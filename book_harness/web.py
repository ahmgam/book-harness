"""Web UI (M4, PLAN.md tasks 4.1 + 4.2).

Library first: every capability here is a pure file-I/O function over the
run directory; the HTTP server at the bottom is a thin stdlib shell over it.

* **Progress view (4.1)** — :func:`overview` summarises ``state.json`` (stage
  cursor) plus on-disk chunk counts.  :func:`progress_stream` turns change
  detection over that snapshot into a Server-Sent-Events stream, so the page
  refreshes itself while a run is going.
* **Review screen (4.2)** — :func:`uncertified_terms` merges ``uncertified``
  glossary entries with the ``proposed`` terms mined by the drift loop (3.6),
  :func:`low_qe_chunks` lists chunks below the QE quality threshold with
  source/draft side-by-side texts.

Human decisions write back to versioned artifacts:

* :func:`apply_term_decision` — sets/updates the reviewed entry in
  ``glossary.yaml`` (``decided_by: human``, never silently certified) and
  **re-queues only the chunks that contain the term**;
* :func:`requeue_chunk` — archives the stored draft (both ``.md`` and
  ``.meta.json`` move to ``04_drafts/requeued/``), which invalidates its
  cache key so the next ``translate`` pass regenerates *only that chunk*.

Every human action is appended to ``06_qa/review-log.jsonl`` for audit.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import yaml

from book_harness import drift as drift_mod
from book_harness.assemble import DraftBook
from book_harness.config import (
    GLOSSARY_FILE,
    QUALITY_THRESHOLD,
    STAGE_DIRS,
    STATE_FILE,
)
from book_harness.glossary import (
    GlossaryEntry,
    glossary_version,
    load_glossary,
    write_glossary,
)
from book_harness.qa import TRANSLATED_KINDS, latest_by_chunk, read_records

logger = logging.getLogger(__name__)

REVIEW_LOG_FILE = "review-log.jsonl"
REVIEW_LOG = Path(STAGE_DIRS["qa"]) / REVIEW_LOG_FILE
DRIFT_CANDIDATES = Path(STAGE_DIRS["glossary"]) / "drift-candidates.jsonl"
REQUEUED_DIR = "requeued"

#: Stage display order (mirrors cli.STAGES; kept local to avoid a cycle).
STAGE_ORDER = [
    "source", "outline-repair", "terms", "glossary", "consistency", "chunk",
    "gate", "translate", "qe", "refine", "compliance", "drift-audit",
    "drift", "artifacts", "assemble",
]

_TERM_NORM_RE = re.compile(r"[_\-/]+")


def _norm(term_en: str) -> str:
    """Canonical term key — same normalisation the drift loop uses."""
    return _TERM_NORM_RE.sub(" ", (term_en or "").lower()).strip()


def _run(run_dir: str | Path, stage: str) -> Path:
    return Path(run_dir) / STAGE_DIRS[stage]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# 4.1 — progress snapshot + SSE stream
# ---------------------------------------------------------------------------

def overview(run_dir: str | Path) -> dict[str, Any]:
    """Progress snapshot (task 4.1): stage cursor + chunk counts + counts of
    items awaiting review. Pure read; safe on a missing run dir."""
    run_dir = Path(run_dir)
    state_path = run_dir / STATE_FILE
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("unreadable state file: %s", state_path)
    stages = state.get("stages", {})

    chunks_dir = _run(run_dir, "chunks")
    total = len(list(chunks_dir.glob("chunk-*.json"))) if chunks_dir.exists() else 0
    drafts_dir = _run(run_dir, "drafts")
    translated = len(list(drafts_dir.glob("chunk-*.meta.json"))) if drafts_dir.exists() else 0

    terms = uncertified_terms(run_dir)
    low = low_qe_chunks(run_dir)
    ordered = [
        {"stage": s, **stages.get(s, {"status": "pending"})}
        for s in STAGE_ORDER
    ]
    extra = [s for s in stages if s not in STAGE_ORDER]
    ordered += [{"stage": s, **stages[s]} for s in sorted(extra)]

    by_stage = {o["stage"]: o for o in ordered}
    tr = by_stage.get("translate", {})
    in_flight = tr.get("status") == "started"
    return {
        "book_id": state.get("book_id", run_dir.name),
        "stages": ordered,
        "translation": {
            "total": total,
            "translated": translated,
            "running": in_flight,
            "percent": round(100.0 * translated / total, 1) if total else None,
        },
        "review": {"uncertified_terms": len(terms), "low_qe_chunks": len(low)},
    }


def progress_stream(
    run_dir: str | Path, poll_seconds: float = 1.0, *, heartbeat: float = 15.0
) -> Iterator[str]:
    """SSE frames (``event: progress``) whenever the overview changes.

    Deduplicated between ticks, comments as heartbeat; ends when the consumer
    closes (the writer notices on the next failed flush).
    """
    last: str | None = None
    since_beat = 0.0
    while True:
        snap = overview(run_dir)
        payload = json.dumps(snap, ensure_ascii=False, sort_keys=True)
        if payload != last:
            last = payload
            yield f"event: progress\ndata: {payload}\n\n"
            since_beat = 0.0
        elif since_beat >= heartbeat:
            yield ": heartbeat\n\n"
            since_beat = 0.0
        time.sleep(poll_seconds)
        since_beat += poll_seconds


# ---------------------------------------------------------------------------
# 4.2 — review items
# ---------------------------------------------------------------------------

_REVIEW_KEYS = (
    "term_en", "term_ar", "source", "wikidata", "page", "scan",
    "status", "decided_by", "reason", "human_decision", "decided_at",
)


def uncertified_terms(run_dir: str | Path) -> list[dict[str, Any]]:
    """``uncertified`` glossary entries + drift ``proposed`` terms (deduped).

    Both piles feed the same review queue: glossary entries are already
    catalogued, drift proposals await a first decision.
    """
    run_dir = Path(run_dir)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    gl_path = _run(run_dir, "glossary") / GLOSSARY_FILE
    if gl_path.exists():
        try:
            _, entries = load_glossary(gl_path)
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("unreadable glossary %s: %s", gl_path, exc)
            entries = []
        version = glossary_version(entries) if entries else ""
        for e in entries:
            if e.get("status") != "uncertified":
                continue
            item = {k: e.get(k) for k in _REVIEW_KEYS}
            item["origin"] = "glossary"
            item["glossary_version"] = version
            item["decided"] = (
                e.get("decided_by") == "human"
                and e.get("human_decision") in ("approved", "rejected")
            )
            out.append(item)
            seen.add(_norm(e.get("term_en") or ""))
    dc_path = run_dir / DRIFT_CANDIDATES
    for c in drift_mod.load_candidates(dc_path):
        if c.get("status") != "proposed":
            continue
        key = _norm(c.get("term") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "term_en": c.get("term"),
            "term_ar": None,
            "status": "proposed",
            "decided_by": None,
            "origin": "drift",
            "reason": "glossary drift: recurring uncovered term",
            "freq": c.get("freq"),
            "df": c.get("df"),
            "chunks": c.get("chunks", []),
            "page": c.get("page"),
            "decided": False,
        })
    out.sort(key=lambda t: (t.get("decided", False), str(t.get("term_en") or "")))
    return out


def low_qe_chunks(
    run_dir: str | Path, threshold: float = QUALITY_THRESHOLD
) -> list[dict[str, Any]]:
    """Chunks whose latest QE score is below ``threshold`` (worst first).

    Each item carries the source/draft text pair per translatable unit so the
    review screen can render them side by side. Refinement overlays are
    applied when present (matching what assembly would print).
    """
    run_dir = Path(run_dir)
    scores = latest_by_chunk(read_records(_run(run_dir, "qa") / "qe-scores.jsonl"))
    if not scores:
        return []
    drafts_dir = _run(run_dir, "drafts")
    if not drafts_dir.exists():
        return []
    drafts = DraftBook(drafts_dir)
    try:
        from book_harness.assemble import _apply_refinements

        _apply_refinements(drafts, Path(run_dir) / STAGE_DIRS["refine"])
    except Exception as exc:                      # refinements are optional
        logger.debug("refinement overlay skipped: %s", exc)

    items: list[dict[str, Any]] = []
    for chunk_id, rec in scores.items():
        score = rec.get("score")
        drafted = drafts.meta_by_chunk.get(chunk_id)
        if drafted is None or score is None or score >= threshold:
            continue
        chunk_path = _run(run_dir, "chunks") / f"{chunk_id}.json"
        units: list[dict[str, Any]] = []
        if chunk_path.exists():
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
            for u in chunk.get("units", []):
                if u.get("kind") not in TRANSLATED_KINDS:
                    continue
                d = drafts.unit(chunk_id, u["uid"])
                units.append({
                    "uid": u.get("uid"),
                    "kind": u.get("kind"),
                    "source": u.get("text", ""),
                    "text": (d or {}).get("text", ""),
                    "score": (rec.get("unit_scores") or {}).get(u.get("uid", "")),
                })
        items.append({
            "chunk_id": chunk_id,
            "score": score,
            "method": rec.get("method"),
            "model": rec.get("model"),
            "section": drafted.get("section", ""),
            "pages": drafted.get("pages", []),
            "cache_key": drafted.get("cache_key", ""),
            "units": units,
        })
    items.sort(key=lambda r: (r["score"], r["chunk_id"]))
    return items


# ---------------------------------------------------------------------------
# Write-back: human decisions (4.2)
# ---------------------------------------------------------------------------

def review_log(run_dir: str | Path, **record: Any) -> None:
    """Append one audit record for a human action taken through the UI."""
    path = Path(run_dir) / REVIEW_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": _now_iso(), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def requeue_chunk(
    run_dir: str | Path,
    chunk_id: str,
    reason: str = "",
    *,
    bust_cache: bool = True,
) -> bool:
    """Re-queue one chunk for re-translation (task 4.2).

    Drafts are archived to ``04_drafts/requeued/`` so their cache key
    disappears from disk: the next ``translate`` pass sees no fresh draft and
    regenerates exactly this chunk. Returns ``False`` when there is nothing
    to invalidate (never translated, or already re-queued).
    """
    run_dir = Path(run_dir)
    drafts_dir = run_dir / STAGE_DIRS["drafts"]
    meta_path = drafts_dir / f"{chunk_id}.meta.json"
    if not meta_path.exists():
        logger.info("requeue: %s has no draft — nothing to invalidate", chunk_id)
        return False
    if bust_cache:
        archive = drafts_dir / REQUEUED_DIR
        archive.mkdir(parents=True, exist_ok=True)
        for name in (f"{chunk_id}.meta.json", f"{chunk_id}.md"):
            src = drafts_dir / name
            if src.exists():
                dest = archive / f"1_{name}"
                i = 2
                while dest.exists():          # keep every archived revision
                    dest = archive / f"{i}_{name}"
                    i += 1
                shutil.move(str(src), str(dest))
    review_log(run_dir, action="requeue", chunk_id=chunk_id, reason=reason)
    logger.info("requeue: %s (cache invalidated; %s)", chunk_id, reason or "no reason given")
    return True


def _update_drift_decision(run_dir: Path, term_en: str, decision: str) -> int:
    """Best-effort status update inside ``drift-candidates.jsonl``."""
    dc_path = run_dir / DRIFT_CANDIDATES
    candidates = drift_mod.load_candidates(dc_path)
    if not candidates:
        return 0
    key = _norm(term_en)
    changed = 0
    for c in candidates:
        if _norm(c.get("term") or "") != key:
            continue
        c["status"] = "confirmed" if decision == "approve" else "rejected"
        c["decision"] = {"approve": "approved by human review",
                         "reject": "rejected by human review"}[decision]
        c["decided_at"] = _now_iso()
        changed += 1
    if changed:
        with open(dc_path, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return changed


def apply_term_decision(
    run_dir: str | Path,
    term_en: str,
    decision: str,                       # approve | reject
    arabic: str = "",
    *,
    requeue: bool = True,
) -> dict[str, Any]:
    """Write one human review decision back to the artifacts (task 4.2).

    ``approve`` requires a reviewed Arabic rendering; the entry is stored with
    ``status: uncertified``/``decided_by: human`` — human review never
    silently certifies (PLAN catch #5) — and chunks containing the term are
    re-queued so only they pick the new rendering up on the next run.
    ``reject`` keeps the term catalogued but marks it unusable.
    """
    decision = decision.lower()
    if decision not in ("approve", "reject"):
        raise ValueError(f"decision must be approve|reject, got {decision!r}")
    arabic = (arabic or "").strip()
    if decision == "approve" and not arabic:
        raise ValueError("approve requires the reviewed Arabic rendering")
    run_dir = Path(run_dir)
    gl_path = _run(run_dir, "glossary") / GLOSSARY_FILE

    entries: list[GlossaryEntry] = []
    if gl_path.exists():
        entries = load_glossary(gl_path)[1]
    key = _norm(term_en)
    entry = next((e for e in entries if _norm(e.get("term_en") or "") == key), None)
    action = "updated"
    if entry is None:
        entry = GlossaryEntry.uncertified(
            term_en, "human review through the web review screen"
        )
        entries.append(entry)
        action = "appended"
    if decision == "approve":
        entry["term_ar"] = arabic
        entry["reason"] = "human-confirmed Arabic rendering (review screen)"
    else:
        entry["reason"] = "human-rejected proposal (review screen)"
    entry["decided_by"] = "human"
    entry["human_decision"] = "approved" if decision == "approve" else "rejected"
    if decision == "approve":
        entry["decided_at"] = _now_iso()

    out_path, version = write_glossary(entries, gl_path)

    requeued: list[str] = []
    if requeue and decision == "approve":
        chunk_ids = drift_mod.affected_chunks(_load_chunk_dicts(run_dir), term_en)
        for cid in chunk_ids:
            requeue_chunk(
                run_dir, cid,
                reason=f"term '{term_en}' approved -> {arabic}",
            )
        requeued = chunk_ids
    elif decision == "reject":
        requeued = []

    _update_drift_decision(run_dir, term_en, decision)
    review_log(
        run_dir, action="term", term=term_en, decision=decision,
        arabic=arabic, glossary_version=version, requeued=requeued,
    )
    logger.info(
        "term decision: %s '%s' = %s (glossary v%s, %d chunk(s) re-queued)",
        decision, term_en, arabic or "-", version, len(requeued),
    )
    return {
        "action": action,
        "glossary_path": str(out_path),
        "glossary_version": version,
        "entry": dict(entry),
        "requeued": requeued,
    }


def _load_chunk_dicts(run_dir: Path) -> list[dict[str, Any]]:
    from book_harness.chunker import load_chunks

    chunks_dir = _run(run_dir, "chunks")
    if not chunks_dir.exists():
        return []
    return [c.to_dict() for c in load_chunks(chunks_dir)]


# ---------------------------------------------------------------------------
# Thin HTTP shell over the library (stdlib-only, no dependency)
# ---------------------------------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>book-harness — review</title>
<style>
  :root { --ink:#1d2430; --muted:#6b7280; --line:#e2e6ea; --accent:#1f6feb; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:system-ui,-apple-system,sans-serif;
         background:#f6f7f9; color:var(--ink); }
  header { background:#fff; border-bottom:1px solid var(--line); padding:.8rem 1.2rem;
           display:flex; align-items:center; gap:1rem; }
  header h1 { font-size:1.05rem; margin:0; }
  #live { font-size:.72rem; padding:.2rem .55rem; border-radius:999px;
          background:#fde68a; color:#7a5c00; }
  #live.on { background:#bbf7d0; color:#155e21; }
  main { max-width:1000px; margin:1.2rem auto; padding:0 1rem; }
  h2 { font-size:.95rem; margin:1.4rem 0 .4rem; color:var(--muted);
       text-transform:uppercase; letter-spacing:.04em; }
  .card { background:#fff; border:1px solid var(--line); border-radius:8px;
          padding:.7rem .9rem; margin:.5rem 0; }
  table { width:100%; border-collapse:collapse; font-size:.88rem; }
  th,td { text-align:left; padding:.35rem .5rem; border-bottom:1px solid #eef1f4;
          vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:.78rem; }
  button { cursor:pointer; border:1px solid var(--line); background:#fff;
           border-radius:6px; padding:.3rem .7rem; font-size:.8rem; }
  button.go { background:var(--accent); border-color:var(--accent); color:#fff; }
  input { border:1px solid var(--line); border-radius:6px; padding:.3rem .5rem;
          font-size:.85rem; width:14rem; }
  input.ar { direction:rtl; width:16rem; }
  .ar { direction:rtl; unicode-bidi:isolate; }
  .muted { color:var(--muted); font-size:.8rem; }
  .bar { height:8px; background:#eef1f4; border-radius:999px; overflow:hidden; }
  .bar i { display:block; height:100%; background:var(--accent); }
  details summary { cursor:pointer; }
  .pairs { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin:.6rem 0; }
  .pairs .src, .pairs .mt { white-space:pre-wrap; font-size:.82rem;
          background:#fafbfc; border:1px solid #eef1f4; border-radius:6px;
          padding:.5rem; max-height:16rem; overflow:auto; }
  .tag { display:inline-block; font-size:.68rem; padding:.1rem .45rem;
         border-radius:999px; background:#eef4ff; color:#1f4fbb; }
  .done { background:#bbf7d0; color:#155e21; }
  .skip { background:#e5e7eb; color:#4b5563; }
</style>
</head>
<body>
<header>
  <h1>book-harness — <span id="book">…</span></h1>
  <span id="live">connecting…</span>
</header>
<main>
  <h2>Progress</h2>
  <div class="card" id="progressCard">
    <div id="barWrap">
      <div class="bar"><i id="barFill" style="width:0"></i></div>
      <p class="muted" id="barText"></p>
    </div>
    <table id="stages"><thead><tr>
      <th>stage</th><th>status</th><th>detail</th>
    </tr></thead><tbody></tbody></table>
  </div>

  <h2>Terms awaiting review</h2>
  <div class="card" id="termsCard">Loading…</div>

  <h2>Chunks below the quality threshold</h2>
  <div class="card" id="qeCard">Loading…</div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function renderProgress(o) {
  $("book").textContent = o.book_id;
  const t = o.translation;
  if (t.total) {
    $("barFill").style.width = (t.percent || 0) + "%";
    $("barText").textContent =
      t.translated + " / " + t.total + " chunks translated (" +
      (t.percent ?? 0) + "%)" + (t.running ? " — running…" : "");
  } else {
    $("barText").textContent = "no chunks on disk yet";
  }
  const tb = $("stages").tBodies[0];
  tb.innerHTML = "";
  for (const s of o.stages) {
    const extra = Object.entries(s)
      .filter(([k]) => !["stage","status","started","finished"].includes(k))
      .map(([k, v]) => k + "=" + (typeof v === "object" ? JSON.stringify(v) : v))
      .join(", ");
    const times = (s.started ? new Date(s.started * 1000).toLocaleString() : "");
    if (s.finished) times += " → " + new Date(s.finished * 1000).toLocaleTimeString();
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + esc(s.stage) + "</td><td><span class='tag "
      + esc(s.status) + "'>" + esc(s.status) + "</span></td><td class='muted'>"
      + esc(extra) + " " + esc(times) + "</td>";
    tb.appendChild(tr);
  }
}

async function loadReview() {
  const r = await fetch("/api/review");
  const d = await r.json();
  renderTerms(d.terms);
  renderQe(d.low_qe);
}

function renderTerms(terms) {
  const card = $("termsCard");
  if (!terms.length) { card.innerHTML = "<p class='muted'>Nothing to review.</p>"; return; }
  let html = "<table><thead><tr><th>term</th><th>current</th><th>origin</th>" +
    "<th>reviewed Arabic</th><th></th></tr></thead><tbody>";
  for (const t of terms) {
    html += "<tr data-term='" + esc(t.term_en) + "'><td><b>" + esc(t.term_en)
      + "</b></td><td>" + esc(t.term_ar || "—") + "</td><td>"
      + esc(t.origin + (t.status ? " / " + t.status : ""))
      + (t.freq ? " <span class='muted'>×" + t.freq + "</span>" : "")
      + "</td><td><input class='ar' data-ar placeholder='الترجمة…'></td>"
      + "<td><button class='go' onclick=\"decide(this,'approve')\">Approve</button>"
      + "<button onclick=\"decide(this,'reject')\">Reject</button></td></tr>";
  }
  card.innerHTML = html + "</tbody></table>";
}

async function decide(btn, decision) {
  const row = btn.closest("tr");
  const en = row.dataset.term;
  const ar = row.querySelector("input.ar").value;
  const body = JSON.stringify({ term_en: en, decision, ar });
  const r = await fetch("/api/terms", { method: "POST",
    headers: {"Content-Type": "application/json"}, body });
  const d = await r.json();
  if (d.error) { alert(d.error); return; }
  alert("Saved (" + d.action + ", glossary v" + d.glossary_version +
    "); " + d.requeued.length + " affected chunk(s) re-queued.");
  loadReview();
}

function renderQe(chunks) {
  const card = $("qeCard");
  if (!chunks.length) { card.innerHTML = "<p class='muted'>All chunks are at or above the threshold.</p>"; return; }
  let html = "";
  for (const c of chunks) {
    html += "<details class='card'><summary><b>" + esc(c.chunk_id) +
      "</b> — QE " + esc(c.score) +
      " <span class='muted'>" + esc(c.method) + " / " + esc(c.model) +
      " · " + esc(c.section) + " · p." + esc((c.pages || []).join(",")) +
      "</span> <button onclick=\"requeue(event,this,'"
      + esc(c.chunk_id) + "')\">Re-queue</button></summary>";
    html += "<div class='pairs'>";
    for (const u of c.units) {
      html += "<div><div class='src'>" + esc(u.source) + "</div>" +
        "<div class='mt ar'>" + esc(u.text) + "</div></div>";
    }
    html += "</div></details>";
  }
  card.innerHTML = html;
}

async function requeue(e, btn, chunkId) {
  e.stopPropagation();
  const r = await fetch("/api/requeue", { method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ chunk_id: chunkId, reason: "queued from review UI" }) });
  const d = await r.json();
  if (d.error) { alert(d.error); return; }
  btn.disabled = true;
  btn.textContent = d.requeued ? "Re-queued" : "Nothing to invalidate";
}

const es = new EventSource("/events");
es.addEventListener("open", () => { $("live").textContent = "live"; $("live").classList.add("on"); });
es.addEventListener("progress", (ev) => { renderProgress(JSON.parse(ev.data)); });
es.addEventListener("error", () => { $("live").textContent = "reconnecting…"; $("live").classList.remove("on"); });

renderProgress({ book_id: "…", stages: [], translation: {} });
loadReview();
</script>
</body>
</html>
"""


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """GET  /                review+progress page (4.1 + 4.2)
       GET  /api/overview    progress snapshot (also the SSE payload)
       GET  /api/review      review items (uncertified terms + low-QE chunks)
       GET  /events          SSE stream of ``event: progress``
       POST /api/terms       {"term_en", "decision", "ar"} -> glossary write-back
       POST /api/requeue     {"chunk_id", "reason"} -> invalidate the draft

    Class attributes ``run_dir`` / ``poll_seconds`` are set by the factory.
    """

    run_dir: Path
    poll_seconds: float = 1.0

    # -- plumbing ----------------------------------------------------------
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            data = {}
        return data if isinstance(data, dict) else {}

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        logger.debug("%s %s", self.address_string(), fmt % args)

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:                                     # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            self._send_html(INDEX_HTML)
        elif path == "/api/overview":
            self._send_json(overview(self.run_dir))
        elif path == "/api/review":
            self._send_json({
                "terms": uncertified_terms(self.run_dir),
                "low_qe": low_qe_chunks(self.run_dir),
            })
        elif path == "/events":
            self._stream_events()
        else:
            self._send_json({"error": f"no route {path}"}, 404)

    def do_POST(self) -> None:                                    # noqa: N802
        path = urlsplit(self.path).path
        body = self._read_json()
        try:
            if path == "/api/terms":
                result = apply_term_decision(
                    self.run_dir,
                    term_en=str(body.get("term_en") or ""),
                    decision=str(body.get("decision") or ""),
                    arabic=str(body.get("ar") or ""),
                )
                self._send_json(result)
            elif path == "/api/requeue":
                chunk_id = str(body.get("chunk_id") or "")
                ok = requeue_chunk(
                    self.run_dir, chunk_id,
                    reason=str(body.get("reason") or ""),
                )
                self._send_json({"chunk_id": chunk_id, "requeued": ok})
            else:
                self._send_json({"error": f"no route {path}"}, 404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:                                  # pragma: no cover
            logger.exception("POST %s failed", path)
            self._send_json({"error": str(exc)}, 500)

    def _stream_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for frame in progress_stream(self.run_dir, self.poll_seconds):
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                                                  # client left
        except Exception:                                         # pragma: no cover
            logger.exception("SSE stream failed")
        self.close_connection = True


def make_server(
    run_dir: str | Path, host: str = "127.0.0.1", port: int = 8000,
    poll_seconds: float = 1.0,
) -> Any:
    """Build (not start) the SSE-capable review server."""
    handler = type(
        "BoundHandler", (ReviewRequestHandler,),
        {"run_dir": Path(run_dir), "poll_seconds": float(poll_seconds)},
    )
    return ThreadingHTTPServer((host, port), handler)


def serve(
    run_dir: str | Path, host: str = "127.0.0.1", port: int = 8000,
    poll_seconds: float = 1.0,
) -> None:
    """Blocking entry point for the CLI (`book-harness serve <run-dir>`)."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    httpd = make_server(run_dir, host=host, port=port, poll_seconds=poll_seconds)
    logger.info("review UI: http://%s:%d/ (%s)", host, httpd.server_address[1], run_dir)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("review UI: shutting down")
    finally:
        httpd.server_close()
