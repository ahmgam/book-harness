"""Quality estimation + QA audits (M2: tasks 2.1, 2.4, 2.5, 2.6).

All QA primitives are file-backed and resumable; every scorer accepts a
duck-typed client (anything with ``.generate()`` / ``.model``) so tests can
stub the model.

* **QE scoring (2.1)** — per chunk ``qe-scores.jsonl``. cometkiwi is used
  when the optional ``unbabel-comet`` stack is installed; the default is an
  LLM judge (documents the NC-license caveat by keeping QE swappable).
* **Compliance audit (2.4)** — checks that every enforced glossary term
  present in a chunk's source also appears in its draft in the certified
  Arabic form, tolerance-aware for Arabic morphology: the shared Arabic
  normalizer (task 1.3) plus clitic-prefix stripping (ال، وال، بال، لل،
  و، ب، …). Output ``compliance.jsonl``.
* **Consistency audit (2.5)** — book-level drift check: the expected
  certified form of a term versus its dictionary alternatives across chunks.
  Output ``consistency.jsonl``.
* **Pre-translation OCR gate (2.6)** — ``page_stats`` + block sanity checks
  flag suspicious pages *before* model spend. Output ``gate.jsonl``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from book_harness.normalize import normalize_term
from book_harness.state import cache_key

logger = logging.getLogger(__name__)

# Refuse to audit un-translatable block kinds.
TRANSLATED_KINDS = frozenset({"text", "title", "caption", "list_item"})

# Arabic clitics tolerated in front of a glossary token (longest first).
# Stripping is applied repeatedly so "للرفع" -> "الرفع" -> "رفع".
_CLITIC_PREFIXES = ("وال", "فال", "كال", "بال", "لل", "ال", "و", "ب", "ل", "ك", "ف", "س")


# ---------------------------------------------------------------------------
# QE: cometkiwi adapter (optional, NC license — kept swappable)
# ---------------------------------------------------------------------------

class CometScorer:
    """``Unbabel/wmt22-cometkiwi-da`` adapter; ``available`` False otherwise."""

    MODEL = "Unbabel/wmt22-cometkiwi-da"

    def __init__(self, model_name: str = MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self.available = False
        try:  # heavy: pulls torch/pytorch-lightning; optional extra
            from comet import load_from_checkpoint  # type: ignore

            self._model = load_from_checkpoint(model_name)
            self.available = True
        except Exception:
            self._model = None

    def score(self, pairs: list[tuple[str, str]]) -> list[float] | None:
        """0..1 quality per (source, translation) pair, or None unusable."""
        if not self.available or not pairs:
            return None
        data = [{"src": s, "mt": t} for s, t in pairs]
        out = self._model.predict(data, batch_size=8, gpus=0)
        return [float(x) for x in out.scores]


# ---------------------------------------------------------------------------
# QE: LLM judge (default)
# ---------------------------------------------------------------------------

def judge_prompt(pairs) -> str:
    """Build the reference-free QE judge prompt for one chunk.

    ``pairs`` are (uid, source, translation) triples from :func:`draft_pairs`.
    """
    lines = [
        "You are a translation quality estimator. Judging an English -> Arabic",
        "translation WITHOUT a reference. Score EACH segment on 0-100:",
        "0 = wrong/broken translation, 50 = rough but understandable,",
        "100 = accurate and fluent professional translation.",
        "Judge accuracy to the source first, fluency second.",
        "Reply ONLY with this JSON: {\"scores\": {\"u0\": <int>, \"u1\": <int>, ...}}",
        "",
    ]
    for uid, src, mt in pairs:
        lines.append(f"### {uid}")
        lines.append(f"SOURCE: {src}")
        lines.append(f"TRANSLATION: {mt}")
        lines.append("")
    return "\n".join(lines)


def parse_scores(reply: str, expected: int) -> dict[int, float]:
    """Robustly extract a ``{unit index: score}`` map from a judge reply.

    Falls back from strict JSON to positional regex to a whole-reply integer,
    so a chatty judge still yields usable numbers; empty dict = unusable.
    """
    data: dict = {}
    blob = _first_balanced(reply, "{")
    if blob:
        try:
            parsed = json.loads(blob)
            raw = parsed.get("scores", parsed) if isinstance(parsed, dict) else parsed
            if isinstance(raw, dict):
                for k, v in raw.items():
                    m = re.fullmatch(r"u?(\d+)", str(k).strip())
                    if m:
                        try:
                            data[int(m.group(1))] = float(v)
                        except (TypeError, ValueError):
                            continue
        except json.JSONDecodeError:
            pass
    if not data:
        for m in re.finditer(r"u(\d+)\D{0,12}(\d{1,3})", reply):
            data.setdefault(int(m.group(1)), float(m.group(2)))
    if not data:  # last resort: any 0-100 integer applies to all units
        m = re.search(r"\b(\d{1,3})\b", reply)
        if m and 0 <= int(m.group(1)) <= 100 and expected:
            data = {i: float(m.group(1)) for i in range(expected)}
    return {i: min(100.0, max(0.0, v)) for i, v in data.items() if 0 <= i < expected}


def _first_balanced(text: str, open_ch: str) -> str | None:
    close_ch = {"{": "}", "[": "]"}[open_ch]
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def draft_pairs(chunk: dict[str, Any], draft: dict[str, Any] | None):
    """Aligned (uid, source, translation) pairs for one chunk, None-safe."""
    if not draft:
        return []
    pairs = []
    for unit in chunk.get("units", []):
        d = _draft_unit(draft, unit["uid"])
        if d and unit.get("kind") in TRANSLATED_KINDS:
            pairs.append((unit["uid"], unit["text"], d.get("text", "")))
    return pairs


def qe_cache_key(
    pairs, *, qe_prompt_version: str, model: str, method: str, chunk_id: str = "",
) -> str:
    """Deterministic identity of one QE unit of work."""
    blob = "\x1e".join(f"{src}\x1f{mt}" for _, src, mt in pairs)
    return cache_key(blob or chunk_id, qe_prompt_version or "", model or "", method)


def qe_score_chunk(
    chunk: dict[str, Any],
    draft: dict[str, Any] | None,
    client: Any,                      # OllamaClient (used when method=judge)
    *,
    qe_prompt_version: str,
    comet: CometScorer | None = None,
    method: str = "auto",             # auto | judge | comet
) -> dict[str, Any]:
    """Score draft quality of one chunk. Never raises: {score: None} on failure."""
    record: dict[str, Any] = {
        "chunk_id": chunk["chunk_id"],
        "method": method if method != "auto" else "judge",
        "model": getattr(client, "model", ""),
        "score": None,
        "unit_scores": {},
        "n_units": 0,
        "finished": None,
    }
    pairs = draft_pairs(chunk, draft)
    if not pairs:
        return record
    all_pairs = [(uid, src, src, mt) for uid, src, mt in pairs]

    scores: dict[int, float] = {}
    if method in ("auto", "comet") and comet is not None:
        comet_scores = comet.score([(src, mt) for _, src, mt in pairs])
        if comet_scores:
            record["method"] = "comet"
            record["model"] = comet.model_name
            scores = {i: round(s * 100, 1) for i, s in enumerate(comet_scores)}
    if not scores and method in ("auto", "judge"):
        try:
            reply = client.generate(judge_prompt(pairs))
            scores = parse_scores(reply, expected=len(pairs))
        except Exception as exc:  # keep the run alive, QE is advisory
            logger.warning("QE judge failed for %s: %s", chunk["chunk_id"], exc)

    usable = [
        v for i, (_, _, _, _) in enumerate(all_pairs)
        if (v := scores.get(i)) is not None
    ]
    if usable:
        record["unit_scores"] = {
            all_pairs[i][0]: scores[i]
            for i in range(len(all_pairs)) if i in scores
        }
        record["score"] = round(sum(usable) / len(usable), 1)
        record["n_units"] = len(all_pairs)
        if client is not None and hasattr(client, "model"):
            record["model"] = client.model
    record["finished"] = _now()
    # key covering every input that changes the score
    record["cache_key"] = qe_cache_key(
        pairs,
        qe_prompt_version=qe_prompt_version or "",
        model=str(record["model"] or ""),
        method=record["method"],
        chunk_id=chunk.get("chunk_id", ""),
    )
    return record


def _draft_unit(draft: dict[str, Any], uid: str) -> dict | None:
    for u in draft.get("units", []):
        if u.get("uid") == uid:
            return u
    return None


# ---------------------------------------------------------------------------
# jsonl I/O
# ---------------------------------------------------------------------------

def write_records(records: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def latest_by_chunk(records: list[dict]) -> dict[str, dict]:
    """Last record per chunk_id wins (records are append-ordered)."""
    out: dict[str, dict] = {}
    for r in records:
        if r.get("chunk_id"):
            out[r["chunk_id"]] = r
    return out


# ---------------------------------------------------------------------------
# Compliance audit (2.4)
# ---------------------------------------------------------------------------

def enforced_entries(entries: list[dict]) -> list[dict]:
    """Glossary entries strong enough to be *enforced* on the draft."""
    return [
        e for e in entries
        if e.get("status") in ("certified", "llm_chosen") and e.get("term_ar")
    ]


def _suffix_variants(token: str) -> frozenset[str]:
    """Token plus forms reachable by stripping clitic prefixes (fixed point)."""
    seen = {token}
    frontier = [token]
    while frontier:
        cur = frontier.pop()
        for cl in _CLITIC_PREFIXES:
            if cur.startswith(cl) and len(cur) > len(cl) + 1:
                nxt = cur[len(cl):]
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    return frozenset(seen)


def _tokens(text: str) -> list[str]:
    """Word tokens for normalized-Arabic matching (punctuation dropped)."""
    return re.findall(r"[\w\u0600-\u06FF]+", normalize_term(text))


def arabic_present(text: str, term_ar: str) -> bool:
    """True when ``term_ar`` occurs in ``text``, tolerating clitic prefixes
    and normalization differences (catch #6: morphology vs. literal match)."""
    g_tokens = _tokens(term_ar)
    t_tokens = _tokens(text)
    if not g_tokens or not t_tokens:
        return False
    g_variants = [_suffix_variants(g) for g in g_tokens]
    t_variants = [_suffix_variants(t) for t in t_tokens]
    n = len(g_tokens)
    for i in range(len(t_tokens) - n + 1):
        window = t_variants[i:i + n]
        ok = True
        for tset, gset in zip(window, g_variants):
            if not (tset & gset):
                ok = False
                break
        if ok:
            return True
    return False


def source_matches(term_en: str, text: str) -> bool:
    """Word-boundary test whether the English source term occurs (task 1.8 style)."""
    import re as _re

    if not term_en or len(term_en) < 3:
        return False
    pat = r"\b" + r"\s+".join(_re.escape(w) for w in term_en.lower().split()) + r"\b"
    return bool(_re.search(pat, text.lower()))


def compliance_audit(
    chunk: dict[str, Any],
    draft: dict[str, Any] | None,
    entries: list[dict],
    *,
    glossary_version: str | None = None,
) -> dict[str, Any]:
    """One compliance record per chunk: enforced-term enforcement."""
    record: dict[str, Any] = {
        "chunk_id": chunk["chunk_id"],
        "glossary_version": glossary_version,
        "checked": 0,
        "hits": 0,
        "violations": [],
        "rate": None,
        "finished": _now(),
    }
    if not draft:
        return record
    enforced = enforced_entries(entries)
    by_uid = {u.get("uid"): u for u in draft.get("units", [])}
    for unit in chunk.get("units", []):
        if unit.get("kind") not in TRANSLATED_KINDS:
            continue
        d = by_uid.get(unit["uid"])
        if not d:
            continue
        pairs = [
            e for e in enforced
            if source_matches(e["term_en"], unit["text"])
        ]
        for e in pairs:
            record["checked"] += 1
            if arabic_present(d.get("text", ""), e["term_ar"]):
                record["hits"] += 1
            else:
                record["violations"].append({
                    "uid": unit["uid"],
                    "term_en": e["term_en"],
                    "term_ar": e["term_ar"],
                })
    total = record["checked"]
    record["rate"] = round(100.0 * record["hits"] / total, 1) if total else None
    return record


def requeue_targets(records: list[dict], floor: float) -> list[dict]:
    """Chunks below the compliance floor with actionable violations."""
    out = []
    for r in records:
        if r.get("violations") and (
            r.get("rate") is None or r["rate"] < floor
        ):
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Consistency audit (2.5)
# ---------------------------------------------------------------------------

def consistency_audit(
    chunk_drafts: list[tuple[dict[str, Any], dict[str, Any] | None]],
    entries: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """One record per enforced English term, book-wide.

    Returns ``(records, summary)``. Drift = a chunk that used a competing
    dictionary rendering (``alternatives``) where the certified form is
    expected.
    """
    records: list[dict] = []
    total_drift = 0
    for e in enforced_entries(entries):
        term = e["term_en"]
        expected = e["term_ar"]
        alternatives = [a for a in (e.get("alternatives") or []) if a]
        occurrences = 0
        expected_chunks: list[str] = []
        alt_hits: dict[str, list[str]] = {a: [] for a in alternatives}
        mixed_chunks: list[str] = []
        drift_chunks: list[str] = []
        for chunk, draft in chunk_drafts:
            if not draft:
                continue
            used_expected = False
            used_alt: list[str] = []
            by_uid = {u.get("uid"): u for u in draft.get("units", [])}
            for unit in chunk.get("units", []):
                if unit.get("kind") not in TRANSLATED_KINDS:
                    continue
                d = by_uid.get(unit["uid"])
                if not d or not source_matches(term, unit["text"]):
                    continue
                occurrences += 1
                if arabic_present(d.get("text", ""), expected):
                    used_expected = True
                for a in alternatives:
                    if arabic_present(d.get("text", ""), a):
                        if chunk["chunk_id"] not in alt_hits[a]:
                            alt_hits[a].append(chunk["chunk_id"])
                        used_alt.append(a)
            if not used_expected and not used_alt:
                continue
            chunk_id = chunk["chunk_id"]
            if used_expected:
                expected_chunks.append(chunk_id)
                if used_alt:
                    mixed_chunks.append(chunk_id)
                    drift_chunks.append(chunk_id)
                    total_drift += 1
            elif used_alt:
                drift_chunks.append(chunk_id)
                total_drift += 1
        records.append({
            "term_en": term,
            "expected": expected,
            "status": e.get("status"),
            "occurrences": occurrences,
            "expected_chunks": len(expected_chunks),
            "alternative_hits": alt_hits,
            "mixed_chunks": mixed_chunks,
            "drift_chunks": drift_chunks,
            "drift": bool(drift_chunks),
            "finished": _now(),
        })
    drift_terms = sum(1 for r in records if r["drift"])
    summary = {
        "terms_checked": len(records),
        "drift_terms": drift_terms,
        "drift_chunks_total": total_drift,
        "finished": _now(),
    }
    return records, summary


# ---------------------------------------------------------------------------
# Pre-translation OCR QA gate (2.6)
# ---------------------------------------------------------------------------

_OK_CHARS = re.compile(
    r"[0-9A-Za-z\u0600-\u06FF\s\-.,/()%…:;'\"!?؟،؛«»()\[\]&=+\-*_]", re.ASCII
)

def junk_ratio(text: str) -> float:
    """Fraction of characters that are not letters/digits/punctuation-ish.

    ASCII + the Arabic block only count as "clean"; letters from other
    scripts (OCR's typical mojibake sources) count as junk.
    """
    text = text.strip()
    if not text:
        return 0.0
    ok = sum(1 for ch in text if _OK_CHARS.match(ch))
    return 1.0 - ok / len(text)


def gate_pages(
    book: Any,                                    # book_harness.book.Book
    *,
    junk_threshold: float = 0.4,
    short_text_chars: int = 12,
    fragmented_min: int = 3,
) -> list[dict[str, Any]]:
    """Per-page sanity records; ``suspect`` pages should get human eyes
    (or a different OCR method) before translation spend."""
    methods = {s.page: s.text_extraction_method for s in getattr(book, "page_stats", []) or []}
    records: list[dict[str, Any]] = []
    pages = getattr(book, "_blocks_by_page", {})
    for page in sorted(pages):
        blocks = pages[page]
        flags: list[str] = []
        short_texts = 0
        junk_max = 0.0
        for b in blocks:
            if b.type.value in ("PageHeader", "PageFooter"):
                continue
            text = b.text
            if b.type.value in ("Text", "Caption", "Footnote") and text:
                r = junk_ratio(text)
                junk_max = max(junk_max, r)
                if r > junk_threshold:
                    flags.append(f"junk@{b.id}")
                if len(text) < short_text_chars:
                    short_texts += 1
            if b.type.value == "SectionHeader" and not text.strip():
                flags.append(f"empty-header@{b.id}")
        if short_texts >= fragmented_min:
            flags.append("fragmented")
        if not blocks:
            flags.append("empty-page")
        records.append({
            "page": page,
            "method": methods.get(page, "?"),
            "blocks": len(blocks),
            "junk_max": round(junk_max, 3),
            "flags": sorted(set(flags)),
            "suspect": bool(flags),
        })
    return records


def write_gate(records: list[dict], qa_dir: str | Path) -> Path:
    return write_records(records, Path(qa_dir) / "gate.jsonl")


def _now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S")
