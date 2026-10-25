"""Guarded refinement loop (M2: tasks 2.2 + 2.3).

estimate -> literal patch -> A/B verify, per chunk:

1. **Estimate** — a *general instruct* model (not the MT model) locates at
   most ``REFINE_MAX_SPANS`` MQM-style error spans; each span must specify an
   exact substring of the draft ("original") plus a "suggestion". Spans whose
   ``original`` does not occur exactly once are dropped (pure-Python literal
   patching later cannot be safe on ambiguity).
2. **Patch** — plain ``str.replace`` on the draft text, no model involved.
3. **Verify** — an A/B judgment by the model, run twice with the candidates
   in swapped positions; a patch is applied only if *both* verdicts prefer
   the patched version. Rejected patches keep the original draft.

The model never rewrites a whole chunk, so errors stay bounded and auditable.

Artifacts (``05_refine/``): per chunk ``chunk-NNNN.spans.json``,
``chunk-NNNN.patches.jsonl``, ``chunk-NNNN.verdicts.jsonl`` and the
authoritative ``chunk-NNNN.refined.json`` (accepted unit texts keyed by a
refinement cache key, consumed by assembly). Refinement of a chunk is
resumable: a fresh ``refined.json`` short-circuits the loop.

QE-budgeted scheduling (2.3) lives in :func:`plan_refinement`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from book_harness.config import REFINE_MAX_SPANS
from book_harness.qa import TRANSLATED_KINDS, _first_balanced, write_records
from book_harness.state import cache_key

logger = logging.getLogger(__name__)

ESTIMATE_SYSTEM = (
    "You are an MQM-trained translation QA analyst. You receive an English "
    "source text and its Arabic translation. Identify translation errors in "
    "the translation only (not the source). Use categories: terminology, "
    "accuracy, fluency, numbers, style, punctuation. Severity: minor, major, "
    "critical.\n"
    "Reply ONLY with a JSON array, one object per error:\n"
    '[{"original": "verbatim substring copied exactly from the translation",'
    ' "suggestion": "corrected replacement substring",'
    ' "category": "terminology", "severity": "major"}]\n'
    "Rules:\n"
    "- `original` MUST be copied character-for-character from the translation.\n"
    "- Do not span across sentence boundaries.\n"
    "- At most %d errors; most severe first. If there are no errors, reply []."
    % REFINE_MAX_SPANS
)

VERIFY_SYSTEM = (
    "You are judging which of two Arabic translations of an English source is "
    "better (adequacy to the source first, fluency second). Reply with a "
    "single letter, A or B, and nothing else."
)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def estimate_spans(
    client: Any,
    source_text: str,
    translated_text: str,
    *,
    max_spans: int = REFINE_MAX_SPANS,
) -> list[dict[str, Any]]:
    """Raw error-spans estimate. Never raises; [] on model/parse failure."""
    prompt = (
        f"SOURCE:\n{source_text}\n\nTRANSLATION:\n{translated_text}\n\n"
        f"Errors (max {max_spans}):"
    )
    try:
        reply = client.generate(prompt, system=ESTIMATE_SYSTEM)
    except Exception as exc:
        logger.warning("estimate failed: %s", exc)
        return []
    return parse_spans(reply)


def parse_spans(reply: str) -> list[dict[str, Any]]:
    """Extract a list of span dicts from the reply; tolerates chatter/blackquotes."""
    blob = _first_balanced(reply, "[")
    if blob is None:
        blob = _single_bracket_wrap(reply)
    if blob is None:
        return []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, list):
        return []
    return [s for s in data if isinstance(s, dict)]


def _single_bracket_wrap(reply: str) -> str | None:
    if reply.count("[") == 1 and reply.count("]") == 1:
        return reply[reply.index("["):reply.index("]") + 1]
    return None


def _estimate_or_none(
    client: Any,
    source_text: str,
    out_text: str,
    *,
    max_spans: int,
    chunk_id: str,
    uid: str,
) -> list[dict[str, Any]]:
    """``estimate_spans`` that degrades to "no spans" on model failure.

    A failed/aborted estimate (e.g. Ollama aborts a token-repeat loop) must
    skip the unit rather than kill the refine stage; logged with the reason.
    """
    try:
        return estimate_spans(client, source_text, out_text, max_spans=max_spans)
    except Exception as exc:
        logger.warning(
            "%s %s: estimate model call failed, unit skipped (%s)",
            chunk_id, uid, exc,
        )
        return []


def scrub_spans(
    spans: list[dict[str, Any]],
    text: str,
    *,
    max_spans: int = REFINE_MAX_SPANS,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep only spans safe for literal patching. Returns (kept, rejected)."""
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for raw in spans:
        original = str(raw.get("original", "")).strip()
        suggestion = str(raw.get("suggestion", "")).strip()
        if not original:
            rejected.append({**raw, "reason": "empty"})
        elif not suggestion:
            rejected.append({**raw, "reason": "no_suggestion"})
        elif original == suggestion:
            rejected.append({**raw, "reason": "no_change"})
        elif original not in text:
            rejected.append({**raw, "reason": "not_found"})
        elif text.count(original) != 1:
            rejected.append({**raw, "reason": "ambiguous"})
        elif len(kept) >= max_spans:
            rejected.append({**raw, "reason": "over_budget"})
        else:
            kept.append({
                "original": original,
                "suggestion": suggestion,
                "category": raw.get("category", ""),
                "severity": raw.get("severity", "minor"),
            })
    return kept, rejected


# ---------------------------------------------------------------------------
# Patching + verification
# ---------------------------------------------------------------------------

def apply_patch(text: str, original: str, suggestion: str) -> tuple[str, str]:
    """One literal substring replacement.

    Returns ``(new_text, status)`` with status ``applied`` | ``not_found`` |
    ``ambiguous`` — the text is modified only when ``original`` occurs exactly
    once, so a patch can never land in the wrong place.
    """
    n = text.count(original)
    if n == 0:
        return text, "not_found"
    if n > 1:
        return text, "ambiguous"
    return text.replace(original, suggestion, 1), "applied"


def verify_patch(
    client: Any,
    source_text: str,
    before: str,
    after: str,
) -> dict[str, Any]:
    """A/B verdict, twice, with swapped positions. Accept only on 2/2.

    A model failure during an ask (e.g. Ollama aborts a degenerate
    token-repeat loop with HTTP 500) counts as a *no* vote: the patch is not
    verified, so it is rejected and the draft stays — never crash the stage.
    """
    def ask(order: str) -> str:
        bodies = "\n".join(
            f"{label}:\n{body}" for label, body in zip(("A", "B"), order)
        )
        reply = client.generate(
            f"SOURCE:\n{source_text}\n\n{bodies}\n\nWhich translation is better (A/B)?",
            system=VERIFY_SYSTEM,
        )
        m = re.search(r"\b([AB])\b", (reply or "").upper())
        return m.group(1) if m else "?"

    try:
        vote1 = ask((before, after))      # patched translation is "B"
        vote2 = ask((after, before))      # patched translation is "A"
    except Exception as exc:
        logger.warning("verify_patch: model call failed, patch rejected (%s)", exc)
        return {"votes": [False, False], "accepted": False, "tally": 0,
                "error": str(exc)}
    votes = [vote1 == "B", vote2 == "A"]
    return {
        "votes": votes,
        "accepted": all(votes),
        "tally": sum(1 for v in votes if v),
    }


# ---------------------------------------------------------------------------
# Per-chunk refinement
# ---------------------------------------------------------------------------

def refine_key(
    chunk: dict[str, Any],
    draft_cache_key: str,
    model: str,
    prompt_version: str,
) -> str:
    sp = ["u0"]
    identity = json.dumps(
        {"section": chunk.get("section_title", ""),
         "units": [u.get("text", "") for u in chunk.get("units", [])]},
        ensure_ascii=False, sort_keys=True,
    )
    return cache_key(identity, draft_cache_key or "", model or "", prompt_version or "")


def refine_chunk(
    chunk: dict[str, Any],
    draft: dict[str, Any] | None,
    client: Any,
    refine_dir: str | Path,
    *,
    draft_cache_key: str,
    prompt_version: str,
    max_spans: int = REFINE_MAX_SPANS,
) -> dict[str, Any]:
    """Refine one drafted chunk; idempotent via the refinement cache key."""
    if not isinstance(chunk, dict):          # accept Chunk objects too
        chunk = chunk.to_dict()
    refine_dir = Path(refine_dir)
    refine_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = chunk["chunk_id"]
    key = refine_key(chunk, draft_cache_key, getattr(client, "model", ""), prompt_version)
    refined_path = refine_dir / f"{chunk_id}.refined.json"

    drilled = _load_json(refined_path)
    if drilled and drilled.get("cache_key") == key:
        logger.debug("%s: fresh refinement exists, skipping", chunk_id)
        return drilled

    record: dict[str, Any] = {
        "chunk_id": chunk_id,
        "cache_key": key,
        "draft_cache_key": draft_cache_key,
        "model": getattr(client, "model", ""),
        "prompt_version": prompt_version,
        "units": [],
        "counters": {"spans": 0, "units_changed": 0, "accepted": 0, "rejected": 0},
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not draft:
        refined_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return record

    draft_by_uid = {u.get("uid"): u for u in draft.get("units", [])}
    all_spans: list[dict[str, Any]] = []
    possible_patches: list[dict[str, Any]] = []
    for unit in chunk.get("units", []):
        uid = unit["uid"]
        d = draft_by_uid.get(uid)
        out_text = (d or {}).get("text", "") or ""
        if not d or unit.get("kind") not in TRANSLATED_KINDS or not out_text.strip():
            continue
        spans, rejected = scrub_spans(
            _estimate_or_none(client, unit["text"], out_text, max_spans=max_spans,
                              chunk_id=chunk_id, uid=uid),
            out_text, max_spans=max_spans,
        )
        all_spans.append({"uid": uid, "raw": spans})
        if rejected:
            all_spans.append({"uid": uid, "rejected": rejected})
        record["counters"]["spans"] += len(spans)
        record["counters"]["rejected"] += len(rejected)
        cur = out_text
        for span in spans:
            patched, status = apply_patch(cur, span["original"], span["suggestion"])
            if status != "applied":
                record["counters"]["rejected"] += 1
                continue
            verdict = verify_patch(client, unit["text"], cur, patched)
            entry = {
                **span,
                "uid": uid,
                "apply_status": status,
                "accepted": verdict["accepted"],
                "votes": verdict["votes"],
            }
            if verdict.get("error"):
                entry["verify_error"] = verdict["error"]
            possible_patches.append(entry)
            if verdict["accepted"]:
                cur = patched
                record["counters"]["accepted"] += 1
            else:
                record["counters"]["rejected"] += 1
                # rejected patches keep the original draft text
        changed = cur != out_text
        if changed:
            record["counters"]["units_changed"] += 1
            record["units"].append({
                "uid": uid,
                "text": cur,
                "changed": changed,
            })
    _write_spans(refine_dir, chunk_id, record, all_spans)
    _write_patches(refine_dir, chunk_id, record, possible_patches)
    refined_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_spans(refine_dir: Path, chunk_id: str, record: dict, spans: list[dict]) -> None:
    (refine_dir / f"{chunk_id}.spans.json").write_text(
        json.dumps(
            {
                "chunk_id": chunk_id,
                "cache_key": record["cache_key"],
                "model": record["model"],
                "prompt_version": record["prompt_version"],
                "spans": spans,
            },
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def _write_patches(refine_dir: Path, chunk_id: str, record: dict, patches: list[dict]) -> None:
    patch_lines, verdict_lines = [], []
    for p in patches:
        patch_lines.append({
            "chunk_id": chunk_id,
            "uid": p["uid"],
            "original": p["original"],
            "suggestion": p["suggestion"],
            "category": p.get("category", ""),
            "severity": p.get("severity", "minor"),
            "apply_status": p["apply_status"],
            "accepted": p["accepted"],
        })
        verdict_lines.append({
            "chunk_id": chunk_id,
            "uid": p["uid"],
            "original": p["original"],
            "votes": p["votes"],
            "accepted": p["accepted"],
        })
    write_records(patch_lines, refine_dir / f"{chunk_id}.patches.jsonl") \
        if patch_lines else _touch(refine_dir / f"{chunk_id}.patches.jsonl")
    write_records(verdict_lines, refine_dir / f"{chunk_id}.verdicts.jsonl") \
        if verdict_lines else _touch(refine_dir / f"{chunk_id}.verdicts.jsonl")


def _touch(path: Path) -> None:
    path.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Refined-text loading for assembly
# ---------------------------------------------------------------------------

def load_refinements(
    refine_dir: str | Path,
    drafts_meta: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Accepted refined unit texts keyed ``{chunk_id: {uid: text}}``.

    A refinement is used only when its recorded ``draft_cache_key`` still
    matches the current draft (a re-translated chunk is not refined text).
    """
    refine_dir = Path(refine_dir)
    out: dict[str, dict[str, str]] = {}
    for path in sorted(refine_dir.glob("chunk-*.refined.json")):
        data = _load_json(path)
        if not data:
            continue
        chunk_id = data.get("chunk_id") or path.name.split(".")[0]
        meta = drafts_meta.get(chunk_id)
        if not meta or meta.get("cache_key") != data.get("draft_cache_key"):
            logger.debug("%s: stale refinement, ignoring", chunk_id)
            continue
        out[chunk_id] = {
            u["uid"]: u["text"] for u in data.get("units", []) if u.get("changed")
        }
        if not out[chunk_id]:
            del out[chunk_id]
    return out


# ---------------------------------------------------------------------------
# QE-budgeted scheduling (2.3)
# ---------------------------------------------------------------------------

def plan_refinement(
    chunks: list[dict[str, Any]],
    drafts_meta: dict[str, dict[str, Any]],
    qe_by_chunk: dict[str, dict[str, Any]] | None = None,
    *,
    quality_threshold: float = 80.0,
    budget: int | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """(chunk, qe-score) to refine, worst score first.

    * chunks without a draft are skipped; chunks without a QE score are
      treated as the worst (score 0) — never silently skipped;
    * chunks scoring at/above ``quality_threshold`` are skipped;
    * ``budget`` caps how many chunks are refined (highest priority kept).
    """
    qe_by_chunk = qe_by_chunk or {}
    candidates: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        if chunk["chunk_id"] not in drafts_meta:
            continue
        rec = qe_by_chunk.get(chunk["chunk_id"])
        score = rec.get("score") if rec else None
        if score is None:
            score = 0.0
        if score >= quality_threshold:
            continue
        candidates.append((float(score), chunk))
    candidates.sort(key=lambda pair: (pair[0], pair[1]["chunk_id"]))
    plan = [(chunk, score) for score, chunk in candidates]
    if budget is not None:
        plan = plan[: max(0, budget)]
    return plan
