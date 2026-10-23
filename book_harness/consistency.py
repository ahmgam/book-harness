"""Glossary consistency stage (post-glossary, one rendering per term).

After the glossary stage resolves candidates against arabterm, every
certified entry with more than one *available* rendering is handed to the
assist model together with the chunks in which the term occurs — documented
during term formation (``occurrences.py``, recorded in ``candidates.jsonl``
with the chunk id). The model picks the rendering that fits the book's own
usage; deterministic dictionary priority keeps applying only as the
fallback.

The stage re-writes ``glossary.yaml`` with a single ``term_ar`` per entry —
a **new glossary version** is issued on every decision change — and
downstream stages (translate, compliance, audits) enforce that one
rendering.

Best-effort contract: the assist model may fail. A missing client, a failed
factory, or an unparseable reply falls back to the deterministic pick with
a warning — the stage never stops the pipeline because adjudication failed.

Artifacts (``02_glossary``):

* ``consistency-cache.jsonl`` — one line per decided term, keyed by term +
  options hash + model + prompt id, so re-runs replay decisions instead of
  repeating model calls; a prompt/model change invalidates them.
* ``consistency-decisions.jsonl`` — the auditable decision log (term,
  options presented, choice, and how it was reached).

Entry updates: ``term_ar`` to the chosen rendering, ``decided_by`` to
``"llm_context"``, ``status`` to ``STATUS_LLM_CHOSEN`` and a reason citing
the chunk count. ``alternatives`` stays untouched, so the decision is
reviewable in ``glossary.yaml`` and downstream consumers keep their
contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from book_harness import config
from book_harness.glossary import (
    STATUS_LLM_CHOSEN,
    STATUS_UNCERTIFIED,
    GlossaryEntry,
)
from book_harness.glossary_pages import _dedupe_key
from book_harness.llm_terms import strip_think
from book_harness.progress import work_bar


class _Decision:
    """Book-keeping for one selectable term (options + cache identity)."""

    def __init__(self, entry: GlossaryEntry) -> None:
        self.term = entry.get("term_en") or ""
        self.options = _entry_options(entry)

    @property
    def key(self) -> str:
        """Cache key: term + menu + prompt id (occurrences excluded —
        context capture may improve without invalidating paid decisions)."""
        blob = "\x1f".join(
            [*self._lines(), config.CONSISTENCY_PROMPT_ID]
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _lines(self) -> list[str]:
        return [self.term, *(f"{ar}|{src or ''}" for ar, src in self.options)]

    def record(
        self, chosen: str | None, via: str, occurrences: int, model: str,
        why: str = "",
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "term_en": self.term,
            "options": [ar for ar, _ in self.options],
            "chosen": chosen,
            "model": model,
            "prompt": config.CONSISTENCY_PROMPT_ID,
            "via": via,
            "occurrences": occurrences,
            "key": self.key,
        }
        if why:
            out["why"] = why
        return out


def _entry_options(entry: GlossaryEntry) -> list[tuple[str, str | None]]:
    """The decision menu: current rendering first, then alternatives.

    The same Arabic string from different dictionaries collapses to the
    first (highest-priority) listing; menu entries are already atomic
    renderings (compound rows were atomized by the glossary builder).
    """
    options: list[tuple[str, str | None]] = [
        (entry.get("term_ar") or "", entry.get("source"))
    ]
    for idx, alt in enumerate(entry.get("alternatives") or []):
        if not alt or any(alt == seen for seen, _ in options):
            continue
        sources = entry.get("alternative_sources") or []
        options.append((alt, sources[idx] if idx < len(sources) else None))
    return options


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _option_lines(options: list[tuple[str, str | None]]) -> str:
    """Numbered option list; the certifying dictionary when known."""
    lines = []
    for idx, (ar, source) in enumerate(options, start=1):
        origin = f" (source: {source})" if source else ""
        lines.append(f"{idx}. `{ar}`{origin}")
    return "\n".join(lines)


def _occurrence_line(rec: dict[str, Any]) -> str:
    """One occurrence: chunk id, section, page, context window."""
    cid = rec.get("chunk_id")
    page = rec.get("page")
    section = rec.get("section")
    where = " · ".join(
        part for part in (
            f"chunk {cid}" if cid else None,
            section if section else None,
            f"page {page}" if page is not None else None,
        ) if part
    ) or "chunk ?"
    return f"- [{where}] {rec.get('context', '')}".rstrip()


def _occurrence_block(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(no occurrences documented; decide on domain usage alone)"
    lines = []
    for rec in records:
        lines.append(_occurrence_line(rec))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def choose_translations(
    entries: list[GlossaryEntry],
    candidates: list[dict[str, Any]],
    assist_factory: Any = None,            # lazily builds the assist client
    out_dir: str | Path | None = None,
    *,
    domains: tuple[str, ...] = config.DEFAULT_DOMAINS,
    logger: Any = logging.getLogger(__name__),
) -> tuple[list[GlossaryEntry], dict[str, int]]:
    """Choose one rendering per term using the chunks it occurs in.

    ``assist_factory`` is lazy: the assist client is only built when a term
    actually needs choosing. Certified entries with no alternatives and
    uncertified entries skip — there is nothing to decide between.

    Returns ``(entries, counts)`` with counts ``{"chosen": n, "fallback": n,
    "skipped": n}``; selected entries are updated in place (one ``term_ar``
    per entry; ``alternatives`` preserved for audit).
    """
    by_term: dict[str, dict[str, Any]] = {
        _dedupe_key(c["term"]): c for c in candidates if c.get("term")
    }
    decisions = _load_decisions(_cache_path(out_dir))
    client: Any = None
    model = ""
    counts = {"chosen": 0, "fallback": 0, "skipped": 0}
    records: list[dict[str, Any]] = []

    with work_bar(entries, desc="consistency", unit="entry", total=len(entries)) as pbar:
        for entry in pbar:
            status = entry.get("status")
            if status == STATUS_UNCERTIFIED:
                counts["skipped"] += 1
                continue
            dec = _Decision(entry)
            if len(dec.options) < 2:
                counts["skipped"] += 1
                continue
            pbar.set_postfix_str(dec.term)
            occurrences = (by_term.get(_dedupe_key(dec.term), {}).get(
                "occurrences") or [])
            cached = decisions.get(dec.key)
            if cached:
                chosen, via, why = cached["chosen"], "cache", ""
                model = str(cached.get("model") or model)
            else:
                if client is None:
                    try:
                        client = assist_factory() if assist_factory else None
                    except Exception as exc:
                        logger.warning(
                            "assist client unavailable for glossary consistency "
                            "(%s); deterministic fallback", exc,
                        )
                if client is None:
                    counts["fallback"] += 1
                    records.append(dec.record(
                        None, "fallback", len(occurrences), "",
                        why="no assist client available",
                    ))
                    continue
                model = str(getattr(client, "model", "") or "")
                chosen, via, why = _ask_model(
                    client, dec, occurrences, domains, logger
                )
            if chosen is None:
                counts["fallback"] += 1
                entry["reason"] = (
                    f"glossary consistency: {why or 'no choice'}; "
                    "kept highest dictionary priority"
                )
            else:
                counts["chosen"] += 1
                entry["term_ar"] = chosen
                entry["decided_by"] = "llm_context"
                entry["status"] = STATUS_LLM_CHOSEN
                entry["reason"] = (
                    f"selected from {len(dec.options)} options using "
                    f"{len(occurrences)} occurrence chunk(s)"
                )
            records.append(dec.record(chosen, via, len(occurrences), model, why))
        pbar.set_description_str(
            "consistency (chosen: {chosen}, fallback: {fallback}, "
            "skipped: {skipped})".format(**counts)
        )

    if records:
        if out_dir is not None:
            _append_records(Path(out_dir) / config.CONSISTENCY_DECISIONS_FILE,
                            records)
            _update_cache(_cache_path(out_dir), decisions, records)
    logger.info(
        "glossary consistency: %d chosen, %d fallback, %d skipped",
        counts["chosen"], counts["fallback"], counts["skipped"],
    )
    return entries, counts


# ---------------------------------------------------------------------------
# Model call + reply parsing
# ---------------------------------------------------------------------------

def _ask_model(
    client: Any,
    dec: _Decision,
    occurrences: list[dict[str, Any]],
    domains: tuple[str, ...],
    logger: Any,
) -> tuple[str | None, str, str]:
    """One model call; ``(chosen_ar, via, reason)`` — chosen None = fallback."""
    prompt = config.CONSISTENCY_USER.format(
        term=dec.term,
        domains="/".join(domains),
        options=_option_lines(dec.options),
        occurrences=_occurrence_block(occurrences),
    )
    try:
        reply = client.generate(
            prompt,
            system=config.CONSISTENCY_SYSTEM,
            options=dict(config.CONSISTENCY_GENERATE_OPTIONS),
        )
    except Exception as exc:
        logger.warning(
            "glossary consistency for %r failed (%s); deterministic fallback",
            dec.term, exc,
        )
        return None, "fallback", f"model call failed ({exc})"
    picked = _parse_reply(strip_think(reply), dec.options)
    if picked is None:
        logger.warning(
            "glossary consistency for %r: ambiguous/invalid reply %r; "
            "deterministic fallback", dec.term, reply[:120],
        )
        return None, "fallback", "ambiguous model reply"
    return picked, "llm", ""


def _parse_reply(
    reply: str, options: list[tuple[str, str | None]]
) -> str | None:
    """Extract the chosen option from a reply (option number, else text)."""
    reply = (reply or "").strip().strip("`\"' ")
    for token in re.findall(r"\b\d+\b", reply):
        idx = int(token)
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
    hits = [ar for ar, _ in options if ar and ar in reply]
    if len(hits) == 1:
        return hits[0]
    return None


# ---------------------------------------------------------------------------
# Cache + audit log
# ---------------------------------------------------------------------------

def _cache_path(out_dir: str | Path | None) -> Path | None:
    return Path(out_dir) / config.CONSISTENCY_CACHE_FILE if out_dir else None


def _load_decisions(path: Path | None) -> dict[str, dict[str, str]]:
    """Decisions keyed by cache key; records of another prompt id ignored."""
    if not path or not path.exists():
        return {}
    decisions: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("prompt") != config.CONSISTENCY_PROMPT_ID
                    or not rec.get("chosen") or not rec.get("key")):
                continue
            decisions[rec["key"]] = rec
    return decisions


def _update_cache(
    path: Path | None, existing: dict[str, dict[str, str]],
    records: list[dict[str, Any]],
) -> None:
    """Append decisions the cache does not know (crash-safe: one line)."""
    fresh = [r for r in records if r["chosen"] and r["key"] not in existing]
    if not (path and fresh):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in fresh:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _append_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
