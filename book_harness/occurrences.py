"""Occurrence capture (context-informed translation selection, PLAN §9).

While the ``terms`` stage forms the candidate list it also documents *where*
each extracted term occurs in the book: one deterministic scan over the
in-memory chunks (``llm_terms.chunk_texts``) finds each candidate verbatim
(case-insensitive, whitespace/hyphen-tolerant) and records the chunk's id,
its page and section plus a text window around the match.

The records are stored in the candidate schema itself — ``occurrences: [
{chunk_id, page, section, context}, ...]`` inside ``candidates.jsonl`` — so
the post-glossary **consistency stage** can replay the book's own usage
(with concrete chunk references) to the assist model without re-reading
the book.

No model is involved here: the scan is fully deterministic and free, and
stage resume is unaffected (re-running re-computes the same records).
"""

from __future__ import annotations

import re
from typing import Any

from book_harness import config

#: A chunk text is OCR prose; matches must sit on token boundaries so
#: "speed" does not match inside "airspeed" (lookbehind / lookahead pair).
_WORD_START = r"(?<!\w)"
_WORD_END = r"(?!\w)"


def term_pattern(term: str) -> re.Pattern | None:
    """Case-insensitive, whitespace/hyphen-tolerant matcher for one term.

    Tokens may drift apart in OCR text ("flight  dynamics", "flight-
    dynamics"), so runs of non-word characters collapse to ``[^\\w]+``;
    everything else is literal. ``None`` when the term has no word tokens.
    """
    tokens = re.findall(r"\w+", term)
    if not tokens:
        return None
    body = r"[^\w]+".join(re.escape(tok) for tok in tokens)
    return re.compile(_WORD_START + body + _WORD_END,
                      re.IGNORECASE | re.UNICODE)


def _windows(text: str, pattern: re.Pattern, context_chars: int) -> list[str]:
    """Clipped context windows (one per distinct match region, sorted)."""
    out: list[str] = []
    last_end = -1
    for m in pattern.finditer(text):
        if m.start() <= last_end:
            continue                      # overlapping with the previous match
        last_end = m.end()
        lo = max(0, m.start() - context_chars)
        hi = min(len(text), m.end() + context_chars)
        window = text[lo:hi]
        if lo > 0:
            window = "…" + window.lstrip()
        if hi < len(text):
            window = window.rstrip() + "…"
        out.append(" ".join(window.split()))
    return out


def attach_occurrences(
    candidates: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    max_per_term: int | None = None,
    context_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Add ``occurrences`` to every candidate record (in place, returned).

    ``chunks`` are ``llm_terms.chunk_texts`` records
    ``{chunk_id, first_page, pages, section, text, source_hash}``. Each
    occurrence is ``{chunk_id, page, section, context}``: the chunk id the
    translation stage uses, the chunk's first page, section path and a
    ``context_chars`` window around the match (newlines collapsed). Capped at
    ``max_per_term`` first-seen occurrences in reading order.
    """
    ctx = config.OCCURRENCE_CONTEXT_CHARS if context_chars is None else context_chars
    cap = config.OCCURRENCES_MAX_PER_TERM if max_per_term is None else max_per_term
    for candidate in candidates:
        pattern = term_pattern(candidate.get("term") or "")
        records: list[dict[str, Any]] = []
        if pattern is not None:
            for chunk in chunks:
                text = chunk.get("text") or ""
                if not text or len(records) >= cap:
                    continue
                for window in _windows(text, pattern, ctx):
                    records.append({
                        "chunk_id": chunk.get("chunk_id"),
                        "page": chunk.get("first_page"),
                        "section": chunk.get("section"),
                        "context": window,
                    })
                    if len(records) >= cap:
                        break
        candidate["occurrences"] = records
    return candidates
