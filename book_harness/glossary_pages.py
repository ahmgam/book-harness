"""Book-glossary harvesting (task 6.2).

If the source book already contains its own back-matter glossary, those
pages are a curated vocabulary list. ``pages.find_glossary_pages`` detects
them from the OCR metadata (glossary section headers); this module hands
each detected page to the assist model, parses the term list it returns,
and mixes the results into the vocabulary-extraction candidates.

Assure-fail contract: when glossary pages were detected, the assist model
is required. A missing client or a failed/empty model reply raises
``RuntimeError`` and stops the ``terms`` stage — a re-run retries it.

Output records reuse the candidate schema (``terms.py``):
``{term, freq, df, page, score, signal}`` with ``signal = "book-glossary"``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from book_harness import config
from book_harness.book import Book, html_to_text

#: Leading bullets/markers the model (or the OCR) may leave on a term line.
_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:[-*•·]|\d{1,3}[.)]\s|[a-z][.)]\s|[أ-ي][.)]\s)+"
)

#: Separator leftovers to trim off a term line's ends.
_TRIM_CHARS = " \t\r\n—–:;,.|*\"'“”„«»`"


def _dedupe_key(term: str) -> str:
    return unicodedata.normalize("NFKC", term.strip().lower())


def parse_term_lines(reply: str) -> list[str]:
    """Parse an assist-model reply into clean, deduped term strings.

    Keeps first-seen casing; drops sentences (length cap), stray bullets and
    annotation leftovers ("term : definition" keeps only "term").
    """
    terms: list[str] = []
    seen: set[str] = set()
    for raw in reply.splitlines():
        line = _LEADING_MARKER_RE.sub("", raw)
        # drop an inline "term : definition" tail if the model added one
        if " — " in line:
            line = line.split(" — ", 1)[0]
        elif " – " in line:
            line = line.split(" – ", 1)[0]
        elif ":" in line and line.index(":") <= config.GLOSSARY_MAX_TERM_CHARS:
            head, tail = line.split(":", 1)
            if 0 < len(tail.strip()) and len(head.strip()) > 0:
                line = head
        line = line.strip(_TRIM_CHARS)
        if not line or len(line) > config.GLOSSARY_MAX_TERM_CHARS:
            continue
        key = _dedupe_key(line)
        if key in seen:
            continue
        seen.add(key)
        terms.append(line)
    return terms


def page_text(book: Book, page: int) -> str:
    """OCR text of one page's content blocks (one line per block)."""
    kinds = [html_to_text(b.html) for b in book.iter_content_blocks(
        pages=range(page, page + 1)
    )]
    return "\n".join(k for k in kinds if k.strip())


def build_candidates(per_page: dict[int, list[str]]) -> list[dict[str, Any]]:
    """Candidate records from ``{page: terms}`` (usable in ``candidates.jsonl``)."""
    stats: dict[str, list[int]] = {}
    display: dict[str, str] = {}
    for page in sorted(per_page):
        for term in per_page[page]:
            key = _dedupe_key(term)
            if key not in stats:
                stats[key] = []
                display[key] = term
            stats[key].append(page)
    return [
        {
            "term": display[key],
            "freq": len(pages),
            "df": len(pages),
            "page": pages[0],
            "score": config.GLOSSARY_PAGE_SCORE,
            "signal": config.GLOSSARY_PAGE_SIGNAL,
        }
        for key, pages in stats.items()
    ]


def extract_page_terms(
    book: Book,
    page_ranges: list[range],
    assist_client: Any,
    *,
    logger=None,
) -> list[dict[str, Any]]:
    """Ask the assist model for the term list on each detected glossary page.

    ``page_ranges`` usually comes from ``pages.find_glossary_pages``. Every
    page must yield at least one term — an empty reply on a detected glossary
    page means the model failed to read it, so the stage fails loudly.

    Returns one candidate record per distinct term.
    """
    if not page_ranges:
        return []
    if assist_client is None:
        raise RuntimeError(
            "book glossary pages detected but no assist client is available; "
            "the terms stage requires it for glossary-page harvesting"
        )
    pages = sorted({p for r in page_ranges for p in r})
    per_page: dict[int, list[str]] = {}

    from book_harness.progress import work_bar

    with work_bar(pages, desc="glossary harvest", unit="page") as pbar:
        for page in pbar:
            pbar.set_postfix_str(f"page {page}")
            text = page_text(book, page)
            if not text.strip():
                raise RuntimeError(
                    f"glossary page {page}: no text extracted from OCR blocks"
                )
            prompt = config.GLOSSARY_HARVEST_USER.format(text=text)
            try:
                reply = assist_client.generate(
                    prompt,
                    system=config.GLOSSARY_HARVEST_SYSTEM,
                    options=dict(config.GENERATE_OPTIONS),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"glossary page {page}: assist model call failed: {exc}"
                ) from exc
            terms = parse_term_lines(reply)
            if not terms:
                raise RuntimeError(
                    f"glossary page {page}: assist model returned no terms: "
                    f"{reply[:120]!r}"
                )
            per_page[page] = terms
            if logger:
                logger.info("glossary page %d: %d terms", page, len(terms))
    candidates = build_candidates(per_page)
    if logger:
        logger.info(
            "book-glossary harvest: %d terms over pages %s",
            len(candidates), pages,
        )
    return candidates


def merge_candidates(
    base: list[dict[str, Any]], extra: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Additively merge harvested candidates into the mined list.

    NFKC-lower dedupe (the ``terms.py`` convention); existing candidates win,
    harvested extras are appended, scores are never recomputed. When a
    duplicate is dropped, its ``occurrences`` (documented contexts) are
    merged into the surviving record (deduped on page+context) so the
    selection stage still sees the full usage picture.
    """
    by_key: dict[str, dict[str, Any]] = {
        _dedupe_key(c["term"]): c for c in base
    }
    merged = list(base)
    for c in extra:
        key = _dedupe_key(c["term"])
        survivor = by_key.get(key)
        if survivor is not None:
            known = {
                (o.get("page"), o.get("section"), o.get("context"))
                for o in (survivor.get("occurrences") or [])
            }
            for occ in (c.get("occurrences") or []):
                ident = (occ.get("page"), occ.get("section"), occ.get("context"))
                if ident not in known:
                    known.add(ident)
                    survivor.setdefault("occurrences", []).append(occ)
            continue
        by_key[key] = c
        merged.append(c)
    return merged


def write_page_terms(candidates: list[dict[str, Any]], path: str | Path) -> int:
    """Write ``glossary-page-terms.jsonl`` (the harvest provenance artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(candidates)
