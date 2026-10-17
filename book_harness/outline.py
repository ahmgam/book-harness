"""Outline repair (task 3.7).

The OCR ``h``-tags occasionally mis-nest sections (a ``1.2.1`` header tagged
``<h2>`` at the same level as the ``1.2`` that contains it, bare chapter-number
headers with no ``<h>`` tag at all). Structure is re-derived from the *section
numbering pattern* in the title itself — ``N`` → depth 1, ``N.M`` → depth 2,
``N.M.K`` → depth 3 — which is far more reliable than the OCR's markup guess.

Rules:

* only ``SectionHeader`` blocks (empty-header artifacts skipped);
* a title starting with a hierarchical number like ``1``, ``2.3`` or ``2.3.1``
  gets ``heading_level`` = number depth, overriding the OCR ``<hN>`` level;
* continuity guard: each multi-part number must extend a section already seen
  (``1.2.1`` may only follow something anchored at ``1.2``/``1``), so stray
  numerals ("1941", exercise/reference numbers) never rewrite the outline;
* unnumbered titles (``Introduction``, ``Appendix A``, …) keep their OCR
  level unchanged — no numbering signal, no guess;
* repair is idempotent, deterministic and pure over the in-memory :class:`Book`.
"""

from __future__ import annotations

import re
from typing import Any

from book_harness.book import BlockType, Book, html_to_text

#: ``1`` / ``1.2`` / ``1.2.3`` at the start of a title (trailing dot tolerated).
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*(.*)$")


def number_depth(title: str) -> int | None:
    """Depth implied by a leading section number, or None for unnumbered
    titles."""
    m = _NUMBERED_RE.match(title.strip())
    if not m:
        return None
    return len(m.group(1).split("."))


def repair_outline(book: Book, *, apply: bool = True) -> list[dict[str, Any]]:
    """Re-derive heading depth from section-numbering patterns (task 3.7).

    Mutates ``SectionHeader.heading_level`` in place and rebuilds
    ``book.outline``. Returns one record per *changed* block:
    ``{block_id, page, number, title, ocr_level, repaired_level}``.
    """
    changes: list[dict[str, Any]] = []
    seen: set[str] = set()        # section numbers seen so far
    changed_any = False
    for b in book.iter_content_blocks():
        if b.type is not BlockType.SECTION_HEADER:
            continue
        title = html_to_text(b.title_text).strip()
        m = _NUMBERED_RE.match(title)
        if not m or not m.group(2):
            continue  # unnumbered title keeps its OCR level
        number = m.group(1)
        if not _continuous(number, seen):
            continue  # stray numeral (not an extending section number)
        seen.add(number)
        depth = len(number.split("."))
        if b.heading_level == depth:
            continue
        changes.append({
            "block_id": b.id,
            "page": b.page,
            "number": number,
            "title": title,
            "ocr_level": b.heading_level,
            "repaired_level": depth,
        })
        if apply:
            b.heading_level = depth
            changed_any = True
    if apply and changed_any:
        book.outline = book._build_outline()
    return changes


def _continuous(number: str, seen: set[str]) -> bool:
    """True when ``N.M.K`` extends sections already anchored in the book."""
    parts = number.split(".")
    if len(parts) == 1:
        return True  # depth-1 chapters are always trusted
    parent = ".".join(parts[:-1])
    if parent in seen:
        return True
    # ``1.3`` following only ``1.2`` is fine: the ancestor *chain* is already
    # anchored once any sibling of the parent was seen
    return any(
        s == ".".join(parts[:i]) or s.startswith(".".join(parts[:i]) + ".")
        for i in range(1, len(parts))
        for s in seen
    )


def repair_summary(changes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact record for ``00_source/outline-repair.json``."""
    return {
        "sections_repaired": len(changes),
        "changes": changes,
    }
