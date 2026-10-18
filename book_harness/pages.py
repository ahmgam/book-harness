"""Page-level helpers for the M5 AI formatting stage.

Two jobs:

* classify each page into a ``PageArchetype`` -- primary classification uses
  the page's BlockType composition from the consolidated ``Book``; a
  markdown-heuristic fallback (``archetype_from_markdown``) covers standalone
  runs where the OCR JSON pair was not kept;
* split ``book.md`` into per-page slices at the page comments the assembler
  writes (``<!-- ===== page N ===== -->``) and re-join them, so the formatting
  stage can cache and verify page by page.

Requires: Python 3.11+ (standard library only).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from enum import Enum
from typing import Iterable, Optional

from book_harness.book import BlockType, CHROME_TYPES


# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

#: Pages up to and including this index are front matter candidates (cover,
#: title, copyright) when they carry almost no text.
FRONT_MATTER_PAGES = 2
#: Maximum total text chars on a page still considered front matter.
FRONT_MATTER_MAX_TEXT = 400
#: The trailing zone in which an index page can be recognised.
INDEX_ZONE = 10
#: Minimum "term, number" lines before a trailing page is an index page.
INDEX_MIN_LINES = 6
#: Minimum enumerated/bullet lines before a page is list-heavy without an
#: explicit ListGroup block.
LIST_MIN_LINES = 3


# ---------------------------------------------------------------------------
# Archetype model
# ---------------------------------------------------------------------------

class PageArchetype(str, Enum):
    """The formatting-personality of one page (PLAN.md, milestone M5)."""

    FRONT_MATTER = "front_matter"
    TOC = "toc"
    INDEX = "index"
    LIST = "list"
    FOOTNOTE = "footnote"
    EQUATION = "equation"
    CODE = "code"
    TABLE = "table"
    FIGURE = "figure"
    NARRATIVE = "narrative"
    DEFAULT = "default"


# ---------------------------------------------------------------------------
# Composition-based classification (primary)
# ---------------------------------------------------------------------------

#: Marker regex: a content line that looks like an index/TOC entry
#: ("term, 12" / "Design …… 5") and nothing else.
_INDEX_LINE_RE = re.compile(r"^\s*\S.{0,60}?[\s,،.…·]+\d{1,4}\s*$")

#: Marker regex: an inline "term, 12" fragment (for texts whose line breaks
#: were flattened away, e.g. by ``html_to_text``).
_INDEX_FRAG_RE = re.compile(r"[^\s,،\d][^\s,،]{0,58}[,،]\s*\d{1,4}(?:\s|$)")

#: Marker regex: a line that begins like a list item.
_BULLET_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s|\d{1,3}[.)]\s|[أ-ي][.)]\s|\(?[a-z][.)]\s)\S"
)


def _index_like(texts: Iterable[str]) -> bool:
    blob = "\n".join(texts)
    lines = [ln for ln in blob.splitlines() if ln.strip()]
    if len(lines) >= INDEX_MIN_LINES:
        hits = sum(1 for ln in lines if len(ln) <= 70 and _INDEX_LINE_RE.match(ln))
        if hits >= INDEX_MIN_LINES and hits * 2 >= len(lines):
            return True
    # flattened text (html_to_text joins everything with spaces): fall back
    # to counting "term, n" fragments
    return len(_INDEX_FRAG_RE.findall(blob)) >= INDEX_MIN_LINES


def _bullet_like(texts: Iterable[str]) -> bool:
    lines = [ln for t in texts for ln in t.splitlines() if ln.strip()]
    hits = sum(1 for ln in lines if _BULLET_LINE_RE.match(ln))
    return hits >= LIST_MIN_LINES


def archetype_from_blocks(
    types: Iterable[BlockType],
    texts: Iterable[str] = (),
    *,
    page: int,
    num_pages: int = 0,
    front_matter_pages: int = FRONT_MATTER_PAGES,
    index_zone: int = INDEX_ZONE,
) -> PageArchetype:
    """Classify one page from its content-block types (+ block texts).

    Priority (first match wins):
    ``front_matter → toc → index → list → footnote → equation → code →
    table → figure → narrative → default``.

    ``texts`` are the plain texts of the same blocks (used for the index and
    the lost-list heuristics); without them only explicit block types fire.
    """
    counts: Counter[BlockType] = Counter(types)
    texts = list(texts)

    # front matter: first pages, almost no text, no explicit structural block
    if (page <= front_matter_pages
            and not counts[BlockType.TABLE_OF_CONTENTS]
            and not counts[BlockType.SECTION_HEADER]
            and not (counts[BlockType.TABLE] or counts[BlockType.TABLE_GROUP])
            and not (counts[BlockType.CODE] and counts[BlockType.EQUATION])
            and sum(len(t) for t in texts) < FRONT_MATTER_MAX_TEXT):
        return PageArchetype.FRONT_MATTER

    if counts[BlockType.TABLE_OF_CONTENTS]:
        return PageArchetype.TOC

    in_index_zone = num_pages > 0 and page > num_pages - index_zone
    if in_index_zone and _index_like(texts):
        return PageArchetype.INDEX

    if (counts[BlockType.LIST_GROUP]
            or (counts[BlockType.TEXT] and _bullet_like(texts))):
        return PageArchetype.LIST

    if counts[BlockType.FOOTNOTE]:
        return PageArchetype.FOOTNOTE
    if counts[BlockType.EQUATION]:
        return PageArchetype.EQUATION
    if counts[BlockType.CODE]:
        return PageArchetype.CODE
    if counts[BlockType.TABLE] or counts[BlockType.TABLE_GROUP]:
        return PageArchetype.TABLE
    if (counts[BlockType.FIGURE] or counts[BlockType.FIGURE_GROUP]
            or counts[BlockType.PICTURE] or counts[BlockType.PICTURE_GROUP]):
        return PageArchetype.FIGURE
    if counts[BlockType.TEXT] or counts[BlockType.SECTION_HEADER]:
        return PageArchetype.NARRATIVE
    return PageArchetype.DEFAULT


def classify_book(book) -> dict[int, PageArchetype]:
    """Archetype per page from the consolidated Book model.

    Chrome blocks (headers/footers) never influence the classification.
    """
    by_page: dict[int, list] = defaultdict(list)
    for b in book.blocks:                    # list order = reading order
        by_page[b.page].append(b)
    num_pages = book.num_pages
    out: dict[int, PageArchetype] = {}
    for pg, blocks in by_page.items():
        content = [b for b in blocks if b.type not in CHROME_TYPES]
        if not content:
            out[pg] = PageArchetype.DEFAULT
            continue
        out[pg] = archetype_from_blocks(
            (b.type for b in content),
            (b.text for b in content),
            page=pg,
            num_pages=num_pages,
        )
    return out


# ---------------------------------------------------------------------------
# Markdown-heuristic classification (fallback, standalone mode)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_FOOTNOTE_RE = re.compile(r"^\[\^\w+\]:", re.MULTILINE)
_EQ_TOKEN_RE = re.compile(r"§\w+-?\d+§")
_DISPLAY_MATH_RE = re.compile(r"\$\$")
_MD_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_TOC_TITLE_RE = re.compile(r"^\s*#{1,3}\s*(contents|المحتويات|فهرس)\s*$",
                           re.MULTILINE | re.IGNORECASE)
_LEADER_RE = re.compile(r"[.…]{2,}")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_ATX_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)


def archetype_from_markdown(
    page_md: str,
    *,
    page: int,
    num_pages: int = 0,
    front_matter_pages: int = FRONT_MATTER_PAGES,
    index_zone: int = INDEX_ZONE,
) -> PageArchetype:
    """Classify one page from its rendered markdown alone.

    Fallback for standalone runs without the OCR JSON pair; weaker than the
    composition-based classifier and only kept in sync with its priority
    order. Heuristic signals: fenced blocks, list markers, footnote
    definitions, placeholder/display-math tokens, markdown tables, image
    links, index- and TOC-like line shapes.
    """
    page_md = _HTML_COMMENT_RE.sub("", page_md)
    lines = [ln.rstrip() for ln in page_md.splitlines() if ln.strip()]
    body_len = sum(len(ln) for ln in lines)

    # front matter: early + tiny, no real headings
    if (page <= front_matter_pages and body_len < FRONT_MATTER_MAX_TEXT
            and not _IMAGE_RE.search(page_md)
            and not _ATX_HEADING_RE.search(page_md)):
        return PageArchetype.FRONT_MATTER

    # toc: many short leader-terminated or numbered-tail lines
    numbered = [ln for ln in lines
                if len(ln) <= 70 and _INDEX_LINE_RE.match(ln)]
    toc_like = (len(numbered) >= INDEX_MIN_LINES
                and any(_LEADER_RE.search(ln) for ln in lines))
    if toc_like or _TOC_TITLE_RE.search(page_md):
        return PageArchetype.TOC

    # index: trailing zone, many "term, N" lines, no leaders
    in_index_zone = num_pages > 0 and page > num_pages - index_zone
    if in_index_zone and len(numbered) >= INDEX_MIN_LINES:
        return PageArchetype.INDEX

    if _bullet_like(lines):
        return PageArchetype.LIST

    if _FOOTNOTE_RE.search(page_md):
        return PageArchetype.FOOTNOTE
    if _DISPLAY_MATH_RE.search(page_md) or _EQ_TOKEN_RE.search(page_md):
        return PageArchetype.EQUATION
    if _FENCE_RE.search(page_md):
        return PageArchetype.CODE
    if len(_MD_TABLE_RE.findall(page_md)) >= 3:
        return PageArchetype.TABLE
    if _IMAGE_RE.search(page_md):
        return PageArchetype.FIGURE
    if _ATX_HEADING_RE.search(page_md) or body_len:
        return PageArchetype.NARRATIVE
    return PageArchetype.DEFAULT


def classify_markdown(
    book_md: str,
    num_pages: int = 0,
) -> dict[int, PageArchetype]:
    """Archetype per page for a standalone markdown-only run."""
    return {
        page: archetype_from_markdown(text, page=page, num_pages=num_pages)
        for page, text in split_pages(book_md)[0].items()
    }


# ---------------------------------------------------------------------------
# book.md <-> per-page slices
# ---------------------------------------------------------------------------

_PAGE_COMMENT_RE = re.compile(
    r"^[ \t]*<!--\s*=+\s*page\s+(\d+)\s*=+\s*-->[ \t]*$", re.MULTILINE
)


def split_pages(book_md: str) -> tuple[dict[int, str], str]:
    """Split ``book.md`` at the assembler's page comments.

    Returns ``(pages, head)``: a ``page number -> page markdown`` mapping
    (each page's text with the leading/trailing blank lines stripped; exactly
    what a formatter should rewrite) and any comment-free head found before
    the first page comment (normally ``""``).
    """
    matches = list(_PAGE_COMMENT_RE.finditer(book_md))
    if not matches:
        return {}, book_md.strip("\n")
    head = book_md[: matches[0].start()].strip("\n")
    pages: dict[int, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(book_md)
        pages[int(m.group(1))] = book_md[m.end(): end].strip("\n")
    return pages, head


def join_pages(pages: dict[int, str], head: str = "") -> str:
    """Re-join per-page slices into canonical ``book.md`` form.

    Inverse of :func:`split_pages` for assembler-canonical input: separators
    are normalised to the single canonical comment form; intra-page text is
    preserved verbatim apart from the stripped boundary blank lines.
    """
    parts: list[str] = []
    if head:
        parts.append(head.strip("\n") + "\n\n")
    for pg in sorted(pages):
        text = pages[pg].strip("\n")
        parts.append(f"<!-- ===== page {pg} ===== -->\n\n{text}\n\n")
    return "".join(parts).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    md_path = sys.argv[1] if len(sys.argv) > 1 else "runs/*/07_output/book.md"
    import glob as _glob
    paths = _glob.glob(md_path)
    if not paths:
        raise SystemExit(f"no file matches {md_path}")
    book_md = open(paths[0], encoding="utf-8").read()
    pages, head = split_pages(book_md)
    print(f"{paths[0]}: {len(pages)} pages, head={bool(head)}")
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
        archetype = archetype_from_markdown(pages[n], page=n)
        print(f"--- page {n} -> {archetype.value} ---")
        print(pages[n][:600])
