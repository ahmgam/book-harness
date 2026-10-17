"""
Consolidated data model for marker/datalab OCR output of a book.

Merges the two pipeline artifacts:

* ``<name>.json``       -- main content: page_info + blocks (html, images, hierarchy)
* ``<name>_meta.json``  -- table_of_contents + page_stats (QA / navigation)

Design notes
------------
* The canonical page index is the ``/page/N/...`` prefix of ``block["id"]``
  (it matches ``page_info`` keys and ``table_of_contents.page_id``).
  The ``block["page"]`` field is unreliable (many-to-one artifact) and is
  ignored.
* Inline JPEGs live in each block's ``images`` map, keyed by the id used in
  the block's ``html`` ``src`` attributes.
* Reading order: blocks grouped by (page index, original list order); the
  list order within a page is already top-to-bottom.
* Section outline is built from ``SectionHeader`` blocks using the
  ``<h1>..<h4>`` level in their html (the meta TOC has heading_level=None).

Requires: Python 3.9+ (standard library only).
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator, Optional


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class BlockType(str, Enum):
    TEXT = "Text"
    SECTION_HEADER = "SectionHeader"
    PAGE_HEADER = "PageHeader"
    PAGE_FOOTER = "PageFooter"
    EQUATION = "Equation"
    CODE = "Code"
    TABLE = "Table"
    LIST_GROUP = "ListGroup"
    FIGURE = "Figure"
    FIGURE_GROUP = "FigureGroup"
    PICTURE = "Picture"
    PICTURE_GROUP = "PictureGroup"
    TABLE_GROUP = "TableGroup"
    TABLE_OF_CONTENTS = "TableOfContents"
    CAPTION = "Caption"
    FOOTNOTE = "Footnote"
    FORM = "Form"
    OTHER = "Other"

    @classmethod
    def _missing_(cls, value: object) -> "BlockType":
        return cls.OTHER


#: Block types that repeat on every page and should be dropped when
#: rendering continuous content.
CHROME_TYPES = frozenset({BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER})


@dataclass
class Block:
    """One merged content block, normalised onto its true page."""

    id: str
    type: BlockType
    html: str
    page: int                      # derived from the id prefix (canonical)
    bbox: tuple[float, float, float, float]
    polygon: list[list[float]]
    section_hierarchy: dict[str, str]
    images: dict[str, str]         # image id -> base64 JPEG
    heading_level: Optional[int] = None   # for SectionHeader: 1..4

    @property
    def text(self) -> str:
        return html_to_text(self.html)

    @property
    def title_text(self) -> str:
        """Text without heading tags (for SectionHeader titles)."""
        return re.sub(r"</?h\d[^>]*>", "", self.html)

    def image_ids(self) -> list[str]:
        return re.findall(r"src='([^']+)'", self.html)


@dataclass
class PageInfo:
    index: int
    width: float
    height: float
    bbox: tuple[float, float, float, float]
    polygon: list[list[float]]


@dataclass
class TocEntry:
    title: str
    page: int
    polygon: list[list[float]]
    level: Optional[int] = None    # filled from the matching SectionHeader

    @property
    def text(self) -> str:
        return re.sub(r"<[^>]+>", "", self.title)


@dataclass
class OutlineNode:
    """Node of the section tree derived from SectionHeader blocks."""
    block: Block
    number: Optional[str]
    title: str
    children: list["OutlineNode"] = field(default_factory=list)
    parent: Optional["OutlineNode"] = None

    def iter_self_and_descendants(self) -> Iterator["OutlineNode"]:
        yield self
        for child in self.children:
            yield from child.iter_self_and_descendants()


@dataclass
class PageStat:
    page: int
    text_extraction_method: str
    block_counts: list
    block_metadata: dict


# --------------------------------------------------------------------------
# HTML -> text / markdown helpers
# --------------------------------------------------------------------------

class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "br":
            self.parts.append(" ")

    def get_text(self) -> str:
        text = " ".join("".join(self.parts).split())
        return text.strip()


def html_to_text(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    parser.close()
    return parser.get_text()


_MATH_RE = re.compile(r"<math([^>]*)>(.*?)</math>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _math_to_md(match: "re.Match[str]") -> str:
    attrs, tex = match.group(1), match.group(2)
    tex = tex.strip().strip("\\")
    if "display=\"block\"" in attrs:
        return f"\n\n$$ {tex} $$\n\n"
    return f"${tex}$"


def html_to_markdown(html: str, image_url_map: Optional[dict[str, str]] = None) -> str:
    """Approximate conversion of a block's html to markdown."""
    out = html
    if image_url_map is not None:
        def repl_img(m: "re.Match[str]") -> str:
            src = m.group(1)
            url = image_url_map.get(src, src)
            return f"<p><img src='{url}'/></p>"
        out = re.sub(r"<img src='([^']+)'>", repl_img, out)
    out = _MATH_RE.sub(_math_to_md, out)
    # headings
    out = re.sub(r"<h1[^>]*>(.*?)</h1>", lambda m: f"\n\n# {html_to_text(m.group(1))}\n\n", out, flags=re.DOTALL)
    out = re.sub(r"<h2[^>]*>(.*?)</h2>", lambda m: f"\n\n## {html_to_text(m.group(1))}\n\n", out, flags=re.DOTALL)
    out = re.sub(r"<h3[^>]*>(.*?)</h3>", lambda m: f"\n\n### {html_to_text(m.group(1))}\n\n", out, flags=re.DOTALL)
    out = re.sub(r"<h4[^>]*>(.*?)</h4>", lambda m: f"\n\n#### {html_to_text(m.group(1))}\n\n", out, flags=re.DOTALL)
    # list items
    out = re.sub(r"<li[^>]*>(.*?)</li>", lambda m: f"\n- {html_to_text(m.group(1))}", out, flags=re.DOTALL)
    out = re.sub(r"</?ul[^>]*>", "\n", out)
    # tables: keep as html (round-trip safe) - markdown tables rarely survive OCR layout
    out = re.sub(r"<pre>(.*?)</pre>", lambda m: f"\n```\n{unescape(m.group(1))}\n```\n", out, flags=re.DOTALL)
    out = re.sub(r"</p>\s*<p[^>]*>", "\n\n", out)
    out = re.sub(r"<p[^>]*>|</p>", "\n", out)
    out = re.sub(r"<b>(.*?)</b>", r"**\1**", out, flags=re.DOTALL)
    out = re.sub(r"<i>(.*?)</i>", r"*\1*", out, flags=re.DOTALL)
    out = _TAG_RE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return unescape(out).strip()


# --------------------------------------------------------------------------
# The consolidated Book class
# --------------------------------------------------------------------------

class Book:
    """All book data from the two OCR JSON files, with rendering support."""

    def __init__(
        self,
        blocks: list[Block],
        pages: dict[int, PageInfo],
        toc: list[TocEntry],
        page_stats: list[PageStat],
        debug_data_path: Optional[str] = None,
    ) -> None:
        self.blocks = blocks
        self.pages = pages
        self.toc = toc
        self.page_stats = page_stats
        self.debug_data_path = debug_data_path

        # --- derived indexes -------------------------------------------
        self._blocks_by_page: dict[int, list[Block]] = {}
        for b in blocks:                       # list order = reading order
            self._blocks_by_page.setdefault(b.page, []).append(b)
        self._blocks_by_id = {b.id: b for b in blocks}

        self._image_index: dict[str, tuple[Block, str]] = {}
        for b in blocks:
            for img_id in b.images:
                self._image_index[img_id] = (b, img_id)

        self.outline = self._build_outline()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_marker_json(
        cls,
        content_path: str | Path,
        meta_path: str | Path | None = None,
    ) -> "Book":
        content_path = Path(content_path)
        if meta_path is None:
            meta_path = content_path.with_name(content_path.stem + "_meta.json")

        with open(content_path, encoding="utf-8") as f:
            raw = json.load(f)

        blocks: list[Block] = []
        for rb in raw.get("blocks", []):
            # Canonical page: the /page/N/ prefix of the id.
            page = int(rb["id"].split("/")[2])
            btype = BlockType(rb["block_type"])
            heading = None
            if btype is BlockType.SECTION_HEADER:
                m = re.match(r"<h(\d)", rb["html"] or "")
                heading = int(m.group(1)) if m else None
            blocks.append(Block(
                id=rb["id"],
                type=btype,
                html=rb.get("html", ""),
                page=page,
                bbox=tuple(rb.get("bbox", (0, 0, 0, 0))),
                polygon=rb.get("polygon", []),
                section_hierarchy=rb.get("section_hierarchy", {}),
                images=rb.get("images", {}),
                heading_level=heading,
            ))

        pages: dict[int, PageInfo] = {}
        for key, info in raw.get("page_info", {}).items():
            bbox = info.get("bbox", [0, 0, 0, 0])
            pages[int(key)] = PageInfo(
                index=int(key),
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
                bbox=tuple(bbox),
                polygon=info.get("polygon", []),
            )

        toc: list[TocEntry] = []
        stats: list[PageStat] = []
        debug_path: Optional[str] = None
        if Path(meta_path).exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            for e in meta.get("table_of_contents", []):
                block = None
                for b in blocks:
                    if (b.type is BlockType.SECTION_HEADER and b.page == e["page_id"]
                            and html_to_text(b.html) == re.sub(r"<[^>]+>", "", e["title"])):
                        block = b
                        break
                toc.append(TocEntry(
                    title=e["title"],
                    page=e["page_id"],
                    polygon=e.get("polygon", []),
                    level=block.heading_level if block else None,
                ))
            for s in meta.get("page_stats", []):
                stats.append(PageStat(
                    page=s["page_id"],
                    text_extraction_method=s.get("text_extraction_method", ""),
                    block_counts=s.get("block_counts", []),
                    block_metadata=s.get("block_metadata", {}),
                ))
            debug_path = meta.get("debug_data_path")

        return cls(blocks=blocks, pages=pages, toc=toc, page_stats=stats,
                   debug_data_path=debug_path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def page(self, index: int) -> list[Block]:
        """Content blocks of one page in reading order."""
        return self._blocks_by_page.get(index, [])

    def block(self, block_id: str) -> Optional[Block]:
        return self._blocks_by_id.get(block_id)

    def image_bytes(self, image_id: str) -> Optional[bytes]:
        """Decode an inline image (lookup covers child images of groups)."""
        entry = self._image_index.get(image_id)
        if entry is None:
            return None
        return base64.b64decode(entry[0].images[image_id])

    def iter_content_blocks(
        self,
        include_chrome: bool = False,
        pages: Optional[range] = None,
    ) -> Iterator[Block]:
        for pg in sorted(self._blocks_by_page):
            if pages is not None and pg not in pages:
                continue
            for b in self._blocks_by_page[pg]:
                if not include_chrome and b.type in CHROME_TYPES:
                    continue
                yield b

    def section_of(self, block: Block, level: str = "2") -> Optional[Block]:
        """Nearest SectionHeader ancestor of a block at a given level."""
        header_id = block.section_hierarchy.get(level)
        if header_id is None:
            return None
        hdr = self._blocks_by_id.get(header_id)
        return hdr if hdr is not None and hdr.type is BlockType.SECTION_HEADER else None

    # ------------------------------------------------------------------
    # Outline
    # ------------------------------------------------------------------
    def _build_outline(self) -> list[OutlineNode]:
        roots: list[OutlineNode] = []
        stack: list[OutlineNode] = []
        for b in self.iter_content_blocks():
            if b.type is not BlockType.SECTION_HEADER:
                continue
            title = html_to_text(b.title_text)
            if not title:                      # empty header artifact
                continue
            level = b.heading_level or 1
            title = html_to_text(b.title_text)
            m = re.match(r"^(\d+(?:\.\d+)*)\b\s*", title)
            number = m.group(1) if m else None
            node = OutlineNode(block=b, number=number, title=title,
                               parent=stack[-1] if stack else None)
            while stack and (stack[-1].block.heading_level or 1) >= level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
        return roots

    def print_outline(self, max_level: int = 2) -> str:
        lines: list[str] = []
        for root in self.outline:
            for node in root.iter_self_and_descendants():
                lvl = node.block.heading_level or 1
                if lvl > max_level:
                    continue
                lines.append("  " * (lvl - 1) + (node.title or "(untitled)"))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _image_url_map(self, blocks: list[Block], mode: str,
                       image_dir: str | None) -> tuple[dict[str, str], dict[str, bytes]]:
        urls: dict[str, str] = {}
        files: dict[str, bytes] = {}
        for b in blocks:
            for img_id in b.image_ids():
                if img_id not in b.images:
                    continue
                if mode == "data-uri":
                    urls[img_id] = f"data:image/jpeg;base64,{b.images[img_id]}"
                else:
                    fname = re.sub(r"^/page/(\d+)/", f"p{int(b.page):03d}_", img_id.strip("/").replace("/", "_")) + ".jpg"
                    urls[img_id] = f"{image_dir or 'images'}/{fname}"
                    files[fname] = base64.b64decode(b.images[img_id])
        return urls, files

    def render_page_html(
        self,
        page: int,
        include_chrome: bool = False,
        images: str = "data-uri",       # "data-uri" | "files" | "omit"
        image_dir: str = "images",
    ) -> str:
        """Render one page as a self-contained-ish HTML fragment."""
        blocks = self.page(page)
        blocks = [b for b in blocks if include_chrome or b.type not in CHROME_TYPES]
        urls, files = self._image_url_map(blocks, images, image_dir)
        parts = [f'<section class="page" data-page="{page}">']
        for b in blocks:
            html = b.html
            if images == "omit":
                html = re.sub(r"<img[^>]*>", "", html)
            else:
                html = re.sub(
                    r"src='([^']+)'",
                    lambda m, b=b: "src='%s'" % urls.get(m.group(1), m.group(1)),
                    html,
                )
                for fname, data in files.items():
                    pass  # caller may persist `files` via save_images()
            parts.append(f'<div class="block {b.type.value}" id="{escape(b.id, quote=True)}">{html}</div>')
        parts.append("</section>")
        return "\n".join(parts)

    def render_html(
        self,
        include_toc: bool = True,
        include_chrome: bool = False,
        images: str = "files",
        image_dir: str = "images",
        title: str = "Flight Simulation Software",
    ) -> str:
        """Whole book as one HTML document."""
        all_blocks = list(self.iter_content_blocks(include_chrome=include_chrome))
        urls, files = self._image_url_map(all_blocks, images, image_dir)
        body: list[str] = []
        if include_toc:
            body.append("<nav id='toc'><h1>Contents</h1><ul>")
            for e in self.toc:
                if e.level and e.level > 2:
                    continue
                body.append(f"<li class='toc-l{e.level or 1}'>"
                            f"<a href='#p{e.page}'>{escape(e.text)}</a></li>")
            body.append("</ul></nav>")
        last_page: Optional[int] = None
        for b in all_blocks:
            if b.page != last_page:
                body.append(f"<section class='page' id='p{b.page}' data-page='{b.page}'>"
                            f"<!-- page {b.page} -->")
                last_page = b.page
            html = b.html
            if images == "omit":
                html = re.sub(r"<img[^>]*>", "", html)
            else:
                html = re.sub(r"src='([^']+)'",
                              lambda m: "src='%s'" % urls.get(m.group(1), m.group(1)), html)
            body.append(f"<div class='block {b.type.value}' id='{escape(b.id, quote=True)}'>{html}</div>")
        body.append("</section>")
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{escape(title)}</title>"
            "<style>"
            "body{font-family:serif;max-width:52em;margin:2em auto;line-height:1.5}"
            "section.page{border-top:1px dashed #bbb;margin-top:2em;padding-top:1em}"
            "nav#toc{column-count:2} .toc-l2{margin-left:1.5em}"
            "table{border-collapse:collapse} th,td{border:1px solid #999;padding:.2em .5em}"
            "pre{background:#f4f4f4;padding:.6em;overflow-x:auto}"
            ".PageFooter,.PageHeader{color:#777;font-size:.85em}"
            "img{max-width:100%}"
            "</style></head><body>\n" + "\n".join(body) + "\n</body></html>"
        )

    def render_markdown(
        self,
        include_chrome: bool = False,
        images: str = "files",
        image_dir: str = "images",
        pages: Optional[range] = None,
    ) -> str:
        """Whole book (or a page range) as Markdown; images as file refs or data URIs."""
        blocks = list(self.iter_content_blocks(include_chrome=include_chrome, pages=pages))
        urls, _ = self._image_url_map(blocks, images, image_dir)
        parts: list[str] = []
        last_page: Optional[int] = None
        for b in blocks:
            if b.page != last_page:
                parts.append(f"\n\n<!-- ===== page {b.page} ===== -->\n\n")
                last_page = b.page
            parts.append(html_to_markdown(b.html, image_url_map=urls))
            parts.append("\n\n")
        return "".join(parts).strip() + "\n"

    def save_html(self, path: str | Path, **kwargs: Any) -> None:
        html = self.render_html(**kwargs)
        Path(path).write_text(html, encoding="utf-8")

    def save_markdown(self, path: str | Path, save_images_to: str | Path | None = None,
                      **kwargs: Any) -> None:
        """Render markdown; optionally write the referenced images to disk."""
        blocks = list(self.iter_content_blocks(include_chrome=kwargs.get("include_chrome", False)))
        urls, files = self._image_url_map(blocks, "files", kwargs.get("image_dir", "images"))
        if save_images_to is not None:
            img_dir = Path(save_images_to)
            img_dir.mkdir(parents=True, exist_ok=True)
            for fname, data in files.items():
                (img_dir / fname).write_bytes(data)
        Path(path).write_text(self.render_markdown(**kwargs), encoding="utf-8")


# --------------------------------------------------------------------------
# CLI smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    main_json = next(base.glob("*.json"))
    if main_json.name.endswith("_meta.json"):
        main_json = next(f for f in base.glob("*.json") if not f.name.endswith("_meta.json"))
    book = Book.from_marker_json(main_json)

    print(f"pages: {book.num_pages}, blocks: {len(book.blocks)}, toc: {len(book.toc)}")
    print("\n--- outline (first 25 lines) ---")
    print("\n".join(book.print_outline(max_level=1).splitlines()[:25]))

    print("\n--- page 12 (Preface) markdown ---")
    print(book.render_markdown(pages=range(12, 13), images="omit")[:600])

    img_id = "/page/25/Figure/2"
    data = book.image_bytes(img_id)
    print(f"\nimage {img_id}: {len(data)} bytes" if data else "image not found")
