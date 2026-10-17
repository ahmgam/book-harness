"""Section-aligned chunking + inline placeholder protection (task 1.7).

Design (PLAN.md catch #8/#9):
* a chunk = consecutive content blocks sharing one outline section
  (ancestor resolved via ``section_hierarchy``), split at paragraph
  (Text-block) boundaries under a char ceiling;
* every chunk records its full ordered block list (translated blocks carry
  *units*, passthrough blocks keep their html) so reassembly by block id is
  exact;
* inline non-translatables inside ``Text`` blocks (LaTeX from ``<math>``,
  URLs, bracketed citations) become placeholder tokens ``§EQ0§`` … before any
  model sees the text; Equation/Code/Table/image blocks never generate units.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from book_harness.book import Block, Book, BlockType, html_to_text
from book_harness.config import CHAR_CEILING, MIN_CHUNK_CHARS

# block types that produce translation units; everything content-y else is
# passthrough, chrome is dropped by Book.iter_content_blocks already.
TRANSLATABLE_TYPES = {
    BlockType.TEXT,
    BlockType.SECTION_HEADER,
    BlockType.CAPTION,
    BlockType.FOOTNOTE,
    BlockType.LIST_GROUP,
}

KIND_TEXT = "text"
KIND_TITLE = "title"
KIND_CAPTION = "caption"
KIND_LIST_ITEM = "list_item"

_MATH_RE = re.compile(r"<math([^>]*)>(.*?)</math>", re.DOTALL)
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE)
_MAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CITE_RE = re.compile(r"\[\d+(?:[,;]\s*\d+)*\]")     # [12], [1, 4]
_TOKEN_RE = re.compile(r"§(?P<kind>[A-Z]+)-(?P<n>\d+)§")


@dataclass
class Unit:
    """One translatable span (text paragraph, heading, caption, list item)."""

    uid: str                       # "u0"…, unique within the chunk
    block_id: str
    kind: str
    text: str = ""                 # placeholder-protected source text
    number: str = ""               # preserved section number for titles
    placeholders: dict[str, str] = field(default_factory=dict)  # token -> original
    item_index: int | None = None  # for list items inside a ListGroup
    item_marker: str = "•"

    def to_dict(self) -> dict[str, Any]:
        out = {
            "uid": self.uid,
            "block_id": self.block_id,
            "kind": self.kind,
            "text": self.text,
        }
        if self.number:
            out["number"] = self.number
        if self.placeholders:
            out["placeholders"] = dict(self.placeholders)
        if self.item_index is not None:
            out["item_index"] = self.item_index
            out["item_marker"] = self.item_marker
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Unit":
        return cls(
            uid=d["uid"],
            block_id=d["block_id"],
            kind=d["kind"],
            text=d["text"],
            number=d.get("number", ""),
            placeholders=dict(d.get("placeholders", {})),
            item_index=d.get("item_index"),
            item_marker=d.get("item_marker", "•"),
        )


@dataclass
class Chunk:
    index: int                      # 1-based
    section_id: str                 # section anchor block id
    section_path: list[dict]        # [{number, title}, ...] root→leaf
    pages: list[int]
    blocks: list[dict[str, Any]]    # ordered refs: {block_id, page, type, units}
    units: list[Unit]

    @property
    def chunk_id(self) -> str:
        return f"chunk-{self.index:04d}"

    @property
    def source_text(self) -> str:
        # complete & deterministic input identity for the cache key
        return "\x1e".join([self.section_path_string()] + [u.text for u in self.units])

    def section_path_string(self) -> str:
        return " > ".join(
            (f"{s.get('number') or ''} {s.get('title')}".strip()) for s in self.section_path
        ) or "(front matter)"

    def source_hash(self) -> str:
        return hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "section_id": self.section_id,
            "section_path": self.section_path,
            "section_title": self.section_path_string(),
            "pages": self.pages,
            "source_hash": self.source_hash(),
            "blocks": self.blocks,
            "units": [u.to_dict() for u in self.units],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            index=d["index"],
            section_id=d.get("section_id", "front"),
            section_path=list(d.get("section_path", [])),
            pages=list(d.get("pages", [])),
            blocks=list(d.get("blocks", [])),
            units=[Unit.from_dict(u) for u in d.get("units", [])],
        )


# ---------------------------------------------------------------------------
# Text extraction with placeholder protection
# ---------------------------------------------------------------------------

class _Protector:
    """Collects placeholder tokens for non-translatable spans in plain text."""

    def __init__(self) -> None:
        self.map: dict[str, str] = {}
        self._n = 0

    def add(self, kind: str, value: str) -> str:
        token = f"§{kind}-{self._n:02d}§"
        self._n += 1
        self.map[token] = value
        return token

    def math_from_html(self, tex: str) -> str:
        return self.add("EQ", f"${tex}$")

    def restore(self, text: str) -> str:
        def sub(m: "re.Match[str]") -> str:
            original = self.map.get(m.group(0))
            return original if original is not None else m.group(0)
        return _TOKEN_RE.sub(sub, text)


def protect_text(html: str, protector: _Protector) -> str:
    """Plain text of a Text/Caption html, with non-translatables tokenized."""
    out = _MATH_RE.sub(
        lambda m: protector.math_from_html(m.group(2).strip()), html
    )
    plain = html_to_text(out)

    def sub(kind: str, m: "re.Match[str]") -> str:
        return protector.add(kind, m.group(0))

    plain = _URL_RE.sub(lambda m: sub("URL", m), plain)
    plain = _MAIL_RE.sub(lambda m: sub("URL", m), plain)
    plain = _CITE_RE.sub(lambda m: sub("CIT", m), plain)
    return plain


# ---------------------------------------------------------------------------
# Extraction & grouping
# ---------------------------------------------------------------------------

def _split_heading_number(title: str) -> tuple[str, str]:
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$", title)
    if not m:
        return "", title.strip()
    return m.group(1), m.group(2).strip()


def extract_units(book: Book, block: Block) -> list[Unit]:
    """Units produced by one translatable block (several for lists)."""
    protector = _Protector()
    if block.type is BlockType.TEXT:
        text = protect_text(block.html, protector)
        if text.strip():
            return [Unit(
                "u0", block.id, KIND_TEXT, text=text,
                placeholders=protector.map,
            )]
        return []
    if block.type in (BlockType.CAPTION, BlockType.FOOTNOTE):
        text = protect_text(block.html, protector)
        if text.strip():
            return [Unit(
                "u0", block.id, KIND_CAPTION, text=text,
                placeholders=protector.map,
            )]
        return []
    if block.type is BlockType.SECTION_HEADER:
        title = html_to_text(block.title_text)
        if not title.strip():
            return []
        number, rest = _split_heading_number(title)
        # placeholder-protect the title too (e.g. §EQ§ in a heading)
        protected = protect_text(f"<p>{rest}</p>", protector) if (
            _MATH_RE.search(rest) or _URL_RE.search(rest)
        ) else rest
        return [Unit(
            "u0", block.id, KIND_TITLE, text=protected,
            number=number, placeholders=protector.map,
        )]
    if block.type is BlockType.LIST_GROUP:
        items = re.findall(r"<li[^>]*>(.*?)</li>", block.html, flags=re.DOTALL | re.IGNORECASE)
        units: list[Unit] = []
        for i, raw in enumerate(items):
            own = _Protector()
            text = protect_text(raw, own)
            if text.strip():
                units.append(Unit(
                    f"u{i}", block.id, KIND_LIST_ITEM, text=text,
                    placeholders=own.map, item_index=i, item_marker="•",
                ))
        return units
    return []


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------

def section_key(block: Block) -> str:
    """Section anchor: nearest (narrowest) available ancestor header id."""
    for level in ("4", "3", "2", "1"):
        ref = block.section_hierarchy.get(level)
        if ref:
            return ref
    return "front"


def section_path(book: Book, block: Block) -> list[dict]:
    """Root→leaf breadcrumb {number, title} of a block's outline ancestors."""
    path: list[dict] = []
    for level in ("1", "2", "4"):  # levels observed in this OCR output
        hdr_id = block.section_hierarchy.get(level)
        if not hdr_id:
            continue
        hdr = book.block(hdr_id)
        if hdr is None or hdr.type is not BlockType.SECTION_HEADER:
            continue
        title = html_to_text(hdr.title_text)
        if not title.strip():
            continue
        number, rest = _split_heading_number(title)
        path.append({"number": number or None, "title": rest})
    dedup: list[dict] = []
    for item in path:
        if not dedup or dedup[-1] != item:
            dedup.append(item)
    return dedup or [{"number": None, "title": "Front matter"}]


def block_weight(book: Book, block: Block) -> int:
    """Translatable char weight of a block (units only + passthrough bonus)."""
    weight = sum(len(u.text) for u in extract_units(book, block))
    if block.type in {
        BlockType.CODE, BlockType.EQUATION, BlockType.TABLE,
        BlockType.TABLE_GROUP, BlockType.TABLE_OF_CONTENTS,
    }:
        weight += min(len(block.html), 400)
    return weight


def plan_chunks(
    book: Book,
    char_ceiling: int = CHAR_CEILING,
    char_floor: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Group content blocks into section-aligned translation chunks."""
    groups: list[tuple[str, list[Block]]] = []
    for b in book.iter_content_blocks():
        if not groups or groups[-1][0] != section_key(b):
            groups.append((section_key(b), [b]))
        else:
            groups[-1][1].append(b)

    chunks: list[Chunk] = []
    for anchor_id, blocks in groups:
        path = section_path(book, blocks[0])
        current: list[Block] = []
        size = 0

        def flush(current_blocks: list[Block]) -> None:
            if not current_blocks:
                return
            units: list[Unit] = []
            block_refs: list[dict[str, Any]] = []
            pages: list[int] = []
            counter = 0
            for blk in current_blocks:
                blk_units = extract_units(book, blk)
                uids = []
                for u in blk_units:
                    u.uid = f"u{counter}"
                    counter += 1
                    units.append(u)
                    uids.append(u.uid)
                block_refs.append({
                    "block_id": blk.id,
                    "page": blk.page,
                    "type": blk.type.value,
                    "units": uids,
                })
                if blk.page not in pages:
                    pages.append(blk.page)
            chunks.append(Chunk(
                index=0,
                section_id=anchor_id,
                section_path=list(path),
                pages=pages,
                blocks=block_refs,
                units=units,
            ))

        for b in blocks:
            weight = block_weight(book, b)
            if current and size + weight > char_ceiling:
                flush(current)
                size = 0
                current = []
            current.append(b)
            size += weight
        flush(current)

    chunks = _merge_dwarfs(chunks, char_floor)
    _renumber_all(chunks)
    return chunks


def _merge_dwarfs(chunks: list[Chunk], floor: int) -> list[Chunk]:
    """Fuse chunks below the floor into their predecessor of the same section
    so the model isn't fed two-line fragments."""
    merged: list[Chunk] = []
    for c in chunks:
        unit_chars = sum(len(u.text) for u in c.units)
        if merged and unit_chars < floor and c.section_id == merged[-1].section_id:
            prev = merged[-1]
            for ref in c.blocks:
                for blkref in prev.blocks:
                    if blkref["block_id"] == ref["block_id"]:
                        blkref["units"].extend(ref["units"])
                        break
                else:
                    prev.blocks.append(ref)
            for pg in c.pages:
                if pg not in prev.pages:
                    prev.pages.append(pg)
            prev.units.extend(c.units)
            _renumber(prev)
            continue
        merged.append(c)
    return merged


def _renumber(chunk: Chunk) -> None:
    counter = 0
    for u in chunk.units:
        u.uid = f"u{counter}"
        counter += 1
    uid_set = {u.uid for u in chunk.units}
    for ref in chunk.blocks:
        ref["units"] = [uid for uid in ref["units"] if uid in uid_set]


def _renumber_all(chunks: list[Chunk]) -> None:
    for i, c in enumerate(chunks, 1):
        c.index = i


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_chunks(chunks: Iterable[Chunk], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in chunks:
        p = out_dir / f"{c.chunk_id}.json"
        p.write_text(
            json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths.append(p)
    return paths


def load_chunks(chunk_dir: str | Path) -> list[Chunk]:
    paths = sorted(Path(chunk_dir).glob("chunk-*.json"))
    chunks = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            chunks.append(Chunk.from_dict(json.load(f)))
    return chunks
