"""Assembly v1 (task 1.9).

Reassembles translated chunks back into ``book.md`` by block id:

* block order and page boundaries come from the consolidated Book model;
* translated units restore numbering and list markers;
* placeholder tokens are substituted back (a token lost in translation is
  logged and the raw token remains visible for inspection);
* passthrough blocks render from their original html via ``html_to_markdown``
  with inline images extracted once to ``images/*.jpg`` under the output dir;
* page comments ``<!-- ===== page N ===== -->`` are preserved;
* ``manifest.json`` records per-chunk model, prompt version, glossary version
  and timestamps (provenance, M3.4 preview).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from book_harness import __version__
from book_harness.book import Block, Book, BlockType, html_to_markdown, html_to_text
from book_harness.chunker import KIND_CAPTION, KIND_LIST_ITEM, KIND_TEXT, KIND_TITLE, Chunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"§(?P<kind>[A-Z]+)-(?P<n>\d+)§")


# ---------------------------------------------------------------------------
# Drafts index
# ---------------------------------------------------------------------------

class DraftBook:
    """Chunk drafts on disk -> uid-indexed translated units."""

    def __init__(self, drafts_dir: str | Path) -> None:
        self.unit_texts: dict[str, dict[str, Any]] = {}
        self.meta_by_chunk: dict[str, dict[str, Any]] = {}
        drafts_dir = Path(drafts_dir)
        for meta_path in sorted(drafts_dir.glob("chunk-*.meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.meta_by_chunk[meta["chunk_id"]] = meta
            for unit in meta.get("units", []):
                # uids are chunk-local -> namespace them
                self.unit_texts[f"{meta['chunk_id']}/{unit['uid']}"] = unit

    def chunk_ids(self) -> list[str]:
        return sorted(self.meta_by_chunk)

    def unit(self, chunk_id: str, uid: str) -> dict[str, Any] | None:
        return self.unit_texts.get(f"{chunk_id}/{uid}")


# ---------------------------------------------------------------------------
# Refinement merge (M2)
# ---------------------------------------------------------------------------

def _apply_refinements(drafts: DraftBook, refine_dir: Path) -> int:
    """Overlay verified refined unit texts onto the drafts, siloed by chunk."""
    overrides = _refinements_cache(refine_dir, drafts.meta_by_chunk)
    applied = 0
    for chunk_id, chunk_overrides in overrides.items():
        for uid, text in chunk_overrides.items():
            unit = drafts.unit_texts.get(f"{chunk_id}/{uid}")
            if unit is None:
                continue
            unit = dict(unit)
            unit["text"] = text
            unit["refined"] = True
            drafts.unit_texts[f"{chunk_id}/{uid}"] = unit
            applied += 1
    if applied:
        logger.info("refined units applied: %d (of %s chunks)", applied,
                    ", ".join(overrides) or 0)
    return applied


def _refinements_cache(
    refine_dir: Path, meta_by_chunk: dict[str, dict[str, Any]]
) -> dict[str, dict[str, str]]:
    """Fresh refinement overrides (thin wrapper around refine.load_refinements)."""
    from book_harness.refine import load_refinements

    return load_refinements(refine_dir, meta_by_chunk)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def extract_images(book: Book, images_dir: Path) -> dict[str, str]:
    """Write every inline image once; map image id -> relative path."""
    images_dir.mkdir(parents=True, exist_ok=True)
    urls: dict[str, str] = {}
    written: set[str] = set()
    for b in book.iter_content_blocks(include_chrome=True):
        for image_id, b64 in b.images.items():
            if image_id in written:
                continue
            base = image_id.strip("/").replace("/", "_").replace(":", "_")
            fname = f"p{b.page:03d}_{base}.jpg"
            (images_dir / fname).write_bytes(base64.b64decode(b64))
            urls[image_id] = f"images/{fname}"
            written.add(image_id)
    return urls


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _unit_text(book: Book, block: Block, unit: dict[str, Any],
               draft: dict[str, Any]) -> str:
    """Format one unit (from chunk data + draft) into markdown."""
    placeholders = unit.get("placeholders") or {}
    text = draft["text"]

    def sub(m: "re.Match[str]") -> str:
        return placeholders.get(m.group(0), m.group(0))

    if placeholders:
        text = _TOKEN_RE.sub(sub, text)
    kind = draft.get("kind", unit.get("kind", KIND_TEXT))
    number = draft.get("number") or unit.get("number") or ""
    if kind == KIND_TITLE:
        level = min(block.heading_level or 2, 6)
        prefix = f"{number} " if number else ""
        return f"\n\n{'#' * level} {prefix}{text}\n\n"
    if kind == KIND_LIST_ITEM:
        return f"- {text}"
    if kind == KIND_CAPTION:
        return f"\n*{text}*\n"
    return f"{text}\n\n"


def _source_unit_text(book: Book, block: Block, unit: dict[str, Any]) -> str:
    """Readable fallback when a unit has no draft (interrupted run)."""
    placeholders = unit.get("placeholders") or {}
    kind = unit.get("kind", KIND_TEXT)

    def sub(m: "re.Match[str]") -> str:
        return placeholders.get(m.group(0), m.group(0))

    if kind == KIND_TITLE:
        level = min(block.heading_level or 2, 6)
        number = unit.get("number", "")
        prefix = f"{number} " if number else ""
        return f"\n\n{'#' * level} {prefix}{unit.get('text', '')}\n\n"
    if kind == KIND_LIST_ITEM:
        return f"- {unit.get('text', '')}"
    text = unit.get("text", "")
    if placeholders:
        text = _TOKEN_RE.sub(sub, text)
    return f"{text}\n\n"


def _passthrough_text(book: Book, block: Block, urls: dict[str, str]) -> str:
    """Markdown for a non-translatable block, image refs resolved."""
    html = re.sub(
        r"src='([^']+)'",
        lambda m: "src='%s'" % urls.get(m.group(1), m.group(1)),
        block.html,
    )
    return html_to_markdown(html, image_url_map=urls) + "\n\n"


def render_book(
    book: Book,
    chunks: list[Chunk],
    drafts: DraftBook,
    out_dir: str | Path,
    book_id: str = "",
    refine_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble ``book.md`` + ``manifest.json``; returns the manifest (M3.4).

    The manifest is the full provenance record: per chunk the translating
    model, prompt version, glossary version, cache key, section, canonical
    pages, unit counts and the finished/duration timestamps; book level adds
    the pipeline version and a generation timestamp. When ``refine_dir`` is
    given, verified refinement results (M2) override the raw draft texts and
    are counted in the manifest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if refine_dir is not None:
        _apply_refinements(drafts, Path(refine_dir))
    urls = extract_images(book, out_dir / "images")

    # block-id -> (chunk, ordered unit dicts); each block lives in exactly
    # one chunk, so the first reference wins.
    by_block: dict[str, tuple[Chunk, list[dict[str, Any]]]] = {}
    for c in chunks:
        dict_units = {u.uid: u.to_dict() for u in c.units}
        for ref in c.blocks:
            if ref["block_id"] in by_block:
                continue
            units = [dict_units[uid] for uid in ref["units"] if uid in dict_units]
            by_block[ref["block_id"]] = (c, units)

    lines: list[str] = []
    last_page: int | None = None

    for b in book.iter_content_blocks():
        if b.page != last_page:
            last_page = b.page
            lines.append(f"\n\n<!-- ===== page {b.page} ===== -->\n\n")
        chunk, units = by_block.get(b.id, (None, []))
        drafted = False
        for unit in units:
            draft = drafts.unit(chunk.chunk_id, unit["uid"]) if chunk else None
            if draft:
                drafted = True
                lines.append(_unit_text(book, b, unit, draft))
            else:
                lines.append(_source_unit_text(book, b, unit))
        if not units:
            lines.append(_passthrough_text(book, b, urls))

    md = "".join(lines).strip()
    md = re.sub(r"\n{4,}", "\n\n\n", md) + "\n"
    (out_dir / "book.md").write_text(md, encoding="utf-8")

    translated_units = sum(
        1 for meta in drafts.meta_by_chunk.values() for _ in meta.get("units", [])
    )
    refined_units = sum(
        1 for u in drafts.unit_texts.values() if u.get("refined")
    )
    refined_chunks = 0
    if refine_dir is not None:
        refinements = _refinements_cache(
            Path(refine_dir), drafts.meta_by_chunk
        )
        refined_chunks = len(refinements)
    manifest: dict[str, Any] = {
        "book_id": book_id or out_dir.parent.name,
        "pipeline_version": __version__,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pages": book.num_pages,
        "blocks": len(book.blocks),
        "images": len(urls),
        "chunks_total": len(chunks),
        "chunks_translated": len(drafts.meta_by_chunk),
        "chunks_refined": refined_chunks,
        "units_translated": translated_units,
        "units_refined": refined_units,
        "chunks": [],
    }
    for chunk_id in drafts.chunk_ids():
        meta = drafts.meta_by_chunk[chunk_id]
        chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
        manifest["chunks"].append({
            "chunk_id": chunk_id,
            "section_title": meta.get("section") or (
                chunk.section_path_string() if chunk else None),
            "pages": meta.get("pages") or (chunk.pages if chunk else None),
            "units": len(meta.get("units", [])),
            "model": meta.get("model"),
            "prompt_version": meta.get("prompt_version"),
            "glossary_version": meta.get("glossary_version"),
            "cache_key": meta.get("cache_key"),
            "source_hash": meta.get("source_hash"),
            "finished": meta.get("finished"),
            "duration_sec": meta.get("duration_sec"),
        })
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
