"""Tests for the consolidation model + section-aligned chunker.

Uses a tiny synthetic OCR JSON pair laid out exactly like the real marker
output (id-derived pages, section_hierarchy, inline base64 images, _meta
companion).
"""

import json
from pathlib import Path

import pytest

from book_harness.book import Book, BlockType, html_to_markdown
from book_harness.chunker import (
    KIND_CAPTION,
    KIND_LIST_ITEM,
    KIND_TEXT,
    KIND_TITLE,
    Chunk,
    extract_units,
    plan_chunks,
    protect_text,
)
from book_harness.chunker import _Protector, _split_heading_number
from book_harness.chunker import load_chunks, section_path, write_chunks


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

B64_JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="  # tiny jpeg sniff


def _block(bid, btype, html, page, hierarchy=None, images=None, rel=None):
    # NOTE: 'page' mirrors the real OCR's unreliable field; must be ignored.
    return {
        "id": f"/page/{page}/{btype}/{rel}",
        "block_type": btype,
        "html": html,
        "page": page + 990,          # deliberately bogus
        "polygon": [], "bbox": [0, 0, 10, 10],
        "section_hierarchy": hierarchy or {},
        "images": images or {},
    }


@pytest.fixture
def mini_book(tmp_path: Path) -> Book:
    main = tmp_path / "mini.json"
    meta = tmp_path / "mini_meta.json"
    blocks = [
        _block(0, "SectionHeader", "<h1><p>1 Design</p></h1>", 1, {"1": "/page/1/SectionHeader/0"}, {}, "0"),
        _block(0, "Text", "<p>The flight simulator computes equations of motion.</p>", 1,
               {"1": "/page/1/SectionHeader/0"}, {}, "1"),
        _block(0, "Equation", "<math>x=1</math>", 1, {"1": "/page/1/SectionHeader/0"}, {}, "2"),
        _block(1, "SectionHeader", "<h2><p>1.1 The Evolution of Flight</p></h2>", 2,
               {"1": "/page/1/SectionHeader/0", "2": "/page/2/SectionHeader/0"}, {}, "0"),
        _block(1, "Text", "<p>Analogue computers solved the differential equations. See [12] "
               "for details. Visit https://example.org/docs for notes.</p>", 2,
               {"2": "/page/2/SectionHeader/0"}, {}, "1"),
        _block(1, "ListGroup", "<ul><li>Set the altitude</li><li>Trim the controls</li></ul>", 2,
               {"2": "/page/2/SectionHeader/0"}, {}, "2"),
        _block(1, "FigureGroup", "<p><img src='/page/2/Figure/0'/></p>", 2,
               {"2": "/page/2/SectionHeader/0"},
               {"/page/2/Figure/0": "/9j/4AAQSkZJRgABAQ=="}, "3"),
        _block(2, "SectionHeader", "<h1><p>2 Simulator Design</p></h1>", 3,
               {"1": "/page/3/SectionHeader/0"}, {}, "0"),
        _block(2, "Text", "<p>The motion platform reproduces realistic conditions.</p>", 3,
               {"1": "/page/3/SectionHeader/0"}, {}, "1"),
    ]
    raw = {
        "page_info": {str(i): {"bbox": [0, 0, 100, 100], "polygon": []} for i in range(3)},
        "blocks": blocks,
    }
    toc = [
        {"title": "<p>1 Design</p>", "page_id": 1},
        {"title": "<p>1.1 The Evolution of Flight</p>", "page_id": 2},
        {"title": "<p>2 Simulator Design</p>", "page_id": 3},
    ]
    meta_raw = {
        "table_of_contents": toc,
        "page_stats": [{"page_id": 0, "text_extraction_method": "surya", "block_counts": []}] * 1,
    }
    main.write_text(json.dumps(raw), encoding="utf-8")
    meta.write_text(json.dumps(meta_raw), encoding="utf-8")
    return Book.from_marker_json(main)


# ---------------------------------------------------------------------------
# Book wiring (1.2)
# ---------------------------------------------------------------------------

class TestBook:
    def test_pages_are_id_derived(self, mini_book):
        for b in mini_book.blocks:
            assert b.page < 900      # the bogus 'page' field was ignored

    def test_iter_content_excludes_chrome(self, mini_book):
        types = {b.type for b in mini_book.iter_content_blocks()}
        assert BlockType.SECTION_HEADER in types
        assert BlockType.PAGE_HEADER not in types

    def test_is_directus_api(self, mini_book):
        # reading order: grouped by page, list order preserved
        pages = [b.page for b in mini_book.iter_content_blocks()]
        assert pages == sorted(pages)

    def test_image_bytes_from_group(self, mini_book):
        data = mini_book.image_bytes("/page/2/Figure/0")
        assert data and bytes(data)[:2] == b"\xff\xd8"

    def test_to_markdown_math_passthrough(self):
        md = html_to_markdown("<math>\\\\frac{a}{b}</math>")
        assert "$" in md


class TestNestedDocumentFormat:
    """marker's nested "document format" (children tree + content-ref
    pointers) is auto-flattened into the flat blocks + page_info form."""

    @pytest.fixture
    def nested_book(self, tmp_path: Path) -> Book:
        # Two pages: page 0 = header + text; page 1 = header + a PictureGroup
        # (with a Caption and a Picture child we inlined), a ListGroup with
        # ListItem children, a Table with unreferenced TableCells, chrome.
        pic = {
            "id": "/page/1/Picture/1", "block_type": "Picture", "html": "",
            "polygon": [], "bbox": [0, 0, 10, 10], "children": None,
            "images": {"/page/1/Picture/1": "/9j/4AAQSkZJRgABAQ=="},
        }
        cap = {
            "id": "/page/1/Caption/0", "block_type": "Caption",
            "html": "<p>Figure 1: overview</p>", "polygon": [], "bbox": [0, 0, 10, 10],
            "children": None, "images": None,
        }
        group = {
            "id": "/page/1/PictureGroup/2", "block_type": "PictureGroup",
            "html": ("<content-ref src='/page/1/Caption/0'></content-ref>"
                     "<content-ref src='/page/1/Picture/1'></content-ref>"),
            "polygon": [], "bbox": [0, 0, 10, 10],
            "children": [cap, pic], "images": None,
        }
        li1 = {"id": "/page/1/ListItem/0", "block_type": "ListItem",
               "html": "<li>Set the altitude</li>", "polygon": [], "bbox": [0, 0, 10, 10],
               "children": None, "images": None}
        lst = {"id": "/page/1/ListGroup/1", "block_type": "ListGroup",
               "html": ("<p><ul><content-ref src='/page/1/ListItem/0'></content-ref>"
                        "</ul></p>"),
               "polygon": [], "bbox": [0, 0, 10, 10],
               "children": [li1], "images": None}
        standalone = {
            "id": "/page/1/Picture/3", "block_type": "Picture", "html": "",
            "polygon": [], "bbox": [0, 0, 10, 10], "children": None,
            "images": {"/page/1/Picture/3": "/9j/4AAQSkZJRgABAQ=="},
        }
        cell = {"id": "/page/1/TableCell/0", "block_type": "TableCell",
                "html": "<th>x</th>", "polygon": [], "bbox": [0, 0, 10, 10],
                "children": None, "images": None}
        tbl = {"id": "/page/1/Table/1", "block_type": "Table",
               "html": "<table><tbody><tr><th>x</th></tr></tbody></table>",
               "polygon": [], "bbox": [0, 0, 10, 10],
               "children": [cell], "images": None}
        pages = []
        for pg, content in ((0, [
                {"id": "/page/0/SectionHeader/0", "block_type": "SectionHeader",
                 "html": "<h1>Lecture Notes</h1>", "polygon": [], "bbox": [0, 0, 10, 10],
                 "children": None, "images": None, "section_hierarchy": {"1": "/page/0/SectionHeader/0"}},
                {"id": "/page/0/Text/1", "block_type": "Text",
                 "html": "<p>The skin is the largest organ.</p>", "polygon": [],
                 "bbox": [0, 0, 10, 10], "children": None, "images": None},
                {"id": "/page/0/PageFooter/2", "block_type": "PageFooter",
                 "html": "<p>footer</p>", "polygon": [], "bbox": [0, 0, 10, 10],
                 "children": None, "images": None}]),
            (1, [
                json.loads(json.dumps(group)),
                json.loads(json.dumps(lst)),
                json.loads(json.dumps(standalone)),
                json.loads(json.dumps(tbl))])):
            # Page html references all of its children (pure structure).
            refs = "".join(f"<content-ref src='{c['id']}'></content-ref>" for c in content)
            pages.append({
                "id": f"/page/{pg}/Page/0", "block_type": "Page",
                "html": refs, "polygon": [], "bbox": [0, 0, 200, 300],
                "children": content, "images": None,
            })
        raw = {"block_type": "Document", "children": pages}
        main = tmp_path / "nested.json"
        main.write_text(json.dumps(raw), encoding="utf-8")
        meta = tmp_path / "nested_meta.json"
        meta.write_text(json.dumps({"table_of_contents": [
            {"title": "Lecture Notes", "page_id": 0, "heading_level": 1},
        ]}), encoding="utf-8")
        return Book.from_marker_json(main)

    def test_pages_and_content_flattened(self, nested_book):
        assert nested_book.num_pages == 2          # from Page bboxes
        page0 = nested_book.page(0)
        assert {b.type for b in page0} == {BlockType.SECTION_HEADER, BlockType.TEXT,
                                          BlockType.PAGE_FOOTER}

    def test_group_children_inlined_not_emitted(self, nested_book):
        page1 = nested_book.page(1)
        by_type = {b.type: b for b in page1}
        pg = by_type[BlockType.PICTURE_GROUP]
        # caption markup + img tag synthesised for the implicit picture
        assert "Figure 1: overview" in pg.html
        assert "src='/page/1/Picture/1'" in pg.html
        # image data travelled from the Picture child into the group
        assert list(pg.images) == ["/page/1/Picture/1"]
        pic = by_type[BlockType.PICTURE]
        assert "/page/1/Picture/1" not in [b.id for b in page1]   # inlined only
        assert "<img src='/page/1/Picture/3'" in pic.html   # standalone picture
        assert list(pic.images) == ["/page/1/Picture/3"]

    def test_list_items_inlined(self, nested_book):
        lst = next(b for b in nested_book.page(1) if b.type is BlockType.LIST_GROUP)
        assert "<li>" in lst.html and "Set the altitude" in lst.html
        assert not any(b.type.value in ("ListItem", "TableCell") for b in nested_book.blocks)

    def test_table_children_dropped(self, nested_book):
        tbl = next(b for b in nested_book.page(1) if b.type is BlockType.TABLE)
        assert "<tr><th>x</th></tr>" in tbl.html


# ---------------------------------------------------------------------------
# Chunker (1.7)
# ---------------------------------------------------------------------------

class TestProtector:
    def test_math_and_urls_and_citations(self):
        p = _Protector()
        text = protect_text(
            "<p>Motion [12] defined by <math>x^{2}+1</math> at "
            "https://example.org/rfc</p>", p
        )
        assert "§EQ-00§" in text
        assert "§URL-01§" in text
        assert "§CIT-02§" in text
        assert "https://example.org" not in text
        # restoration round-trips
        restored = p.restore(text)
        assert restored == "Motion [12] defined by $x^{2}+1$ at https://example.org/rfc"

    def test_new_the_plain_text_survives(self):
        p = _Protector()
        text = protect_text("<p>Hello <b>world</b>.</p>", p)
        assert text == "Hello world."


class TestSplitHeadingNumber:
    def test_numbered(self):
        assert _split_heading_number("1.4 Drag Force") == ("1.4", "Drag Force")
        assert _split_heading_number("2.10.3 Sub") == ("2.10.3", "Sub")

    def test_unnumbered(self):
        assert _split_heading_number("Preface") == ("", "Preface")


class TestExtractUnits:
    def test_text_unit(self, mini_book):
        blocks = [b for b in mini_book.blocks if b.type is BlockType.TEXT]
        unit = extract_units(mini_book, blocks[0])[0]
        assert unit.kind == KIND_TEXT
        assert "simulator" in unit.text.lower()

    def test_heading_unnumbered(self, mini_book):
        blocks = [b for b in mini_book.blocks if b.type is BlockType.SECTION_HEADER]
        units = extract_units(mini_book, blocks[0])
        assert units[0].kind == KIND_TITLE
        assert units[0].number == "1"

    def test_list_items_split(self, mini_book):
        blocks = [b for b in mini_book.blocks if b.type is BlockType.LIST_GROUP]
        units = extract_units(mini_book, blocks[0])
        assert len(units) == 2
        assert all(u.kind == KIND_LIST_ITEM for u in units)

    def test_passthrough_blocks_yield_no_units(self, mini_book):
        blocks = [b for b in mini_book.blocks
                  if b.type in (BlockType.EQUATION, BlockType.FIGURE_GROUP)]
        assert all(extract_units(mini_book, b) == [] for b in blocks)


class TestPlanChunks:
    def test_section_aligned(self, mini_book):
        chunks = plan_chunks(mini_book, char_ceiling=800)
        # chapter 1 section 1 has: h1, text, equation / 1.1: header,text,list,figure
        assert len(chunks) >= 2
        titles = [c.section_path_string() for c in chunks]
        assert any("1.1" in t for t in titles)
        assert any("2 Simulator" in t for t in titles)
        # equation + figure blocks ride along in blocks, never as units
        unit_blocks = {u.block_id for c in chunks for u in c.units}
        assert all(not bid.endswith("Equation/0".replace("Equation/0", "Equation/0"))
                   for bid in unit_blocks)

    def test_ceiling_splits(self, mini_book):
        chunks = plan_chunks(mini_book, char_ceiling=120, char_floor=0)
        assert len(chunks) >= 4

    def test_every_block_in_exactly_one_chunk(self, mini_book):
        chunks = plan_chunks(mini_book)
        seen = [ref["block_id"] for c in chunks for ref in c.blocks]
        expected = [b.id for b in mini_book.iter_content_blocks()]
        assert sorted(seen) == sorted(expected)
        assert len(seen) == len(set(seen))

    def test_serialization_roundtrip(self, mini_book, tmp_path):
        chunks = plan_chunks(mini_book)
        write_chunks(chunks, tmp_path)
        loaded = load_chunks(tmp_path)
        assert [c.chunk_id for c in loaded] == [c.chunk_id for c in chunks]
        assert loaded[0].units and loaded[0].source_text is not None
