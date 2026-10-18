"""Tests for page archetypes + book.md page slicing (M5 task 5.1)."""

import textwrap

import pytest

from book_harness.book import Block, BlockType, PageInfo
from book_harness.book import Book
from book_harness.pages import (
    PageArchetype,
    archetype_from_blocks,
    archetype_from_markdown,
    classify_book,
    classify_markdown,
    join_pages,
    split_pages,
)


# ---------------------------------------------------------------------------
# Fixture: a Book assembled from Block dataclass instances directly
# ---------------------------------------------------------------------------

def _block(bid, btype, text, page):
    return Block(
        id=bid, type=btype, html=f"<p>{text}</p>", page=page,
        bbox=(0, 0, 10, 10), polygon=[], section_hierarchy={}, images={},
    )


@pytest.fixture
def mixed_book() -> Book:
    """Pages wired to hit the distinct archetypes of the priority list."""
    blocks = [
        # 1 front matter: cover/title, only Other blocks + tiny text
        _block("/page/1/Text/0", BlockType.TEXT, "Flight Simulation Software", 1),
        # 2 TOC
        _block("/page/2/TableOfContents/0", BlockType.TABLE_OF_CONTENTS,
               "1 Design ... 5", 2),
        # 3 narrative body
        _block("/page/3/SectionHeader/0", BlockType.SECTION_HEADER,
               "1 Simulator Design", 3),
        _block("/page/3/Text/0", BlockType.TEXT,
               "The flight simulator integrates models of aircraft motion.", 3),
        # 4 list
        _block("/page/4/ListGroup/0", BlockType.LIST_GROUP,
               "Set the altitude Trim the controls", 4),
        # 5 footnote
        _block("/page/5/Footnote/0", BlockType.FOOTNOTE,
               "Analogue computers were widely used in the 1970s.", 5),
        # 6 equation
        _block("/page/6/Equation/0", BlockType.EQUATION, "x = y", 6),
        # 7 code
        _block("/page/7/Code/0", BlockType.CODE, "def f(): pass", 7),
        # 8 table
        _block("/page/8/Table/0", BlockType.TABLE, "a b c", 8),
        # 9 figure
        _block("/page/9/FigureGroup/0", BlockType.FIGURE_GROUP, "fig", 9),
        # 10 narrative
        _block("/page/10/Text/0", BlockType.TEXT,
               "Long-body text continuing the section narrative.", 10),
        # 11 index entries (trailing page)
        _block("/page/11/Text/0", BlockType.TEXT,
               "altitude, 12\nairspeed, 17\nautopilot, 3\ncontrol laws, 41\n"
               "gain margin, 55\nphase margin, 60", 11),
    ]
    pages = {i: PageInfo(index=i, width=100, height=100, bbox=(0, 0, 100, 100),
                         polygon=[])
             for i in range(1, 12)}
    return Book(blocks=blocks, pages=pages, toc=[], page_stats=[])


# ---------------------------------------------------------------------------
# Composition-based classification (primary)
# ---------------------------------------------------------------------------

class TestArchetypeFromBlocks:
    def test_priority_order(self):
        # one page carrying several structural types -> strongest trigger wins
        arc = archetype_from_blocks(
            [BlockType.TEXT, BlockType.LIST_GROUP, BlockType.FIGURE], page=9)
        assert arc is PageArchetype.LIST

    def test_each_explicit_trigger(self):
        cases = [
            (BlockType.TABLE_OF_CONTENTS, PageArchetype.TOC),
            (BlockType.LIST_GROUP, PageArchetype.LIST),
            (BlockType.FOOTNOTE, PageArchetype.FOOTNOTE),
            (BlockType.EQUATION, PageArchetype.EQUATION),
            (BlockType.CODE, PageArchetype.CODE),
            (BlockType.TABLE, PageArchetype.TABLE),
            (BlockType.TABLE_GROUP, PageArchetype.TABLE),
            (BlockType.FIGURE, PageArchetype.FIGURE),
            (BlockType.PICTURE_GROUP, PageArchetype.FIGURE),
            (BlockType.TEXT, PageArchetype.NARRATIVE),
            (BlockType.OTHER, PageArchetype.DEFAULT),
        ]
        for btype, expected in cases:
            assert archetype_from_blocks([btype], page=9) is expected

    def test_early_textual_page_is_front_matter(self):
        assert archetype_from_blocks(
            [BlockType.TEXT], ["Flight Simulation Software"],
            page=2, num_pages=8) is PageArchetype.FRONT_MATTER

    def test_dense_early_page_is_not_front_matter(self):
        long_text = "This is a real paragraph of body text. " * 30
        assert archetype_from_blocks(
            [BlockType.TEXT], [long_text],
            page=2, num_pages=8) is PageArchetype.NARRATIVE

    def test_index_found_in_trailing_zone(self):
        entries = "altitude, 12\nairspeed, 17\nautopilot, 3\ncorrection, 8\n" \
                  "drag, 21\nelevator, 30"
        assert archetype_from_blocks(
            [BlockType.TEXT], [entries],
            page=200, num_pages=205) is PageArchetype.INDEX

    def test_no_index_outside_trailing_zone(self):
        entries = "altitude, 12\nairspeed, 17\nautopilot, 3\ncorrection, 8\n" \
                  "drag, 21\nelevator, 30"
        assert archetype_from_blocks(
            [BlockType.TEXT], [entries],
            page=50, num_pages=205) is PageArchetype.NARRATIVE

    def test_lost_list_without_list_group(self):
        text = "1. Set the altitude\n2. Trim the controls\n3. Run the test"
        assert archetype_from_blocks(
            [BlockType.TEXT], [text], page=20) is PageArchetype.LIST

    def test_no_implicit_list_on_short_enumeration(self):
        text = "Chapter 6 covers the following. 2. See the appendix."
        assert archetype_from_blocks(
            [BlockType.TEXT], [text], page=20) is PageArchetype.NARRATIVE

    def test_chrome_never_needed(self):
        # a page whose only content types are non-listed -> default
        assert archetype_from_blocks(
            [BlockType.FORM, BlockType.OTHER], page=40) is PageArchetype.DEFAULT


class TestClassifyBook:
    def test_full_book(self, mixed_book):
        got = classify_book(mixed_book)
        expected = {
            1: PageArchetype.FRONT_MATTER,
            2: PageArchetype.TOC,
            3: PageArchetype.NARRATIVE,
            4: PageArchetype.LIST,
            5: PageArchetype.FOOTNOTE,
            6: PageArchetype.EQUATION,
            7: PageArchetype.CODE,
            8: PageArchetype.TABLE,
            9: PageArchetype.FIGURE,
            10: PageArchetype.NARRATIVE,
            11: PageArchetype.INDEX,
        }
        assert got == expected

    def test_empty_page_is_default(self, mixed_book):
        empty = Book(blocks=[], pages={5: PageInfo(
            index=5, width=1, height=1, bbox=(0, 0, 1, 1), polygon=[])},
            toc=[], page_stats=[])
        assert classify_book(empty) == {}


# ---------------------------------------------------------------------------
# Markdown-heuristic classification (fallback)
# ---------------------------------------------------------------------------

class TestArchetypeFromMarkdown:
    def test_front_matter(self):
        md = "Flight Simulation Software\nDavid Allerton"
        assert archetype_from_markdown(
            md, page=2, num_pages=100) is PageArchetype.FRONT_MATTER

    def test_toc(self):
        md = textwrap.dedent("""\
            Design …… 5
            Two-dimensional theory …… 41
            Dynamics of an aircraft …… 83
            Simulation software design …… 146
            Motion platform …… 212
            Validation data …… 267
        """)
        assert archetype_from_markdown(
            md, page=4, num_pages=100) is PageArchetype.TOC

    def test_index(self):
        md = "altitude, 12\nairspeed, 17\nautopilot, 3\ncorrection, 8\ndrag, 21\nelevator, 30"
        assert archetype_from_markdown(
            md, page=160, num_pages=165) is PageArchetype.INDEX

    def test_lost_list(self):
        md = "- Set the altitude\n- Trim the controls\n- Run the test case"
        assert archetype_from_markdown(
            md, page=30, num_pages=165) is PageArchetype.LIST

    def test_footnote(self):
        md = "[^1]: Analogue computers were widely used in the 1970s.\n[^2]: See chapter 4."
        assert archetype_from_markdown(
            "Body text.\n" + md, page=40, num_pages=165) is PageArchetype.FOOTNOTE

    def test_equation(self):
        assert archetype_from_markdown(
            "Thus we obtain\n\n$$ x = a y + b $$\n", page=41) is PageArchetype.EQUATION
        assert archetype_from_markdown(
            "Substituting §EQ0§ gives the result.", page=41) is PageArchetype.EQUATION

    def test_code(self):
        md = "```python\ndef f():\n    return 1\n```\n"
        assert archetype_from_markdown(
            md, page=42, num_pages=165) is PageArchetype.CODE

    def test_table(self):
        md = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |"
        assert archetype_from_markdown(
            md, page=43, num_pages=165) is PageArchetype.TABLE

    def test_figure(self):
        md = "![image](images/p001_x.jpg)"
        assert archetype_from_markdown(
            md, page=44, num_pages=165) is PageArchetype.FIGURE

    def test_narrative_and_default(self):
        assert archetype_from_markdown(
            "A body paragraph.", page=45, num_pages=165) is PageArchetype.NARRATIVE
        assert archetype_from_markdown(
            "<!-- remark only -->", page=46, num_pages=165) is PageArchetype.DEFAULT


class TestClassifyMarkdown:
    def test_pages_map_to_request(self):
        book_md = textwrap.dedent("""\
            <!-- ===== page 1 ===== -->

            Flight Simulation Software

            <!-- ===== page 3 ===== -->

            An introduction to simulation techniques and trade-offs.
            """)
        got = classify_markdown(book_md, num_pages=9)
        assert got == {
            1: PageArchetype.FRONT_MATTER,
            3: PageArchetype.NARRATIVE,
        }


def test_prompt_cards_cover_archetypes():
    from book_harness.config import (
        FORMAT_PROMPT_VERSION,
        PAGE_FORMAT_ARCHETYPES,
        PAGE_FORMAT_SYSTEM,
    )
    assert set(PAGE_FORMAT_ARCHETYPES) == {a.value for a in PageArchetype}
    assert FORMAT_PROMPT_VERSION == "fmt-v2"
    assert "whitespace-insensitive" in PAGE_FORMAT_SYSTEM


def test_format_role_never_inherits_mt_models():
    """M5 formatting must resolve general instruct models, not the MT list."""
    from book_harness.config import FORMAT_MODELS, _FORMAT_MODEL_DEFAULTS
    assert not any("hy-mt2" in m for m in _FORMAT_MODEL_DEFAULTS)
    assert all(m for m in FORMAT_MODELS)   # env override stays well-formed


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------
CANONICAL = (
    "<!-- ===== page 3 ===== -->\n\n"
    "First page content.\n\n"
    "<!-- ===== page 4 ===== -->\n\n"
    "Second page content.\n"
)


class TestSliceRoundTrip:
    def test_split_extracts_numbers_and_text(self):
        pages, head = split_pages(CANONICAL)
        assert head == ""
        assert pages == {3: "First page content.", 4: "Second page content."}

    def test_join_is_inverse_of_split(self):
        pages, head = split_pages(CANONICAL)
        assert join_pages(pages, head) == CANONICAL

    def test_join_inserts_head_before_first_comment(self):
        out = join_pages({7: "text"}, head="Prelude")
        assert out == ("Prelude\n\n<!-- ===== page 7 ===== -->\n\n"
                       "text\n")

    def test_split_handles_missing_comments(self):
        text = "No page markers here."
        assert split_pages(text) == ({}, text)

    def test_split_tolerates_varied_equals_runs(self):
        md = "<!-- = page 12 = -->\n\ncontent"
        pages, _ = split_pages(md)
        assert pages == {12: "content"}

    def test_normalizes_noncanonical_separator_whitespace(self):
        md = "text\n\n\n\n<!-- ===== page 9 ===== -->\n\n\n\ntail text\n"
        out = join_pages(*split_pages(md))
        assert out == "text\n\n<!-- ===== page 9 ===== -->\n\ntail text\n"

    def test_join_is_idempotent(self):
        once = join_pages(*split_pages(CANONICAL))
        twice = join_pages(*split_pages(once))
        assert twice == once
