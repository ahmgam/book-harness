"""Book-glossary harvesting tests (task 6.1–6.3)."""

import json
from pathlib import Path

import pytest

from book_harness import cli
from book_harness.book import Block, BlockType, PageInfo
from book_harness.book import Book
from book_harness.config import Settings
from book_harness.glossary_pages import (
    build_candidates,
    extract_page_terms,
    merge_candidates,
    parse_term_lines,
    write_page_terms,
)
from book_harness.pages import find_glossary_pages


# ---------------------------------------------------------------------------
# Fixture: a Book assembled from Block dataclass instances directly
# ---------------------------------------------------------------------------

def _block(bid, btype, text, page, level=None):
    html = f"<p>{text}</p>"
    if level:
        html = f"<h{level}>{text}</h{level}>"
    return Block(
        id=bid, type=btype, html=html, page=page,
        bbox=(0, 0, 10, 10), polygon=[], section_hierarchy={}, images={},
        heading_level=level,
    )


def _book(blocks, num_pages):
    pages = {i: PageInfo(index=i, width=100, height=100, bbox=(0, 0, 100, 100),
                         polygon=[])
             for i in range(1, num_pages + 1)}
    return Book(blocks=blocks, pages=pages, toc=[], page_stats=[])


@pytest.fixture
def glossary_book() -> Book:
    """A book whose glossary section runs pages 6–7, appendices after."""
    blocks = [
        _block("/page/1/SectionHeader/0", BlockType.SECTION_HEADER,
               "1 Design", 1, level=1),
        _block("/page/1/Text/0", BlockType.TEXT,
               "The flight simulator integrates aircraft motion models.", 1),
        # glossary header on page 6, letter-group subheaders on both pages
        _block("/page/6/SectionHeader/0", BlockType.SECTION_HEADER,
               "Glossary", 6, level=1),
        _block("/page/6/SectionHeader/1", BlockType.SECTION_HEADER,
               "A", 6, level=2),
        _block("/page/6/Text/0", BlockType.TEXT,
               "aerodynamics, 7", 6),
        _block("/page/6/Text/1", BlockType.TEXT,
               "airspeed: indicated airspeed definition placeholder", 6),
        _block("/page/7/SectionHeader/0", BlockType.SECTION_HEADER,
               "B", 7, level=2),
        _block("/page/7/Text/0", BlockType.TEXT,
               "boundary layer — flow region near the surface", 7),
        # appendix terminates the glossary section
        _block("/page/8/SectionHeader/0", BlockType.SECTION_HEADER,
               "Appendix A", 8, level=1),
        _block("/page/8/Text/0", BlockType.TEXT, "Tables.", 8),
    ]
    return _book(blocks, num_pages=9)


# ---------------------------------------------------------------------------
# Detection (OCR metadata only: glossary section headers)
# ---------------------------------------------------------------------------

class TestFindGlossaryPages:
    def test_header_span_ends_at_next_same_level_header(self, glossary_book):
        assert find_glossary_pages(glossary_book) == [range(6, 8)]

    def test_deeper_subheaders_do_not_cut_short(self, glossary_book):
        # "A"/"B" letter groups are level 2, inside the level-1 glossary
        ranges = find_glossary_pages(glossary_book)
        assert all(r.stop == 8 for r in ranges)

    def test_glossary_extends_to_end_without_terminator(self):
        blocks = [
            _block("/page/2/SectionHeader/0", BlockType.SECTION_HEADER,
                   "Glossary of Terms and Abbreviations", 2, level=1),
            _block("/page/2/Text/0", BlockType.TEXT, "altitude: height", 2),
            _block("/page/3/Text/0", BlockType.TEXT, "airspeed — speed", 3),
        ]
        assert find_glossary_pages(_book(blocks, num_pages=4)) == [range(2, 5)]

    def test_no_glossary(self, glossary_book):
        stripped = [
            b for b in glossary_book.blocks
            if b.text != "Glossary"
        ]
        assert find_glossary_pages(_book(stripped, num_pages=9)) == []

    def test_whole_word_match_only(self):
        blocks = [
            _block("/page/1/SectionHeader/0", BlockType.SECTION_HEADER,
                   "Glossarial conventions in prose", 1, level=1),
        ]
        assert find_glossary_pages(_book(blocks, num_pages=2)) == []

    def test_running_head_folio_is_stripped_and_ranges_merged(self):
        """OCR sometimes promotes running heads ("xxii Glossary") to
        SectionHeaders; the folio token is ignored and adjacent ranges
        merge into one glossary span."""
        blocks = [
            _block("/page/15/SectionHeader/0", BlockType.SECTION_HEADER,
                   "Aerospace Series Preface", 15, level=2),
            _block("/page/16/SectionHeader/0", BlockType.SECTION_HEADER,
                   "Glossary", 16, level=2),
            _block("/page/18/Text/0", BlockType.TEXT, "ALT altitude", 18),
            _block("/page/19/SectionHeader/0", BlockType.SECTION_HEADER,
                   "xxii Glossary", 19, level=2),
            _block("/page/20/SectionHeader/0", BlockType.SECTION_HEADER,
                   "About the Author", 20, level=2),
        ]
        assert find_glossary_pages(_book(blocks, num_pages=21)) == [range(16, 20)]


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------

class TestParseTermLines:
    def test_bullets_and_numbering_stripped(self):
        reply = "- aerodynamics\n1. airspeed\n• roll rate"
        assert parse_term_lines(reply) == ["aerodynamics", "airspeed", "roll rate"]

    def test_definition_tail_dropped(self):
        assert parse_term_lines("airspeed: measured flow velocity") == ["airspeed"]
        assert parse_term_lines("altitude — height above datum") == ["altitude"]
        assert parse_term_lines("altitude – height above datum") == ["altitude"]

    def test_colon_without_tail_kept(self):
        # "HMS:4" style noise (empty tail) must not be torn apart
        reply = "fuel cell\nrate of climb"
        assert parse_term_lines(reply) == ["fuel cell", "rate of climb"]

    def test_sentences_rejected(self):
        line = "The trailing edge of the aileron is deflected upward " \
            "while the leading edge extends outward beyond thirty chars"
        assert parse_term_lines(line) == []

    def test_dedupe_keeps_first_casing(self):
        assert parse_term_lines("Airspeed\nairspeed\nAIRSPEED") == ["Airspeed"]

    def test_preserves_inner_capitalization_and_hyphens(self):
        assert parse_term_lines("Lift-to-Drag Ratio\nangle-of-attack") == [
            "Lift-to-Drag Ratio", "angle-of-attack",
        ]


# ---------------------------------------------------------------------------
# Extraction + merge
# ---------------------------------------------------------------------------

class FakeAssist:
    """Assist-model double: canned replies keyed by page text substring."""

    def __init__(self, replies: dict[str, str]):
        self.replies = replies
        self.calls: list[str] = []

    def generate(self, prompt, *, system=None, options=None):
        self.calls.append(prompt)
        for needle, reply in self.replies.items():
            if needle in prompt:
                return reply
        raise AssertionError("unexpected prompt")


class TestExtractPageTerms:
    def test_records_schema(self, glossary_book):
        client = FakeAssist({
            "aerodynamics": "aerodynamics\nairspeed",
            "boundary layer": "boundary layer\n- Angle of attack",
        })
        terms = extract_page_terms(glossary_book, [range(6, 8)], client)
        by_term = {t["term"].lower(): t for t in terms}
        assert by_term["aerodynamics"] == {
            "term": "aerodynamics", "freq": 1, "df": 1, "page": 6,
            "score": 10.0, "signal": "book-glossary",
        }
        assert by_term["boundary layer"]["page"] == 7
        # same term harvested twice => freq/df 2, first page kept
        c2 = FakeAssist({
            "aerodynamics": "angle of attack",
            "boundary layer": "Angle Of Attack",
        })
        terms2 = extract_page_terms(glossary_book, [range(6, 8)], c2)
        assert terms2[0] == {
            "term": "angle of attack", "freq": 2, "df": 2, "page": 6,
            "score": 10.0, "signal": "book-glossary",
        }

    def test_one_call_per_page(self, glossary_book):
        client = FakeAssist({
            "aerodynamics": "aerodynamics", "boundary layer": "boundary",
        })
        extract_page_terms(glossary_book, [range(6, 8)], client)
        assert len(client.calls) == 2

    def test_no_ranges_no_model(self, glossary_book):
        assert extract_page_terms(glossary_book, [], None) == []

    def test_missing_client_fails_loudly(self, glossary_book):
        with pytest.raises(RuntimeError, match="assist client"):
            extract_page_terms(glossary_book, [range(6, 8)], None)

    def test_model_failure_fails_loudly(self, glossary_book):
        class Boom:
            def generate(self, *a, **k):
                raise ConnectionError("ollama down")

        with pytest.raises(RuntimeError, match="assist model call failed"):
            extract_page_terms(glossary_book, [range(6, 8)], Boom())

    def test_empty_reply_fails_loudly(self, glossary_book):
        client = FakeAssist({"aerodynamics": "\n\n"})
        with pytest.raises(RuntimeError, match="returned no terms"):
            extract_page_terms(glossary_book, [range(6, 8)], client)


class TestMergeCandidates:
    def test_additive_and_deduped(self):
        base = [
            {"term": "flight dynamics", "score": 2.5, "freq": 4, "df": 3,
             "page": 12, "signal": "ngram"},
        ]
        extra = [
            {"term": "Flight Dynamics", "score": 10.0, "freq": 1, "df": 1,
             "page": 9, "signal": "book-glossary"},   # dup, dropped
            {"term": "weathervaning", "score": 10.0, "freq": 1, "df": 1,
             "page": 9, "signal": "book-glossary"},
        ]
        merged = merge_candidates(base, extra)
        assert [c["term"] for c in merged] == ["flight dynamics", "weathervaning"]
        assert merged[0]["score"] == 2.5       # never re-scored
        assert merged[1]["signal"] == "book-glossary"

    def test_write_page_terms_roundtrip(self, tmp_path: Path):
        recs = [{"term": "Boundary Layer", "freq": 1, "df": 1, "page": 6,
                 "score": 10.0, "signal": "book-glossary"}]
        path = tmp_path / "out" / "glossary-page-terms.jsonl"
        assert write_page_terms(recs, path) == 1
        loaded = [json.loads(ln) for ln in path.read_text().splitlines()]
        assert loaded == recs


# ---------------------------------------------------------------------------
# Stage wiring: stage_terms blends the harvest into candidates.jsonl
# ---------------------------------------------------------------------------

class TestStageTermsWiring:
    def test_merge_into_candidates(self, glossary_book, tmp_path, monkeypatch):
        settings = Settings(ocr_file=Path("dummy.json"))
        settings.run_root = tmp_path
        out_dir = settings.stage_dir(stage="glossary")

        monkeypatch.setattr(
            cli.llm_terms_mod, "extract_candidates",
            lambda book, client, **kw: [{
                "term": "flight dynamics", "score": 2.5, "freq": 4,
                "df": 3, "page": 12, "signal": "llm"}],
        )
        monkeypatch.setattr(cli, "_extract_client", lambda s: object())
        monkeypatch.setattr(cli.pages_mod, "find_glossary_pages",
                            lambda book: [range(6, 8)])
        monkeypatch.setattr(cli, "_assist_client", lambda s: FakeAssist({
            "aerodynamics": "aerodynamics", "boundary layer": "boundary layer",
        }))

        candidates = cli.stage_terms(settings, glossary_book)
        terms = {c["term"].lower() for c in candidates}
        assert "flight dynamics" in terms and "aerodynamics" in terms

        harvest = [json.loads(ln) for ln in (
            out_dir / "glossary-page-terms.jsonl"
        ).read_text().splitlines()]
        assert {r["term"].lower() for r in harvest} == {
            "aerodynamics", "boundary layer",
        }

    def test_no_pages_leaves_pipeline_untouched(self, glossary_book, tmp_path,
                                                monkeypatch):
        settings = Settings(ocr_file=Path("dummy.json"))
        settings.run_root = tmp_path
        out_dir = settings.stage_dir(stage="glossary")
        llm = [{
            "term": "flight dynamics", "score": 2.5, "freq": 4, "df": 3,
            "page": 12, "signal": "llm",
        }]
        monkeypatch.setattr(
            cli.llm_terms_mod, "extract_candidates",
            lambda book, client, **kw: llm,
        )
        monkeypatch.setattr(cli, "_extract_client", lambda s: object())
        monkeypatch.setattr(cli.pages_mod, "find_glossary_pages",
                            lambda book: [])

        assert cli.stage_terms(settings, glossary_book) == llm
        assert not (out_dir / "glossary-page-terms.jsonl").exists()

    def test_dedup_against_mined_list(self, glossary_book, tmp_path,
                                      monkeypatch):
        settings = Settings(ocr_file=Path("dummy.json"))
        settings.run_root = tmp_path
        settings.stage_dir(stage="glossary")
        monkeypatch.setattr(
            cli.llm_terms_mod, "extract_candidates",
            lambda book, client, **kw: ([{
                "term": "Boundary Layer", "score": 3.0, "freq": 9,
                "df": 5, "page": 3, "signal": "llm"}]),
        )
        monkeypatch.setattr(cli, "_extract_client", lambda s: object())
        monkeypatch.setattr(cli.pages_mod, "find_glossary_pages",
                            lambda book: [range(6, 8)])
        monkeypatch.setattr(cli, "_assist_client", lambda s: FakeAssist({
            "aerodynamics": "boundary layer",
            "boundary layer": "boundary layer",
        }))

        candidates = cli.stage_terms(settings, glossary_book)
        assert len(candidates) == 1
        assert candidates[0]["signal"] == "llm"     # LLM record untouched
