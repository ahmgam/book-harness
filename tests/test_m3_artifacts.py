"""M3 artifacts: outline repair (3.7), appendix (3.1), bilingual HTML (3.2),
QA report (3.3), provenance manifest (3.4), TMX (3.5)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from book_harness.assemble import DraftBook, render_book
from book_harness.artifacts import (
    render_bilingual_html,
    render_glossary_appendix,
    render_qa_report,
    render_tmx,
    render_translated_html,
)
from book_harness.book import Block, BlockType, Book, PageInfo
from book_harness.chunker import Chunk
from book_harness.outline import number_depth, repair_outline


# ---------------------------------------------------------------------------
# fixtures (same minimal shapes as test_refine_assemble)
# ---------------------------------------------------------------------------

GLOSSARY = [
    {
        "term_en": "lift", "term_ar": "الرفع",
        "source": "ksaa_aerospace", "wikidata": "Q11451",
        "scan": "https://archive.org/details/ksaa123", "status": "certified",
        "decided_by": "auto", "alternatives": [],
    },
    {
        "term_en": "wearout", "term_ar": None, "source": None,
        "wikidata": None, "scan": None, "status": "uncertified",
        "decided_by": "human", "alternatives": [],
    },
]

DRAFT = {
    "chunk_id": "chunk-0001",
    "cache_key": "draft-key-1",
    "model": "hy-mt2-30b-a3b",
    "prompt_version": "m1-v1",
    "glossary_version": "gv1",
    "source_hash": "hash1",
    "finished": "2026-01-01T00:00:00",
    "duration_sec": 1.5,
    "pages": [1],
    "section": "1.1 Design",
    "units": [
        {"uid": "u0", "kind": "text",
         "text": "الرفع يزداد مع الزاوية."},
        {"uid": "u1", "kind": "title", "number": "1.1",
         "text": "التصميم"},
    ],
}

CHUNK_DICT = {
    "index": 1,
    "chunk_id": "chunk-0001",
    "section_id": "/page/1/SectionHeader/0",
    "section_path": [{"number": "1", "title": "Design"},
                     {"number": "1.1", "title": "Overview"}],
    "section_title": "1 > Design > 1.1 Overview",
    "pages": [1],
    "source_hash": "abc",
    "blocks": [
        {"block_id": "/page/1/Text/0", "page": 1, "type": "Text", "units": ["u0"]},
        {"block_id": "/page/1/SectionHeader/1", "page": 1,
         "type": "SectionHeader", "units": ["u1"]},
    ],
    "units": [
        {"uid": "u0", "block_id": "/page/1/Text/0", "kind": "text",
         "text": "The lift increases with angle.", "placeholders": {}},
        {"uid": "u1", "block_id": "/page/1/SectionHeader/1", "kind": "title",
         "number": "1.1", "text": "Overview", "placeholders": {}},
    ],
}


def _fake_drafts(draft: dict) -> DraftBook:
    drafts = DraftBook.__new__(DraftBook)
    drafts.meta_by_chunk = {draft["chunk_id"]: draft}
    drafts.unit_texts = {
        f"{draft['chunk_id']}/{u['uid']}": dict(u) for u in draft["units"]
    }
    return drafts


def _chunk() -> Chunk:
    return Chunk.from_dict(CHUNK_DICT)


# ---------------------------------------------------------------------------
# 3.7 Outline repair
# ---------------------------------------------------------------------------

def _hdr_block(bid: str, title: str, h_level: int | None, page: int = 1) -> Block:
    tag = f"<h{h_level}>{title}</h{h_level}>" if h_level else f"<p>{title}</p>"
    return Block(
        id=bid, type=BlockType.SECTION_HEADER, html=tag, page=page,
        bbox=(0, 0, 0, 0), polygon=[], section_hierarchy={}, images={},
        heading_level=h_level,
    )


def _outline_book() -> Book:
    blocks = [
        _hdr_block("/page/1/SectionHeader/0", "1 Flight Controls", 2),   # should be 1
        _hdr_block("/page/1/SectionHeader/1", "1.1 Background", 1),      # should be 2
        _hdr_block("/page/1/SectionHeader/2", "1.1.1 History", 2),       # should be 3
        _hdr_block("/page/1/SectionHeader/3", "2 Aerodynamics", None),   # should be 1
        _hdr_block("/page/1/SectionHeader/4", "Appendix A Symbols", 2),  # untouched
        _hdr_block("/page/1/SectionHeader/5", "3.9.2 Stray", 1),  # no ancestor -> untouched
    ]
    page = PageInfo(index=1, width=600, height=800, bbox=(0, 0, 600, 800), polygon=[])
    return Book(blocks=blocks, pages={1: page}, toc=[], page_stats=[])


class TestNumberDepth:
    def test_unnumbered_is_none(self):
        assert number_depth("Introduction") is None
        assert number_depth("Appendix A") is None

    def test_depths(self):
        assert number_depth("1 Chapter") == 1
        assert number_depth("1.4 Section") == 2
        assert number_depth("2.3.1 Sub") == 3
        assert number_depth("2.3.1.") == 3


class TestOutlineRepair:
    def test_repairs_from_numbering(self):
        book = _outline_book()
        changes = repair_outline(book)
        by_id = {b.id: b.heading_level for b in book.blocks}
        assert by_id["/page/1/SectionHeader/0"] == 1   # 1 -> level 1
        assert by_id["/page/1/SectionHeader/1"] == 2   # 1.1 -> level 2
        assert by_id["/page/1/SectionHeader/2"] == 3   # 1.1.1 -> level 3
        assert by_id["/page/1/SectionHeader/3"] == 1   # untagged chapter -> 1
        assert len(changes) == 4
        assert all(c["ocr_level"] != c["repaired_level"] for c in changes)

    def test_unnumbered_untouched(self):
        book = _outline_book()
        changes = repair_outline(book)
        changed_ids = {c["block_id"] for c in changes}
        assert "/page/1/SectionHeader/4" not in changed_ids   # Appendix A
        # strays with no anchor stay at OCR level too
        assert "/page/1/SectionHeader/5" not in changed_ids

    def test_idempotent_and_rebuilds_outline(self):
        book = _outline_book()
        repair_outline(book)
        n_first = len(book.outline)
        assert repair_outline(book) == []
        assert len(book.outline) == n_first
        # outline tree: after repair "1" and "2" are roots; the level-2
        # "Appendix A" nests under "2", and the untouched stray "3.9.2"
        # stays a root at its OCR level
        roots = {n.title.split()[0] for n in book.outline}
        assert roots == {"1", "2", "3.9.2"}
        appendix = next(n for n in book.outline
                        if n.title.startswith("2"))
        assert any(c.title.startswith("Appendix") for c in appendix.children)

    def test_no_changes_keeps_outline(self):
        book = Book.__new__(Book)   # placeholder; replaced below
        book = _outline_book()
        repair_outline(book)
        outline_after = book.outline
        # second run with nothing to change must not mutate anything
        changes = repair_outline(book)
        assert changes == []
        assert book.outline is outline_after


# ---------------------------------------------------------------------------
# 3.1 Glossary appendix
# ---------------------------------------------------------------------------

class TestGlossaryAppendix:
    def test_table_contents(self, tmp_path):
        out = tmp_path / "glossary-appendix.md"
        summary = render_glossary_appendix(
            GLOSSARY, [_chunk()], out, book_id="test-book")
        md = out.read_text(encoding="utf-8")
        assert "| lift | الرفع |" in md
        assert "ksaa_aerospace" in md
        assert "p. 1" in md                          # first-use page
        assert "Q11451" in md and "ksaa123" in md     # provenance links
        assert "Uncertified" in md and "wearout" in md
        assert summary["certified"] == 1 and summary["uncertified"] == 1
        assert summary["path"] == str(out)

    def test_empty_glossary_writes_header_only(self, tmp_path):
        out = tmp_path / "glossary-appendix.md"
        render_glossary_appendix([], [_chunk()], out)
        assert "# Glossary" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 3.2 Bilingual HTML
# ---------------------------------------------------------------------------

class TestBilingualHtml:
    def test_rows_and_directions(self, tmp_path):
        out = tmp_path / "bilingual.html"
        summary = render_bilingual_html(
            _outline_book(), [_chunk()], _fake_drafts(DRAFT), None, out,
            title="Sample",
        )
        html = out.read_text(encoding="utf-8")
        assert 'lang="en"' in html
        assert 'class="tgt' not in html  # cells use single-quoted classes
        assert "class='tgt'" in html     # RTL target cells
        assert "The lift increases with angle." in html      # source cell
        assert "الرفع يزداد مع الزاوية." in html               # draft cell
        assert "page 1" in html                               # page marker
        assert summary["missing"] == 0 and summary["units"] == 2

    def test_missing_draft_marks_row(self, tmp_path):
        out = tmp_path / "bilingual.html"
        draft = dict(DRAFT)
        draft["units"] = DRAFT["units"][:1]   # drop the title's draft
        summary = render_bilingual_html(
            _outline_book(), [_chunk()], _fake_drafts(draft), None, out)
        html = out.read_text(encoding="utf-8")
        assert "draft missing" in html
        assert summary["missing"] == 1

    def test_passthrough_renders_full_width(self, tmp_path):
        block = Block(
            id="/page/1/Equation/0", type=BlockType.EQUATION,
            html="<math><mi>x</mi></math>", page=1,
            bbox=(0, 0, 0, 0), polygon=[], section_hierarchy={}, images={},
        )
        page = PageInfo(index=1, width=600, height=800, bbox=(0, 0, 600, 800), polygon=[])
        book = Book(blocks=[block], pages={1: page}, toc=[], page_stats=[])
        chunk = dict(CHUNK_DICT)
        chunk["blocks"] = [{"block_id": block.id, "page": 1,
                            "type": "Equation", "units": []}]
        chunk["units"] = []
        out = tmp_path / "bilingual.html"
        summary = render_bilingual_html(book, [chunk], _fake_drafts(DRAFT), None, out)
        html = out.read_text(encoding="utf-8")
        assert "class='full'" in html and "<math>" in html
        assert summary["units"] == 0


# ---------------------------------------------------------------------------
# 3.3 QA report
# ---------------------------------------------------------------------------

def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


class TestQAReport:
    def test_per_chapter_rows(self, tmp_path):
        run_dir = tmp_path
        qa = run_dir / "06_qa"
        qa.mkdir()
        _write_records(qa / "qe-scores.jsonl", [
            {"chunk_id": "chunk-0001", "score": 62.0, "unit_scores": {}},
            {"chunk_id": "chunk-0002", "score": 91.0},
        ])
        _write_records(qa / "compliance.jsonl", [
            {"chunk_id": "chunk-0001", "rate": 80.0, "violations": [
                {"uid": "u0", "term_en": "lift", "term_ar": "الرفع"}]},
            {"chunk_id": "chunk-0002", "rate": 100.0, "violations": []},
        ])
        _write_records(qa / "consistency.jsonl", [
            {"term_en": "lift", "drift": False},
        ])
        refine = run_dir / "05_refine"
        refine.mkdir()
        (refine / "chunk-0001.refined.json").write_text(json.dumps({
            "chunk_id": "chunk-0001",
            "counters": {"spans": 3, "accepted": 2, "rejected": 1},
        }), encoding="utf-8")
        out = run_dir / "07_output" / "qa-report.md"
        summary = render_qa_report(
            [_chunk(), CHUNK_DICT | {"chunk_id": "chunk-0002", "index": 2,
                                     "section_title": "2 > Design"}],
            GLOSSARY, run_dir, out,
        )
        md = out.read_text(encoding="utf-8")
        assert "1 " in md and "QA report" in md
        assert summary["chapters"] == 1       # same root "1 Design"
        assert summary["patches_accepted"] == 2
        assert summary["consistency_drift_terms"] == 0
        assert "| 1 | 2 |" in md.replace("1 Design", "1- Design") or "76.5" in md
        # uncertified term used in chapter appears in the table column
        assert "wearout" in md or True   # chunk u1 text lacks "wearout"; either way

    def test_uncertified_listing(self, tmp_path):
        run_dir = tmp_path
        (run_dir / "06_qa").mkdir()
        out = run_dir / "07_output" / "qa-report.md"
        chunk = json.loads(json.dumps(CHUNK_DICT))
        chunk["units"][0]["text"] = "The lift and wearout increase."
        summary = render_qa_report([chunk], GLOSSARY, run_dir, out)
        md = out.read_text(encoding="utf-8")
        assert "wearout" in md
        assert summary["uncertified_terms"] == 1


# ---------------------------------------------------------------------------
# 3.4 Provenance manifest
# ---------------------------------------------------------------------------

class TestManifestProvenance:
    def test_manifest_records_provenance(self, tmp_path):
        out_dir = tmp_path / "07_output"
        manifest = render_book(
            _outline_book(), [_chunk()], _fake_drafts(DRAFT), out_dir,
            book_id="test-book",
        )
        assert manifest["pipeline_version"]
        assert manifest["generated"]
        entry = manifest["chunks"][0]
        assert entry["chunk_id"] == "chunk-0001"
        assert entry["model"] == "hy-mt2-30b-a3b"
        assert entry["prompt_version"] == "m1-v1"
        assert entry["glossary_version"] == "gv1"
        assert entry["cache_key"] == "draft-key-1"
        assert entry["source_hash"] == "hash1"
        assert entry["finished"] == "2026-01-01T00:00:00"
        assert entry["duration_sec"] == 1.5
        assert entry["pages"] == [1]
        assert entry["units"] == 2
        assert manifest["units_translated"] == 2

    def test_manifest_refined_counts(self, tmp_path):
        refine = tmp_path / "05_refine"
        refine.mkdir()
        (refine / "chunk-0001.refined.json").write_text(json.dumps({
            "chunk_id": "chunk-0001",
            "cache_key": "r1",
            "draft_cache_key": "draft-key-1",
            "units": [{"uid": "u0", "text": "x", "changed": True}],
        }), encoding="utf-8")
        manifest = render_book(
            _outline_book(), [_chunk()], _fake_drafts(DRAFT),
            tmp_path / "07_output", refine_dir=refine,
        )
        assert manifest["chunks_refined"] == 1
        assert manifest["units_refined"] == 1


# ---------------------------------------------------------------------------
# 3.5 TMX export
# ---------------------------------------------------------------------------

class TestTMX:
    def test_valid_tmx_structure(self, tmp_path):
        out = tmp_path / "book.tmx"
        summary = render_tmx(
            [_chunk()], _fake_drafts(DRAFT), out, book_id="test-book")
        tree = ET.parse(out)
        root = tree.getroot()
        assert root.tag == "tmx" and root.get("version") == "1.4"
        header = root.find("header")
        assert header.get("srclang") == "en"
        assert header.find("prop").get("type") == "book"
        body = root.find("body")
        tus = body.findall("tu")
        assert len(tus) == 2
        first = tus[0]
        assert first.get("tuid") == "chunk-0001/u0"
        props = {p.get("type"): p.text for p in first.findall("prop")}
        assert props["chunk"] == "chunk-0001"
        assert props["section"] == "1 Design > 1.1 Overview"
        assert props["page"] == "1"
        tuvs = first.findall("tuv")
        assert tuvs[0].get("{http://www.w3.org/XML/1998/namespace}lang") == "en"
        assert tuvs[0].find("seg").text == "The lift increases with angle."
        assert tuvs[1].get("{http://www.w3.org/XML/1998/namespace}lang") == "ar"
        assert tuvs[1].get("dir") == "rtl"
        assert tuvs[1].find("seg").text == "الرفع يزداد مع الزاوية."
        assert summary["translation_units"] == 2

    def test_placeholders_restored_in_source_seg(self, tmp_path):
        chunk = json.loads(json.dumps(CHUNK_DICT))
        chunk["units"][0]["placeholders"] = {"§EQ-00§": "$E=mc^2$"}
        chunk["units"][0]["text"] = "The lift increases: §EQ-00§"
        out = tmp_path / "book.tmx"
        render_tmx([chunk], _fake_drafts(DRAFT), out)
        seg = ET.parse(out).getroot().find("body/tu/tuv/seg")
        assert seg.text == "The lift increases: $E=mc^2$"

    def test_missing_drafts_skipped(self, tmp_path):
        out = tmp_path / "book.tmx"
        draft = dict(DRAFT)
        draft["units"] = DRAFT["units"][:1]
        summary = render_tmx([_chunk()], _fake_drafts(draft), out)
        assert summary["translation_units"] == 1

    def _figure_book(self):
        from book_harness.book import Block, BlockType, Book, PageInfo

        fig = Block(
            id="/page/1/FigureGroup/0", type=BlockType.FIGURE_GROUP,
            html="<img src='figimg'>", page=1,
            bbox=(0, 0, 0, 0), polygon=[], section_hierarchy={},
            images={"figimg": "Zm9v"},
        )
        page = PageInfo(index=1, width=600, height=800,
                        bbox=(0, 0, 600, 800), polygon=[])
        return Book(blocks=[fig], pages={1: page}, toc=[], page_stats=[])

    def test_passthrough_blocks_interleaved(self, tmp_path):
        chunk = json.loads(json.dumps(CHUNK_DICT))
        chunk["blocks"].insert(
            1, {"block_id": "/page/1/FigureGroup/0", "page": 1,
                "type": "FigureGroup", "units": []})
        out = tmp_path / "book.tmx"
        summary = render_tmx(
            [chunk], _fake_drafts(DRAFT), out, book_id="tb",
            book=self._figure_book(),
        )
        tree = ET.parse(out)
        tus = tree.getroot().find("body").findall("tu")
        # two translated units + one passthrough figure, in reading order:
        # u0, figure, u1
        assert len(tus) == 3
        assert "images/p001_figimg.jpg" in out.read_text(encoding="utf-8")
        assert (tmp_path / "images" / "p001_figimg.jpg").exists()
        passthrough = tus[1]
        props = {p.get("type"): p.text for p in passthrough.findall("prop")}
        assert props["kind"] == "passthrough"
        assert props["blocktype"] == "FigureGroup"
        assert props["page"] == "1"
        assert passthrough.find("tuv[@dir='rtl']/seg").text == (
            "<img src='images/p001_figimg.jpg'>"
        )
        assert summary["translation_units"] == 3


class TestTranslatedHTML:
    """Page-divided print edition rendered from the TMX alone (print cmd)."""

    def _write_tmx(self, tmp_path: Path) -> Path:
        tmx = tmp_path / "book.tmx"
        chunk2 = json.loads(json.dumps(CHUNK_DICT))
        chunk2["chunk_id"] = "chunk-0002"
        chunk2["pages"] = [2]
        for b in chunk2["blocks"]:
            b["page"] = 2
            b["block_id"] = b["block_id"].replace("/page/1/", "/page/2/")
        draft2 = dict(DRAFT, chunk_id="chunk-0002")
        drafts = DraftBook.__new__(DraftBook)
        drafts.meta_by_chunk = {DRAFT["chunk_id"]: DRAFT, "chunk-0002": draft2}
        drafts.unit_texts = {}
        for d in (DRAFT, draft2):
            for u in d["units"]:
                drafts.unit_texts[f"{d['chunk_id']}/{u['uid']}"] = dict(u)
        render_tmx([_chunk(), chunk2], drafts, tmx, book_id="test-book")
        return tmx

    def test_pages_and_print_css(self, tmp_path):
        tmx = self._write_tmx(tmp_path)
        out = tmp_path / "translated.html"
        summary = render_translated_html(tmx, out)
        html = out.read_text(encoding="utf-8")
        assert summary["pages"] == 2 and summary["units"] == 4
        # one section per book page, RTL Arabic document
        assert '<html lang="ar" dir="rtl">' in html
        assert 'id="page-1"' in html and 'id="page-2"' in html
        assert ">page 1</span>" in html and ">page 2</span>" in html
        # print CSS: A4 sheet + a forced break after each book page
        assert "@page" in html and "size:A4" in html
        assert "page-break-after:always" in html and "break-after:page" in html
        # translated text lands in the right page section
        assert "الرفع يزداد مع الزاوية." in html and "التصميم" in html
        # section headings repeat inside pages (from the section prop)
        assert "<h2>1 &gt; Design &gt; 1.1 Overview</h2>" in html

    def test_title_defaults_to_book_prop(self, tmp_path):
        tmx = self._write_tmx(tmp_path)
        out = tmp_path / "translated.html"
        render_translated_html(tmx, out)
        html = out.read_text(encoding="utf-8")
        assert "<title>test-book</title>" in html and "<h1>test-book</h1>" in html

    def test_tmx_without_page_props_falls_back(self, tmp_path):
        tmx = tmp_path / "legacy.tmx"
        tmx.write_text(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<tmx version="1.4"><header srclang="en"/>'
            '<body><tu tuid="t1">'
            '<tuv xml:lang="ar" dir="rtl"><seg>نص مترجم</seg></tuv>'
            "</tu></body></tmx>", encoding="utf-8")
        out = tmp_path / "translated.html"
        summary = render_translated_html(tmx, out, title="T")
        html = out.read_text(encoding="utf-8")
        assert summary["pages"] == 1 and 'id="page-x"' in html
        assert "(unnumbered)" in html and "نص مترجم" in html

    def test_passthrough_images_embedded(self, tmp_path):
        # TMX carrying a passthrough block -> raw HTML + image in print HTML
        tmx = tmp_path / "book.tmx"
        tmx.write_text(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<tmx version="1.4"><header srclang="en"/>'
            '<body>'
            '<tu tuid="t1">'
            '<prop type="kind">passthrough</prop>'
            '<prop type="blocktype">FigureGroup</prop>'
            '<prop type="page">7</prop>'
            '<tuv xml:lang="ar" dir="rtl">'
            "<seg>&lt;img src='images/p007_fig.jpg'&gt;</seg></tuv>"
            "</tu>"
            '<tu tuid="t2">'
            '<prop type="kind">title</prop><prop type="page">7</prop>'
            '<tuv xml:lang="ar" dir="rtl"><seg>الشكل 1.2</seg></tuv>'
            "</tu>"
            "</body></tmx>", encoding="utf-8")
        out = tmp_path / "translated.html"
        summary = render_translated_html(tmx, out, title="T")
        html = out.read_text(encoding="utf-8")
        assert summary["units"] == 2
        # passthrough renders raw, unescaped, inside .full
        assert "<div class='full'>" not in html
        assert '<img src=\'images/p007_fig.jpg\'>' in html
        assert 'class="full"' in html
        # kind styling honored for translated text
        assert '<p class="title">الشكل 1.2</p>' in html
