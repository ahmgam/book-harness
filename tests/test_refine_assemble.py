"""Integration: refinement -> assembly overlay, and compliance re-queue keys."""

import json
from pathlib import Path

from book_harness.assemble import DraftBook, render_book
from book_harness.book import Block, BlockType, Book, PageInfo
from book_harness.chunker import Chunk
from book_harness.translate import OllamaClient, SummaryTracker, translate_chunk


class TransStub(OllamaClient):
    model = "stub-mt"

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt, *, system=None, options=None):
        self.calls.append(prompt)
        return "الرفع يزداد مع الزاوية."


class Assistant(OllamaClient):
    """Estimates one span; verify always prefers the patch (no MONTHLY)."""

    model = "stub-assist"

    def __init__(self, spans):
        self.spans = json.dumps(spans)

    def generate(self, prompt, *, system=None, options=None):
        if system and "MQM" in system:
            return self.spans
        if "Which translation is better" in prompt:
            bodies = prompt.split("A:\n", 1)[1]
            a, b = bodies.split("\nB:\n", 1)
            return "B" if "MONTHLY" in a else "A"
        return ""


GLOSSARY = [{"term_en": "lift", "term_ar": "الرفع", "status": "certified"}]


def _chunk_dict(**over):
    chunk = {
        "index": 1,
        "chunk_id": "chunk-0001",
        "section_id": "/page/1/SectionHeader/0",
        "section_path": [{"number": "1.1", "title": "Design"}],
        "section_title": "1.1 Design",
        "pages": [1],
        "source_hash": "abc",
        "blocks": [{"block_id": "/page/1/Text/0", "page": 1,
                    "type": "Text", "units": ["u0"]}],
        "units": [
            {"uid": "u0", "block_id": "/page/1/Text/0", "kind": "text",
             "text": "The lift increases with angle.", "placeholders": {}},
        ],
    }
    chunk.update(over)
    return chunk


def _draft(unit_text: str = "الرفع يزداد مع الزاوية.MONTHLY.") -> dict:
    return {
        "chunk_id": "chunk-0001",
        "cache_key": "draft-key-1",
        "units": [{"uid": "u0", "kind": "text", "text": unit_text}],
    }


def _book() -> Book:
    block = Block(
        id="/page/1/Text/0", type=BlockType.TEXT,
        html="<p>The lift increases with angle.</p>", page=1,
        bbox=(0, 0, 0, 0), polygon=[], section_hierarchy={}, images={},
    )
    page = PageInfo(index=1, width=600, height=800, bbox=(0, 0, 600, 800), polygon=[])
    return Book(blocks=[block], pages={1: page}, toc=[], page_stats=[])


def _fake_drafts(draft: dict) -> DraftBook:
    drafts = DraftBook.__new__(DraftBook)
    drafts.meta_by_chunk = {draft["chunk_id"]: draft}
    drafts.unit_texts = {
        f"{draft['chunk_id']}/{u['uid']}": dict(u) for u in draft["units"]
    }
    return drafts


def _refine(tmp_path: Path, draft: dict) -> dict:
    from book_harness.refine import refine_chunk

    spans = [{"original": "الزاوية.MONTHLY.", "suggestion": "الزاوية."}]
    refine_dir = tmp_path / "05_refine"
    return refine_chunk(
        Chunk.from_dict(_chunk_dict()), draft, Assistant(spans), refine_dir,
        draft_cache_key=draft["cache_key"], prompt_version="r1",
    )


class TestAssemblyOverlay:
    def test_refined_text_wins_in_book_md(self, tmp_path):
        draft = _draft()
        rec = _refine(tmp_path, draft)
        assert rec["units"][0]["text"] == "الرفع يزداد مع الزاوية."

        drafts = _fake_drafts(draft)
        render_book(
            _book(), [Chunk.from_dict(_chunk_dict())], drafts,
            tmp_path / "07_output", refine_dir=tmp_path / "05_refine",
        )
        md = (tmp_path / "07_output" / "book.md").read_text()
        assert "الرفع يزداد مع الزاوية." in md
        assert "MONTHLY" not in md

    def test_stale_refinement_ignored(self, tmp_path):
        _refine(tmp_path, _draft())
        drafts = _fake_drafts(_draft())
        drafts.meta_by_chunk["chunk-0001"]["cache_key"] = "re-translated-key"
        render_book(
            _book(), [Chunk.from_dict(_chunk_dict())], drafts,
            tmp_path / "07_output", refine_dir=tmp_path / "05_refine",
        )
        md = (tmp_path / "07_output" / "book.md").read_text()
        assert "MONTHLY" in md

    def test_load_refinements_drop_empty(self, tmp_path: Path):
        from book_harness.refine import load_refinements

        _refine(tmp_path, _draft())
        overrides = load_refinements(
            tmp_path / "05_refine", {"chunk-0001": {"cache_key": "draft-key-1"}}
        )
        assert overrides == {"chunk-0001": {"u0": "الرفع يزداد مع الزاوية."}}


class TestEmphasisRequeue:
    def test_emphasis_changes_cache_key_and_prompt(self, tmp_path):
        client = TransStub()
        summaries = SummaryTracker(client, tmp_path / "sum.json", period=100)
        translate_chunk(
            _chunk_dict(), client, GLOSSARY, tmp_path,
            prompt_version="m1-v1", glossary_version="gv1", summaries=summaries,
        )
        calls_after_draft = len(client.calls)
        emphasis = [("lift", "قوة الرفع الإجبارية")]
        translate_chunk(
            _chunk_dict(), client, GLOSSARY, tmp_path,
            prompt_version="m1-v1", glossary_version="gv1", summaries=summaries,
            emphasis=emphasis,
        )
        # the emphasis run invalidates the fresh draft and re-translates
        assert len(client.calls) > calls_after_draft
        assert any("MANDATORY" in c and "قوة الرفع الإجبارية" in c for c in client.calls)
        # and calling again with the same emphasis is a fresh-cache no-op
        calls_now = len(client.calls)
        translate_chunk(
            _chunk_dict(), client, GLOSSARY, tmp_path,
            prompt_version="m1-v1", glossary_version="gv1", summaries=summaries,
            emphasis=emphasis,
        )
        assert len(client.calls) == calls_now
