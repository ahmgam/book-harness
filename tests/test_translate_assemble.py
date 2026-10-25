"""Tests for translate flow + assembly with a stubbed Ollama client."""

import json
from pathlib import Path

import pytest

from book_harness.assemble import DraftBook, render_book
from book_harness.translate import (
    OllamaClient,
    SummaryTracker,
    match_terms,
    restore_placeholders,
    translate_chunk,
)


class StubClient(OllamaClient):
    """Model that echoes a pseudo-translation; records prompts."""

    def __init__(self):
        self.model = "stub-model"
        self.calls: list[str] = []
        self.summary_calls = 0

    def generate(self, prompt, *, system=None, options=None):
        if "Summarize" in prompt:
            self.summary_calls += 1
            return "stub summary of topics"
        self.calls.append(prompt)
        terms = ""
        for line in prompt.splitlines():
            if "translates to" in line:      # "`X` translates to `Y`"
                terms += line.split("translates to", 1)[1].strip("`* ") + " "
        last_line = prompt.rsplit("\n", 1)[-1]
        return f"[Ar] {terms}{last_line}"


def _chunk(chunk_id="chunk-0001", units=None, **kw):
    base = {
        "chunk_id": chunk_id,
        "section_title": "1 Design > 1.1 The Evolution",
        "pages": [1],
        "source_hash": "abc",
        "units": units or [
            {"uid": "u0", "block_id": "/page/1/Text/0", "kind": "text",
             "text": "The lift increases with angle.",
             "placeholders": {}},
            {"uid": "u1", "block_id": "/page/1/Text/1", "kind": "text",
             "text": "Equation §EQ-00§ describes the drag coefficient. See https://rfc.io",
             "placeholders": {"§EQ-00§": "$C_d = x^2$", }},
        ],
    }
    base.update(kw)
    return base


GLOSSARY = [
    {"term_en": "drag coefficient", "term_ar": "معامل السحب", "status": "certified"},
    {"term_en": "lift", "term_ar": "الرفع", "status": "certified"},
]


class TestMatchTerms:
    def test_case_insensitive_word_boundary(self):
        pairs = match_terms("The Lift coefficient of DRAG COEFFICIENT", GLOSSARY)
        assert ("drag coefficient", "معامل السحب") in pairs
        assert ("lift", "الرفع") in pairs

    def test_respects_word_boundaries(self):
        pairs = match_terms("airlifty slippery", GLOSSARY)
        assert pairs == []


class TestPlaceholders:
    def test_restore_roundtrip(self):
        unit = {"uid": "u0", "placeholders": {"§EQ-00§": "$C_d = x^2$"}}
        out = restore_placeholders(unit, "معامل السحب §EQ-00§ يحدده")
        assert out == "معامل السحب $C_d = x^2$ يحدده"

    def test_unknown_token_kept(self):
        unit = {"uid": "u0", "placeholders": {"§EQ-00§": "$x$"}}
        out = restore_placeholders(unit, "§EQ-01§ lost")
        assert "§EQ-01§" in out


class TestTranslateChunk:
    def test_prompt_contains_glossary_and_context(self, tmp_path):
        client = StubClient()
        summaries = SummaryTracker(client, tmp_path / "sum.json", period=100)
        meta = translate_chunk(
            _chunk(), client, GLOSSARY, tmp_path,
            prompt_version="m1-v1", glossary_version="gv1", summaries=summaries,
        )
        assert meta["chunk_id"] == "chunk-0001"
        texts = [u["text"] for u in meta["units"]]
        assert any("معامل السحب" in t for t in texts), texts
        # placeholder original restored in the draft unit
        assert any("$C_d = x^2$" in t for t in texts)
        assert any("https://rfc.io" in t for t in texts)
        # 'lift' pair matched in unit 0 prompt
        assert any("الرفع" in c for c in client.calls)
        # files written
        assert (tmp_path / "chunk-0001.meta.json").exists()
        assert (tmp_path / "chunk-0001.md").exists()

    def test_resume_skips_when_fresh(self, tmp_path):
        client = StubClient()
        summaries = SummaryTracker(client, tmp_path / "sum.json", period=100)
        translate_chunk(_chunk(), client, GLOSSARY, tmp_path,
                        prompt_version="m1-v1", glossary_version="gv1",
                        summaries=summaries)
        client2 = StubClient()
        translate_chunk(_chunk(), client2, GLOSSARY, tmp_path,
                        prompt_version="m1-v1", glossary_version="gv1",
                        summaries=summaries)
        assert client2.calls == []          # no model calls: fresh draft

    def test_cache_key_invalidates_on_inputs(self, tmp_path):
        client = StubClient()
        summaries = SummaryTracker(client, tmp_path / "sum.json", period=100)
        translate_chunk(_chunk(), client, GLOSSARY, tmp_path,
                        prompt_version="m1-v1", glossary_version="gv1",
                        summaries=summaries)
        client2 = StubClient()
        # same inputs -> skip
        translate_chunk(_chunk(), client2, GLOSSARY, tmp_path,
                        prompt_version="m1-v1", glossary_version="gv1",
                        summaries=summaries)
        # changed model name (tag) -> different cache key -> retranslate
        client3_model = _chunk()
        for u in client3_model["units"]:
            u["text"] = u["text"] + " tweaked"
        translate_chunk(client3_model, StubClient(), GLOSSARY, tmp_path,
                        prompt_version="m1-v1", glossary_version="gv1",
                        summaries=summaries)
        metas = list(tmp_path.glob("chunk-0001.meta.json"))
        data = json.loads(metas[0].read_text())
        assert data["units"][0]["text"].endswith("tweaked")


class TestSummaryTracker:
    def test_refresh_period(self, tmp_path):
        client = StubClient()
        t = SummaryTracker(client, tmp_path / "s.json", period=2)
        for i in range(2):
            t.note_completion("ch1", f"source {i}", f"target {i}")
        assert t.summary_for("ch1") == "stub summary of topics"
        assert (tmp_path / "s.json").exists()


class TestAssemble:
    @pytest.fixture
    def book(self):
        # reuse the mini-book fixture builder by direct import
        tests = json.loads((Path(__file__).parent / "test_book_chunker.py")
                           .read_text(encoding="utf-8"))
        return tests

    def _mini_book(self, tmp_path):
        from tests.test_book_chunker import _block
        main = tmp_path / "mini.json"
        blocks = [
            _block(0, "SectionHeader", "<h1><p>1 Design</p></h1>", 1,
                   {"1": "/page/1/SectionHeader/0"}, None, "0"),
            _block(0, "Text", "<p>The lift increases.</p>", 1,
                   {"1": "/page/1/SectionHeader/0"}, None, "1"),
            _block(0, "Equation", "<math>y=f(x)</math>", 1,
                   {"1": "/page/1/SectionHeader/0"}, None, "2"),
        ]
        raw = {
            "page_info": {"0": {"bbox": [0, 0, 10, 10], "polygon": []},
                          "1": {"bbox": [0, 0, 10, 10], "polygon": []}},
            "blocks": [],
        }
        # blocks fake owners
        for blk in blocks:
            raw["blocks"].append(blk)
        main.write_text(json.dumps(raw), encoding="utf-8")
        from book_harness.book import Book
        return Book.from_marker_json(main)

    def test_render_book(self, tmp_path):
        book = self._mini_book(tmp_path)
        chunks = []
        from book_harness.chunker import plan_chunks
        chunks = plan_chunks(book)
        # stub a draft for the text unit only
        chunk = chunks[0]
        units = chunk.to_dict()["units"]
        meta = {
            "chunk_id": chunk.chunk_id,
            "model": "stub", "prompt_version": "m1-v1", "glossary_version": "gv1",
            "cache_key": "k", "finished": "now", "duration_sec": 1.0,
            "units": [{"uid": u["uid"], "block_id": u["block_id"],
                       "kind": u["kind"], "text": f"[Ar] {u['text']}",
                       "number": u.get("number", "")} for u in units],
        }
        ballots = tmp_path / "04_drafts"
        ballots.mkdir()
        (ballots / f"{chunk.chunk_id}.meta.json").write_text(json.dumps(meta))
        out = tmp_path / "07_output"
        manifest = render_book(book, chunks, DraftBook(ballots), out, book_id="mini")
        md = (out / "book.md").read_text(encoding="utf-8")
        assert "<!-- ===== page 1 ===== -->" in md
        assert "[Ar] Design" in md or "Design" in md
        # math passthrough survived
        assert "y" in md and "$" in md
        assert manifest["chunks_total"] == len(chunks)


# ---------------------------------------------------------------------------
# Server-side refusal surfacing (HTTP 500 body → RuntimeError)
# ---------------------------------------------------------------------------

class TestHttpPostErrorDetail:
    def test_http_error_body_surfaced(self, monkeypatch):
        import io
        import urllib.error

        from book_harness.translate import OllamaClient, _http_error_detail

        exc = urllib.error.HTTPError(
            "http://x/api/generate", 500, "Internal Server Error", None,
            io.BytesIO(b'{"error":"prediction aborted, token repeat limit reached"}'),
        )
        assert _http_error_detail(exc) == \
            "prediction aborted, token repeat limit reached"

        # direct: _post wraps a raised HTTPError with the body reason
        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError(
                "http://x/api/generate", 500, "Internal Server Error", None,
                io.BytesIO(b'{"error":"prediction aborted, token repeat limit reached"}'),
            )

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        client = OllamaClient(host="http://x")
        with pytest.raises(RuntimeError, match="token repeat limit"):
            client._post("/api/generate", {"prompt": "hi"})
