"""Tests for LLM-based term extraction v2 (book_harness/llm_terms.py).

Uses the same synthetic mini book as ``test_book_chunker.py`` plus a fake
extract client with canned replies; no real model is involved.
"""

import json
from pathlib import Path

import pytest

from book_harness import config
from book_harness.book import Book
from book_harness import llm_terms

_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _block(bid, btype, html, page, hierarchy=None, rel=None):
    return {
        "id": f"/page/{page}/{btype}/{rel}",
        "block_type": btype,
        "html": html,
        "page": page + 990,          # deliberately bogus; ids drive pages
        "polygon": [], "bbox": [0, 0, 10, 10],
        "section_hierarchy": hierarchy or {},
        "images": {},
    }


@pytest.fixture
def mini_book(tmp_path: Path) -> Book:
    blocks = [
        _block(0, "SectionHeader", "<h1><p>1 Design</p></h1>", 1,
               {"1": "/page/1/SectionHeader/0"}, "0"),
        _block(0, "Text",
               "<p>The flight simulator computes equations of motion.</p>", 1,
               {"1": "/page/1/SectionHeader/0"}, "1"),
        _block(2, "SectionHeader", "<h1><p>2 Simulation</p></h1>", 3,
               {"1": "/page/3/SectionHeader/0"}, "0"),
        _block(2, "Text",
               "<p>The motion platform reproduces realistic flight dynamics.</p>", 3,
               {"1": "/page/3/SectionHeader/0"}, "1"),
    ]
    raw = {
        "page_info": {str(i): {"bbox": [0, 0, 100, 100], "polygon": []}
                      for i in range(4)},
        "blocks": blocks,
    }
    meta_raw = {
        "table_of_contents": [],
        "page_stats": [{"page_id": 0, "text_extraction_method": "surya",
                        "block_counts": []}] * 1,
    }
    main = tmp_path / "mini.json"
    meta = tmp_path / "mini_meta.json"
    main.write_text(json.dumps(raw), encoding="utf-8")
    meta.write_text(json.dumps(meta_raw), encoding="utf-8")
    return Book.from_marker_json(main)


class FakeExtract:
    """Canned replies keyed by substring of the prompt; counts calls."""

    model = "fake-extract:0b"

    def __init__(self, replies: dict[str, str], default: str = "") -> None:
        self.replies = replies
        self.default = default
        self.calls = 0
        self.last_prompt = None

    def generate(self, prompt, *, system, options):
        self.calls += 1
        self.last_prompt = prompt
        for key, reply in self.replies.items():
            if key in prompt:
                return reply
        return self.default


# ---------------------------------------------------------------------------
# strip_think
# ---------------------------------------------------------------------------

class TestStripThink:
    def test_removes_think_block(self):
        reply = (
            _THINK_OPEN + "Let me look at the text." + _THINK_CLOSE +
            "\nflight dynamics\n"
        )
        assert llm_terms.strip_think(reply) == "flight dynamics"

    def test_plain_reply_untouched(self):
        assert llm_terms.strip_think("wing loading\n") == "wing loading"


# ---------------------------------------------------------------------------
# Chunking for extraction
# ---------------------------------------------------------------------------

class TestChunkTexts:
    def test_chunks_in_reading_order(self, mini_book):
        chunks = llm_terms.chunk_texts(mini_book)
        assert len(chunks) >= 2
        pages = [c["first_page"] for c in chunks]
        assert pages == sorted(pages)
        assert all(c["text"] for c in chunks)
        assert all(c["source_hash"] for c in chunks)
        assert all(c["section"] for c in chunks)

    def test_source_hash_is_stable(self, mini_book):
        first = llm_terms.chunk_texts(mini_book)
        second = llm_terms.chunk_texts(mini_book)
        assert [c["source_hash"] for c in first] == \
            [c["source_hash"] for c in second]


# ---------------------------------------------------------------------------
# Per-chunk extraction
# ---------------------------------------------------------------------------

class TestExtractChunkTerms:
    def test_one_generate_call_per_chunk(self):
        client = FakeExtract({})
        terms = llm_terms.extract_chunk_terms(
            client, "1 Design", "chunk body", logger=None
        )
        assert client.calls == 1
        assert terms == []   # default reply is empty → parses to nothing

    def test_canned_reply_parsed(self):
        reply = "- flight simulator\nequations of motion — describes it\n"
        client = FakeExtract({}, default=reply)
        terms = llm_terms.extract_chunk_terms(client, "1 Design", "text")
        assert terms == ["flight simulator", "equations of motion"]

    def test_think_block_stripped_before_parsing(self):
        reply = (_THINK_OPEN + "hmm" + _THINK_CLOSE + "\nwing loading\n")
        client = FakeExtract({}, default=reply)
        terms = llm_terms.extract_chunk_terms(client, "1 Design", "text")
        assert terms == ["wing loading"]

    def test_word_limit_in_prompt(self):
        client = FakeExtract({})
        llm_terms.extract_chunk_terms(client, "1 Design", "text")
        assert (
            "at most "
            f"{config.EXTRACT_MAX_TERM_WORDS} words" in client.last_prompt
        )

    def test_phrases_over_word_limit_rejected(self):
        reply = (
            "psoriasis\nguttate psoriasis\n"
            "psoriasis with joint involvement\n"          # 4 words → drop
            "nail pitting\n"
            "generalised pustular psoriasis with fever\n"  # 5 words → drop
        )
        client = FakeExtract({}, default=reply)
        terms = llm_terms.extract_chunk_terms(client, "1 Design", "text")
        assert terms == ["psoriasis", "guttate psoriasis", "nail pitting"]

    def test_single_word_terms_kept(self):
        client = FakeExtract({}, default="melanoma\nnystatin\n")
        terms = llm_terms.extract_chunk_terms(client, "1 Design", "text")
        assert terms == ["melanoma", "nystatin"]

    def test_generate_error_raises(self):
        class Boom:
            model = "boom"

            def generate(self, *a, **k):
                raise ConnectionError("server down")

        with pytest.raises(RuntimeError, match="model call failed"):
            llm_terms.extract_chunk_terms(Boom(), "1 Design", "text")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestBuildCandidates:
    def test_freq_df_page_score(self):
        cands = llm_terms.build_candidates([
            (2, ["flight dynamics", "simulator"]),
            (5, ["flight dynamics"]),
            (9, ["wing loading"]),
        ])
        by_term = {c["term"]: c for c in cands}
        fd = by_term["flight dynamics"]
        assert fd["freq"] == 2 and fd["df"] == 2
        assert fd["page"] == 2
        assert fd["score"] == round(
            config.EXTRACT_SCORE_BASE + config.EXTRACT_SCORE_STEP, 3)
        assert fd["signal"] == config.EXTRACT_SIGNAL
        assert by_term["simulator"]["freq"] == 1
        assert by_term["simulator"]["score"] == config.EXTRACT_SCORE_BASE

    def test_case_insensitive_dedupe_keeps_first_casing(self):
        cands = llm_terms.build_candidates([
            (1, ["Flight Dynamics"]),
            (3, ["flight dynamics"]),
        ])
        assert len(cands) == 1
        assert cands[0]["term"] == "Flight Dynamics"
        assert cands[0]["freq"] == 2

    def test_sorted_by_freq_then_term_and_capped(self):
        terms = [f"term {i:03d}" for i in range(30)]
        cands = llm_terms.build_candidates(
            [(1, terms)] * 3,   # every term in 3 chunks
            max_candidates=10,
        )
        assert len(cands) == 10
        assert cands[0]["freq"] == 3

    def test_empty_input(self):
        assert llm_terms.build_candidates([]) == []


# ---------------------------------------------------------------------------
# Thinking disabled for the extract role
# ---------------------------------------------------------------------------

class TestThinkOption:
    def test_extract_call_passes_think_false(self, mini_book, tmp_path):
        seen: list[dict] = []

        class Recorder(FakeExtract):
            def generate(self, prompt, *, system, options):
                seen.append(options)
                self.calls += 1
                return ""

        llm_terms.extract_candidates(
            mini_book, Recorder({}), cache_path=tmp_path / "cache.jsonl")
        assert seen
        assert all(o.get("think") is False for o in seen)

    def test_ollama_hoists_think_to_top_level(self, monkeypatch):
        from book_harness.translate import OllamaClient

        payloads: list[tuple[str, dict]] = []

        def fake_post(self, path, payload, timeout=None):
            payloads.append((path, payload))
            return {"response": "ok"}

        monkeypatch.setattr(OllamaClient, "_post", fake_post)
        client = OllamaClient(host="http://x")
        client.model = "qwen3.5:4b"
        out = client.generate("hi", system="s", options={"think": False})
        assert out == "ok"
        path, payload = payloads[0]
        assert path == "/api/generate"
        assert payload["think"] is False
        assert "think" not in payload["options"]

    def test_ollama_pins_think_false_by_default(self, monkeypatch):
        """PLAN: ``think: False`` is pinned in ``GENERATE_OPTIONS`` — Ollama
        hoists it to the request top level so thinking-capable assist models
        (gemma4:26b) can never burn the context window on a think channel."""
        from book_harness.translate import OllamaClient

        payloads: list[dict] = []

        def fake_post(self, path, payload, timeout=None):
            payloads.append(payload)
            return {"response": "ok"}

        monkeypatch.setattr(OllamaClient, "_post", fake_post)
        client = OllamaClient(host="http://x")
        client.model = "some-model"
        client.generate("hi")
        assert payloads[0]["think"] is False
        assert "think" not in payloads[0]["options"]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestExtractCandidates:
    def test_none_client_raises(self, mini_book):
        with pytest.raises(RuntimeError, match="no extract client"):
            llm_terms.extract_candidates(mini_book, None)

    def test_one_call_per_chunk_and_output(self, mini_book, tmp_path):
        client = FakeExtract({
            "simulator": "flight dynamics\nsimulator design\n",
            "motion platform": "motion platform\n",
        })
        cache = tmp_path / "extract-cache.jsonl"
        cands = llm_terms.extract_candidates(mini_book, client, cache_path=cache)
        by_term = {c["term"]: c for c in cands}
        assert by_term["flight dynamics"]["freq"] == 1
        assert by_term["motion platform"]["freq"] == 1
        assert all(c["signal"] == "llm" for c in cands)
        assert client.calls == 2   # one per chunk
        # cache file has one record per call
        records = [json.loads(ln) for ln in cache.read_text().splitlines()]
        assert len(records) == client.calls

    def test_empty_replies_accepted(self, mini_book, tmp_path):
        client = FakeExtract({}, default="")
        cands = llm_terms.extract_candidates(
            mini_book, client, cache_path=tmp_path / "cache.jsonl")
        assert cands == []
        assert client.calls == 2

    def test_cache_hit_skips_generate(self, mini_book, tmp_path):
        cache = tmp_path / "extract-cache.jsonl"
        first = FakeExtract({"flight": "flight dynamics\n"})
        run1 = llm_terms.extract_candidates(mini_book, first, cache_path=cache)
        assert first.calls == 2   # one per chunk

        served = {"n": 0}

        class NoCall(FakeExtract):
            def __init__(self) -> None:
                self.replies, self.default, self.calls = {}, "", 0
                self.model = "fake-extract:0b"

            def generate(self, prompt, *, system, options):
                served["n"] += 1
                return ""

        second = NoCall()
        run2 = llm_terms.extract_candidates(mini_book, second, cache_path=cache)
        assert served["n"] == 0           # everything served from the cache
        assert run2 == run1

    def test_prompt_or_model_change_invalidates_cache(self, mini_book, tmp_path,
                                                      monkeypatch):
        cache = tmp_path / "extract-cache.jsonl"
        first = FakeExtract({"flight": "flight dynamics\n"})
        llm_terms.extract_candidates(mini_book, first, cache_path=cache)

        other = FakeExtract({})
        other.model = "other-model:0b"
        llm_terms.extract_candidates(mini_book, other, cache_path=cache)
        assert other.calls == 2   # not served from the first model's cache


# ---------------------------------------------------------------------------
# LLM verification pass (PLAN §9)
# ---------------------------------------------------------------------------

class FakeAssist:
    """Canned verification replies; counts calls."""

    model = "fake-assist:0b"

    def __init__(self, reply: str = "", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0
        self.last_prompt = None

    def generate(self, prompt, *, system, options):
        self.calls += 1
        self.last_prompt = prompt
        if self.fail:
            raise ConnectionError("assist down")
        return self.reply


def _cand(term, freq=1, **extra):
    return {"term": term, "freq": freq, "df": freq, "page": 1, "score": 1.0,
            "signal": "llm", **extra}


class TestParseVerdicts:
    def test_keep_and_drop_lines(self):
        verdicts = llm_terms.parse_verdicts(
            "flight dynamics: keep\nhair: drop\npsoriasis: keep\n")
        key = llm_terms._dedupe_key
        assert verdicts == {key("flight dynamics"): True,
                            key("hair"): False,
                            key("psoriasis"): True}

    def test_numbering_stripped(self):
        verdicts = llm_terms.parse_verdicts(
            "1. flight dynamics: keep\n2) hair: drop\n")
        assert llm_terms.parse_verdicts("flight dynamics: keep") == \
            {llm_terms._dedupe_key("flight dynamics"): True}
        key = llm_terms._dedupe_key
        assert verdicts[key("flight dynamics")] is True
        assert verdicts[key("hair")] is False

    def test_think_block_stripped(self):
        reply = (_THINK_OPEN + "judging..." + _THINK_CLOSE
                 + "\nflight dynamics: keep\n")
        verdicts = llm_terms.parse_verdicts(reply)
        assert verdicts == {llm_terms._dedupe_key("flight dynamics"): True}

    def test_unparseable_lines_ignored(self):
        assert llm_terms.parse_verdicts("just some prose\n") == {}


class TestVerifyCandidates:
    def test_empty_candidates_noop(self):
        assert llm_terms.verify_candidates([], FakeAssist()) == []

    def test_no_client_returns_input(self):
        cands = [_cand("flight dynamics")]
        assert llm_terms.verify_candidates(cands, None) is cands

    def test_drops_verdicted_terms_keeps_unanswered(self, tmp_path):
        cands = [_cand("flight dynamics"), _cand("hair"),
                 _cand("never mentioned")]
        client = FakeAssist("flight dynamics: keep\nhair: drop\n")
        out = llm_terms.verify_candidates(
            cands, client, cache_path=tmp_path / "verify-cache.jsonl")
        assert [c["term"] for c in out] == ["flight dynamics",
                                            "never mentioned"]

    def test_cache_hit_skips_generate(self, tmp_path):
        cache = tmp_path / "verify-cache.jsonl"
        cands = [_cand("flight dynamics"), _cand("hair")]
        first = FakeAssist("flight dynamics: keep\nhair: drop\n")
        llm_terms.verify_candidates(cands, first, cache_path=cache)
        assert first.calls == 1

        second = FakeAssist("everything: keep\n")
        out = llm_terms.verify_candidates(cands, second, cache_path=cache)
        assert second.calls == 0          # served entirely from cache
        assert [c["term"] for c in out] == ["flight dynamics"]

    def test_failed_call_keeps_batch(self, tmp_path):
        cands = [_cand("flight dynamics"), _cand("hair")]
        client = FakeAssist(fail=True)
        out = llm_terms.verify_candidates(
            cands, client, cache_path=tmp_path / "verify-cache.jsonl")
        assert out == cands               # graceful: unchanged

    def test_prompt_or_model_change_invalidates_cache(self, tmp_path,
                                                      monkeypatch):
        cache = tmp_path / "verify-cache.jsonl"
        cands = [_cand("flight dynamics")]
        llm_terms.verify_candidates(cands, FakeAssist("flight dynamics: keep\n"),
                                    cache_path=cache)

        other = FakeAssist("flight dynamics: keep\n")
        other.model = "other-assist:0b"
        llm_terms.verify_candidates(cands, other, cache_path=cache)
        assert other.calls == 1           # not served from the first cache

        monkeypatch.setattr(config, "VERIFY_PROMPT_ID", "verify-v2")
        third = FakeAssist("flight dynamics: keep\n")
        llm_terms.verify_candidates(cands, third, cache_path=cache)
        assert third.calls == 1

    def test_batching(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "VERIFY_BATCH_SIZE", 2)
        cands = [_cand(f"term {i:03d}") for i in range(5)]
        client = FakeAssist("all: keep\n")
        out = llm_terms.verify_candidates(
            cands, client, cache_path=tmp_path / "verify-cache.jsonl")
        assert client.calls == 3          # ceil(5 / 2)
        assert len(out) == 5
