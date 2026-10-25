"""Tests for the guarded refinement loop (M2: 2.2) and QE scheduling (2.3)."""

import json
from pathlib import Path

from book_harness.refine import (
    apply_patch,
    estimate_spans,
    load_refinements,
    parse_spans,
    plan_refinement,
    refine_chunk,
    scrub_spans,
    verify_patch,
)


class StubAssistant:
    """Duck-typed Ollama client: returns canned estimate/verify replies."""

    def __init__(self, spans_reply="[]", votes=("A", "A")):
        self.model = "stub-assist"
        self.spans_reply = spans_reply
        self.votes = list(votes)
        self.calls: list[str] = []

    def generate(self, prompt, *, system=None, options=None):
        self.calls.append(prompt)
        if system and "MQM" in system:
            return self.spans_reply
        # verify call: look for the SOURCE block and which body is patched
        if "Which translation is better" in prompt:
            if not self.votes:
                return "A"
            # vote with the next prepared letter
            return self._next_vote(prompt)
        return ""

    def _next_vote(self, prompt):
        vote = self.votes.pop(0)
        self.votes.append(vote if False else vote)  # keep list non-empty
        # decide A or B by matching which body appears where: simplified tests
        return vote


def _chunk(units=None):
    return {
        "chunk_id": "chunk-0001",
        "section_title": "1 Design",
        "units": units or [
            {"uid": "u0", "block_id": "/page/1/Text/0", "kind": "text",
             "text": "The lift is produced by the wing."},
        ],
    }


def _draft(units=None):
    return {
        "chunk_id": "chunk-0001",
        "cache_key": "draft-key-1",
        "units": units or [
            {"uid": "u0", "kind": "text", "text": "الرفع تنتجه الجناح.",},
        ],
    }


class TestParseSpans:
    def test_clean_json(self):
        reply = '[{"original": "x", "suggestion": "y", "category": "terminology", "severity": "major"}]'
        spans = parse_spans(reply)
        assert len(spans) == 1
        assert spans[0]["original"] == "x"

    def test_wrapped_in_chatter(self):
        reply = 'Here are the errors:\n```json\n[{"original": "xx", "suggestion": "yy"}]\n```'
        assert parse_spans(reply)[0]["original"] == "xx"

    def test_garbage_returns_empty(self):
        assert parse_spans("I cannot help with that.") == []

    def test_non_dict_items_dropped(self):
        assert parse_spans('["oops", {"original": "a", "suggestion": "b"}]') == [
            {"original": "a", "suggestion": "b"}
        ]


class TestScrubSpans:
    def test_not_found_rejected(self):
        kept, rejected = scrub_spans(
            [{"original": "missing", "suggestion": "x"}], "text")
        assert kept == [] and rejected[0]["reason"] == "not_found"

    def test_ambiguous_rejected(self):
        kept, rejected = scrub_spans(
            [{"original": "the", "suggestion": "EL"}],
            "the lift over the wing")
        assert kept == [] and rejected[0]["reason"] == "ambiguous"

    def test_no_change_rejected(self):
        kept, rejected = scrub_spans(
            [{"original": "same", "suggestion": "same"}], "same word")
        assert rejected[0]["reason"] == "no_change"

    def test_budget_cap(self):
        spans = [{"original": f"g{i}", "suggestion": f"G{i}"} for i in range(20)]
        kept, rejected = scrub_spans(spans, "g0 g1 g2", max_spans=2)
        assert len(kept) == 2 and len(rejected) == 18


class TestApplyPatch:
    def test_applied_once(self):
        new, status = apply_patch("aaa bbb ccc", "bbb", "BBB")
        assert (new, status) == ("aaa BBB ccc", "applied")

    def test_not_found(self):
        assert apply_patch("abc", "zzz", "Z")[1] == "not_found"

    def test_ambiguous_never_applied(self):
        out, status = apply_patch("x x x", "x", "y")
        assert out == "x x x" and status == "ambiguous"


class TestVerifyPatch:
    def test_verify_only_accepts_unanimous(self):
        # A "correct" judge that always picks the body containing the marker FIX
        class FixPicker:
            def generate(self, prompt, **kw):
                bodies = prompt.split("A:\n", 1)[1]
                a_body, b_body = bodies.split("\nB:\n", 1)
                return "A" if "FIX" in a_body else "B"

        class NeverFix:
            def generate(self, prompt, **kw):
                bodies = prompt.split("A:\n", 1)[1]
                a_body, b_body = bodies.split("\nB:\n", 1)
                return "A" if "FIX" not in a_body else "B"

        class HalfFix:
            seq = ("B", "B")   # first call picks the fix, second call refuses it

            def __init__(self):
                self.votes = iter(self.seq)

            def generate(self, prompt, **kw):
                bodies = prompt.split("A:\n", 1)[1]
                a_body, b_body = bodies.split("\nB:\n", 1)
                return next(self.votes)

        src = "wing lift"
        before, after = "الرفع القديم", "الرفع FIX الجديد"
        # votes always flag the patched side (2nd arg) regardless of position
        assert verify_patch(FixPicker(), src, before, after)["accepted"] is True
        assert verify_patch(FixPicker(), src, before, after)["votes"] == [True, True]
        # and flag the original when the patch is passed as the first body
        assert verify_patch(FixPicker(), src, after, before)["votes"] == [False, False]

        bad = verify_patch(NeverFix(), src, before, after)
        assert bad["accepted"] is False and bad["votes"] == [False, False]

        half = verify_patch(HalfFix(), src, before, after)
        assert half["votes"] == [True, False]
        assert half["accepted"] is False          # 1/2 votes are not enough

    def test_model_failure_rejects_patch(self):
        """Ollama aborts (HTTP 500 'token repeat limit reached') land as a
        rejected patch, not a stage crash (PLAN: failed verify keeps draft)."""
        class Boom:
            def generate(self, prompt, **kw):
                raise RuntimeError(
                    "Ollama HTTP 500 at /api/generate: "
                    "prediction aborted, token repeat limit reached"
                )

        verdict = verify_patch(Boom(), "src", "before", "after")
        assert verdict["accepted"] is False
        assert verdict["votes"] == [False, False]
        assert "token repeat limit" in verdict["error"]


class TestRefineChunk:
    def test_accepted_patch_lands_in_refined(self, tmp_path: Path):
        spans = json.dumps([{
            "original": "الجناح错了",
            "suggestion": "",
        }])  # will be scrubbed out (empty suggestion)
        client = StubAssistant(spans_reply=spans)
        rec = refine_chunk(_chunk(), _draft(), client, tmp_path,
                           draft_cache_key="draft-key-1", prompt_version="r1")
        assert rec["counters"]["spans"] == 0

    def test_end_to_end_accepted(self, tmp_path: Path):
        # draft has typo "الجناحك" -> spans propose "الجناحك" -> "الجناح"
        spans_reply = json.dumps([{
            "original": "الرفع انتجه",
            "suggestion": "الرفع تنتجه",
            "category": "accuracy",
            "severity": "major",
        }])
        draft_text = "الرفع انتجه الجناح."

        class Client:
            model = "stub-assist"

            def generate(self, prompt, *, system=None, options=None):
                if system and "MQM" in system:
                    return spans_reply
                # verify: always choose the body containing "تنتجه"
                bodies = prompt.split("A:\n", 1)[1]
                a_body, b_body = bodies.split("\nB:\n", 1)
                return "A" if "تنتجه" in a_body else "B"

        chunk = _chunk()
        draft = _draft([{"uid": "u0", "kind": "text", "text": draft_text}])
        rec = refine_chunk(chunk, draft, Client(), tmp_path,
                           draft_cache_key="draft-key-1", prompt_version="r1")
        assert rec["counters"]["spans"] == 1
        assert rec["counters"]["accepted"] == 1
        assert rec["units"][0]["text"] == "الرفع تنتجه الجناح."
        assert rec["units"][0]["changed"] is True
        # artifacts
        patch_lines = [json.loads(l) for l in
                       (tmp_path / "chunk-0001.patches.jsonl").read_text().splitlines()]
        assert patch_lines[0]["accepted"] is True
        verdict = json.loads((tmp_path / "chunk-0001.verdicts.jsonl").read_text().splitlines()[0])
        assert verdict["accepted"] is True

    def test_resumable_and_stale_draft_ignored(self, tmp_path: Path):
        client = StubAssistant(spans_reply="[]")

        class CountingClient(StubAssistant):
            def __init__(self):
                super().__init__(spans_reply="[]")
                self.n = 0

            def generate(self, prompt, **kw):
                self.n += 1
                return "[]"

        cc = CountingClient()
        refine_chunk(_chunk(), _draft(), cc, tmp_path,
                     draft_cache_key="draft-key-1", prompt_version="r1")
        after_first = cc.n
        refine_chunk(_chunk(), _draft(), cc, tmp_path,
                     draft_cache_key="draft-key-1", prompt_version="r1")
        assert cc.n == after_first          # second run: zero model calls

        # a changed draft cache key marks the refinement stale and re-runs
        new_rec = refine_chunk(_chunk(), _draft(), cc, tmp_path,
                               draft_cache_key="draft-key-2", prompt_version="r1")
        assert new_rec["draft_cache_key"] == "draft-key-2"
        assert cc.n > after_first

    def test_load_refinements_matches_fresh_drafts_only(self, tmp_path: Path):
        client = StubAssistant(spans_reply="[]")
        refine_chunk(_chunk(), _draft(), client, tmp_path,
                     draft_cache_key="draft-key-1", prompt_version="r1")
        # fresh: meta cache_key matches refined draft_cache_key
        drafts_meta = {"chunk-0001": {"cache_key": "draft-key-1"}}
        assert load_refinements(tmp_path, drafts_meta) == {}
        # stale: key mismatch -> nothing applied
        stale_meta = {"chunk-0001": {"cache_key": "other-key"}}
        assert load_refinements(tmp_path, stale_meta) == {}


class TestPlanRefinement:
    def _drafts(self, ids):
        return {cid: {"cache_key": "k"} for cid in ids}

    def test_worst_first(self):
        chunks = [{"chunk_id": f"chunk-{i}", "units": []} for i in range(3)]
        scores = {
            "chunk-0": {"score": 90.0},
            "chunk-1": {"score": 40.0},
            "chunk-2": {"score": 60.0},
        }
        plan = plan_refinement(chunks, self._drafts(["chunk-0", "chunk-1", "chunk-2"]),
                              scores, quality_threshold=80.0)
        assert [c["chunk_id"] for c, _ in plan] == ["chunk-1", "chunk-2"]

    def test_unscored_chunks_go_first(self):
        chunks = [{"chunk_id": "chunk-a", "units": []},
                  {"chunk_id": "chunk-b", "units": []}]
        scores = {"chunk-b": {"score": 30.0}}
        plan = plan_refinement(chunks, self._drafts(["chunk-a", "chunk-b"]), scores)
        assert plan[0][0]["chunk_id"] == "chunk-a"
        assert plan[0][1] == 0.0

    def test_budget_and_undrafted_skip(self):
        chunks = [{"chunk_id": f"chunk-{i}", "units": []} for i in range(4)]
        drafts = self._drafts(["chunk-0", "chunk-1", "chunk-2", "chunk-3"])
        drafts.pop("chunk-3")
        plan = plan_refinement(chunks, drafts, {}, budget=2)
        assert len(plan) == 2
        assert all(c["chunk_id"] != "chunk-3" for c, _ in plan)
