"""Glossary consistency stage tests (one rendering per term, post-glossary).

The stage was formerly the embedded "translation selection" phase; its
artifacts are now ``consistency-decisions.jsonl`` / ``consistency-cache.jsonl``
and the prompt shows chunk references (chunk id · section · page).
"""

import json
from pathlib import Path

from book_harness.glossary import (
    STATUS_CERTIFIED,
    STATUS_LLM_CHOSEN,
    STATUS_UNCERTIFIED,
    GlossaryEntry,
)
from book_harness.consistency import choose_translations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entry(
    term, ar, alternatives=(), sources=None, status=STATUS_CERTIFIED,
    decided_by="auto", source="damascus_physics",
):
    return GlossaryEntry(
        term_en=term, term_ar=ar, source=source, wikidata="Q1",
        page=7, scan=None, status=status, decided_by=decided_by,
        alternatives=list(alternatives),
        alternative_sources=list(sources or []),
        reason="highest dictionary priority",
    )


def _occ(term, chunk_id="chunk-0007", page=12, section="1 Design", context=None):
    return {
        "term": term, "freq": 2, "df": 2, "page": 1, "score": 1.0,
        "signal": "llm",
        "occurrences": [
            {"chunk_id": chunk_id, "page": page, "section": section,
             "context": context or f".. the {term} reacts to gusts .."},
        ],
    }


class FakeAssist:
    """Assist-model double: canned replies keyed by prompt substring."""

    def __init__(self, replies: dict[str, str]):
        self.replies = replies
        self.calls: list[str] = []
        self.model = "assist-mock"

    def generate(self, prompt, *, system=None, options=None):
        self.calls.append(prompt)
        for needle, reply in self.replies.items():
            if needle in prompt:
                return reply
        raise AssertionError("unexpected prompt")


class _Silent:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _run(entries, candidates, out_dir, replies):
    """Run consistency with a canned model over a tmp dir; return results."""
    client = FakeAssist(replies)
    entries, counts = choose_translations(
        entries, candidates, lambda: client, out_dir, logger=_Silent()
    )
    return entries, counts, client


# ---------------------------------------------------------------------------
# Stage updates entries + writes artifacts
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_model_pick_updates_entry_and_logs(self, tmp_path: Path):
        ent = _entry("cell", "خانة", ["خلية"], ["at_physics"])
        entries, counts, client = _run(
            [ent], [_occ("cell")], tmp_path, replies={"cell": "2"},
        )
        assert counts == {"chosen": 1, "kept": 0, "overridden": 0,
                           "fallback": 0, "skipped": 0}
        assert client.calls, "model must be consulted"
        # occurrence context reached the prompt — with the chunk reference
        prompt = client.calls[0]
        assert "reacts to gusts" in prompt
        assert "[chunk chunk-0007 · 1 Design · page 12]" in prompt
        e = entries[0]
        assert e["term_ar"] == "خلية"
        assert e["decided_by"] == "llm_context"
        assert e["status"] == STATUS_LLM_CHOSEN
        assert e["alternatives"] == ["خلية"]        # menu preserved
        log = [json.loads(ln) for ln in (
            tmp_path / "consistency-decisions.jsonl").read_text().splitlines()]
        assert log and log[0]["chosen"] == "خلية"
        assert log[0]["options"] == ["خانة", "خلية"]
        assert log[0]["prompt"] == "consistency-v1"
        cache = [json.loads(ln) for ln in (
            tmp_path / "consistency-cache.jsonl").read_text().splitlines()]
        assert cache and cache[0]["chosen"] == "خلية"

    def test_number_reply_maps_to_option_order(self, tmp_path: Path):
        ent = _entry("cell", "خانة", ["خلية", "ميناء"], ["a", "b"])
        entries, counts, _client = _run(
            [ent], [], tmp_path, replies={"cell": "3"},
        )
        assert counts["chosen"] == 1
        assert entries[0]["term_ar"] == "ميناء"

    def test_cache_replay_skips_model(self, tmp_path: Path):
        # populate the cache on a first pass
        _run([_entry("cell", "خانة", ["خلية"], ["at_physics"])],
             [_occ("cell")], tmp_path, replies={"cell": "2"})
        fresh = _entry("cell", "خانة", ["خلية"], ["at_physics"])

        def boom():
            raise AssertionError("model must not be consulted on cache hit")

        entries, counts = choose_translations(
            [fresh], [_occ("cell")], boom, tmp_path, logger=_Silent()
        )
        assert counts["chosen"] == 1
        assert entries[0]["term_ar"] == "خلية"
        assert entries[0]["decided_by"] == "llm_context"

    def test_model_failure_falls_back(self, tmp_path: Path):
        class Boom:
            model = "assist-mock"

            def generate(self, *a, **k):
                raise ConnectionError("server down")

        ent = _entry("cell", "خانة", ["خلية"], ["at_physics"])
        entries, counts = choose_translations(
            [ent], [_occ("cell")], lambda: Boom(), tmp_path, logger=_Silent()
        )
        assert counts["fallback"] == 1
        e = entries[0]
        assert e["term_ar"] == "خانة"                # deterministic keep
        assert e["status"] == STATUS_CERTIFIED
        assert "kept highest dictionary priority" in e["reason"]
        log = [json.loads(ln) for ln in (
            tmp_path / "consistency-decisions.jsonl").read_text().splitlines()]
        assert log[0]["chosen"] is None and log[0]["via"] == "fallback"

    def test_facility_failure_falls_back_too(self, tmp_path: Path):
        def broken_factory():
            raise RuntimeError("no server reachable")

        ent = _entry("cell", "خانة", ["خلية"], ["at_physics"])
        entries, counts = choose_translations(
            [ent], [], broken_factory, tmp_path, logger=_Silent()
        )
        assert counts["fallback"] == 1
        assert entries[0]["term_ar"] == "خانة"

    def test_ambiguous_reply_falls_back(self, tmp_path: Path):
        ent = _entry("cell", "خانة", ["خلية"], ["at_physics"])
        entries, counts, _client = _run(
            [ent], [_occ("cell")], tmp_path, replies={"cell": "7"},
        )
        assert counts["fallback"] == 1
        assert entries[0]["term_ar"] == "خانة"

    def test_text_reply_matched_when_no_number(self, tmp_path: Path):
        ent = _entry("cell", "خانة", ["خلية"], ["at_physics"])
        entries, counts, _client = _run(
            [ent], [], tmp_path, replies={"cell": "I would pick خلية."},
        )
        assert counts["chosen"] == 1
        assert entries[0]["term_ar"] == "خلية"

    def test_no_client_falls_back_without_crash(self, tmp_path: Path):
        ent = _entry("cell", "خانة", ["خلية"], ["at_physics"])
        entries, counts = choose_translations(
            [ent], [], None, tmp_path, logger=_Silent()
        )
        assert counts["fallback"] == 1
        assert entries[0]["term_ar"] == "خانة"

    def test_uncertified_and_unoccurred_single_option_skip(
        self, tmp_path: Path,
    ):
        # uncertified: nothing to decide — a lone certified form with NO
        # documented occurrences has no usage to audit against either
        unc = _entry("quantum fizzle", None, status=STATUS_UNCERTIFIED)
        single = _entry("lift", "الرفع")
        _, counts, client = _run(
            [unc, single], [], tmp_path, replies={"lift": "keep"},
        )
        assert counts["chosen"] == 0
        assert counts["kept"] == 0
        assert counts["overridden"] == 0
        assert counts["skipped"] == 2
        assert not client.calls
        assert not (tmp_path / "consistency-decisions.jsonl").exists()

    def test_options_carry_dictionary_sources_in_prompt(self, tmp_path: Path):
        ent = _entry("cell", "خانة", ["خلية"], ["at_physics"])
        _entries, _counts, client = _run(
            [ent], [], tmp_path, replies={"cell": "1"},
        )
        assert "خانة" in client.calls[0] and "خلية" in client.calls[0]
        assert "at_physics" in client.calls[0]

    def test_prompt_mentions_chunks_section_title(self, tmp_path: Path):
        ent = _entry("cell", "خانة", ["خلية"], ["at_physics"])
        _entries, _counts, client = _run(
            [ent], [_occ("cell")], tmp_path, replies={"cell": "1"},
        )
        assert "Chunks of this book where the term occurs" in client.calls[0]


class TestOverrideCheck:
    """Lone certified forms are re-audited against documented occurrences:
    the model either keeps the form or overrides it with a domain-fitting
    rendering."""

    # Arabic glyphs are visually ambiguous in source — code points are the
    # contract, so every form below is spelled out with unicode escapes.
    FORM = "\u067a\u0627\u0631\u0629 \u0627\u0644\u062a\u062d\u0643\u0645"      # plane (certified)
    MUSTAWEY = "\u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u062a\u062d\u0643\u0645"  # control level
    TABAQA = "\u0637\u0628\u0642\u0629 \u0627\u0644\u062a\u062d\u0643\u0645"     # control layer
    MARQA = "\u0645\u0631\u0642\u0629 \u0627\u0644\u062d\u0627\u0644\u0629"
    DICT = ("\u0645\u0639\u062c\u0645 \u0645\u0635\u0637\u0644\u062d\u0627\u062a"
            " \u0627\u0644\u0639\u0644\u0645 \u0648\u0627\u0644\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627"
            " (\u062a.1)")

    def _control_plane(self):
        # mirrors the real glossary entry: single certified form, no
        # alternatives — a dictionary form the book's own usage disagrees
        # with
        return _entry(
            "control plane", self.FORM, source=self.DICT,
        )

    def test_override_replaces_certified_form(self, tmp_path: Path):
        ent = self._control_plane()
        entries, counts, client = _run(
            [ent],
            [_occ("control plane", chunk_id="chunk-0034", page=18,
                  section="Chapter 2: Two control planes",
                  context=".. as part of a behavioral control plane ..")],
            tmp_path,
            replies={"control plane":
                     f"override to `{self.MUSTAWEY}`"},
        )
        assert counts["overridden"] == 1
        assert counts["skipped"] == 0
        assert client.calls and self.FORM in client.calls[0]
        assert "behavioral control plane" in client.calls[0]
        e = entries[0]
        assert e["term_ar"] == self.MUSTAWEY
        assert e["status"] == STATUS_LLM_CHOSEN
        assert e["decided_by"] == "llm_context"
        assert "consistency override" in e["reason"]
        # dictionary provenance survives the override — the citation is
        # what made it auditable
        assert e["source"] == self.DICT
        assert e["page"] == 7
        log = [json.loads(ln) for ln in (
            tmp_path / "consistency-decisions.jsonl").read_text().splitlines()]
        assert log[0]["via"] == "override"
        assert log[0]["prompt"] == "consistency-override-v1"
        cache = [json.loads(ln) for ln in (
            tmp_path / "consistency-cache.jsonl").read_text().splitlines()]
        assert cache[0]["chosen"] == e["term_ar"]

    def test_override_keep_preserves_certified_form(self, tmp_path: Path):
        ent = self._control_plane()
        entries, counts, client = _run(
            [ent], [_occ("control plane")], tmp_path,
            replies={"control plane": "keep"},
        )
        assert counts["kept"] == 1
        assert counts["overridden"] == 0
        assert client.calls
        e = entries[0]
        assert e["term_ar"] == self.FORM       # untouched
        assert e["status"] == STATUS_CERTIFIED
        assert e["decided_by"] == "auto"
        log = [json.loads(ln) for ln in (
            tmp_path / "consistency-decisions.jsonl").read_text().splitlines()]
        assert log[0]["via"] == "keep"
        assert log[0]["chosen"] == self.FORM

    def test_override_keep_replays_from_cache(self, tmp_path: Path):
        _run([self._control_plane()], [_occ("control plane")], tmp_path,
             replies={"control plane": "keep"})
        fresh = self._control_plane()

        def boom():
            raise AssertionError("model must not be consulted on cache hit")

        entries, counts = choose_translations(
            [fresh], [_occ("control plane")], boom, tmp_path,
            logger=_Silent(),
        )
        assert counts["kept"] == 1
        assert entries[0]["term_ar"] == self.FORM

    def test_override_without_backticks_uses_first_arabic_span(
        self, tmp_path: Path,
    ):
        ent = self._control_plane()
        entries, counts, _client = _run(
            [ent], [_occ("control plane")], tmp_path,
            replies={"control plane": f"override to: {self.TABAQA}"},
        )
        assert counts["overridden"] == 1
        assert entries[0]["term_ar"] == self.TABAQA

    def test_override_claim_without_form_falls_back(self, tmp_path: Path):
        ent = self._control_plane()
        entries, counts, _client = _run(
            [ent], [_occ("control plane")], tmp_path,
            replies={"control plane": "override, it does not fit"},
        )
        assert counts["fallback"] == 1
        assert counts["overridden"] == 0
        e = entries[0]
        assert e["term_ar"] == self.FORM       # certified kept
        assert e["status"] == STATUS_CERTIFIED
        assert "kept certified rendering" in e["reason"]
        # a fallback was not a decision — it is not cached
        assert not (tmp_path / "consistency-cache.jsonl").exists()

    def test_override_reply_quoting_certified_form_keeps(self, tmp_path: Path):
        ent = self._control_plane()
        entries, counts, _client = _run(
            [ent], [_occ("control plane")], tmp_path,
            replies={"control plane": f"override to `{self.FORM}`, keep as-is"},
        )
        assert counts["kept"] == 1
        assert counts["overridden"] == 0
        assert entries[0]["term_ar"] == self.FORM

    def test_llm_chosen_single_option_is_checked_too(self, tmp_path: Path):
        # a glossary-stage LLM pick is not immune to a context mismatch
        ent = _entry("state_db_bridge",
                     "\u062c\u0633\u0631 \u062d\u0627\u0644\u0629",
                     status=STATUS_LLM_CHOSEN, decided_by="llm")
        entries, counts, _client = _run(
            [ent], [_occ("state_db_bridge")], tmp_path,
            replies={"state_db_bridge":
                     f"override to `{self.MARQA}`"},
        )
        assert counts["overridden"] == 1
        assert entries[0]["term_ar"] == self.MARQA

