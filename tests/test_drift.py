"""Glossary drift loop (task 3.6): mining, proposals, human confirmation."""

import json
from pathlib import Path

from book_harness import drift as drift_mod
from book_harness.glossary import GlossaryEntry, glossary_version, load_glossary, write_glossary


def _chunk(chunk_id: str, text: str, page: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "pages": [page],
        "blocks": [{"block_id": f"/page/{page}/Text/0", "page": page,
                    "type": "Text", "units": ["u0"]}],
        "units": [{"uid": "u0", "block_id": f"/page/{page}/Text/0",
                   "kind": "text", "text": text, "placeholders": {}}],
    }


ENTRIES = [GlossaryEntry(  # "lift" subsumes "lift distribution" etc.
    term_en="lift", term_ar="الرفع", source="ksaa", wikidata=None, page=None,
    scan=None, status="certified", decided_by="auto", alternatives=[],
)]


class TestMining:
    def test_finds_recurring_uncovered_term(self):
        chunks = [
            _chunk("chunk-0001", "The boundary layer thickens with speed.", 1),
            _chunk("chunk-0002", "Boundary layer separation follows.", 2),
            _chunk("chunk-0003", "The boundary layer grows unbounded.", 3),
        ]
        out = drift_mod.mine_drift_terms(chunks, ENTRIES)
        terms = [c["term"] for c in out]
        assert "boundary layer" in terms
        rec = next(c for c in out if c["term"] == "boundary layer")
        assert rec["freq"] == 3 and rec["df"] == 3
        assert rec["page"] == 1
        assert set(rec["chunks"]) == {"chunk-0001", "chunk-0002", "chunk-0003"}

    def test_skips_subsumed_terms(self):
        chunks = [
            _chunk("chunk-0001", "The lift equation changed.", 1),
            _chunk("chunk-0002", "The lift equation holds.", 2),
            _chunk("chunk-0003", "The lift equation holds.", 3),
        ]
        out = drift_mod.mine_drift_terms(chunks, ENTRIES)
        assert "lift equation" not in [c["term"] for c in out]

    def test_threshold_and_dedup(self):
        chunks = [
            _chunk("chunk-0001", "Boundary layer stuff.", 1),
            _chunk("chunk-0002", "Boundary layer things.", 2),
        ]
        out = drift_mod.mine_drift_terms(chunks, ENTRIES, min_df=3)
        assert out == []

    def test_gram_quality_filter(self):
        """Placeholders, single letters and Latin abbreviation noise stay out."""
        chunks = [
            _chunk("chunk-0001", "See §EQ-00§ as shown: e.g. x y and et al.", 1),
            _chunk("chunk-0002", "s §URL-00§ i.e. y z and et al.", 2),
            _chunk("chunk-0003", "e.g. y and et al gleam.", 3),
        ]
        out = drift_mod.mine_drift_terms(chunks, ENTRIES)
        terms = [c["term"] for c in out]
        for junk in ("x y", "y z", "e g", "et al", "eq-00", "url-00"):
            assert junk not in terms

    def test_ordering_by_score(self):
        chunks = []
        for i in range(1, 6):
            chunks.append(_chunk(f"c{i}", "The specific impulse matters.", i))
        for i in range(3):
            chunks.append(_chunk(f"d{i}", "The flutter margin is small.", i + 10))
        out = drift_mod.mine_drift_terms(chunks, ENTRIES)
        terms = [c["term"] for c in out]
        assert "specific impulse" in terms
        assert "flutter margin" in terms
        impulses = next(c for c in out if c["term"] == "specific impulse")
        assert impulses["freq"] == 5 and impulses["df"] == 5
        # scores are sorted descending
        scores = [c["score"] for c in out]
        assert scores == sorted(scores, reverse=True)


class TestCandidateIO:
    def test_round_trip(self, tmp_path):
        chunks = [_chunk(f"c{i}", "The boundary layer thickens.", i)
                  for i in range(1, 4)]
        candidates = drift_mod.mine_drift_terms(chunks, ENTRIES)
        path = tmp_path / "drift-candidates.jsonl"
        n = drift_mod.write_candidates(candidates, path, glossary_version="gv1")
        assert n == len(candidates)
        loaded = drift_mod.load_candidates(path)
        assert [(r["term"], r["freq"], r["df"], r["chunks"]) for r in loaded] == [
            (c["term"], c["freq"], c["df"], c["chunks"]) for c in candidates
        ]
        assert all(r["status"] == "proposed" and r["decision"] is None for r in loaded)
        assert all(r["glossary_version"] == "gv1" for r in loaded)

    def test_load_missing_file(self, tmp_path):
        assert drift_mod.load_candidates(tmp_path / "nope.jsonl") == []


class TestConfirm:
    def test_parse_args(self):
        pairs = drift_mod.parse_confirm_args(["lift=الرفع", "wearout"])
        assert pairs == [{"en": "lift", "ar": "الرفع"},
                         {"en": "wearout", "ar": ""}]

    def test_append_and_skip_existing(self):
        new, appended = drift_mod.append_confirmed_entries(
            list(ENTRIES),
            [{"en": "boundary layer", "ar": "الطبقة الحدّية"},
             {"en": "lift", "ar": "x"}],
        )
        assert len(appended) == 1
        entry = new[-1]
        assert isinstance(entry, GlossaryEntry)
        assert entry["status"] == "uncertified"      # never silently certified
        assert entry["term_ar"] == "الطبقة الحدّية"
        assert entry["decided_by"] == "human"

    def test_no_arabic_means_no_translation(self):
        new, appended = drift_mod.append_confirmed_entries(
            list(ENTRIES), [{"en": "flutter margin", "ar": ""}])
        assert new[-1]["term_ar"] is None
        assert appended and new[-1]["decided_by"] == "human"


class TestAffectedChunks:
    def test_only_matching_chunks(self):
        chunks = [_chunk("c1", "The lift changes."), _chunk("c2", "Noise.")]
        assert drift_mod.affected_chunks(chunks, "lift") == ["c1"]
        assert drift_mod.affected_chunks(chunks, "flutter") == []


def _write_glossary_with_entries(tmp_path: Path):
    path = tmp_path / "glossary.yaml"
    write_glossary(list(ENTRIES), path)
    return path


def test_confirm_then_retranslate_loop(tmp_path, monkeypatch):
    """End-to-end file loop: confirm appends and bumps the glossary version."""
    path = _write_glossary_with_entries(tmp_path)
    old_version, entries = load_glossary(path)

    chunks = [_chunk("c1", "The boundary layer thickens, and the lift drops.")]
    entries, appended = drift_mod.append_confirmed_entries(
        entries, [{"en": "boundary layer", "ar": "الطبقة الحدّية"}])
    write_glossary(entries, path)
    new_version, reloaded = load_glossary(path)
    assert old_version != new_version
    assert len(reloaded) == 2
    assert appended[0]["term_en"] == "boundary layer"
    # and only the chunk that mentions the new term is affected
    assert drift_mod.affected_chunks(chunks, "boundary layer") == ["c1"]
