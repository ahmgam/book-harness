"""Tests for the deterministic technical-term focus filters (PLAN §9).

``book_harness/term_filter.py`` — the wordfreq backend is stubbed per test so
the zipf thresholds are exact and no real data file is needed.
"""

import pytest

from book_harness import config
from book_harness import term_filter as tf


@pytest.fixture(autouse=True)
def stub_zipf(monkeypatch):
    """Deterministic zipf stub: everyday words high, jargon words 0."""
    everyday = {"hair": 5.6, "money": 5.9, "house": 6.0, "system": 5.2,
                "mix": 4.9}
    monkeypatch.setattr(
        tf, "_ZIPF_FN", lambda word, lang: everyday.get(word, 0.0))
    monkeypatch.setattr(tf, "_ZIPF_CACHE", {})
    return everyday


# ---------------------------------------------------------------------------
# Per-chunk word-shape / stopword filter
# ---------------------------------------------------------------------------

class TestShapeReject:
    def test_stopword_single_word_rejected(self):
        assert tf.shape_reject("of") == "stopword"
        assert tf.shape_reject("The") == "stopword"
        assert tf.shape_reject("also") == "stopword"

    def test_connective_adverbs_rejected(self):
        assert tf.shape_reject("however") == "stopword"
        assert tf.shape_reject("therefore") == "stopword"

    def test_short_tokens_rejected(self):
        assert tf.shape_reject("am") == "stopword"   # stopword hit first
        assert tf.shape_reject("io") == "too-short"  # 2 chars, not stopword

    def test_numbers_rejected(self):
        assert tf.shape_reject("1234") == "number"
        assert tf.shape_reject("1997") == "number"   # year
        assert tf.shape_reject("3,5") == "number"

    def test_roman_numerals_rejected(self):
        assert tf.shape_reject("VII") == "roman-numeral"
        assert tf.shape_reject("xiv") == "roman-numeral"

    def test_roman_words_not_dropped(self):
        # real words that read as Roman numerals only matter when written
        # all-upper or all-lower ("mix" is a strict Roman reading — accepted
        # trade-off), but normal mixed-case prose words stay
        assert tf.shape_reject("civil") is None
        assert tf.shape_reject("Dim") is None

    def test_numbers_inside_phrase_rejected(self):
        assert tf.shape_reject("part 1234") == "number"

    def test_phrase_all_stopwords_rejected(self):
        assert tf.shape_reject("of the") == "all-stopwords"
        assert tf.shape_reject("in with") == "all-stopwords"

    def test_technical_terms_pass(self):
        assert tf.shape_reject("flight dynamics") is None
        assert tf.shape_reject("psoriasis") is None
        assert tf.shape_reject("wikipedia") is None


class TestFilterChunkTerms:
    def test_mixed_reply(self):
        terms = ["of", "the", "flight dynamics", "VII", "1997",
                 "wikipedia", "in with", "psoriasis"]
        assert tf.filter_chunk_terms(terms) == [
            "flight dynamics", "wikipedia", "psoriasis"]


# ---------------------------------------------------------------------------
# Commonness (zipf) gate
# ---------------------------------------------------------------------------

class TestPassesCommonness:
    def test_rare_single_word_kept(self):
        assert tf.passes_commonness("psoriasis", 1) is True

    def test_everyday_word_dropped_at_low_freq(self):
        assert tf.passes_commonness("hair", 1) is False
        assert tf.passes_commonness("money", 3) is False

    def test_everyday_word_jargon_escape_hatch(self):
        assert tf.passes_commonness(
            "hair", config.EXTRACT_JARGON_MIN_FREQ) is True
        assert tf.passes_commonness(
            "hair", config.EXTRACT_JARGON_MIN_FREQ + 1) is True

    def test_phrase_needs_one_rare_token(self):
        # "hair system": "hair" is everyday, "system" (stubbed 5.2) is above
        # the rare threshold → no rare token → dropped
        assert tf.passes_commonness("hair system", 1) is False
        # one rare token saves the phrase
        assert tf.passes_commonness("hair dynamics", 1) is True

    def test_unknown_words_treated_as_rare(self):
        # stub returns 0.0 for unknowns → fail-open
        assert tf.passes_commonness("flowmiosis", 1) is True


class TestGateCandidates:
    @staticmethod
    def _cand(term, freq):
        return {"term": term, "freq": freq, "df": freq, "page": 1,
                "score": 1.0, "signal": "llm"}

    def test_everyday_low_freq_dropped(self):
        out = tf.gate_candidates([self._cand("hair", 1),
                                  self._cand("psoriasis", 1)])
        assert [c["term"] for c in out] == ["psoriasis"]

    def test_everyday_jargon_freq_kept(self):
        out = tf.gate_candidates(
            [self._cand("hair", config.EXTRACT_JARGON_MIN_FREQ)])
        assert [c["term"] for c in out] == ["hair"]

    def test_min_freq_filter(self):
        out = tf.gate_candidates(
            [self._cand("psoriasis", 1), self._cand("melanoma", 2)],
            min_freq=2)
        assert [c["term"] for c in out] == ["melanoma"]
