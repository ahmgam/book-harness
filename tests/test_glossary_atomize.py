"""Glossary atomization tests: compound dictionary rows → atomic renderings.

Seeded from the real arabterm rows that leaked option lists into
translations (the "all term options in place" defect) before the
consistency-stage rework.
"""

from book_harness.glossary import GlossaryBuilder, split_renderings


# Real rows that previously produced compound term_ar values.
COMPOUND_TREATMENT = "مُعامَلَة، مُعالَجَة (كِيماويَّة)؛ مُداوَاة، مُعالَجة، عِلاج"
COMPOUND_HAIR = "شَعْر، شَعَرَة؛ وَبَر؛ فُسْحَة شَعْريِّة؛ شَعْريّ"
COMPOUND_NAILS = "مِسمار؛ ظُفر؛ سَمَّرَ، ثَبَّتَ بمِسمار، دَسَرَ"
COMPOUND_SCALP = "قَشَرَ، كَشَطَ؛ فَروةُ أو جِلدَةُ الرأس"

# Khatib "treatment" row (verified against arabterm.db).
KHATIB_TREATMENT = "مُعامَلَة، مُعالَجَة (كِيماويَّة)؛ مُداوَاة، مُعالَجة، عِلاج"


class TestSplitRenderings:
    def test_treatment_row(self):
        assert split_renderings(COMPOUND_TREATMENT) == [
            "مُعامَلَة",
            "مُعالَجَة (كِيماويَّة)",
            "مُداوَاة",
            "مُعالَجة",
            "عِلاج",
        ]

    def test_parenthetical_qualifier_stays_attached(self):
        parts = split_renderings("مُعَامَلَة (لِلْمِيَاه)")
        assert parts == ["مُعَامَلَة (لِلْمِيَاه)"]

    def test_hair_row(self):
        assert split_renderings(COMPOUND_HAIR) == [
            "شَعْر", "شَعَرَة", "وَبَر", "فُسْحَة شَعْريِّة", "شَعْريّ",
        ]

    def test_verb_gloss_parts_dropped(self):
        # nails: the trailing clause is a verb usage note — dropped whole
        assert split_renderings(COMPOUND_NAILS) == ["مِسمار", "ظُفر"]

    def test_scalp_row_keeps_noun_clause(self):
        # verb clause dropped, noun clause survives
        assert split_renderings(COMPOUND_SCALP) == ["فَروةُ أو جِلدَةُ الرأس"]

    def test_verb_only_row_is_empty(self):
        assert split_renderings("قَشَرَ، كَشَطَ") == []
        assert split_renderings("سَمَّرَ، دَسَرَ") == []

    def test_atomic_row_unchanged(self):
        assert split_renderings("معالجة، علاج") == ["معالجة", "علاج"]
        assert split_renderings("معالجة") == ["معالجة"]

    def test_dedup_and_order(self):
        assert split_renderings("عدوى؛ عدوى، إصابة") == ["عدوى", "إصابة"]

    def test_empty_and_blank(self):
        assert split_renderings("") == []
        assert split_renderings(None) == []      # type: ignore[arg-type]

    def test_overlong_part_dropped(self, monkeypatch):
        from book_harness import config
        monkeypatch.setattr(config, "GLOSSARY_MAX_TERM_CHARS", 8)
        assert split_renderings("علاج، كلمةٌ كبيرةٌ جدًّا في حدٍّ") == ["علاج"]


class _FakeArabterm:
    """Duck-typed lookup: one dictionary, hits as given."""

    def __init__(self, hits):
        self._hits = hits

    def lookup(self, term):
        return self._hits


def _hit(ar, source="khatib_scientific_technical", source_tech=None):
    return {
        "ar": ar, "en": "treatment", "fr": None,
        "source": source,
        "source_tech": source_tech or source,
        "wikidata": None, "page": None, "scan": None,
        "description": None, "uri": None,
    }


class TestResolveAtomized:
    def test_entry_gets_atomic_term_and_alternatives(self):
        b = GlossaryBuilder(_FakeArabterm([_hit(KHATIB_TREATMENT)]), domains=("*",))
        e = b.resolve("treatment", {"term": "treatment"})
        assert e["term_ar"] == "مُعامَلَة"
        assert e["alternatives"] == [
            "مُعالَجَة (كِيماويَّة)", "مُداوَاة", "مُعالَجة", "عِلاج",
        ]
        assert "atomized" in e["reason"]

    def test_verb_only_hits_leave_entity_uncertified(self):
        b = GlossaryBuilder(_FakeArabterm([_hit("قَشَرَ، كَشَطَ")]), domains=("*",))
        e = b.resolve("scalp", {"term": "scalp"})
        assert e["status"] == "uncertified"
        assert "verb glosses" in e["reason"]

    def test_atomic_parts_deduped_across_dictionaries(self):
        h1 = _hit("معالجة", source="at_chemistry")
        h2 = _hit("علاج", source="at_hygienics_human_body")
        b = GlossaryBuilder(_FakeArabterm([h1, h2]), domains=("*",))
        e = b.resolve("treatment", {"term": "treatment"})
        assert e["term_ar"] == "معالجة"
        assert e["alternatives"] == ["علاج"]

    def test_parts_inherit_source_citation(self):
        # dictionary priority sorts damascus (420) before at_* (400)
        h1 = _hit("معالجة، معاملة", source="at_chemistry")
        h2 = _hit("علاج", source="damascus_chemistry")
        b = GlossaryBuilder(_FakeArabterm([h1, h2]), domains=("*",))
        e = b.resolve("treatment", {"term": "treatment"})
        assert e["term_ar"] == "علاج"
        assert e["source"] == "damascus_chemistry"
        assert e["alternative_sources"] == ["at_chemistry", "at_chemistry"]
