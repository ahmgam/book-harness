"""Tests for state/cache keys and the glossary conflict policy (1.6, 1.10)."""

from pathlib import Path

from book_harness.glossary import (
    STATUS_CERTIFIED,
    STATUS_LLM_CHOSEN,
    STATUS_UNCERTIFIED,
    GlossaryBuilder,
    dictionary_allowed,
    dictionary_sort_key,
    glossary_version,
    load_glossary,
    write_glossary,
)
from book_harness.state import RunState, cache_key


class FakeArabterm:
    """Duck-typed Arabterm returning canned hits per term."""

    def __init__(self, table: dict[str, list[dict]]) -> None:
        self.table = table

    # real Arabterm.lookup(term) signature
    def lookup(self, term: str) -> list[dict]:
        return self.table.get(term.lower(), [])


def _hit(ar, source="KSU glossary", wikidata="Q1", page=7, scan="http://s/7"):
    return {"ar": ar, "source": source, "wikidata": wikidata, "page": page, "scan": scan}


class TestCacheKeys:
    def test_deterministic(self):
        k1 = cache_key("source", "gv1", "llama3.1:8b", "m1-v1")
        k2 = cache_key("source", "gv1", "llama3.1:8b", "m1-v1")
        assert k1 == k2 and len(k1) == 64

    def test_every_input_matters(self):
        kwargs = dict(source="source", glossary_version="gv1",
                      model="llama3.1:8b", prompt_version="m1-v1")
        variants = [
            dict(source="source2"),
            dict(glossary_version="gv2"),
            dict(model="other"),
            dict(prompt_version="m1-v2"),
        ]
        keys = {cache_key(**kwargs)}
        for delta in variants:
            keys.add(cache_key(**{**kwargs, **delta}))
        assert len(keys) == 5

    def test_state_roundtrip(self, tmp_path: Path):
        st = RunState(tmp_path / "runA")
        st.mark("translate", status="done", chunks=7)
        st2 = RunState(tmp_path / "runA")
        assert st2.is_done("translate")
        assert st2.stage_status("source") is None


class TestConflictPolicy:
    def test_priority_ordering(self):
        assert dictionary_sort_key("ksaa_finance_economics") > dictionary_sort_key(
            "khatib_scientific_technical"
        )
        assert dictionary_sort_key("aerospace_engineering") > dictionary_sort_key(
            "dstt_1"
        )
        assert dictionary_sort_key("at_physics") > FALLBACK_PRIORITY_SENTINEL

    def test_domain_whitelist(self):
        assert dictionary_allowed("aerospace_engineering", strict_domains())
        # finance dictionaries are outside the aerospace/software whitelist
        assert not dictionary_allowed("ksaa_finance_economics", strict_domains())


FALLBACK_PRIORITY_SENTINEL = 100


def strict_domains():
    return ("aerospace",)


class TestBuilder:
    def test_single_certified_hit(self):
        at = FakeArabterm({"lift": [_hit("الرفع", source="aerospace_engineering")]})
        e = builder(["lift"], at)[0]
        assert e["status"] == STATUS_CERTIFIED
        assert e["term_ar"] == "الرفع"
        assert e["decided_by"] == "auto"
        assert e["wikidata"] == "Q1"

    def test_priority_picks_best_dictionary(self):
        at = FakeArabterm({
            "cell": [
                _hit("خلية", source="at_physics"),           # unified: 400
                _hit("خانة", source="damascus_physics"),     # Damascus: 420
            ],
        })
        e = builder(["cell"], at)[0]
        assert e["term_ar"] == "خانة"
        assert e["status"] == STATUS_CERTIFIED
        assert e["alternatives"] == ["خلية"]

    def test_domain_whitelist_excludes(self):
        at = FakeArabterm({
            "cell": [_hit("خلية", source="at_physics"), _hit("خانة", source="at_physics")],
            "interest": [_hit("فائدة", source="ksaa_finance_economics")]  # excluded
        })
        e_interest = builder(["cell", "interest"], at)[1]
        assert e_interest["status"] == STATUS_UNCERTIFIED
        assert e_interest["reason"].startswith("all hits excluded")

    def test_no_hit_is_uncertified(self):
        at = FakeArabterm({})
        e = builder(["quantum fizzle"], at)[0]
        assert e["status"] == STATUS_UNCERTIFIED
        assert e["term_ar"] is None

    def test_llm_breaks_ties(self):
        at = FakeArabterm({
            "panel": [
                _hit("لوحة", source="damascus_maths"),
                _hit("صفيحة", source="damascus_physics"),
            ],
        })
        assert dictionary_sort_key("damascus_maths") == dictionary_sort_key(
            "damascus_physics"
        )
        e = builder(["panel"], at, assist=lambda p: "صفيحة")[0]
        assert e["status"] == STATUS_LLM_CHOSEN
        assert e["term_ar"] == "صفيحة"
        assert "tied" in e["reason"]

    def test_llm_failure_falls_back_deterministically(self):
        at = FakeArabterm({
            "panel": [
                _hit("لوحة", source="damascus_maths"),
                _hit("صفيحة", source="damascus_physics"),
            ],
        })

        def boom(_p):
            raise RuntimeError("no ollama")

        e = builder(["panel"], at, assist=boom)[0]
        assert e["status"] == STATUS_CERTIFIED
        assert e["term_ar"] == "لوحة"  # first listed among equals


def builder(terms, at, assist=None):
    candidates = [{"term": t, "freq": 3, "df": 2, "page": 12, "score": 5.0, "signal": "ngram"}
                  for t in terms]
    return GlossaryBuilder(at, domains=("aerospace", "software"),
                           assist_model=assist).build(candidates)


class TestSerialization:
    def test_yaml_roundtrip(self, tmp_path: Path):
        at = FakeArabterm({"lift": [_hit("الرفع", source="aerospace_engineering")]})
        entries = builder(["lift"], at)
        path, version = write_glossary(entries, tmp_path / "glossary.yaml")
        v2, loaded = load_glossary(path)
        assert v2 == version
        assert loaded[0]["term_ar"] == "الرفع"
        assert loaded[0]["status"] == STATUS_CERTIFIED

    def test_version_changes_with_content(self):
        e1 = [dict(term_en="lift", term_ar="الرفع", source="KSU", status=STATUS_CERTIFIED)]
        e2 = [dict(term_en="lift", term_ar="الرفع", source="other", status=STATUS_CERTIFIED)]
        assert glossary_version(e1) != glossary_version(e2)
        assert glossary_version(e1) == glossary_version([dict(e1[0])])
