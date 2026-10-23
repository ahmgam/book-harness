"""Occurrence capture tests (context-informed translation selection)."""

from book_harness.occurrences import attach_occurrences, term_pattern

_CHUNK = dict(chunk_id="chunk-0007", first_page=12, pages=[12, 13],
              section="1 Design", text=None, source_hash="h")


def _chunks(*texts):
    out = []
    for i, text in enumerate(texts):
        ch = dict(_CHUNK)
        ch["text"] = text
        ch["section"] = f"1.{i} Section {i}"
        ch["first_page"] = 10 + i
        ch["chunk_id"] = f"chunk-{i:04d}"
        out.append(ch)
    return out


def _one(term):
    return [{"term": term, "freq": 1, "df": 1, "page": 1,
             "score": 1.0, "signal": "llm"}]


class TestTermPattern:
    def test_case_insensitive(self):
        assert term_pattern("Flight Dynamics").search("the flight dynamics model")

    def test_whitespace_and_hyphen_tolerant(self):
        assert term_pattern("flight dynamics").search("flight  dynamics")
        assert term_pattern("angle-of-attack").search("angle-of-attack sensor")
        assert term_pattern("angle-of-attack").search("angle of attack sensor")
        assert term_pattern("flight dynamics").search("flight-dynamics model")

    def test_no_false_substring_match(self):
        assert not term_pattern("speed").search("airspeed indicator")
        assert not term_pattern("cell").search("cells divide")  # plural boundary
        assert term_pattern("cell").search("fuel-cell stack")   # boundary ok
        assert term_pattern("cell").search("(cell)")            # punctuation ok

    def test_nonword_term_gives_none(self):
        assert term_pattern("!!!") is None


class TestAttachOccurrences:
    def test_page_section_context_captured(self):
        texts = ["The flight simulator integrates aircraft motion models "
                 "with stochastic wind fields."]
        recs = attach_occurrences(_one("flight simulator"), _chunks(*texts))
        occ = recs[0]["occurrences"]
        assert len(occ) == 1
        assert occ[0]["chunk_id"] == "chunk-0000"
        assert occ[0]["page"] == 10
        assert occ[0]["section"] == "1.0 Section 0"
        assert "flight simulator integrates" in occ[0]["context"]

    def test_context_window_bounded_and_ellipsized(self):
        text = ("pad " * 200) + "hyperloop guidance" + (" pad " * 200)
        recs = attach_occurrences(_one("hyperloop guidance"), _chunks(text),
                                  context_chars=30)
        context = recs[0]["occurrences"][0]["context"]
        assert context.startswith("…") and context.endswith("…")
        assert len(context) <= 30 + len("hyperloop guidance") + 30 + 4

    def test_multiple_occurrences_in_reading_order(self):
        texts = ["twist thought here", "middle has twist thought too",
                 "no match here"]
        recs = attach_occurrences(_one("twist thought"), _chunks(*texts))
        pages = [o["page"] for o in recs[0]["occurrences"]]
        assert pages == [10, 11]

    def test_cap_per_term(self):
        text = "twist thought " * 20
        recs = attach_occurrences(_one("twist thought"), _chunks(text),
                                  max_per_term=3)
        assert len(recs[0]["occurrences"]) == 3

    def test_no_match_yields_empty_list(self):
        recs = attach_occurrences(_one("quantum fizzle"), _chunks("nothing"))
        assert recs[0]["occurrences"] == []

    def test_record_schema_untouched(self):
        recs = attach_occurrences(_one("twist thought"), _chunks("twist thought"))
        rec = recs[0]
        assert rec["freq"] == 1 and rec["signal"] == "llm"
        assert set(rec) == {
            "term", "freq", "df", "page", "score", "signal", "occurrences",
        }
