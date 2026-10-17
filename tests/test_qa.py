"""Tests for QA: QE scoring, compliance, consistency, OCR gate (M2 2.1/2.4/2.5/2.6)."""

import json
from pathlib import Path

from book_harness.book import Block, Book, BlockType, PageInfo
from book_harness.qa import (
    CometScorer,
    arabic_present,
    compliance_audit,
    consistency_audit,
    enforced_entries,
    gate_pages,
    junk_ratio,
    judge_prompt,
    parse_scores,
    qe_score_chunk,
    requeue_targets,
)
from book_harness.translate import translate_chunk


class StubJudge:
    def __init__(self, reply='{"scores": {"u0": 88, "u1": 92}}'):
        self.model = "stub-judge"
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, options=None):
        self.prompts.append(prompt)
        return self.reply


GLOSSARY = [
    {"term_en": "drag coefficient", "term_ar": "معامل السحب", "status": "certified"},
    {"term_en": "lift", "term_ar": "الرفع", "status": "certified",
     "alternatives": ["الإجاءة"]},
    {"term_en": "tie break", "term_ar": "منظومة", "status": "llm_chosen"},
    {"term_en": "simulator", "term_ar": None, "status": "uncertified"},
]


def _chunk(units=None):
    return {
        "chunk_id": "chunk-0001",
        "section_title": "1.1 Design",
        "units": units or [
            {"uid": "u0", "block_id": "/page/1/Text/0", "kind": "text",
             "text": "The lift increases with angle."},
            {"uid": "u1", "block_id": "/page/1/Text/1", "kind": "text",
             "text": "Equation §EQ-00§ relates the lift to airspeed."},
        ],
    }


def _draft(units):
    return {"chunk_id": "chunk-0001", "units": units}


class TestQeScoring:
    def test_judge_scores_parsed(self):
        chunk = _chunk()
        draft = _draft([
            {"uid": "u0", "text": "الرفع يزداد مع الزاوية."},
            {"uid": "u1", "text": "`EQ` يربط الرفع بسرعة الهواء."},
        ])
        judge = StubJudge()
        rec = qe_score_chunk(chunk, draft, judge, qe_prompt_version="qe-m2-v1")
        assert rec["score"] == 90.0
        assert rec["unit_scores"] == {"u0": 88.0, "u1": 92.0}
        assert rec["method"] == "judge"
        assert rec["cache_key"]
        # prompt got both pairs, and is judging without a reference
        p = judge.prompts[0]
        assert "The lift increases with angle." in p
        assert "الرفع يزداد مع الزاوية." in p

    def test_chatting_judge_falls_back(self):
        chunk = _chunk()
        draft = _draft([{"uid": "u0", "text": "x"}])
        judge = StubJudge(reply="The translation quality scored 73 of 100 overall.")
        rec = qe_score_chunk(chunk, draft, judge, qe_prompt_version="v")
        assert rec["score"] in (73.0,) or rec["unit_scores"]

    def test_unusable_judge_yields_none(self):
        chunk = _chunk()
        draft = _draft([{"uid": "u0", "text": "x"}])
        judge = StubJudge(reply="no numbers here")
        rec = qe_score_chunk(chunk, draft, judge, qe_prompt_version="v")
        assert rec["score"] is None

    def test_missing_draft_short_circuits(self):
        rec = qe_score_chunk(_chunk(), None, StubJudge(), qe_prompt_version="v")
        assert rec["score"] is None and rec["n_units"] == 0


class TestParseScores:
    def test_json_block(self):
        out = parse_scores('Sure! {"scores": {"u0": 89, "u1": 45}} hope that helps', expected=2)
        assert out == {0: 89.0, 1: 45.0}

    def test_regex_fallback(self):
        out = parse_scores("u0: 77 of 100, u1 -> 30", expected=2)
        assert out == {0: 77.0, 1: 30.0}

    def test_out_of_range_clamped(self):
        out = parse_scores('{"scores": {"u0": 250, "u1": -3}}', expected=2)
        assert out[0] == 100.0 and out[1] == 0.0

    def test_single_integer_bonus(self):
        out = parse_scores("I think the quality is 66 / 100", expected=2)
        assert out == {0: 66.0, 1: 66.0}


class TestArabicPresent:
    def test_plain_match(self):
        assert arabic_present("يمر معامل السحب عبر الحساب", "معامل السحب")

    def test_clitic_prefixes_tolerated(self):
        # والـ / بالـ / للـ prefixed glossary forms still count as compliance
        assert arabic_present("تزداد قوة الرفع مع الزاوية", "الرفع")
        assert arabic_present("بحساب والرفع", "الرفع")
        assert arabic_present("للرفع دور مهم", "الرفع")

    def test_absence_detected(self):
        assert not arabic_present("ترتفع قوة الرفع الفرعية", "معامل السحب")
        assert not arabic_present("السحب فقط", "معامل السحب")

    def test_diacritics_ignored(self):
        assert arabic_present("قوة الرفع ~", "الرفَع")


class TestComplianceAudit:
    def test_hits_and_violations(self):
        chunk = _chunk([
            {"uid": "u0", "kind": "text",
             "text": "The lift increases with drag coefficient."},
            {"uid": "u1", "kind": "text", "text": "lift is here too."},
        ])
        draft = _draft([
            {"uid": "u0", "text": "يزداد الرفع بحسب معامل السحب."},
            {"uid": "u1", "text": "لكن ترجمة لا تستعمل المصطلح المعتمد هنا."},
        ])
        rec = compliance_audit(chunk, draft, GLOSSARY, glossary_version="gv1")
        # u0: lift + drag coefficient enforced; u1: lift enforced
        assert rec["checked"] == 3
        violation = {(v["uid"], v["term_en"]) for v in rec["violations"]}
        assert ("u1", "lift") in violation
        assert ("u0", "lift") not in violation      # والـ clitic tolerated
        assert ("u0", "drag coefficient") not in violation
        assert rec["rate"] == round(100 * 2 / 3, 1)
        assert rec["glossary_version"] == "gv1"

    def test_full_violation_requeues(self):
        rec = compliance_audit(
            _chunk(), _draft([{"uid": "u0", "text": "ترجمة بدون أي مصطلح."}]),
            GLOSSARY,
        )
        assert rec["rate"] == 0.0
        assert len(requeue_targets([rec], floor=95.0)) == 1

    def test_uncertified_terms_never_enforced(self):
        chunk = _chunk([{"uid": "u0", "kind": "text", "text": "simulator here"}])
        draft = _draft([{"uid": "u0", "text": "محاكي"}])
        rec = compliance_audit(chunk, draft, GLOSSARY)
        assert rec["checked"] == 0 and rec["rate"] is None

    def test_missing_draft(self):
        rec = compliance_audit(_chunk(), None, GLOSSARY)
        assert rec["checked"] == 0


class TestConsistencyAudit:
    def test_expected_and_drift(self):
        entries = [
            {"term_en": "lift", "term_ar": "الرفع", "status": "certified",
             "alternatives": ["الإجاءة"]},
        ]
        c1 = _chunk([{"uid": "u0", "kind": "text", "text": "lift on page one"}])
        d1 = _draft([{"uid": "u0", "text": "قوة الرفع"}])
        c2 = _chunk([{"uid": "u0", "kind": "text",
                      "text": "lift on page two"}])
        d2 = _draft([{"uid": "u0", "text": "كلمة الإجاءة هنا"}])
        records, summary = consistency_audit(
            [(c1, d1), (c2, d2)], entries
        )
        (rec,) = records
        assert rec["occurrences"] == 2
        assert rec["expected_chunks"] == 1
        assert rec["drift_chunks"] == ["chunk-0001"]
        # chunk-0001 dedupes: both chunks share the id, so one drift entry
        assert summary["drift_terms"] == 1

    def test_no_term_occurrences(self):
        entries = [{"term_en": "yaw", "term_ar": "الانحراف", "status": "certified"}]
        chunk = _chunk([{"uid": "u0", "kind": "text", "text": "nothing relevant"}])
        records, summary = consistency_audit([(chunk, None)], entries)
        # no draft -> skipped, record exists with zero occurrences
        assert records[0]["occurrences"] == 0
        assert summary["drift_terms"] == 0


def _page_info(page: int) -> PageInfo:
    return PageInfo(index=page, width=600, height=800, bbox=(0, 0, 600, 800), polygon=[])


def _block(page: int, idx: int, btype: str, html: str) -> Block:
    return Block(
        id=f"/page/{page}/{btype}/{idx}", type=BlockType(btype), html=html,
        page=page, bbox=(0, 0, 0, 0), polygon=[], section_hierarchy={}, images={},
    )


class TestOcrGate:
    def _book(self, blocks, page_stats=None):
        pages = {b.page: _page_info(b.page) for b in blocks}
        book = Book(blocks=blocks, pages=pages, toc=[], page_stats=page_stats or [])
        return book

    def test_junk_ratio(self):
        assert junk_ratio("Clean Arabic sentence مع نص.") == 0.0
        assert junk_ratio("Æ□◊Ù§¤") > 0.9
        assert junk_ratio("") == 0.0

    def test_flags_junk_and_empty_headers(self):
        blocks = [
            _block(1, 0, "Text", "<p>Sensible Arabic text، with numbers 12.</p>"),
            _block(2, 0, "Text", "<p>◊◊◊ ÙÃÃ€ §§§</p>"),
            _block(2, 1, "SectionHeader", "<h2></h2>"),
        ]
        book = self._book(blocks)
        records = gate_pages(book)
        p1 = next(r for r in records if r["page"] == 1)
        p2 = next(r for r in records if r["page"] == 2)
        assert not p1["suspect"]
        assert p2["suspect"]
        assert any(f.startswith("junk@") for f in p2["flags"])
        assert any(f.startswith("empty-header@") for f in p2["flags"])

    def test_fragmented_short_blocks_flagged(self):
        blocks = [
            _block(3, 0, "Text", "<p>short</p>"),
            _block(3, 1, "Text", "<p>tiny</p>"),
            _block(3, 2, "Text", "<p>bits</p>"),
        ]
        book = self._book(blocks)
        (rec,) = gate_pages(book)
        assert "fragmented" in rec["flags"]

    def test_page_stats_method_recorded(self):
        class Stat:
            page = 7
            text_extraction_method = "ocr"

        book = self._book([_block(7, 0, "Text", "<p>fine text here</p>")], [Stat()])
        (rec,) = gate_pages(book)
        assert rec["method"] == "ocr" and not rec["suspect"]

    def test_chrome_blocks_ignored(self):
        blocks = [
            _block(4, 0, "Text", "<p>main text here</p>"),
            _block(4, 1, "PageHeader", "<p>◊◊◊</p>"),
        ]
        (rec,) = gate_pages(self._book(blocks))
        assert not rec["suspect"]


class TestEnforcedEntries:
    def test_statuses(self):
        enforced = enforced_entries(GLOSSARY)
        assert [e["term_en"] for e in enforced] == [
            "drag coefficient", "lift", "tie break"
        ]


class TestCometScorerFallback:
    def test_unavailable_is_soft(self):
        scorer = CometScorer(model_name="definitely/missing-model")
        assert scorer.available is False
        assert scorer.score([("src", "mt")]) is None
