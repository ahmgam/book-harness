"""Tests for the pure parts of the M5 formatting stage (tasks 5.3 + 5.5)."""

from pathlib import Path

from book_harness.format import (
    cosmetic_clean,
    format_book,
    needs_format,
    verify_format,
)
from book_harness.pages import PageArchetype


# ---------------------------------------------------------------------------
# needs_format (5.3)
# ---------------------------------------------------------------------------

class TestCosmeticClean:
    def test_collapses_blank_runs_and_doubles_spaces(self):
        md = "para one.\n\n\n\n\n معادلة  الحركة\n\npara two. \n"
        out = cosmetic_clean(md)
        assert out == "para one.\n\n معادلة الحركة\n\npara two.\n"

    def test_idempotent(self):
        md = "معادلة  الحركة\n\n\n\nنص\n"
        assert cosmetic_clean(cosmetic_clean(md)) == cosmetic_clean(md)


class TestNeedsFormat:
    def test_clean_paragraph_is_skipped(self):
        md = "نص الفقرة الأولى كامل في سطر واحد.\n\nنص الفقرة الثانية كامل أيضا."
        assert needs_format(md) == (False, [])

    def test_newline_run(self):
        md = "first paragraph.\n\n\n\n\nsecond paragraph.\n"
        needs, reasons = needs_format(md)
        assert needs and "newline-run>=3" in reasons

    def test_double_space(self):
        md = "معادلة  الحركة"
        needs, reasons = needs_format(md)
        assert needs and "double-space" in reasons

    def test_glued_image(self):
        md = "بالتالي ![img](images/p001_x.jpg) نحصل على النتيجة."
        needs, reasons = needs_format(md)
        assert needs and "glued-image" in reasons

    def test_lone_image_is_clean(self):
        md = "قبلها فقرة كاملة.\n\n![img](images/p001_x.jpg)\n\nفقرة ثانية."
        assert needs_format(md)[0] is False

    def test_soft_wrapped_paragraphs(self):
        md = ("The flight simulator integrates the complete set of motion\n"
              "equations as described in chapter four and beyond.\n"
              "\n"
              "The simulation model reproduces the same behaviour at a\n"
              "rate of one frame per simulated second.\n")
        needs, reasons = needs_format(md)
        assert needs and any(r.startswith("soft-wrapped") for r in reasons)

    def test_hard_wrapped_inside_code_fence_is_clean(self):
        md = "نص تمهيدي كامل.\n\n```python\nx = 1\ny = 2\n```\n"
        assert needs_format(md)[0] is False

    def test_broken_lost_list(self):
        md = ("الخطوات كما يلي\n2. Trim the controls\n3. Run the test case")
        needs, reasons = needs_format(md)
        assert needs and any(r.startswith("broken-list") for r in reasons)

    def test_entry_style_page_is_not_soft_flagged(self):
        # bibliography: separate entries, no sentence-ending punctuation
        lines = "\n\n".join(
            f"ديفيد أليرتون. مبادئ محاكاة الطيران رقم {i}" if i % 4 == 0
            else f"مؤلف رقم {i}. كتاب تقني عن الطيران رقم {i}"
            for i in range(6))
        # ~6 entry lines, few punctuated -> entry page, no soft flag
        assert needs_format(lines)[0] is False

    def test_proper_list_after_colon_is_clean(self):
        md = ("الخطوات كما يلي:\n\n1. Set the altitude\n2. Trim the controls\n"
              "3. Run the test case")
        assert needs_format(md)[0] is False


# ---------------------------------------------------------------------------
# verify_format (5.5)
# ---------------------------------------------------------------------------

class TestVerifyFormat:
    def test_pure_whitespace_reflow_passes(self):
        original = ("معادلة\n\nالحركة\n\nx = 1\n\n![img](images/a.jpg)\n\n"
                    "[^1]: حاشية.\n")
        formatted = "معادلة الحركة x = 1 ![img](images/a.jpg)\n[^1]: حاشية.\n"
        ok, violations = verify_format(original, formatted)
        assert ok, violations

    def test_rewording_detected_by_letters(self):
        original = "هذه فقرة كاملة تحتوي على معلومات تقنية مفصلة عن النظام."
        formatted = "فقرة كاملة معلومات تقنية النظام."
        ok, violations = verify_format(original, formatted)
        assert not ok and any(v.startswith("letters-drifted") for v in violations)

    def test_lost_list_reconstruction_passes(self):
        # line unwrapping + '- ' bullet markers: letters may grow slightly
        original = "الخطوات\n\nأولا التركيب\nثانيا التشغيل\nثالثا التحقق\n"
        formatted = "- التركيب\n- التشغيل\n- التحقق\n"
        ok, violations = verify_format(original, formatted)
        assert not ok                    # dropped 'الخطوات' + أولا/ثانيا/ثالثا
        assert any(v.startswith("letters-drifted") for v in violations)

    def test_list_marker_additions_pass(self):
        original = "خطوات كثيرة جدا مع نص طويل كفاية\n\nتعليمات أخرى"
        formatted = "1. خطوات كثيرة جدا جدا مع نص طويل كفاية\n\n1. تعليمات أخرى"
        ok, violations = verify_format(original, formatted)
        assert ok, violations            # only markers + duplicate words added

    def test_fenced_code_must_survive_byte_identical(self):
        original = "```python\nx = 1\ny = 2\n```"
        formatted = "نص replaced body"
        ok, violations = verify_format(original, formatted)
        assert not ok and "fenced-code-changed" in violations

    def test_image_links_must_be_identical(self):
        original = "![img](images/a.jpg)"
        formatted = "![مرجع](images/a.jpg)"
        ok, violations = verify_format(original, formatted)
        assert not ok and "image-links-changed" in violations

    def test_footnote_defs_must_survive(self):
        original = "[^1]: حاشية.\n[^2]: أخرى."
        formatted = "[^1]: حاشية."
        ok, violations = verify_format(original, formatted)
        assert not ok and "footnotes-changed" in violations

    def test_display_math_count_must_survive(self):
        original = "$$ x = y $$"
        formatted = "x = y نص"
        ok, violations = verify_format(original, formatted)
        assert not ok and "display-math-changed" in violations

    def test_renumbering_digits_within_tolerance(self):
        original = "1. بند أول كامل\n2. بند ثان كامل\n..."
        formatted = "1. بند أول كامل بين\n2. بند ثان كامل بين\n3. بند ثالث بين"
        ok, violations = verify_format(original, formatted)
        assert ok, violations

    def test_token_count_must_survive(self):
        original = "بالمعادلة §EQ0§ نحصل"
        formatted = "بالمعادلة نحصل على"
        ok, violations = verify_format(original, formatted)
        assert not ok and "tokens-changed" in violations


# ---------------------------------------------------------------------------
# format_book stage (5.4, with a fake client)
# ---------------------------------------------------------------------------

class _FakeClient:
    """Assist-role stand-in: the defective page gets one content-destroying
    attempt and a clean retry, exercising the verifier + fallback path."""

    model = "fake-assist"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.bad_done = False

    def pick_model(self, override=None) -> str:      # pragma: no cover
        return self.model

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        if "IMPORTANT" not in prompt and not self.bad_done:
            # rule violation: drop several words past the letter floor
            self.bad_done = True
            return "wrapped text containing qty intact a short note"
        return ("wrapped text containing qty intact and multiple words here "
                "continuation of the numeric qty content beyond the first "
                "line and still more narrative content rolled on to a "
                "third line")


GOOD_MD = (
    "<!-- ===== page 5 ===== -->\n\n"
    "clean single-line text.\n\n"
    "<!-- ===== page 6 ===== -->\n\n"
    "wrapped text containing qty intact\n"
)


class TestFormatBook:
    def _book_md(self):
        return (
            "<!-- ===== page 5 ===== -->\n\n"
            "clean   single-line   text.\n\n"
            "<!-- ===== page 6 ===== -->\n\n"
            "wrapped text containing qty intact and multiple words here\n"
            "continuation of the numeric qty content beyond the first line\n"
            "and still more narrative content rolled on to a third line\n"
        )

    def _run(self, tmp_path: Path, client):
        from book_harness.pages import classify_markdown

        book_md = self._book_md()
        summary, report = format_book(
            book_md,
            classify_markdown(book_md, num_pages=6),
            client=client,
            out_path=tmp_path / "out" / "book.md",
            format_dir=tmp_path / "fmt",
            report_path=tmp_path / "qa" / "format-report.jsonl",
        )
        return summary, report

    def test_stage_flow_with_fallback_and_retry(self, tmp_path):
        client = _FakeClient()
        summary, report = self._run(tmp_path, client)
        out = (tmp_path / "out" / "book.md").read_text(encoding="utf-8")
        # page 6 needed the model; its first attempt was rejected and the
        # strict retry succeeded -> the unwrapped page lands in the output
        assert "wrapped text containing qty intact" in out
        assert "текст" not in out
        # page 5 has no reason-triggered edits, so only page 6 model use
        assert len(client.calls) >= 2          # 1 bad attempt + 1 retry
        assert summary["fallbacks"] == 0       # retry succeeded
        # report: one record per page incl. reasons
        recs = [ln for ln in (tmp_path / "qa" / "format-report.jsonl")
                .read_text(encoding="utf-8").splitlines() if ln]
        assert len(recs) == 2

    def test_cache_reuses_previous_formatting(self, tmp_path):
        client = _FakeClient()
        summary, _ = self._run(tmp_path, client)
        fresh = _FakeClient()
        summary2, _ = self._run(tmp_path, fresh)
        assert summary2["cache_hits"] >= 1
        assert len(fresh.calls) <= len(client.calls)

    def test_in_place_result_is_idempotent(self, tmp_path):
        client = _FakeClient()
        summary, _ = self._run(tmp_path, client)
        once = (tmp_path / "out" / "book.md").read_text(encoding="utf-8")
        tmp2 = tmp_path / "second"
        tmp2.mkdir()
        (tmp2 / "book.md_src").write_text(once, encoding="utf-8")
        # run again on the formatted markdown with a fresh tmp dirs
        from book_harness.pages import classify_markdown

        summary2, report2 = format_book(
            once,
            classify_markdown(once, num_pages=6),
            client=_FakeClient(),
            out_path=tmp2 / "book.md",
            format_dir=tmp2 / "fmt",
            report_path=tmp2 / "qa" / "format-report.jsonl",
        )
        twice = (tmp2 / "book.md").read_text(encoding="utf-8")
        assert twice == once        # second run changes nothing
