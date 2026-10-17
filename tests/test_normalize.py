"""Tests for the shared Arabic normalizer (task 1.3)."""

from book_harness.normalize import normalize_ar, normalize_term, strip_diacritics


class TestStripDiacritics:
    def test_removes_harakat(self):
        assert strip_diacritics("مُحَمَّد") == "محمد"
        assert strip_diacritics("القَلْبُ") == "القلب"

    def test_removes_tanwin_and_sukun(self):
        assert strip_diacritics("كِتَابًا") == "كتابا"
        assert strip_diacritics("مِنْ") == "من"

    def test_removes_tatweel(self):
        assert strip_diacritics("رــســول") == "رسول"
        assert strip_diacritics("حيَّـران") == "حيران"

    def test_keeps_consonants(self):
        assert strip_diacritics("سلام") == "سلام"


class TestNormalizeAlef:
    def test_unifies_alef_variants(self):
        assert normalize_ar("أَحمد") == normalize_ar("احمد") == "احمد"
        assert normalize_ar("إِسْلام") == "اسلام"
        assert normalize_ar("آمين") == normalize_ar("امين")
        assert normalize_ar("ٱلله") == normalize_ar("الله") == "الله"

    def test_alif_maqsura_to_ya(self):
        assert normalize_ar("على") == normalize_ar("علي") == "علي"
        assert normalize_ar("مصطفى") == "مصطفي"

    def test_ta_marbuta_to_ha(self):
        assert normalize_ar("مدرسة") == "مدرسه"
        # normalization is idempotent
        twice = normalize_ar(normalize_ar("مدرسة"))
        assert twice == "مدرسه"


class TestNormalizeAggressive:
    def test_folded_hamza_carriers(self):
        assert normalize_ar("مرؤوس", aggressive=True) == "مرووس"
        assert normalize_ar("مؤمن", aggressive=True) == normalize_ar("مومن", aggressive=True)
        assert normalize_ar("سماء", aggressive=True) == "سما"
        # ئ folds onto a full ya (so 'رئيس' keeps its ya in both worlds,
        # 'رئيس' and 'ريس' only agree in the aggressive world).
        assert normalize_ar("رئيس", aggressive=True) == "رييس"
        assert normalize_ar("رئيس", aggressive=True) == "رييس"

    def test_light_keeps_hamza_carriers(self):
        assert normalize_ar("مرؤوس") == "مرؤوس"
        assert normalize_ar("رئيس") == "رئيس"


class TestNormalizerBehavior:
    def test_daftar_examples(self):
        # classic pair: same trunk, different spelling forms
        assert normalize_ar("أثاث") == normalize_ar("اثاث")
        assert normalize_ar("سُئل") == normalize_ar("سئل")

    def test_digits_folded(self):
        assert normalize_ar("٢٠٢٤") == "2024"

    def test_term_matching_helper_is_aggressive(self):
        assert normalize_term("مُؤمّن") == normalize_term("مومن")
        assert normalize_term("تأسّس") == normalize_term("تاسس")

    def test_mixed_content_passthrough(self):
        src = "The value of $y(0)$ setting"
        assert normalize_ar("غير عربي 123 ABC", aggressive=True) == "غير عربي 123 ABC"
