"""Shared Arabic text normalization.

Used by glossary enforcement and (later) the compliance audit, so every
component agrees on one canonical form of an Arabic token.

Policy (light by default, stable for matching):
* remove diacritics/tashkeel/additional marks and tatweel;
* unify alef variants (أ إ آ ٱ ...) to bare alef;
* map alif maqsura (ى) -> ya (ي);
* map ta marbuta (ة) -> ha (ه);
* optional aggressive mode additionally folds hamza carriers
  (ؤ -> و, ئ -> ي, standalone hamza ء -> strip).
"""

from __future__ import annotations

import re
import unicodedata

# Harakat + non-spacing marks: fathatan..sukun (064B-0652), 0653-065F,
# superscript alef (0670), small high ligatures; tatweel (0640).
_MARKS_RE = re.compile("[\u0640\u064b-\u0655\u0670\u06d6-\u06ed]")

_ALEF_VARIANTS = "\u0623\u0625\u0622\u0671"        # أ إ آ ٱ -> ا
_TASLIA = "\u0649"                                  # ى -> ي
_TA_MARBUTA = "\u0629"                              # ة -> ه
_HAMZA_ON_WAW = "\u0624"                            # ؤ -> و
_HAMZA_ON_YA = "\u0626"                             # ئ -> ي
_STANDALONE_HAMZA = "\u0621"                        # ء -> stripped (aggressive)

# Arabic-Indic + extended digits -> ASCII (matching helps QA comparisons)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def strip_diacritics(text: str) -> str:
    """Remove tashkeel, Quranic marks, small ligatures, and tatweel."""
    return _MARKS_RE.sub("", text)


def normalize_ar(text: str, *, aggressive: bool = False) -> str:
    """Normalize Arabic text to the light canonical form (or aggressive)."""
    if not text:
        return text
    nit = unicodedata.normalize("NFKC", text)
    nit = strip_diacritics(nit)
    nit = nit.translate(_ARABIC_DIGITS)
    out = []
    for ch in nit:
        if ch in _ALEF_VARIANTS:
            out.append("\u0627")
        elif ch == _TASLIA:
            out.append("\u064a")
        elif ch == _TA_MARBUTA:
            out.append("\u0647")
        elif aggressive and ch == _HAMZA_ON_WAW:
            out.append("\u0648")
        elif aggressive and ch == _HAMZA_ON_YA:
            out.append("\u064a")
        elif aggressive and ch == _STANDALONE_HAMZA:
            continue
        else:
            out.append(ch)
    # collapse whitespace caused by removed marks
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_term(term: str, *, aggressive: bool = True) -> str:
    """Normalization for glossary-term matching (aggressive by default)."""
    return normalize_ar(term, aggressive=aggressive)
