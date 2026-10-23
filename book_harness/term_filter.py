"""Deterministic technical-term focus filters (PLAN §9).

Two layers keep everyday words out of the term candidates:

* ``filter_chunk_terms()`` — cheap, per-chunk word-shape/stopword rules run
  right after the model reply is parsed: stopwords, 1-2 char tokens, pure
  digits/years, standalone Roman numerals, and phrases whose every token is a
  stopword never become candidates.
* ``gate_candidates()`` — a commonness gate at candidate aggregation, backed
  by ``wordfreq`` Zipf scores. Everyday words (zipf ≥ ``EXTRACT_ZIPF_FILTER``)
  survive only when reported by ≥ ``EXTRACT_JARGON_MIN_FREQ`` chunks —
  pervasive, book-wide use is exactly the "domain jargon" signal
  (``"hair"`` in a dermatology book). ``wordfreq`` is imported lazily and any
  failure degrades to "unknown → rare" so a broken dependency cannot silently
  shrink the candidate list.

Both layers are pure functions over plain term strings / candidate records;
the LLM verification pass on top of them lives in ``llm_terms``.
"""

from __future__ import annotations

import re
from typing import Any

from book_harness import config

#: English stopword / function-word list (NLTK-style, vendored so the only
#: new dependency remains ``wordfreq``). Single-word candidates hitting this
#: are dropped before aggregation.
STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "within", "would",
    "you", "your", "yours", "yourself", "yourselves",
    # frequent connective adverbs/discourse words that model replies pick up
    "also", "however", "therefore", "thus", "hence", "moreover",
    "furthermore", "nevertheless", "e.g", "i.e", "etc", "via", "using",
    "used", "based", "given", "respectively", "following", "various",
})

#: Standalone Roman numeral (strict form) — matched case-sensitively on an
#: all-upper or all-lower candidate, e.g. ``VII`` / ``xiv``. Words like
#: "mix" (legal Roman reading MIX = 1009) are only dropped when written
#: as the numeral usually appears: all-uppercase or all-lowercase letters.
_ROMAN_RE = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)

#: Pure numbers: integers, decimals with ``.`` or ``,``, four-digit years.
_NUMBER_RE = re.compile(r"^\d{1,4}([.,]\d+)?$")


def _is_roman(term: str) -> bool:
    return (term.isalpha() and (term.isupper() or term.islower())
            and bool(_ROMAN_RE.match(term.upper())))


def shape_reject(term: str) -> str | None:
    """Rejection reason for one cleaned term under the word-shape rules.

    ``None`` means keep. Applied after ``parse_term_lines`` so the shape of
    every reply line is known-good (the word-length cap is enforced upstream
    in ``llm_terms.extract_chunk_terms``).
    """
    words = term.split()
    if not words:
        return "empty"
    for w in words:
        if _NUMBER_RE.match(w):
            return "number"
        if _is_roman(w):
            return "roman-numeral"
    if len(words) == 1:
        w = words[0]
        if term.lower() in STOPWORDS:
            return "stopword"
        if len(w) <= 2:
            return "too-short"
    elif all(w.lower() in STOPWORDS for w in words):
        return "all-stopwords"
    return None


def filter_chunk_terms(terms: list[str]) -> list[str]:
    """Per-chunk cheap pass: stopword + word-shape rules (format is fine)."""
    return [t for t in terms if shape_reject(t) is None]


# ---------------------------------------------------------------------------
# Commonness (zipf) gate
# ---------------------------------------------------------------------------

_ZIPF_CACHE: dict[str, float] = {}
_ZIPF_FN: Any = None


def _zipf_fn() -> Any:
    """Lazily import ``wordfreq.zipf_frequency``; a stub returning 0 on
    import/other failure (unknown → rare → fail-open)."""
    global _ZIPF_FN
    if _ZIPF_FN is None:
        try:
            from wordfreq import zipf_frequency
            _ZIPF_FN = zipf_frequency
        except Exception:
            _ZIPF_FN = lambda *_a, **_k: 0.0  # noqa: E731
    return _ZIPF_FN


def term_zipf(word: str) -> float:
    """Zipf score (0-8) of one word; cached, unknown words score 0."""
    key = word.strip().lower()
    if not key:
        return 0.0
    if key not in _ZIPF_CACHE:
        try:
            value = float(_zipf_fn()(key, "en"))
        except Exception:
            value = 0.0
        _ZIPF_CACHE[key] = value
    return _ZIPF_CACHE[key]


def passes_commonness(term: str, freq: int) -> bool:
    """Commonness gate for one candidate with its chunk frequency.

    * single word: keep when not everyday English (zipf <
      ``EXTRACT_ZIPF_FILTER``); everyday words need the domain-jargon escape
      hatch — reported by ≥ ``EXTRACT_JARGON_MIN_FREQ`` chunks.
    * phrase (2-3 words): keep when at least one token is rare (zipf <
      ``EXTRACT_ZIPF_RARE``; unknown tokens score 0 → rare).
    """
    words = term.split()
    if len(words) == 1:
        if term_zipf(term) < config.EXTRACT_ZIPF_FILTER:
            return True
        return freq >= config.EXTRACT_JARGON_MIN_FREQ
    return any(term_zipf(w) < config.EXTRACT_ZIPF_RARE for w in words)


def gate_candidates(
    candidates: list[dict[str, Any]],
    *,
    min_freq: int | None = None,
) -> list[dict[str, Any]]:
    """Frequency-aware aggregation gate over candidate records.

    Applies ``EXTRACT_MIN_FREQ`` and the zipf commonness gate
    (``passes_commonness``). Order and records are preserved so the caller
    can still rank/cap afterwards.
    """
    if min_freq is None:
        min_freq = config.EXTRACT_MIN_FREQ
    return [
        c for c in candidates
        if c["freq"] >= min_freq and passes_commonness(c["term"], c["freq"])
    ]
