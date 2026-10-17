"""Glossary drift loop (task 3.6 / PLAN.md catch #7).

After the first translation pass the drafts reveal terms the phase-1
extraction missed. The loop:

1. **Mine** recurring English n-grams across the chunk *source* texts that
   the current glossary does not cover (no glossary term whose token
   sequence subsumes them);
2. **Propose** them — ``02_glossary/drift-candidates.jsonl`` with frequency,
   chunk spread, first-use page — as ``uncertified`` terms awaiting human
   review (the M4 review screen reads the same file);
3. **Confirm** (human): accepted candidates are appended to ``glossary.yaml``
   with ``status: uncertified`` (never silently mixed into the certified
   set), and chunks containing the confirmed terms are re-queued — *only*
   those chunks, via the ``emphasis`` cache-key hook that the M2 compliance
   re-queue already uses.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from book_harness.config import STOPWORDS
from book_harness.qa import source_matches
from book_harness.terms import RUBBER_WORDS, _WORD_RE

#: Placeholder tokens inside unit source text (§EQ-00§, §URL-03§, …) are
#: protected non-translatable spans — never term candidates.
_TOKEN_RE = re.compile(r"§[A-Za-z]+-\d+§")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Mining
# ---------------------------------------------------------------------------

def _unit_docs(
    chunks: list[Any],
) -> list[tuple[str, int, str]]:                 # (chunk_id, page, text)
    """(chunk id, canonical page, source text) per translatable unit."""
    out: list[tuple[str, int, str]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            chunk = chunk.to_dict()
        by_uid: dict[str, dict[str, Any]] = {}
        for ref in chunk.get("blocks", []):
            for uid in ref.get("units", []):
                by_uid[uid] = ref
        pages = chunk.get("pages") or [0]
        for unit in chunk.get("units", []):
            if unit.get("kind") not in ("text", "title", "caption", "list_item"):
                continue
            ref = by_uid.get(unit.get("uid", ""))
            page = (ref or {}).get("page") or pages[0]
            text = unit.get("text", "")
            if text.strip():
                out.append((chunk.get("chunk_id", ""), int(page), text))
    return out


def _term_tokens(term_en: str) -> tuple[str, ...]:
    """Word tokens of one glossary term, whitespace/case-normalised."""
    norm = unicodedata.normalize("NFKC", re.sub(r"[_\-/]+", " ", term_en.lower()))
    return tuple(_WORD_RE.findall(norm))


def _meaningful(gram: tuple[str, ...]) -> bool:
    """Gram quality: every word is alphabetic, single letters excluded, and
    there is at least one content word (len > 2, not a stopword). This keeps
    "angle of attack" but rejects "e g", "x y", numbers and pure fillers."""
    if not all(w.isalpha() and len(w) > 1 for w in gram):
        return False
    if not any(len(w) > 2 and w not in STOPWORDS for w in gram):
        return False
    if all(w in STOPWORDS for w in gram):
        return False
    return True


def _covered(gram: tuple[str, ...], known: list[tuple[str, ...]]) -> bool:
    """True when the gram is already subsumed by some glossary term's words,
    or subsumes one itself (``lift equation`` when ``lift`` is glossed)."""
    n = len(gram)
    for tokens in known:
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == gram:
                return True
        if len(tokens) < n and _contained(tokens, gram):
            return True
    return False


def _contained(tokens: tuple[str, ...], gram: tuple[str, ...]) -> bool:
    for i in range(len(gram) - len(tokens) + 1):
        if tuple(gram[i:i + len(tokens)]) == tokens:
            return True
    return False


def mine_drift_terms(
    chunks: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    *,
    min_freq: int = 3,
    min_df: int = 2,
    min_n: int = 2,
    max_n: int = 3,
    max_candidates: int = 60,
) -> list[dict[str, Any]]:
    """Recurring uncovered n-grams across chunk sources -> drift candidates.

    Output records ``{term, freq, df, chunks, page, score}`` (most term-like
    first); deterministic and pure.
    """
    docs = _unit_docs(chunks)
    if not docs:
        return []
    known = [t for t in (_term_tokens(e.get("term_en") or "") for e in entries) if t]

    freq: Counter[tuple[str, ...]] = Counter()
    df: Counter[tuple[str, ...]] = Counter()
    first_page: dict[tuple[str, ...], int] = {}
    gram_chunks: dict[tuple[str, ...], set[str]] = {}
    canonical: dict[tuple[str, ...], str] = {}
    for chunk_id, page, text in docs:
        words = _WORD_RE.findall(_TOKEN_RE.sub(" ", text))
        lowered = [w.lower() for w in words]
        grams: set[tuple[str, ...]] = set()
        for n in range(min_n, max_n + 1):
            for i in range(len(lowered) - n + 1):
                gram = tuple(lowered[i:i + n])
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                if any(w in RUBBER_WORDS for w in gram):
                    continue
                if not _meaningful(gram):
                    continue
                if _covered(gram, known):
                    continue
                freq[gram] += 1
                grams.add(gram)
                gram_chunks.setdefault(gram, set()).add(chunk_id)
                if gram not in first_page or page < first_page[gram]:
                    first_page[gram] = page
                if gram not in canonical:
                    canonical[gram] = " ".join(words[i:i + n])
        for gram in grams:
            df[gram] += 1

    candidates: list[dict[str, Any]] = []
    for gram, f in freq.items():
        d = df[gram]
        if f < min_freq or d < min_df:
            continue
        length_bonus = 1.0 + 0.35 * (len(gram) - 1)
        candidates.append({
            "term": canonical.get(gram, " ".join(gram)),
            "freq": f,
            "df": d,
            "chunks": sorted(gram_chunks.get(gram, set())),
            "page": first_page.get(gram),
            "score": round(f * math.log(1 + len(docs) / max(1, d)) * length_bonus, 3),
        })
    candidates.sort(key=lambda c: (-c["score"], c["term"]))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in candidates:
        key = unicodedata.normalize("NFKC", c["term"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
        if len(deduped) >= max_candidates:
            break
    return deduped


# ---------------------------------------------------------------------------
# 2. Propose / I/O
# ---------------------------------------------------------------------------

def write_candidates(
    candidates: list[dict[str, Any]],
    path: str | Path,
    *,
    glossary_version: str = "",
) -> int:
    """Write ``drift-candidates.jsonl`` (one proposed term per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            rec = {
                **c,
                "status": "proposed",          # uncertified until human-confirmed
                "decision": None,              # filled by human review
                "glossary_version": glossary_version,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(candidates)


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``drift-candidates.jsonl`` file; [] when missing/corrupt."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# 3. Confirm (human decision)
# ---------------------------------------------------------------------------

def parse_confirm_args(pairs: list[str]) -> list[dict[str, str]]:
    """``["lift=الرفع", "wearout"]`` -> ``[{en, ar-or-''}]`` (drift --confirm)."""
    out = []
    for raw in pairs:
        en, _, ar = raw.partition("=")
        en = en.strip()
        if en:
            out.append({"en": en, "ar": ar.strip() or ""})
    return out


def affected_chunks(chunks: list[dict[str, Any]], term_en: str) -> list[str]:
    """Chunk ids whose source text contains the confirmed term."""
    return [
        c["chunk_id"] for c in chunks
        if any(source_matches(term_en, u.get("text", ""))
               for u in c.get("units", []))
    ]


def append_confirmed_entries(
    entries: list[dict[str, Any]],
    confirms: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Append human-confirmed drift terms as ``uncertified`` glossary entries.

    Returns ``(entries, appended)``; already-present terms are skipped.
    Never fabricates: an entry without a human-supplied Arabic keeps
    ``term_ar: None`` and stays out of enforcement.
    """
    from book_harness.glossary import GlossaryEntry

    existing = {
        re.sub(r"[_\-/]+", " ", (e.get("term_en") or "").lower()).strip()
        for e in entries
    }
    appended: list[dict[str, Any]] = []
    for cf in confirms:
        key = re.sub(r"[_\-/]+", " ", cf["en"].lower()).strip()
        if key in existing:
            logger.info("drift confirm: '%s' already in glossary, skipped", cf["en"])
            continue
        entry = GlossaryEntry.uncertified(
            cf["en"], "glossary drift: recurring term mined from the drafts"
        )
        if cf["ar"]:
            entry["term_ar"] = cf["ar"]
        entry["decided_by"] = "human"
        entries.append(entry)
        appended.append(entry)
        existing.add(key)
    return entries, appended
