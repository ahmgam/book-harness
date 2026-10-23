"""Glossary builder v1 (task 1.6).

Looks term candidates up in arabterm (filtered by the book's domain →
dictionary whitelist), resolves conflicts with a *deterministic* dictionary
priority, and only consults the LLM when two sources tie — recording the
reason either way. Every entry is a citation: source dictionary, Wikidata ID,
printed page, archive.org scan URL, plus ``status`` and ``decided_by``.

Dictionary rows frequently hold several synonym renderings in one
``arabic`` field (joined by ``؛`` / ``،``); ``split_renderings`` atomizes them
so each option the consistency stage (`consistency.py`) chooses from is a
single clean rendering.

Output: ``glossary.yaml`` with a ``version`` that feeds the chunk cache keys.
Uncertified fallback (PLAN.md catch #5): uncovered terms are written with
``status: uncertified`` and no fabrication — the suggested rendering is to
keep the English term in parentheses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from book_harness import config

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][\w'\-/]*")

STATUS_CERTIFIED = "certified"
STATUS_LLM_CHOSEN = "llm_chosen"
STATUS_UNCERTIFIED = "uncertified"


def dictionary_sort_key(dictionary: str) -> int:
    for pattern, value in config.DICTIONARY_PRIORITY:
        if pattern.lower() in (dictionary or "").lower():
            return value
    return config.FALLBACK_DICTIONARY_PRIORITY


def dictionary_allowed(dictionary: str, domains: Iterable[str]) -> bool:
    """True when a dictionary passes the domain whitelist (``*`` = ungated)."""
    allowed: list[str] | None = None
    for d in domains:
        entries = config.DOMAIN_DICTIONARIES.get(d)
        if entries is None:
            continue
        allowed = (allowed or []) + entries
    if allowed is None:
        return True
    hay = (dictionary or "").lower()
    return any(pat.lower() in hay for pat in allowed)


# ---------------------------------------------------------------------------
# Compound renderings -> atomic options
# ---------------------------------------------------------------------------

#: Arabic comma (U+060C), Arabic semicolon (U+061B) — the separators the
#: dictionaries use to stack synonyms inside one ``arabic`` field.
_CLAUSE_RE = re.compile(r"\s*؛\s*")
_PART_RE = re.compile(r"\s*،\s*")
#: The dictionary's perfective verb glosses («سَمَّرَ», «دَسَرَ») end in a
#: fatha-marked letter; declined nouns end in ة/دamma/other marks instead.
_VERB_TOKEN_RE = re.compile(r"[\u0621-\u064A]\u064E$")

_GLOSS_LIMIT_FALLBACK = 60


def _gloss_limit() -> int:
    return max(1, getattr(config, "GLOSSARY_MAX_TERM_CHARS", None)
               or _GLOSS_LIMIT_FALLBACK)


def _arabic_tokens(part: str) -> list[str]:
    """Letter-plus-diacritic runs; the tokens verb detection looks at."""
    return [
        t for t in re.findall(r"[\u0621-\u06FF][\u0621-\u06FF\u064B-\u0652]*", part)
        if t.strip()
    ]


def _is_verb_gloss(part: str) -> bool:
    """True when a rendering part is a verb paraphrase (usage note, not a
    translation): some word-token ends in the perfective fatha form the
    dictionaries append as "also said" glosses (``قَشَرَ، كَشَطَ``).
    Verb glosses are dropped, per the confirmed single-nominal policy.
    """
    return any(_VERB_TOKEN_RE.search(t) for t in _arabic_tokens(part))


def split_renderings(ar: str) -> list[str]:
    """Atomize one dictionary ``arabic`` string into single renderings.

    Splits on the Arabic semicolon (clause boundary) then the Arabic comma
    (synonym boundary). Guards:

    * a parenthetical qualifier (``(كِيماويَّة)``) stays attached to the
      preceding part — it is a domain note, not an independent rendering;
    * verb-gloss parts are dropped entirely (may empty the row — the caller
      treats an empty result as "row is a usage note, no rendering");
    * parts longer than ``GLOSSARY_MAX_TERM_CHARS`` are dropped;
    * duplicate parts collapse (first occurrence wins, priority order).
    """
    text = (ar or "").strip()
    if not text:
        return []
    parts: list[str] = []
    for clause in _CLAUSE_RE.split(text):
        for piece in _PART_RE.split(clause.strip()):
            piece = piece.strip()
            if not piece:
                continue
            if piece.startswith("(") and piece.endswith(")"):
                if parts:                       # domain note on the previous
                    parts[-1] = f"{parts[-1]} {piece}".strip()
                continue
            if _is_verb_gloss(piece) or len(piece) > _gloss_limit():
                continue
            if piece not in parts:
                parts.append(piece)
    return parts


def _expand_hit(h: dict) -> list[str]:
    """Atomic renderings of one dictionary hit (may be empty — verb row)."""
    return split_renderings(h.get("ar") or "")


class GlossaryEntry(dict):
    """Dict-style entry; keys follow the cited-artifact contract.

    ``alternatives`` holds the non-selected Arabic renderings,
    ``alternative_sources`` the parallel dictionary names (kept for the
    translation-selection prompt; same length, unknown when merged from
    older runs).
    """

    @classmethod
    def from_hits(
        cls,
        term: str,
        hits: list[dict],
        chosen: dict | None,
        *,
        alternatives: list[str],
        alternative_sources: list[str] | None = None,
        status: str,
        decided_by: str,
        reason: str = "",
    ) -> "GlossaryEntry":
        chosen = chosen or {}
        return cls(
            term_en=term,
            term_ar=chosen.get("ar"),
            source=chosen.get("source"),
            wikidata=chosen.get("wikidata"),
            page=chosen.get("page"),
            scan=chosen.get("scan"),
            status=status,
            decided_by=decided_by,
            alternatives=alternatives,
            alternative_sources=alternative_sources or [],
            reason=reason,
        )

    @classmethod
    def uncertified(cls, term: str, reason: str) -> "GlossaryEntry":
        return cls(
            term_en=term,
            term_ar=None,
            source=None,
            wikidata=None,
            page=None,
            scan=None,
            status=STATUS_UNCERTIFIED,
            decided_by="auto",
            alternatives=[],
            alternative_sources=[],
            reason=reason,
        )


class GlossaryBuilder:
    def __init__(
        self,
        arabterm: Any,                       # book_harness.arabterm.Arabterm
        domains: Iterable[str] = config.DEFAULT_DOMAINS,
        assist_model: Any = None,            # optional LLM for tie-breaking
        glossary_limit: int | None = None,
    ) -> None:
        self.arabterm = arabterm
        self.domains = tuple(domains)
        self.assist_model = assist_model
        self.limit = glossary_limit

    # -- public ----------------------------------------------------------
    def build(self, candidates: list[dict]) -> list[GlossaryEntry]:
        entries: list[GlossaryEntry] = []
        for c in candidates[: self.limit or len(candidates)]:
            term = (c.get("term") or "").strip()
            if not term:
                continue
            entries.append(self.resolve(term, c))
        return entries

    def resolve(self, term: str, candidate: dict) -> GlossaryEntry:
        hits = self.lookup(term)
        certified = [
            h for h in hits
            if dictionary_allowed(
                h.get("source_tech") or h.get("source") or "", self.domains
            )
        ]
        seen, unique = set(), []
        for h in sorted(
            certified,
            key=lambda h: -dictionary_sort_key(h.get("source_tech") or h.get("source") or ""),
        ):
            ar = h.get("ar")
            if not ar or ar in seen:
                continue
            seen.add(ar)
            unique.append(h)
        if not unique:
            return GlossaryEntry.uncertified(
                term,
                "no dictionary hit" if not hits
                else "all hits excluded by domain whitelist",
            )
        # Atomize: every dictionary row contributes its atomic renderings
        # (compound rows split, verb-gloss parts dropped), in dictionary
        # priority order — the options the consistency stage chooses from.
        options: list[tuple[str, str, dict]] = []
        seen_parts: set[str] = set()
        split_notes = 0
        for h in unique:
            parts = _expand_hit(h)
            if len(parts) > 1:
                split_notes += 1
            for part in parts:
                if part in seen_parts:
                    continue
                seen_parts.add(part)
                options.append(
                    (part, h.get("source") or h.get("source_tech") or "", h)
                )
        if not options:
            return GlossaryEntry.uncertified(
                term,
                "all dictionary renderings are verb glosses "
                "(usage notes, not terms)",
            )
        best_part, best_source, best_hit = options[0]
        best_key = dictionary_sort_key(
            best_hit.get("source_tech") or best_hit.get("source") or ""
        )
        tied = [h for h in unique
                if dictionary_sort_key(h.get("source_tech") or h.get("source") or "") == best_key]
        alternatives = [part for part, _, _ in options[1:]]
        alternative_sources = [source for _, source, _ in options[1:]]
        chosen = dict(best_hit)
        chosen["ar"] = best_part
        chosen["source"] = best_source
        reason = (
            ("highest dictionary priority; tied candidates exist but "
             "LLM adjudication unavailable" if len(tied) > 1
             else "highest dictionary priority")
        )
        if split_notes:
            reason += (f"; {split_notes} compound dictionary "
                       f"entr{'y' if split_notes == 1 else 'ies'} atomized")
        if len(tied) > 1 and self.assist_model is not None:
            hit, tie_reason, via_llm = self._llm_adjudicate(
                term, best_hit, tied
            )
            pick = _expand_hit(hit)
            chosen = dict(hit)
            chosen["ar"] = pick[0] if pick else hit.get("ar")
            chosen["source"] = hit.get("source") or hit.get("source_tech") or ""
            extra = "; compound dictionary entries atomized" if len(pick) > 1 else ""
            return GlossaryEntry.from_hits(
                term, unique, chosen, alternatives=alternatives,
                alternative_sources=alternative_sources,
                status=STATUS_LLM_CHOSEN if via_llm else STATUS_CERTIFIED,
                decided_by="llm" if via_llm else "auto",
                reason=tie_reason + extra,
            )
        return GlossaryEntry.from_hits(
            term, unique, chosen, alternatives=alternatives,
            alternative_sources=alternative_sources,
            status=STATUS_CERTIFIED, decided_by="auto",
            reason=reason,
        )

    # -- internals ---------------------------------------------------------
    def lookup(self, term: str) -> list[dict]:
        return self.arabterm.lookup(term)



    def _llm_adjudicate(self, term: str, best: dict, tied: list[dict]) -> tuple[dict, str, bool]:
        options = "\n".join(
            f"- `{h['ar']}` ({h.get('source', '?')})" for h in tied
        )
        prompt = (
            f"For the English technical term '{term}', several certified Arabic "
            f"translations are available:\n{options}\n\n"
            "Choose the translation most appropriate for an "
            f"{'/'.join(self.domains)} technical book. "
            "Reply with the chosen Arabic term only."
        )
        try:
            reply = self.assist_model(prompt)
            reply = (reply or "").strip().strip("`\"' ")
            for h in tied:
                if h["ar"] in reply:
                    return h, f"LLM chose among {len(tied)} tied dictionary matches", True
        except Exception as exc:  # LLM unavailable — deterministic fallback
            logger.warning("LLM adjudication failed (%s); deterministic fallback", exc)
        return best, "LLM tie-break failed; kept highest dictionary priority", False


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def glossary_version(entries: list[GlossaryEntry] | str) -> str:
    if not isinstance(entries, str):
        blob = json.dumps(
            [
                [e.get("term_en"), e.get("term_ar"), e.get("source"), e.get("status")]
                for e in entries
            ],
            ensure_ascii=False, sort_keys=True,
        )
    else:
        blob = entries
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def canonical_term(term_en: str) -> str:
    """Case-insensitive lookup key for injection matching."""
    return _WORD_RE.sub(" ", term_en.lower()).strip()


def write_glossary(
    entries: list[GlossaryEntry], out_path: str | Path
) -> tuple[Path, str]:
    out_path = Path(out_path)
    version = glossary_version(entries)
    payload = {
        "version": version,
        "domains": None,
        "entries": [dict(e) for e in entries],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, width=100)
    return out_path, version


def load_glossary(path: str | Path) -> tuple[str, list[GlossaryEntry]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("version", ""), [GlossaryEntry(e) for e in data.get("entries", [])]
