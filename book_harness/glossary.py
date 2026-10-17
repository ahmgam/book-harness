"""Glossary builder v1 (task 1.6).

Looks term candidates up in arabterm (filtered by the book's domain →
dictionary whitelist), resolves conflicts with a *deterministic* dictionary
priority, and only consults the LLM when two sources tie — recording the
reason either way. Every entry is a citation: source dictionary, Wikidata ID,
printed page, archive.org scan URL, plus ``status`` and ``decided_by``.

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
from book_harness.terms import GlinerExtractor

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


class GlossaryEntry(dict):
    """Dict-style entry; keys follow the cited-artifact contract."""

    @classmethod
    def from_hits(
        cls,
        term: str,
        hits: list[dict],
        chosen: dict | None,
        *,
        alternatives: list[str],
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
        best = unique[0]
        best_key = dictionary_sort_key(best.get("source_tech") or best.get("source") or "")
        tied = [h for h in unique
                if dictionary_sort_key(h.get("source_tech") or h.get("source") or "") == best_key]
        alternatives = [h["ar"] for h in unique[1:]]
        if len(tied) > 1 and self.assist_model is not None:
            chosen, reason, via_llm = self._llm_adjudicate(term, best, tied)
            return GlossaryEntry.from_hits(
                term, unique, chosen, alternatives=alternatives,
                status=STATUS_LLM_CHOSEN if via_llm else STATUS_CERTIFIED,
                decided_by="llm" if via_llm else "auto", reason=reason,
            )
        return GlossaryEntry.from_hits(
            term, unique, best, alternatives=alternatives,
            status=STATUS_CERTIFIED, decided_by="auto",
            reason=("highest dictionary priority; tied candidates exist but "
                    "LLM adjudication unavailable" if len(tied) > 1
                    else "highest dictionary priority"),
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
