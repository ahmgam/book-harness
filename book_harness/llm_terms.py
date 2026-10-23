"""LLM-based term extraction v2 (PLAN §8).

The whole extraction pass is model-driven now: the book is chunked with the
existing chunker (``chunker.plan_chunks``, in memory — nothing is written to
``03_chunks``) and every chunk is handed to a small instruct model, which lists
the glossary terms it sees. The model output is parsed with the same
one-term-per-line contract as the book-glossary harvest
(``glossary_pages.parse_term_lines``).

Assure-fail contract: the extract model is required. A missing client or a
failed ``generate()`` call raises ``RuntimeError`` and stops the ``terms``
stage. An *empty* reply on a content chunk is accepted (and logged) — a
math/table-only chunk can legitimately name no terms.

Technical-term focus (PLAN §9): replies pass the deterministic
``term_filter`` rules (stopwords, word shape) and a frequency-aware zipf
commonness gate at aggregation; a final LLM verification pass
(``verify_candidates``) drops field-inappropriate candidates *gracefully*
(no assist model → skipped, never fails the stage).

Output records reuse the candidate schema (``terms.py``):
``{term, freq, df, page, score, signal}`` with ``signal = "llm"``.
``freq``/``df`` count the chunks which reported the term; ``page`` is the first
reporting chunk's minimum page.

Replies are cached in ``02_glossary/extract-cache.jsonl`` keyed by the chunk's
source hash + model + prompt id, so re-running the stage (or the pipeline's
stage-resume) never pays for the same chunk twice.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from book_harness import config
from book_harness.book import Book
from book_harness.chunker import CHAR_CEILING, MIN_CHUNK_CHARS, plan_chunks
from book_harness.glossary_pages import _dedupe_key, parse_term_lines
from book_harness.progress import work_bar
from book_harness.term_filter import filter_chunk_terms, gate_candidates

# Reasoning-style small models may emit a thinking block; strip every
# such block before line parsing. Tags are assembled from pieces to keep
# them out of doc/literal defaults.
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"


# ---------------------------------------------------------------------------
# Chunk texts
# ---------------------------------------------------------------------------

def chunk_texts(
    book: Book,
    *,
    char_ceiling: int = CHAR_CEILING,
    char_floor: int = MIN_CHUNK_CHARS,
) -> list[dict[str, Any]]:
    """Chunk the book in memory and describe each chunk for extraction.

    Returns ``{chunk_id, first_page, pages, section, text, source_hash}`` per
    chunk, in reading order. ``text`` is the chunk's unit text
    (placeholder-protected, newline-joined); ``source_hash`` identifies the
    chunk content for the extraction cache; ``chunk_id`` is the same id
    ``03_chunks/chunk-NNNN.json`` uses, so occurrence records can reference
    real translation chunks.
    """
    out: list[dict[str, Any]] = []
    for c in plan_chunks(book, char_ceiling=char_ceiling, char_floor=char_floor):
        text = "\n".join(u.text for u in c.units).strip()
        if not text:
            continue
        out.append({
            "chunk_id": c.chunk_id,
            "first_page": min(c.pages) if c.pages else None,
            "pages": list(c.pages),
            "section": c.section_path_string(),
            "text": text,
            "source_hash": c.source_hash(),
        })
    return out


# ---------------------------------------------------------------------------
# Reply handling
# ---------------------------------------------------------------------------

def strip_think(reply: str) -> str:
    """Remove reasoning *think* blocks small models may emit."""
    out = reply
    while _THINK_OPEN in out and _THINK_CLOSE in out:
        start = out.index(_THINK_OPEN)
        end = out.index(_THINK_CLOSE) + len(_THINK_CLOSE)
        out = out[:start] + out[end:]
    return out.strip()


def extract_chunk_terms(
    client: Any,
    section: str,
    text: str,
    *,
    logger=None,
) -> list[str]:
    """One extract-model call for one chunk; returns clean, deduped terms.

    Term-length policy: at most ``config.EXTRACT_MAX_TERM_WORDS`` words —
    enforced twice (prompt rule + post-parse filter), so multiword phrases
    never reach the candidates.
    """
    prompt = config.EXTRACT_USER.format(
        section=section, text=text,
        max_words=config.EXTRACT_MAX_TERM_WORDS,
    )
    try:
        reply = client.generate(
            prompt,
            system=config.EXTRACT_SYSTEM,
            options=dict(config.EXTRACT_GENERATE_OPTIONS),
        )
    except Exception as exc:
        raise RuntimeError(
            f"term extraction: model call failed for chunk section "
            f"{section!r}: {exc}"
        ) from exc
    parsed = parse_term_lines(strip_think(reply))
    # Hard filter: terms, not phrases — at most EXTRACT_MAX_TERM_WORDS words.
    terms = [
        t for t in parsed
        if 0 < len(t.split()) <= config.EXTRACT_MAX_TERM_WORDS
    ]
    # Technical-term focus filter (PLAN §9): stopwords + word shape.
    clean = filter_chunk_terms(terms)
    dropped = len(terms) - len(clean)
    if dropped and logger:
        logger.info("chunk %r: %d non-technical lines dropped", section, dropped)
    if logger:
        logger.info("chunk %r: %d terms", section, len(clean))
    return clean


# ---------------------------------------------------------------------------
# Candidate aggregation
# ---------------------------------------------------------------------------

def build_candidates(
    per_chunk: list[tuple[int | None, list[str]]],
    *,
    max_candidates: int = config.EXTRACT_MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Candidate records from ``[(first_page, terms), ...]`` per chunk.

    ``freq``/``df`` = number of chunks that reported the term (deduped inside
    a chunk by ``parse_term_lines``); ``page`` = the first reporting chunk's
    minimum page; ``score`` grows with chunk frequency
    (``EXTRACT_SCORE_BASE + EXTRACT_SCORE_STEP * (freq - 1)``).
    The PLAN §9 commonness gate applies before the cap: everyday single words
    survive only via the ``EXTRACT_JARGON_MIN_FREQ`` escape hatch.
    """
    chunks: dict[str, set[int]] = {}   # dedupe key -> chunk indexes
    display: dict[str, str] = {}
    first_page: dict[str, int | None] = {}
    for idx, (page, terms) in enumerate(per_chunk):
        for term in terms:
            key = _dedupe_key(term)
            if key not in chunks:
                chunks[key] = set()
                display[key] = term
                first_page[key] = page
            chunks[key].add(idx)
    scored = []
    for key, idx_set in chunks.items():
        freq = len(idx_set)
        score = config.EXTRACT_SCORE_BASE + config.EXTRACT_SCORE_STEP * (freq - 1)
        scored.append({
            "term": display[key],
            "freq": freq,
            "df": freq,
            "page": first_page[key],
            "score": round(score, 3),
            "signal": config.EXTRACT_SIGNAL,
        })
    # Commonness gate (PLAN §9): everyday single words are kept only via the
    # chunk-frequency jargon escape hatch; phrases need one rare token.
    scored = gate_candidates(scored)
    scored.sort(key=lambda c: (-c["freq"], c["term"]))
    return scored[:max_candidates]


# ---------------------------------------------------------------------------
# Reply cache
# ---------------------------------------------------------------------------

def load_replies(cache_path: str | Path, *, model: str) -> dict[str, list[str]]:
    """Load cached replies keyed by chunk source hash for this model+prompt.

    Records whose model or prompt id differ from the current run are ignored
    (a model or prompt change invalidates them).
    """
    path = Path(cache_path)
    if not path.exists():
        return {}
    replies: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("model") != model or rec.get("prompt") != config.EXTRACT_PROMPT_ID:
                continue
            replies[rec["source_hash"]] = list(rec["terms"])
    return replies


def append_reply(cache_path: str | Path, rec: dict[str, Any]) -> None:
    """Append one extraction record to the cache (crash-safe: one line)."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_candidates(
    book: Book,
    extract_client: Any,
    *,
    char_ceiling: int = CHAR_CEILING,
    char_floor: int = MIN_CHUNK_CHARS,
    max_candidates: int = config.EXTRACT_MAX_CANDIDATES,
    cache_path: str | Path | None = None,
    logger=None,
) -> list[dict[str, Any]]:
    """Extract candidate terms for the whole book via the extract model.

    One ``generate()`` call per chunk (cache-hit chunks are skipped). Fails
    loudly when the client is missing or a call errors; empty replies on
    content chunks are accepted with a log line.
    """
    if extract_client is None:
        raise RuntimeError(
            "no extract client is available; the terms stage requires the "
            "extract model (BOOK_HARNESS_EXTRACT_MODELS) for term extraction"
        )
    chunks = chunk_texts(book, char_ceiling=char_ceiling, char_floor=char_floor)
    if not chunks:
        return []
    model = getattr(extract_client, "model", "")
    cached = load_replies(cache_path, model=model) if cache_path else {}
    per_chunk: list[tuple[int | None, list[str]]] = []
    empties = 0
    served = 0
    with work_bar(
        chunks, desc="term extraction", unit="chunk", total=len(chunks)
    ) as pbar:
        for chunk in pbar:
            if chunk["source_hash"] in cached:
                terms = cached[chunk["source_hash"]]
                served += 1
            else:
                terms = extract_chunk_terms(
                    extract_client, chunk["section"], chunk["text"], logger=logger
                )
                if cache_path:
                    append_reply(cache_path, {
                        "model": model,
                        "prompt": config.EXTRACT_PROMPT_ID,
                        "source_hash": chunk["source_hash"],
                        "section": chunk["section"],
                        "pages": chunk["pages"],
                        "terms": terms,
                    })
            if not terms:
                empties += 1
            per_chunk.append((chunk["first_page"], terms))
        pbar.set_description_str(
            f"term extraction ({served} cached, {empties} empty)"
        )
    candidates = build_candidates(per_chunk, max_candidates=max_candidates)
    if logger:
        logger.info(
            "llm extraction: %d chunks (%d served from cache, %d empty) -> "
            "%d distinct terms",
            len(chunks), served, empties, len(candidates),
        )
    return candidates


# ---------------------------------------------------------------------------
# LLM verification pass (PLAN §9, task B)
# ---------------------------------------------------------------------------

def parse_verdicts(reply: str) -> dict[str, bool]:
    """Parse a verification reply into ``{dedupe_key: kept?}``.

    Contract: one word per line, ``word: keep`` / ``word: drop``. Unanswered
    words are simply absent from the mapping (the caller keeps them — the
    deterministic filters already had their say).
    """
    verdicts: dict[str, bool] = {}
    for raw in strip_think(reply).splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        head, tail = line.rsplit(":", 1)
        head = re.sub(r"^\s*\(?\d+[.)]?\)?\s*", "", head).strip(" .;")
        tail = tail.strip().lower().rstrip(" .;")
        if not head:
            continue
        if tail.startswith("keep"):
            verdicts[_dedupe_key(head)] = True
        elif tail.startswith("drop"):
            verdicts[_dedupe_key(head)] = False
    return verdicts


def _batch_key(terms: list[dict[str, Any]]) -> str:
    """Stable cache key for one verification batch (terms, in order)."""
    return hashlib.md5(
        json.dumps([_dedupe_key(c["term"]) for c in terms],
                   ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()


def load_verdicts(
    cache_path: str | Path, *, model: str
) -> dict[str, dict[str, bool]]:
    """Load cached verdict batches for this model (prompt id respected)."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    batches: dict[str, dict[str, bool]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("model") != model or rec.get("prompt") != config.VERIFY_PROMPT_ID:
                continue
            batches[rec["batch"]] = {
                k: bool(v) for k, v in rec.get("verdicts", {}).items()
            }
    return batches


def append_verdicts(cache_path: str | Path, rec: dict[str, Any]) -> None:
    """Append one verification batch record (crash-safe: one line)."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _verify_words_block(batch: list[dict[str, Any]]) -> str:
    """Selectable rendering of one verification batch for ``VERIFY_USER``."""
    lines = []
    for idx, cand in enumerate(batch, 1):
        sections: list[str] = []
        for occ in cand.get("occurrences") or []:
            section = occ.get("section")
            if section and section not in sections:
                sections.append(section)
        context = (
            "sections: " + ", ".join(sections[:6])
            if sections
            else (f"page {cand['page']}" if cand.get("page") is not None else "unknown context")
        )
        lines.append(
            f"{idx}. {cand['term']} — {cand['freq']} chunks | {context}"
        )
    return "\n".join(lines)


def verify_candidates(
    candidates: list[dict[str, Any]],
    assist_client: Any,
    *,
    cache_path: str | Path | None = None,
    logger=None,
) -> list[dict[str, Any]]:
    """LLM expertise filter over the top candidates (field-agnostic layer).

    Batches of ``config.VERIFY_BATCH_SIZE`` candidates are sent to the assist
    model, which answers ``word: keep|drop`` per line; dropped candidates are
    removed from the list. The pass is *graceful by contract* (PLAN §9): with
    no assist client, a failed call, or a missing cache file it returns the
    candidates unchanged with a warning — the deterministic filters already
    ran; this layer only upgrades accuracy. Replies are cached in
    ``verify-cache.jsonl`` keyed by ``(batch terms hash, model, prompt id)``.
    """
    if not candidates:
        return []
    if assist_client is None:
        if logger:
            logger.warning(
                "no assist model available; LLM term verification skipped"
            )
        return candidates
    model = getattr(assist_client, "model", "")
    cached = load_verdicts(cache_path, model=model) if cache_path else {}
    batches = [
        candidates[i:i + config.VERIFY_BATCH_SIZE]
        for i in range(0, len(candidates), config.VERIFY_BATCH_SIZE)
    ]
    kept: list[dict[str, Any]] = []
    served = 0
    for batch in batches:
        bkey = _batch_key(batch)
        verdicts = cached.get(bkey)
        if verdicts is None:
            prompt = config.VERIFY_USER.format(words=_verify_words_block(batch))
            try:
                reply = assist_client.generate(
                    prompt,
                    system=config.VERIFY_SYSTEM,
                    options=dict(config.VERIFY_GENERATE_OPTIONS),
                )
            except Exception as exc:
                if logger:
                    logger.warning(
                        "term verification call failed (%s); batch kept unchanged",
                        exc,
                    )
                kept.extend(batch)
                continue
            verdicts = parse_verdicts(reply)
            if cache_path:
                append_verdicts(cache_path, {
                    "model": model,
                    "prompt": config.VERIFY_PROMPT_ID,
                    "batch": bkey,
                    "terms": [c["term"] for c in batch],
                    "verdicts": verdicts,
                })
        else:
            served += 1
        dropped = 0
        for cand in batch:
            verdict = verdicts.get(_dedupe_key(cand["term"]))
            if verdict is False:
                dropped += 1
                continue
            kept.append(cand)
        if dropped and logger:
            logger.info("verify: %d non-technical candidates dropped", dropped)
    if logger:
        logger.info(
            "llm verification: %d batches (%d served from cache) -> "
            "%d of %d candidates kept",
            len(batches), served, len(kept), len(candidates),
        )
    return kept


def write_replies(replies: list[dict[str, Any]], path: str | Path) -> int:
    """Rewrite a full extraction cache file (used by tests/tools)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in replies:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(replies)
