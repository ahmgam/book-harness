"""Thin CLI over the library (task 1.11).

    book-harness run <main-ocr-json> [options]      # full pipeline
    book-harness stages                             # describe the stages
    book-harness serve runs/<book-id>               # M4: review UI (progress + review)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from book_harness import config
from book_harness import glossary as glossary_mod
from book_harness import terms as terms_mod
from book_harness.arabterm import Arabterm, ensure_db
from book_harness.book import Book
from book_harness.chunker import plan_chunks, write_chunks
from book_harness.config import Settings, setup_logging
from book_harness.state import RunState

logger = logging.getLogger("book-harness")

STAGES = [
    "source", "outline-repair", "terms", "glossary", "chunk", "gate",
    "translate", "qe", "refine", "compliance", "consistency", "drift",
    "artifacts", "assemble", "format",
]


# ---------------------------------------------------------------------------
# Stage implementations (each is idempotent / resumable)
# ---------------------------------------------------------------------------

def stage_source(settings: Settings, book: Book) -> Path:
    src_dir = settings.stage_dir(stage="source")
    summary = {
        "book_id": settings.book_id,
        "ocr_file": str(settings.ocr_file),
        "pages": book.num_pages,
        "blocks": len(book.blocks),
        "toc_entries": len(book.toc),
        "outline": book.print_outline(max_level=1),
    }
    (src_dir / "book-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return src_dir


def stage_terms(settings: Settings, book: Book) -> list[dict]:
    out = settings.stage_dir(stage="glossary")
    candidates, gliner_used = terms_mod.extract_candidates(book, logger=logger)
    n = terms_mod.write_candidates(candidates, out / config.CANDIDATES_FILE)
    logger.info("extracted %d term candidates (GLiNER: %s)", n, gliner_used)
    return candidates


def stage_glossary(
    settings: Settings, candidates: list[dict]
) -> tuple[list[dict], str, Path]:
    out = settings.stage_dir(stage="glossary")
    db_path = ensure_db()
    with Arabterm.from_db(db_path) as at:
        builder = glossary_mod.GlossaryBuilder(at, domains=settings.domains)
        entries = builder.build(candidates)
    path, version = glossary_mod.write_glossary(entries, out / config.GLOSSARY_FILE)
    logger.info("glossary: %d entries, version %s -> %s", len(entries), version, path)
    version, entries = glossary_mod.load_glossary(path)
    return entries, version, path


def stage_chunk(settings: Settings, book: Book) -> list:
    out = settings.stage_dir(stage="chunks")
    chunks = plan_chunks(
        book, char_ceiling=settings.char_ceiling, char_floor=settings.char_floor
    )
    paths = write_chunks(chunks, out)
    logger.info("chunked into %d chunks -> %s", len(paths), out)
    return chunks


def stage_translate(settings: Settings, chunks: list) -> int:
    from book_harness.translate import SummaryTracker, build_client, translate_chunk

    if settings.pages:
        lo, hi = settings.pages
        chunks = [
            c for c in chunks
            if lo <= min(c.pages) and max(c.pages) < hi
        ]
        logger.info("page range %d..%d: %d chunks", lo, hi - 1, len(chunks))
    if settings.max_chunks:
        chunks = chunks[: settings.max_chunks]
    drafts_dir = settings.stage_dir(stage="drafts")
    client = build_client(
        settings.ollama_host,
        timeout=settings.ollama_timeout,
        api=settings.api,
        api_key=settings.api_key,
        model=settings.model,
    )
    _, entries, _ = _load_glossary(settings)
    summaries = SummaryTracker(
        client,
        settings.run_dir() / config.STAGE_DIRS["drafts"] / "chapter-summary.json",
        period=settings.summary_period,
    )
    done = 0
    for c in chunks:
        translate_chunk(
            c.to_dict(), client, entries, drafts_dir,
            prompt_version=settings.prompt_version,
            glossary_version=_glossary_version(settings),
            summaries=summaries,
        )
        done += 1
        logger.info("translated %s (%d/%d)", c.chunk_id, done, len(chunks))
    return done


def _load_glossary(settings: Settings) -> tuple[str, list[dict], Path]:
    path = settings.run_dir() / config.STAGE_DIRS["glossary"] / config.GLOSSARY_FILE
    version, entries = glossary_mod.load_glossary(path)
    return version, entries, path


def _glossary_version(settings: Settings) -> str:
    return _load_glossary(settings)[0]


def _drafts_meta(settings: Settings) -> dict[str, dict]:
    from book_harness.assemble import DraftBook

    drafts = DraftBook(settings.stage_dir(stage="drafts"))
    return drafts.meta_by_chunk


def _translation_client(settings: Settings) -> Any:
    """MT-role client: chunk translation / re-translation; honours the
    ``BOOK_HARNESS_TRANSLATION_*`` role envs (see config.TRANSLATION_MODELS)."""
    from book_harness.translate import build_client

    return build_client(
        settings.ollama_host,
        timeout=settings.ollama_timeout,
        preferred=config.TRANSLATION_MODELS,
        api=settings.api,
        api_key=settings.api_key,
        model=settings.model,
    )


def _assist_client(settings: Settings, preferred: list[str] | None = None) -> Any:
    """Instruct-role client (suffix/verify/judging); ``preferred`` narrows a
    sub-role (e.g. ``config.QE_MODELS`` for the QE judge)."""
    from book_harness.translate import build_client

    return build_client(
        settings.ollama_host,
        timeout=settings.ollama_timeout,
        preferred=preferred or config.ASSIST_MODELS,
        api=settings.api,
        api_key=settings.api_key,
        model=settings.model,          # explicit model also steers QA roles
    )


def _format_client(settings: Settings, model_override: str | None = None) -> Any:
    """M5 formatting role: general instruct preference list (``config.
    FORMAT_MODELS``), never the dedicated MT model."""
    from book_harness.translate import build_client

    return build_client(
        settings.ollama_host,
        timeout=settings.ollama_timeout,
        preferred=config.FORMAT_MODELS,
        api=settings.api,
        api_key=settings.api_key,
        model=model_override,
    )


# ---------------------------------------------------------------------------
# M2 stages
# ---------------------------------------------------------------------------

def stage_gate(settings: Settings, book: Book) -> dict:
    """OCR QA gate: flag suspicious pages before model spend (task 2.6)."""
    from book_harness.qa import gate_pages, write_gate

    records = gate_pages(book)
    write_gate(records, settings.stage_dir(stage="qa"))
    flagged = [r for r in records if r["suspect"]]
    summary = {
        "pages": len(records),
        "flagged": len(flagged),
        "suspicious_pages": [r["page"] for r in flagged][:50],
    }
    if flagged:
        logger.warning(
            "OCR QA gate: %d/%d pages flagged (junk/fragmented/empty headers); "
            "see 06_qa/gate.jsonl", len(flagged), len(records),
        )
    else:
        logger.info("OCR QA gate: clean (%d pages)", len(records))
    return summary


def stage_qe(settings: Settings, chunks: list) -> dict[str, dict]:
    """Reference-free QE scoring per chunk (task 2.1) -> qe-scores.jsonl."""
    from book_harness.qa import (
        CometScorer,
        draft_pairs,
        latest_by_chunk,
        qe_cache_key,
        qe_score_chunk,
        read_records,
        write_records,
    )

    drafts_meta = _drafts_meta(settings)
    comet = None
    if settings.qe_method in ("auto", "comet"):
        comet = CometScorer()
        if not comet.available:
            logger.info("cometkiwi not installed; using the LLM judge for QE")
    elif settings.qe_method == "comet":
        comet = CometScorer()
        if not comet.available:
            raise RuntimeError("qe-method=comet but the comet stack is unavailable")
    client = None if comet and comet.available and settings.qe_method == "comet" \
        else _assist_client(settings, preferred=config.QE_MODELS)
    effective_method = (
        settings.qe_method if settings.qe_method != "auto"
        else ("comet" if comet and comet.available else "judge")
    )
    effective_model = (
        comet.model_name if effective_method == "comet" else getattr(client, "model", "")
    )
    qa_dir = settings.stage_dir(stage="qa")
    path = qa_dir / "qe-scores.jsonl"
    scores: dict[str, dict] = latest_by_chunk(read_records(path))
    scored = 0
    for c in chunks:
        cd = c.to_dict()
        draft = drafts_meta.get(c.chunk_id)
        if not draft:
            continue
        # resume: skip when draft+scorer identity is unchanged
        key = qe_cache_key(
            draft_pairs(cd, draft),
            qe_prompt_version=config.QE_PROMPT_VERSION,
            model=str(effective_model or ""),
            method=effective_method,
            chunk_id=c.chunk_id,
        )
        existing = scores.get(c.chunk_id)
        if existing and existing.get("cache_key") == key and existing.get("finished"):
            continue
        record = qe_score_chunk(
            cd, draft, client,
            qe_prompt_version=config.QE_PROMPT_VERSION,
            comet=comet, method=settings.qe_method,
        )
        scores[c.chunk_id] = record
        scored += 1
        logger.info(
            "qe %s: score=%s (%s, %d units)",
            c.chunk_id, record["score"], record["method"], len(record["unit_scores"]),
        )
    write_records(scores.values(), path)
    return scores


def stage_refine(settings: Settings, chunks: list) -> int:
    """QE-budgeted refinement (tasks 2.2 + 2.3)."""
    from book_harness.refine import plan_refinement, refine_chunk

    drafts_meta = _drafts_meta(settings)
    scores = {}
    scores_path = settings.run_dir() / config.STAGE_DIRS["qa"] / "qe-scores.jsonl"
    if scores_path.exists():
        from book_harness.qa import latest_by_chunk, read_records

        scores = latest_by_chunk(read_records(scores_path))
    plan = plan_refinement(
        [c.to_dict() for c in chunks], drafts_meta, scores,
        quality_threshold=settings.quality_threshold,
        budget=settings.refine_budget,
    )
    skipped = len(drafts_meta) - len(plan)
    logger.info(
        "refine: %d chunk(s) planned (worst first; %d above %.0f threshold or undrafted)",
        len(plan), max(0, skipped), settings.quality_threshold,
    )
    client = _assist_client(settings)
    refine_dir = settings.run_dir() / config.STAGE_DIRS["refine"]
    done = 0
    for chunk_dict, score in plan:
        refined = refine_chunk(
            chunk_dict, drafts_meta[chunk_dict["chunk_id"]], client, refine_dir,
            draft_cache_key=drafts_meta[chunk_dict["chunk_id"]].get("cache_key", ""),
            prompt_version=config.REFINE_PROMPT_VERSION,
            max_spans=settings.max_spans,
        )
        done += 1
        counters = refined["counters"]
        logger.info(
            "refined %s (qe=%.0f): %d spans, %d patches accepted, %d rejected (%d/%d)",
            chunk_dict["chunk_id"], score, counters["spans"],
            counters["accepted"], counters["rejected"], done, len(plan),
        )
    return done


def _compliance_records(settings: Settings, chunks: list, entries: list[dict]) -> list[dict]:
    from book_harness.qa import compliance_audit

    records = []
    drafts_meta = _drafts_meta(settings)
    glossary_version = _glossary_version(settings)
    for c in chunks:
        rec = compliance_audit(
            c.to_dict(), drafts_meta.get(c.chunk_id), entries,
            glossary_version=glossary_version,
        )
        records.append(rec)
    return records


def stage_compliance(settings: Settings, chunks: list, entries: list[dict]) -> list[dict]:
    """Glossary enforcement audit -> compliance.jsonl, with re-queue pass."""
    from book_harness.qa import requeue_targets, write_records
    from book_harness.translate import SummaryTracker, translate_chunk

    qa_dir = settings.stage_dir(stage="qa")
    path = qa_dir / "compliance.jsonl"
    records = _compliance_records(settings, chunks, entries)
    write_records(records, path)

    if not settings.requeue:
        _log_compliance(records)
        return records

    requeue = requeue_targets(records, settings.compliance_floor)
    if not requeue:
        _log_compliance(records)
        return records

    # Targeted re-translation (2.4): violated terms become a hard emphasis and
    # enter the cache key, so only violating chunks are regenerated.
    drafts_dir = settings.stage_dir(stage="drafts")
    client = _translation_client(settings)
    summaries = SummaryTracker(
        client,
        drafts_dir / "chapter-summary.json",
        period=settings.summary_period,
    )
    glossary_version = _glossary_version(settings)
    violated_terms = {
        r["chunk_id"]: [
            (v["term_en"], v["term_ar"]) for v in r["violations"]
        ]
        for r in requeue
    }
    for c in chunks:
        if c.chunk_id not in violated_terms:
            continue
        logger.info(
            "re-queuing %s for targeted re-translation (%d violations)",
            c.chunk_id, len(violated_terms[c.chunk_id]),
        )
        translate_chunk(
            c.to_dict(), client, entries, drafts_dir,
            prompt_version=settings.prompt_version,
            glossary_version=glossary_version,
            summaries=summaries,
            emphasis=violated_terms[c.chunk_id],
        )
    after = _compliance_records(settings, [
        c for c in chunks if c.chunk_id in violated_terms
    ], entries)
    requeue_ids = {r["chunk_id"] for r in requeue}
    records = [t for t in records if t["chunk_id"] not in requeue_ids] + after
    write_records(records, path)
    _log_compliance(records)
    return records


def _log_compliance(records: list[dict]) -> None:
    rated = [r for r in records if r.get("rate") is not None]
    if not rated:
        logger.info("compliance: no enforced terms matched any chunk source")
        return
    below = [r for r in rated if r["rate"] < 95.0]
    logger.info(
        "compliance: %d rated chunks, mean rate %.1f%%, below-floor: %d",
        len(rated),
        sum(r["rate"] for r in rated) / len(rated),
        len(below),
    )


def stage_consistency(settings: Settings, chunks: list, entries: list[dict]) -> dict:
    """Book-level terminology drift audit (task 2.5) -> consistency.jsonl."""
    from book_harness.qa import consistency_audit, write_records

    drafts_meta = _drafts_meta(settings)
    chunk_drafts = [
        (c.to_dict(), drafts_meta.get(c.chunk_id)) for c in chunks
    ]
    records, summary = consistency_audit(chunk_drafts, entries)
    write_records(records, settings.stage_dir(stage="qa") / "consistency.jsonl")
    logger.info(
        "consistency: %d/%d enforced terms drift from the certified rendering",
        summary["drift_terms"], summary["terms_checked"],
    )
    for r in records:
        if r["drift"]:
            alts = {form: ids for form, ids in r["alternative_hits"].items() if ids}
            logger.info(
                "drift: '%s' (expected %s) alternatives used: %s",
                r["term_en"], r["expected"], alts,
            )
    return summary


def stage_assemble(settings: Settings, book: Book) -> Path:
    from book_harness.assemble import DraftBook, render_book
    from book_harness.chunker import load_chunks

    chunks = load_chunks(settings.run_dir() / config.STAGE_DIRS["chunks"])
    drafts = DraftBook(settings.stage_dir(stage="drafts"))
    out_dir = settings.stage_dir(stage="output")
    render_book(
        book, chunks, drafts, out_dir,
        book_id=settings.book_id,
        refine_dir=settings.run_dir() / config.STAGE_DIRS["refine"],
    )
    logger.info("assembled -> %s/book.md", out_dir)
    return out_dir / "book.md"


# ---------------------------------------------------------------------------
# M3 stages
# ---------------------------------------------------------------------------

def stage_outline_repair(settings: Settings, book: Book) -> dict:
    """Re-derive heading depth from section-numbering patterns (task 3.7)."""
    from book_harness.outline import repair_outline, repair_summary

    changes = repair_outline(book)
    summary = repair_summary(changes)
    out = settings.stage_dir(stage="source") / "outline-repair.json"
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if changes:
        logger.warning(
            "outline repair: re-levelled %d section header(s) from numbering "
            "patterns (see %s)", len(changes), out,
        )
    else:
        logger.info("outline repair: OCR heading levels are consistent")
    return summary


def stage_drift(settings: Settings, chunks: list) -> dict:
    """Mine uncovered recurring terms from the drafted chunks (task 3.6)."""
    from book_harness import drift as drift_mod

    _, entries, _ = _load_glossary(settings)
    candidates = drift_mod.mine_drift_terms(
        [c.to_dict() for c in chunks], entries
    )
    out = settings.stage_dir(stage="glossary") / "drift-candidates.jsonl"
    n = drift_mod.write_candidates(
        candidates, out, glossary_version=_glossary_version(settings)
    )
    if candidates:
        preview = ", ".join(c["term"] for c in candidates[:10])
        logger.warning(
            "glossary drift: %d recurring uncovered term(s) proposed for "
            "human review -> %s (top: %s)", n, out, preview,
        )
    else:
        logger.info("glossary drift: no uncovered recurring terms")
    return {"candidates": n, "path": str(out)}


def stage_artifacts(settings: Settings, book: Book, chunks: list) -> dict:
    """Glossary appendix + bilingual HTML + QA report + TMX (tasks 3.1-3.5)."""
    from book_harness.artifacts import (
        render_bilingual_html,
        render_glossary_appendix,
        render_qa_report,
        render_translated_html,
        render_tmx,
    )
    from book_harness.assemble import DraftBook
    from book_harness.chunker import load_chunks

    run_dir = settings.run_dir()
    _, entries, _ = _load_glossary(settings)
    chunks = load_chunks(run_dir / config.STAGE_DIRS["chunks"])
    drafts = DraftBook(settings.stage_dir(stage="drafts"))
    refine_dir = run_dir / config.STAGE_DIRS["refine"]
    out_dir = settings.stage_dir(stage="output")

    summary: dict = {}
    if settings.appendix:
        summary["appendix"] = render_glossary_appendix(
            entries, [c.to_dict() for c in chunks], out_dir / "glossary-appendix.md",
            book_id=settings.book_id,
        )
    if settings.bilingual:
        summary["bilingual"] = render_bilingual_html(
            book, [c.to_dict() for c in chunks], drafts, refine_dir,
            out_dir / "bilingual.html",
            title=f"{settings.book_id} — bilingual edition",
        )
    if settings.tmx:
        summary["tmx"] = render_tmx(
            [c.to_dict() for c in chunks], drafts, out_dir / "book.tmx",
            book_id=settings.book_id, book=book,
        )
        if settings.translated_html:
            summary["translated_html"] = render_translated_html(
                out_dir / "book.tmx", out_dir / "translated.html",
                title=f"{settings.book_id} — الطبعة العربية",
            )
    if settings.qa_report:
        summary["qa_report"] = render_qa_report(
            [c.to_dict() for c in chunks], entries, run_dir,
            out_dir / "qa-report.md",
            quality_threshold=settings.quality_threshold,
            compliance_floor=settings.compliance_floor,
        )
    logger.info("artifacts written to %s: %s", out_dir, ", ".join(summary))
    return summary


# ---------------------------------------------------------------------------
# CLI drift-confirm command (3.6): human step of the drift loop
# ---------------------------------------------------------------------------

def drift_confirm(
    settings: Settings,
    confirms: list[dict[str, str]],
) -> dict:
    """Append confirmed drift terms to the glossary and re-translate only the
    chunks that contain them (cache keys change only for those chunks)."""
    from book_harness import drift as drift_mod
    from book_harness.chunker import load_chunks
    from book_harness.translate import SummaryTracker, build_client, translate_chunk

    run_dir = settings.run_dir()
    chunks = load_chunks(run_dir / config.STAGE_DIRS["chunks"])
    version, entries, glossary_path = _load_glossary(settings)

    entries, appended = drift_mod.append_confirmed_entries(entries, confirms)
    if not appended:
        logger.info("drift confirm: nothing new to append")
        return {"appended": 0, "chunks": []}
    glossary_mod.write_glossary(entries, glossary_path)
    new_version = glossary_mod.glossary_version(entries)
    logger.info(
        "drift confirm: %d term(s) appended -> glossary version %s",
        len(appended), new_version,
    )

    if not settings.translate:
        logger.info(
            "drift confirm: re-translation skipped (--no-rerun); re-run the "
            "translate stage to apply the new terms")
        return {"appended": len(appended), "chunks": []}
    if not any(c.get("ar") for c in confirms):
        logger.info(
            "drift confirm: no Arabic forms supplied — nothing to inject into "
            "translation; the terms are catalogued as uncertified")
        return {"appended": len(appended), "chunks": []}

    # affected chunks only: emphasis enters the per-chunk cache key so the
    # other chunks' translated drafts stay fresh (same hook the M2 compliance
    # re-queue uses).
    drafts_dir = settings.stage_dir(stage="drafts")
    client = build_client(
        settings.ollama_host,
        timeout=settings.ollama_timeout,
        api=settings.api,
        api_key=settings.api_key,
        model=settings.model,
    )
    summaries = SummaryTracker(
        client, drafts_dir / "chapter-summary.json",
        period=settings.summary_period,
    )
    enforced = [(c["en"], c["ar"]) for c in confirms if c["ar"]]
    rerun: list[str] = []
    for c in chunks:
        chunk_dict = c.to_dict()
        pairs = [
            (en, ar) for en, ar in enforced
            if drift_mod.affected_chunks([chunk_dict], en)
        ]
        if not pairs:
            continue
        logger.info(
            "drift confirm: re-queuing %s (%d new terms in source)",
            c.chunk_id, len(pairs),
        )
        translate_chunk(
            c.to_dict(), client, entries, drafts_dir,
            prompt_version=settings.prompt_version,
            glossary_version=new_version,
            summaries=summaries,
            emphasis=pairs,
        )
        rerun.append(c.chunk_id)
    state = RunState(run_dir)
    state.mark(
        "drift-confirm", status="done", terms=[e["term_en"] for e in appended],
        rerun=rerun,
    )
    state.save()
    return {"appended": len(appended), "chunks": rerun}



# ---------------------------------------------------------------------------
# M5 stage
# ---------------------------------------------------------------------------

def stage_format(settings: Settings, book: Book) -> dict:
    """AI formatting pass, archetype card per page (task 5.4 / M5)."""
    from book_harness.format import format_book
    from book_harness.pages import classify_book

    book_md_path = settings.stage_dir(stage="output") / "book.md"
    book_md = book_md_path.read_text(encoding="utf-8")
    client = _format_client(settings)
    summary, _report = format_book(
        book_md,
        classify_book(book),
        client=client,
        model=str(getattr(client, "model", "") or ""),
        out_path=book_md_path,
        format_dir=settings.stage_dir(stage="format"),
        report_path=settings.stage_dir(stage="format_qa") / "format-report.jsonl",
        force=settings.force_format,
        pages_limit=settings.format_pages,
    )
    return summary


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def run(settings: Settings) -> int:
    run_dir = settings.run_dir()
    state = RunState(run_dir)
    logger.info("run dir: %s", run_dir)

    book = Book.from_marker_json(settings.ocr_file)
    logger.info(
        "book: %d pages, %d blocks, %d TOC entries",
        book.num_pages, len(book.blocks), len(book.toc),
    )

    src_summary = run_dir / config.STAGE_DIRS["source"] / "book-summary.json"
    if not src_summary.exists():
        state.mark("source", status="started")
        stage_source(settings, book)
        state.mark("source", status="done", pages=book.num_pages,
                   blocks=len(book.blocks))
        logger.info("stage source: done")

    state.mark("outline-repair", status="started")
    outline_summary = stage_outline_repair(settings, book)
    state.mark("outline-repair", status="done",
               repaired=outline_summary["sections_repaired"])

    state.mark("terms", status="started")
    candidates = stage_terms(settings, book)
    state.mark("terms", status="done", candidates=len(candidates))

    state.mark("glossary", status="started")
    entries, version, _path = stage_glossary(settings, candidates)
    state.mark("glossary", status="done", entries=len(entries), version=version)

    state.mark("chunk", status="started")
    chunks = stage_chunk(settings, book)
    state.mark("chunk", status="done", chunks=len(chunks))

    state.mark("gate", status="started")
    gate_summary = stage_gate(settings, book)
    state.mark("gate", status="done", flagged=gate_summary["flagged"])

    if not settings.translate:
        logger.info("translation skipped (--no-translate)")
        state.save()
        return 0

    state.mark("translate", status="started")
    n_translated = stage_translate(settings, chunks)
    state.mark("translate", status="done", chunks=n_translated)

    _, entries, _ = _load_glossary(settings)

    state.mark("qe", status="started")
    if settings.qe_method != "none":
        scores = stage_qe(settings, chunks)
        state.mark("qe", status="done", chunks=len(scores))
    else:
        scores = {}
        state.mark("qe", status="done", skipped=True)

    state.mark("refine", status="started")
    if settings.refine:
        n_refined = stage_refine(settings, chunks)
        state.mark("refine", status="done", chunks=n_refined)
    else:
        state.mark("refine", status="done", skipped=True)

    state.mark("compliance", status="started")
    stage_compliance(settings, chunks, entries)
    state.mark("compliance", status="done")

    state.mark("consistency", status="started")
    stage_consistency(settings, chunks, entries)
    state.mark("consistency", status="done")

    state.mark("drift", status="started")
    if not settings.no_drift:
        drift_summary = stage_drift(settings, chunks)
        state.mark("drift", status="done", candidates=drift_summary["candidates"])
    else:
        state.mark("drift", status="done", skipped=True)

    state.mark("artifacts", status="started")
    if not settings.no_artifacts:
        stage_artifacts(settings, book, chunks)
    state.mark("artifacts", status="done", skipped=settings.no_artifacts)

    state.mark("assemble", status="started")
    stage_assemble(settings, book)
    state.mark("assemble", status="done")
    logger.info("book.md at %s", run_dir / config.STAGE_DIRS["output"])

    state.mark("format", status="started")
    if not settings.no_format:
        fmt_summary = stage_format(settings, book)
        state.mark("format", status="done", **fmt_summary)
        logger.info("formatted book.md: %s", fmt_summary)
    else:
        state.mark("format", status="done", skipped=True)
    state.save()
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_server_flags(p: argparse.ArgumentParser) -> None:
    """Host/api/key flags shared by every command that talks to a model
    server. Default serving backend stays Ollama ('auto' only auto-detects
    OpenAI-compatible servers from a host URL ending in /v1)."""
    p.add_argument("--ollama-host", default=config.OLLAMA_HOST,
                   help="model server base URL (Ollama default; a URL ending "
                        "in /v1 uses the OpenAI-compatible API)")
    p.add_argument("--api", choices=["auto", "ollama", "openai"],
                   default=config.API_DEFAULT,
                   help="wire protocol: auto-detected from the host URL "
                        "(default) or forced")
    p.add_argument("--api-key", default=config.OPENAI_API_KEY,
                   help="API key for OpenAI-compatible endpoints "
                        "(env OPENAI_API_KEY); sent as a Bearer header")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="book-harness",
        description="OCR JSON -> cited glossary -> translation -> markdown",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the full pipeline")
    run_p.add_argument("ocr_json", type=Path, help="main OCR JSON file")
    run_p.add_argument("--run-root", type=Path, default=Path("runs"))
    run_p.add_argument("--model", default=None,
                       help="force a translation model (Ollama name or ID)")
    _add_server_flags(run_p)
    run_p.add_argument("--domains", nargs="+", default=list(config.DEFAULT_DOMAINS))
    run_p.add_argument("--char-ceiling", type=int, default=config.CHAR_CEILING)
    run_p.add_argument("--char-floor", type=int, default=config.MIN_CHUNK_CHARS)
    run_p.add_argument("--glossary-limit", type=int, default=None,
                       help="cap glossary entries (debug)")
    run_p.add_argument("--max-chunks", type=int, default=None,
                       help="translate only the first N chunks (debug)")
    run_p.add_argument("--pages", nargs=2, type=int, metavar=("LO", "HI"),
                       help="smoke run: restrict translation to pages LO..HI")
    run_p.add_argument("--no-translate", action="store_true",
                       help="stop after chunking")
    run_p.add_argument("--qe-method", choices=["auto", "judge", "comet", "none"],
                       default=config.QE_METHOD_DEFAULT,
                       help="QE scorer: cometkiwi if installed, else LLM judge")
    run_p.add_argument("--no-refine", action="store_true",
                       help="skip the estimate/patch/verify refinement loop")
    run_p.add_argument("--no-requeue", action="store_true",
                       help="do not re-translate compliance-violating chunks")
    run_p.add_argument("--quality-threshold", type=float,
                       default=config.QUALITY_THRESHOLD,
                       help="skip refinement for chunks scoring >= this (0-100)")
    run_p.add_argument("--compliance-floor", type=float,
                       default=config.COMPLIANCE_FLOOR,
                       help="re-translate chunks below this compliance %%")
    run_p.add_argument("--refine-budget", type=int, default=config.REFINE_BUDGET,
                       help="max chunks refined per pass (None = all below threshold)")
    run_p.add_argument("--max-spans", type=int, default=config.REFINE_MAX_SPANS,
                       help="max MQM error spans per unit (default 12)")
    run_p.add_argument("--no-drift", action="store_true",
                       help="skip the glossary drift-mining stage (3.6)")
    run_p.add_argument("--no-translated-html", action="store_true",
                       help="skip the page-divided print edition HTML "
                            "(rendered from the TMX)")
    run_p.add_argument("--no-artifacts", action="store_true",
                       help="skip appendix/bilingual/TMX/QA-report generation (3.1-3.5)")
    run_p.add_argument("--no-format", action="store_true",
                       help="skip the M5 AI formatting pass after assembly")
    run_p.add_argument("--force-format", action="store_true",
                       help="send every page to the format model, not only "
                            "pages flagged by needs_format (M5)")
    run_p.add_argument("--log-level", default="INFO")

    sub.add_parser("stages", help="show pipeline stages")

    bake_p = sub.add_parser(
        "bakeoff",
        help="M2: sample-chunk model comparison (run --no-translate first)",
    )
    bake_p.add_argument("ocr_json", type=Path, help="main OCR JSON file")
    bake_p.add_argument("--models", nargs="+",
                        default=["hy-mt2-30b-a3b", "hy-mt2-7b", "translategemma-12b",
                                 "qwen3.5-27b"],
                        help="candidate translation models (Ollama names)")
    bake_p.add_argument("--chunks", type=int, default=30,
                        help="sample size (evenly spread across the book)")
    bake_p.add_argument("--judge", default=None,
                        help="force a QE judge model (default: assist preference)")
    bake_p.add_argument("--run-root", type=Path, default=Path("runs"))
    bake_p.add_argument("--ollama-timeout", type=int, default=config.OLLAMA_TIMEOUT)
    _add_server_flags(bake_p)
    bake_p.add_argument("--max-chunks", type=int, default=None)
    bake_p.add_argument("--pages", nargs=2, type=int, metavar=("LO", "HI"))
    bake_p.add_argument("--domains", nargs="+", default=list(config.DEFAULT_DOMAINS))
    bake_p.add_argument("--log-level", default="INFO")

    drift_p = sub.add_parser(
        "drift",
        help="M3.6: glossary drift — mine recurring uncovered terms and "
             "confirm them (append + re-run affected chunks only)",
    )
    drift_p.add_argument("ocr_json", type=Path, help="main OCR JSON file")
    drift_p.add_argument("--confirm", action="append", default=[],
                         metavar='TERM[=ARABIC]',
                         help="confirm a proposed term, optionally with the "
                              "human-reviewed Arabic (=ARABIC); repeatable")
    drift_p.add_argument("--min-freq", type=int, default=3,
                         help="min occurrences across the book")
    drift_p.add_argument("--min-df", type=int, default=2,
                         help="min distinct chunks the term appears in")
    drift_p.add_argument("--no-rerun", action="store_true",
                         help="append confirmed entries without re-translating")
    drift_p.add_argument("--model", default=None, help="force a translation model")
    drift_p.add_argument("--run-root", type=Path, default=Path("runs"))
    drift_p.add_argument("--domains", nargs="+", default=list(config.DEFAULT_DOMAINS))
    _add_server_flags(drift_p)
    drift_p.add_argument("--log-level", default="INFO")

    serve_p = sub.add_parser(
        "serve",
        help="M4: review UI — live stage/chunk progress (SSE) + human review "
             "of uncertified terms and low-QE chunks",
    )
    serve_p.add_argument("run_dir", type=Path,
                         help="run directory (runs/<book-id>)")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--poll-seconds", type=float, default=1.0,
                         help="SSE poll interval for state/chunk changes")
    serve_p.add_argument("--log-level", default="INFO")

    print_p = sub.add_parser(
        "print",
        help="standalone: page-divided, print-ready HTML of the translated "
             "book, rendered from a TMX (each sheet = one book page)",
    )
    print_p.add_argument("tmx", type=Path,
                         help="TMX file (e.g. 07_output/book.tmx)")
    print_p.add_argument("-o", "--output", type=Path, default=None,
                         help="output HTML (default: translated.html next to "
                              "the TMX)")
    print_p.add_argument("--title", default=None,
                         help="document title (default: the TMX book prop)")
    print_p.add_argument("--log-level", default="INFO")

    fmt_p = sub.add_parser(
        "format",
        help="M5 standalone: AI formatting pass on an already-assembled run "
             "(archetype cards per page; verified layout-only edits)",
    )
    fmt_p.add_argument("run_dir", type=Path,
                       help="run directory containing 07_output/book.md "
                            "(runs/<book-id>)")
    fmt_p.add_argument("--pages", nargs=2, type=int, metavar=("LO", "HI"),
                       default=None,
                       help="restrict formatting to pages LO..HI (inclusive)")
    fmt_p.add_argument("--force", action="store_true",
                       help="send every page to the model, not only pages "
                            "flagged by needs_format")
    fmt_p.add_argument("--model", default=None,
                       help="force a formatting model (assist preference by "
                            "default)")
    fmt_p.add_argument("-o", "--out", type=Path, default=None,
                       help="write the formatted markdown here instead of "
                            "updating 07_output/book.md in place")
    fmt_p.add_argument("--ollama-timeout", type=int, default=config.OLLAMA_TIMEOUT)
    _add_server_flags(fmt_p)
    fmt_p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stages":
        for i, s in enumerate(STAGES, 1):
            print(f"{i}. {s}")
        return 0

    if args.command == "print":                 # TMX in, HTML out — no OCR needed
        from book_harness.artifacts import render_translated_html

        setup_logging(args.log_level)
        tmx: Path = args.tmx.resolve()
        if not tmx.exists():
            logger.error("TMX not found: %s", tmx)
            return 2
        out = (args.output or tmx.parent / "translated.html").resolve()
        summary = render_translated_html(tmx, out, title=args.title)
        logger.info(
            "print edition: %d pages, %d units -> %s",
            summary["pages"], summary["units"], summary["path"],
        )
        return 0

    if args.command == "serve":                 # needs no OCR input
        from book_harness import web

        setup_logging(args.log_level)
        try:
            web.serve(
                args.run_dir, host=args.host, port=args.port,
                poll_seconds=args.poll_seconds,
            )
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 2
        return 0

    if args.command == "format":            # M5 standalone: post-assembly pass
        from book_harness.format import format_book
        from book_harness.pages import classify_book, classify_markdown
        from book_harness.translate import build_client

        setup_logging(args.log_level)
        run_dir = args.run_dir.resolve()
        book_md_path = run_dir / "07_output" / "book.md"
        if not book_md_path.exists():
            logger.error("no assembled book at %s", book_md_path)
            return 2
        book_md = book_md_path.read_text(encoding="utf-8")

        # Page composition: the consolidated Book model when the OCR JSON
        # pair was kept in 00_source; markdown heuristics otherwise.
        book = None
        src = run_dir / "00_source"
        if src.exists():
            cands = [
                p for p in src.glob("*.json")
                if not p.name.endswith("_meta.json")
                and p.name not in ("book-summary.json", "outline-repair.json")
            ]
            if cands:
                book = Book.from_marker_json(cands[0])
                logger.info("composition from %s", cands[0])
        archetypes = classify_book(book) if book is not None \
            else classify_markdown(book_md)
        logger.info(
            "composed %d page archetypes (%s source)", len(archetypes),
            "Book" if book is not None else "markdown",
        )

        client = build_client(
            args.ollama_host,
            timeout=args.ollama_timeout,
            preferred=config.FORMAT_MODELS,
            api=args.api,
            api_key=args.api_key,
            model=args.model,
        )
        out_path = args.out.resolve() if args.out else book_md_path
        summary, _report = format_book(
            book_md,
            archetypes,
            client=client,
            model=str(getattr(client, "model", "") or ""),
            out_path=out_path,
            format_dir=run_dir / "08_format",
            report_path=run_dir / "09_qa" / "format-report.jsonl",
            force=args.force,
            pages_limit=tuple(args.pages) if args.pages else None,
        )
        state = RunState(run_dir)
        state.mark("format", status="done", **summary)
        state.save()
        logger.info("formatted -> %s: %s", out_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    setup_logging(args.log_level)
    ocr_file: Path = args.ocr_json.resolve()
    if not ocr_file.exists():
        logger.error("OCR JSON not found: %s", ocr_file)
        return 2
    if ocr_file.name.endswith("_meta.json"):
        logger.error("pass the MAIN OCR json (not the _meta.json)")
        return 2

    settings = Settings(
        run_root=args.run_root,
        ocr_file=ocr_file,
        model=getattr(args, "model", None),
        ollama_host=args.ollama_host,
        ollama_timeout=getattr(args, "ollama_timeout", config.OLLAMA_TIMEOUT),
        api=getattr(args, "api", None),
        api_key=getattr(args, "api_key", "") or "",
        domains=tuple(args.domains),
        char_ceiling=getattr(args, "char_ceiling", config.CHAR_CEILING),
        char_floor=getattr(args, "char_floor", config.MIN_CHUNK_CHARS),
        glossary_limit=getattr(args, "glossary_limit", None),
        max_chunks=getattr(args, "max_chunks", None),
        pages=tuple(args.pages) if getattr(args, "pages", None) else None,
        translate=not getattr(args, "no_translate", False)
        and not getattr(args, "no_rerun", False),
        log_level=args.log_level,
        qe_method=getattr(args, "qe_method", "auto"),
        refine=getattr(args, "refine", True),
        requeue=getattr(args, "requeue", True),
        quality_threshold=getattr(args, "quality_threshold", config.QUALITY_THRESHOLD),
        compliance_floor=getattr(args, "compliance_floor", config.COMPLIANCE_FLOOR),
        refine_budget=getattr(args, "refine_budget", config.REFINE_BUDGET),
        max_spans=getattr(args, "max_spans", config.REFINE_MAX_SPANS),
        no_drift=getattr(args, "no_drift", False),
        translated_html=not getattr(args, "no_translated_html", False),
        no_artifacts=getattr(args, "no_artifacts", False),
        no_format=getattr(args, "no_format", False),
        force_format=getattr(args, "force_format", False),
    )
    try:
        if args.command == "bakeoff":
            from book_harness.bakeoff import run_bakeoff

            report = run_bakeoff(
                settings, models=args.models, n_chunks=args.chunks,
                judge_model=args.judge,
            )
            logger.info("bake-off report: %s", report)
            return 0
        if args.command == "drift":
            from book_harness import drift as drift_mod
            from book_harness.chunker import load_chunks

            run_dir = settings.run_dir()
            if getattr(args, "confirm", None):
                confirms = drift_mod.parse_confirm_args(args.confirm)
                result = drift_confirm(settings, confirms)
                logger.info("drift confirm result: %s", result)
                return 0
            # mining: propose recurring uncovered terms from existing chunks
            chunks = load_chunks(run_dir / config.STAGE_DIRS["chunks"])
            _, entries, _ = _load_glossary(settings)
            candidates = drift_mod.mine_drift_terms(
                [c.to_dict() for c in chunks], entries,
                min_freq=args.min_freq, min_df=args.min_df,
            )
            out = settings.stage_dir(stage="glossary") / "drift-candidates.jsonl"
            drift_mod.write_candidates(
                candidates, out, glossary_version=_glossary_version(settings)
            )
            for c in candidates:
                logger.info(
                    "drift candidate: %s (freq %d, chunks %d, page %s)",
                    c["term"], c["freq"], c["df"], c.get("page"),
                )
            logger.info(
                "%d proposed terms -> %s (confirm with `drift --confirm TERM=AR`)",
                len(candidates), out,
            )
            return 0
        return run(settings)
    except ConnectionError as e:
        logger.error("ollama connection failed: %s", e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
