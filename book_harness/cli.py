"""Thin CLI over the library (task 1.11).

    book-harness run <main-ocr-json> [options]      # full pipeline
    book-harness stages                             # describe the stages
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
    "source", "terms", "glossary", "chunk", "gate", "translate",
    "qe", "refine", "compliance", "consistency", "assemble",
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
    from book_harness.translate import (
        OllamaClient,
        SummaryTracker,
        translate_chunk,
    )

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
    client = OllamaClient(
        host=settings.ollama_host, timeout=settings.ollama_timeout,
        preferred=config.TRANSLATION_MODELS,
    )
    client.pick_model(settings.model)
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


def _assist_client(settings: Settings):
    from book_harness.translate import OllamaClient

    client = OllamaClient(
        host=settings.ollama_host,
        timeout=settings.ollama_timeout,
        preferred=config.ASSIST_MODELS,
    )
    client.pick_model(settings.model)          # explicit model also steers QA roles
    return client


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
        else _assist_client(settings)
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
    client = _assist_client(settings)
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
        state.mark("source", status="done")
        logger.info("stage source: done")

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

    state.mark("assemble", status="started")
    stage_assemble(settings, book)
    state.mark("assemble", status="done")
    logger.info("book.md at %s", run_dir / config.STAGE_DIRS["output"])
    state.save()
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="book-harness",
        description="OCR JSON -> cited glossary -> translation -> markdown",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the full pipeline")
    run_p.add_argument("ocr_json", type=Path, help="main OCR JSON file")
    run_p.add_argument("--run-root", type=Path, default=Path("runs"))
    run_p.add_argument("--model", default=None, help="force a translation model (Ollama)")
    run_p.add_argument("--ollama-host", default=config.OLLAMA_HOST)
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
    bake_p.add_argument("--ollama-host", default=config.OLLAMA_HOST)
    bake_p.add_argument("--ollama-timeout", type=int, default=config.OLLAMA_TIMEOUT)
    bake_p.add_argument("--max-chunks", type=int, default=None)
    bake_p.add_argument("--pages", nargs=2, type=int, metavar=("LO", "HI"))
    bake_p.add_argument("--domains", nargs="+", default=list(config.DEFAULT_DOMAINS))
    bake_p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stages":
        for i, s in enumerate(STAGES, 1):
            print(f"{i}. {s}")
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
        domains=tuple(args.domains),
        char_ceiling=getattr(args, "char_ceiling", config.CHAR_CEILING),
        char_floor=getattr(args, "char_floor", config.MIN_CHUNK_CHARS),
        glossary_limit=getattr(args, "glossary_limit", None),
        max_chunks=getattr(args, "max_chunks", None),
        pages=tuple(args.pages) if getattr(args, "pages", None) else None,
        translate=not getattr(args, "no_translate", False),
        log_level=args.log_level,
        qe_method=getattr(args, "qe_method", "auto"),
        refine=getattr(args, "refine", True),
        requeue=getattr(args, "requeue", True),
        quality_threshold=getattr(args, "quality_threshold", config.QUALITY_THRESHOLD),
        compliance_floor=getattr(args, "compliance_floor", config.COMPLIANCE_FLOOR),
        refine_budget=getattr(args, "refine_budget", config.REFINE_BUDGET),
        max_spans=getattr(args, "max_spans", config.REFINE_MAX_SPANS),
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
        return run(settings)
    except ConnectionError as e:
        logger.error("ollama connection failed: %s", e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
