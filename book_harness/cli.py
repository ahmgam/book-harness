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

STAGES = ["source", "terms", "glossary", "chunk", "translate", "assemble"]


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


def stage_assemble(settings: Settings, book: Book) -> Path:
    from book_harness.assemble import DraftBook, render_book
    from book_harness.chunker import load_chunks

    chunks = load_chunks(settings.run_dir() / config.STAGE_DIRS["chunks"])
    drafts = DraftBook(settings.stage_dir(stage="drafts"))
    out_dir = settings.stage_dir(stage="output")
    render_book(book, chunks, drafts, out_dir, book_id=settings.book_id)
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

    if not settings.translate:
        logger.info("translation skipped (--no-translate)")
        state.save()
        return 0

    state.mark("translate", status="started")
    n_translated = stage_translate(settings, chunks)
    state.mark("translate", status="done", chunks=n_translated)

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
    run_p.add_argument("--log-level", default="INFO")

    sub.add_parser("stages", help="show pipeline stages")
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
        model=args.model,
        ollama_host=args.ollama_host,
        domains=tuple(args.domains),
        char_ceiling=args.char_ceiling,
        char_floor=args.char_floor,
        glossary_limit=args.glossary_limit,
        max_chunks=args.max_chunks,
        pages=tuple(args.pages) if args.pages else None,
        translate=not args.no_translate,
        log_level=args.log_level,
    )
    try:
        return run(settings)
    except ConnectionError as e:
        logger.error("ollama connection failed: %s", e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
