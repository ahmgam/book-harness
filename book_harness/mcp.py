"""MCP tool implementations (M3.5, PLAN.md task 3.5.1).

Library first: the four MCP tools are plain functions here (pure file I/O
plus the shared model-server client), and ``mcp/server.py`` is a thin stdlib
JSON-RPC-over-stdio shell that maps them onto MCP ``tools/list`` /
``tools/call``. Nothing here speaks MCP itself, so everything is testable
and reusable on its own.

Tools:

* ``lookup_term(term, lang, domain)`` — arabterm lookup returning *citations*
  (source dictionary, Wikidata ID, printed page, archive.org scan), optionally
  filtered through the domain → dictionary whitelist (PLAN.md catch #2).
* ``get_glossary(book_id, term)`` — a run's ``glossary.yaml``: version plus
  cited entries, optionally narrowed to one term.
* ``translate_chunk(text, glossary, context)`` — translate one unit of text
  through the same model-server client (Ollama by default, OpenAI-compatible
  on ``/v1`` hosts), Hy-MT2 terminology template + context, the same prompt
  path the pipeline uses (task 1.8 machinery), with optional model override.
* ``list_runs()`` — every run directory under the runs root with its stage
  cursor (``state.json``), chunk/draft counts and glossary version.

Run root resolution: ``BOOK_HARNESS_RUNS`` env var, default ``./runs`` — the
same convention the CLI uses as its default ``--run-root``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from book_harness import config
from book_harness.config import GLOSSARY_FILE, STAGE_DIRS, STATE_FILE

TOOL_NAMES = ("lookup_term", "get_glossary", "translate_chunk", "list_runs")

RUNS_ENV = "BOOK_HARNESS_RUNS"


# ---------------------------------------------------------------------------
# Runs root
# ---------------------------------------------------------------------------

def runs_root(run_root: str | Path | None = None) -> Path:
    if run_root is not None:
        return Path(run_root)
    return Path(os.environ.get(RUNS_ENV, "runs"))


def run_dir(book_id: str, run_root: str | Path | None = None) -> Path:
    return runs_root(run_root) / book_id


# ---------------------------------------------------------------------------
# lookup_term — arabterm citations, domain-filtered
# ---------------------------------------------------------------------------

def get_arabterm() -> Any:
    """Lazily open (download-once) arabterm.db; kept separate for tests."""
    from book_harness.arabterm import Arabterm, ensure_db

    return Arabterm.from_db(ensure_db())


def lookup_term(
    term: str,
    lang: str = "en",
    domain: str | None = None,
    *,
    arabterm: Any = None,
) -> dict[str, Any]:
    """Certified dictionary hits for a term, as citation dicts.

    ``lang`` selects the lookup column (``en`` or ``ar``); ``domain`` filters
    hits through the same domain → dictionary whitelist the glossary builder
    uses, so an agent gets the same translations the pipeline would choose.
    """
    term = (term or "").strip()
    if not term:
        raise ValueError("term must be a non-empty string")
    lang = "ar" if str(lang).lower().startswith("ar") else "en"
    arabterm = arabterm or get_arabterm()
    if lang == "ar":
        hits = arabterm.lookup_arabic(term)
    else:
        hits = arabterm.lookup_term(term)

    if domain:
        from book_harness.glossary import dictionary_allowed

        hits = [
            h for h in hits
            if dictionary_allowed(h.get("source_tech") or h.get("source") or "", [domain])
        ]
    return {
        "term": term,
        "lang": lang,
        "domain": domain,
        "hits": hits,
    }


# ---------------------------------------------------------------------------
# get_glossary — a run's cited glossary
# ---------------------------------------------------------------------------

def _norm_key(text: str) -> str:
    """Case/punctuation-insensitive key for term filtering.

    ``glossary.canonical_term`` is an injection matcher (it erases whole
    words), so a simpler normalization is used here.
    """
    import re

    return re.sub(r"[_\W]+", " ", (text or "").lower()).strip()


def get_glossary(
    book_id: str,
    term: str | None = None,
    *,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Glossary version + cited entries for one run directory.

    With ``term`` set, only entries whose English (normalized) or Arabic
    (diacritic-folded) form matches the query are returned.
    """
    from book_harness.glossary import load_glossary
    from book_harness.normalize import normalize_ar

    path = run_dir(book_id, run_root) / STAGE_DIRS["glossary"] / GLOSSARY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"no glossary for book_id '{book_id}' at {path} "
            f"(run the glossary stage first)"
        )
    version, entries = load_glossary(path)
    query = (term or "").strip()
    if query:
        if any("\u0600" <= ch <= "\u06ff" for ch in query):
            want, needle = normalize_ar(query), normalize_ar
        else:
            want, needle = _norm_key(query), _norm_key
        entries = [e for e in entries if needle(e.get("term_en") or "") == want
                   or needle(e.get("term_ar") or "") == want]
    by_status: dict[str, int] = {}
    for e in entries:
        key = e.get("status") or "unknown"
        by_status[key] = by_status.get(key, 0) + 1
    return {
        "book_id": book_id,
        "version": version,
        "path": str(path),
        "term": query or None,
        "count": len(entries),
        "by_status": by_status,
        "entries": [dict(e) for e in entries],
    }


# ---------------------------------------------------------------------------
# translate_chunk — one unit of text through the pipeline prompt path
# ---------------------------------------------------------------------------

def normalize_glossary(glossary: Any) -> list[dict[str, str]]:
    """Accept {term_en|en|english, term_ar|ar|arabic} dicts (or a wrapping
    ``{"entries": [...]}``) and return injectable entries."""
    if isinstance(glossary, dict):
        glossary = glossary.get("entries") or glossary.get("glossary") or []
    out: list[dict[str, str]] = []
    for item in glossary or []:
        if not isinstance(item, dict):
            continue
        en = (item.get("term_en") or item.get("en") or item.get("english")
              or item.get("term") or "")
        ar = (item.get("term_ar") or item.get("ar") or item.get("arabic") or "")
        if en.strip() and ar.strip():
            out.append({"term_en": en.strip(), "term_ar": ar.strip()})
    return out


def make_client(model: str | None = None) -> Any:
    """Model-server client, default Ollama; an OpenAI-compatible wire
    protocol is selected via ``BOOK_HARNESS_API=openai`` or a host URL ending
    in ``/v1`` (test seam: monkeypatch this function)."""
    from book_harness.translate import build_client

    host = os.environ.get("OLLAMA_HOST", config.OLLAMA_HOST)
    timeout = int(os.environ.get(
        "BOOK_HARNESS_OLLAMA_TIMEOUT", str(config.OLLAMA_TIMEOUT)
    ))
    return build_client(
        host,
        timeout=timeout,
        model=model,
        api=os.environ.get("BOOK_HARNESS_API", "auto"),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


def translate_chunk(
    text: str,
    glossary: Any,
    context: str = "",
    *,
    model: str | None = None,
    kind: str = "text",
) -> dict[str, Any]:
    """Translate one unit of text with glossary injection + prior context.

    ``glossary`` is a list of ``{en, ar}`` pairs (key aliases accepted). Only
    terms actually occurring in ``text`` are injected, exactly like the
    pipeline's per-unit injection (task 1.8 / catch #12 scope).
    """
    if not (text or "").strip():
        raise ValueError("text must be a non-empty string")
    from book_harness.translate import glossary_lines, match_terms, translate_unit

    entries = normalize_glossary(glossary)
    client = make_client(model)
    try:
        lines = glossary_lines(match_terms(text, entries))
        out = translate_unit(client, text, lines, context or "", kind)
    except ConnectionError:
        raise
    except Exception as exc:
        raise RuntimeError(f"translation failed: {exc}") from exc
    return {"model": client.model, "kind": kind, "translation": out}


# ---------------------------------------------------------------------------
# list_runs — stage cursors + counts across the runs root
# ---------------------------------------------------------------------------

def _looks_like_run(path: Path) -> bool:
    if (path / STATE_FILE).exists():
        return True
    return any((path / d).is_dir() for d in STAGE_DIRS.values())


def run_summary(run_path: str | Path) -> dict[str, Any]:
    run_path = Path(run_path)
    stages: dict[str, Any] = {}
    state_path = run_path / STATE_FILE
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                stages = {
                    k: v.get("status")
                    for k, v in (json.load(f).get("stages") or {}).items()
                }
        except (json.JSONDecodeError, OSError):
            stages = {"error": "unreadable state.json"}

    chunks_dir = run_path / STAGE_DIRS["chunks"]
    chunks = sum(1 for p in chunks_dir.glob("chunk-*.json")) if chunks_dir.is_dir() else 0
    drafts_dir = run_path / STAGE_DIRS["drafts"]
    translated = sum(1 for p in drafts_dir.glob("*.meta.json")) if drafts_dir.is_dir() else 0

    glossary: dict[str, Any] = None
    glossary_path = run_path / STAGE_DIRS["glossary"] / GLOSSARY_FILE
    if glossary_path.exists():
        try:
            from book_harness.glossary import load_glossary

            version, entries = load_glossary(glossary_path)
            glossary = {
                "version": version,
                "entries": len(entries),
                "certified": sum(
                    1 for e in entries if e.get("status") == "certified"
                ),
            }
        except Exception:
            glossary = {"error": "unreadable glossary.yaml"}

    output_dir = run_path / STAGE_DIRS["output"]
    book_md = (output_dir / "book.md").exists() if output_dir.is_dir() else False

    return {
        "book_id": run_path.name,
        "path": str(run_path),
        "stages": stages,
        "chunks": chunks,
        "translated": translated,
        "glossary": glossary,
        "assembled": book_md,
    }


def list_runs(run_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Every run directory under the runs root, sorted by book_id."""
    root = runs_root(run_root)
    if not root.is_dir():
        return []
    return [
        run_summary(p) for p in sorted(root.iterdir())
        if p.is_dir() and _looks_like_run(p)
    ]
