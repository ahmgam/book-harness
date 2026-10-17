"""Model bake-off (M2: task 2.7).

Runs the same sample of chunks through every installed candidate MT model
(e.g. ``hy-mt2-30b-a3b``, ``hy-mt2-7b``, ``translategemma``, ``qwen3.5``),
then scores each run with the QE judge and the compliance audit, aggregates
per model, writes ``bakeoff.json`` + ``bakeoff.md``, and recommends a default
model empirically (composite = 0.6*QE + 0.4*compliance%).

Requires a prepared run directory (``book-harness run --no-translate``) for
the chunk list and glossary; translation happens in isolated output dirs so a
bake-off never touches the real drafts.

Judging uses the *assist* model universe (``config.ASSIST_MODELS``), falling
back to the model under test when nothing else is installed (caveat is noted
in the report).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from book_harness import config
from book_harness.assemble import DraftBook
from book_harness.qa import compliance_audit, qe_score_chunk
from book_harness.translate import SummaryTracker, build_client, translate_chunk

logger = logging.getLogger(__name__)

QE_WEIGHT = 0.6
COMPLIANCE_WEIGHT = 0.4


def _pick_sample(chunks: list[Any], n: int) -> list[Any]:
    """Evenly spaced slice stratified across the book (not just the head)."""
    if len(chunks) <= n:
        return list(chunks)
    step = len(chunks) / n
    return [chunks[int(i * step)] for i in range(n)]


def score_run(
    chunks: list[Any],
    drafts_dir: Path,
    entries: list[dict],
    glossary_version: str,
    qe_client: Any,
) -> list[dict]:
    """QE + compliance records for every drafted chunk in one run dir."""
    drafts = DraftBook(drafts_dir)
    qe_records, compliance_records = [], []
    for chunk in chunks:
        draft = drafts.meta_by_chunk.get(chunk.chunk_id)
        if not draft:
            continue
        cd = chunk.to_dict()
        qe_records.append(qe_score_chunk(
            cd, draft, qe_client,
            qe_prompt_version=config.QE_PROMPT_VERSION,
        ))
        rec = compliance_audit(
            cd, draft, entries, glossary_version=glossary_version
        )
        compliance_records.append(rec)
    return qe_records, compliance_records


def aggregate(qe_records: list[dict], compliance_records: list[dict]) -> dict:
    qe_scores = [r["score"] for r in qe_records if r.get("score") is not None]
    rates = [r["rate"] for r in compliance_records if r.get("rate") is not None]
    qe_mean = round(sum(qe_scores) / len(qe_scores), 1) if qe_scores else None
    comp_mean = round(sum(rates) / len(rates), 1) if rates else None
    composite = None
    if qe_mean is not None or comp_mean is not None:
        composite = round(
            QE_WEIGHT * (qe_mean or 0.0) + COMPLIANCE_WEIGHT * (comp_mean or 0.0), 1
        )
    return {
        "chunks_scored": len(qe_records),
        "qe_mean": qe_mean,
        "compliance_mean": comp_mean,
        "composite": composite,
    }


def run_bakeoff(
    settings: Any,
    models: list[str],
    n_chunks: int = 30,
    judge_model: str | None = None,
) -> Path:
    """Translate a chunk sample per candidate model; score + report."""
    from book_harness.chunker import load_chunks

    run_dir = settings.run_dir()
    chunks_dir = run_dir / config.STAGE_DIRS["chunks"]
    glossary_path = run_dir / config.STAGE_DIRS["glossary"] / config.GLOSSARY_FILE
    if not chunks_dir.exists() or not glossary_path.exists():
        raise SystemExit(
            "bake-off needs a prepared run: 'book-harness run ... --no-translate' first"
        )
    from book_harness.glossary import load_glossary

    chunks = load_chunks(chunks_dir)
    if settings.pages:
        lo, hi = settings.pages
        chunks = [c for c in chunks if lo <= min(c.pages) and max(c.pages) < hi]
    if settings.max_chunks:
        chunks = chunks[: settings.max_chunks]
    sample = _pick_sample(chunks, n_chunks)
    _, entries, _ = _load_glossary(settings)
    gv, _, _ = _load_glossary(settings)

    out_dir = run_dir / config.STAGE_DIRS["qa"] / "bakeoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    qe_client = _judge_client(settings, judge_model)
    results: dict[str, dict[str, Any]] = {}

    for model in models:
        slug = model.replace("/", "_").replace(":", "-")
        dir_m = out_dir / slug
        drafts_dir = dir_m / "04_drafts"
        started = time.time()
        translated = 0
        try:
            client = build_client(
                settings.ollama_host,
                timeout=settings.ollama_timeout,
                preferred=[model],
                api=settings.api,
                api_key=settings.api_key,
                model=model,
            )
        except (RuntimeError, ConnectionError, ValueError) as exc:
            results[model] = {"error": str(exc)}
            logger.warning("bake-off: model %s unavailable: %s", model, exc)
            continue
        summaries = SummaryTracker(
            client, drafts_dir / "chapter-summary.json",
            period=settings.summary_period,
        )
        for c in sample:
            translate_chunk(
                c.to_dict(), client, entries, drafts_dir,
                prompt_version="bakeoff-v1", glossary_version=gv,
                summaries=summaries,
            )
            translated += 1
        duration = round(time.time() - started, 1)
        qe_rec, comp_rec = score_run(
            sample, drafts_dir, entries, gv, qe_client
        )
        agg = aggregate(qe_rec, comp_rec)
        agg.update({
            "model": client.model,
            "chunks_translated": translated,
            "duration_sec": duration,
            "judge": getattr(qe_client, "model", ""),
            "drafts_dir": str(drafts_dir),
        })
        results[model] = agg
        logger.info("bake-off %s: %s", model, agg)

    _write_report(results, out_dir / "bakeoff.json", out_dir / "bakeoff.md")
    return out_dir / "bakeoff.md"


def _judge_client(settings: Any, judge_model: str | None) -> Any:
    """QE judge = an assist model, never the model under test when avoidable."""
    client: Any
    try:
        client = build_client(
            settings.ollama_host,
            timeout=settings.ollama_timeout,
            preferred=config.QE_MODELS,
            api=settings.api,
            api_key=settings.api_key,
            model=judge_model,
        )
        if judge_model and judge_model.rstrip(":") not in (client.model or ""):
            # same family as a bake-off candidate — note the bias in logs
            logger.warning("judge %s is also a bake-off candidate; scores may be biased", client.model)  # noqa: E501
    except (RuntimeError, ConnectionError, ValueError) as exc:
        logger.warning("no dedicated judge: %s; QE scoring disabled in report", exc)
        return _NullClient()
    return client


class _NullClient:
    """QE fallback when no judge model exists at all."""

    model = ""


def _write_report(
    results: dict[str, dict[str, Any]],
    json_path: Path,
    md_path: Path,
) -> None:
    scored = {m: r for m, r in results.items() if "error" not in r}
    ranked = sorted(
        scored.items(), key=lambda kv: kv[1].get("composite") or -1, reverse=True
    )
    payload = {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "composite_formula": f"{QE_WEIGHT}*qe + {COMPLIANCE_WEIGHT}*compliance",
        "recommended": ranked[0][0] if ranked else None,
        "models": results,
    }
    import json

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["# Model bake-off", ""]
    if ranked:
        lines += [
            f"Recommended (highest composite): **{ranked[0][0]}**", "",
            "| model | QE mean | compliance % | composite | chunks | duration s |",
            "|---|---|---|---|---|---|",
        ]
        for name, r in scored.items():
            lines.append(
                f"| {r['model'] or name} | {r['qe_mean']} | {r['compliance_mean']} "
                f"| {r['composite']} | {r['chunks_scored']} | {r['duration_sec']} |"
            )
    for name, r in results.items():
        if "error" in r:
            lines += ["", f"*{name}: unavailable — {r['error']}*"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_glossary(settings: Any) -> tuple[str, list[dict], Path]:
    from book_harness.glossary import load_glossary

    path = settings.run_dir() / config.STAGE_DIRS["glossary"] / config.GLOSSARY_FILE
    version, entries = load_glossary(path)
    return version, entries, path
