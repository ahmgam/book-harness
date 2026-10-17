"""M3 output artifacts (tasks 3.1, 3.2, 3.3, 3.5).

All writers are pure file I/O over the on-disk stage artifacts and never talk
to a model, so they are cheap, idempotent and re-runnable at any time:

* **Glossary appendix (3.1)** — ``07_output/glossary-appendix.md``:
  the cited back-matter glossary (term, translation, source dictionary,
  first-use page, Wikidata/scan link) grouped by status.
* **Bilingual HTML (3.2)** — ``07_output/bilingual.html``:
  paragraph-aligned side-by-side edition built from chunk pairs (English
  source left, Arabic draft right), passthrough blocks rendered once
  full-width.
* **QA report (3.3)** — ``07_output/qa-report.md``:
  per-chapter QE distribution, compliance %, patch counts and the
  uncertified terms used in each chapter, plus book-level totals.
* **TMX export (3.5)** — ``07_output/book.tmx``:
  TMX 1.4 translation memory of the aligned source/translation pairs.

The provenance manifest (3.4) is written by :func:`render_book` in
``assemble.py`` — every chunk already carries model, prompt version, glossary
version and timestamps there.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
import xml.sax.saxutils as _xml
from pathlib import Path
from typing import Any

from book_harness.book import Book
from book_harness.glossary import STATUS_CERTIFIED, STATUS_LLM_CHOSEN, STATUS_UNCERTIFIED
from book_harness.qa import TRANSLATED_KINDS, read_records, source_matches

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"§(?P<kind>[A-Z]+)-(?P<n>\d+)§")


def chunk_dicts(chunks: list[Any]) -> list[dict[str, Any]]:
    """Normalize a mix of Chunk objects / dicts into dicts."""
    return [c.to_dict() if not isinstance(c, dict) else c for c in chunks]


# ---------------------------------------------------------------------------
# 3.1 Glossary appendix
# ---------------------------------------------------------------------------

def record_first_use(
    entries: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> dict[str, int]:
    """First canonical page each glossary term is used on, for citations."""
    usage: dict[str, int] = {}
    index = _chunk_unit_index(chunks)
    for e in entries:
        term = e.get("term_en") or ""
        if not term:
            continue
        for page, text in index:
            if term in usage:
                break
            if source_matches(term, text):
                usage[term] = page
    return usage


def _chunk_unit_index(
    chunks: list[dict[str, Any]]
) -> list[tuple[int, str]]:
    """(page, source text) for every translatable unit across all chunks."""
    out: list[tuple[int, str]] = []
    for chunk in chunks:
        by_uid: dict[str, dict[str, Any]] = {}
        for ref in chunk.get("blocks", []):
            for uid in ref.get("units", []):
                by_uid[uid] = ref
        for unit in chunk.get("units", []):
            ref = by_uid.get(unit.get("uid", ""))
            page = (ref or {}).get("page")
            if page is None:
                page = (chunk.get("pages") or [0])[0]
            out.append((int(page), unit.get("text", "")))
    return out


def render_glossary_appendix(
    entries: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    out_path: str | Path,
    *,
    book_id: str = "",
) -> dict[str, Any]:
    """Write ``glossary-appendix.md``; returns a status summary."""
    chunks = chunk_dicts(chunks)
    usage = record_first_use(entries, chunks)
    grouped: dict[str, list[dict[str, Any]]] = {
        STATUS_CERTIFIED: [], STATUS_LLM_CHOSEN: [], STATUS_UNCERTIFIED: [],
    }
    for e in entries:
        grouped.setdefault(e.get("status") or STATUS_UNCERTIFIED, []).append(e)

    lines = [
        "# Glossary",
        "",
        "Certified terms come from the arabterm dictionaries with full"
        " provenance: source dictionary, Wikidata concept ID, archive.org"
        " scan. Uncertified terms await human review (catch #5).",
        "",
    ]
    counts: dict[str, int] = {}
    for status in (STATUS_CERTIFIED, STATUS_LLM_CHOSEN, STATUS_UNCERTIFIED):
        group = grouped.get(status) or []
        counts[status] = len(group)
        if not group:
            continue
        heading = {
            STATUS_CERTIFIED: "Certified (dictionary candidates)",
            STATUS_LLM_CHOSEN: "LLM-adjudicated",
            STATUS_UNCERTIFIED: "Uncertified (human review pending)",
        }[status]
        lines += [f"## {heading}", "", "| English | Arabic | Source | Page | Provenance |",
                  "|---|---|---|---:|---|"]
        for e in sorted(group, key=lambda e: (e.get("term_en") or "").lower()):
            term = e.get("term_en") or ""
            ar = e.get("term_ar") or "—"
            source = e.get("source") or ("human-confirmed" if e.get("decided_by") == "human" else "—")
            page = usage.get(term)
            page_s = f"p. {page}" if page else "—"
            links = []
            if e.get("wikidata"):
                links.append(f"[Q]({e['wikidata']})" if e["wikidata"].startswith("http")
                             else e["wikidata"])
            if e.get("scan"):
                links.append(f"[scan]({e['scan']})")
            lines.append(f"| {term} | {ar} | {source} | {page_s} | {' · '.join(links) or '—'} |")
        lines.append("")

    md = "\n".join(lines).rstrip() + "\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    summary = {"terms": len(entries), **counts, "path": str(out_path)}
    logger.info("glossary appendix: %d terms -> %s", len(entries), out_path)
    return summary


# ---------------------------------------------------------------------------
# 3.2 Bilingual HTML
# ---------------------------------------------------------------------------

_BILINGUAL_CSS = """\
body{font-family:Georgia,'Times New Roman',serif;margin:2rem auto;max-width:70rem;padding:0 1rem;color:#222}
h1{font-size:1.6rem}
.section{margin:2.5rem 0 1.5rem;border-bottom:2px solid #bbb;padding-bottom:.3rem}
.section h2{margin:0;font-size:1.15rem}
.section .meta{color:#777;font-size:.85rem;margin-top:.2rem}
.row{display:grid;grid-template-columns:1fr 1fr;gap:0 2rem;border-bottom:1px solid #eee;padding:.45rem 0}
.row img{max-width:100%}
.page-marker{grid-column:1/-1;color:#888;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;padding:1rem 0 .3rem;border-bottom:1px dotted #ccc}
.src{direction:ltr;text-align:left}
.tgt{direction:rtl;text-align:right}
.caption{font-size:.92rem;color:#555}
.full{grid-column:1/-1;padding:.6rem 0;border-bottom:1px solid #eee}
.missing{color:#a33;font-style:italic}
.list{margin:0;padding-inline-start:1.2rem}
table{border-collapse:collapse}td,th{border:1px solid #999;padding:2px 6px}
.passtype{float:right;color:#999;font-size:.72rem}
"""


def render_bilingual_html(
    book: Book,
    chunks: list[dict[str, Any]],
    drafts: Any,                             # assemble.DraftBook
    refine_dir: str | Path | None,
    out_path: str | Path,
    *,
    title: str = "Bilingual edition",
) -> dict[str, Any]:
    """Paragraph-aligned side-by-side English/Arabic edition (task 3.2).

    Rows are per translatable unit (source cell English, draft cell Arabic
    with ``dir="rtl"``); passthrough blocks (equations, code, tables, figures)
    render once, full width, with their original markup.
    """
    from book_harness.assemble import _apply_refinements, extract_images

    if refine_dir is not None:
        _apply_refinements(drafts, Path(refine_dir))
    out_path = Path(out_path)
    out_dir = out_path.parent
    urls = extract_images(book, out_dir / "images")
    chunks = chunk_dicts(chunks)

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{_xml.escape(title)}</title>",
        f"<style>{_BILINGUAL_CSS}</style></head><body>",
        f"<h1>{_xml.escape(title)}</h1>",
    ]
    stats = {"units": 0, "missing": 0, "rows": 0}
    last_page: int | None = None
    for chunk in chunks:
        dict_units = {u["uid"]: u for u in chunk.get("units", [])}
        section = chunk.get("section_title") or "(section)"
        parts.append(
            f'<section class="section" id="{_xml.escape(chunk["chunk_id"])}">'
            f"<h2>{_xml.escape(section)}</h2>"
            f"<div class='meta'>{_xml.escape(chunk['chunk_id'])}"
            + (f" · pages {chunk.get('pages')}" if chunk.get("pages") else "")
            + "</div></section>"
        )
        last_page = None
        for ref in chunk.get("blocks", []):
            page = ref.get("page")
            if page is not None and page != last_page:
                last_page = page
                parts.append(f'<div class="row"><div class="page-marker">'
                             f"page {page}</div></div>")
            uids = ref.get("units", [])
            if not uids:
                parts.append(_passthrough_row(book, ref, urls, stats))
                continue
            for uid in uids:
                unit = dict_units.get(uid) or {"uid": uid, "kind": "text", "text": ""}
                src = _restore(unit)
                drafted = drafts.unit(chunk["chunk_id"], uid)
                tgt = (drafted or {}).get("text")
                if not tgt:
                    tgt = None
                    stats["missing"] += 1
                else:
                    stats["units"] += 1
                stats["rows"] += 1
                parts.append(_unit_row(unit, src, tgt))
    parts += ["</body></html>"]
    html = "\n".join(parts)
    out_path.write_text(html, encoding="utf-8")
    logger.info(
        "bilingual edition: %d units (%d without draft) -> %s",
        stats["units"], stats["missing"], out_path,
    )
    return {**stats, "path": str(out_path)}


def _restore(unit: dict[str, Any]) -> str:
    """Source text with placeholder tokens substituted back."""
    text = unit.get("text", "")
    placeholders = unit.get("placeholders") or {}
    if not placeholders:
        return text

    def sub(m: "re.Match[str]") -> str:
        return placeholders.get(m.group(0), m.group(0))

    return _TOKEN_RE.sub(sub, text)


def _unit_row(unit: dict[str, Any], src: str, tgt: str | None) -> str:
    kind = unit.get("kind", "text")
    tag_open, tag_close = _kind_wrap(kind)
    tgt_cell = (f"<div class='tgt'>{tag_open}{_xml.escape(tgt)}{tag_close}</div>"
                if tgt is not None
                else "<div class='tgt missing'>— <em>draft missing</em></div>")
    return (
        f"<div class='row'>"
        f"<div class='src'>{tag_open}{_xml.escape(src)}{tag_close}</div>"
        f"{tgt_cell}</div>"
    )


def _kind_wrap(kind: str) -> tuple[str, str]:
    if kind == "title":
        return "<strong>", "</strong>"
    if kind == "caption":
        return "<em><span class='caption'>", "</span></em>"
    return "", ""


def _passthrough_row(
    book: Book, ref: dict[str, Any], urls: dict[str, str], stats: dict[str, int]
) -> str:
    block = book.block(ref["block_id"])
    if block is None:
        return ""
    html = re.sub(
        r"src='([^']+)'",
        lambda m: "src='%s'" % urls.get(m.group(1), m.group(1)),
        block.html,
    )
    stats["rows"] += 1
    return (
        f"<div class='full'>"
        f"<span class='passtype'>{_xml.escape(block.type.value)}</span>"
        f"{html}</div>"
    )


# ---------------------------------------------------------------------------
# 3.3 QA report
# ---------------------------------------------------------------------------

def render_qa_report(
    chunks: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    run_dir: str | Path,
    out_path: str | Path,
    *,
    quality_threshold: float = 80.0,
    compliance_floor: float = 95.0,
) -> dict[str, Any]:
    """Write ``qa-report.md``: per-chapter QE distribution, compliance %,
    patch counts and uncertified terms in use (task 3.3)."""
    from book_harness.config import STAGE_DIRS

    run_dir = Path(run_dir)
    qa_dir = run_dir / STAGE_DIRS["qa"]
    refine_dir = run_dir / STAGE_DIRS["refine"]
    qe = read_records(qa_dir / "qe-scores.jsonl")
    compliance = read_records(qa_dir / "compliance.jsonl")
    consistency = read_records(qa_dir / "consistency.jsonl")
    chunks = chunk_dicts(chunks)
    qe_by_chunk = {r["chunk_id"]: r for r in reversed(qe)}
    comp_by_chunk = {r["chunk_id"]: r for r in reversed(compliance)}

    # refinement counters per chunk (05_refine/*.refined.json)
    refine_counters: dict[str, dict[str, int]] = {}
    for path in sorted(refine_dir.glob("chunk-*.refined.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            refine_counters[path.name.split(".")[0]] = data.get("counters") or {}
        except (OSError, json.JSONDecodeError):
            continue

    uncertified = [e for e in entries if e.get("status") == STATUS_UNCERTIFIED
                   and e.get("term_en")]

    # group chunks by their root outline node (chapter)
    chapters: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for c in chunks:
        path = c.get("section_path") or [{"number": None, "title": "Front matter"}]
        root = path[0]
        label = (f"{root.get('number') or ''} {root.get('title')}"
                 .strip()) or "(front matter)"
        if label not in chapters:
            chapters[label] = []
            order.append(label)
        chapters[label].append(c)

    rows = []
    for label in order:
        group = chapters[label]
        chunk_ids = {c["chunk_id"] for c in group}
        scores = [r["score"] for r in (qe_by_chunk.get(i) for i in chunk_ids)
                  if r and r.get("score") is not None]
        rates = [r["rate"] for r in (comp_by_chunk.get(i) for i in chunk_ids)
                 if r and r.get("rate") is not None]
        violations = sum(len(r.get("violations") or [])
                         for r in (comp_by_chunk.get(i) for i in chunk_ids) if r)
        spans = sum(rc.get("spans", 0) for i in chunk_ids
                    if (rc := refine_counters.get(i)))
        accepted = sum(rc.get("accepted", 0) for i in chunk_ids
                       if (rc := refine_counters.get(i)))
        rejected = sum(rc.get("rejected", 0) for i in chunk_ids
                       if (rc := refine_counters.get(i)))
        used_terms = sorted({
            e["term_en"] for e in uncertified
            if any(source_matches(e["term_en"], u.get("text", ""))
                   for c in group for u in c.get("units", []))
        })
        rows.append({
            "chapter": label,
            "chunks": len(group),
            "qe": scores,
            "qe_mean": round(statistics.mean(scores), 1) if scores else None,
            "qe_below_threshold": sum(1 for s in scores if s < quality_threshold),
            "compliance_mean": round(statistics.mean(rates), 1) if rates else None,
            "compliance_below_floor": sum(1 for r in rates if r < compliance_floor),
            "violations": violations,
            "spans": spans,
            "patches_accepted": accepted,
            "patches_rejected": rejected,
            "uncertified_used": used_terms,
        })

    book_scores = [s for r in rows for s in r["qe"]]
    all_rates = [r["compliance_mean"] for r in rows if r["compliance_mean"] is not None]
    drift_terms = sum(1 for r in consistency if r.get("drift"))
    summary = {
        "chapters": len(rows),
        "chunks": sum(r["chunks"] for r in rows),
        "qe_mean": round(statistics.mean(book_scores), 1) if book_scores else None,
        "qe_min": min(book_scores) if book_scores else None,
        "compliance_mean": round(statistics.mean(all_rates), 1) if all_rates else None,
        "consistency_drift_terms": drift_terms,
        "uncertified_terms": len(uncertified),
        "patches_accepted": sum(r["patches_accepted"] for r in rows),
        "patches_rejected": sum(r["patches_rejected"] for r in rows),
        "path": str(out_path),
    }
    _write_report_md(out_path, rows, summary, quality_threshold, compliance_floor)
    logger.info(
        "QA report: %d chapters, mean QE %s, compliance %s%% -> %s",
        summary["chapters"], summary["qe_mean"], summary["compliance_mean"], out_path,
    )
    return summary


def _write_report_md(
    out_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    quality_threshold: float,
    compliance_floor: float,
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        "# QA report",
        f"*generated {now}*",
        "",
        f"**Chapters:** {summary['chapters']} · **chunks:** {summary['chunks']} ·"
        f" mean QE **{summary['qe_mean']}** (min {summary['qe_min']}) ·"
        f" mean compliance **{summary['compliance_mean']}%** ·"
        f" consistency drift terms **{summary['consistency_drift_terms']}** ·"
        f" uncertified glossary terms **{summary['uncertified_terms']}** ·"
        f" patches **{summary['patches_accepted']} accepted /"
        f" {summary['patches_rejected']} rejected**",
        "",
        "| Chapter | Chunks | QE mean | QE range | QE < %.0f | Compliance %% |"
        " Below floor | Violations | Spans | Patches ± | Uncertified terms |"
        % quality_threshold,
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lo = min(r["qe"]) if r["qe"] else None
        hi = max(r["qe"]) if r["qe"] else None
        rng = f"{lo:.0f}–{hi:.0f}" if lo is not None else "—"
        terms = ", ".join(r["uncertified_used"]) or "—"
        lines.append(
            f"| {_xml.escape(r['chapter'])} | {r['chunks']} |"
            f" {r['qe_mean'] if r['qe_mean'] is not None else '—'} | {rng} |"
            f" {r['qe_below_threshold']} |"
            f" {r['compliance_mean'] if r['compliance_mean'] is not None else '—'} |"
            f" {r['compliance_below_floor']} | {r['violations']} | {r['spans']} |"
            f" {r['patches_accepted']} / {r['patches_rejected']} | {_xml.escape(terms)} |"
        )
    lines += ["", f"Compliance floor {compliance_floor:.0f}%;"
              " chunks below the floor were re-queued for targeted re-translation.",
              ""]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 3.5 TMX export
# ---------------------------------------------------------------------------

def render_tmx(
    chunks: list[dict[str, Any]],
    drafts: Any,                             # assemble.DraftBook
    out_path: str | Path,
    *,
    source_lang: str = "en",
    target_lang: str = "ar",
    book_id: str = "",
) -> dict[str, Any]:
    """TMX 1.4 translation memory of aligned chunk pairs (task 3.5).

    One ``<tu>`` per translated unit (source/translation), section and chunk
    recorded as ``<prop>``; draft-missing units are skipped, not invented.
    """
    from book_harness import __version__

    chunks = chunk_dicts(chunks)
    pairs: list[dict[str, str]] = []
    for chunk in chunks:
        for unit in chunk.get("units", []):
            if unit.get("kind") not in TRANSLATED_KINDS:
                continue
            d = drafts.unit(chunk["chunk_id"], unit["uid"]) if drafts else None
            if not d or not (d.get("text") or "").strip():
                continue
            pairs.append({
                "chunk_id": chunk["chunk_id"],
                "section": chunk.get("section_title", ""),
                "uid": unit["uid"],
                "src": _restore(unit),
                "tgt": d["text"],
            })

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<tmx version="1.4">',
        f'  <header creationtool="book-harness" creationtoolversion="{__version__}"'
        f' segtype="paragraph" o-tmf="plain text" adminlang="{source_lang}"'
        f' datatype="plaintext" srclang="{_xml.escape(source_lang)}">'
        + (f'<prop type="book">{_xml.escape(book_id)}</prop></header>'
           if book_id else "</header>"),
        "  <body>",
    ]
    for p in pairs:
        tuid = _xml.escape(p["chunk_id"] + "/" + p["uid"])
        block = [
            f'    <tu tuid="{tuid}">',
        ]
        if p["section"]:
            block.append(f'      <prop type="section">{_xml.escape(p["section"])}</prop>')
        block.append(f'      <prop type="chunk">{_xml.escape(p["chunk_id"])}</prop>')
        block.append(
            f'      <tuv xml:lang="{_xml.escape(source_lang)}">'
            f"<seg>{_xml.escape(p['src'])}</seg></tuv>"
        )
        block.append(
            f'      <tuv xml:lang="{_xml.escape(target_lang)}" dir="rtl">'
            f"<seg>{_xml.escape(p['tgt'])}</seg></tuv>"
        )
        block.append("    </tu>")
        lines.extend(block)
    lines += ["  </body>", "</tmx>", ""]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("TMX export: %d translation units -> %s", len(pairs), out_path)
    return {"translation_units": len(pairs), "path": str(out_path)}
