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
* **Translated print edition** — ``07_output/translated.html``:
  RTL Arabic-only, page-divided HTML rendered *from the TMX*; sections map
  1:1 to book pages with print CSS (``@page`` + ``break-after: page``) so
  printing splits correctly.

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


def leaf_section(chunk: dict[str, Any]) -> tuple[str, int]:
    """Display heading for a chunk's section: ``(label, depth)`` (1-based).

    A chunk's breadcrumb (``section_path``, root→leaf) describes the full
    hierarchy; the heading shows only the leaf's own section title, wrapped
    in a heading tag matching the depth the section sits at.  Falls back to
    taking the last ``>``-separated part of the legacy ``section_title``
    string when the breadcrumb is missing (or is "(front matter)").
    """
    path = [s for s in (chunk.get("section_path") or [])
            if str(s.get("title") or "").strip()]
    if path:
        leaf = path[-1]
        parts = (leaf.get("number"), leaf.get("title"))
        label = " ".join(str(p) for p in parts if str(p or "").strip()).strip()
        return label or "(section)", min(max(len(path), 1), 6)
    raw = str(chunk.get("section_title") or "").strip()
    parts = [p.strip() for p in raw.split(">") if p.strip()]
    if parts:
        return parts[-1], min(max(len(parts), 1), 6)
    return "(front matter)", 1


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
.section h1,.section h2,.section h3,.section h4,.section h5,.section h6{margin:0;font-size:1.15rem}
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
        section, depth = leaf_section(chunk)
        heading_tag = f"h{min(depth, 6)}"
        parts.append(
            f'<section class="section" id="{_xml.escape(chunk["chunk_id"])}">'
            f"<{heading_tag}>{_xml.escape(section)}</{heading_tag}>"
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


def _restore(unit: dict[str, Any], text: str | None = None) -> str:
    """Text with placeholder tokens substituted back; defaults to the source
    text, pass *text* to restore a translated variant (draft target)."""
    text = unit.get("text", "") if text is None else text
    placeholders = unit.get("placeholders") or {}
    if not placeholders:
        return text

    def sub(m: "re.Match[str]") -> str:
        return placeholders.get(m.group(0), m.group(0))

    return _TOKEN_RE.sub(sub, text)


def _rewire_images(html: str, urls: dict[str, str]) -> str:
    """Rewrite ``src='…'`` attributes in block HTML to extracted file paths."""
    return re.sub(
        r"src='([^']+)'",
        lambda m: "src='%s'" % urls.get(m.group(1), m.group(1)),
        html,
    )


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
# 3.5 Translated print edition (stands alone on the TMX)
# ---------------------------------------------------------------------------

_PRINT_CSS = """\
@page{size:A4;margin:18mm 16mm 20mm;@bottom-center{content:"— " counter(page) " —";font-size:.75rem;color:#888}}
html{font-size:12pt}
body{margin:0;background:#fff;color:#111;direction:rtl;text-align:right;
  font-family:'Noto Naskh Arabic','Amiri','Scheherazade New','Traditional Arabic','Tahoma',serif;
  line-height:1.75}
h1{font-size:1.5rem;text-align:center;margin:0 0 1.5rem}
h2{font-size:1.05rem;color:#333;border-bottom:1px solid #ddd;padding-bottom:.2rem;margin:1.2rem 0 .6rem}
h3{font-size:.95rem;color:#444;margin:.9rem 0 .3rem}
h4,h5,h6{font-size:.9rem;color:#555;margin:.8rem 0 .3rem}
p{margin:.35rem 0;text-align:justify;hyphens:none}
p.title{font-weight:700;font-size:1.05rem}
p.caption{font-size:.92rem;color:#555;text-align:center}
.full{margin:.6rem 0;page-break-inside:avoid;break-inside:avoid-page}
.full img{max-width:100%;height:auto;display:block;margin:.5rem auto}
.full .passtype{float:left;color:#a09890;font-size:.68rem;font-weight:400;
  text-transform:inherit;letter-spacing:0}
.full .figmark{clear:both;color:#a09890;text-align:left;font-size:.68rem}
.full table{border-collapse:collapse;width:100%}
.full td,.full th{border:1px solid #999;padding:4px 8px}
.full .list{margin:.3rem 0;padding-inline-start:1.4rem}
.full math{font-size:1.05em}
.full pre,.full code{direction:ltr;text-align:left;font-family:monospace;font-size:.85rem;
  background:#f4f4f4;padding:.3rem .5rem;border-radius:4px;unicode-bidi:embed}
.page{page-break-after:always;break-after:page;page-break-inside:avoid;break-inside:avoid-page}
.page:last-of-type{page-break-after:auto;break-after:auto}
.page header{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:2px solid #999;padding-bottom:.2rem;margin-bottom:.6rem;
  font-size:.72rem;color:#888;text-transform:uppercase;letter-spacing:.08em}
@media screen{
  body{background:#525659}
  .page{background:#fff;margin:1.2rem auto;max-width:38rem;padding:1.4rem 1.8rem;
    border-radius:6px;box-shadow:0 1px 8px rgba(0,0,0,.35)}
}
@media print{
  body{background:#fff}
  .page{padding:0;max-width:none;margin:0}
}
"""


def _section_depth_prop(raw: str | None) -> int | None:
    """Parsed ``section-depth`` prop, ``None`` when absent/garbage/out of 1..6."""
    if not raw:
        return None
    try:
        depth = int(raw)
    except ValueError:
        return None
    return depth if 1 <= depth <= 6 else None


def render_translated_html(
    tmx_path: str | Path,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    """Page-divided, print-ready HTML of the translated book from a TMX.

    Reads ``07_output/book.tmx`` (each ``<tu>`` carries its book page in a
    ``<prop type="page">``), groups translated units by page and emits RTL
    Arabic sections that each map to one printed page: ``@page`` size/margins
    and ``break-after: page`` per section, so printing the page from a browser
    reproduces the book's pagination.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(tmx_path).getroot()
    if title is None:
        prop = root.find("header/prop[@type='book']")
        title = (prop.text if prop is not None and prop.text
                 else "Translated edition")

    units: list[dict[str, Any]] = []
    last_page: int | None = None
    for tu in root.iter("tu"):
        props = {pr.get("type"): (pr.text or "") for pr in tu.findall("prop")}
        if props.get("page"):
            try:
                last_page = int(props["page"])
            except ValueError:
                pass
        seg = None
        for tuv in tu.findall("tuv"):
            lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang") or ""
            if lang.lower().startswith("ar"):
                seg = tuv.find("seg")
                break
        text = "".join(seg.itertext()) if seg is not None else None
        if not (text or "").strip():
            continue
        units.append({
            "page": last_page,
            "section": props.get("section-title") or props.get("section") or "",
            "depth": _section_depth_prop(props.get("section-depth")),
            "kind": (props.get("kind") or "text").lower(),
            "blocktype": props.get("blocktype") or "",
            "chunk": props.get("chunk") or "",
            "text": text,
        })

    # group by page; TMX without page props lands in one "(unnumbered)" bucket
    buckets: dict[Any, list[dict[str, Any]]] = {}
    order: list[Any] = []
    for u in units:
        key = u["page"] if u["page"] is not None else None
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(u)

    numbered = [(p, buckets[p]) for p in sorted(o for o in order if o is not None)]
    unnumbered = [(None, buckets[p]) for p in order if p is None]
    page_groups = numbered + unnumbered

    parts = [
        "<!DOCTYPE html>",
        '<html lang="ar" dir="rtl"><head><meta charset="utf-8">',
        f"<title>{_xml.escape(title)}</title>",
        f"<style>{_PRINT_CSS}</style></head><body>",
        f"<h1>{_xml.escape(title)}</h1>",
    ]
    total = len(page_groups)
    for pos, (page, group) in enumerate(page_groups, 1):
        label = f"page {page}" if page is not None else "(unnumbered)"
        parts.append(
            f'<section class="page" id="page-{page if page is not None else "x"}">'
            f"<header><span>{_xml.escape(label)}</span>"
            f"<span>{pos} / {total}</span></header>"
        )
        last_section: tuple[str, int | None] | None = None
        for u in group:
            heading = (u["section"], u.get("depth"))
            if u["section"] and heading != last_section:
                last_section = heading
                tag = f"h{min(u.get('depth') or 2, 6)}"
                parts.append(f"<{tag}>{_xml.escape(u['section'])}</{tag}>")
            if u["kind"] == "passthrough":
                html = (u["text"] or "").rstrip()
                if "figure" in u["blocktype"].lower():
                    html += ('<div class="figmark">'
                             + _xml.escape(u["blocktype"]) + "</div>")
                parts.append(f'<div class="full">{html}</div>')
            elif u["kind"] == "title":
                parts.append(f'<p class="title">{_xml.escape(u["text"])}</p>')
            elif u["kind"] == "caption":
                parts.append(f'<p class="caption">{_xml.escape(u["text"])}</p>')
            else:
                parts.append(f"<p>{_xml.escape(u['text'])}</p>")
        parts.append("</section>")
    parts += ["</body></html>"]
    html = "\n".join(parts)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info(
        "translated print edition: %d pages, %d units -> %s",
        total, len(units), out_path,
    )
    return {"pages": total, "units": len(units), "path": str(out_path)}


# ---------------------------------------------------------------------------
# 3.6 TMX export
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

def render_tmx(
    chunks: list[dict[str, Any]],
    drafts: Any,                             # assemble.DraftBook
    out_path: str | Path,
    *,
    source_lang: str = "en",
    target_lang: str = "ar",
    book_id: str = "",
    book: Any | None = None,                 # Book, to embed passthrough blocks
) -> dict[str, Any]:
    """TMX 1.4 translation memory of aligned chunk pairs (task 3.5).

    One ``<tu>`` per translated unit (source/translation), section, page and
    chunk recorded as ``<prop>``; draft-missing units are skipped, not
    invented.  When *book* is given the non-translatable passthrough blocks
    (figures, pictures, tables, equations, code) are interleaved in reading
    order as ``kind="passthrough"`` units carrying their rendered HTML —
    image ``src`` attributes are rewritten to ``images/…`` copies written next
    to the TMX — so the print edition can render the full layout from the TMX
    alone.  Target placeholders are restored with the unit's placeholder map
    so inline equations/links survive into the TMX too.
    """
    from book_harness import __version__
    from book_harness.assemble import extract_images

    out_dir = Path(out_path).parent
    urls: dict[str, str] = {}
    if book is not None:
        urls = extract_images(book, out_dir / "images")

    chunks = chunk_dicts(chunks)
    pairs: list[dict[str, Any]] = []
    for chunk in chunks:
        ref_by_uid: dict[str, dict[str, Any]] = {}
        for ref in chunk.get("blocks", []):
            for uid in ref.get("units", []):
                ref_by_uid[uid] = ref

        def _page(ref: dict[str, Any] | None) -> int | None:
            page = ref.get("page") if ref else None
            if page is None:
                page = (chunk.get("pages") or [None])[0]
            return int(page) if page is not None else None

        def _section() -> str:
            return chunk.get("section_title", "")

        section_label, section_depth = leaf_section(chunk)

        seen_units: set[str] = set()

        def _unit(uid: str, ref: dict[str, Any] | None) -> None:
            if uid in seen_units:
                return
            seen_units.add(uid)
            unit = dict_units.get(uid)
            if not unit or unit.get("kind") not in TRANSLATED_KINDS:
                return
            d = drafts.unit(chunk["chunk_id"], uid) if drafts else None
            if not d or not (d.get("text") or "").strip():
                return
            pairs.append({
                "kind": unit.get("kind", "text"),
                "chunk_id": chunk["chunk_id"],
                "section": _section(),
                "section_label": section_label,
                "section_depth": section_depth,
                "uid": uid,
                "page": _page(ref),
                "src": _restore(unit),
                "tgt": _restore(unit, d["text"]),
            })

        dict_units: dict[str, dict[str, Any]] = {}
        for unit in chunk.get("units", []):
            dict_units.setdefault(unit.get("uid", ""), unit)

        # reading order: block refs first, passthrough blocks in between;
        # units not attached to any block ref are appended afterwards
        if book is not None:
            for ref in chunk.get("blocks", []):
                if not ref.get("units"):
                    block = book.block(ref["block_id"])
                    if block is None:
                        continue
                    pairs.append({
                        "kind": "passthrough",
                        "blocktype": block.type.value,
                        "chunk_id": chunk["chunk_id"],
                        "section": _section(),
                        "section_label": section_label,
                        "section_depth": section_depth,
                        "uid": ref["block_id"].lstrip("/"),
                        "page": _page(ref),
                        "src": "",
                        "tgt": _rewire_images(block.html, urls),
                    })
                for uid in ref.get("units", []):
                    _unit(uid, ref)
        else:
            for unit in chunk.get("units", []):
                _unit(unit.get("uid", ""), ref_by_uid.get(unit.get("uid", "")))
        for uid in dict_units:
            _unit(uid, ref_by_uid.get(uid))


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
        block.append(f'      <prop type="kind">{_xml.escape(p["kind"])}</prop>')
        if p.get("blocktype"):
            block.append(
                f'      <prop type="blocktype">{_xml.escape(p["blocktype"])}</prop>'
            )
        if p["section"]:
            block.append(f'      <prop type="section">{_xml.escape(p["section"])}</prop>')
        if p.get("section_label"):
            block.append(
                f'      <prop type="section-title">{_xml.escape(p["section_label"])}</prop>'
            )
            block.append(f'      <prop type="section-depth">{p.get("section_depth", 1)}</prop>')
        if p["page"] is not None:
            block.append(f'      <prop type="page">{p["page"]}</prop>')
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
