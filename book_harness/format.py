"""AI-assisted per-page formatting (M5).

Pure, deterministic parts of the stage:

* :func:`needs_format` -- decide whether a page reaches the model at all
  (task 5.3): it reports heuristic format defects (soft-wrapped paragraphs,
  broken enumerations, runs of blank lines, doubled spaces, glued image
  lines). Pages without defects never incur a model call.
* :func:`cosmetic_clean` -- deterministic whitespace fixes applied before
  the gate (and before the model sees the page).
* :func:`verify_format` -- guardrail (task 5.5): a formatted page may only
  differ from its source on a whitespace-insensitive basis plus the
  whitelisted structural edits (list markers, fences, image alt lines ...).
  Violations trigger one strict retry, then the original text is kept.

* :func:`format_book` -- the LLM stage (task 5.4): per-page model calls with
  a cache under ``08_format/cache``, archetype-specific prompt cards from
  ``config`` and verifier-enforced fallback.

The CLI wiring (task 5.6) builds on :func:`format_book`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from book_harness.pages import PageArchetype
from book_harness.config import (
    FORMAT_PROMPT_VERSION,
    PAGE_FORMAT_ARCHETYPES,
    PAGE_FORMAT_SYSTEM,
)
from book_harness.progress import work_bar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# needs_format heuristics (task 5.3)
# ---------------------------------------------------------------------------

_NEWLINE_RUN_RE = re.compile(r"\n{4,}")
_DOUBLE_SPACE_RE = re.compile(r"(\S)  +(\S)")
_NUM_MARKER_RE = re.compile(r"^\s*(\d{1,3}[.)]|[a-z][.)]|[أ-ي][.)])\s+\S")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_FENCE_RE = re.compile(r"^\s*```")
_SOFT_CHARS = ".!?:;…،؛؟\"'”’«»)]}"
#: Closing typographic chars that legitimately end a wrapped line.
_CHARACTER_CLOSERS = ")}\u00bb\u201d\u2019"


def cosmetic_clean(page_md: str) -> str:
    """Deterministic whitespace fixes that never need a model.

    Applied before the gate (and before the model sees the page): collapse
    runs of blank lines to ONE blank line, squeeze inner doubled spaces and
    strip trailing whitespace. Pure-whitespace edits are always verifier-safe.
    """
    text = page_md.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _DOUBLE_SPACE_RE.sub(r"\1 \2", text)
    return text


def needs_format(page_md: str) -> tuple[bool, list[str]]:
    """Should this page be sent to the formatting model?

    Returns ``(needs, reasons)``; ``reasons`` is empty for clean pages.
    """
    text = page_md.replace("\r\n", "\n")
    reasons: list[str] = []
    if _NEWLINE_RUN_RE.search(text):
        reasons.append("newline-run>=3")
    if _DOUBLE_SPACE_RE.search(text):
        reasons.append("double-space")
    # an image must stand alone on its line
    for m in _IMAGE_RE.finditer(text):
        if _line_of(text, m.start()).strip() != m.group(0):
            reasons.append("glued-image")
            break

    in_code = False
    soft = 0
    broken = 0
    body_len = 0
    body_lines = 0
    punct_lines = 0
    prev = ""
    for raw in text.splitlines():
        s = raw.strip()
        if _FENCE_RE.match(s):
            in_code = not in_code
            prev = s
            continue
        if in_code or not s:
            if s:
                prev = s
            continue
        marker = _NUM_MARKER_RE.match(s)
        if marker and prev and not _NUM_MARKER_RE.match(prev.strip()):
            end = prev.rstrip()[-1:]
            if end and end not in ":.!…؟،؛?":
                broken += 1
        if (prev and not marker
                and not (s[0] in "#>|!-" or s == "---")
                and len(prev.rstrip()) > 24
                and prev.rstrip()[-1] not in _SOFT_CHARS
                and not prev.rstrip()[-1].isdigit()
                and prev.rstrip()[-1] not in _CHARACTER_CLOSERS):
            soft += 1
        if s[0] not in "#>|!":
            body_lines += 1
            body_len += len(s)
            if s[-1] in _SOFT_CHARS:
                punct_lines += 1
        prev = s

    if soft >= 2 and not _entry_style(punct_lines, body_lines, body_len):
        reasons.append(f"soft-wrapped×{soft}")
    if broken:
        reasons.append(f"broken-list×{broken}")
    return (bool(reasons), reasons)


def _entry_style(punct_lines: int, body_lines: int, body_len: int) -> bool:
    """True for pages whose lines are separate entries (bibliography,
    reference lists, catalogues): almost no line ends with sentence
    punctuation and lines are short. Their line breaks are structure, not
    wrapping defects, so they must not trigger an unwrap pass (which could
    merge entries into paragraphs)."""
    if body_lines < 4 or not body_len:
        return False
    avg_len = body_len / body_lines
    return (punct_lines / body_lines) < 0.2 and avg_len <= 60


def _line_of(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start: end if end != -1 else len(text)]


# ---------------------------------------------------------------------------
# verify_format guardrail (task 5.5)
# ---------------------------------------------------------------------------

_FENCE_BODY_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_FOOTNOTE_DEF_RE = re.compile(r"\[\^\w+\]:")
_MATH_TOKEN_RE = re.compile(r"§\w+-?\d+§")


def _no_ws(text: str) -> str:
    """NFKC-normalise and drop every whitespace char."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def _count(text: str) -> tuple[int, int, int]:
    arabic = latin = digit = 0
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            arabic += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
        elif ch.isdigit():
            digit += 1
    return arabic, latin, digit


def verify_format(original: str, formatted: str) -> tuple[bool, list[str]]:
    """Is ``formatted`` a layout-only rewrite of ``original``?

    Checks (in order):

    1. fenced code bodies must be byte-identical multisets;
    2. image links, footnote definitions, ``§token§`` and ``$$`` counts must
       be unchanged;
    3. total letter count may only grow (lost list markers add letters such
       as ``a)``/``أ)``) and never shrink beyond a small rim;
    4. digit count stays within a small tolerance (list renumbering);
       auto-pass when the whitespace-insensitive texts are simply equal.

    Returns ``(ok, violations)``.
    """
    violations: list[str] = []

    if _no_ws(original) == _no_ws(formatted):
        return True, []

    if re.search(r"</?page\b[^>]*>", formatted, re.IGNORECASE):
        violations.append("wrapper-tag")

    if Counter(_FENCE_BODY_RE.findall(original)) != Counter(
            _FENCE_BODY_RE.findall(formatted)):
        violations.append("fenced-code-changed")

    if Counter(_IMAGE_RE.findall(original)) != Counter(
            _IMAGE_RE.findall(formatted)):
        violations.append("image-links-changed")

    for name, rx in (("footnotes", _FOOTNOTE_DEF_RE), ("tokens", _MATH_TOKEN_RE)):
        if Counter(rx.findall(original)) != Counter(rx.findall(formatted)):
            violations.append(f"{name}-changed")
    if original.count("$$") != formatted.count("$$"):
        violations.append("display-math-changed")

    or_a, or_l, or_d = _count(_no_ws(original))
    fm_a, fm_l, fm_d = _count(_no_ws(formatted))
    letters_o, letters_f = or_a + or_l, fm_a + fm_l

    floor = letters_o - max(3, letters_o // 100)
    ceil = letters_o + 40 + letters_o // 50
    if not floor <= letters_f <= ceil:
        violations.append(
            f"letters-drifted({letters_o}->{letters_f}; floor {floor} ceil {ceil})")
    if abs(fm_d - or_d) > 4 + or_d // 50:
        violations.append(f"digits-drifted({or_d}->{fm_d})")

    return (not violations), violations


# ---------------------------------------------------------------------------
# LLM stage (task 5.4)
# ---------------------------------------------------------------------------

def _cache_key(page_md: str, archetype: str, model: str,
               prompt_version: str = FORMAT_PROMPT_VERSION) -> str:
    blob = "\x00".join([page_md, archetype, model, prompt_version])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _build_messages(archetype: PageArchetype, page_md: str,
                    violations: list[str] | None = None) -> tuple[str, str]:
    """(system, user) pair: base card + archetype card + the page itself."""
    card = PAGE_FORMAT_ARCHETYPES[archetype.value]
    user = (
        f"{card}\n\n"
        "Format the page inside <page>...</page> below.\n"
        f"<page>\n{page_md}\n</page>\n\n"
        "Output only the formatted page markdown, WITHOUT the <page> "
        "markers themselves."
    )
    if violations:
        user += (
            "\n\nIMPORTANT: your previous output violated the rules "
            f"({'; '.join(violations)}). Re-output the page: layout changes "
            "ONLY, every word and every character of content preserved, and "
            "no <page> markers."
        )
    return PAGE_FORMAT_SYSTEM, user


_PAGE_TAG_RE = re.compile(r"</?page\b[^>]*>\s*", re.IGNORECASE)


def _sanitize(output: str) -> str:
    """Strip echoed <page>/</page> wrapper tags (models sometimes repeat
    the delimiters); the verifier reports residuals as violations too."""
    return _PAGE_TAG_RE.sub("", output).strip()


def format_page(client: Any, archetype: PageArchetype, page_md: str,
                violations: list[str] | None = None) -> str:
    system, user = _build_messages(archetype, page_md, violations)
    return _sanitize(client.generate(user, system=system) or "")


def format_book(
    book_md: str,
    archetypes: dict[int, PageArchetype],
    *,
    client: Any,
    out_path: Path,
    format_dir: Path,
    report_path: Path,
    model: str = "",
    prompt_version: str = FORMAT_PROMPT_VERSION,
    force: bool = False,
    pages_limit: tuple[int, int] | None = None,
) -> dict:
    """Format every page of ``book_md``; write ``out_path`` + report.

    Pipeline per page: deterministic ``cosmetic_clean`` -> gate
    (``needs_format``, unless ``force``) -> cached or fresh LLM attempt ->
    :func:`verify_format` (one strict retry, then fall back to the original
    page text). Only the final page texts replace ``book_md``.

    Returns a summary dict; the report (one jsonl record per page) also
    captures the cache key, archetype, reasons and verifier outcome.
    """
    from book_harness.pages import join_pages, split_pages

    pages, head = split_pages(book_md)
    if pages_limit:
        lo, hi = pages_limit
        pages = {p: t for p, t in pages.items() if lo <= p <= hi}
    fmt_dir = Path(format_dir)
    cache_dir = fmt_dir / "cache"
    pages_dir = fmt_dir / "pages"
    for d in (cache_dir, pages_dir, Path(report_path).parent):
        d.mkdir(parents=True, exist_ok=True)

    report: list[dict] = []
    formatted: dict[int, str] = {}
    counters = Counter()
    ordered = sorted(pages.items())
    with work_bar(ordered, desc="formatting", unit="page") as pbar:
        for pg, original in pbar:
            pbar.set_postfix_str(f"page {pg}")
            archetype = archetypes.get(pg, PageArchetype.DEFAULT)
            cleaned = cosmetic_clean(original)
            needs, reasons = needs_format(cleaned)
            rec: dict[str, Any] = {
                "page": pg, "archetype": archetype.value,
                "cosmetic": cleaned != original, "reasons": reasons,
                "cache": False, "fallback": None, "verifier_ok": True,
            }
            if force or needs:
                counters["model_pages"] += 1
                key = _cache_key(cleaned, archetype.value, model, prompt_version)
                cache_path = cache_dir / f"{key}.md"
                text = None
                if cache_path.exists():
                    text = cache_path.read_text(encoding="utf-8")
                    rec["cache"] = True
                    counters["cache_hits"] += 1
                else:
                    try:
                        text = format_page(client, archetype, cleaned)
                        ok, violations = verify_format(cleaned, text)
                        if not ok:
                            retry = format_page(client, archetype, cleaned,
                                                violations=violations)
                            ok2, _viol2 = verify_format(cleaned, retry)
                            if ok2:
                                text = retry
                                rec["retried"] = True
                    except Exception as exc:   # noqa: BLE001 - keep going
                        logger.warning("page %d: model call failed: %s", pg, exc)
                        rec["fallback"] = f"model-error: {exc}"
                    if not text or not text.strip():
                        text = cleaned
                        if not rec["fallback"]:
                            rec["fallback"] = "empty-output"
                    if not rec["fallback"]:
                        cache_path.write_text(text, encoding="utf-8")
            else:
                text = cleaned
                if rec["cosmetic"]:
                    counters["cosmetic_only"] += 1

            ok, _viol = verify_format(original, text)
            rec["verifier_ok"] = ok
            rec["changed"] = text.strip() != original.strip()
            if not ok and not rec.get("fallback"):
                rec["fallback"] = "verifier-violation"
                text = original
                rec["changed"] = text.strip() != original.strip()
            if rec.get("fallback"):
                counters["fallbacks"] += 1
            if rec["changed"]:
                counters["changed_pages"] += 1
            report.append(rec)
            formatted[pg] = text
            (pages_dir / f"page-{pg:03d}.md").write_text(
                text if text.endswith("\n") else text + "\n", encoding="utf-8"
            )
        pbar.set_description_str(
            "formatting ({mp} via model, {ch} cached, {fb} fallbacks)".format(
                mp=counters["model_pages"], ch=counters["cache_hits"],
                fb=counters["fallbacks"],
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(join_pages(formatted, head), encoding="utf-8")
    with open(report_path, "w", encoding="utf-8") as f:
        for rec in report:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "pages": len(report),
        "model_pages": counters["model_pages"],
        "cache_hits": counters["cache_hits"],
        "cosmetic_only": counters["cosmetic_only"],
        "changed_pages": counters["changed_pages"],
        "fallbacks": counters["fallbacks"],
        "out_path": str(out_path),
    }
    return summary, report
