"""Central configuration: paths, model names, prompt versions, policies.

All values are overridable through :class:`Settings` (the CLI builds one from
flags); module-level constants are the defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env auto-load (before every constant below reads os.environ)
# ---------------------------------------------------------------------------

#: Path of the `.env` file that was auto-loaded, if any (diagnostics).
DOTENV_LOADED: str | None = None


def _dotenv_candidate() -> str | None:
    """First existing `.env`: current working directory, then the project
    root (the directory above ``book_harness/``)."""
    root = Path(__file__).resolve().parent.parent
    for cand in (Path.cwd() / ".env", root / ".env"):
        try:
            if cand.is_file():
                return str(cand)
        except OSError:
            continue
    return None


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """One ``KEY=VALUE`` line → ``(key, value)``; ``None`` for blanks,
    comments and garbage.

    Tolerates shell-style ``export KEY=…``, a trailing ``# comment`` on
    unquoted values, and surrounding quotes. An empty value keeps the key
    (sets it to ``""``, mirroring shell semantics), and a leading ``~`` is
    home-expanded like ``source`` would do.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].lstrip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip()
    if not key or not (key[0].isalpha() or key[0] == "_"):
        return None
    if len(value) >= 2 and value[0] in "\"'":
        quote = value[0]
        close = value.find(quote, 1)
        value = value[1:close] if close > 0 else value[1:].strip()
    else:
        for idx, ch in enumerate(value):
            if ch == "#" and (idx == 0 or value[idx - 1] in " \t"):
                value = value[:idx].strip()
                break
    if value.startswith("~"):          # mirror `source`: ~ expands to the home
        value = os.path.expanduser(value)
    return key, value


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> str | None:
    """Populate ``os.environ`` from a ``KEY=VALUE`` file (no new dependency).

    ``path=None`` auto-detects the run's ``.env`` (cwd first, then the
    project root). Already-exported variables win unless ``override=True`` —
    an explicit export beats the file, mirroring shell sourcing. Values keep
    everything after the first ``=`` (API keys may contain ``=``); quotes
    are stripped. Returns the loaded path or ``None``.
    """
    global DOTENV_LOADED
    file = str(path) if path else _dotenv_candidate()
    if not file:
        return None
    try:
        text = Path(file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug(".env unreadable (%s): %s", file, exc)
        return None
    for raw in text.splitlines():
        pair = _parse_dotenv_line(raw)
        if pair is None:
            continue
        key, value = pair
        if override or key not in os.environ:
            os.environ[key] = value
    DOTENV_LOADED = file
    logger.debug(".env loaded: %s", file)
    return file


#: Module-import load point: every consumer of ``book_harness.config`` —
#: CLI, MCP server, library imports — has its ``.env`` applied before the
#: constants below are resolved.
DOTENV_LOADED = load_dotenv()

# ---------------------------------------------------------------------------
# Artifact layout (PLAN.md §4)
# ---------------------------------------------------------------------------

STAGE_DIRS = {
    "source": "00_source",
    "profile": "01_profile",
    "glossary": "02_glossary",
    "chunks": "03_chunks",
    "drafts": "04_drafts",
    "refine": "05_refine",
    "qa": "06_qa",
    "output": "07_output",
    "format": "08_format",          # M5: per-page formatting artifacts
    "format_qa": "09_qa",           # M5: format-report.jsonl
}

# ---------------------------------------------------------------------------
# Models / serving
# ---------------------------------------------------------------------------

TRANSLATION = "en"
TARGET = "ar"

#: Role-based model assignment (session-level env override).
#:
#: Roles map to pipeline *jobs*, not prompts:
#:
#: ================ ====================================================
#: Role             Used by
#: ================ ====================================================
#: ``translation``  MT work: chunks, drift-confirmed + compliance
#:                  re-translation, bake-off candidates, MCP translate
#: ``assist``       Refinement (MQM estimate / A/B verify) + rolling
#:                  chapter summaries
#: ``qe``           QE LLM-judge scoring + bake-off judge (``--judge``
#:                  still overrides)
#: ``format``       M5 per-page AI formatting (layout-only rewrite);
#:                  general instruct models, NEVER the MT model
#: ================ ====================================================
#:
#: Each role resolves its own preference list: the first entry installed on
#: the server wins (via ``/v1/models`` or ``/api/tags``). Precedence:
#: ``--model`` flag > role env var > role default (= the previous level's
#: list), so the three levels collapse to today's defaults when no env is
#: set: translation == assist defaults to the hard-coded list, QE defaults
#: to assist.
#:
#: Variables (plural first, a singular alias is accepted as a one-element
#: list):
#:
#: * ``BOOK_HARNESS_TRANSLATION_MODELS`` / ``BOOK_HARNESS_TRANSLATION_MODEL``
#: * ``BOOK_HARNESS_ASSIST_MODELS`` / ``BOOK_HARNESS_ASSIST_MODEL``
#: * ``BOOK_HARNESS_QE_MODELS`` / ``BOOK_HARNESS_QE_MODEL``
#: * ``BOOK_HARNESS_FORMAT_MODELS`` / ``BOOK_HARNESS_FORMAT_MODEL``
#: * ``BOOK_HARNESS_EXTRACT_MODELS`` / ``BOOK_HARNESS_EXTRACT_MODEL``
#:
#: Values: whitespace- or comma-separated Ollama names / server model IDs,
#: e.g. ``BOOK_HARNESS_TRANSLATION_MODELS="hy-mt2-30b-a3b, qwen2.5:14b"``.

_MODEL_DEFAULTS = ["hy-mt2-30b-a3b", "hy-mt2-7b", "qwen2.5:14b", "llama3.1:8b"]


def _env_models(plural: str, singular: str, fallback: list[str]) -> list[str]:
    """Preference list from env (comma/whitespace separated), else fallback."""
    raw = os.environ.get(plural) or os.environ.get(singular) or ""
    items = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
    return items or fallback


TRANSLATION_MODELS = _env_models(
    "BOOK_HARNESS_TRANSLATION_MODELS",
    "BOOK_HARNESS_TRANSLATION_MODEL",
    _MODEL_DEFAULTS,
)
ASSIST_MODELS = _env_models(
    "BOOK_HARNESS_ASSIST_MODELS",
    "BOOK_HARNESS_ASSIST_MODEL",
    TRANSLATION_MODELS,
)
QE_MODELS = _env_models("BOOK_HARNESS_QE_MODELS", "BOOK_HARNESS_QE_MODEL", ASSIST_MODELS)

#: M5 formatting role (task 5.6): general instruct models by preference,
#: with small generic instructs as fallback. Deliberately does NOT inherit
#: the translation list — a dedicated MT model should never be reused for
#: the layout-writing pass.
_FORMAT_MODEL_DEFAULTS = [
    "gemma4:26b",                   # 26B general instruct
    "k2-horizon",                   # 36B general instruct
    "qwen2.5:14b", "llama3.1:8b",   # generic instruct fallbacks
]
FORMAT_MODELS = _env_models(
    "BOOK_HARNESS_FORMAT_MODELS",
    "BOOK_HARNESS_FORMAT_MODEL",
    _FORMAT_MODEL_DEFAULTS,
)

#: Term-extraction role: the small instruct model that reads the book chunk by
#: chunk and lists the glossary terms itself (v2 extraction, PLAN §8). Small
#: models are fine — the output is a short term list, and the task runs once
#: per chunk. Never honours the MT ``--model`` override.
_EXTRACT_MODEL_DEFAULTS = [
    "qwen3.5:4b",
    "qwen2.5:3b-instruct", "llama3.2:3b", "qwen2.5:7b",
]
EXTRACT_MODELS = _env_models(
    "BOOK_HARNESS_EXTRACT_MODELS",
    "BOOK_HARNESS_EXTRACT_MODEL",
    _EXTRACT_MODEL_DEFAULTS,
)

#: Default serving backend is Ollama (native ``/api/generate``). A host URL
#: ending in ``/v1`` auto-selects the OpenAI-compatible chat-completions API
#: instead (vLLM / llama.cpp server / LM Studio …); override with
#: ``BOOK_HARNESS_API`` / the CLI ``--api`` flag.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.environ.get("BOOK_HARNESS_OLLAMA_TIMEOUT", "900"))
API_DEFAULT = os.environ.get("BOOK_HARNESS_API", "auto")  # auto | ollama | openai
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

#: Base generate options for every model role. ``think: False`` is pinned
#: here, not per-role: some assist-capable models (e.g. ``gemma4:26b``) run
#: a thinking channel that silently eats the context window when allowed —
#: with ``num_ctx: 4096`` the page-format prompt exhausted the window inside
#: thinking (``done_reason: length``, empty response). Tasks are
#: temperature-0 formatting/verdict missions, not reasoning; the Ollama
#: client hoists ``think`` to the request top level, OpenAI-compatible
#: servers ignore it (only ``OPENAI_OPTION_KEYS`` are forwarded).
GENERATE_OPTIONS = {
    "temperature": 0.0, "num_ctx": 4096, "seed": 42, "think": False,
}

# ---------------------------------------------------------------------------
# Prompt versions (bump to invalidate the chunk cache)
# ---------------------------------------------------------------------------

PROMPT_VERSION = "m1-v1"

#: Hy-MT2 ships an official terminology template: "`X` translates to `Y`"
#: lines. Used verbatim in the glossary section of the prompt.
GLOSSARY_TEMPLATE = "`{en}` translates to `{ar}`"

SYSTEM_PROMPT = (
    "You are a professional technical translator from English to Arabic. "
    "Translate the user text into formal Modern Standard Arabic, keeping the "
    "technical register. Rules:\n"
    "1. Use the supplied terminology glossary exactly — every listed term must "
    "appear in the output in its given Arabic form.\n"
    "2. Never translate placeholder tokens (§EQ0§, §URL0§, §CIT0§, …); copy "
    "them through unchanged.\n"
    "3. Do not add explanations, notes, or the source text. Output only the "
    "Arabic translation.\n"
    "4. Keep numbers, units, and equation references as in the source."
)

#: Non-dictionary English stopwords for n-gram mining.
STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from
    had has have how if in into is it its may might must no not of on or our
    shall should since so some such than that the their them then there these
    they this those through to was we were what when which while who whose
    will with within without would you your via figure table appendix chapter
    section note notes eq fig section sectionref above below each both other
    more most only same one two three first second third new used using use
    also however thus therefore whereas during between because both""".split()
)

# ---------------------------------------------------------------------------
# Quality estimation + refinement (M2: tasks 2.1 / 2.3 / 2.4)
# ---------------------------------------------------------------------------

#: Prompt versions for the M2 roles (bump to invalidate QE/refine artifacts).
QE_PROMPT_VERSION = "qe-m2-v1"
REFINE_PROMPT_VERSION = "refine-m2-v1"

#: Chunks scoring at/above this QE score are skipped by refinement (2.3).
#: Score scale is 0-100 regardless of scoring method.
QUALITY_THRESHOLD = 80.0

#: Default QE scorer selection: "auto" prefers cometkiwi when the optional
#: stack is installed, falling back to the LLM judge.
QE_METHOD_DEFAULT = "auto"

#: Chunks below this compliance rate (%) are re-queued for targeted
#: re-translation (2.4).
COMPLIANCE_FLOOR = 95.0#: Hard cap on the number of chunks refined in one pass (None = budgetless;
#: worst-scoring chunks go first either way).
REFINE_BUDGET: int | None = None

#: Maximum error spans the estimator may report per unit (blockquote budget).
REFINE_MAX_SPANS = 12

#: MQM categories the estimator is asked to use (regex-agnostic, display only).
MQM_CATEGORIES = (
    "terminology",
    "accuracy",
    "fluency",
    "terminology/consistency",
    "numbers",
    "style",
    "punctuation",
)

# ---------------------------------------------------------------------------
# AI page formatting (M5: tasks 5.2 / PLAN.md milestone 5)
# ---------------------------------------------------------------------------

#: Bump to invalidate the per-page format cache (independent of the
#: translation PROMPT_VERSION).
FORMAT_PROMPT_VERSION = "fmt-v2"

#: Base instruction card (system prompt) shared by every archetype. The
#: archetype-specific card from PAGE_FORMAT_ARCHETYPES is appended in the
#: user message. See PLAN.md M5 "Instruction cards".
PAGE_FORMAT_SYSTEM = (
    "You are a markdown layout formatter for an Arabic RTL technical book.\n"
    "You receive ONE page of the book. Fix only its layout and structure.\n"
    "Rules:\n"
    "1. Never reword, translate, summarize, reorder blocks, or add content. "
    "Every word must survive: the output is verified against the input on a "
    "whitespace-insensitive basis and violations are discarded.\n"
    "2. Never touch: placeholder tokens (§EQ0§, §URL0§, §CIT0§, …), image "
    "markdown ![…](images/…), numeric content, or anything inside fenced "
    "code blocks.\n"
    "3. When nearly every line is a distinct short entry (bibliography, "
    "reference list, credits, author lists, revision history), keep one "
    "entry per line: never merge entries into paragraphs.\n"
    "4. A paragraph is one line. Exactly one blank line separates blocks. "
    "Headings use ATX form (#). One space after sentence punctuation.\n"
    "5. The only structural changes allowed: heading levels, --- separators, "
    "bullet/ordered markers and their indentation, > callouts, and "
    "bold/italic emphasis that mirrors emphasis in the source.\n"
    "6. Output only the formatted page markdown — no notes, no fences "
    "around the whole page, no explanations."
)

#: One instruction card per page archetype (task 5.1's classifier output).
PAGE_FORMAT_ARCHETYPES: dict[str, str] = {
    "narrative": (
        "Page archetype: narrative body text.\n"
        "- Join hyphen-broken and soft-wrapped lines into single-line "
        "paragraphs.\n"
        "- Collapse runs of newlines to exactly one blank line.\n"
        "- Drop a leaked repeated running header/footer (the same line "
        "repeating across neighboring pages).\n"
        "- Never merge a paragraph into a following heading or list item."
    ),
    "list": (
        "Page archetype: lists. Reconstruct list structure the OCR lost.\n"
        "- Lines beginning with 1. / ١. / a) / أ) / - / • / * inside body "
        "text are list items: re-emit them as markdown list items. Keep "
        "Arabic marker text as written (e.g. '1.' in Arabic RTL order).\n"
        "- Renumber only consecutive ordered sequences.\n"
        "- Nested items: 2-space indent per level, maximum 3 levels.\n"
        "- Never merge two items; never split an item across a blank line."
    ),
    "table": (
        "Page archetype: tables.\n"
        "- Pipe tables: one row per line, uniform column count (escape "
        "stray | with \\|), regenerate the separator row, drop empty OCR "
        "column stubs. Cell text never wraps across lines.\n"
        "- HTML passthrough tables: leave the structure exactly as-is."
    ),
    "figure": (
        "Page archetype: figures/pictures.\n"
        "- Image lines ![…](…) stand alone: no text glued to them.\n"
        "- Captions: a single italic line directly after their image.\n"
        "- Strip scan artifacts (stray | or stray quote glyphs)."
    ),
    "equation": (
        "Page archetype: equations.\n"
        "- Never edit, reorder, or split placeholder/equation tokens or "
        "equation references; only clean the surrounding narrative spacing."
    ),
    "code": (
        "Page archetype: code.\n"
        "- Wrap stray code fragments in fenced blocks with a best-guess "
        "language tag; inline code in backticks.\n"
        "- Preserve spacing inside fences EXACTLY (no unwrapping)."
    ),
    "toc": (
        "Page archetype: table of contents.\n"
        "- One entry per line, page numbers verbatim.\n"
        "- Consistent indentation per level; leader dots (……) normalized "
        "to a single '… N' suffix; the whole page as one consistent list.\n"
        "- Original order preserved; never sort."
    ),
    "footnote": (
        "Page archetype: footnotes.\n"
        "- Definitions: [^n]: text, one per line; inline markers [^n].\n"
        "- Never merge footnotes with body text."
    ),
    "front_matter": (
        "Page archetype: front matter (cover/title/edition).\n"
        "- Title as a heading level; subtitle/author/edition lines kept "
        "separate.\n"
        "- Strip page-number and printer noise; preserve line emphasis."
    ),
    "index": (
        "Page archetype: index.\n"
        "- One entry per line; keep the 'term, sub-entry, number' pattern "
        "and commas.\n"
        "- Original order preserved; never sort."
    ),
    "default": (
        "Page archetype: mixed. Apply only the base rules; light cleanup."
    ),
}


# ---------------------------------------------------------------------------
# Chunking policy (task 1.7)
# ---------------------------------------------------------------------------

CHAR_CEILING = 1200          # char ceiling per chunk
MIN_CHUNK_CHARS = 200        # merge dwarfs into the previous chunk up to this
SUMMARY_PERIOD = 4           # refresh rolling chapter summary every N chunks

# ---------------------------------------------------------------------------
# arabterm (tasks 1.4 / 1.6)
# ---------------------------------------------------------------------------

ARABTERM_URLS = [
    "https://github.com/forzagreen/arabterm/raw/main/arabterm.db.gz",
]

#: Deterministic conflict priority (PLAN.md catch #4): domain-specific and
#: KSAA-unified glossariess outrank the grand "Librairie du Liban"-style
#: multi-dictionaries. Suffixes are matched case-insensitively against the
#: arabterm ``name_tech`` slug.
DICTIONARY_PRIORITY = [
    ("ksaa", 520),
    ("aerospace", 500),
    ("jordan_computing", 480),
    ("scs_information_technology", 470),
    ("at_information_tech", 460),
    ("damascus", 420),
    ("unified", 400),               # المعجم الموحد (at_*) unified glossaries
    ("at_", 400),                   # ...same family matched by slug prefix
    ("infosec", 380),
    ("dstt", 340),                  # Science & Technology Terms vol.1
    ("khatib_scientific_technical", 300),   # Librairie du Liban
    ("military", 280),
    ("electropedia", 280),
    ("arabeyes", 260),
]
FALLBACK_DICTIONARY_PRIORITY = 100

#: domain -> dictionary whitelist. Keys are arabterm ``name_tech`` patterns.
#: ``"*"`` means ungated (no whitelist), i.e. every dictionary is trusted.
DOMAIN_DICTIONARIES = {
    "aerospace": [
        "aerospace", "military", "electronic_warfare", "remote_sensing",
        "meteorology", "physics", "mathematics", "dstt", "khatib",
        "damascus",
    ],
    "software": [
        "information_tech", "information_technology", "computing", "arabeyes",
        "infosec", "electropedia", "electrical_engineering", "siwar_sdaia",
        "damascus", "dstt", "khatib", "mathematics",
    ],
    "*": None,
}

#: Which extracted-term sources count as "book domain" for the lookup.
DEFAULT_DOMAINS = ("aerospace", "software")

# ---------------------------------------------------------------------------
# Run state / cache
# ---------------------------------------------------------------------------

STATE_FILE = "state.json"
GLOSSARY_FILE = "glossary.yaml"
CANDIDATES_FILE = "candidates.jsonl"
GLOSSARY_PAGE_TERMS_FILE = "glossary-page-terms.jsonl"

#: Extraction-reply cache (LLM extraction v2, PLAN §8): one line per chunk
#: call, keyed by chunk source hash + model + prompt id; re-runs skip paid calls.
EXTRACT_CACHE_FILE = "extract-cache.jsonl"
STATE_VERSION = 1

# ---------------------------------------------------------------------------
# Book-glossary harvesting (task 6.2/6.3)
# ---------------------------------------------------------------------------

#: Fixed score for terms harvested from the book's own glossary pages.
#: The author curated this list, so it outranks every mined/NER signal
#: (n-gram scores average ≲ 3; GLiNER-only finds use 8.0).
GLOSSARY_PAGE_SCORE = 10.0

#: Candidate record signal for book-glossary harvests; mixed into
#: ``candidates.jsonl`` additively (deduped, no re-scoring of mined terms).
GLOSSARY_PAGE_SIGNAL = "book-glossary"

#: Reject model replies whose "term" lines exceed this many characters —
#: they are definitions or sentences, not terms.
GLOSSARY_MAX_TERM_CHARS = 60

# ---------------------------------------------------------------------------
# Term extraction v2 (fully LLM-based; PLAN §8)
# ---------------------------------------------------------------------------

#: Candidate record signal for LLM chunk extraction — the only extraction
#: signal; mixed with ``book-glossary`` harvest records additively.
EXTRACT_SIGNAL = "llm"

#: Candidate score from chunk frequency: ``EXTRACT_SCORE_BASE +
#: EXTRACT_SCORE_STEP * (freq_in_chunks - 1)``. A term reported by exactly one
#: chunk scores 1.0 and ranks below book-glossary records
#: (``GLOSSARY_PAGE_SCORE`` = 10.0).
EXTRACT_SCORE_BASE = 1.0
EXTRACT_SCORE_STEP = 0.25

#: Default candidate cap for the LLM extraction (same as v1).
EXTRACT_MAX_CANDIDATES = 400

#: Maximum number of words an extracted term may span — terms, not phrases:
#: "biopsy" and "excisional biopsy" are fine, a 4+ word run is a phrase and
#: gets rejected (prompt rule + hard post-parse filter).
EXTRACT_MAX_TERM_WORDS = 3

#: System card for the chunk term extraction (extract role, temperature 0).
EXTRACT_SYSTEM = (
    "You extract vocabulary terms from text of a technical book.\n"
    "The text may contain OCR noise; read through it and identify the "
    "technical terms, methods, components and domain vocabulary it uses.\n"
    "A term is vocabulary a specialist translator of this book would look up "
    "in a domain dictionary: jargon, method names, component or concept "
    "names of the book's field. Ordinary words a layperson already knows are "
    "NOT terms — do not list them (wrong: of, the, with, house, money, "
    "work). An everyday word qualifies only if this book clearly uses it as "
    "domain jargon with a special meaning.\n"
    "Output is verified downstream: be precise and literal, never invent, "
    "translate or paraphrase anything.\n"
)

#: User-message template: one chunk's section title + placeholder-protected text.
EXTRACT_USER = (
    "The following text was OCR-extracted from one chunk of a technical "
    "book (section: {section}). List the vocabulary terms it contains.\n"
    "\n"
    "Rules:\n"
    "1. One term per line — nothing else on the line.\n"
    "2. Terms must be at most {max_words} words long (1 to {max_words} "
    "words). Never extract a longer phrase or a full heading.\n"
    "3. Terms only: no definitions, explanations, translations, or "
    "annotations.\n"
    "4. Write each term exactly as it appears in the text (verbatim "
    "spelling, capitalization preserved); fix only obvious OCR typos.\n"
    "5. No bullets, numbering, or markdown formatting.\n"
    "6. Technical/domain vocabulary only — no everyday English words "
    "(function words like \"of\" or \"the\", and common general words like "
    "\"hair\" or \"money\" are never terms).\n"
    "7. If the chunk contains no vocabulary terms, output nothing.\n"
    "\n"
    "Example — for a chunk discussing flight-control software that "
    "mentions the transport aircraft's sidewall stick, a correct reply "
    "has exactly the lines:\n"
    "sidewall stick\n"
    "flight envelope\n"
    "fly-by-wire\n"
    "\n"
    "Chunk text:\n"
    "\n"
    "{text}\n"
)

#: Prompt-card identity recorded in the extraction cache; bump when the
#: prompt templates above change so cached replies are invalidated.
EXTRACT_PROMPT_ID = "extract-v3"

#: Generate options for the extract model. ``think`` is disabled: the task is
#: a short verbatim term list, not reasoning. The Ollama client hoists the
#: ``think`` key out of ``options`` into the request top level
#: (``/api/generate`` semantics); OpenAI-compatible servers ignore it
#: (only ``OPENAI_OPTION_KEYS`` are forwarded).
EXTRACT_GENERATE_OPTIONS = {**GENERATE_OPTIONS, "think": False}

# ---------------------------------------------------------------------------
# Technical-term focus filters (PLAN §9)
# ---------------------------------------------------------------------------

#: Zipf ceiling (``wordfreq`` scale, 0-8) above which a single-word candidate
#: counts as everyday English rather than jargon. Real-data check: everyday
#: words cluster ≳5.0 ("hair" 5.08, "model" 5.13, "data" 5.34, "money" 5.64)
#: while genuine jargon sits below ~4.5 ("fuselage" 3.09, "psoriasis" 2.89);
#: borderline academic words ("theory" 4.88, "analysis" 4.91) survive the
#: deterministic gate and get judged contextually by the LLM verify pass.
EXTRACT_ZIPF_FILTER = 5.0

#: Zipf ceiling for the rare-token rule inside multi-word phrases: a phrase
#: passes the commonness gate if at least one token sits below this value.
EXTRACT_ZIPF_RARE = 4.8

#: Chunk-frequency escape hatch for the zipf gate: a single word whose zipf is
#: above ``EXTRACT_ZIPF_FILTER`` is still kept when reported by at least this
#: many chunks — pervasive, book-wide use is the domain-jargon signal
#: ("hair" in a dermatology book).
EXTRACT_JARGON_MIN_FREQ = 4

#: Minimum chunk frequency for a candidate to survive aggregation below the
#: top-N cut. 1 keeps the current behavior (all chunk-reported terms).
EXTRACT_MIN_FREQ = 1

#: System card for the candidate verification pass (assist role,
#: temperature 0): the field-agnostic expertise filter that knows whether a
#: word is jargon *in this book*.
VERIFY_SYSTEM = (
    "You judge whether listed words are technical vocabulary of the book "
    "they were extracted from.\n"
    "A technical term is vocabulary a specialist translator would look up "
    "in a domain dictionary: jargon, method names, component or concept "
    "names of the book's field. Ordinary everyday words are not terms, "
    "unless the book clearly uses them as domain jargon with a special "
    "meaning (then they are terms).\n"
    "Use each word's chunk frequency and the sections it appears in: a word "
    "used pervasively across the book's sections is likely domain jargon.\n"
    "Output is verified downstream: decide for every listed word, never "
    "invent, translate or paraphrase anything.\n"
)

#: User-message template: one verification batch — one numbered word per
#: line with its frequency and sections; the model answers with one word
#: per line under the same numbering.
VERIFY_USER = (
    "Below is a list of vocabulary words extracted from one book. For each "
    "word, report whether it is a technical term of this book.\n"
    "\n"
    "Rules:\n"
    "1. Answer with one word per line — exactly the listed word, then\n"
    "   \"keep\" or \"drop\" separated by a colon (example: "
    "\"flight envelope: keep\").\n"
    "2. Answer for every listed word, in the same order.\n"
    "3. \"keep\": domain jargon, a method/component name, or an everyday "
    "word clearly used with a special meaning in this book.\n"
    "4. \"drop\": everyday vocabulary, function words, generic phrases — "
    "words the book's field does not treat as terminology.\n"
    "\n"
    "Words (word: frequency in chunks | sections):\n"
    "\n"
    "{words}\n"
)

#: Prompt-card identity recorded in the verification cache.
VERIFY_PROMPT_ID = "verify-v1"

#: Verification replies cache file (same layout as ``extract-cache.jsonl``).
VERIFY_CACHE_FILE = "verify-cache.jsonl"

#: Candidates per verification call — keeps one call's output within a small
#: model's safe output length.
VERIFY_BATCH_SIZE = 100

#: Generate options for the verification pass. ``think`` is disabled: the
#: task is a short per-line verdict list, not reasoning.
VERIFY_GENERATE_OPTIONS = {**GENERATE_OPTIONS, "think": False}

#: System card for the book-glossary harvest (assist role, temperature 0).
GLOSSARY_HARVEST_SYSTEM = (
    "You extract vocabulary terms from one page of a technical book's own "
    "glossary section.\n"
    "The text may contain OCR noise; read through it and list the terms it "
    "defines.\n"
    "Output is verified downstream: be precise and literal, never invent, "
    "translate or paraphrase anything.\n"
)

#: User-message template: the OCR text of one glossary page only.
GLOSSARY_HARVEST_USER = (
    "The following text was OCR-extracted from one page of a book's own "
    "glossary section. List the vocabulary terms it contains.\n"
    "\n"
    "Rules:\n"
    "1. One term per line — nothing else on the line.\n"
    "2. Terms only: no definitions, explanations, translations, or "
    "annotations.\n"
    "3. Write each term exactly as it appears on the page (verbatim "
    "spelling, capitalization preserved); fix only obvious OCR typos.\n"
    "4. No bullets, numbering, or markdown formatting.\n"
    "5. If the page contains no glossary entries, output nothing.\n"
    "\n"
    "Glossary page text:\n"
    "\n"
    "{text}\n"
)


# ---------------------------------------------------------------------------
# Occurrence capture + translation selection (context-informed glossary)
# ---------------------------------------------------------------------------

#: Per-match context window: characters kept on each side of a found term
#: when documenting its occurrences in ``candidates.jsonl`` (record field
#: ``occurrences``) and replayed to the select model.
OCCURRENCE_CONTEXT_CHARS = 150

#: Cap on documented occurrences per term — keeps the selection prompt
#: bounded for terms that saturate a book (e.g. "software").
OCCURRENCES_MAX_PER_TERM = 8

#: System card for the glossary consistency stage (assist role, temperature
#: 0): pick among certified atomic dictionary renderings using the book's own
#: usage, traced to the chunks the term occurs in.
CONSISTENCY_SYSTEM = (
    "You choose which certified Arabic translation fits an English technical "
    "term as it is used in a book.\n"
    "Several terminology dictionaries certify different Arabic renderings; "
    "the occurrences show — with chunk references — how the book actually "
    "uses the term.\n"
    "Ground decisions in the supplied occurrences, not prior preference.\n"
)

#: User-message template: term, numbered dictionary options, then documented
#: occurrences (chunk id + section + page + context window). The model
#: replies with the chosen option number only.
CONSISTENCY_USER = (
    "English technical term: {term}\n"
    "Book domain: {domains}\n"
    "\n"
    "Certified Arabic translations from terminology dictionaries:\n"
    "{options}\n"
    "\n"
    "Chunks of this book where the term occurs (with surrounding context):\n"
    "{occurrences}\n"
    "\n"
    "Which translation best fits the term's usage in these chunks?\n"
    "Reply with the number of the chosen translation only."
)

#: Prompt-card identity recorded in the consistency cache; bump when the
#: prompt templates above change so cached decisions are invalidated.
CONSISTENCY_PROMPT_ID = "consistency-v1"

#: Override check for a *single-option* certified entry: several dictionaries
#: agreeing is common, a wrong *field idiom* is rarer — but the dictionary
#: wins on citation, not on fit. The assist model re-audits a lone certified
#: rendering against the book's occurrences and either keeps it or overrides
#: it with a domain-fitting rendering. The model replies ``keep`` or
#: ``override to `ARABIC RENDERING` ``.
CONSISTENCY_OVERRIDE_SYSTEM = (
    "You audit one dictionary-certified Arabic translation against how the "
    "book actually uses the term.\n"
    "Terminology dictionaries certify forms for their own field (aviation, "
    "military, general technology); a certified form can be idiomatic there "
    "and wrong in this book's domain.\n"
    "Judge only from the supplied occurrences; when in doubt, keep the "
    "certified form.\n"
)

#: User-message template for the override check: the single certified form
#: + its dictionary, then documented occurrences (chunk id · section · page +
#: context window). The reply is ``keep`` or ``override to `ARABIC RENDERING` ``.
CONSISTENCY_OVERRIDE_USER = (
    "English technical term: {term}\n"
    "Book domain: {domains}\n"
    "\n"
    "Dictionary-certified Arabic translation:\n"
    "1. `{form}` (source: {source})\n"
    "\n"
    "Chunks of this book where the term occurs (with surrounding context):\n"
    "{occurrences}\n"
    "\n"
    "If the certified translation fits the term's usage in these chunks, "
    "reply exactly: keep\n"
    "If it does not fit — for example it is a calque from another field — "
    "override it and reply exactly: override to `ARABIC RENDERING`\n"
    "with a single, domain-fitting Arabic rendering for this book."
)

#: Prompt-card identity for the override check; cached in the same
#: consistency-cache.jsonl under its own key (options + this prompt id).
CONSISTENCY_OVERRIDE_PROMPT_ID = "consistency-override-v1"

#: Generate options for the consistency model. ``think`` is disabled like the
#: extraction card: the task is picking a numbered option, not reasoning.
CONSISTENCY_GENERATE_OPTIONS = {**GENERATE_OPTIONS, "think": False}

#: Consistency decision cache: one line per decided term, keyed by term +
#: options hash + model + prompt id; re-runs replay decisions without
#: paying for the same model call twice.
CONSISTENCY_CACHE_FILE = "consistency-cache.jsonl"

#: Auditable decision log of the consistency stage: one line per decided
#: term (term, options presented, choice, and how it was reached).
CONSISTENCY_DECISIONS_FILE = "consistency-decisions.jsonl"


def setup_logging(level: str = "INFO") -> None:
    # Log through tqdm.write when progress bars are active so log lines land
    # above the bars instead of clobbering them (book_harness.progress falls
    # back to plain stderr when tqdm is missing).
    try:
        from book_harness.progress import TqdmLoggingHandler

        stream_handler: logging.Handler = TqdmLoggingHandler()
    except ImportError:                            # pragma: no cover
        stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[stream_handler],
    )


@dataclass
class Settings:
    """Runtime settings resolved from defaults + CLI flags."""

    run_root: Path = Path("runs")
    ocr_file: Path | None = None
    model: str | None = None                 # force a translation model
    ollama_host: str = OLLAMA_HOST
    ollama_timeout: int = OLLAMA_TIMEOUT
    api: str | None = None                   # auto (default) | ollama | openai
    api_key: str = OPENAI_API_KEY            # Bearer key for openai endpoints
    domains: tuple[str, ...] = DEFAULT_DOMAINS
    char_ceiling: int = CHAR_CEILING
    char_floor: int = MIN_CHUNK_CHARS
    glossary_limit: int | None = None        # cap on glossary entries
    max_chunks: int | None = None            # cap on chunks translated
    pages: tuple[int, int] | None = None     # inclusive page range for a smoke run
    summary_period: int = SUMMARY_PERIOD
    prompt_version: str = PROMPT_VERSION
    translate: bool = True
    log_level: str = "INFO"
    skip_stages: list[str] = field(default_factory=list)
    # --- M2 (refinement + QA) ---
    qe_method: str = "auto"                     # auto | judge | comet | none
    refine: bool = True
    requeue: bool = True                        # re-translate compliance violators
    quality_threshold: float = QUALITY_THRESHOLD
    compliance_floor: float = COMPLIANCE_FLOOR
    refine_budget: int | None = REFINE_BUDGET
    max_spans: int = REFINE_MAX_SPANS
    # --- M3 (artifacts + drift) ---
    no_drift: bool = False                      # skip the glossary drift-mining stage
    no_artifacts: bool = False                  # skip appendix/bilingual/TMX/QA report
    appendix: bool = True
    bilingual: bool = True
    tmx: bool = True
    translated_html: bool = True                # page-divided print edition (from TMX)
    qa_report: bool = True
    # --- M5 (AI formatting pass) ---
    no_format: bool = False
    force_format: bool = False
    format_pages: tuple[int, int] | None = None

    @property
    def book_id(self) -> str:
        return slugify(self.ocr_file.stem if self.ocr_file else "book")

    def run_dir(self) -> Path:
        return self.run_root / self.book_id

    def stage_dir(self, run: Path | None = None, stage: str = "output") -> Path:
        base = (run or self.run_dir()) / STAGE_DIRS[stage]
        base.mkdir(parents=True, exist_ok=True)
        return base


def slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:64] or "book"
