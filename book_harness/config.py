"""Central configuration: paths, model names, prompt versions, policies.

All values are overridable through :class:`Settings` (the CLI builds one from
flags); module-level constants are the defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

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

#: Default serving backend is Ollama (native ``/api/generate``). A host URL
#: ending in ``/v1`` auto-selects the OpenAI-compatible chat-completions API
#: instead (vLLM / llama.cpp server / LM Studio …); override with
#: ``BOOK_HARNESS_API`` / the CLI ``--api`` flag.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.environ.get("BOOK_HARNESS_OLLAMA_TIMEOUT", "900"))
API_DEFAULT = os.environ.get("BOOK_HARNESS_API", "auto")  # auto | ollama | openai
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

GENERATE_OPTIONS = {"temperature": 0.0, "num_ctx": 8192, "seed": 42}

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
COMPLIANCE_FLOOR = 95.0

#: Hard cap on the number of chunks refined in one pass (None = budgetless;
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
STATE_VERSION = 1


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
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
    qa_report: bool = True

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
