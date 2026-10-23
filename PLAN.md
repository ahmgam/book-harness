# Book-Harness — Implementation Plan

> Concrete build plan derived from [main-idea.md](./main-idea.md) (the idea), [brainstorming.md](./brainstorming.md) (the analysis), and [ocr-output-analysis.md](./ocr-output-analysis.md) (the OCR input contract).
> Pipeline is end-to-end from PDF: marker/datalab OCR → `book.py` consolidation → glossary → translation → refinement → assembly.

---

## 0. Guiding principles

1. **Library first.** The core is a Python package with pure file I/O. CLI, MCP server, and web UI are thin shells over it.
2. **Everything on disk.** Every stage writes versioned artifacts; any stage can be re-run or resumed from its files. Cache key per chunk = `hash(source_chunk + glossary_version + model + prompt_version)`.
3. **Structural passthrough.** Translatability is decided by block type from the OCR model, not heuristics. Equations, code, images, and table structure never reach a model.
4. **Cited glossary.** Every glossary entry carries provenance (dictionary, Wikidata, page, scan URL) and a status (`certified | llm_chosen | uncertified`).
5. **Bounded model edits.** The model never rewrites a whole chunk; refinement is estimate → literal patch → A/B verify.

---

## 1. Repository layout

```text
book-harness/
  book_harness/
    __init__.py
    config.py            # paths, model names, prompt versions, policies
    book.py              # (exists) OCR JSON pair → Book model
    profile.py           # Phase 0: domain, style, entity profiling
    terms.py             # tokenisation support + candidates.jsonl writer (drift.py shares the constants)
    llm_terms.py         # term extraction v2: small instruct model, one call per chunk (PLAN §8)
    glossary.py          # arabterm lookup, conflict policy, glossary.yaml writer
    chunker.py           # section-aligned chunking + inline placeholders
    translate.py         # Ollama/vLLM client, glossary injection, rolling context
    consistency.py       # consistency stage: one rendering per term (post-glossary LLM adjudication)
    refine.py            # estimate → patch → verify loop
    qa.py                # QE scoring, compliance audit, drift audit (consistency audit)
    assemble.py          # markdown/HTML output, glossary appendix, manifest
    normalize.py         # Arabic text normalization (shared by enforcement + QA)
    state.py             # run state.json, resume logic, cache keys
    cli.py               # thin CLI over the library
  mcp/                   # (M3.5) MCP agent workflow guide (AGENT.md); the server itself lives in book_harness/mcp_server.py to avoid shadowing the official `mcp` SDK package
  web/                   # (M4) review UI
  runs/<book-id>/        # per-book artifacts (see §4)
  tests/
  pyproject.toml
```

---

## 2. Milestones

### M1 — MVP: OCR → glossary → translation → markdown (idea validation)

**Goal:** one book in, translated markdown out, with a cited glossary. No refinement.

| # | Task | Details | Depends on |
|---|---|---|---|
| 1.1 | Package skeleton | `pyproject.toml`, `config.py` with model names + prompt versions, logging | — |
| 1.2 | Wire `book.py` into the package | Move/adopt existing `Book` class; expose `iter_content_blocks`, `outline`, `image_bytes` | — |
| 1.3 | Arabic normalizer | `normalize.py`: strip diacritics, unify alef/ya/ta-marbuta; unit tests on tricky cases | — |
| 1.4 | arabterm loader | Download `arabterm.db.gz`, gunzip, index `dictionary`/`term` tables; domain → dictionary whitelist mapping | 1.1 |
| 1.5 | Term extraction v1 | GLiNER (`urchade/gliner_multi-v2.1`) with custom labels + noun-phrase/n-gram TF-IDF mining; `candidates.jsonl` | 1.2 |
| 1.6 | Glossary builder v1 | Lookup candidates in arabterm (domain-filtered); deterministic conflict priority; LLM adjudication only on ties; `glossary.yaml` with citations + status | 1.4, 1.5 |
| 1.7 | Chunker v1 | Section-aligned chunks from outline tree; paragraph-boundary splits with char ceiling; inline placeholders inside `Text` blocks only; `chunk-NNNN.json` with block ids + hash | 1.2 |
| 1.8 | Translation client | Ollama client for Hy-MT2 (terminology prompt template) with fallback models; glossary injection; prior-paragraph + rolling chapter summary context; per-chunk `.meta.json` | 1.7 |
| 1.9 | Assembly v1 | Reassemble chunks by block id → `book.md`; images extracted to disk; page comments preserved | 1.8 |
| 1.10 | State + resume | `state.json` stage cursor; skip completed chunks by cache key | 1.8 |
| 1.11 | CLI | `book-harness run <ocr-json>` orchestrating 1.2→1.9 | all |
| 1.12 | Validation run | Full run on the Flight Simulation book; manual review of 3 chapters | 1.11 |

**Exit criteria:** translated `book.md` for the sample book; glossary with ≥80% certified coverage of extracted terms; resume works after `kill -9` mid-translation.

### M2 — Accuracy: refinement + QA

| # | Task | Details | Depends on |
|---|---|---|---|
| 2.1 | QE scoring | cometkiwi (or LLM-judge fallback) per chunk → `qe-scores.jsonl` | 1.8 |
| 2.2 | Refinement loop | 2a estimate (≤12 MQM spans, general instruct model) → 2b literal Python substring patches → 2c A/B verify with swapped positions ×2; rejected patches keep draft | 2.1 |
| 2.3 | QE-budgeted scheduling | Refine worst-scoring chunks first; skip chunks above quality threshold | 2.1, 2.2 |
| 2.4 | Compliance audit | Normalized Arabic glossary enforcement check per chunk → `compliance.jsonl`; violations re-queued for targeted re-translation | 1.3, 1.6 |
| 2.5 | Consistency audit | Book-level: same source term → one target form everywhere; drift report | 2.4 |
| 2.6 | Pre-translation OCR QA gate | `page_stats` + block sanity checks flag suspicious pages before model spend | 1.2 |
| 2.7 | Model bake-off | Same 30–50 chunks through Hy-MT2-7B / Hy-MT2-30B-A3B / TranslateGemma-12b / Qwen3.5-27B; score with COMET + compliance metric; pick empirically | 2.1 |

**Exit criteria:** refinement measurably improves QE scores on a held-out sample; compliance ≥95% on glossary terms; bake-off report with a chosen default model.

### M3 — Artifacts & robustness

| # | Task | Details | Depends on |
|---|---|---|---|
| 3.1 | Glossary appendix | Back-matter glossary: term, translation, source dictionary, page, Wikidata link | 1.6 |
| 3.2 | Bilingual HTML | Paragraph-aligned side-by-side edition from chunk pairs | 1.9 |
| 3.3 | QA report | Per chapter: QE distribution, compliance %, patch counts, uncertified terms | 2.x |
| 3.4 | Provenance manifest | Per chunk: model, prompt version, glossary version, timestamps | 1.10 |
| 3.5 | TMX export | Aligned pairs → translation memory file | 1.9 |
| 3.6 | Glossary drift loop | Mine draft for new recurring terms → `uncertified` → human confirm → re-run affected chunks only | 1.6, 1.8 |
| 3.7 | Outline repair | Re-derive heading depth from section-numbering patterns when OCR `h`-tags mis-nest | 1.2 |

### M3.5 — MCP server

| # | Task | Details |
|---|---|---|
| 3.5.1 | MCP tools | `lookup_term(en|ar, domain)`, `get_glossary(book_id)`, `translate_chunk(text, glossary, context)`, `list_runs` |
| 3.5.2 | Agent workflow doc | Prompt guide so Claude/Copilot can drive a full book translation through the tools |

### M4 — Web UI (last)

| # | Task | Details |
|---|---|---|
| 4.1 | Progress view | SSE streaming of stage/chunk progress from `state.json` |
| 4.2 | Review screen | Human review of `uncertified` terms and low-QE chunks; decisions write back to `glossary.yaml` / re-queue chunks |

### M5 — AI-aided post-processing per page archetype

> AI (local LLM, assist role) reformats each assembled page of `book.md`
> according to its **page archetype**, fixing text alignment, extra newlines,
> lost list elements and other format defects — with deterministic guardrails
> so wording never changes. Also runnable standalone on already-extracted runs.

**Design decisions:**

1. **Page archetype = derived concept.** The codebase has no page-level type
   (`book.py` only defines per-block `BlockType`). A new classifier maps each
   page's BlockType composition to one archetype.
2. **Placement: new `format` stage after `assemble`.** Operates on the final
   assembled Arabic `book.md`, split at the existing page comments
   (`<!-- ===== page N ===== -->`). Post-processing before translation would be
   reflowed anyway; after translation the formatting sticks.
   **Standalone mode:** the stage can also be run by itself on an
   already-extracted run directory (`book-harness format <run-dir>`) — it
   reads that run's `07_output/book.md` + `manifest.json` (and the OCR JSON
   pair only for page composition, falling back to heuristic composition when
   missing), reformatting in place with the same per-page cache and verifier.
   No translation, refinement, or QA stages are re-run.
3. **Base + per-archetype prompt cards** stored in `config.py`
   (`PAGE_FORMAT_SYSTEM`, `PAGE_FORMAT_ARCHETYPES`), with a separate
   `FORMAT_PROMPT_VERSION` so formatting runs don't invalidate translation
   caches.
4. **Assist models only** (`ASSIST_MODELS` list, temperature 0) — this is
   formatting, not translation.
5. **Bounded edits with verification** (mirrors principle 5): deterministic
   whitespace-insensitive structural diff per page; on violation → one retry
   with a stricter reminder → fall back to original page text + log entry.
6. **Skip clean pages:** deterministic `needs_format()` scorer gates which
   pages reach the model (same philosophy as QE-budgeted refinement).
7. **No reflow of RTL semantics:** formatting instructions are newline/
   structure only; no left/right alignment heuristics.
8. **Idempotent + standalone-runnable:** re-running formatting on an
   already-formatted page yields no changes (per-page cache + boundary
   re-split at page comments make it safe), and the stage works equally from
   the full pipeline or on its own against an existing run directory.

**Page archetype taxonomy** (first match by priority wins):

| Archetype | Trigger (BlockType mix on the page) | Priority |
|---|---|---|
| `front_matter` | book's first N pages, mostly short/Other blocks (cover, title, edition) | 1 |
| `toc` | contains `TableOfContents` | 2 |
| `index` | trailing pages: many short Text lines ending in page numbers | 3 |
| `list` | `ListGroup` present, or ≥3 list-item units / enumerated-line pattern | 4 |
| `footnote` | `Footnote` blocks present | 5 |
| `equation` | `Equation` present | 6 |
| `code` | `Code` present | 7 |
| `table` | `Table`/`TableGroup` present | 8 |
| `figure` | `Figure`/`Picture`/`FigureGroup` present | 9 |
| `narrative` | dominated by `Text`/`SectionHeader` | 10 |
| `default` | anything else / heavy mix | 11 |

**Instruction cards per archetype:**

*Base card (all pages):*
- You are a markdown layout formatter for an Arabic RTL technical book.
- Only layout/structure may change; never reword, translate, summarize,
  reorder blocks, or add content. Every word must survive (output is verified
  against the input on a whitespace-insensitive basis).
- Never touch: `§EQ0§`-style placeholder tokens, image markdown
  `![…](images/…)`, numeric content, anything inside fenced code.
- Paragraph = one line; exactly one blank line between blocks; headings in
  ATX (`#`) form, max one per line; single space after sentence punctuation.
- Freedom whitelist (only these): heading levels, `---` separators, bullet /
  ordered markers and indentation, `>` callouts, bold/italic emphasis that
  mirrors source emphasis.

*Narrative:* join hyphen-broken and soft-wrapped lines into single-line
paragraphs; collapse newline runs to one blank line; drop a leaked repeated
running header/footer line (repeating across neighboring pages); never merge a
paragraph into a following heading or list item.

*List:* reconstruct lost list structure — lines beginning `1.` `١.` `a)` `أ)`
`-` `•` `*` inside Text re-emitted as markdown list items (keep Arabic marker
text); renumber only consecutive ordered sequences; nesting = 2-space indent
per level, max 3; never merge items; never split an item onto a line after a
blank. Source-side `ListGroup`/item cues provided.

*Table:* pipe-tables: one row per line, uniform column count (escape stray
`|`), regenerate separator row, drop empty OCR column stubs, no cell wrapping.
HTML passthrough tables: verify only, structural edits forbidden.

*Figure:* image lines stand alone (nothing glued to them); captions = one
italic line immediately after their image; strip scan artifacts (stray `|`,
stray quote glyphs).

*Equation:* never edit, reorder, or split `§EQ…§` tokens or equation
references; only clean surrounding narrative spacing.

*Code:* wrap stray code fragments in fenced blocks with best-guess language;
preserve intra-fence spacing exactly (no unwrapping); inline code → backticks.

*TOC:* one entry per line; page numbers verbatim; consistent indentation per
level; leaders `……` normalized to a single `… N`; whole page as one
consistent list; original order preserved.

*Footnote:* definitions as `[^n]: text`, one per line; inline markers as
`[^n]`; never merge footnotes with body text.

*Front-matter:* title → `#`; subtitle/author/edition lines kept separate;
strip page-number/printer noise; preserve line emphasis.

*Index:* one entry per line; keep `term, sub-entry, number` pattern and
commas; original order preserved (no sorting).

*Default/mixed:* base card only + light cleanup.

**Tasks:**

| # | Task | Details | Depends on |
|---|---|---|---|
| 5.1 | Archetype classifier | New `book_harness/pages.py`: per-page BlockType composition (from `Book`) → archetype; `page_lines()` helper to split/merge `book.md` at page comments; unit tests on the Flight Simulation run | — |
| 5.2 | Prompt cards + config | `config.py`: `PAGE_FORMAT_SYSTEM` (base card), `PAGE_FORMAT_ARCHETYPES` dict, `FORMAT_PROMPT_VERSION = "fmt-v1"`; archetype-specific user-message builder | 5.1 |
| 5.3 | Needs-format scorer | Pure function `needs_format(page_md) -> tuple[bool, list[str]]` with reasons (line-break density in paragraphs, broken enumerations, newline runs ≥3, doubled spaces, glued image/caption lines); unit tests | 5.1 |
| 5.4 | Format stage (LLM calls) | `book_harness/format.py`: per-page call via existing `OllamaClient`/`OpenAIClient` with assist-model list; cache dir `08_format/cache/` keyed `hash(page_md + archetype + model + FORMAT_PROMPT_VERSION)`; writes `08_format/pages/page-NNN.md`; `state.json` stage `format` | 5.2, 5.3 |
| 5.5 | Guardrail verifier | `verify_format(original, formatted)`: NFKC-normalize, strip all whitespace → char-signature counts (Arabic/Latin letters, digits) must match; placeholders + image links + fenced-code bodies byte-identical; list-marker additions only tolerated whitespace-insensitively; violation → 1 retry → original fallback + log | 5.4 |
| 5.6 | CLI wiring | Add `format` to `STAGES` (after `assemble`); `--skip-format` / `--force-format` flags; report `09_qa/format-report.jsonl` (page, archetype, changed, reasons, violations); **standalone subcommand** `book-harness format <run-dir>` that locates the run's `07_output/book.md`, composition source and settings from `state.json`/manifest and runs only this stage (reformat in place; `--out` optional to write a copy instead) | 5.4, 5.5 |
| 5.7 | Validation run | Full run on the Flight Simulation book; manually review ~10 pages of at least 4 different archetypes; confirm lists restored, paragraphs unwrapped, zero content-loss verifier hits reaching the output silently | all |

**Exit criteria:** every processed page passes the structural verifier or fell
back to the original (logged); list pages show restored bullets; no placeholder
or image loss anywhere; resume works mid-format; running
`book-harness format <run-dir>` on a completed run produces identical results
to the in-pipeline stage (idempotency confirmed).

**Risks / mitigations:**

| Risk | Mitigation |
|---|---|
| Model rewrites content instead of formatting | Structural verifier + fallback (5.5) |
| Formatting echoed as content (verbatim regression) | Explicit "output only the formatted page" instruction + verifier catch (5.5) |
| Cost blowup across 500+ pages | `needs_format()` gate (5.3); assist-family models only (5.2) |
| LLM destructively repairs tables | Deterministic pipe-table repair preferred; LLM only when repair fails (5.2 table card) |

### M6 — Book-glossary harvesting (glossary pages → vocabulary candidates)

> If the source book already contains its own glossary (back-matter
> "Glossary" pages), those pages are a high-precision vocabulary list the
> author curated. Detect them, hand them to the assist model to extract the
> vocabulary terms listed on each page, and mix the resulting candidates into
> the normal vocabulary-extraction output before the glossary builder runs.
> The rest of the pipeline proceeds unchanged.

**Design decisions:**

1. **Detection is purely OCR metadata.** The OCR does not tag pages as
   glossary (no such block type or page category in the OCR contract), but
   its section structure does: a `SectionHeader` whose title names a
   glossary (whole-word match; a folio token from running heads such as
   "xxii Glossary" is stripped before matching, since OCR sometimes
   promotes running heads into headers). The section spans from that
   block's page until just before the next header of the same-or-higher
   level; overlapping/adjacent spans merge. No text-shape heuristics: a
   book without a glossary header yields no harvest (deliberate — a
   metadata-only contract is deterministic and explains itself).
2. **Extraction via the assist model, fail-fast.** Each glossary page's text
   is sent to the assist role (`_assist_client(settings)`, temperature 0 via
   `config.GENERATE_OPTIONS`) with a prompt asking for one term per line,
   verbatim, definitions omitted. If glossary pages were detected and the
   assist client cannot be built or any page call fails/timeouts, the `terms`
   stage raises and the pipeline stops (state.json records it; a re-run
   retries the stage). No silent degradation.
3. **Purely additive merge, no re-scoring.** After `terms.extract_candidates`
   finishes, the glossary-page term records
   (`{term, freq: 1, df: <glossary pages>, page: <glossary page>,
   score: GLOSSARY_PAGE_SCORE, signal: "book-glossary"}`) are appended and
   deduped against mined candidates on the NFKC-lower key (`terms.py`
   convention). Mined candidates are never re-scored.
4. **Everything downstream is unchanged.** The merged list goes into
   `candidates.jsonl`, so `GlossaryBuilder.build()`, arabterm lookup,
   conflict policy, LLM tie-break, `glossary.yaml` versioning, per-chunk
   cache keys, translation injection, and QA compliance all run normally on
   the union. Glossary pages are not assumed to carry Arabic equivalents —
   the standard lookup flow applies to these terms.
5. **Provenance artifact.** The raw glossary-page harvest is written separately
   (`02_glossary/glossary-page-terms.jsonl`, mirroring
   `drift-candidates.jsonl`) before the merge, and detection info lands in
   `state.json` / the source summary (`book_glossary_pages`, `page_terms`).

**Tasks:**

| # | Task | Details | Depends on |
|---|---|---|---|
| 6.1 | Glossary-page detector | `pages.py`: `_GLOSSARY_TITLE_RE` (whole-word `glossary`) folio-token stripper + `_GLOSSARY_FOLIO_TOKEN_RE`, new public `find_glossary_pages(book) -> list[range]` (ordered, merged contiguous ranges); unit tests: header span bounds, letter-group subheaders don't cut short, whole-word only, folio running heads, empty result | — |
| 6.2 | Page-term extraction | New `book_harness/glossary_pages.py`: `extract_page_terms(book, page_ranges, assist_client)` — page text via `book.iter_content_blocks(pages=...)`, one assist-model call per page (batchable), line parser (strip bullets/numbering, NFKC-lower dedupe), candidate-record builder; fail-fast on missing client / failed call | 6.1 |
| 6.3 | Stage wiring + config | `config.py`: `GLOSSARY_PAGE_TERMS_FILE`, `GLOSSARY_PAGE_SCORE`, `GLOSSARY_PAGE_SIGNAL`, `GLOSSARY_MAX_TERM_CHARS`, harvest prompt cards; `cli.stage_terms`: run detection → extraction → merge (dedupe) → write `candidates.jsonl` with the union + `glossary-page-terms.jsonl`; state/summary logging (`book_glossary_pages`, `page_terms`); no-op path when no pages detected | 6.2 |
| 6.4 | Validation run | Flight Simulation book: detector finds the front-matter Glossary (pages 16–19); assist-model spike harvests 145 terms with per-page call audit; full unit suite green | 6.3 |

**Exit criteria:** books without glossary pages take the untouched path (zero
behavior change); a book with a glossary yields page-extracted terms mixed
into `candidates.jsonl` with `signal: "book-glossary"`, all flowing through
the standard glossary build; missing/broken assist model stops the run
instead of silently dropping the harvest.

**Risks / mitigations:**

| Risk | Mitigation |
|---|---|
| OCR promotes running heads ("xxii Glossary") into SectionHeaders | Folio tokens stripped before title matching (6.1); validated on the Flight Simulation book |
| Assist model lists definitions instead of terms | Prompt fixes output shape (one term per line); parser rejects lines over `GLOSSARY_MAX_TERM_CHARS` |
| LLM output drift (paraphrased terms) | Verbatim-terms prompt at temperature 0; provenance artifact (6.3) makes review possible |
| OCR mashes several entries onto one line | The assist model reads the whole page and splits terms; per-page calls, not per-entry parsing |

---

## 3. Model & infrastructure decisions

| Concern | Decision | Fallback |
|---|---|---|
| Translation model | Hy-MT2-30B-A3B (Apache-2.0, official terminology template) | Hy-MT2-7B GGUF via Ollama; llama.cpp/vLLM for 30B |
| Refine/verify model | Qwen3.5-27B or gemma-4-26B-A4B-it (general instruct) | — |
| Term extraction | GLiNER multi-v2.1 (CPU-friendly) | gliner2 ecosystem |
| QE | cometkiwi-da (CC-BY-NC — personal use only) | LLM-judge |
| Terminology DB | arabterm.db (SQLite, direct queries) | MCP wrapper later |
| Serving | Ollama first; vLLM if 30B-A3B GGUF unavailable | — |
| Language | Python 3.11+, stdlib-first where possible | — |

**Early verification tasks (do first, cheap):**
- [ ] Confirm Hy-MT2-30B-A3B GGUF availability for Ollama; else plan vLLM.
- [ ] Download arabterm.db; verify domain→dictionary mapping covers the sample book's domain (aerospace/software).
- [ ] Run GLiNER on 3 sample chapters; measure term-candidate precision by hand on 50 candidates.

---

## 4. Run artifact layout (per book)

```text
runs/<book-id>/
  00_source/          OCR JSON pair (main + _meta), consolidated Book model, extracted images
  01_profile/         domain.json, style.md, entities.jsonl
  02_glossary/        candidates.jsonl (mined + book-glossary harvest), glossary-page-terms.jsonl (M6 provenance), glossary.yaml (versioned), audit.md
  03_chunks/          chunk-0001.json (block ids, source, placeholders, hash)
  04_drafts/          chunk-0001.md + .meta.json (model, prompt, glossary ver)
  05_refine/          chunk-0001.spans.json, patches.jsonl, verdicts.jsonl
  06_qa/              qe-scores.jsonl, compliance.jsonl, consistency.jsonl
  07_output/          book.md / book.html, glossary-appendix.md, manifest.json
  08_format/          pages/page-NNN.md, cache/ (per-page format cache)
  09_qa/              format-report.jsonl
  state.json          stage cursor for resume
```

---

## 5. Known caveats & proposed solutions

The analysis in [brainstorming.md](./brainstorming.md) surfaced several catches in the original idea. Each is listed here with the solution baked into the plan (task references in parentheses).

### Glossary phase

| # | Catch | Proposed solution | Where in plan |
|---|---|---|---|
| 1 | **GLiNER finds entities, not terms.** Zero-shot NER catches people/orgs/places, but book terminology is mostly multiword technical noun phrases ("reinforcement learning", "supply chain") that NER misses. | 3-signal extraction pipeline: GLiNER (with custom labels like `technical_term`) + noun-phrase/n-gram mining with TF-IDF rarity scoring + an LLM adjudication pass that merges, dedupes, and classifies candidates (term vs. name vs. noise). | 1.5, 1.6 |
| 2 | **Same word, different domains, different translations.** "Cell", "field", "interest" translate differently by domain; a flat lookup produces wrong terms. | **Phase 0 book profiling**: classify book/chapters into domains, then filter the arabterm lookup by a domain → dictionary whitelist (arabterm is already organized as separate dictionaries — military, finance, linguistics, legal, aerospace…). | 1.4, 1.6 |
| 3 | **A glossary pair is not a citation.** Without provenance, "certified" is unverifiable. | Every glossary entry is a cited artifact: source dictionary name, Wikidata ID, printed page, archive.org scan URL, plus `status` (`certified | llm_chosen | uncertified`) and `decided_by` (`auto | llm | human`). | 1.6 |
| 4 | **LLM conflict resolution is nondeterministic.** When dictionaries disagree, letting the model "choose" each time yields inconsistent priorities. | Deterministic conflict policy encoded up front (e.g., KSAA unified glossaries > Librairie du Liban > others); LLM choice only on ties, with the reason recorded. | 1.6 |
| 5 | **Uncertified terms degrade the glossary silently.** Scientific books always contain terms no dictionary covers. | Explicit uncertified-term policy: transliteration, keep-English-in-parentheses, or LLM coinage — all flagged `uncertified` for human review, never silently mixed in. | 1.6, 3.6 |
| 6 | **Arabic morphology breaks naive enforcement.** Prefixes/suffixes (ال، و، ب، ات، ين…) make literal string matching produce false violations and false compliance. | Shared Arabic normalizer (strip diacritics, unify alef/ya/ta-marbuta) + stem-aware matching (CAMeL Tools), used by both enforcement and the compliance audit. | 1.3, 2.4 |
| 7 | **The glossary drifts after translation starts.** Drafting reveals terms extraction missed. | Glossary freeze point (versioned artifact) before translation + one drift loop: mine the draft for new recurring terms → append as `uncertified` → human confirms → re-run only affected chunks. | 3.6 |
| 8 | **The book may already have its own glossary pages.** The author's own back-matter glossary is a curated, high-precision vocabulary list that naive n-gram mining overlooks as a signal. | Detect glossary pages from OCR metadata (whole-word glossary `SectionHeader` titles, folio-stripped), hand them to the assist model to list the terms on each page, and merge those candidates additively into `candidates.jsonl` (dedupe, signal `book-glossary`) before the glossary builder runs; fail-fast if the assist model is unavailable while glossary pages exist. | M6 (6.1–6.4) |

### Translation phase

| # | Catch | Proposed solution | Where in plan |
|---|---|---|---|
| 8 | **Fixed 1200-char chunks cut mid-sentence and mid-markdown.** | Chunk on structure: consecutive typed content blocks within one outline section, split at paragraph boundaries with a char ceiling, keyed by block id for exact reassembly. | 1.7 |
| 9 | **Models destroy non-translatables** — equations, code, numbers, citations, URLs. | Structural passthrough by block type (Equation/Code/images/tables never reach the model) + inline placeholder protection only inside `Text` blocks. | 1.7 |
| 10 | **Prior-paragraph context is too local.** The model doesn't know what the chapter is arguing. | Add a rolling chapter summary (one paragraph, updated every N chunks) alongside the prior paragraph. | 1.8 |
| 11 | **Style inconsistency across 500 chunks.** Without a style guide the book reads like inconsistent essays. | Extract a style profile from 2–3 sample chapters in Phase 0 (tone, person, terminology conventions) and inject it into every prompt. | 1.8 |
| 12 | **Hy-MT2 can translate but can't judge.** A dedicated MT model is not built for MQM span-locating or A/B verification. | Split roles: dedicated MT model (Hy-MT2 / TranslateGemma) translates; a general instruct model (Qwen3.5-27B / gemma-4-26B-A4B) does estimate and verify. | 2.2 |
| 13 | **Refinement is always ×3 cost.** Running estimate/patch/verify on every chunk wastes compute on already-good chunks. | Reference-free QE scoring (cometkiwi) per chunk; spend the expensive passes on worst-scoring chunks first. | 2.1, 2.3 |
| 14 | **Per-chunk QA misses book-level drift.** The same term can map to different forms in different chapters. | Cross-chunk consistency audit: one source term → one target form everywhere; drift reported as a QA metric. | 2.5 |

### OCR input

| # | Catch | Proposed solution | Where in plan |
|---|---|---|---|
| 15 | **`block["page"]` is unreliable** (only 210 distinct values for 510 pages). | Always use the id-derived page (`/page/N/...`); `book.py` already does this. | 1.2 |
| 16 | **Heading levels are occasionally inconsistent** (bare chapter-number headers, 4 empty `SectionHeader` blocks). | Outline repair: re-derive heading depth from section-numbering patterns (`N.` / `N.M.` / `N.M.K`) when OCR `h`-tags mis-nest. | 3.7 |
| 17 | **OCR noise propagates into translation** — garbage-in/garbage-out. | Pre-translation QA gate using `page_stats` (extraction method, block counts) + block-level sanity checks to flag suspicious pages before spending model calls. | 2.6 |
| 18 | **OCR table layouts rarely survive re-flowing** into Markdown pipe tables. | Keep table HTML as-is (round-trip safe); translate cell text only, never re-flow the markup. | 1.7, 1.9 |

### Operational

| # | Catch | Proposed solution | Where in plan |
|---|---|---|---|
| 19 | **Hy-MT2-30B-A3B GGUF availability in Ollama is unverified.** | Verify early (§3 checklist); fallbacks: Hy-MT2-7B GGUF, llama.cpp, or vLLM with FP8. | §3 |
| 20 | **QE model license (CC-BY-NC) blocks commercial use.** | Keep QE swappable; LLM-judge fallback. | 2.1 |
| 21 | **Interruption mid-run loses work.** | Everything-on-disk + `state.json` stage cursor; cache key per chunk = `hash(source_chunk + glossary_version + model + prompt_version)` so only affected chunks re-run. | 1.10 |

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 30B-A3B not available in Ollama | Verify GGUF early (§3); fallback to 7B or vLLM |
| Arabic morphology false positives in enforcement | Shared normalizer (1.3) with stem-aware matching; tune on real violations |
| QE model NC license blocks commercial use | LLM-judge fallback; keep QE swappable |
| OCR noise → garbage-in/garbage-out | Pre-translation QA gate (2.6); treat OCR quality as a gate |
| Glossary silently degrades on uncovered terms | Explicit uncertified policy + drift loop (3.6) |
| UI scope creep | M4 last; files-first core first |

---

## 7. Immediate next steps

1. Create package skeleton + `config.py` (1.1) and adopt `book.py` (1.2).
2. Implement the Arabic normalizer with tests (1.3).
3. Download arabterm.db and build the loader + domain mapping (1.4).
4. Run the GLiNER extraction spike on 3 chapters (1.5 spike).
5. Verify Hy-MT2-30B-A3B GGUF availability (§3 checklist).

---

## 8. Refactor: fully LLM-based glossary extraction (2026-10-20)

> Supersedes §1.5's deterministic extraction. The `terms` stage becomes LLM-driven:
> the book is chunked with the **existing chunker** and every chunk is sent to a
> **small instruct model** (selected via `BOOK_HARNESS_EXTRACT_MODELS`, e.g.
> `qwen3.5:4b`) which extracts the glossary terms itself. Nothing downstream
> (arabterm glossary build, translation, QA, drift, artifacts) changes.

### Design decisions

| Decision | Choice |
|---|---|
| Old signals | Both removed (n-gram mining + GLiNER) |
| Model selection | New role env `BOOK_HARNESS_EXTRACT_MODELS` / `BOOK_HARNESS_EXTRACT_MODEL` |
| Chunking | Reuse `chunker.plan_chunks()` inside the terms stage (in memory, not written to `03_chunks`) |
| Book-glossary harvest | Kept (`glossary_pages.py`); merged additively as today |
| Contract | Fail-loud: no model reachable → the terms stage stops (no silent n-gram fallback) |

### Steps

1. **`config.py`**: new `EXTRACT_MODELS` role (via `_env_models`, defaults
   `["qwen3.5:4b", "qwen2.5:3b-instruct", "llama3.2:3b", "qwen2.5:7b"]`),
   `EXTRACT_SIGNAL = "llm"`, scoring constants
   (`score = 1.0 + 0.25 * (freq_in_chunks - 1)`), word limit
   `EXTRACT_MAX_TERM_WORDS = 3` (terms, not phrases — enforced in the
   prompt **and** as a hard post-parse filter), and prompt cards
   `EXTRACT_SYSTEM` / `EXTRACT_USER` (`{section}` + `{text}`).
   The extract role also disables thinking (`EXTRACT_GENERATE_OPTIONS`
   with `think: false`; the Ollama client hoists `think` to the request top level).
2. **New module `book_harness/llm_terms.py`** (blueprint: `glossary_pages.py`):
   - `chunk_texts()` — `plan_chunks` → `(first_page, pages, section_title, text)`
     with `text = "\n".join(u.text for u in chunk.units)`.
   - `strip_think()` — removes `<think>…</think>` from reasoning models.
   - `extract_chunk_terms()` — one `generate()` per chunk → `parse_term_lines()`.
   - `build_candidates()` — candidate schema `{term, freq, df, page, score, signal="llm"}`,
     freq/df = number of reporting chunks, page = first chunk's min page, sorted `(-freq, term)`, capped.
   - `extract_candidates()` — orchestrator; exceptions → `RuntimeError`; empty replies on
     content chunks accepted (logged).
   - Reply cache `02_glossary/extract-cache.jsonl` keyed by
     `(chunk source_hash, model, prompt id)` so re-runs skip paid calls.
3. **Slim `terms.py`**: keep `RUBBER_WORDS` / `_WORD_RE` (imported by `drift.py`)
   and `write_candidates`; delete `mine_ngrams`, `GlinerExtractor`, `extract_candidates`.
4. **`cli.py`**: `_extract_client()` (never honours the MT `--model` override);
   `stage_terms` calls `llm_terms.extract_candidates(...)` then merges the
   book-glossary harvest as before; log line reflects the extract model.
5. **Cleanup**: remove the `gliner` pip extra, `.env.example` + `README.md` mention
   the new role; historical `PLAN.md` §1.5 / `brainstorming.md` stay as-is.
6. **Tests**: new `tests/test_llm_terms.py` (per-chunk calls, think-strip,
   aggregation, dedupe, cap, fail-loud, cache); update the `stage_terms` wiring
   test in `tests/test_glossary_pages.py`; `test_drift.py` stays green untouched.

### Risks / side effects

- **Glossary version churn**: new candidates → new `glossary.yaml` version → all
  chunk draft cache keys invalidate once after this refactor (by design).
- **Small-model quality**: temperature 0 + verbatim rule + char-capped parsing +
  arabterm certification filter downstream; drift mining still catches remnants.
- **Runtime**: ~1 LLM call per chunk (~1200 chars in, short output), bounded by
  the extraction cache on re-runs.
