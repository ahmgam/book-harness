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
    terms.py             # term extraction: GLiNER + n-gram mining + LLM adjudication
    glossary.py          # arabterm lookup, conflict policy, glossary.yaml writer
    chunker.py           # section-aligned chunking + inline placeholders
    translate.py         # Ollama/vLLM client, glossary injection, rolling context
    refine.py            # estimate → patch → verify loop
    qa.py                # QE scoring, compliance audit, consistency audit
    assemble.py          # markdown/HTML output, glossary appendix, manifest
    normalize.py         # Arabic text normalization (shared by enforcement + QA)
    state.py             # run state.json, resume logic, cache keys
    cli.py               # thin CLI over the library
  mcp/                   # (M3.5) MCP server wrapping glossary + translation
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
  02_glossary/        candidates.jsonl, glossary.yaml (versioned), audit.md
  03_chunks/          chunk-0001.json (block ids, source, placeholders, hash)
  04_drafts/          chunk-0001.md + .meta.json (model, prompt, glossary ver)
  05_refine/          chunk-0001.spans.json, patches.jsonl, verdicts.jsonl
  06_qa/              qe-scores.jsonl, compliance.jsonl, consistency.jsonl
  07_output/          book.md / book.html, glossary-appendix.md, manifest.json
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
