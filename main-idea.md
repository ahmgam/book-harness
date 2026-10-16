# Translation AI Workflow

## Phase 1 : Glossary generation
the workflow should be able to generate Glossary for the whole book, we can divide this part into two ideas
### 1. generating Glossary
we can do this programmatically using [`GLiNER`](https://github.com/urchade/GLiNER), which can generate it without AI aid.
### 2. getting certitifed translation
we can do this through project like [`arabterm`](https://forzagreen.github.io/arabterm/), so we can produce unified acceptable translation for scientific terms.
it might need to build special MCP to do this part
if there is multiple translation for the same term, we can let LLM choose from the available options.

## Phase 2 : Translation
in this phase, we will relay on specialized local model for translation, like `translategemma2`, maybe also `Qwen3.5-A3B` and `gemma4:26b`.
Claude recommended `Hy-MT2-30B-A3B`
it has two stages:
### 1. translation
book split into ~1200-char chunks per chapter; each chunk translated via Ollama with glossary injected + prior-paragraph context; glossary enforced post-hoc; results cached and streamed via SSE; resumable.
### 2. Refinement
3-phase guarded refinement per chunk:
   - **2a Estimate** (refinement model locates ≤12 error spans, MQM-style)
   - **2b Patch** (pure-Python literal substring replacements, no model)
   - **2c Verify** (verifier model judges A/B with swapped positions, twice; rejected patches keep the original draft — the model can never rewrite the whole chunk)
## Notes
1. every process should be stored in files, so no need to rerun any of them
2. progress of translation is also saved in files, to if the process interrupted, it can be resumed (track progress)
3. the output should be markdown or html (as original file)