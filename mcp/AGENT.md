# Agent workflow guide — driving a book translation through MCP (task 3.5.2)

The MCP server (`book-harness-mcp`) exposes four tools so an agent
(Claude, Copilot, or any MCP client) can drive or extend a book translation
without touching the pipeline code. This document is the prompt guide for
such an agent: connect to the server, then follow the workflow below.

Every tool returns a JSON text block; on failure you get `isError: true`
with an `error: …` text block — read it, fix the arguments, retry.

> **Everything on disk.** The pipeline and the MCP server share one contract
> (`PLAN.md` §4). Runs live in `runs/<book-id>/` (override the location with
> env `BOOK_HARNESS_RUNS` for the server process). Long work is never kept
> in the conversation: stages write artifacts, tools read them.

## The tools

| Tool | When to use | Returns |
|---|---|---|
| `lookup_term(term, lang, domain)` | before coining a translation; terminology questions | certified dictionary hits **with citations** (dictionary, Wikidata ID, printed page, archive.org scan) |
| `get_glossary(book_id, term?)` | at the start of every task on a book; when a chunk needs term pairs | glossary `version` + cited entries with `status` (`certified \| llm_chosen \| uncertified`) |
| `translate_chunk(text, glossary, context, kind?)` | one paragraph/heading/caption at a time | `{model, kind, translation}` |
| `list_runs()` | first — discover what exists | per-run stage statuses, chunk/translated counts, glossary version |

Rules an agent must respect:

1. **Never invent terminology.** For any recurring technical term, use
   `lookup_term` first. If the certified hit exists, use it *exactly* as the
   Arabic form given. If there is no hit, mark the pair `uncertified` in
   your notes — do not silently certify your own coinage (PLAN.md catch #5).
2. **Never translate whole chapters in one call.** `translate_chunk` is for
   one *unit* of text (a paragraph, heading, caption or list item).
3. **Preserve non-translatables.** Chunk unit texts contain placeholder
   tokens (`§EQ0§`, `§URL0§`, `§CIT0§`) protecting equations, URLs and
   citations. Copy them through the translation untouched; if a token is
   lost, the paragraph must be retried (catch #9).
4. **Keep provenance.** When you write a draft, record (see step 4) model,
   prompt version, glossary version and the chunk's cache key, so resume
   works (catch #21).

## Workflow A — translate a prepared book

Prerequisite: the *sourcing* stages ran once on the machine (they need the
OCR JSON, GLiNER/n-gram extraction and arabterm):

```bash
book-harness run <main-ocr-json> --no-translate     # prepare chunks + glossary
```

That creates `runs/<book-id>/{02_glossary,03_chunks}/…`. Then the agent does
**all the translation** entirely through MCP:

### 1. Discover the run(s)

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
 "params": {"name": "list_runs", "arguments": {}}}
```

Pick the run with `chunks > 0`. Note `glossary.version` — it goes into every
draft metadata. If a run already has `translated` drafts, stop there and
treat the run as partially complete: resume at the first chunk whose
`.meta.json` is missing (or whose `cache_key` differs), not at the top.

### 2. Load the glossary

```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call",
 "params": {"name": "get_glossary", "arguments": {"book_id": "control-systems"}}}
```

Keep the entries list (or just the `version`). Each chunk only needs the
pairs whose English form actually appears in it — pass the whole list to
`translate_chunk` and it filters, or filter yourself to keep calls small.
Dictionary-hit terms can also be fetched lazily with
`lookup_term("flight envelope", "en", "aerospace")`.

### 3. Translate chunk by chunk, maintaining context

For each `run_root/03_chunks/chunk-NNNN.json` (ascending order), for each
object in its `units` array:

- `unit.text` — placeholder-protected English source; pair with
  `unit.kind` (`text | title | caption | list_item`) as the `kind`
  argument;
- `glossary` — the pairs from step 2 (or a narrowed subset);
- `context` — either the *previous translated paragraph* or the rolling
  chapter summary. Post-M2 runs also store a per-chapter summary at
  `04_drafts/chapter-summary.json`; use it as `context` when set
  (catches #10, #11).

```json
{"jsonrpc": "2.0", "id": 10, "method": "tools/call",
 "params": {"name": "translate_chunk", "arguments": {
   "text": "The lift equation relates CL to the wing loading.",
   "glossary": [{"en": "lift coefficient", "ar": "معامل الرفع"},
                {"en": "wing loading", "ar": "الحمل الجناحي"}],
   "context": "Previous paragraph translation:\nيعرف بحر الطيران ..."
 }}}
```

Sanity-check each result: placeholders all present, Arabic text non-empty,
the injectable in-context forms used. Then **write the units back** into
`runs/<book-id>/04_drafts/chunk-NNNN.{md,meta.json}` — use the object
returned by `translate_chunk`, and render the list of translated units:

```python
meta = {
  "chunk_id": "chunk-0007",
  "cache_key": hash("…source…\x1fglossary_version\x1fmodel\x1fprompt_version"),
  "model": result["model"], "prompt_version": "m1-v1",
  "glossary_version": "<from step 2>", "section": "<chunk.section_title>",
  "pages": chunk["pages"], "units": [{"uid": u["uid"], "kind": u["kind"],
            "number": u.get("number", ""), "text": <Arabic text>}],
  "finished": "<iso-timestamp>",
}
```

The key ingredients: `cache_key = sha256(source \\x1f glossary_version \\x1f
model \\x1f prompt_version)` where `source` = JSON dump of
`{"section": chunk["section_title"], "units": [u["text"] …]}` — exactly
`book_harness.state.cache_key(...)`. On a later run the draft with a
matching key short-circuits; a changed glossary version automatically
invalidates only the affected chunks.

### 4. Assemble

When all chunks have drafts, tell the user to run:

```bash
book-harness run <main-ocr-json>          # translate stage resumes: skipped
```

Translation is a no-op (fresh cache keys), and the pipeline continues with
QE → refinement → compliance → assembly, producing `book.md`, the bilingual
edition, TMX, QA report and manifest. The agent's drafts are identical in
shape to the pipeline's, so nothing downstream changes.

## Workflow B — answer terminology questions / bilingual review

* "What does X mean in domain D?" → `lookup_term("X", "en", "D")`; cite the
  dictionary + page/scan URL in the answer.
* "How is X translated in the *Control Systems* run?" →
  `get_glossary("control-systems", "X")`; report the status: `certified`
  (dictionary-cited), `llm_chosen` (tie-broken, reason recorded), or
  `uncertified` (needs human review — recommend `book-harness serve` for the
  review UI).

## Configuration

| Env var | Effect |
|---|---|
| `BOOK_HARNESS_RUNS` | runs root for `get_glossary` / `list_runs` (default `./runs`) |
| `OLLAMA_HOST` | model server base URL for `translate_chunk` (default `http://localhost:11434`); a URL ending in `/v1` switches to the OpenAI-compatible wire protocol |
| `BOOK_HARNESS_API` | force the wire protocol: `auto` (default) \| `ollama` \| `openai` |
| `OPENAI_API_KEY` | Bearer key sent to OpenAI-compatible endpoints |
| `BOOK_HARNESS_OLLAMA_TIMEOUT` | seconds per generate call (default 900) |

An agent **can** act as a human reviewer for `uncertified` terms only in the
sense of proposing pairs (see `book-harness drift … --confirm`) — final
confirmation stays a human CLI step through this milestone.
