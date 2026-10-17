# OCR Output Analysis — *Flight Simulation Software* (David Allerton)

Analysis of the two JSON artifacts produced by the marker/datalab OCR pipeline,
plus the proposed consolidation into a single data model (`book.py`).

## 1. The files

| File | Size | Role |
|---|---|---|
| `_David_Allerton_Flight_Simulation_Software_Design,_Development_and.json` | 19 MB | Main content: pages + content blocks |
| `_David_Allerton_Flight_Simulation_Software_Design,_Development_and_meta.json` | 479 KB | Pipeline metadata: TOC, page stats, debug path |

Both come from the same OCR run and are complementary halves of one dataset.

## 2. Main JSON structure

```json
{
  "page_info": {
    "0": { "bbox": [0, 0, 504, 720], "polygon": [[0,0],[504,0],[504,720],[0,720]] },
    "...": "… 510 pages total (keys 0–509) …"
  },
  "blocks": [
    {
      "id": "/page/0/Picture/0",
      "block_type": "Picture",
      "html": "<p><img src='/page/0/Picture/0'></p>",
      "page": 26,                                  // ← unreliable, see §4
      "polygon": [[...], ...],
      "bbox": [159.86, 0.0, 497.7, 360.56],
      "section_hierarchy": { "1": "/page/0/SectionHeader/2" },
      "images": { "/page/0/Picture/0": "<base64 JPEG>" }
    }
  ]
}
```

### 2.1 `page_info` — 510 pages

- Keys `0`–`509`, one entry per PDF page.
- Each page is 504 × 720 pt (`bbox` + corner `polygon`).
- This is the **authoritative page grid** of the document.

### 2.2 `blocks` — 4,908 merged content blocks

Every block has an identical key set:
`id, block_type, html, page, polygon, bbox, section_hierarchy, images`.

Block type counts:

| Type | Count | Notes |
|---|---:|---|
| `Text` | 2058 | Paragraphs |
| `PageHeader` | 1371 | Repeated running heads (rendering chrome) |
| `Code` | 373 | `<pre>` listings |
| `Equation` | 336 | `<math>` containing **LaTeX source** |
| `SectionHeader` | 307 | `h1`–`h4` headings; the book's outline |
| `FigureGroup` | 168 | Figure + caption merged; child image lives in the group's `images` |
| `ListGroup` | 124 | `<ul>/<li>` |
| `PictureGroup` | 57 | |
| `Table` | 27 | `<table>` html |
| `TableGroup` | 22 | Caption + table merged |
| `PageFooter` | 17 | Chrome |
| `TableOfContents` | 16 | Printed TOC of the book itself |
| `Figure` / `Caption` / `Picture` | 12 / 11 / 4 | |
| `Footnote` | 4 | |
| `Form` | 1 | |

Key facts:

- **`html`** holds the rendered content. `<math>` elements carry LaTeX
  (e.g. `\frac{d^2y(t)}{dt^2} + ...` with `\tag{2.1}` equation numbers).
- **`images`** maps an image *id* → inline base64 JPEG (magic bytes `\xff\xd8\xff\xe0`).
  241 images total. Child figures referenced by `src='...'` inside a
  `FigureGroup`/`PictureGroup` exist **only in that group's `images` map**,
  not as separate blocks — the image index must be built globally, per group.
- **`section_hierarchy`** maps hierarchy level (`"1"`, `"2"`, `"4"`) → the id of
  the nearest `SectionHeader` ancestor. All **12,120 references resolve** —
  no dangling links — so every block can be placed in the chapter/section tree.

## 3. Meta JSON structure

```json
{
  "table_of_contents": [
    { "title": "Flight Simulation Software", "heading_level": null,
      "page_id": 0, "polygon": [[...]] }
    // 307 entries
  ],
  "page_stats": [
    { "page_id": 0, "text_extraction_method": "surya",   // or "pdftext"
      "block_counts": [["Line", 13], ["Span", 7], ...],
      "block_metadata": { "llm_request_count": 0, ... } }
    // 510 entries
  ],
  "debug_data_path": "debug_data/_David_Allerton_..."
}
```

- **`table_of_contents`** (307 entries = exactly the 307 `SectionHeader` blocks)
  is a *flat* title → `page_id` list. `heading_level` is always `null`, so the
  hierarchy must be derived from the `h1..h4` tags in the `SectionHeader` html
  (11 × h1, 115 × h2, 177 × h4, 4 empty).
- **`page_stats`** is diagnostic only: raw *pre-merge* surya/pdftext block counts
  (`Line`, `Span`, …) and LLM usage counters. Useful for QA, not for rendering.
- **`debug_data_path`** points to pipeline debug output (not present here).

## 4. How the files relate (join keys)

```
page_info keys (0..509)  ←→  block id prefix "/page/N/..."  ←→  toc[].page_id
```

- The meta TOC aligns with the block id page numbering, verified by content:
  the "Preface" `SectionHeader` sits on id-page 12 and the TOC entry
  "Preface" has `page_id: 12`.
- `block["page"]` is **unreliable** — a many-to-one artifact: only **210 distinct
  values** for 510 pages (range 6–990), inconsistent with the TOC. It should be
  ignored in favour of the id-derived page.
- Reading order: within a page, the block list order is correct (top→down,
  e.g. id-page 12 lists PageHeader → SectionHeader → Text×5 → ListGroup by
  increasing y). Across pages, however, the list is **not ordered**, so blocks
  must be grouped by id-page and each group kept in list order.

## 5. Consolidation — `book.py`

A single standard-library-only `Book` class merges both files into one model
with rendering support.

### Data model

- `Block` — normalized block: `id`, `type` (enum), `html`, canonical `page`
  (derived from the id prefix), `bbox`, `polygon`, `section_hierarchy`,
  `images`, `heading_level` (for `SectionHeader`, parsed from `h1..h4`).
- `PageInfo` — page index, width, height.
- `TocEntry` — title, page, matched heading level, polygon.
- `OutlineNode` — node of the section tree: block, parsed section number
  (`1.4`, `2.3.1`, …), title, children.
- `PageStat` — per-page QA statistics from the meta file.

### Construction

```python
from book import Book
book = Book.from_marker_json("..._.json")   # locates the _meta.json automatically
```

- Canonical page = `int(block["id"].split("/")[2])`; the `page` field is ignored.
- Indexes built once: blocks by page (reading order preserved), blocks by id,
  a **global image index** (so child images inside groups are addressable), and
  the nested outline tree (empty `SectionHeader` artifacts skipped).
- Meta TOC entries are matched back to their `SectionHeader` blocks to recover
  heading levels.

### Access API

| Call | Purpose |
|---|---|
| `book.page(n)` | Content blocks of page *n* in reading order |
| `book.block(id)` | Block lookup by id |
| `book.image_bytes(image_id)` | Decoded JPEG bytes (covers group children) |
| `book.section_of(block, level)` | Nearest `SectionHeader` ancestor via `section_hierarchy` |
| `book.outline` / `book.print_outline()` | Nested chapter/section tree |
| `book.toc` | Flat TOC with recovered levels |
| `book.iter_content_blocks(...)` | All content, chrome (`PageHeader`/`PageFooter`) excluded by default, optional page range |

### Rendering

Markdown (`render_markdown` / `save_markdown`):
- HTML→Markdown conversion: LaTeX extracted from `<math>` as `$…$` / `$$…$$`,
  tables preserved as HTML, code blocks fenced, lists converted, bold/italic.
- Image references resolved to files on disk (`save_images_to=…`) or omitted;
  page boundaries kept as `<!-- ===== page N ===== -->` comments.
- `PageHeader`/`PageFooter` dropped unless requested.

HTML (`render_html` / `save_html` / `render_page_html`):
- One styled document: TOC `<nav>` linking to per-page `<section id="pN">`
  elements, per-block `<div>`s typed by `block_type`.
- Images as file refs, `data:` URIs (single self-contained file), or omitted.

### Verified on this book

- 510 pages, 4,908 blocks, 307 TOC entries.
- Full render: `book.md` (~1.2 MB) + `book.html` (~1.6 MB) + 241 JPEGs extracted.
- Outline, e.g.: *1 Design of an Engineering Flight Simulator → 1.1 The Evolution
  of Flight Simulation → 1.8.3 Analogue-to-Digital Conversion …*, chapters 1–10
  plus front matter (Preface, Glossary) and back matter (References, Appendices).

## 6. Known caveats (inherent to the OCR source)

1. `block["page"]` is junk; always use the id-derived page.
2. Heading levels are occasionally inconsistent (a bare "1" chapter-number
   header appears as its own section; four `SectionHeader` blocks have empty
   html and are skipped in the outline).
3. A block-level fallback could re-derive heading depth purely from the section
   numbering pattern (`N.` vs `N.M.` vs `N.M.K`) for a cleaner hierarchy if the
   OCR's `h`-tag levels mis-nest.
4. Table html is kept as-is in Markdown output (round-trip safe) since OCR
   table layouts rarely survive re-flowing into Markdown pipe tables.
