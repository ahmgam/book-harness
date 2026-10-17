"""Book-Harness: OCR JSON -> cited glossary -> translation -> markdown.

Pipeline stages (see PLAN.md, milestones M1..M3):

    00_source     OCR JSON pair -> consolidated Book model + extracted images
    (outline)     outline repair from section-numbering patterns
    02_glossary   term extraction -> arabterm lookup -> glossary.yaml
    03_chunks     section-aligned chunks with inline placeholders
    04_drafts     Ollama translation with glossary + rolling context
    06_qa         QE scores, compliance + consistency audits, OCR gate
    05_refine     estimate -> literal patch -> A/B verify refinement
    06_qa/drift   glossary drift mining -> uncertified candidates
    07_output     book.md + provenance manifest + glossary appendix,
                  bilingual HTML + QA report + TMX translation memory

Everything lands on disk; every stage can be re-run or resumed.
"""

__version__ = "0.1.0"

from book_harness.book import Book, Block, BlockType, OutlineNode, PageInfo, PageStat, TocEntry

__all__ = [
    "Book",
    "Block",
    "BlockType",
    "OutlineNode",
    "PageInfo",
    "PageStat",
    "TocEntry",
    "__version__",
]
