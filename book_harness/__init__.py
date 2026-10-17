"""Book-Harness: OCR JSON -> cited glossary -> translation -> markdown.

Pipeline stages (see PLAN.md, milestone M1):

    00_source     OCR JSON pair -> consolidated Book model + extracted images
    02_glossary   term extraction -> arabterm lookup -> glossary.yaml
    03_chunks     section-aligned chunks with inline placeholders
    04_drafts     Ollama translation with glossary + rolling context
    07_output     reassembled book.md + manifest

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
