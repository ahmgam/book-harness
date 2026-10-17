"""Backward-compatibility shim: the canonical Book model now lives inside the
package (``book_harness.book``). Existing scripts importing ``from book import
Book`` keep working unchanged."""

from book_harness.book import (  # noqa: F401
    CHROME_TYPES,
    Block,
    BlockType,
    Book,
    PageInfo,
    PageStat,
    TocEntry,
    OutlineNode,
    html_to_markdown,
    html_to_text,
)

if __name__ == "__main__":  # old CLI smoke test keeps working
    import sys
    from pathlib import Path

    book = Book.from_marker_json(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
    print(f"pages: {book.num_pages}, blocks: {len(book.blocks)}, toc: {len(book.toc)}")
