"""Term extraction v1 (task 1.5).

Two signals, later merged by the glossary builder:

1. **n-gram mining** (always available): word n-grams (2..4) extracted from
   translatable text, stopword-filtered, scored by TF-IDF-style rarity across
   blocks/sections. Frequent phrases that are rare elsewhere in the book are
   term candidates.
2. **GLiNER** (optional, task spike): ``urchade/gliner_multi-v2.1`` with custom
   labels (``technical_term``, ``method_name``). Requires the ``gliner`` extra
   and network access to fetch the model; gracefully disabled otherwise.

Output: ``candidates.jsonl`` — one JSON line per candidate:
``{term, freq, df, pages, score, signal}``.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from book_harness.book import Book, BlockType, html_to_text
from book_harness.config import STOPWORDS

# ---------------------------------------------------------------------------
# GLiNER adapter (optional)
# ---------------------------------------------------------------------------

GLINER_LABELS = ["technical term", "method name", "process", "equipment"]

#: Multi-word phrases mined from the text are also what GLiNER would tag as
#: entities; these entity labels are kept *out* of the custom-sequence pass.
GLINER_ENTITY_LABELS = ["person", "organization", "location"]


class GlinerExtractor:
    """Thin adapter over ``gliner``; ``available`` False when unsupported."""

    def __init__(self, model_name: str = "urchade/gliner_multi-v2.1") -> None:
        self.model_name = model_name
        self._model = None
        self.available = False
        try:
            from gliner import GLiNER  # type: ignore

            self._model = GLiNER.from_pretrained(model_name)
            self.available = True
        except Exception:  # not installed / model fetch failed / no torch
            self._model = None

    def extract(self, texts: list[str], threshold: float = 0.45) -> list[dict]:
        if not self.available:
            return []
        results: list[dict] = []
        batch = 16
        for i in range(0, len(texts), batch):
            for labels, kind in ((GLINER_LABELS, "gliner"), (GLINER_ENTITY_LABELS, "gliner-entity")):
                spans = self._model.predict_entities(
                    texts[i:i + batch], labels, threshold=threshold,
                )
                for entity in spans:
                    results.append({
                        "term": entity["text"].strip(),
                        "label": entity.get("label", ""),
                        "signal": kind,
                    })
        return results


# ---------------------------------------------------------------------------
# n-gram mining
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][\w'\-/]*")


#: Position-independent verb/junk words: an n-gram containing any of these
#: anywhere is sentence filler, not a term ("care is needed", "then applied").
RUBBER_WORDS = frozenset(
    """be been being has have had do does did can may will must might care
    needed needs makes made such this that these there then thus very quite
    often given shown followed way hence also become becomes some almost much
    see seen maybe rarely usually where while""".split()
)


def _is_junk_gram(gram: list[str]) -> bool:
    return any(w in RUBBER_WORDS for w in gram)


def _doc_text(block) -> str:
    return html_to_text(block.html)


def mine_ngrams(
    book: Book,
    *,
    min_n: int = 2,
    max_n: int = 4,
    min_freq: int = 2,
    min_df: int = 2,
    max_candidates: int = 400,
) -> list[dict[str, Any]]:
    """Stopword-filtered n-gram mining with TF-IDF-style rarity scoring."""
    docs: list[tuple[int, list[str]]] = []   # (page, word list)
    for b in book.iter_content_blocks():
        if b.type not in {
            BlockType.TEXT, BlockType.SECTION_HEADER, BlockType.CAPTION,
            BlockType.FOOTNOTE, BlockType.LIST_GROUP,
        }:
            continue
        words = _WORD_RE.findall(_doc_text(b).lower())
        if words:
            docs.append((b.page, words))
    if not docs:
        return []

    n_docs = len(docs)
    freq: Counter[str] = Counter()
    df: Counter[str] = Counter()
    first_page: dict[str, int] = {}
    for page, words in docs:
        grams = set()
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                gr = words[i:i + n]
                if gr[0] in STOPWORDS or gr[-1] in STOPWORDS:
                    continue
                if all(w in STOPWORDS for w in gr):
                    continue
                if _is_junk_gram(gr):
                    continue
                if not any(len(w) > 2 and w not in STOPWORDS for w in gr):
                    continue
                gram = " ".join(gr)
                freq[gram] += 1
                grams.add(gram)
                if gram not in first_page:
                    first_page[gram] = page
        for gram in grams:
            df[gram] += 1

    # split each doc's indexed text is already lowercased; keep canonical casing
    # from the original text for pretty output.
    canonical: dict[str, str] = {}
    for b in book.iter_content_blocks():
        if b.type not in {
            BlockType.TEXT, BlockType.SECTION_HEADER, BlockType.CAPTION,
            BlockType.FOOTNOTE, BlockType.LIST_GROUP,
        }:
            continue
        text = _doc_text(b)
        words = _WORD_RE.findall(text)
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                gr = " ".join(words[i:i + n])
                if gr.lower() not in canonical:
                    canonical[gr.lower()] = gr

    candidates = []
    for gram, f in freq.items():
        d = df[gram]
        if f < min_freq or d < min_df or d > n_docs * 0.5:
            continue  # too rare, or everywhere (not term-like)
        idf = math.log(n_docs / d) + 1.0
        n_grams = len(gram.split())
        length_bonus = 1.0 + 0.35 * (n_grams - 1)
        candidates.append({
            "term": canonical.get(gram, gram),
            "freq": f,
            "df": d,
            "page": first_page.get(gram),
            "score": round(f * idf * length_bonus, 3),
            "signal": "ngram",
        })
    candidates.sort(key=lambda c: (-c["score"], c["term"]))
    deduped: list[dict] = []
    seen_lower = set()
    for c in candidates:  # prefer the highest-scored casing/dup
        key = unicodedata.normalize("NFKC", c["term"].lower())
        if key in seen_lower:
            continue
        seen_lower.add(key)
        deduped.append(c)
        if len(deduped) >= max_candidates:
            break
    return deduped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_candidates(
    book: Book,
    *,
    use_gliner: bool = True,
    max_candidates: int = 400,
    min_freq: int = 2,
    min_df: int = 2,
    logger=None,
) -> tuple[list[dict[str, Any]], bool]:
    """Extract candidates; returns (candidates, gliner_used)."""
    candidates = mine_ngrams(book, min_freq=min_freq, min_df=min_df,
                             max_candidates=max_candidates * 2)
    gliner_used = False
    if use_gliner:
        extr = GlinerExtractor()
        if extr.available:
            texts = [
                html_to_text(b.html)
                for b in book.iter_content_blocks()
                if b.type in {BlockType.TEXT, BlockType.SECTION_HEADER}
            ][:400]
            hits = {}
            for e in extr.extract(texts):
                hits[e["term"].lower()] = e
            for c in candidates:
                key = c["term"].lower()
                if key in hits:
                    c["signal"] = "ngram+gliner"
                    c["label"] = hits[key]["label"]
            # add GLiNER-only finds before frequency cut as low-priority adds
            ngram_keys = {c["term"].lower() for c in candidates}
            for e in hits.values():
                if e["term"].lower() not in ngram_keys:
                    candidates.append({
                        "term": e["term"], "freq": 1, "df": 1, "page": None,
                        "score": 8.0, "signal": e["signal"],
                        "label": e["label"],
                    })
            candidates.sort(key=lambda c: (-c["score"], c["term"]))
            gliner_used = True
        elif logger:
            logger.info(
                "GLiNER unavailable (not installed or model not reachable); "
                "using n-gram mining only"
            )
    candidates = candidates[:max_candidates]
    return candidates, gliner_used


def write_candidates(candidates: list[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(candidates)
