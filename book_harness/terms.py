"""Term extraction support (v2; PLAN §8).

Extraction itself is fully LLM-based now — see ``llm_terms.py`` (one small
instruct model call per chunk). This module keeps the shared tokenisation
constants used by ``drift.py`` (``RUBBER_WORDS``, ``_WORD_RE``) and the
``candidates.jsonl`` writer shared by all candidate producers.

Output schema (unchanged): one JSON line per candidate —
``{term, freq, df, page, score, signal}``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z][\w'\-/]*")


#: Position-independent verb/junk words: an n-gram containing any of these
#: anywhere is sentence filler, not a term ("care is needed", "then applied").
RUBBER_WORDS = frozenset(
    """be been being has have had do does did can may will must might care
    needed needs makes made such this that these there then thus very quite
    often given shown followed way hence also become becomes some almost much
    see seen maybe rarely usually where while""".split()
)


def write_candidates(candidates: list[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(candidates)
