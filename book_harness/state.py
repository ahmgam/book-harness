"""Run state + cache keys (tasks 1.10 / §5-21).

``state.json`` is the stage cursor for a run. Chunk-level resume does not
depend on it: drafts are keyed by content hash, so re-running a stage skips
any chunk whose draft already exists with the same
``hash(source_chunk + glossary_version + model + prompt_version)``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from book_harness.config import STATE_FILE, STATE_VERSION


def cache_key(source: str, glossary_version: str, model: str, prompt_version: str) -> str:
    """Deterministic key for a translation unit of work."""
    blob = "\x1f".join(
        (
            source,
            glossary_version or "",
            model or "",
            prompt_version or "",
        )
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RunState:
    """Track stage cursors and per-chunk results for one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / STATE_FILE
        self.data: dict[str, Any] = {
            "version": STATE_VERSION,
            "book_id": self.run_dir.name,
            "created": _now(),
            "stages": {},
        }
        self.load()

    # -- stages --------------------------------------------------------
    def load(self) -> None:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def mark(self, stage: str, *, status: str = "done", **info: Any) -> None:
        entry = self.data["stages"].setdefault(stage, {})
        entry.update(info)
        entry["status"] = status
        if status == "started":
            entry.setdefault("started", _now())
        elif status == "done":
            entry["finished"] = _now()
        self.save()

    def stage_status(self, stage: str) -> str | None:
        return self.data["stages"].get(stage, {}).get("status")

    def is_done(self, stage: str) -> bool:
        return self.stage_status(stage) == "done"


def _now() -> float:
    return round(time.time(), 3)
