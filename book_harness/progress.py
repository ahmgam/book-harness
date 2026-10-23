"""tqdm-based progress reporting for the pipeline.

Two kinds of bars:

- **Stage bar** (:class:`StagePipeline`, used by ``cli.run``): one persistent
  bar with a tick per pipeline stage. Its description names the *running*
  stage and the postfix lists the *remaining* stages.
- **Work bars** (:func:`work_bar`): transient bars for long-running loops
  (per chunk / unit / page / entry / model). They stack *under* the stage bar
  automatically, so a unit-level bar nests inside the chunk-level bar.

Bars render only on a TTY — when stdout/stderr is piped or captured (tests,
``serve``, CI), tqdm disables itself and the logging below falls back to
plain stderr, so behaviour is unchanged for non-interactive use.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

try:
    from tqdm import tqdm
except ImportError:  # graceful degradation when tqdm is not installed
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

#: Position slots currently held by live bars (simple stacking so nested
#: bars do not clobber each other; 0 is reserved for the stage bar while a
#: :class:`StagePipeline` is active).
_LIVE_POSITIONS: set[int] = set()

#: The active pipeline when inside a ``with StagePipeline(...):`` block.
_ACTIVE: "StagePipeline | None" = None


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that prints through ``tqdm.write`` so log lines are
    emitted *above* any live bar instead of clobbering it.

    Falls back to plain stderr writes when tqdm is unavailable.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
        except Exception:                        # pragma: no cover
            self.handleError(record)
            return
        if tqdm is not None:
            tqdm.write(msg, file=sys.stderr, nolock=False)
        else:                                    # pragma: no cover
            sys.stderr.write(msg)


def _acquire_position() -> int | None:
    """Reserve the next free stacking slot below the active stage bar."""
    if tqdm is None:
        return None
    start = 1 if _ACTIVE is not None else 0
    pos = start
    while pos in _LIVE_POSITIONS:
        pos += 1
    _LIVE_POSITIONS.add(pos)
    return pos


def _release_position(pos: int | None) -> None:
    if pos is not None:
        _LIVE_POSITIONS.discard(pos)


class _WorkBar(tqdm):  # type: ignore[misc]
    """tqdm bar that releases its stacking slot when closed."""

    def __init__(self, slot: int | None, *args: Any, **kwargs: Any) -> None:
        self._slot = slot
        super().__init__(*args, **kwargs)

    def close(self) -> None:
        slot = getattr(self, "_slot", None)
        super().close()
        _release_position(slot)
        self._slot = None


def work_bar(
    iterable: Sequence | Iterator | None = None,
    *,
    desc: str,
    unit: str = "unit",
    total: int | None = None,
    **tqdm_kwargs: Any,
) -> Any:
    """Create a progress bar for a long-running work loop.

    Usage::

        with work_bar(chunks, desc="translate", unit="chunk",
                      total=len(chunks)) as pbar:
            for c in pbar:                     # every ``next()`` ticks the bar
                pbar.set_postfix_str(c.chunk_id)
                translate_chunk(c, ...)

    The bar is nested below the active stage bar (``position`` stacking), is
    transient (``leave=False``) and auto-disables when stderr is not a TTY.
    Skipped iterations (``continue``) still tick the bar because the loop
    drives ``tqdm.__iter__``. Returns the bare iterable (no bar) when tqdm
    is not installed.
    """
    if tqdm is None:                           # pragma: no cover
        return iterable if iterable is not None else iter(())
    slot = _acquire_position()
    tqdm_kwargs.setdefault("disable", None)    # auto-disable on non-TTY
    return _WorkBar(
        slot, iterable=iterable, desc=desc, unit=unit, total=total,
        leave=False, position=slot or 0, **tqdm_kwargs,
    )


class StagePipeline:
    """Persistent tqdm bar tracking the pipeline's stages.

    One tick per finished stage; the description shows the running stage,
    the postfix shows the remaining ones::

        chunk:  38%|███▌      | 6/16 [05:12<04:01, remaining: translate,
        qe, refine, ...]

    Individual stages wrap their work in ``stages.stage(name)`` context
    managers; stage functions themselves do not need to know about it.
    """

    def __init__(self, stages: Sequence[str], *, desc: str = "pipeline") -> None:
        self.stages = list(stages)
        self.desc = desc
        self._bar: Any = None

    def __enter__(self) -> "StagePipeline":
        global _ACTIVE
        _ACTIVE = self
        return self

    def __exit__(self, *exc: Any) -> bool:
        global _ACTIVE
        _ACTIVE = None
        if self._bar is not None:
            self._bar.set_description_str("✔ pipeline")
            self._bar.close()
            self._bar = None
        return False

    # -- internals ----------------------------------------------------------

    def _ensure_bar(self) -> Any:
        if self._bar is None and tqdm is not None:
            bar_kwargs: dict[str, Any] = {}
            bar_kwargs.setdefault("disable", None)  # auto-disable on non-TTY
            self._bar = tqdm(
                total=len(self.stages), desc="", unit="stage",
                position=0, leave=True, **bar_kwargs,
            )
        return self._bar

    def _remaining_str(self, idx: int) -> str:
        remaining = self.stages[idx + 1:]
        return f"remaining: {', '.join(remaining)}" if remaining else "final stage"

    # -- stage transitions ---------------------------------------------------

    def stage(self, name: str) -> Any:
        """Context manager for one stage: marks it running, then finished."""

        @contextmanager
        def _ctx() -> Iterator[None]:
            self.start(name)
            try:
                yield
            except Exception:
                self.finish(name, status="failed")
                raise
            else:
                self.finish(name)

        return _ctx()

    def start(self, name: str) -> None:
        bar = self._ensure_bar()
        if bar is None:                        # pragma: no cover
            return
        idx = self.stages.index(name)
        bar.set_description_str(f"stage {idx + 1}/{len(self.stages)}: {name}")
        bar.set_postfix_str(self._remaining_str(idx))
        bar.refresh()

    def finish(self, name: str, status: str = "done") -> None:
        bar = self._ensure_bar()
        if bar is None:                        # pragma: no cover
            return
        bar.update(1)
        idx = self.stages.index(name)
        bar.set_postfix_str(f"{name}: {status}; {self._remaining_str(idx)}")
        bar.refresh()


def planned_stages(settings: Any, *, needs_source: bool) -> list[str]:
    """The stage list a ``run`` will actually execute, for the stage bar.

    Skipped stages (``--no-translate``, ``qe-method none``, ...) are excluded
    so the bar's total matches reality.
    """
    stages: list[str] = []
    if needs_source:
        stages.append("source")
    stages += ["outline-repair", "terms", "glossary", "consistency", "chunk", "gate"]
    if not settings.translate:
        return stages
    stages.append("translate")
    if settings.qe_method != "none":
        stages.append("qe")
    if settings.refine:
        stages.append("refine")
    stages += ["compliance", "drift-audit"]
    if not settings.no_drift:
        stages.append("drift")
    if not settings.no_artifacts:
        stages.append("artifacts")
    stages.append("assemble")
    if not settings.no_format:
        stages.append("format")
    return stages
