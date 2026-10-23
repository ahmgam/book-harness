"""Progress bars: stage tracking + work-loop bars (book_harness.progress)."""

from book_harness.config import Settings
from book_harness.progress import (
    StagePipeline,
    _LIVE_POSITIONS,
    planned_stages,
    work_bar,
)


class FakeSettings:
    translate = True
    qe_method = "auto"
    refine = True
    no_drift = False
    no_artifacts = False
    no_format = False


def _settings(**overrides) -> Settings:
    s = FakeSettings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# planned_stages

def test_planned_stages_default_run():
    stages = planned_stages(_settings(), needs_source=False)
    assert stages == [
        "outline-repair", "terms", "glossary", "consistency", "chunk", "gate",
        "translate", "qe", "refine", "compliance", "drift-audit", "drift",
        "artifacts", "assemble", "format",
    ]


def test_planned_stages_source_only_when_needed():
    out = planned_stages(_settings(), needs_source=False)
    assert "source" not in out
    assert planned_stages(_settings(), needs_source=True)[0] == "source"


def test_planned_stages_no_translate_stops_early():
    stages = planned_stages(_settings(translate=False, qe_method="none"),
                            needs_source=False)
    assert stages[-1] == "gate"
    assert "translate" not in stages


def test_planned_stages_skips_disabled_stages():
    stages = planned_stages(
        _settings(qe_method="none", refine=False, no_drift=True,
                  no_artifacts=True, no_format=True),
        needs_source=True,
    )
    for absent in ("qe", "refine", "drift", "artifacts", "format"):
        assert absent not in stages
    assert stages[-1] == "assemble"


# ---------------------------------------------------------------------------
# StagePipeline

def test_stage_pipeline_tracks_running_and_remaining():
    import io

    out = io.StringIO()
    frames: list[str] = []

    class Recording:
        def set_description_str(self, s):
            frames.append(s)

        def set_postfix_str(self, s):
            frames.append(s)

        def update(self, n=1):
            pass

        def refresh(self):
            pass

        def close(self):
            pass

    pipeline = StagePipeline(["a", "b", "c"])

    def fake_bar():
        rec = Recording()
        pipeline._bar = rec            # so __exit__ sees and closes it
        return rec

    pipeline._ensure_bar = fake_bar    # type: ignore[method-assign]
    with pipeline as p:
        with p.stage("a"):
            pass
        with p.stage("b"):
            pass
    # descriptions announce the running stage and its position
    assert frames[0] == "stage 1/3: a"
    assert f"stage 2/3: b" in frames
    # 'c' never ran, so it is never announced
    assert all(f != "stage 3/3: c" for f in frames)
    # the pipeline closes with a finished marker
    assert frames[-1].startswith("✔")


def test_stage_pipeline_remaining_list():
    seen: list[str] = []

    class Recording:
        def set_description_str(self, s):
            pass

        def set_postfix_str(self, s):
            seen.append(s)

        def update(self, n=1):
            pass

        def refresh(self):
            pass

        def close(self):
            pass

    pipeline = StagePipeline(["a", "b", "c"])

    def fake_bar():
        rec = Recording()
        pipeline._bar = rec            # so __exit__ sees and closes it
        return rec

    pipeline._ensure_bar = fake_bar    # type: ignore[method-assign]
    with pipeline as p:
        with p.stage("a"):
            pass
    assert "remaining: b, c" in seen
    assert seen[-1].startswith("a: done;")


# ---------------------------------------------------------------------------
# work_bar: stacking slots are released on close

def test_work_bar_releases_slots():
    before = set(_LIVE_POSITIONS)
    bar = work_bar([1, 2, 3], desc="x", unit="it", total=3)
    assert _LIVE_POSITIONS - before
    for _ in bar:
        pass
    bar.close()
    assert _LIVE_POSITIONS == before


def test_work_bar_nested_positions_do_not_collide():
    before = set(_LIVE_POSITIONS)
    with work_bar([1, 2], desc="outer", unit="it") as outer:
        for _ in outer:
            with work_bar(["u"], desc="inner", unit="it") as inner:
                for _ in inner:
                    pass
    assert _LIVE_POSITIONS == before
