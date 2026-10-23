"""Role-based model assignment via env (translation / assist / qe).

Module-level role constants are read at import, so the env parsing itself is
tested through ``config._env_models`` with monkeypatched env, and the stage
wiring through ``cli._assist_client`` with a stubbed ``build_client``.
"""

import book_harness.config as config
from book_harness import cli


def test_defaults_when_env_unset(monkeypatch):
    """Clean env → built-in defaults via ``_env_models`` resolution.

    Module-level constants are resolved at import and may legitimately come
    from an auto-loaded ``.env`` (they are asserted separately); the env
    parsing and the role inheritance cascade are asserted here live.
    """
    monkeypatch.delenv("BOOK_HARNESS_TRANSLATION_MODELS", raising=False)
    monkeypatch.delenv("BOOK_HARNESS_TRANSLATION_MODEL", raising=False)
    monkeypatch.delenv("BOOK_HARNESS_ASSIST_MODELS", raising=False)
    monkeypatch.delenv("BOOK_HARNESS_ASSIST_MODEL", raising=False)
    monkeypatch.delenv("BOOK_HARNESS_QE_MODELS", raising=False)
    monkeypatch.delenv("BOOK_HARNESS_QE_MODEL", raising=False)
    resolve = config._env_models
    translation = resolve(
        "BOOK_HARNESS_TRANSLATION_MODELS", "BOOK_HARNESS_TRANSLATION_MODEL",
        config._MODEL_DEFAULTS,
    )
    assert translation == config._MODEL_DEFAULTS
    # cascade: unset roles inherit the previous level's list
    assert resolve(
        "BOOK_HARNESS_ASSIST_MODELS", "BOOK_HARNESS_ASSIST_MODEL", translation
    ) == translation
    assert resolve(
        "BOOK_HARNESS_QE_MODELS", "BOOK_HARNESS_QE_MODEL", translation
    ) == translation


def test_env_models_parse_lists_commas_and_singular(monkeypatch):
    monkeypatch.setenv("BOOK_HARNESS_X_MODELS", "a, b c")
    assert config._env_models("BOOK_HARNESS_X_MODELS", "BOOK_HARNESS_X_MODEL",
                              ["fallback"]) == ["a", "b", "c"]
    # singular alias acts as a one-element list
    monkeypatch.delenv("BOOK_HARNESS_X_MODELS")
    monkeypatch.setenv("BOOK_HARNESS_X_MODEL", "solo")
    assert config._env_models("BOOK_HARNESS_X_MODELS", "BOOK_HARNESS_X_MODEL",
                              ["fallback"]) == ["solo"]
    # blank / whitespace-only values fall back
    monkeypatch.setenv("BOOK_HARNESS_X_MODEL", "   ")
    assert config._env_models("BOOK_HARNESS_X_MODELS", "BOOK_HARNESS_X_MODEL",
                              ["fallback"]) == ["fallback"]


def test_role_env_overrides_resolve(monkeypatch):
    monkeypatch.setenv("BOOK_HARNESS_TRANSLATION_MODELS", "mt-a,mt-b")
    monkeypatch.setenv("BOOK_HARNESS_ASSIST_MODEL", "assist-a")
    monkeypatch.setenv("BOOK_HARNESS_QE_MODEL", "qe-a")
    assert config._env_models(
            "BOOK_HARNESS_TRANSLATION_MODELS",
                "BOOK_HARNESS_TRANSLATION_MODEL",
            config._MODEL_DEFAULTS) == ["mt-a", "mt-b"]
    assert config._env_models(
        "BOOK_HARNESS_ASSIST_MODELS", "BOOK_HARNESS_ASSIST_MODEL",
        config.TRANSLATION_MODELS) == ["assist-a"]
    assert config._env_models(
        "BOOK_HARNESS_QE_MODELS", "BOOK_HARNESS_QE_MODEL",
        config.ASSIST_MODELS) == ["qe-a"]


def test_stage_role_wiring(monkeypatch, tmp_path):
    """QE stage uses the QE list; compliance re-queue uses the MT list;
    refine keeps the assist list."""
    from book_harness.config import Settings

    captured: list[list[str]] = []

    def fake_build_client(host, *, timeout=None, preferred=None,
                          model=None, api=None, api_key=None):
        captured.append(preferred or [])
        return lambda prompt, **kw: ""          # duck-type client

    import book_harness.translate as tr

    monkeypatch.setattr(tr, "build_client", fake_build_client)
    settings = Settings(run_root=tmp_path, ocr_file=tmp_path / "x.json")

    cli._assist_client(settings, preferred=config.QE_MODELS)
    cli._assist_client(settings)
    cli._translation_client(settings)
    assert captured == [list(config.QE_MODELS),
                        list(config.ASSIST_MODELS),
                        list(config.TRANSLATION_MODELS)]


def test_bakeoff_judge_uses_qe_role(monkeypatch, tmp_path):
    import book_harness.bakeoff as bakeoff

    captured: list[list[str]] = []

    def fake_build_client(host="x", *, preferred=None, **kw):
        captured.append(preferred or [])
        return bakeoff._NullClient()

    import book_harness.translate as tr

    monkeypatch.setattr(tr, "build_client", fake_build_client)
    monkeypatch.setattr(bakeoff, "build_client", fake_build_client)

    class S:                                    # minimal settings duck
        ollama_host = "http://x"
        ollama_timeout = 5
        api = None
        api_key = ""
        model = None

    client = bakeoff._judge_client(S(), judge_model=None)
    assert captured[-1] == list(config.QE_MODELS)
    assert isinstance(client, bakeoff._NullClient)     # picked fallback fine
