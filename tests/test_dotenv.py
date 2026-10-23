"""``.env`` auto-load tests (config.load_dotenv)."""

import os
from pathlib import Path

import book_harness.config as config
from book_harness.config import _parse_dotenv_line, load_dotenv


class TestParseDotenvLine:
    def test_basic_and_export_prefix(self):
        assert _parse_dotenv_line("KEY=value") == ("KEY", "value")
        assert _parse_dotenv_line("export KEY=value") == ("KEY", "value")
        assert _parse_dotenv_line("  export   KEY=value") == ("KEY", "value")

    def test_comments_and_blanks_ignored(self):
        assert _parse_dotenv_line("# comment") is None
        assert _parse_dotenv_line("") is None
        assert _parse_dotenv_line("   ") is None

    def test_inline_comment_stripped_only_when_spaced(self):
        assert _parse_dotenv_line("KEY=value # note") == ("KEY", "value")
        assert _parse_dotenv_line("KEY=value#nospace") == (
            "KEY", "value#nospace")
        assert _parse_dotenv_line('KEY="v # x" # tail') == ("KEY", "v # x")

    def test_quotes_stripped(self):
        assert _parse_dotenv_line('KEY="double"') == ("KEY", "double")
        assert _parse_dotenv_line("KEY='single'") == ("KEY", "single")
        assert _parse_dotenv_line('KEY="unclosed') == ("KEY", "unclosed")

    def test_value_may_contain_equals(self):
        # API keys/signatures legitimately contain =
        assert _parse_dotenv_line("KEY=a=b=c") == ("KEY", "a=b=c")

    def test_empty_value_keeps_key(self):
        assert _parse_dotenv_line("KEY=") == ("KEY", "")

    def test_tilde_home_expansion(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        assert _parse_dotenv_line("KEY=~/.cache/x") == (
            "KEY", "/home/tester/.cache/x")
        assert _parse_dotenv_line("KEY=plain~") == ("KEY", "plain~")

    def test_garbage_keys_rejected(self):
        assert _parse_dotenv_line("=novalue") is None
        assert _parse_dotenv_line("no equals here") is None
        assert _parse_dotenv_line("'quoted'=v") is None


class TestLoadDotenv:
    def test_explicit_path_sets_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("DOTENV_TEST_KEY", raising=False)
        f = tmp_path / "custom.env"
        f.write_text("DOTENV_TEST_KEY=from-file\n", encoding="utf-8")
        assert load_dotenv(f) == str(f)
        assert os.environ["DOTENV_TEST_KEY"] == "from-file"
        assert config.DOTENV_LOADED == str(f)

    def test_existing_export_wins_unless_override(self, tmp_path: Path,
                                                  monkeypatch):
        monkeypatch.setenv("DOTENV_TEST_KEY", "from-export")
        f = tmp_path / "custom.env"
        f.write_text("DOTENV_TEST_KEY=from-file\n", encoding="utf-8")
        load_dotenv(f)
        assert os.environ["DOTENV_TEST_KEY"] == "from-export"
        load_dotenv(f, override=True)
        assert os.environ["DOTENV_TEST_KEY"] == "from-file"

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_dotenv(tmp_path / "absent.env") is None

    def test_autodetect_prefers_cwd_over_project_root(
        self, tmp_path: Path, monkeypatch,
    ):
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        (cwd / ".env").write_text("DOTENV_TEST_CWD=1\n", encoding="utf-8")
        monkeypatch.delenv("DOTENV_TEST_CWD", raising=False)
        monkeypatch.chdir(cwd)
        loaded = load_dotenv()               # path=None: auto-detect
        assert loaded == str(cwd / ".env")
        assert os.environ["DOTENV_TEST_CWD"] == "1"

    def test_import_time_autoload_did_the_job(self):
        """The module-import load point fired: either no .env exists, or the
        recorded path is the repo .env (constants below were resolved after
        it was applied)."""
        if config.DOTENV_LOADED:
            assert Path(config.DOTENV_LOADED).name == ".env"
            assert Path(config.DOTENV_LOADED).is_file()
