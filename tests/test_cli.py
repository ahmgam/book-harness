"""CLI smoke tests (task 1.11) + M3 command surface."""

from book_harness.cli import build_parser, main


class TestParser:
    def test_defaults(self):
        args = build_parser().parse_args(["run", "book.json"])
        assert str(args.ocr_json) == "book.json"
        assert args.char_ceiling == 1200
        assert args.domains == ["aerospace", "software"]
        assert not args.no_translate

    def test_variant_flags(self):
        args = build_parser().parse_args([
            "run", "b.json", "--pages", "30", "45", "--max-chunks", "5",
            "--model", "llama3.1:8b", "--no-translate", "--glossary-limit", "50",
        ])
        assert args.pages == [30, 45]
        assert args.max_chunks == 5
        assert args.model == "llama3.1:8b"
        assert args.no_translate
        assert args.glossary_limit == 50

    def test_stages_listing(self, capsys):
        assert main(["stages"]) == 0
        out = capsys.readouterr().out
        assert "source" in out and "assemble" in out

    def test_meta_file_rejected(self, tmp_path):
        meta = tmp_path / "book_meta.json"
        meta.write_text("{}")
        assert main(["run", str(meta)]) == 2  # logged error, exit code


class TestM3Commands:
    def test_stages_listing_includes_m3(self, capsys):
        assert main(["stages"]) == 0
        out = capsys.readouterr().out
        assert "outline-repair" in out
        assert "drift" in out and "artifacts" in out

    def test_run_m3_flags(self):
        args = build_parser().parse_args(["run", "book.json", "--no-drift", "--no-artifacts"])
        assert args.no_drift and args.no_artifacts

    def test_drift_defaults(self):
        args = build_parser().parse_args(["drift", "book.json"])
        assert args.min_freq == 3 and args.min_df == 2
        assert args.confirm == [] and args.no_rerun is False

    def test_drift_confirm_flags(self):
        args = build_parser().parse_args([
            "drift", "book.json",
            "--confirm", "boundary layer=الطبقة الحدّية",
            "--confirm", "wearout",
        ])
        assert args.confirm == ["boundary layer=الطبقة الحدّية", "wearout"]
        assert args.model is None

    def test_drift_missing_ocr_rejected(self, tmp_path):
        assert main(["drift", str(tmp_path / "none.json")]) == 2
