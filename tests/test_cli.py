"""CLI smoke tests (task 1.11)."""

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
