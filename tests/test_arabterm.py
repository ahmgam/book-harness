"""Tests for the arabterm loader against a tiny fixture db."""

import sqlite3
from pathlib import Path

import pytest

from book_harness.arabterm import Arabterm, _is_sqlite

SCHEMA = """
CREATE TABLE dictionary (
  id INTEGER PRIMARY KEY,
  name_tech TEXT, name_arabic TEXT, name_english TEXT, name_french TEXT,
  nbr_entries INTEGER, wikidata_id TEXT,
  archive_org_url TEXT, page_offset INTEGER, created_at TEXT
);
CREATE TABLE term (
  id INTEGER PRIMARY KEY,
  dictionary_id INTEGER REFERENCES dictionary(id),
  arabic TEXT, english TEXT, french TEXT, description TEXT,
  page INTEGER, uri TEXT
);
"""


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "arabterm.db"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute(
        "INSERT INTO dictionary VALUES (1,'at_physics','المعجم الفيزيائي','Physics',NULL,"
        " 2,'Q1156917','https://archive.org/details/phys001',10, '2020-01-01')"
    )
    con.execute("INSERT INTO dictionary VALUES (2,'at_information_tech','تقنية المعلومات','Information Tech',NULL, 2, NULL, NULL, NULL, '2020-01-02')")
    con.execute("INSERT INTO term VALUES (1, 1, 'الرفع', 'lift', 'portance', NULL, 214, NULL)")
    con.execute("INSERT INTO term VALUES (2, 1, 'معامل السحب', 'drag coefficient', NULL, 'force', 301, NULL)")
    con.execute("INSERT INTO term VALUES (3, 2, 'تجميع', 'assembly', NULL, NULL, 44, NULL)")
    con.execute("INSERT INTO term VALUES (4, 2, 'التجميع', 'assembly', NULL, NULL, 45, NULL)")
    con.commit()
    con.close()
    return path


class TestLookup:
    def test_exact_hit(self, db):
        at = Arabterm.from_db(db)
        hits = at.lookup("Drag Coefficient")     # case-insensitive
        assert len(hits) == 1
        h = hits[0]
        assert h["ar"] == "معامل السحب"
        assert h["source"] == "المعجم الفيزيائي"
        assert h["source_tech"] == "at_physics"
        assert h["wikidata"] == "Q1156917"
        assert h["page"] == 301
        assert h["description"] == "force"

    def test_scan_url_formula(self, db):
        at = Arabterm.from_db(db)
        h = at.lookup("lift")[0]
        # printed page 214, offset 10 -> archive page n223
        assert h["scan"] == "https://archive.org/details/phys001/page/n223/mode/1up"

    def test_no_scan_when_metadata_missing(self, db):
        at = Arabterm.from_db(db)
        assert at.lookup("assembly")[0]["scan"] is None

    def test_arabic_lookup(self, db):
        at = Arabterm.from_db(db)
        hits = at.lookup_arabic("الرفع")
        assert len(hits) == 1 and hits[0]["en"] == "lift"

    def test_miss_returns_empty(self, db):
        at = Arabterm.from_db(db)
        assert at.lookup("quantum entanglement sequencer") == []

    def test_multiple_hits_collected(self, db):
        at = Arabterm.from_db(db)
        hits = at.lookup("assembly")
        assert len(hits) == 2
        assert {h["page"] for h in hits} == {44, 45}


def test_is_sqlite_guard(db, tmp_path):
    assert _is_sqlite(db)
    bogus = tmp_path / "bad.db"
    bogus.write_bytes(b"definitely not sqlite")
    assert not _is_sqlite(bogus)
