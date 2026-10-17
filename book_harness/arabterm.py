"""arabterm loader (task 1.4).

arabterm (github.com/forzagreen/arabterm) ships certified Arabic/English/French
technical dictionaries as one SQLite file (``arabterm.db.gz``) with full
provenance built in:

* ``dictionary``: ``name_tech`` (slug), ``name_arabic``, ``name_english``,
  ``wikidata_id``, ``archive_org_url``, ``page_offset`` (P8330)
* ``term``: ``arabic``, ``english``, ``french``, ``description``, ``page``, ``uri``

Every lookup returns translation *citations*: source dictionary, Wikidata ID,
printed page and a resolvable archive.org scan URL
(``page`` + ``page_offset`` - 1 → ``/page/nN/mode/1up``).

The db is downloaded once into a platform cache dir and unpacked there.
"""

from __future__ import annotations

import gzip
import logging
import sqlite3
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from book_harness import config

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = Path(
    __import__("os").environ.get(
        "BOOK_HARNESS_CACHE", Path.home() / ".cache" / "book-harness"
    )
)
DB_NAME = "arabterm.db"


def ensure_db(cache_dir: Path | None = None) -> Path:
    """Return the path of a locally unpacked arabterm.db, downloading once."""
    cache_dir = Path(cache_dir or DEFAULT_DB_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / DB_NAME
    if db_path.exists() and _is_sqlite(db_path):
        return db_path

    last_error: Exception | None = None
    for url in config.ARABTERM_URLS:
        try:
            logger.info("downloading arabterm.db.gz from %s …", url)
            with urllib.request.urlopen(url, timeout=120) as resp:
                blob = resp.read()
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".gz", delete=False
            ) as tmp:
                tmp.write(blob)
                gz_path = Path(tmp.name)
            with gzip.open(gz_path, "rb") as gz, open(db_path, "wb") as out:
                out.write(gz.read())
            gz_path.unlink(missing_ok=True)
            logger.info("arabterm.db ready: %s (%.1f MB)",
                        db_path, db_path.stat().st_size / 1e6)
            return db_path
        except Exception as exc:            # try next mirror
            last_error = exc
            logger.warning("download failed from %s: %s", url, exc)
    raise RuntimeError(
        "could not obtain arabterm.db; place it manually at "
        f"{db_path} (last error: {last_error})"
    )


def _is_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def _scan_url(dict_row: sqlite3.Row, page: int | None) -> str | None:
    url = dict_row["archive_org_url"]
    offset = dict_row["page_offset"]
    if not url or offset is None or page is None:
        return None
    n = page + offset - 1
    if n < 0:
        return url
    return f"{url}/page/n{n}/mode/1up"


_HIT_SQL = """
    SELECT t.arabic, t.english, t.french, t.description, t.page, t.uri,
           t.dictionary_id,
           d.*
      FROM term t
      JOIN dictionary d ON d.id = t.dictionary_id
"""


class Arabterm:
    """Read-only helper around arabterm.db"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.con.row_factory = sqlite3.Row
        self._dict_cache: dict[int, dict] = {}

    # -- lifecycle -----------------------------------------------------------
    @classmethod
    def from_db(cls, db_path: str | Path) -> "Arabterm":
        return cls(db_path)

    def __enter__(self) -> "Arabterm":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- dictionaries -----------------------------------------------------
    def dictionaries(self) -> list[dict[str, Any]]:
        rows = self.con.execute(
            "SELECT id, name_tech, name_arabic, name_english, name_french, "
            "nbr_entries, wikidata_id, archive_org_url, page_offset "
            "FROM dictionary ORDER BY name_tech"
        ).fetchall()
        return [dict(r) for r in rows]

    def _dict(self, dictionary_id: int) -> dict:
        if dictionary_id not in self._dict_cache:
            row = self.con.execute(
                "SELECT id, name_tech, name_arabic, name_english, name_french, "
                "wikidata_id, archive_org_url, page_offset "
                "FROM dictionary WHERE id = ?", (dictionary_id,)
            ).fetchone()
            self._dict_cache[dictionary_id] = dict(row or {})
        return self._dict_cache[dictionary_id]

    # -- lookup ------------------------------------------------------------
    _RUBBER_WORDS = {
        "is", "are", "was", "were", "be", "been", "being", "has", "have",
        "had", "do", "does", "did", "can", "may", "will", "must", "might",
        "care", "needed", "needs", "makes", "made", "such", "this", "that",
        "these", "there", "then", "thus", "very", "quite", "often", "given",
        "shown", "followed", "value", "way", "hence", "also", "become",
        "becomes", "see", "seen", "some", "almost", "much", "using", "used",
    }

    def lookup_term(self, term: str, *, exact: bool = True) -> list[dict]:
        """Lookup an English term; each hit is a citation-ready dict."""
        needle = term.strip().lower()
        attempts = [needle]
        # light lemmatization: 'flight simulators' -> 'flight simulator';
        # drop a leading article ('the flight simulator' -> 'flight simulator')
        if needle.startswith("the "):
            attempts.append(needle[4:])
        if needle.endswith("ies") and len(needle) > 4:
            attempts.append(needle[:-3] + "y")
        elif needle.endswith(("ses", "xes", "zes", "ches", "shes")):
            attempts.append(needle[:-2])
        elif needle.endswith("es") and not needle.endswith("les"):
            attempts.append(needle[:-2])
        elif needle.endswith("s") and not needle.endswith("ss"):
            attempts.append(needle[:-1])
        for attempt in attempts:
            if not attempt:
                continue
            rows = self.con.execute(
                _HIT_SQL + " WHERE LOWER(TRIM(t.english)) = ? ORDER BY t.id",
                (attempt,),
            ).fetchall()
            hits = self._hit_rows(rows)
            if hits:
                return hits
        # phrase prefix fallback ('low pass' → 'low pass filter') — top few
        sql = _HIT_SQL + (
            " WHERE LOWER(TRIM(t.english)) LIKE ? || ' %'"
            " AND LOWER(TRIM(t.english)) != ? ORDER BY t.id LIMIT 5"
        )
        rows = self.con.execute(sql, (needle, needle)).fetchall()
        return self._hit_rows(rows)

    # alias kept close to glossary.GlossaryBuilder's duck-typing
    def lookup(self, term: str, *args, **kwargs) -> list[dict]:
        return self.lookup_term(term, *args, **kwargs)

    def lookup_arabic(self, term_ar: str) -> list[dict]:
        sql = _HIT_SQL + " WHERE LOWER(TRIM(t.arabic)) = ? ORDER BY t.id"
        return self._hit_rows(
            self.con.execute(sql, (term_ar.strip().lower(),)).fetchall()
        )

    def _hit_rows(self, rows: Iterable[sqlite3.Row]) -> list[dict]:
        out = []
        for r in rows:
            d = self._dict(r["dictionary_id"])
            out.append({
                "ar": r["arabic"],
                "en": r["english"],
                "fr": r["french"],
                "source": d.get("name_arabic") or d.get("name_tech"),
                "source_tech": d.get("name_tech"),
                "wikidata": d.get("wikidata_id"),
                "page": r["page"],
                "scan": _scan_url(d, r["page"]),
                "description": r["description"],
                "uri": r["uri"],
            })
        return out
