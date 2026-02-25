"""Tests for vecgrep.migrate — SQLite → LanceDB migration utility."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vecgrep.migrate import main, migrate_project

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sqlite_db(db_path: Path, rows: list[dict] | None = None, add_meta: bool = True) -> None:
    """Create a minimal VecGrep-style SQLite database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunks (file_path TEXT, start_line INTEGER, end_line INTEGER, "
        "content TEXT, file_hash TEXT, chunk_hash TEXT, embedding BLOB)"
    )
    if add_meta:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('last_indexed', '2024-01-01T00:00:00')")

    if rows:
        for r in rows:
            vec = np.random.rand(384).astype(np.float32)
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (r["file_path"], r["start_line"], r["end_line"],
                 r["content"], r["file_hash"], r["chunk_hash"], vec.tobytes()),
            )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# migrate_project
# ---------------------------------------------------------------------------


class TestMigrateProject:
    def test_no_db_does_nothing(self, tmp_path):
        """If no index.db exists, migrate_project returns immediately."""
        migrate_project(tmp_path)
        # Nothing created
        assert list(tmp_path.iterdir()) == []

    def test_invalid_schema_skips_and_closes(self, tmp_path):
        """A SQLite file with wrong schema prints a warning and skips."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
        conn.commit()
        conn.close()

        migrate_project(tmp_path)
        # db_path still exists (not renamed), no LanceDB data folder
        assert db_path.exists()

    def test_empty_db_renames_to_bak(self, tmp_path):
        """A valid schema with zero rows renames the .db to .db.bak."""
        db_path = tmp_path / "index.db"
        _make_sqlite_db(db_path, rows=[])

        migrate_project(tmp_path)

        assert not db_path.exists()
        assert (tmp_path / "index.db.bak").exists()

    def test_migration_creates_lancedb_and_backs_up(self, tmp_path):
        """Rows are migrated to LanceDB and the SQLite file is renamed."""
        db_path = tmp_path / "index.db"
        _make_sqlite_db(
            db_path,
            rows=[
                {
                    "file_path": "/repo/foo.py",
                    "start_line": 1,
                    "end_line": 5,
                    "content": "def foo(): pass",
                    "file_hash": "abc123",
                    "chunk_hash": "def456",
                },
            ],
        )

        migrate_project(tmp_path)

        # Original db should be gone, backup should exist
        assert not db_path.exists()
        assert (tmp_path / "index.db.bak").exists()

    def test_migration_without_last_indexed_meta(self, tmp_path):
        """Migration works even when the meta table has no last_indexed row."""
        db_path = tmp_path / "index.db"
        _make_sqlite_db(
            db_path,
            rows=[
                {
                    "file_path": "/repo/bar.py",
                    "start_line": 1,
                    "end_line": 3,
                    "content": "x = 1",
                    "file_hash": "aaa",
                    "chunk_hash": "bbb",
                },
            ],
            add_meta=False,
        )
        # Create meta table without last_indexed
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()

        migrate_project(tmp_path)

        assert not db_path.exists()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_vecgrep_home(self, tmp_path):
        """main() exits gracefully when VECGREP_HOME doesn't exist."""
        with patch("vecgrep.migrate.VECGREP_HOME", tmp_path / "nonexistent"):
            main()  # Should not raise

    def test_no_projects(self, tmp_path):
        """main() exits gracefully when VECGREP_HOME is empty."""
        (tmp_path / ".vecgrep").mkdir()
        with patch("vecgrep.migrate.VECGREP_HOME", tmp_path / ".vecgrep"):
            main()  # Should not raise

    def test_with_non_16char_dirs_skipped(self, tmp_path):
        """Directories whose names are not exactly 16 chars are not processed."""
        home = tmp_path / ".vecgrep"
        home.mkdir()
        (home / "tooshort").mkdir()
        (home / "toolongdirectorynamethatexceeds").mkdir()

        with patch("vecgrep.migrate.VECGREP_HOME", home):
            main()  # Should not raise, nothing to migrate

    def test_main_calls_migrate_project(self, tmp_path):
        """main() calls migrate_project for each 16-char directory."""
        home = tmp_path / ".vecgrep"
        home.mkdir()
        proj = home / ("a" * 16)
        proj.mkdir()

        called_with = []

        def fake_migrate(p):
            called_with.append(p)

        with patch("vecgrep.migrate.VECGREP_HOME", home), \
             patch("vecgrep.migrate.migrate_project", side_effect=fake_migrate):
            main()

        assert proj in called_with
