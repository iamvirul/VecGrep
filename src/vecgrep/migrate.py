
"""Utility script to migrate existing SQLite VecGrep indexes to LanceDB."""

import sqlite3
from pathlib import Path

import numpy as np

from vecgrep.server import VECGREP_HOME
from vecgrep.store import VectorStore


def migrate_project(project_dir: Path) -> None:
    db_path = project_dir / "index.db"
    if not db_path.exists():
        return

    print(f"Found legacy SQLite index at {db_path}...")

    # Read legacy data
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute("SELECT * FROM chunks").fetchall()
        meta_rows = dict(conn.execute("SELECT * FROM meta").fetchall())
    except sqlite3.OperationalError:
        print(f"Skipping {db_path} (not a valid VecGrep index)")
        conn.close()
        return

    conn.close()

    if not rows:
        print("No chunks found. Renaming to .bak and skipping migration.")
        db_path.rename(db_path.with_suffix(".db.bak"))
        return

    print(f"Migrating {len(rows)} chunks from {project_dir.name} to LanceDB...")

    prepared_rows = []
    vectors = []

    for row in rows:
        d = dict(row)
        # We need to construct a new vector from the BLOB
        vec = np.frombuffer(d["embedding"], dtype=np.float32)
        vectors.append(vec)

        # SQLite db doesn't have mtime and size, so we initialize to zeros
        # This will force the first index run to re-stat everything,
        # which is extremely fast anyway and ensures correctness.
        prepared_rows.append({
            "file_path": d["file_path"],
            "start_line": d["start_line"],
            "end_line": d["end_line"],
            "content": d["content"],
            "file_hash": d["file_hash"],
            "chunk_hash": d["chunk_hash"],
            "mtime": 0.0,
            "size": 0,
        })

    vectors = np.array(vectors)

    # Ingest into LanceDB
    with VectorStore(project_dir) as store:
        store.add_chunks(prepared_rows, vectors)

        if "last_indexed" in meta_rows:
            store._meta_table.add([{"key": "last_indexed", "value": meta_rows["last_indexed"]}])

    # Backup the old SQLite db
    backup_path = db_path.with_suffix(".db.bak")
    db_path.rename(backup_path)
    print(f"Successfully migrated {project_dir.name}. Old database backed up to {backup_path.name}")


def main() -> None:
    if not VECGREP_HOME.exists():
        print(f"No VecGrep home directory found at {VECGREP_HOME}.")
        return

    projects = [p for p in VECGREP_HOME.iterdir() if p.is_dir() and len(p.name) == 16]

    if not projects:
        print("No VecGrep projects found.")
        return

    print(f"Found {len(projects)} potential VecGrep projects. Scanning for legacy indexes...")
    for project_dir in projects:
        migrate_project(project_dir)

    print("Migration complete.")


if __name__ == "__main__":
    main()
