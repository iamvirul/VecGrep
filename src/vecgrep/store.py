
"""LanceDB-backed vector store."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

# Minimum chunks needed before building an ANN index (IVF-PQ requires enough data)
_INDEX_MIN_ROWS = 256

# Define the LanceDB Schema
schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("file_path", pa.string()),
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("content", pa.string()),
    pa.field("file_hash", pa.string()),
    pa.field("chunk_hash", pa.string()),
    pa.field("mtime", pa.float64()),
    pa.field("size", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), 384)),
])

_file_stats_schema = pa.schema([
    pa.field("file_path", pa.string()),
    pa.field("file_hash", pa.string()),
    pa.field("mtime", pa.float64()),
    pa.field("size", pa.int64()),
])

_meta_schema = pa.schema([
    pa.field("key", pa.string()),
    pa.field("value", pa.string()),
])


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.index_dir / "lancedb")
        self._db = lancedb.connect(self.db_path)

        existing = set(self._db.list_tables().tables)

        self.table_name = "chunks"
        if self.table_name not in existing:
            self._table = self._db.create_table(self.table_name, schema=schema)
        else:
            self._table = self._db.open_table(self.table_name)

        # Per-file stats table: one row per file, O(files) reads for change detection
        self._file_stats_table_name = "file_stats"
        if self._file_stats_table_name not in existing:
            self._file_stats_table = self._db.create_table(
                self._file_stats_table_name, schema=_file_stats_schema
            )
        else:
            self._file_stats_table = self._db.open_table(self._file_stats_table_name)

        self.meta_table_name = "meta"
        if self.meta_table_name not in existing:
            self._meta_table = self._db.create_table(self.meta_table_name, schema=_meta_schema)
        else:
            self._meta_table = self._db.open_table(self.meta_table_name)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *args: object) -> None:
        pass  # LanceDB doesn't require explicit closing

    # ------------------------------------------------------------------
    # File stats helpers (O(files), not O(chunks))
    # ------------------------------------------------------------------

    def get_file_stats(self) -> dict[str, dict]:
        """Return {file_path: {file_hash, mtime, size}} — reads from file_stats table (O(files))."""
        if self._file_stats_table.count_rows() == 0:
            return {}
        rows = self._file_stats_table.search().limit(None).to_list()
        return {
            r["file_path"]: {
                "file_hash": r["file_hash"],
                "mtime": r["mtime"],
                "size": r["size"],
            }
            for r in rows
        }

    def _upsert_file_stat(self, file_path: str, file_hash: str, mtime: float, size: int) -> None:
        """Insert or replace one row in the file_stats table."""
        safe = file_path.replace("'", "''")
        self._file_stats_table.delete(f"file_path = '{safe}'")
        self._file_stats_table.add([{
            "file_path": file_path,
            "file_hash": file_hash,
            "mtime": mtime,
            "size": size,
        }])

    def get_file_hashes(self) -> dict[str, str]:
        """Compatibility method for legacy callers."""
        return {fp: s["file_hash"] for fp, s in self.get_file_stats().items()}

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def delete_file_chunks(self, file_path: str) -> None:
        """Remove all chunks and the file stat entry for a given file."""
        safe = file_path.replace("'", "''")
        self._table.delete(f"file_path = '{safe}'")
        self._file_stats_table.delete(f"file_path = '{safe}'")

    def add_chunks(self, rows: list[dict], vectors: np.ndarray) -> None:
        """
        Insert chunk rows with their embeddings and update the file_stats table.

        rows: list of dicts with keys: file_path, start_line, end_line,
              content, file_hash, chunk_hash, mtime, size
        vectors: float32 array of shape (len(rows), 384)
        """
        if len(rows) != len(vectors):
            raise ValueError("rows/vectors length mismatch")

        data = []
        for i, r in enumerate(rows):
            row_id = f"{r['file_path']}_{r['start_line']}_{r['end_line']}_{r['chunk_hash']}"
            data.append({
                "id": row_id,
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "content": r["content"],
                "file_hash": r["file_hash"],
                "chunk_hash": r["chunk_hash"],
                "mtime": r.get("mtime", 0.0),
                "size": r.get("size", 0),
                "vector": vectors[i].tolist(),
            })

        if data:
            self._table.add(data)

        # Update file_stats — one upsert per unique file in this batch
        seen: dict[str, dict] = {}
        for r in rows:
            seen[r["file_path"]] = r
        for fp, r in seen.items():
            self._upsert_file_stat(fp, r["file_hash"], r.get("mtime", 0.0), r.get("size", 0))

    def replace_file_chunks(
        self, file_path: str, rows: list[dict], vectors: np.ndarray
    ) -> None:
        """Replace chunks for a file: add new rows first, then delete old ones by ID.

        Adding before deleting means a crash leaves duplicates rather than gaps.
        Duplicates are harmless and cleaned up on the next index run, whereas
        gaps (from delete-then-add crashes) would cause files to disappear from search.
        """
        if len(rows) != len(vectors):
            raise ValueError("rows/vectors length mismatch")

        # Snapshot existing IDs before we add anything
        safe = file_path.replace("'", "''")
        existing_rows = (
            self._table.search()
            .where(f"file_path = '{safe}'")
            .select(["id"])
            .limit(None)
            .to_list()
        ) if self._table.count_rows() > 0 else []
        old_ids = [r["id"] for r in existing_rows]

        # Add new chunks first — crash here leaves both old and new (duplicates, not gaps)
        if rows:
            self.add_chunks(rows, vectors)

        # Now remove only the old rows by their IDs
        if old_ids:
            escaped = ", ".join(f"'{i.replace(chr(39), chr(39)+chr(39))}'" for i in old_ids)
            self._table.delete(f"id IN ({escaped})")

    # ------------------------------------------------------------------
    # ANN index
    # ------------------------------------------------------------------

    def build_index(self) -> bool:
        """Build an IVF-PQ ANN index when the table has enough rows.

        Returns True if an index was built, False if skipped (too few rows).
        """
        n = self._table.count_rows()
        if n < _INDEX_MIN_ROWS:
            return False
        num_partitions = min(256, max(1, n // 64))
        self._table.create_index(
            "vector",
            index_type="IVF_PQ",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=16,
            replace=True,
        )
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int = 8) -> list[dict]:
        """
        Cosine similarity search via LanceDB natively.
        Returns list of dicts: {file_path, start_line, end_line, content, score}.
        """
        if self._table.count_rows() == 0:
            return []

        results = (
            self._table.search(query_vec.tolist())
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

        parsed_results = []
        for r in results:
            # _distance is returned by LanceDB. For cosine, smaller distance = more similar.
            # Convert to similarity score: score = 1 - distance
            score = 1.0 - r.get("_distance", 0.0)
            parsed_results.append({
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "content": r["content"],
                "score": float(score),
            })

        return parsed_results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        total_chunks = self._table.count_rows()
        total_files = self._file_stats_table.count_rows()

        meta_rows = self._meta_table.search().limit(None).to_list()
        last_indexed = "never"
        for r in meta_rows:
            if r["key"] == "last_indexed":
                last_indexed = r["value"]
                break

        db_path = Path(self.db_path)
        db_size = (
            sum(f.stat().st_size for f in db_path.glob("**/*") if f.is_file())
            if db_path.exists()
            else 0
        )

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "last_indexed": last_indexed,
            "index_size_bytes": db_size,
        }

    def touch_last_indexed(self) -> None:
        ts = datetime.now(UTC).isoformat()
        self._meta_table.delete("key = 'last_indexed'")
        self._meta_table.add([{"key": "last_indexed", "value": ts}])
