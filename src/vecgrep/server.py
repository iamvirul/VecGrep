
"""FastMCP server exposing VecGrep tools."""

from __future__ import annotations

import atexit
import fnmatch
import hashlib
import logging
import os
import threading
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from vecgrep.chunker import chunk_file
from vecgrep.embedder import _detect_device, embed
from vecgrep.store import VectorStore

_log = logging.getLogger(__name__)


def _stop_all_observers() -> None:
    """Stop all background watchdog threads on process exit."""
    for observer in list(_OBSERVER_REGISTRY.values()):
        try:
            observer.stop()
            observer.join(timeout=2)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

mcp = FastMCP("vecgrep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECGREP_HOME = Path.home() / ".vecgrep"

ALWAYS_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "eggs",
    ".eggs",
    "htmlcov",
}

ALWAYS_SKIP_PATTERNS = [
    "*.min.js",
    "*.bundle.js",
    "*.lock",
    "*.pyc",
    "*.class",
    "*.o",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.DS_Store",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.whl",
    "*.egg",
]

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go",
    ".java", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".rb", ".swift", ".kt", ".cs", ".md", ".txt", ".yaml",
    ".yml", ".toml", ".json", ".sh", ".bash", ".zsh",
    ".fish", ".html", ".css", ".scss", ".less", ".sql",
    ".graphql", ".proto", ".tf", ".hcl", ".dockerfile",
    ".vue", ".svelte",
}

MAX_FILE_BYTES = 512 * 1024  # 512 KB — skip very large files
EMBED_BATCH = 64

# ---------------------------------------------------------------------------
# Per-path indexing locks
# ---------------------------------------------------------------------------

_LOCK_REGISTRY: dict[str, threading.Lock] = {}
_LOCK_REGISTRY_LOCK = threading.Lock()


def _get_index_lock(path: str) -> threading.Lock:
    """Return (and create if needed) a per-path threading.Lock."""
    with _LOCK_REGISTRY_LOCK:
        if path not in _LOCK_REGISTRY:
            _LOCK_REGISTRY[path] = threading.Lock()
        return _LOCK_REGISTRY[path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_hash(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def _get_store(path: str) -> VectorStore:
    index_dir = VECGREP_HOME / _project_hash(path)
    return VectorStore(index_dir)


def _sha256_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _load_gitignore(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    patterns: list[str] = []
    if gitignore.exists():
        for line in gitignore.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


# ---------------------------------------------------------------------------
# Background File Watcher
# ---------------------------------------------------------------------------

_OBSERVER_REGISTRY: dict[str, Observer] = {}
atexit.register(_stop_all_observers)


class LiveSyncHandler(FileSystemEventHandler):
    def __init__(self, root_path: str, gitignore_patterns: list[str]):
        self.root_path = root_path
        self.gitignore_patterns = gitignore_patterns
        self._debounce_timers: dict[str, threading.Timer] = {}
        self._debounce_delay = 2.0  # seconds

    def _process_file(self, file_path_str: str) -> None:
        file_path = Path(file_path_str)
        if not file_path.exists() or not file_path.is_file():
            return

        try:
            rel = str(file_path.relative_to(self.root_path))
        except ValueError:
            rel = str(file_path)

        if _is_ignored_by_gitignore(rel, self.gitignore_patterns) or _should_skip_file(file_path):
            return

        # Perform targeted index for this single file
        lock = _get_index_lock(self.root_path)
        if not lock.acquire(blocking=False):
            return  # Wait for full index to finish or let a future event pick it up

        try:
            with _get_store(self.root_path) as store:
                # Fast track: check if actually changed by stat
                # (watchdog can be trigger-happy with save sequences)
                stats = store.get_file_stats()
                fp_str = str(file_path)

                try:
                    stat_res = file_path.stat()
                    current_mtime = stat_res.st_mtime
                    current_size = stat_res.st_size
                except OSError:
                    return

                existing = stats.get(fp_str)
                if (
                    existing
                    and existing["mtime"] == current_mtime
                    and existing["size"] == current_size
                ):
                    return

                # It changed, process it
                file_hash = _sha256_file(file_path)
                chunks = chunk_file(fp_str)

                if not chunks:
                    store.delete_file_chunks(fp_str)
                    return

                rows = [
                    {
                        "file_path": fp_str,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "content": c.content,
                        "file_hash": file_hash,
                        "chunk_hash": _sha256_str(c.content),
                        "mtime": current_mtime,
                        "size": current_size,
                    }
                    for c in chunks
                ]
                texts = [c.content for c in chunks]

                batch_vecs: list[np.ndarray] = []
                for i in range(0, len(texts), EMBED_BATCH):
                    batch_vecs.append(embed(texts[i : i + EMBED_BATCH]))
                vecs = np.concatenate(batch_vecs)

                store.replace_file_chunks(fp_str, rows, vecs)
                store.touch_last_indexed()
        except Exception:
            _log.exception("LiveSync error processing %s", file_path_str)
        finally:
            lock.release()

    def _schedule_processing(self, file_path: str) -> None:
        if file_path in self._debounce_timers:
            self._debounce_timers[file_path].cancel()

        def _run_and_cleanup() -> None:
            self._process_file(file_path)
            self._debounce_timers.pop(file_path, None)

        timer = threading.Timer(self._debounce_delay, _run_and_cleanup)
        self._debounce_timers[file_path] = timer
        timer.start()

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_processing(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_processing(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            root = self.root_path
            fp = event.src_path
            if fp in self._debounce_timers:
                self._debounce_timers[fp].cancel()

            def _run_and_cleanup() -> None:
                self._delete_file(root, fp)
                self._debounce_timers.pop(fp, None)

            timer = threading.Timer(self._debounce_delay, _run_and_cleanup)
            self._debounce_timers[fp] = timer
            timer.start()

    def _delete_file(self, root: str, file_path: str) -> None:
        lock = _get_index_lock(root)
        if lock.acquire(blocking=False):
            try:
                with _get_store(root) as store:
                    store.delete_file_chunks(file_path)
            finally:
                lock.release()

def _ensure_watcher(root: Path, gitignore: list[str]) -> None:
    root_str = str(root)
    with _LOCK_REGISTRY_LOCK:
        if root_str not in _OBSERVER_REGISTRY:
            observer = Observer()
            handler = LiveSyncHandler(root_str, gitignore)
            observer.schedule(handler, root_str, recursive=True)
            observer.start()
            _OBSERVER_REGISTRY[root_str] = observer


def _is_ignored_by_gitignore(rel_path: str, patterns: list[str]) -> bool:
    parts = Path(rel_path).parts
    for pattern in patterns:
        # Match against the full relative path
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # Match against each path component
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _should_skip_file(file_path: Path) -> bool:
    name = file_path.name
    for pattern in ALWAYS_SKIP_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return True
    try:
        if file_path.stat().st_size > MAX_FILE_BYTES:
            return True
    except OSError:
        return True
    return False


def _walk_files(root: Path, gitignore_patterns: list[str]) -> list[Path]:
    """Collect all indexable files under root."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in ALWAYS_SKIP_DIRS
            and not _is_ignored_by_gitignore(
                str(Path(dirpath).relative_to(root) / d), gitignore_patterns
            )
        ]
        for fname in filenames:
            fp = Path(dirpath) / fname
            try:
                rel = str(fp.relative_to(root))
            except ValueError:
                rel = str(fp)
            if _is_ignored_by_gitignore(rel, gitignore_patterns):
                continue
            if _should_skip_file(fp):
                continue
            files.append(fp)
    return files


def _do_index(path: str, force: bool = False, watch: bool = False) -> str:
    root = Path(path).resolve()
    if not root.exists():
        return f"Error: path does not exist: {path}"

    lock = _get_index_lock(str(root))
    if not lock.acquire(blocking=False):
        return f"Error: indexing of {path} is already in progress"

    try:
        gitignore = _load_gitignore(root)
        all_files = _walk_files(root, gitignore)

        with _get_store(str(root)) as store:
            existing_stats = {} if force else store.get_file_stats()

            # Orphan cleanup: remove chunks for files no longer on disk
            all_file_strs = {str(fp) for fp in all_files}
            orphan_paths = set(existing_stats) - all_file_strs
            for p in orphan_paths:
                store.delete_file_chunks(p)

            files_changed = 0
            files_skipped = 0
            files_skipped_chunking = 0
            total_new_chunks = 0

            for fp in all_files:
                fp_str = str(fp)

                try:
                    stat_res = fp.stat()
                    current_mtime = stat_res.st_mtime
                    current_size = stat_res.st_size
                except OSError:
                    files_skipped += 1
                    continue

                if not force:
                    existing = existing_stats.get(fp_str)
                    if (
                        existing
                        and existing["mtime"] == current_mtime
                        and existing["size"] == current_size
                    ):
                        files_skipped += 1
                        continue

                # File changed or is new, calculate hash for metadata
                file_hash = _sha256_file(fp)
                is_existing = fp_str in existing_stats

                chunks = chunk_file(fp_str)
                if not chunks:
                    files_skipped_chunking += 1
                    # If it existed before but now has no chunks, clear it
                    if is_existing:
                        store.delete_file_chunks(fp_str)
                    continue

                rows = [
                    {
                        "file_path": fp_str,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "content": c.content,
                        "file_hash": file_hash,
                        "chunk_hash": _sha256_str(c.content),
                        "mtime": current_mtime,
                        "size": current_size,
                    }
                    for c in chunks
                ]
                texts = [c.content for c in chunks]

                batch_vecs: list[np.ndarray] = []
                for i in range(0, len(texts), EMBED_BATCH):
                    batch_vecs.append(embed(texts[i : i + EMBED_BATCH]))
                vecs = np.concatenate(batch_vecs)

                if is_existing:
                    store.replace_file_chunks(fp_str, rows, vecs)
                else:
                    store.add_chunks(rows, vecs)

                files_changed += 1
                total_new_chunks += len(chunks)

            store.touch_last_indexed()

            # Build ANN index when files changed (enables sub-linear search on large repos)
            if files_changed > 0:
                store.build_index()

        # Only start the background watcher when explicitly requested
        if watch:
            _ensure_watcher(root, gitignore)

        return (
            f"Indexed {files_changed} file(s), {total_new_chunks} chunk(s) added, "
            f"{len(orphan_paths)} orphan(s) removed "
            f"({files_skipped} file(s) skipped, unchanged)"
        )
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def index_codebase(path: str, force: bool = False, watch: bool = False) -> str:
    """
    Index a codebase directory for semantic search.

    Walks the directory, extracts semantic code chunks using AST analysis,
    embeds them locally with sentence-transformers, and stores in a vector index.
    Subsequent calls skip unchanged files (incremental updates).

    Args:
        path: Absolute path to the codebase root directory.
        force: If True, re-index all files even if unchanged.
        watch: If True, start a background watcher for live sync on file changes.

    Returns:
        Summary: files indexed, chunks added, files skipped.
    """
    try:
        return _do_index(path, force=force, watch=watch)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def search_code(query: str, path: str, top_k: int = 8) -> str:
    """
    Semantically search an indexed codebase for code relevant to a query.

    Embeds the query and performs cosine similarity search against indexed
    code chunks, returning the most semantically relevant snippets with
    file paths and line numbers.

    If the codebase is not yet indexed, it will be indexed automatically first.

    Args:
        query: Natural language description of what you're looking for.
               E.g. "how does authentication work", "database connection setup"
        path: Absolute path to the codebase root directory.
        top_k: Number of results to return (default 8, max 20).

    Returns:
        Formatted list of matching code chunks with file:line references and
        similarity scores.
    """
    try:
        if len(query) > 500:
            return "Error: query is too long (max 500 characters)"
        if not query.strip():
            return "Error: query must not be empty"

        top_k = max(1, min(top_k, 20))
        root = Path(path).resolve()

        # Check if index has data
        with _get_store(str(root)) as store:
            needs_index = store.status()["total_chunks"] == 0

        index_result: str | None = None
        if needs_index:
            index_result = _do_index(str(root), force=False)

        with _get_store(str(root)) as store:
            if store.status()["total_chunks"] == 0:
                msg = f"No indexable files found in {path}."
                if index_result:
                    msg += f"\n(Index attempt: {index_result})"
                return msg

            query_vec = embed([query])[0]
            results = store.search(query_vec, top_k=top_k)

        if not results:
            return "No results found. Try re-indexing with index_codebase()."

        lines = [f"Top {len(results)} results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            try:
                rel = str(Path(r["file_path"]).relative_to(root))
            except ValueError:
                rel = r["file_path"]
            lines.append(
                f"[{i}] {rel}:{r['start_line']}-{r['end_line']} (score: {r['score']:.2f})"
            )
            lines.append(r["content"])
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_index_status(path: str) -> str:
    """
    Get the status of the vector index for a codebase.

    Args:
        path: Absolute path to the codebase root directory.

    Returns:
        Index statistics: file count, chunk count, last indexed time, disk usage.
    """
    try:
        root = Path(path).resolve()
        with _get_store(str(root)) as store:
            s = store.status()

        size_mb = s["index_size_bytes"] / (1024 * 1024)
        device = _detect_device()
        device_label = {"cuda": "CUDA (GPU)", "mps": "Metal (Apple Silicon)", "cpu": "CPU"}.get(
            device, device
        )
        return (
            f"Index status for: {root}\n"
            f"  Files indexed:  {s['total_files']}\n"
            f"  Total chunks:   {s['total_chunks']}\n"
            f"  Last indexed:   {s['last_indexed']}\n"
            f"  Index size:     {size_mb:.1f} MB\n"
            f"  Compute device: {device_label}"
        )
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
