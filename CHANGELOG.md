# Changelog

All notable changes to VecGrep are documented here.

---

## [1.5.0] — 2026-02-28

### Breaking Changes

- **Store backend replaced**: SQLite + numpy vector store replaced with LanceDB.
  Existing indexes at `~/.vecgrep/` must be re-indexed after upgrading —
  simply run `index_codebase` again on your project.

### Added

**Performance**
- **fastembed ONNX backend** — default embedding backend switched from
  sentence-transformers (PyTorch) to fastembed (ONNX Runtime).
  MCP server startup: ~6.6s → ~1.25s. First embed call: ~2–3s → ~100ms.
- **IVF-PQ ANN index** — `build_index()` creates an approximate nearest
  neighbour index after indexing for sub-linear search on large codebases.
- **`file_stats` table** — O(files) change detection table. Incremental
  re-indexing now reads one row per file instead of scanning all chunks.
- **Auto device detection** — selects Metal (Apple Silicon), CUDA (NVIDIA),
  or CPU automatically for the torch backend.

**Configuration**
- `VECGREP_BACKEND` env var — `onnx` (default, fast) or `torch` (any HF model).
- `VECGREP_MODEL` env var — override the default HuggingFace model ID.
- Default model switched to [`isuruwijesiri/all-MiniLM-L6-v2-code-search-512`](https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512),
  fine-tuned for semantic code search (384-dim, ~80MB one-time download).

**MCP tools**
- `stop_watching` — new tool to explicitly stop watching a codebase path.
- Watch state persisted to `~/.vecgrep/watched.json` and restored on restart.
- Background startup sync via daemon thread — server starts immediately.

**Developer experience**
- `.githooks/pre-commit` lint check — run `git config core.hooksPath .githooks`
  once per clone to enable.

### Fixed

- `build_index()` passed `"vector"` as positional arg to `create_index()`,
  mapping to `metric` instead of `vector_column_name`.
- `list_tables()` returned `ListTablesResponse` instead of a plain list,
  causing `set()` to fail with an unhashable type error.

### Performance summary

| Metric | 1.0.0 | 1.5.0 |
|---|---|---|
| MCP server startup | ~6.6s | ~1.25s |
| Model load (first embed) | ~2–3s | ~100ms |
| Change detection | O(chunks) SHA-256 | O(files) mtime+size |

---

## [1.0.0] — 2026-02-22

First stable release.

### Added

**Core features**
- AST-based code chunking via `tree-sitter-languages` for Python, JavaScript/TypeScript, Rust, Go, Java, C/C++, Ruby, Swift, Kotlin, and C#
- Local embeddings using `all-MiniLM-L6-v2` (384-dim, ~80 MB one-time download, no API key required)
- SQLite-backed vector store at `~/.vecgrep/<project-hash>/index.db`
- Cosine similarity search with in-memory embedding cache (eliminates per-query full-table scan)
- Incremental indexing via SHA-256 file hashing — unchanged files are skipped on re-index
- Orphan cleanup — chunks for deleted files are removed on the next index run
- Three MCP tools: `index_codebase`, `search_code`, `get_index_status`

**Robustness**
- Per-path threading lock prevents concurrent indexing corruption
- Atomic file updates via `replace_file_chunks()` (DELETE + INSERT in a single transaction)
- Context-manager support on `VectorStore` guarantees connection closure on exceptions
- `top_k` clamped to `max(1, min(top_k, 20))` — negative values no longer crash
- Query validation: empty queries and queries over 500 characters return a clean error string
- `followlinks=False` in directory walk prevents symlink loops
- Try/except on all MCP tool functions — errors surface as readable strings, not tracebacks

**Tooling**
- 52 tests across `test_store`, `test_chunker`, `test_embedder`, and `test_server`
- CI with ruff lint, pytest with coverage upload to Codecov, and non-blocking pyright
- Published to PyPI — install with `pip install vecgrep` or run directly with `uvx vecgrep`

### Changed

- `pyproject.toml`: tightened dependency version ranges, added dev extras (`pytest`, `ruff`, `pyright`)

---

## [0.1.0] — 2026-02-22 *(pre-release, not published)*

Initial working prototype — basic indexing and search functional but without hardening, tests, or PyPI packaging.
