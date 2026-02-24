# VecGrep

[![CI](https://github.com/iamvirul/VecGrep/actions/workflows/ci.yml/badge.svg)](https://github.com/iamvirul/VecGrep/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/iamvirul/VecGrep/branch/main/graph/badge.svg)](https://codecov.io/gh/iamvirul/VecGrep)
[![Discussions](https://img.shields.io/github/discussions/iamvirul/VecGrep)](https://github.com/iamvirul/VecGrep/discussions)

Cursor-style semantic code search as an MCP plugin for Claude Code.

Instead of grepping 50 files and sending 30,000 tokens to Claude, VecGrep returns the top 8 semantically relevant code chunks (~1,600 tokens). That's a **~95% token reduction** for codebase queries.

## How it works

1. **Chunk** — Parses source files with tree-sitter to extract semantic units (functions, classes, methods)
2. **Embed** — Encodes each chunk locally using `all-MiniLM-L6-v2` (384-dim, ~80MB one-time download)
3. **Store** — Saves embeddings + metadata in SQLite under `~/.vecgrep/<project_hash>/`
4. **Search** — Cosine similarity over all embeddings returns the most relevant snippets

Incremental re-indexing via SHA256 file hashing skips unchanged files.

## Architecture

![VecGrep architecture diagram](.github/res/diagram.jpeg)

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

> **Note:** Python 3.12 is required — `tree-sitter-languages` does not yet have wheels for Python 3.13+.

```bash
pip install vecgrep                        # standard pip
uv tool install --python 3.12 vecgrep     # uv tool (recommended)
```

## Claude Code integration

Run once — works for every project:

```bash
claude mcp add --scope user vecgrep -- vecgrep
```

This installs VecGrep as a persistent binary and registers it in your user config (`~/.claude.json`) so it's available globally across all projects. Starts instantly — no download delay on Claude Code launch.

## Usage with Claude

You don't trigger VecGrep manually - Claude decides when to call the tools based on what you ask.

| What you say to Claude | Tool invoked |
|---|---|
| "Index my project at /Users/me/myapp" | `index_codebase` |
| "How does authentication work in this codebase?" | `search_code` |
| "Find where database connections are set up" | `search_code` |
| "How many files are indexed?" | `get_index_status` |

**Typical first-time flow:**

```
You:    "Search for how payments are handled in /Users/me/myapp"
Claude: [calls index_codebase automatically since no index exists]
Claude: [calls search_code with your query]
Claude: "Here's how payments work — in src/payments.py:42..."
```

After the first index, subsequent searches skip unchanged files automatically — no re-indexing needed unless your code changes.

## Tools

### `index_codebase(path, force=False)`

Index a project directory. Skips unchanged files on subsequent calls.

```
index_codebase("/path/to/myproject")
# → "Indexed 142 file(s), 1847 chunk(s) added (0 file(s) skipped, unchanged)"
```

### `search_code(query, path, top_k=8)`

Semantic search. Auto-indexes if no index exists.

```
search_code("how does user authentication work", "/path/to/myproject")
```

Returns formatted snippets with file paths, line numbers, and similarity scores:

```
[1] src/auth.py:45-72 (score: 0.87)
def authenticate_user(token: str) -> User:
    ...

[2] src/middleware.py:12-28 (score: 0.81)
...
```

### `get_index_status(path)`

Check index statistics.

```
Index status for: /path/to/myproject
  Files indexed:  142
  Total chunks:   1847
  Last indexed:   2026-02-22T07:20:31+00:00
  Index size:     28.4 MB
```

## Supported languages

Python, JavaScript/TypeScript, Rust, Go, Java, C/C++, Ruby, Swift, Kotlin, C#

All other text files fall back to sliding-window line chunks.

## Index location

`~/.vecgrep/<sha256-of-project-path>/index.db`

Each project gets its own isolated index. Delete the directory to wipe the index.

## Community

| | |
|---|---|
| ❓ **Questions** | [Start a Q&A discussion](https://github.com/iamvirul/VecGrep/discussions/new?category=q-a) |
| 💡 **Ideas** | [Share an idea](https://github.com/iamvirul/VecGrep/discussions/new?category=ideas) |
| 🚀 **Show & Tell** | [Share how you use VecGrep](https://github.com/iamvirul/VecGrep/discussions/new?category=show-and-tell) |
| 🐛 **Bugs** | [Open an issue](https://github.com/iamvirul/VecGrep/issues/new) |
