"""Unit tests for chunk_file."""

from __future__ import annotations

from vecgrep.chunker import MAX_CHUNK_CHARS, SLIDING_WINDOW_LINES, chunk_file


class TestPythonFile:
    def test_python_file_yields_at_least_one_chunk(self, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(
            "def foo():\n    return 1\n\ndef bar():\n    return 2\n",
            encoding="utf-8",
        )
        chunks = chunk_file(str(src))
        assert len(chunks) >= 1

    def test_chunk_content_matches_source(self, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text("def foo():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(str(src))
        assert any("def foo" in c.content for c in chunks)

    def test_start_line_is_one_indexed(self, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text("def foo():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(str(src))
        assert all(c.start_line >= 1 for c in chunks)

    def test_end_line_gte_start_line(self, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text("def foo():\n    x = 1\n    return x\n", encoding="utf-8")
        chunks = chunk_file(str(src))
        assert all(c.end_line >= c.start_line for c in chunks)


class TestUnknownExtension:
    def test_falls_back_to_sliding_window(self, tmp_path):
        src = tmp_path / "data.xyz"
        lines = [f"line {i}" for i in range(60)]
        src.write_text("\n".join(lines), encoding="utf-8")
        chunks = chunk_file(str(src))
        assert len(chunks) >= 1
        assert chunks[0].start_line == 1

    def test_sliding_window_overlaps(self, tmp_path):
        src = tmp_path / "data.xyz"
        lines = [f"line {i}" for i in range(SLIDING_WINDOW_LINES + 10)]
        src.write_text("\n".join(lines), encoding="utf-8")
        chunks = chunk_file(str(src))
        assert len(chunks) >= 2


class TestEdgeCases:
    def test_empty_file_returns_empty(self, tmp_path):
        src = tmp_path / "empty.py"
        src.write_text("", encoding="utf-8")
        assert chunk_file(str(src)) == []

    def test_whitespace_only_returns_empty(self, tmp_path):
        src = tmp_path / "blank.py"
        src.write_text("   \n\n  \n", encoding="utf-8")
        assert chunk_file(str(src)) == []

    def test_nonexistent_file_returns_empty(self, tmp_path):
        assert chunk_file(str(tmp_path / "nope.py")) == []

    def test_oversized_function_splits(self, tmp_path):
        # Create a function body long enough to exceed MAX_CHUNK_CHARS
        body_lines = [f"    x_{i} = {i}" for i in range(200)]
        src_text = "def big_fn():\n" + "\n".join(body_lines) + "\n"
        assert len(src_text) > MAX_CHUNK_CHARS
        src = tmp_path / "big.py"
        src.write_text(src_text, encoding="utf-8")
        chunks = chunk_file(str(src))
        assert len(chunks) >= 2
        assert all(len(c.content) <= MAX_CHUNK_CHARS for c in chunks)


class TestRustFile:
    def test_rust_chunk_contains_fn_or_struct(self, tmp_path):
        src = tmp_path / "lib.rs"
        src.write_text(
            "pub fn hello() -> &'static str {\n    \"hello\"\n}\n\n"
            "pub struct Point { x: f64, y: f64 }\n",
            encoding="utf-8",
        )
        chunks = chunk_file(str(src))
        assert len(chunks) >= 1
        combined = " ".join(c.content for c in chunks)
        assert "fn" in combined or "struct" in combined


class TestNoTreeSitter:
    def test_falls_back_to_sliding_window_when_tree_sitter_unavailable(self, tmp_path):
        from unittest.mock import patch
        src = tmp_path / "mod.py"
        src.write_text("def foo():\n    return 1\n", encoding="utf-8")
        with patch("vecgrep.chunker.HAS_TREE_SITTER", False):
            chunks = chunk_file(str(src))
        assert len(chunks) >= 1
        assert chunks[0].start_line == 1
