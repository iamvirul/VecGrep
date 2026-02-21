"""Unit tests for AST chunking with mocks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vecgrep.chunker import chunk_file


def test_ast_chunks_mocked(tmp_path):
    # Mock node structure
    # Source:
    # def foo():
    #     pass
    #
    # Lines:
    # 0: def foo():
    # 1:     pass
    # 2:

    # Root node
    root = MagicMock()
    root.type = "module"
    root.children = []

    # Function node
    func_node = MagicMock()
    func_node.type = "function_definition"
    func_node.start_point = (0, 0)
    func_node.end_point = (1, 8)
    func_node.children = []

    root.children.append(func_node)

    # Mock tree
    tree = MagicMock()
    tree.root_node = root

    # Mock parser
    parser = MagicMock()
    parser.parse.return_value = tree

    # Mock get_parser
    with patch("tree_sitter_languages.get_parser", return_value=parser) as mock_get_parser:
        # Create file
        src = tmp_path / "mocked.py"
        src.write_text("def foo():\n    pass\n", encoding="utf-8")

        chunks = chunk_file(str(src))

        # Verify get_parser called
        mock_get_parser.assert_called()

        # Verify chunks
        assert len(chunks) == 1
        assert chunks[0].language == "python"
        assert "def foo():" in chunks[0].content
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2

def test_ast_chunks_mocked_large_function(tmp_path):
    # Mock a large function node that triggers splitting
    # 200 lines

    root = MagicMock()
    root.type = "module"

    func_node = MagicMock()
    func_node.type = "function_definition"
    func_node.start_point = (0, 0)
    func_node.end_point = (200, 10)
    func_node.children = []

    root.children = [func_node]

    tree = MagicMock()
    tree.root_node = root

    parser = MagicMock()
    parser.parse.return_value = tree

    with patch("tree_sitter_languages.get_parser", return_value=parser):
        src = tmp_path / "large.py"
        # Generate content with 201 lines
        lines = [f"line {i}" for i in range(201)]
        content = "\n".join(lines)
        src.write_text(content, encoding="utf-8")

        # We need to ensure content length > MAX_CHUNK_CHARS (1800)
        # 200 lines * ~7 chars = 1400. Need more.
        # Let's make lines longer.
        lines = [f"line {i} " * 10 for i in range(201)]
        content = "\n".join(lines)
        src.write_text(content, encoding="utf-8")

        chunks = chunk_file(str(src))

        # Should be split
        assert len(chunks) > 1
        assert chunks[0].language == "python"

def test_ast_chunks_no_matching_nodes_fallback(tmp_path):
    # Mock tree with no interesting nodes
    root = MagicMock()
    root.type = "module"
    root.children = [] # No children

    tree = MagicMock()
    tree.root_node = root

    parser = MagicMock()
    parser.parse.return_value = tree

    with patch("tree_sitter_languages.get_parser", return_value=parser):
        src = tmp_path / "fallback.py"
        src.write_text("print('hello')\n", encoding="utf-8")

        chunks = chunk_file(str(src))

        # Should fall back to sliding window (1 chunk)
        assert len(chunks) == 1
        # Sliding window still sets language to python
        assert chunks[0].language == "python"
        assert chunks[0].start_line == 1

def test_ast_chunks_empty_target_types(tmp_path):
    # This covers the 'if not target_types' check
    # We need to mock CHUNK_NODE_TYPES to return empty list for python

    # We still need get_parser to work
    parser = MagicMock()
    parser.parse.return_value = MagicMock()

    with patch("tree_sitter_languages.get_parser", return_value=parser):
        with patch.dict("vecgrep.chunker.CHUNK_NODE_TYPES", {"python": []}):
            src = tmp_path / "empty_types.py"
            src.write_text("def foo(): pass\n", encoding="utf-8")

            chunks = chunk_file(str(src))

            # Should fall back to sliding window
            assert len(chunks) == 1
            assert chunks[0].language == "python"

