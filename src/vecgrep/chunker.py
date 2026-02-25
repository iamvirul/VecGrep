"""AST-based code chunking using tree-sitter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from tree_sitter_languages import get_parser  # type: ignore

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

MAX_CHUNK_CHARS = 1800  # ~512 tokens
SLIDING_WINDOW_LINES = 50
SLIDING_WINDOW_OVERLAP = 25

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".cs": "c_sharp",
}

# Node types to extract per language
CHUNK_NODE_TYPES: dict[str, list[str]] = {
    "python": [
        "function_definition",
        "async_function_definition",
        "class_definition",
        "decorated_definition",
    ],
    "javascript": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "export_statement",
    ],
    "typescript": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "export_statement",
    ],
    "tsx": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "export_statement",
    ],
    "rust": [
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
    ],
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "interface_type",
    ],
    "java": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
    ],
    "c": [
        "function_definition",
        "struct_specifier",
    ],
    "cpp": [
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    ],
    "ruby": [
        "method",
        "singleton_method",
        "class",
        "module",
    ],
    "swift": [
        "function_declaration",
        "class_declaration",
        "struct_declaration",
        "protocol_declaration",
        "extension_declaration",
    ],
    "kotlin": [
        "function_declaration",
        "class_declaration",
        "object_declaration",
        "interface_declaration",
    ],
    "c_sharp": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
        "property_declaration",
    ],
}


@dataclass
class Chunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str


def _split_large_chunk(content: str, file_path: str, start_line: int, language: str) -> list[Chunk]:
    """Split oversized chunks by lines."""
    lines = content.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        batch = lines[i : i + SLIDING_WINDOW_LINES]
        chunk_content = "\n".join(batch)
        chunks.append(
            Chunk(
                content=chunk_content,
                file_path=file_path,
                start_line=start_line + i,
                end_line=start_line + i + len(batch) - 1,
                language=language,
            )
        )
        i += SLIDING_WINDOW_LINES - SLIDING_WINDOW_OVERLAP
        if i >= len(lines):
            break
    return chunks


def _sliding_window_chunks(source: str, file_path: str, language: str) -> list[Chunk]:
    """Fallback: sliding window over lines for unsupported languages."""
    lines = source.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        batch = lines[i : i + SLIDING_WINDOW_LINES]
        content = "\n".join(batch)
        chunks.append(
            Chunk(
                content=content,
                file_path=file_path,
                start_line=i + 1,
                end_line=i + len(batch),
                language=language,
            )
        )
        step = SLIDING_WINDOW_LINES - SLIDING_WINDOW_OVERLAP
        i += step
        if i >= len(lines):
            break
    return chunks


def _ast_chunks(source: str, file_path: str, language: str) -> list[Chunk]:
    """Extract semantic chunks from AST nodes."""
    if not HAS_TREE_SITTER:
        return _sliding_window_chunks(source, file_path, language)
    try:
        parser = get_parser(language)
    except Exception:
        return _sliding_window_chunks(source, file_path, language)

    tree = parser.parse(source.encode())
    target_types = set(CHUNK_NODE_TYPES.get(language, []))
    if not target_types:
        return _sliding_window_chunks(source, file_path, language)

    lines = source.splitlines()
    chunks: list[Chunk] = []
    seen_ranges: set[tuple[int, int]] = set()

    def visit(node) -> None:
        if node.type in target_types:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            span = (start_line, end_line)
            if span not in seen_ranges:
                seen_ranges.add(span)
                content = "\n".join(lines[start_line : end_line + 1])
                if len(content) > MAX_CHUNK_CHARS:
                    sub = _split_large_chunk(content, file_path, start_line + 1, language)
                    chunks.extend(sub)
                else:
                    chunks.append(
                        Chunk(
                            content=content,
                            file_path=file_path,
                            start_line=start_line + 1,
                            end_line=end_line + 1,
                            language=language,
                        )
                    )
                # Don't recurse into matched nodes to avoid duplication
                return
        for child in node.children:
            visit(child)

    visit(tree.root_node)

    # If no semantic nodes found, fall back to sliding window
    if not chunks:
        return _sliding_window_chunks(source, file_path, language)

    return chunks


def chunk_file(file_path: str) -> list[Chunk]:
    """Chunk a source file into semantic units."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    language = LANGUAGE_MAP.get(suffix)

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if not source.strip():
        return []

    if language and language in CHUNK_NODE_TYPES:
        return _ast_chunks(source, file_path, language)
    else:
        lang_label = language or path.suffix.lstrip(".") or "text"
        return _sliding_window_chunks(source, file_path, lang_label)
