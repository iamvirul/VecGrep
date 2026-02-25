"""Unit tests for VectorStore."""

from __future__ import annotations

import pytest

from tests.conftest import make_rows, make_unit_vecs
from vecgrep.store import VectorStore


class TestSearch:
    def test_self_match_score_is_one(self, store):
        vecs = make_unit_vecs(2)
        rows = make_rows(2)
        store.add_chunks(rows, vecs)

        results = store.search(vecs[0], top_k=1)
        assert len(results) == 1
        assert abs(results[0]["score"] - 1.0) < 1e-5
        assert results[0]["content"] == "content_0"

    def test_empty_store_returns_empty(self, store):
        vec = make_unit_vecs(1)[0]
        assert store.search(vec) == []

    def test_top_k_limits_results(self, store):
        vecs = make_unit_vecs(5)
        store.add_chunks(make_rows(5), vecs)
        assert len(store.search(vecs[0], top_k=3)) == 3

    def test_results_have_expected_keys(self, store):
        vecs = make_unit_vecs(1)
        store.add_chunks(make_rows(1), vecs)
        result = store.search(vecs[0], top_k=1)[0]
        assert {"file_path", "start_line", "end_line", "content", "score"} <= result.keys()





class TestDeleteFileChunks:
    def test_removes_chunks(self, store):
        vecs = make_unit_vecs(3)
        store.add_chunks(make_rows(3), vecs)
        assert store.status()["total_chunks"] == 3

        store.delete_file_chunks("a.py")
        assert store.status()["total_chunks"] == 0

    def test_delete_nonexistent_is_noop(self, store):
        store.delete_file_chunks("nonexistent.py")
        assert store.status()["total_chunks"] == 0


class TestReplaceFileChunks:
    def test_only_new_chunks_remain(self, store):
        vecs_old = make_unit_vecs(2)
        store.add_chunks(make_rows(2), vecs_old)

        new_vecs = make_unit_vecs(1, seed=99)
        new_rows = [
            {
                "file_path": "a.py",
                "start_line": 10,
                "end_line": 20,
                "content": "new_content",
                "file_hash": "newhash",
                "chunk_hash": "newchunkhash",
            }
        ]
        store.replace_file_chunks("a.py", new_rows, new_vecs)

        assert store.status()["total_chunks"] == 1
        results = store.search(new_vecs[0], top_k=5)
        assert len(results) == 1
        assert results[0]["content"] == "new_content"

    def test_raises_on_length_mismatch(self, store):
        vecs = make_unit_vecs(2)
        with pytest.raises(ValueError, match="mismatch"):
            store.replace_file_chunks("a.py", make_rows(1), vecs)


class TestContextManager:
    def test_enter_returns_self(self, tmp_path):
        with VectorStore(tmp_path / "cm") as store:
            assert isinstance(store, VectorStore)

    def test_exit_does_not_interfere(self, tmp_path):
        with VectorStore(tmp_path / "cm") as store:
            pass
        # LanceDB doesn't throw if we access it outside the block
        assert store.status()["total_chunks"] == 0


class TestStatus:
    def test_empty_store(self, store):
        s = store.status()
        assert s["total_chunks"] == 0
        assert s["total_files"] == 0
        assert s["last_indexed"] == "never"

    def test_reflects_added_chunks(self, store):
        vecs = make_unit_vecs(3)
        store.add_chunks(make_rows(3), vecs)
        s = store.status()
        assert s["total_chunks"] == 3
        assert s["total_files"] == 1

    def test_touch_last_indexed(self, store):
        store.touch_last_indexed()
        assert store.status()["last_indexed"] != "never"


class TestGetFileHashes:
    def test_empty(self, store):
        assert store.get_file_hashes() == {}

    def test_returns_correct_mapping(self, store):
        vecs = make_unit_vecs(2)
        rows = make_rows(2, file_path="src/foo.py", file_hash="abc123")
        store.add_chunks(rows, vecs)
        hashes = store.get_file_hashes()
        assert hashes == {"src/foo.py": "abc123"}

    def test_multiple_files(self, store):
        vecs = make_unit_vecs(2)
        rows_a = make_rows(1, file_path="a.py", file_hash="hash_a")
        rows_b = make_rows(1, file_path="b.py", file_hash="hash_b")
        store.add_chunks(rows_a, vecs[:1])
        store.add_chunks(rows_b, vecs[1:])
        hashes = store.get_file_hashes()
        assert hashes == {"a.py": "hash_a", "b.py": "hash_b"}
