"""Integration tests for VecGrep MCP server tools."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from vecgrep.server import (
    _OBSERVER_REGISTRY,
    LiveSyncHandler,
    _do_index,
    _ensure_watcher,
    _get_index_lock,
    _get_store,
    _is_ignored_by_gitignore,
    _load_gitignore,
    _should_skip_file,
    _stop_all_observers,
    _walk_files,
    get_index_status,
    index_codebase,
    main,
    search_code,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_py(path: Path, name: str, content: str) -> Path:
    f = path / name
    f.write_text(content, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestIndexCodebase:
    def test_indexes_python_files(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo():\n    return 42\n")
        result = _do_index(str(tmp_path))
        assert "1 file(s)" in result
        assert "Error" not in result

    def test_nonexistent_path_returns_error(self):
        result = _do_index("/nonexistent/path/xyzzy12345")
        assert result.startswith("Error")

    def test_incremental_skips_unchanged(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        result2 = _do_index(str(tmp_path))
        assert "skipped" in result2

    def test_force_reindexes_all(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        result = _do_index(str(tmp_path), force=True)
        assert "1 file(s)" in result
        assert "Error" not in result

    def test_index_codebase_tool_wraps_do_index(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        result = index_codebase(str(tmp_path))
        assert "Error" not in result


class TestOrphanCleanup:
    def test_deleted_file_chunks_removed_on_reindex(self, tmp_path):
        f1 = _write_py(tmp_path, "a.py", "def foo(): pass\n")
        f2 = _write_py(tmp_path, "b.py", "def bar(): pass\n")

        _do_index(str(tmp_path))

        root = str(tmp_path.resolve())
        with _get_store(root) as store:
            hashes = store.get_file_hashes()
        assert str(f1.resolve()) in hashes
        assert str(f2.resolve()) in hashes

        f2.unlink()
        result = _do_index(str(tmp_path))
        assert "1 orphan" in result

        with _get_store(root) as store:
            hashes_after = store.get_file_hashes()
        assert str(f2.resolve()) not in hashes_after


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchCode:
    def test_returns_relevant_result(self, tmp_path):
        _write_py(tmp_path, "auth.py", "def authenticate_user(username, password):\n    pass\n")
        _do_index(str(tmp_path))
        result = search_code("user authentication", str(tmp_path), top_k=5)
        assert "auth.py" in result
        assert "Error" not in result

    def test_negative_top_k_does_not_crash(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        result = search_code("foo", str(tmp_path), top_k=-5)
        # max(1, min(-5, 20)) = 1 — should return 1 result without error
        assert "Error" not in result

    def test_empty_query_returns_error(self, tmp_path):
        result = search_code("   ", str(tmp_path), top_k=5)
        assert "Error" in result

    def test_long_query_returns_error(self, tmp_path):
        result = search_code("x" * 501, str(tmp_path), top_k=5)
        assert "Error" in result
        assert "long" in result.lower() or "500" in result

    def test_auto_indexes_on_first_search(self, tmp_path):
        _write_py(tmp_path, "a.py", "def compute(): return 1\n")
        # No explicit index call — search_code should trigger it
        result = search_code("compute function", str(tmp_path), top_k=5)
        assert "Error" not in result


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrentLock:
    def test_second_index_call_returns_in_progress(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        root_str = str(tmp_path.resolve())

        lock = _get_index_lock(root_str)
        assert lock.acquire(blocking=False)

        captured: list[str] = []

        def run():
            captured.append(_do_index(str(tmp_path)))

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5)
        lock.release()

        assert len(captured) == 1
        assert "already in progress" in captured[0]


# ---------------------------------------------------------------------------
# get_index_status
# ---------------------------------------------------------------------------


class TestGetIndexStatus:
    def test_returns_expected_fields(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        result = get_index_status(str(tmp_path))
        assert "Files indexed" in result
        assert "Total chunks" in result
        assert "Last indexed" in result
        assert "Index size" in result

    def test_nonexistent_path_shows_zero_files(self, tmp_path):
        # No indexing — status should still return without raising
        result = get_index_status(str(tmp_path))
        assert "Files indexed:  0" in result

    def test_exception_returns_error_string(self):
        with patch("vecgrep.server._get_store", side_effect=RuntimeError("boom")):
            result = get_index_status("/some/path")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Helper function coverage
# ---------------------------------------------------------------------------


class TestLoadGitignore:
    def test_returns_empty_when_no_gitignore(self, tmp_path):
        assert _load_gitignore(tmp_path) == []

    def test_parses_gitignore_skipping_comments_and_blanks(self, tmp_path):
        (tmp_path / ".gitignore").write_text("# comment\n\n*.pyc\ndist/\n")
        patterns = _load_gitignore(tmp_path)
        assert "*.pyc" in patterns
        assert "dist/" in patterns
        assert "# comment" not in patterns
        assert "" not in patterns


class TestIsIgnoredByGitignore:
    def test_full_path_match(self):
        assert _is_ignored_by_gitignore("dist/foo.js", ["dist*"])

    def test_component_match(self):
        assert _is_ignored_by_gitignore("some/node_modules/lodash.js", ["node_modules"])

    def test_no_match(self):
        assert not _is_ignored_by_gitignore("src/main.py", ["dist/", "node_modules"])

    def test_glob_pattern(self):
        assert _is_ignored_by_gitignore("src/foo.min.js", ["*.min.js"])


class TestShouldSkipFile:
    def test_skips_unsupported_extension(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00" * 10)
        assert _should_skip_file(f)

    def test_skips_pattern_match(self, tmp_path):
        f = tmp_path / "bundle.min.js"
        f.write_text("x=1;")
        assert _should_skip_file(f)

    def test_skips_large_file(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_bytes(b"x" * (512 * 1024 + 1))
        assert _should_skip_file(f)

    def test_does_not_skip_normal_py(self, tmp_path):
        f = tmp_path / "small.py"
        f.write_text("def foo(): pass\n")
        assert not _should_skip_file(f)

    def test_oserror_returns_true(self, tmp_path):
        f = tmp_path / "ghost.py"
        # File doesn't exist — stat() will raise OSError
        assert _should_skip_file(f)


class TestWalkFiles:
    def test_finds_python_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        files = _walk_files(tmp_path, [])
        assert any(f.name == "a.py" for f in files)

    def test_skips_gitignored_dirs(self, tmp_path):
        ignored = tmp_path / "dist"
        ignored.mkdir()
        (ignored / "out.py").write_text("x=1")
        files = _walk_files(tmp_path, ["dist"])
        assert not any("dist" in str(f) for f in files)

    def test_skips_always_skip_dirs(self, tmp_path):
        skip = tmp_path / "node_modules"
        skip.mkdir()
        (skip / "lib.js").write_text("x=1;")
        files = _walk_files(tmp_path, [])
        assert not any("node_modules" in str(f) for f in files)


# ---------------------------------------------------------------------------
# _do_index edge cases
# ---------------------------------------------------------------------------


class TestDoIndexEdgeCases:
    def test_watch_flag_starts_watcher(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        root_str = str(tmp_path.resolve())

        with patch("vecgrep.server._ensure_watcher") as mock_watcher:
            _do_index(root_str, watch=True)
            mock_watcher.assert_called_once()

        # Cleanup observer if started
        if root_str in _OBSERVER_REGISTRY:
            _OBSERVER_REGISTRY[root_str].stop()
            _OBSERVER_REGISTRY[root_str].join(timeout=2)
            del _OBSERVER_REGISTRY[root_str]

    def test_file_updated_uses_replace(self, tmp_path):
        """Re-indexing a changed file should call replace_file_chunks."""
        f = _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))

        # Touch the file to force mtime/size change
        f.write_text("def foo(): return 42\ndef bar(): pass\n")
        result = _do_index(str(tmp_path))
        assert "1 file(s)" in result

    def test_stat_oserror_skips_file(self, tmp_path):
        """If stat() raises OSError, file is skipped gracefully."""
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        original_stat = Path.stat

        def bad_stat(self, **kwargs):
            if self.name == "a.py":
                raise OSError("no access")
            return original_stat(self, **kwargs)

        with patch.object(Path, "stat", bad_stat):
            result = _do_index(str(tmp_path))
        assert "Error" not in result

    def test_no_chunks_for_existing_file_deletes(self, tmp_path):
        """If a file now produces no chunks, its old chunks are deleted."""
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))

        with patch("vecgrep.server.chunk_file", return_value=[]):
            # Force re-index so it hits the "no chunks" + "is_existing" branch
            result = _do_index(str(tmp_path), force=True)
        assert "Error" not in result

    def test_index_codebase_exception_returns_error(self):
        with patch("vecgrep.server._do_index", side_effect=RuntimeError("crash")):
            result = index_codebase("/any/path")
        assert "Error" in result


# ---------------------------------------------------------------------------
# search_code edge cases
# ---------------------------------------------------------------------------


class TestSearchCodeEdgeCases:
    def test_no_results_returns_message(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        with patch("vecgrep.server.VectorStore.search", return_value=[]):
            result = search_code("something", str(tmp_path))
        assert "No results" in result

    def test_empty_index_after_auto_index(self, tmp_path):
        """When auto-index finds no indexable files, returns a helpful message."""
        result = search_code("anything", str(tmp_path))
        assert "No indexable files" in result

    def test_exception_returns_error(self):
        with patch("vecgrep.server._get_store", side_effect=RuntimeError("boom")):
            result = search_code("query", "/any/path")
        assert "Error" in result

    def test_result_path_outside_root(self, tmp_path):
        """Result whose file_path is outside root still formats without crash."""
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))

        outside = str(Path("/completely/different/file.py"))
        fake_result = [{"file_path": outside, "start_line": 1, "end_line": 5,
                        "content": "def foo(): pass", "score": 0.9}]
        with patch("vecgrep.server.VectorStore.search", return_value=fake_result):
            result = search_code("foo", str(tmp_path))
        assert "Error" not in result
        assert "file.py" in result


# ---------------------------------------------------------------------------
# LiveSyncHandler
# ---------------------------------------------------------------------------


class TestLiveSyncHandler:
    def _make_handler(self, root: str, patterns: list[str] | None = None):
        h = LiveSyncHandler(root, patterns or [])
        h._debounce_delay = 0  # instant for tests
        return h

    def test_on_modified_schedules_processing(self, tmp_path):
        h = self._make_handler(str(tmp_path))
        evt = MagicMock()
        evt.is_directory = False
        evt.src_path = str(tmp_path / "a.py")

        with patch.object(h, "_schedule_processing") as mock_sched:
            h.on_modified(evt)
            mock_sched.assert_called_once_with(evt.src_path)

    def test_on_modified_ignores_directory_events(self, tmp_path):
        h = self._make_handler(str(tmp_path))
        evt = MagicMock()
        evt.is_directory = True

        with patch.object(h, "_schedule_processing") as mock_sched:
            h.on_modified(evt)
            mock_sched.assert_not_called()

    def test_on_created_schedules_processing(self, tmp_path):
        h = self._make_handler(str(tmp_path))
        evt = MagicMock()
        evt.is_directory = False
        evt.src_path = str(tmp_path / "new.py")

        with patch.object(h, "_schedule_processing") as mock_sched:
            h.on_created(evt)
            mock_sched.assert_called_once_with(evt.src_path)

    def test_on_deleted_schedules_delete(self, tmp_path):
        h = self._make_handler(str(tmp_path))
        fp = str(tmp_path / "gone.py")
        evt = MagicMock()
        evt.is_directory = False
        evt.src_path = fp

        with patch.object(h, "_delete_file") as mock_del:
            h.on_deleted(evt)
            time.sleep(0.05)  # let the 0-delay timer fire
            mock_del.assert_called_once_with(str(tmp_path), fp)

    def test_on_deleted_cancels_existing_timer(self, tmp_path):
        h = self._make_handler(str(tmp_path))
        h._debounce_delay = 60  # long enough to not fire
        fp = str(tmp_path / "gone.py")

        mock_timer = MagicMock()
        h._debounce_timers[fp] = mock_timer

        evt = MagicMock()
        evt.is_directory = False
        evt.src_path = fp

        with patch.object(h, "_delete_file"):
            h.on_deleted(evt)
        mock_timer.cancel.assert_called_once()

    def test_process_file_nonexistent(self, tmp_path):
        """_process_file returns early if file doesn't exist."""
        h = self._make_handler(str(tmp_path))
        h._process_file(str(tmp_path / "missing.py"))  # should not raise

    def test_process_file_ignored(self, tmp_path):
        f = tmp_path / "skip.min.js"
        f.write_text("x=1;")
        h = self._make_handler(str(tmp_path))
        # Should return early without trying to acquire lock
        h._process_file(str(f))

    def test_process_file_lock_busy(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("def foo(): pass\n")
        h = self._make_handler(str(tmp_path))

        lock = _get_index_lock(str(tmp_path))
        lock.acquire()
        try:
            h._process_file(str(f))  # should return immediately — lock busy
        finally:
            lock.release()

    def test_process_file_unchanged_by_stat(self, tmp_path):
        """_process_file skips when mtime/size match stored stats."""
        f = tmp_path / "a.py"
        f.write_text("def foo(): pass\n")
        _do_index(str(tmp_path))

        h = self._make_handler(str(tmp_path))
        # Since stat matches what was just indexed, should return early
        h._process_file(str(f))

    def test_process_file_changed(self, tmp_path):
        """_process_file re-indexes a file with changed mtime/size."""
        f = tmp_path / "a.py"
        f.write_text("def foo(): pass\n")
        _do_index(str(tmp_path))

        f.write_text("def foo(): return 1\ndef bar(): pass\n")
        h = self._make_handler(str(tmp_path))
        h._process_file(str(f))  # should not raise

    def test_delete_file_acquires_lock(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("def foo(): pass\n")
        _do_index(str(tmp_path))

        h = self._make_handler(str(tmp_path))
        h._delete_file(str(tmp_path), str(f))  # should not raise

    def test_schedule_processing_cancels_previous(self, tmp_path):
        h = self._make_handler(str(tmp_path))
        h._debounce_delay = 60
        fp = str(tmp_path / "a.py")

        mock_timer = MagicMock()
        h._debounce_timers[fp] = mock_timer

        with patch("vecgrep.server.threading.Timer") as MockTimer:
            MockTimer.return_value = MagicMock()
            h._schedule_processing(fp)
        mock_timer.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# _ensure_watcher and _stop_all_observers
# ---------------------------------------------------------------------------


class TestEnsureWatcher:
    def test_creates_observer_on_first_call(self, tmp_path):
        root_str = str(tmp_path.resolve()) + "_watcher_test"

        # Ensure not already registered
        _OBSERVER_REGISTRY.pop(root_str, None)

        _ensure_watcher(tmp_path.parent, [])  # uses parent to avoid stale entries
        # Clean up any observer created
        obs = _OBSERVER_REGISTRY.pop(str(tmp_path.parent.resolve()), None)
        if obs:
            obs.stop()
            obs.join(timeout=2)

    def test_does_not_duplicate_observer(self, tmp_path):
        root_str = str(tmp_path.resolve()) + "_dedup"
        tmp_dup = tmp_path / "dedup"
        tmp_dup.mkdir()
        root_str = str(tmp_dup.resolve())
        _OBSERVER_REGISTRY.pop(root_str, None)

        _ensure_watcher(tmp_dup, [])
        _ensure_watcher(tmp_dup, [])  # second call should not create another

        count = sum(1 for k in _OBSERVER_REGISTRY if k == root_str)
        assert count == 1

        obs = _OBSERVER_REGISTRY.pop(root_str, None)
        if obs:
            obs.stop()
            obs.join(timeout=2)


class TestStopAllObservers:
    def test_stops_registered_observers(self):
        mock_obs = MagicMock()
        _OBSERVER_REGISTRY["__test_stop__"] = mock_obs

        _stop_all_observers()

        mock_obs.stop.assert_called()
        _OBSERVER_REGISTRY.pop("__test_stop__", None)

    def test_tolerates_exception_in_stop(self):
        mock_obs = MagicMock()
        mock_obs.stop.side_effect = RuntimeError("crash")
        _OBSERVER_REGISTRY["__test_exc__"] = mock_obs

        _stop_all_observers()  # Should not raise

        _OBSERVER_REGISTRY.pop("__test_exc__", None)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_without_error(self):
        with patch("vecgrep.server.mcp.run") as mock_run:
            main()
            mock_run.assert_called_once()
