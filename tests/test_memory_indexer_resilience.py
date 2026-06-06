"""MemoryIndexer per-source isolation (exposed by engaging the bge-small embedder).

Switching the vault embedder forces a rebuild, which surfaced a robustness bug: all
sources were indexed under one try/except, so a single stale source (knowledge_graph.json
schema drift) aborted the whole rebuild and left the vault stuck "degraded". _run_source
now isolates each source: a stale/un-parseable source is recorded + skipped (mtime left
unset so it retries), while a genuine MemoryIntegrityError still propagates as fatal.
"""
import pytest

from tar_lab.memory.vault import MemoryIndexer, MemoryIntegrityError


def _raise(exc):
    def _f():
        raise exc
    return _f


def _indexer():
    idx = MemoryIndexer.__new__(MemoryIndexer)  # bypass __init__ (no store/vault needed)
    idx._metrics_mtime = None
    idx._graph_mtime = None
    return idx


def test_stale_source_is_skipped_not_fatal(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    idx = _indexer()
    failures: list[str] = []
    # a source that raises a normal error is recorded + skipped, NOT raised
    idx._run_source("knowledge_graph", src, "_graph_mtime", _raise(ValueError("schema drift")), failures)
    assert failures and failures[0].startswith("knowledge_graph")
    assert idx._graph_mtime is None  # mtime left unset -> retried next sync


def test_memory_integrity_error_still_fatal(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    idx = _indexer()
    failures: list[str] = []
    with pytest.raises(MemoryIntegrityError):
        idx._run_source("metrics", src, "_metrics_mtime", _raise(MemoryIntegrityError("embedder/dim")), failures)


def test_good_source_sets_mtime(tmp_path):
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    idx = _indexer()
    failures: list[str] = []
    idx._run_source("metrics", src, "_metrics_mtime", lambda: None, failures)
    assert failures == [] and idx._metrics_mtime is not None


def test_missing_source_is_noop(tmp_path):
    idx = _indexer()
    failures: list[str] = []
    idx._run_source("metrics", tmp_path / "does_not_exist.json", "_metrics_mtime", _raise(RuntimeError("boom")), failures)
    assert failures == [] and idx._metrics_mtime is None  # absent source: skipped, no failure
