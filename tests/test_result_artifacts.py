from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tar_lab.result_artifacts import (
    load_latest_phase_comparisons,
    write_append_only_result_pair,
    write_canonical_comparison_result,
)
from tar_lab.canonical_registry import (
    DeterministicRecomputeFailure,
    ManifestIntegrityMismatchedError,
    ProvenanceSiblingInvalidError,
    RegistryLockTimeout,
    _acquire_registry_lock,
    _check_gate_1_sibling,
    _check_gate_2_manifest,
    _check_gate_3_recompute,
    _compute_env_hash,
    quarantine_unregistered_results,
)


def _prepare_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "tar_state" / "comparisons").mkdir(parents=True, exist_ok=True)
    return workspace


# ── existing result_artifacts tests ───────────────────────────────────────────

def test_write_append_only_result_pair_refuses_overwrite(tmp_path: Path) -> None:
    workspace = _prepare_workspace(tmp_path)
    result_path = workspace / "tar_state" / "experiments" / "exp-1" / "result.json"
    payload = {"value": 1}
    env_payload = {"env": "snapshot"}

    artifacts = write_append_only_result_pair(
        result_path=result_path,
        payload=payload,
        env_payload=env_payload,
    )

    assert artifacts["result_path"].exists()
    assert artifacts["env_path"].exists()
    assert json.loads(artifacts["result_path"].read_text(encoding="utf-8")) == payload
    assert json.loads(artifacts["env_path"].read_text(encoding="utf-8")) == env_payload

    with pytest.raises(FileExistsError):
        write_append_only_result_pair(
            result_path=result_path,
            payload=payload,
            env_payload=env_payload,
        )


def test_write_canonical_comparison_result_creates_indexed_unique_artifact(tmp_path: Path) -> None:
    workspace = _prepare_workspace(tmp_path)
    payload = {"phase": 10, "value": 1}
    env_payload = {"env": "snapshot"}

    artifacts = write_canonical_comparison_result(
        workspace=workspace,
        logical_name="phase10_baseline",
        payload=payload,
        env_payload=env_payload,
        phase_number=10,
        source_script="phase10_baseline.py",
    )

    assert artifacts["result_path"].name.startswith("phase10_baseline__")
    assert artifacts["env_path"].name.startswith("phase10_baseline__")
    index_lines = artifacts["index_path"].read_text(encoding="utf-8").splitlines()
    assert len(index_lines) == 1
    record = json.loads(index_lines[0])
    assert record["logical_name"] == "phase10_baseline"
    assert record["phase_number"] == 10


def test_load_latest_phase_comparisons_prefers_canonical_index_over_legacy(tmp_path: Path) -> None:
    workspace = _prepare_workspace(tmp_path)
    comparisons = workspace / "tar_state" / "comparisons"
    legacy_path = comparisons / "phase10_baseline.json"
    legacy_path.write_text(json.dumps({"phase": 10, "source": "legacy"}), encoding="utf-8")

    write_canonical_comparison_result(
        workspace=workspace,
        logical_name="phase10_baseline",
        payload={"phase": 10, "source": "canonical"},
        env_payload={"env": "snapshot"},
        phase_number=10,
        source_script="phase10_baseline.py",
    )

    loaded = load_latest_phase_comparisons(workspace)
    assert loaded[10]["source"] == "canonical"


# ── canonical_registry helpers ────────────────────────────────────────────────

def _make_result_dir(tmp_path: Path, run_id: str = "my_run") -> Path:
    """Create workspace/tar_state/experiments/<run_id>/ and return the exp dir."""
    exp_dir = tmp_path / "workspace" / "tar_state" / "experiments" / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _make_env_json(exp_dir: Path, *, git_head: str = "abc123def456",
                   manifest_path: str = "/manifests/m.json",
                   manifest_hash: str = "aabbcc") -> Path:
    env_path = exp_dir / "result_env.json"
    env_path.write_text(json.dumps({
        "git": {"head": git_head},
        "authorization": {
            "manifest_path": manifest_path,
            "manifest_hash": manifest_hash,
        },
    }), encoding="utf-8")
    return env_path


def _make_manifest_file(tmp_path: Path, *, include_content_hash: bool = True) -> tuple[Path, str]:
    """Return (manifest_path, correct_hash) for a minimal valid manifest."""
    from tar_lab.manifest import compute_manifest_hash
    data: dict = {
        "manifest_schema": "tar_execution_manifest_v1",
        "manifest_id": "test-manifest-gate2",
        "created_at": "2026-01-01T00:00:00Z",
        "authorised_by": "test_user",
        "purpose": "Gate 2 test",
        "experiments": [{
            "experiment_id": "test-exp",
            "name": "Test Experiment",
            "allowed_datasets": [],
            "allowed_methods": [],
            "allowed_seeds": [],
            "time_limit_h": 48.0,
            "run_limit": 1,
            "notes": "",
        }],
        "global_time_limit_h": 48.0,
    }
    if include_content_hash:
        data["content_hash"] = "UNSIGNED"
    p = tmp_path / "test_manifest.json"
    p.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
    h = compute_manifest_hash(p)
    return p, h


def _clean_git_mock(manifest_path: Path):
    """Return a subprocess.run mock that simulates a clean, tracked manifest."""
    def _mock_run(cmd, **kwargs):
        m = MagicMock()
        if "status" in cmd:
            m.stdout = ""
            m.returncode = 0
        elif "ls-files" in cmd:
            m.stdout = str(manifest_path)
            m.returncode = 0
        else:
            m.stdout = ""
            m.returncode = 0
        return m
    return _mock_run


def _make_seed_result(n: int = 3) -> tuple[list[dict], float, float, float, float]:
    """Return (seed_results, mean_f, std_f, mean_a, std_a) for n seeds."""
    forgetting = [0.10 + i * 0.05 for i in range(n)]
    accuracy = [0.90 - i * 0.03 for i in range(n)]
    seed_results = [
        {"seed": i, "forgetting": f, "accuracy": a}
        for i, (f, a) in enumerate(zip(forgetting, accuracy))
    ]
    f_arr = np.array(forgetting)
    a_arr = np.array(accuracy)
    mean_f = float(np.mean(f_arr))
    std_f = float(np.std(f_arr, ddof=1))
    mean_a = float(np.mean(a_arr))
    std_a = float(np.std(a_arr, ddof=1))
    return seed_results, mean_f, std_f, mean_a, std_a


# ── Gate 1 tests ──────────────────────────────────────────────────────────────

def test_gate1_passes_valid_env_sibling(tmp_path: Path) -> None:
    exp_dir = _make_result_dir(tmp_path)
    result_path = exp_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    _make_env_json(exp_dir)

    fields = _check_gate_1_sibling(result_path)
    assert fields["git_head"] == "abc123def456"
    assert fields["manifest_hash"] == "aabbcc"
    assert fields["manifest_path"] == "/manifests/m.json"


def test_gate1_fails_missing_env_sibling(tmp_path: Path) -> None:
    exp_dir = _make_result_dir(tmp_path)
    result_path = exp_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    # No result_env.json written.
    with pytest.raises(ProvenanceSiblingInvalidError, match="not found"):
        _check_gate_1_sibling(result_path)


def test_gate1_fails_missing_manifest_hash(tmp_path: Path) -> None:
    exp_dir = _make_result_dir(tmp_path)
    result_path = exp_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    (exp_dir / "result_env.json").write_text(json.dumps({
        "git": {"head": "abc"},
        "authorization": {"manifest_path": "/m.json"},
        # manifest_hash deliberately absent
    }), encoding="utf-8")
    with pytest.raises(ProvenanceSiblingInvalidError, match="manifest_hash"):
        _check_gate_1_sibling(result_path)


def test_gate1_fails_malformed_env_json(tmp_path: Path) -> None:
    exp_dir = _make_result_dir(tmp_path)
    result_path = exp_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    (exp_dir / "result_env.json").write_text("not valid json {{", encoding="utf-8")
    with pytest.raises(ProvenanceSiblingInvalidError, match="not valid JSON"):
        _check_gate_1_sibling(result_path)


# ── Gate 2 tests ──────────────────────────────────────────────────────────────

def test_gate2_passes_matching_hash(tmp_path: Path) -> None:
    manifest_path, correct_hash = _make_manifest_file(tmp_path)
    env_fields = {
        "git_head": "abc",
        "manifest_path": str(manifest_path),
        "manifest_hash": correct_hash,
        "manifest_id": "",
    }
    with patch(
        "tar_lab.canonical_registry.subprocess.run",
        side_effect=_clean_git_mock(manifest_path),
    ):
        _check_gate_2_manifest(env_fields, tmp_path)  # must not raise


def test_gate2_fails_dirty_working_tree(tmp_path: Path) -> None:
    manifest_path, correct_hash = _make_manifest_file(tmp_path)
    env_fields = {
        "git_head": "abc",
        "manifest_path": str(manifest_path),
        "manifest_hash": correct_hash,
        "manifest_id": "",
    }

    def _dirty_mock(cmd, **kwargs):
        m = MagicMock()
        if "status" in cmd:
            m.stdout = " M test_manifest.json\n"
            m.returncode = 0
        else:
            m.stdout = ""
            m.returncode = 0
        return m

    with patch("tar_lab.canonical_registry.subprocess.run", side_effect=_dirty_mock):
        with pytest.raises(ManifestIntegrityMismatchedError, match="not cleanly committed"):
            _check_gate_2_manifest(env_fields, tmp_path)


def test_gate2_fails_hash_mismatch(tmp_path: Path) -> None:
    manifest_path, _ = _make_manifest_file(tmp_path)
    env_fields = {
        "git_head": "abc",
        "manifest_path": str(manifest_path),
        "manifest_hash": "0000000000000000deadbeef",  # wrong hash
        "manifest_id": "",
    }
    with patch(
        "tar_lab.canonical_registry.subprocess.run",
        side_effect=_clean_git_mock(manifest_path),
    ):
        with pytest.raises(ManifestIntegrityMismatchedError, match="hash mismatch"):
            _check_gate_2_manifest(env_fields, tmp_path)


def test_gate2_passes_absent_content_hash(tmp_path: Path) -> None:
    """Corroborating Field absent from manifest; Authoritative Invariant holds → pass."""
    manifest_path, correct_hash = _make_manifest_file(tmp_path, include_content_hash=False)
    env_fields = {
        "git_head": "abc",
        "manifest_path": str(manifest_path),
        "manifest_hash": correct_hash,
        "manifest_id": "",
    }
    with patch(
        "tar_lab.canonical_registry.subprocess.run",
        side_effect=_clean_git_mock(manifest_path),
    ):
        _check_gate_2_manifest(env_fields, tmp_path)  # must not raise


def test_gate2_fails_untracked_manifest(tmp_path: Path) -> None:
    """Binding 1: ls-files returns non-zero (untracked manifest) → ManifestIntegrityMismatchedError.

    git status returns empty (clean), but git ls-files --error-unmatch returns
    returncode=1, meaning the manifest is not tracked by git.  Gate 2 must fail.
    """
    manifest_path, correct_hash = _make_manifest_file(tmp_path)
    env_fields = {
        "git_head": "abc",
        "manifest_path": str(manifest_path),
        "manifest_hash": correct_hash,
        "manifest_id": "",
    }

    def _untracked_mock(cmd, **kwargs):
        m = MagicMock()
        if "status" in cmd:
            m.stdout = ""   # clean — passes dirty check
            m.returncode = 0
        elif "ls-files" in cmd:
            m.stdout = ""
            m.returncode = 1  # non-zero: manifest is not tracked
        else:
            m.stdout = ""
            m.returncode = 0
        return m

    with patch("tar_lab.canonical_registry.subprocess.run", side_effect=_untracked_mock):
        with pytest.raises(ManifestIntegrityMismatchedError, match="not tracked by git"):
            _check_gate_2_manifest(env_fields, tmp_path)


# ── Gate 3 tests ──────────────────────────────────────────────────────────────

def test_gate3_passes_matching_aggregates(tmp_path: Path) -> None:
    seed_results, mean_f, std_f, mean_a, std_a = _make_seed_result(4)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "seed_results": seed_results,
        "mean_forgetting": mean_f,
        "std_forgetting": std_f,
        "mean_accuracy": mean_a,
        "std_accuracy": std_a,
    }), encoding="utf-8")

    recomputed = _check_gate_3_recompute(result_path)
    assert abs(recomputed["mean_forgetting"] - mean_f) < 1e-9
    assert abs(recomputed["std_forgetting"] - std_f) < 1e-9


def test_gate3_fails_mean_mismatch(tmp_path: Path) -> None:
    seed_results, mean_f, std_f, mean_a, std_a = _make_seed_result(4)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "seed_results": seed_results,
        "mean_forgetting": mean_f + 0.1,  # deliberately wrong
        "std_forgetting": std_f,
        "mean_accuracy": mean_a,
        "std_accuracy": std_a,
    }), encoding="utf-8")

    with pytest.raises(DeterministicRecomputeFailure, match="mean_forgetting"):
        _check_gate_3_recompute(result_path)


def test_gate3_independence(tmp_path: Path, monkeypatch) -> None:
    """_check_gate_3_recompute must not depend on tar_lab.result_artifacts.

    Removing that module from sys.modules causes ImportError on any lazy import
    or attribute lookup. The gate must succeed despite the module being absent.
    """
    monkeypatch.delitem(sys.modules, "tar_lab.result_artifacts", raising=False)

    seed_results, mean_f, std_f, mean_a, std_a = _make_seed_result(3)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "seed_results": seed_results,
        "mean_forgetting": mean_f,
        "std_forgetting": std_f,
        "mean_accuracy": mean_a,
        "std_accuracy": std_a,
    }), encoding="utf-8")

    # Must complete without accessing tar_lab.result_artifacts.
    recomputed = _check_gate_3_recompute(result_path)
    assert abs(recomputed["mean_forgetting"] - mean_f) < 1e-9


# ── Lockfile / finally-release tests ─────────────────────────────────────────

def test_lock_normal_acquire_and_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    handle = _acquire_registry_lock(lock_path)
    try:
        assert lock_path.exists()
        content = json.loads(lock_path.read_text(encoding="utf-8"))
        assert content["pid"] == os.getpid()
    finally:
        handle.close()
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass
    assert not lock_path.exists()


def test_lock_finally_releases_on_exception(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    with pytest.raises(RuntimeError, match="simulated"):
        lock_handle = _acquire_registry_lock(lock_path)
        try:
            assert lock_path.exists()
            raise RuntimeError("simulated error in critical section")
        finally:
            lock_handle.close()
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass
    assert not lock_path.exists()


def _make_stale_lock(lock_path: Path) -> None:
    """Write a lock file with a dead-looking PID and an mtime > 60s ago."""
    lock_path.write_text(
        json.dumps({"pid": 999999999, "acquired_at": 0.0}),
        encoding="utf-8",
    )
    old_mtime = time.time() - 120
    os.utime(lock_path, (old_mtime, old_mtime))


def test_lock_stale_fnf_loser(tmp_path: Path, monkeypatch) -> None:
    """Race loser #1: os.unlink raises FileNotFoundError → RegistryLockTimeout."""
    lock_path = tmp_path / "test.lock"
    _make_stale_lock(lock_path)
    monkeypatch.setattr("tar_lab.canonical_registry._pid_alive", lambda pid: False)

    with patch(
        "tar_lab.canonical_registry.os.unlink",
        side_effect=FileNotFoundError("already gone"),
    ):
        with pytest.raises(RegistryLockTimeout, match="FileNotFoundError"):
            _acquire_registry_lock(lock_path)


def test_lock_stale_permission_loser(tmp_path: Path, monkeypatch) -> None:
    """Race loser #2: os.unlink raises PermissionError → RegistryLockTimeout."""
    lock_path = tmp_path / "test.lock"
    _make_stale_lock(lock_path)
    monkeypatch.setattr("tar_lab.canonical_registry._pid_alive", lambda pid: False)

    with patch(
        "tar_lab.canonical_registry.os.unlink",
        side_effect=PermissionError("active handle"),
    ):
        with pytest.raises(RegistryLockTimeout, match="PermissionError"):
            _acquire_registry_lock(lock_path)


def test_lock_stale_exists_after_unlink_loser(tmp_path: Path, monkeypatch) -> None:
    """Race loser #3: unlink succeeds but re-acquire open('x') raises FileExistsError."""
    lock_path = tmp_path / "test.lock"
    _make_stale_lock(lock_path)
    monkeypatch.setattr("tar_lab.canonical_registry._pid_alive", lambda pid: False)

    original_unlink = os.unlink

    def unlink_and_race(path):
        original_unlink(path)
        # Simulate another process grabbing the lock immediately.
        Path(path).write_text(
            json.dumps({"pid": os.getpid() + 1, "acquired_at": time.time()}),
            encoding="utf-8",
        )

    monkeypatch.setattr("tar_lab.canonical_registry.os.unlink", unlink_and_race)

    with pytest.raises(RegistryLockTimeout, match="FileExistsError"):
        _acquire_registry_lock(lock_path)


def test_lock_fresh_lock_raises_timeout(tmp_path: Path) -> None:
    """A lock file with mtime < 60s raises RegistryLockTimeout; no unlink attempted."""
    lock_path = tmp_path / "test.lock"
    # Fresh lock — mtime is right now (default for newly written files).
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "acquired_at": time.time()}),
        encoding="utf-8",
    )
    # No utime call — mtime is current, age < 60s.
    with pytest.raises(RegistryLockTimeout, match="<= 60s"):
        _acquire_registry_lock(lock_path)
    # Lock file must still exist (no reclaim attempted).
    assert lock_path.exists()


# ── Discovery / quarantine tests ──────────────────────────────────────────────

def test_quarantine_unregistered_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    exp_dir = workspace / "tar_state" / "experiments" / "my_run"
    exp_dir.mkdir(parents=True)
    (exp_dir / "result.json").write_text("{}", encoding="utf-8")
    (exp_dir / "result_env.json").write_text("{}", encoding="utf-8")

    quarantine_set, is_clean = quarantine_unregistered_results(workspace, [])
    assert quarantine_set == ["my_run"]
    assert is_clean is False


def test_quarantine_hash_mismatch_in_index(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    exp_dir = workspace / "tar_state" / "experiments" / "my_run"
    exp_dir.mkdir(parents=True)
    env_bytes = b'{"git": {"head": "abc"}}'
    (exp_dir / "result.json").write_text("{}", encoding="utf-8")
    (exp_dir / "result_env.json").write_bytes(env_bytes)

    actual_hash = hashlib.sha256(env_bytes).hexdigest()
    wrong_hash = "00" * (len(actual_hash) // 2)

    index_records = [{"run_id": "my_run", "env_snapshot_hash": wrong_hash}]
    quarantine_set, is_clean = quarantine_unregistered_results(workspace, index_records)
    assert quarantine_set == ["my_run"]
    assert is_clean is False


def test_quarantine_clean_when_registered(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    exp_dir = workspace / "tar_state" / "experiments" / "my_run"
    exp_dir.mkdir(parents=True)
    env_bytes = b'{"git": {"head": "abc"}}'
    (exp_dir / "result.json").write_text("{}", encoding="utf-8")
    (exp_dir / "result_env.json").write_bytes(env_bytes)

    actual_hash = hashlib.sha256(env_bytes).hexdigest()
    index_records = [{"run_id": "my_run", "env_snapshot_hash": actual_hash}]
    quarantine_set, is_clean = quarantine_unregistered_results(workspace, index_records)
    assert quarantine_set == []
    assert is_clean is True


def test_quarantine_no_experiments_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # No tar_state/experiments directory — function must not raise.
    quarantine_set, is_clean = quarantine_unregistered_results(workspace, [])
    assert quarantine_set == []
    assert is_clean is True
