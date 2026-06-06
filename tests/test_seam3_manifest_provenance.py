"""Seam 3 (2026-06-06) — manifest->env provenance regression lock.

The 10-agent assessment flagged that `authorization.manifest_hash` was landing
`null` in env snapshots, so `verify_canonical_3gate` never conferred publication.
Root cause was a SUPERVISED phase-script that ran a pre-fix script — NOT the
autonomous orchestrator path. The autonomous path is structurally sound because
`load_and_verify_manifest` REFUSES a manifest whose content_hash is still the
UNSIGNED sentinel (manifest.py), so `orchestrator._active_manifest.content_hash`
can never be null/sentinel once set.

These tests LOCK that guarantee end-to-end with the real production helpers:
  (1) the autonomous auto-manifest yields a non-null, non-sentinel content_hash;
  (2) an env snapshot carrying that hash + git.head clears canonical gates 1 & 2.

Hermetic: every test runs against a throwaway temp git repo so the live branch
is never touched (the orchestrator auto-manifest path git-commits).
The orchestrator import is kept function-local so the Windows torch-DLL pytest
flake (which only bites modules that transitively import torch while the live
daemon holds it) cannot block collection of the torch-free function-level test.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(["init"], repo)
    _git(["config", "user.email", "seam3@test.local"], repo)
    _git(["config", "user.name", "Seam3 Test"], repo)
    (repo / "seed.txt").write_text("seed", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "init"], repo)


def _write_signed_manifest(repo: Path, exp_id: str = "seam3-exp") -> tuple[Path, str]:
    """Replicate _auto_generate_manifest's exact hashing sequence with the real helper."""
    from tar_lab.manifest import compute_manifest_hash

    auto_dir = repo / "manifests" / "auto"
    auto_dir.mkdir(parents=True, exist_ok=True)
    path = auto_dir / "seam3.json"
    payload = {
        "manifest_id": "manifest-auto-seam3",
        "manifest_schema": "tar_execution_manifest_v1",
        "created_at": "2026-06-06T00:00:00+00:00",
        "authorised_by": "TAR Director (autonomous)",
        "purpose": "seam3 provenance regression",
        "global_time_limit_h": 1.0,
        "experiments": [
            {
                "experiment_id": exp_id,
                "name": "seam3",
                "allowed_datasets": ["split_cifar10"],
                "allowed_methods": ["tcl"],
                "allowed_seeds": [0],
                "time_limit_h": 1.0,
                "run_limit": 1,
                "notes": "regression",
            }
        ],
        "content_hash": "UNSIGNED",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    digest = compute_manifest_hash(path)
    payload["content_hash"] = digest
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    return path, digest


def test_autonomous_result_clears_gates_1_and_2(tmp_path):
    """Function-level (torch-free): a result whose env carries the auto-manifest's
    content_hash + git.head clears canonical gate-1 (env sibling) and gate-2
    (committed manifest + hash match). Gate-3 (deterministic recompute) is a
    separate concern and may legitimately fail for a synthetic aggregate."""
    from tar_lab.manifest import load_and_verify_manifest
    from tar_lab.result_artifacts import collect_environment_snapshot
    from tar_lab.canonical_registry import verify_canonical_3gate

    repo = tmp_path / "repo"
    _init_repo(repo)
    path, digest = _write_signed_manifest(repo)
    assert digest and digest != "UNSIGNED"
    _git(["add", "--", str(path)], repo)
    _git(["commit", "-m", "auto-manifest"], repo)

    # The autonomous binding source: the loader populates content_hash with the real
    # digest (and would have refused the UNSIGNED sentinel outright).
    manifest = load_and_verify_manifest(path, repo_root=repo)
    assert manifest.content_hash == digest
    assert manifest.content_hash != "UNSIGNED"

    ws = tmp_path / "ws"
    ws.mkdir()
    env = collect_environment_snapshot(
        repo_root=repo,
        workspace=ws,
        config={},
        trigger="seam3_test",
        source_script="test_seam3_manifest_provenance.py",
        manifest_path=str(path),
        manifest_hash=manifest.content_hash,
    )
    assert env["git"]["head"], "git.head must be non-null"
    assert env["authorization"]["manifest_hash"] == digest, "manifest_hash must flow into env"

    comp = ws / "tar_state" / "comparisons"
    comp.mkdir(parents=True)
    res = comp / "seam3__x.json"
    envp = comp / "seam3__x_env.json"
    res.write_text(json.dumps({"logical_name": "seam3", "aggregate": {}}), encoding="utf-8")
    envp.write_text(json.dumps(env), encoding="utf-8")

    ok, reason = verify_canonical_3gate(res, repo_root=repo)
    # Provenance gates (1 & 2) must clear. Only the deterministic recompute (gate-3)
    # may fail for this synthetic aggregate — that is out of Seam 3's scope.
    assert ok or reason.startswith("gate3"), f"unexpected gate-1/2 failure: {reason}"


def test_autonomous_auto_manifest_content_hash_nonnull(tmp_path, monkeypatch):
    """Orchestrator-path: _auto_generate_manifest must leave the active manifest's
    content_hash non-null and non-sentinel (the value _save_result freezes into the
    env). Hermetic via a monkeypatched _REPO temp git repo so the live branch is
    never committed to. Import is function-local to dodge the torch-DLL flake."""
    monkeypatch.delenv("TAR_MANIFEST_PATH", raising=False)
    import tar_experiment_orchestrator as orch_mod
    from tar_experiment_orchestrator import ExperimentSpec

    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.setattr(orch_mod, "_REPO", repo)

    workspace = tmp_path / "ws"
    (workspace / "tar_state").mkdir(parents=True)
    orch = orch_mod.ExperimentOrchestrator(workspace)
    orch.set_autonomous(True)

    spec = ExperimentSpec(
        name="seam3",
        project_id="seam3-proj",
        hypothesis_name="seam3-hyp",
        dataset="split_cifar10",
        method="tcl",
        seeds=[0],
        config_overrides={},
        estimated_runtime_h=1.0,
    )
    manifest = orch._auto_generate_manifest(spec)

    # the binding source read by _save_result: getattr(self._active_manifest, "content_hash")
    assert orch._active_manifest is not None
    assert orch._active_manifest.content_hash
    assert orch._active_manifest.content_hash != "UNSIGNED"
    assert manifest.content_hash == orch._active_manifest.content_hash
