"""
Resume launcher for the HPC claim-validation suite (seeds 4-8).

The first run (launched 2026-05-15, started seeds at 18:57 UTC) processed
seeds [42, 0, 1, 2, 3] completely (all 6 methods each, 30/60 method-runs)
before being killed at 2026-05-16 ~15:03 UTC by a blanket
`Get-Process python | Stop-Process -Force` cleanup command.

Seed 4 was incomplete (5/6 methods, killed mid high_penalty_conservative).
It is re-run in full here for provenance cleanliness.

This launcher targets seeds [4, 5, 6, 7, 8] via runtime_context["resume_seeds"]
so the locked spec fields (seeds, config_overrides.min_seed_list) are not
changed and the validation-suite drift check continues to pass.

The canonical result produced by this run covers seeds [4, 5, 6, 7, 8].
The human combines with the first run's per-seed data for the n=10 analysis.

Do NOT use run_claim_validation_hpc_once.py for this resume — that launcher
would attempt the full seed list and re-run the already-complete seeds.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_experiment_orchestrator import ExperimentOrchestrator
from tar_storage import ensure_workspace_layout, resolve_workspace

REPO = Path(__file__).resolve().parent
WORKSPACE = ensure_workspace_layout(resolve_workspace(REPO), repo_root=REPO)
TARGET_ID = "claim_validation_hpc_suite"
MANIFEST_PATH = REPO / "manifests" / "claim_validation_hpc_suite_20260511.json"

RESUME_SEEDS = [4, 5, 6, 7, 8]
COMPLETED_SEEDS = [42, 0, 1, 2, 3]
METHOD_ORDER = [
    "sgd_baseline",
    "ewc_lambda_100",
    "ewc_lambda_1000",
    "si_c_0_01",
    "tcl_baseline",
    "high_penalty_conservative",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log(message: str) -> None:
    print(f"[{_now_iso()}] {message}", flush=True)


def _stat_audit_path(name: str) -> Path:
    root = WORKSPACE / "tar_state" / "stat_audit"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}__{_stamp()}.json"


def _load_validation_state() -> dict[str, Any]:
    path = WORKSPACE / "tar_state" / "stabilisation_mode.json"
    if not path.exists():
        raise RuntimeError(f"Validation mode state missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Validation mode state is not a JSON object.")
    return raw


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_blocker(reason: str, detail: dict[str, Any] | None = None) -> Path:
    path = WORKSPACE / "tar_state" / "stat_audit" / f"resume_blocker_{_stamp()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# HPC Resume Blocker\n\n"
        f"**Timestamp:** {_now_iso()}\n\n"
        f"**Reason:** {reason}\n\n"
        f"**Detail:**\n```json\n{json.dumps(detail or {}, indent=2)}\n```\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    # ── Validate stabilisation mode ─────────────────────────────────────────
    validation = _load_validation_state()
    if not validation.get("active"):
        blocker = _write_blocker("Stabilisation mode is not active.", {"state": validation})
        raise RuntimeError(f"Stabilisation mode not active. Blocker written: {blocker}")
    if validation.get("primary_validation_experiment_id") != TARGET_ID:
        actual = validation.get("primary_validation_experiment_id")
        blocker = _write_blocker(
            f"Primary experiment mismatch: expected {TARGET_ID!r}, got {actual!r}.",
            {"stabilisation_state": validation},
        )
        raise RuntimeError(f"Primary experiment mismatch. Blocker: {blocker}")

    _log("Stabilisation mode confirmed. Initialising HPC resume for seeds 4-8...")

    # ── Load orchestrator and reconcile stale state ─────────────────────────
    orch = ExperimentOrchestrator(WORKSPACE)
    orch.reconcile_runtime_state()

    running = orch.get_running()
    if running:
        blocker = _write_blocker(
            "Queue still has running experiment(s) after reconcile.",
            {"running": [spec.id for spec in running]},
        )
        raise RuntimeError(f"Queue still running. Blocker: {blocker}")

    # ── Load and verify manifest ────────────────────────────────────────────
    from tar_lab.manifest import load_and_verify_manifest
    manifest = load_and_verify_manifest(MANIFEST_PATH, REPO)
    manifest_id = manifest.manifest_id
    _log(f"Manifest verified: {manifest_id}")
    orch.set_manifest(MANIFEST_PATH)

    # ── Locate and validate the spec ────────────────────────────────────────
    spec = orch._specs.get(TARGET_ID)
    if spec is None:
        blocker = _write_blocker(
            f"Experiment {TARGET_ID!r} not found in queue after reconcile.",
            {"available": list(orch._specs.keys())},
        )
        raise RuntimeError(f"Spec not found. Blocker: {blocker}")

    if spec.status not in ("pending", "stalled"):
        blocker = _write_blocker(
            f"Experiment status is {spec.status!r}, expected pending or stalled after reconcile.",
            {"status": spec.status, "stage": spec.stage, "error": spec.error},
        )
        raise RuntimeError(f"Bad spec status {spec.status!r}. Blocker: {blocker}")

    _log(f"Spec found: status={spec.status} stage={spec.stage} seeds={spec.seeds}")

    # ── Inject resume_seeds into runtime_context (not drift-checked) ────────
    # Locked fields spec.seeds and config_overrides.min_seed_list are NOT
    # changed, so the validation-suite drift check continues to pass.
    # _run_hpc_validation_suite reads resume_seeds from runtime_context and
    # filters the seed list before running.
    spec.runtime_context = dict(spec.runtime_context or {})
    spec.runtime_context["resume_seeds"] = list(RESUME_SEEDS)
    spec.runtime_context["resume_reason"] = (
        "First run killed at 2026-05-16 15:03 UTC by blanket Stop-Process. "
        "Seeds 42,0,1,2,3 complete. Seed 4 partial (5/6 methods), re-run in full."
    )
    spec.status = "pending"
    spec.stage = "queued"
    spec.error = ""
    spec.pid = 0
    orch._save()
    _log(f"Spec updated: resume_seeds={spec.runtime_context['resume_seeds']}")

    # ── Execute ─────────────────────────────────────────────────────────────
    _log(f"Launching resume under manifest {MANIFEST_PATH.name} ({manifest_id}).")
    _log(f"Seeds to run: {RESUME_SEEDS}  Methods: {METHOD_ORDER}")

    result = orch._execute(spec)

    # ── Read PID from lock file (worker updates it to completed on exit) ────
    pid_lock = WORKSPACE / "tar_state" / "run_locks" / f"{TARGET_ID}.pid"
    worker_pid: int | None = None
    lock_data: dict[str, Any] = {}
    if pid_lock.exists():
        try:
            lock_data = json.loads(pid_lock.read_text(encoding="utf-8"))
            worker_pid = int(lock_data.get("pid") or 0) or None
        except Exception:
            pass

    kill_exclusion = (
        f"Get-Process python | Where-Object {{ $_.Id -ne {worker_pid} }} | Stop-Process"
        if worker_pid
        else "Get-Process python | Stop-Process  # WARNING: no PID recorded — check run_locks/"
    )

    # ── Write launch audit ──────────────────────────────────────────────────
    audit_payload: dict[str, Any] = {
        "launcher": "run_claim_validation_hpc_resume.py",
        "launched_at": _now_iso(),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_id": manifest_id,
        "experiment_id": TARGET_ID,
        "seeds_run": list(RESUME_SEEDS),
        "seeds_already_complete": list(COMPLETED_SEEDS),
        "methods": list(METHOD_ORDER),
        "pid_lock_path": str(pid_lock),
        "worker_pid": worker_pid,
        "worker_pid_lock_status": lock_data.get("status", "unknown"),
        "kill_exclusion_command": kill_exclusion,
        "result": None if result is None else {
            "experiment_id": result.experiment_id,
            "verdict": result.verdict,
            "status": result.status,
            "result_path": result.result_path,
            "notes": result.notes,
        },
    }
    note_path = _stat_audit_path("hpc_resume_launch")
    _write_json(note_path, audit_payload)
    _log(f"Launch audit written to {note_path}")
    _log(f"FUTURE SAFE CLEANUP: {kill_exclusion}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure: dict[str, Any] = {
            "launcher": "run_claim_validation_hpc_resume.py",
            "failed_at": _now_iso(),
            "manifest_path": str(MANIFEST_PATH),
            "experiment_id": TARGET_ID,
            "seeds_attempted": list(RESUME_SEEDS),
            "error": str(exc),
        }
        note_path = (WORKSPACE / "tar_state" / "stat_audit" / f"hpc_resume_launch__{_stamp()}.json")
        note_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(note_path, failure)
        _log(f"REFUSED/FAILED: {exc}")
        _log(f"Failure audit written to {note_path}")
        raise
