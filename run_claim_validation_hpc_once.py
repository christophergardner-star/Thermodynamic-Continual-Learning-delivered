from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_experiment_orchestrator import ExperimentOrchestrator
from tar_storage import ensure_workspace_layout, resolve_workspace


REPO = Path(__file__).resolve().parent
WORKSPACE = ensure_workspace_layout(resolve_workspace(REPO), repo_root=REPO)
TARGET_ID = "claim_validation_hpc_suite"
MANIFEST_PATH = REPO / "manifests" / "claim_validation_hpc_suite_20260511.json"
POLL_INTERVAL_S = 60.0
MAX_WAIT_HOURS = 24.0
BLOCKER_PATTERNS = (
    "run_rerun_chain.py",
    "phase10_baseline.py",
    "phase11_ablation.py",
    "phase12_ewc_sweep.py",
    "phase13_si_sweep.py",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log(message: str) -> None:
    line = f"[{_now_iso()}] {message}"
    print(line, flush=True)


def _audit_note_path() -> Path:
    root = WORKSPACE / "tar_state" / "stat_audit"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"claim_validation_hpc_launcher__{_stamp()}.json"


def _powershell_python_processes() -> list[dict[str, Any]]:
    cmd = [
        "powershell",
        "-Command",
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe' } | "
        "Select-Object ProcessId, Name, CommandLine | ConvertTo-Json -Depth 3",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    data = json.loads(proc.stdout)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def _blocking_processes() -> list[dict[str, Any]]:
    me = os.getpid()
    blockers: list[dict[str, Any]] = []
    for rec in _powershell_python_processes():
        pid = int(rec.get("ProcessId", 0) or 0)
        cmd = str(rec.get("CommandLine", "") or "")
        if pid == me or not cmd:
            continue
        low = cmd.lower()
        if any(pattern in low for pattern in BLOCKER_PATTERNS):
            blockers.append({"pid": pid, "cmd": cmd})
    return blockers


def _load_validation_state() -> dict[str, Any]:
    path = WORKSPACE / "tar_state" / "stabilisation_mode.json"
    if not path.exists():
        raise RuntimeError(f"Validation mode state missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Validation mode state is not a JSON object.")
    return raw


def _write_audit(payload: dict[str, Any]) -> Path:
    path = _audit_note_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _wait_for_nonprimary_runs() -> None:
    deadline = time.time() + (MAX_WAIT_HOURS * 3600.0)
    while True:
        blockers = _blocking_processes()
        if not blockers:
            _log("No blocking non-primary rerun processes remain.")
            return
        if time.time() >= deadline:
            raise TimeoutError(
                "Timed out waiting for current non-primary rerun processes to finish."
            )
        short = ", ".join(f"{b['pid']}:{b['cmd']}" for b in blockers)
        _log(f"Waiting for current non-primary rerun work to finish: {short}")
        time.sleep(POLL_INTERVAL_S)


def main() -> int:
    validation = _load_validation_state()
    if not validation.get("active"):
        raise RuntimeError("Stabilisation mode is not active.")
    if validation.get("primary_validation_experiment_id") != TARGET_ID:
        raise RuntimeError(
            f"Validation mode primary experiment is {validation.get('primary_validation_experiment_id')!r}, "
            f"not {TARGET_ID!r}."
        )

    _log("Validation mode matches the queued primary HPC claim-validation suite.")
    _wait_for_nonprimary_runs()

    orch = ExperimentOrchestrator(WORKSPACE)
    orch.reconcile_runtime_state()
    running = orch.get_running()
    if running:
        raise RuntimeError(
            f"Queue already has running experiment(s): {[spec.id for spec in running]}"
        )

    pending = orch.get_pending()
    if not pending:
        raise RuntimeError("No pending experiments remain in the queue.")
    if pending[0].id != TARGET_ID:
        raise RuntimeError(
            f"Queue head is {pending[0].id!r}, not the authorised target {TARGET_ID!r}."
        )

    orch.set_manifest(MANIFEST_PATH)
    spec = orch._specs.get(TARGET_ID)
    if spec is None:
        raise RuntimeError(f"Target experiment {TARGET_ID!r} not found after reload.")
    if spec.status != "pending":
        raise RuntimeError(
            f"Target experiment {TARGET_ID!r} is not pending (status={spec.status!r})."
        )

    _log(f"Launching authorised experiment {TARGET_ID} using manifest {MANIFEST_PATH.name}.")
    result = orch._execute(spec)
    summary = {
        "launcher": "run_claim_validation_hpc_once.py",
        "launched_at": _now_iso(),
        "manifest_path": str(MANIFEST_PATH),
        "experiment_id": TARGET_ID,
        "result": None if result is None else {
            "experiment_id": result.experiment_id,
            "verdict": result.verdict,
            "status": result.status,
            "result_path": result.result_path,
            "notes": result.notes,
        },
    }
    note = _write_audit(summary)
    _log(f"Launch audit written to {note}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure = {
            "launcher": "run_claim_validation_hpc_once.py",
            "failed_at": _now_iso(),
            "manifest_path": str(MANIFEST_PATH),
            "experiment_id": TARGET_ID,
            "error": str(exc),
        }
        note = _write_audit(failure)
        _log(f"REFUSED/FAILED: {exc}")
        _log(f"Failure audit written to {note}")
        raise
