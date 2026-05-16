"""
Workspace-local experiment worker.

Used when the orchestrator prepares a non-default Python environment for a run.
The worker executes one experiment inside that interpreter while sharing the
same queue/result state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from tar_experiment_orchestrator import ExperimentOrchestrator
from tar_storage import ensure_workspace_layout, resolve_workspace


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_pid_lock(workspace: Path, experiment_id: str, status: str = "running", **extra: object) -> Path:
    """Write a PID lock file so a human can identify and exclude this process before cleanup."""
    lock_dir = workspace / "tar_state" / "run_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{experiment_id}.pid"
    pid = os.getpid()
    payload: dict[str, object] = {
        "pid": pid,
        "experiment_id": experiment_id,
        "manifest_id": os.environ.get("TAR_MANIFEST_PATH", "unknown"),
        "started_at_utc": extra.pop("started_at_utc", _now_iso()),
        "status": status,
        "message": (
            "PROTECTED — authorised HPC validation, do not Stop-Process"
            if status == "running"
            else f"Run {status}"
        ),
        "kill_exclusion_command": (
            f"Get-Process python | Where-Object {{ $_.Id -ne {pid} }} | Stop-Process"
        ),
    }
    payload.update(extra)
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return lock_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single TAR experiment inside a prepared interpreter.")
    parser.add_argument("--workspace", default=str(resolve_workspace(_REPO)))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = ensure_workspace_layout(Path(args.workspace), repo_root=_REPO)

    started_at = _now_iso()
    lock_path = _write_pid_lock(workspace, args.experiment_id, started_at_utc=started_at)
    print(f"[worker] PID lock written: {lock_path}  (PID={os.getpid()})", flush=True)

    try:
        orch = ExperimentOrchestrator(workspace)
        result = orch.execute_by_id(
            args.experiment_id,
            skip_preflight=args.skip_preflight,
            force_in_process=True,
        )
        _write_pid_lock(
            workspace, args.experiment_id, "completed",
            started_at_utc=started_at, completed_at_utc=_now_iso(),
        )
        return 0 if result is not None else 1
    except Exception:
        _write_pid_lock(
            workspace, args.experiment_id, "failed",
            started_at_utc=started_at, failed_at_utc=_now_iso(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
