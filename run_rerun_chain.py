"""
Rerun chain: Phase 12 → Phase 13
Polls the canonical index for Phase 11 completion, then runs Phase 12,
then polls for Phase 12 completion and runs Phase 13.
Each step is gated by its signed manifest (RAIL 3).
"""
import json
import os
import subprocess
import sys
import time
import datetime
from pathlib import Path

_repo = str(Path(__file__).resolve().parent)
sys.path.insert(0, _repo)
from tar_storage import ensure_workspace_layout, resolve_workspace
from tar_lab.runtime_ledger import acquire_runtime_lease, release_runtime_lease, RuntimeLeaseError
from tar_lab.validation import build_validation_state

_ws = ensure_workspace_layout(resolve_workspace(Path(_repo)), repo_root=Path(_repo))
_INDEX = _ws / "tar_state" / "comparisons" / "canonical_results_index.jsonl"

POLL_INTERVAL_S = 60
_CHAIN_STATE = _ws / "tar_state" / "rerun_chain_state.json"


def _write_chain_state(statuses: dict[str, str]) -> None:
    """Write all chain steps with their current status for the dashboard to read."""
    import datetime
    steps = []
    for step in CHAIN:
        steps.append({
            "phase":        step["phase"],
            "logical_name": step["logical_name"],
            "script":       step["script"],
            "manifest_id":  Path(step["manifest"]).stem,
            "log":          step["log"],
            "wait_for":     step["wait_for"],
            "status":       statuses.get(step["logical_name"], "pending"),
        })
    _CHAIN_STATE.parent.mkdir(parents=True, exist_ok=True)
    _CHAIN_STATE.write_text(
        json.dumps({"steps": steps, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def _clear_chain_state() -> None:
    if _CHAIN_STATE.exists():
        _CHAIN_STATE.unlink()


def _indexed_logical_names() -> set[str]:
    if not _INDEX.exists():
        return set()
    names: set[str] = set()
    for line in _INDEX.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            names.add(json.loads(line).get("logical_name", ""))
        except json.JSONDecodeError:
            pass
    return names


def _wait_for(logical_name: str) -> None:
    print(f"[chain] Waiting for '{logical_name}' in canonical index...", flush=True)
    while logical_name not in _indexed_logical_names():
        time.sleep(POLL_INTERVAL_S)
        print(f"  [chain] still waiting for {logical_name}...", flush=True)
    print(f"[chain] '{logical_name}' confirmed in index.", flush=True)


def _write_active_rerun(phase: int, logical_name: str, script: str, manifest: str, log: str) -> None:
    state = {
        "phase": phase,
        "logical_name": logical_name,
        "script": script,
        "manifest_id": manifest,
        "log": log,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running",
    }
    out = _ws / "tar_state" / "active_rerun.json"
    out.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _clear_active_rerun() -> None:
    out = _ws / "tar_state" / "active_rerun.json"
    if out.exists():
        out.unlink()


def _run_step(script: str, manifest: str, log: str, phase: int, logical_name: str) -> None:
    print(f"\n[chain] Launching {script} (manifest={manifest})...", flush=True)
    env = os.environ.copy()
    env["TAR_MANIFEST_PATH"] = manifest
    env["PYTHONIOENCODING"] = "utf-8"
    log_path = Path(log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lease = acquire_runtime_lease(
            _ws,
            component_id=f"rerun_chain:{logical_name}",
            component_kind="rerun_phase",
            experiment_id=logical_name,
            manifest_id=Path(manifest).stem,
            manifest_path=str((Path(_repo) / manifest) if not Path(manifest).is_absolute() else Path(manifest)),
            owner_component="run_rerun_chain",
            source_script=Path(__file__).name,
            conflict_keys=[f"experiment:{logical_name}", f"rerun_chain:{phase}"],
            stale_timeout_s=12 * 3600.0,
            extra={"phase": phase, "log": str(log_path)},
        )
    except RuntimeLeaseError as exc:
        print(f"[chain] REFUSED: {exc}", flush=True)
        sys.exit(1)
    _write_active_rerun(phase, logical_name, script, manifest, log)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=_repo,
            env=env,
            stdout=logf,
            stderr=logf,
        )
    _clear_active_rerun()
    if proc.returncode != 0:
        release_runtime_lease(
            _ws,
            lease_id=str(lease.get("lease_id", "") or ""),
            final_status="failed",
            completion_reason=f"{script} exited {proc.returncode}",
        )
        print(f"[chain] ERROR: {script} exited {proc.returncode}. Chain halted.", flush=True)
        sys.exit(proc.returncode)
    build_validation_state(_ws, persist=True)
    release_runtime_lease(
        _ws,
        lease_id=str(lease.get("lease_id", "") or ""),
        final_status="complete",
        completion_reason=f"{script} completed successfully",
    )
    print(f"[chain] {script} complete.", flush=True)


CHAIN = [
    {
        "wait_for":     "phase11_ablation",
        "phase":        12,
        "logical_name": "phase12_ewc_sweep",
        "script":       "phase12_ewc_sweep.py",
        "manifest":     "manifests/phase12_rerun_20260511.json",
        "log":          str(_ws / "tar_state" / "stat_audit" / "phase12_rerun_20260511.log"),
    },
    {
        "wait_for":     "phase12_ewc_sweep",
        "phase":        13,
        "logical_name": "phase13_si_sweep",
        "script":       "phase13_si_sweep.py",
        "manifest":     "manifests/phase13_rerun_20260511.json",
        "log":          str(_ws / "tar_state" / "stat_audit" / "phase13_rerun_20260511.log"),
    },
]

print("[chain] Rerun chain started: Phase 11 -> Phase 12 -> Phase 13", flush=True)
_statuses: dict[str, str] = {step["logical_name"]: "pending" for step in CHAIN}
_write_chain_state(_statuses)

for step in CHAIN:
    _wait_for(step["wait_for"])
    _statuses[step["logical_name"]] = "running"
    _write_chain_state(_statuses)
    _run_step(step["script"], step["manifest"], step["log"],
              phase=step["phase"], logical_name=step["logical_name"])
    _statuses[step["logical_name"]] = "complete"
    _write_chain_state(_statuses)

print("\n[chain] All reruns complete: Phase 11, 12, 13.", flush=True)
_clear_chain_state()
