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
from pathlib import Path

_repo = str(Path(__file__).resolve().parent)
sys.path.insert(0, _repo)
from tar_storage import ensure_workspace_layout, resolve_workspace

_ws = ensure_workspace_layout(resolve_workspace(Path(_repo)), repo_root=Path(_repo))
_INDEX = _ws / "tar_state" / "comparisons" / "canonical_results_index.jsonl"

POLL_INTERVAL_S = 60


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


def _run_step(script: str, manifest: str, log: str) -> None:
    print(f"\n[chain] Launching {script} (manifest={manifest})...", flush=True)
    env = os.environ.copy()
    env["TAR_MANIFEST_PATH"] = manifest
    log_path = Path(log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=_repo,
            env=env,
            stdout=logf,
            stderr=logf,
        )
    if proc.returncode != 0:
        print(f"[chain] ERROR: {script} exited {proc.returncode}. Chain halted.", flush=True)
        sys.exit(proc.returncode)
    print(f"[chain] {script} complete.", flush=True)


CHAIN = [
    {
        "wait_for":  "phase11_ablation",
        "script":    "phase12_ewc_sweep.py",
        "manifest":  "manifests/phase12_rerun_20260511.json",
        "log":       str(_ws / "tar_state" / "stat_audit" / "phase12_rerun_20260511.log"),
    },
    {
        "wait_for":  "phase12_ewc_sweep",
        "script":    "phase13_si_sweep.py",
        "manifest":  "manifests/phase13_rerun_20260511.json",
        "log":       str(_ws / "tar_state" / "stat_audit" / "phase13_rerun_20260511.log"),
    },
]

print("[chain] Rerun chain started: Phase 11 -> Phase 12 -> Phase 13", flush=True)
for step in CHAIN:
    _wait_for(step["wait_for"])
    _run_step(step["script"], step["manifest"], step["log"])

print("\n[chain] All reruns complete: Phase 11, 12, 13.", flush=True)
