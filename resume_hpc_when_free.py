"""
resume_hpc_when_free.py — one-shot watcher that resumes the HPC n=20 replication.

CONTEXT: on 2026-06-03 the HPC replication (run_hpc_replication.py) died at 6/20
seeds and was mislabeled "completed", so it will NOT auto-rerun. run_hpc_replication.py
HAS checkpoint-resume (it skips seeds already in hpc_replication_checkpoint.json), so a
re-launch continues from seed 15 to reach n=20.

This watcher waits until the GPU is free (e.g. HP Selection has finished and nothing
else is training), then asks the dashboard to launch run_hpc_replication.py under
Python 3.11 (the dashboard endpoint already uses the correct CUDA interpreter and has
anti-contention guards). It launches HPC AT MOST ONCE, then exits.

Safety:
  - respects tar_state/daemon_paused.flag (exits without launching if present)
  - requires tar_state/execution_enabled.flag
  - never launches while the GPU is busy (avoids contention on the 4 GB GTX 1650)
  - skips if HPC is already complete (SPRT-decided or seeds_run >= required)
  - delegates the actual launch + PID tracking + guards to the dashboard
    /api/phase2/launch endpoint (falls back to a direct Python 3.11 launch if the
    dashboard is unreachable)
  - bounded: gives up after MAX_WAIT_H hours

Usage:  pythonw resume_hpc_when_free.py      (launch detached; logs to tar_state/logs)
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_PY311 = Path(r"C:\Users\cgard\AppData\Local\Programs\Python\Python311\python.exe")

_TAR_STATE_CANDIDATES = [
    Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state"),
    _REPO / "tar_state",
]
TAR_STATE = next((p for p in _TAR_STATE_CANDIDATES if p.exists()), _REPO / "tar_state")
LOG = TAR_STATE / "logs" / "resume_hpc_watcher.log"
CHECKPOINT = TAR_STATE / "comparisons" / "hpc_replication_checkpoint.json"
PAUSE_FLAG = TAR_STATE / "daemon_paused.flag"
EXEC_FLAG = TAR_STATE / "execution_enabled.flag"

DASHBOARD_URL = "http://localhost:7860/api/phase2/launch"
SCRIPT_KEY = "run_hpc_replication"
REQUIRED_SEEDS = 20

POLL_S = 60
MAX_WAIT_H = 24
GPU_FREE_UTIL = 30        # %
GPU_FREE_MEM_MIB = 900    # MiB
FREE_STREAK_NEEDED = 2    # consecutive free polls (~2 min) before launching


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _hpc_complete() -> bool:
    try:
        ckpt = json.loads(CHECKPOINT.read_text(encoding="utf-8")) if CHECKPOINT.exists() else {}
    except Exception:
        ckpt = {}
    seeds_run = int(ckpt.get("seeds_run", len(ckpt.get("per_seed_results", []) or [])) or 0)
    sprt_log = ckpt.get("sprt_log", []) or []
    decided = bool(sprt_log) and str(sprt_log[-1].get("decision", "")) in {"accept_H1", "accept_H0"}
    return decided or seeds_run >= REQUIRED_SEEDS


def _gpu_free() -> bool:
    """True if the GPU is idle enough to start a run (no training in progress)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        if not out:
            return False
        util_s, mem_s = out[0].split(",")
        util, mem = int(util_s.strip()), int(mem_s.strip())
        return util <= GPU_FREE_UTIL and mem <= GPU_FREE_MEM_MIB
    except Exception:
        return False


def _launch_via_dashboard() -> bool:
    req = urllib.request.Request(
        DASHBOARD_URL,
        data=json.dumps({"script": SCRIPT_KEY}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("ok"):
            _log(f"LAUNCHED via dashboard: pid={body.get('pid')} log={body.get('log')}")
            return True
        _log(f"dashboard refused launch (will retry): {body.get('error')}")
        return False
    except Exception as exc:
        _log(f"dashboard unreachable ({exc}); trying direct launch")
        return _launch_direct()


def _launch_direct() -> bool:
    py = _PY311 if _PY311.exists() else None
    if py is None:
        _log("ERROR: Python 3.11 not found for direct launch; giving up this attempt")
        return False
    log_path = TAR_STATE / "logs" / "hpc_resume_stdout.log"
    try:
        log_fh = open(log_path, "a", encoding="utf-8")
        DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            [str(py), str(_REPO / "run_hpc_replication.py")],
            cwd=str(_REPO), stdout=log_fh, stderr=subprocess.STDOUT,
            creationflags=DETACHED,
        )
        # mirror the dashboard's phase2_pids tracking so panels stay consistent
        pids_path = TAR_STATE / "phase2_pids.json"
        try:
            pids = json.loads(pids_path.read_text(encoding="utf-8")) if pids_path.exists() else {}
        except Exception:
            pids = {}
        pids[SCRIPT_KEY] = {
            "pid": proc.pid, "script": "run_hpc_replication.py",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path), "python": str(py),
        }
        pids_path.write_text(json.dumps(pids, indent=2), encoding="utf-8")
        _log(f"LAUNCHED directly under Python 3.11: pid={proc.pid} log={log_path}")
        return True
    except Exception as exc:
        _log(f"ERROR direct launch failed: {exc}")
        return False


def main() -> None:
    _log(f"watcher start: waiting to resume HPC (script_key={SCRIPT_KEY}, required_seeds={REQUIRED_SEEDS})")
    deadline = time.time() + MAX_WAIT_H * 3600
    free_streak = 0
    last_beat = 0.0
    while time.time() < deadline:
        if time.time() - last_beat > 600:  # heartbeat every ~10 min so the wait is observable
            _log("waiting for a GPU-free window to resume HPC...")
            last_beat = time.time()
        if PAUSE_FLAG.exists():
            _log("daemon_paused.flag present — exiting without launching")
            return
        if _hpc_complete():
            _log("HPC already complete (SPRT-decided or seeds_run >= required) — nothing to do; exiting")
            return
        if not EXEC_FLAG.exists():
            _log("execution_enabled.flag missing — waiting")
            free_streak = 0
            time.sleep(POLL_S)
            continue
        if _gpu_free():
            free_streak += 1
            _log(f"GPU appears free ({free_streak}/{FREE_STREAK_NEEDED})")
            if free_streak >= FREE_STREAK_NEEDED:
                _log("GPU sustained free — attempting HPC resume launch")
                if _launch_via_dashboard():
                    _log("resume launched; watcher exiting (one-shot)")
                    return
                free_streak = 0  # launch refused/failed; back off and retry
        else:
            free_streak = 0
        time.sleep(POLL_S)
    _log(f"MAX_WAIT_H={MAX_WAIT_H}h elapsed without launching — giving up; re-run watcher to retry")


if __name__ == "__main__":
    main()
