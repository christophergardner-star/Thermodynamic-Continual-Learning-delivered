"""
phase2_local_sequencer.py — one-shot sequencer for the LOCAL confirmatory runs.

Purpose: after run_hpc_replication finishes on the GTX 1650, automatically
launch run_mechanistic_ablation (the next local-feasible Phase-2 confirmatory
run), so the ramp's phase2 gate keeps draining overnight without a human
babysitting the GPU. Modeled on resume_hpc_when_free.py.

Behaviour:
  1. Poll every 2 min while the HPC replication is running.
  2. If HPC's tracked PID dies but its checkpoint says INCOMPLETE (SPRT
     undecided and seeds_run < REQUIRED_SEEDS): relaunch it (checkpoint-resume
     makes this safe) — at most MAX_RELAUNCHES times.
  3. When HPC is complete and the GPU is free (2 consecutive polls):
     launch run_mechanistic_ablation via the dashboard endpoint ONCE, then exit.

Safety (same rails as resume_hpc_when_free):
  - exits without launching if tar_state/daemon_paused.flag is present
  - requires tar_state/execution_enabled.flag
  - never launches while the GPU is busy (4 GB GTX 1650, one run at a time)
  - delegates launch + PID tracking + anti-contention guards to the dashboard
    /api/phase2/launch endpoint (sends the X-TAR-Control CSRF header)
  - bounded: gives up after MAX_WAIT_H hours

Usage: launched detached by the session that started the HPC run; logs to
tar_state/logs/phase2_sequencer.log.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TAR_STATE_CANDIDATES = [
    Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state"),
    _REPO / "tar_state",
]
TAR_STATE = next((p for p in _TAR_STATE_CANDIDATES if p.exists()), _REPO / "tar_state")
LOG = TAR_STATE / "logs" / "phase2_sequencer.log"
CHECKPOINT = TAR_STATE / "comparisons" / "hpc_replication_checkpoint.json"
PIDS_PATH = TAR_STATE / "phase2_pids.json"
PAUSE_FLAG = TAR_STATE / "daemon_paused.flag"
EXEC_FLAG = TAR_STATE / "execution_enabled.flag"

DASHBOARD_URL = "http://localhost:7860/api/phase2/launch"
HPC_KEY = "run_hpc_replication"
NEXT_KEY = "run_mechanistic_ablation"
REQUIRED_SEEDS = 20

POLL_S = 120
MAX_WAIT_H = 48
MAX_RELAUNCHES = 3
GPU_FREE_UTIL = 30
GPU_FREE_MEM_MIB = 900
FREE_STREAK_NEEDED = 2


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=20,
        )
        return str(pid) in (out.stdout or "")
    except Exception:
        return True  # conservative: assume alive on query failure


def _tracked_pid(key: str) -> int:
    try:
        pids = json.loads(PIDS_PATH.read_text(encoding="utf-8"))
        return int(pids.get(key, {}).get("pid", 0) or 0)
    except Exception:
        return 0


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
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        if not out:
            return False
        util_s, mem_s = out[0].split(",")
        return int(util_s.strip()) <= GPU_FREE_UTIL and int(mem_s.strip()) <= GPU_FREE_MEM_MIB
    except Exception:
        return False


def _launch(script_key: str) -> bool:
    req = urllib.request.Request(
        DASHBOARD_URL,
        data=json.dumps({"script": script_key}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-TAR-Control": "1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("ok"):
            _log(f"LAUNCHED {script_key}: pid={body.get('pid')} log={body.get('log')}")
            return True
        _log(f"dashboard refused {script_key}: {body.get('error')}")
        return False
    except Exception as exc:
        _log(f"launch {script_key} failed: {exc}")
        return False


def main() -> int:
    _log(f"sequencer start: {HPC_KEY} -> {NEXT_KEY} (poll={POLL_S}s, max {MAX_WAIT_H}h)")
    if not EXEC_FLAG.exists():
        _log("execution_enabled.flag absent — refusing to sequence. exit.")
        return 1

    deadline = time.time() + MAX_WAIT_H * 3600
    relaunches = 0
    free_streak = 0

    while time.time() < deadline:
        if PAUSE_FLAG.exists():
            _log("daemon_paused.flag present — standing down without launching. exit.")
            return 0

        hpc_pid = _tracked_pid(HPC_KEY)
        hpc_running = _pid_alive(hpc_pid)

        if hpc_running:
            free_streak = 0
            time.sleep(POLL_S)
            continue

        # HPC not running: relaunch if incomplete, advance if complete.
        if not _hpc_complete():
            if relaunches >= MAX_RELAUNCHES:
                _log(f"HPC incomplete after {relaunches} relaunches — giving up (human review needed). exit.")
                return 1
            if _gpu_free():
                relaunches += 1
                _log(f"HPC pid dead but checkpoint incomplete — relaunch #{relaunches} (checkpoint-resume).")
                _launch(HPC_KEY)
            time.sleep(POLL_S)
            continue

        # HPC complete -> wait for a stable-free GPU, then launch the ablation once.
        if _gpu_free():
            free_streak += 1
        else:
            free_streak = 0
        if free_streak >= FREE_STREAK_NEEDED:
            _log("HPC complete + GPU free — launching mechanistic ablation.")
            if _launch(NEXT_KEY):
                _log("sequencer done: ablation launched. exit.")
                return 0
            # dashboard refused (e.g. contention) — retry next poll
            free_streak = 0
        time.sleep(POLL_S)

    _log(f"gave up after {MAX_WAIT_H}h without completing the sequence. exit.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
