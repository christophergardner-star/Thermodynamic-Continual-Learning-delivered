"""
tar_execution_watcher.py
Waits for lambda-probe to complete, then replaces the planning-only daemon
with a fresh one that picks up execution_enabled.flag and runs the queue.
Exits after the new daemon is running.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE   = Path("E:/TAR/Thermodynamic-Continual-Learning-delivered")
PYTHON      = Path("C:/Users/cgard/TAR/.venv/Scripts/python.exe")
DAEMON_SCRIPT = Path(__file__).resolve().parent / "tar_living_research.py"
PLANNING_ONLY_PID = 26672
TARGET_EXP  = "director-hyperparameter-robustness-lambda-probe"
QUEUE_FILE  = WORKSPACE / "tar_state" / "experiment_queue.json"
LEDGER_FILE = WORKSPACE / "tar_state" / "runtime_ledger.json"
POLL_S      = 60


def _lambda_probe_running() -> bool:
    try:
        with open(LEDGER_FILE, encoding="utf-8-sig") as f:
            ledger = json.load(f)
        for lease in ledger.get("leases", []):
            if (
                lease.get("experiment_id") == TARGET_EXP
                and lease.get("status") == "running"
            ):
                return True
    except Exception:
        pass
    try:
        with open(QUEUE_FILE, encoding="utf-8-sig") as f:
            q = json.load(f)
        for exp in q.get("experiments", []):
            if exp.get("id") == TARGET_EXP and exp.get("status") == "running":
                return True
    except Exception:
        pass
    return False


def _pid_alive(pid: int) -> bool:
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object Id"],
            capture_output=True, text=True, timeout=10,
        )
        return str(pid) in r.stdout
    except Exception:
        return False


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(WORKSPACE / "tar_state" / "logs" / "execution_watcher.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


_log(f"Execution watcher started. Watching {TARGET_EXP} (planning-only daemon PID {PLANNING_ONLY_PID}).")
_log("Will restart daemon in execution mode once experiment completes.")

while _lambda_probe_running():
    _log(f"  {TARGET_EXP} still running — rechecking in {POLL_S}s")
    time.sleep(POLL_S)

_log(f"{TARGET_EXP} completed or no longer running.")

if _pid_alive(PLANNING_ONLY_PID):
    _log(f"Stopping planning-only daemon (PID {PLANNING_ONLY_PID})...")
    subprocess.run(
        ["powershell", "-Command",
         f"Stop-Process -Id {PLANNING_ONLY_PID} -Force -ErrorAction SilentlyContinue"],
    )
    time.sleep(3)
    _log("Daemon stopped.")
else:
    _log(f"PID {PLANNING_ONLY_PID} already gone.")

_log("Starting fresh daemon (execution_enabled.flag is present — will run queue autonomously).")
subprocess.Popen(
    [str(PYTHON), str(DAEMON_SCRIPT), "--daemon"],
    cwd=str(DAEMON_SCRIPT.parent),
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
)

_log("Fresh daemon started. Watcher exiting.")
