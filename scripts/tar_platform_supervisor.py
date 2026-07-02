"""
TAR platform supervisor — the watchdog's watchdog.

Kills the single-point-of-failure identified in the state audit: nothing
respawned the top-level watchdog if IT died, so a reboot or a watchdog crash
left the whole stack silently dead until a human re-ran it.

This script is idempotent and cheap. Run it on a schedule (see
install_tar_platform_task.ps1 — at logon + every N minutes):

  1. If tar_state/watchdog_autostart.enabled is ABSENT -> do nothing.
     (Durable opt-in. Deleting the flag disables auto-(re)start without
     touching the scheduled task.)
  2. If a watchdog is already alive (watchdog.lock.json pid running) -> nothing.
  3. Otherwise -> relaunch via `python tar_living_research.py --platform`,
     the blessed entrypoint. The watchdog enforces single-instance and RAIL-3
     (it will not (re)start execution-adjacent services without a committed
     manifest), and the autonomy ramp remains fail-closed — so resurrecting
     the platform NEVER runs an experiment without the existing human gates.

SAFETY: this only keeps the SUPERVISOR alive. It does not confirm the ramp,
arm the operator, enable RunPod, or clear daemon_paused.flag. Execution stays
exactly as human-gated as before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_STATE = _REPO / "tar_state"
_AUTOSTART_FLAG = _STATE / "watchdog_autostart.enabled"
_LOCK = _STATE / "watchdog.lock.json"
_LOG = _STATE / "logs" / "supervisor.log"


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _watchdog_alive() -> bool:
    try:
        lock = json.loads(_LOCK.read_text(encoding="utf-8"))
        return _pid_alive(int(lock.get("pid") or 0))
    except Exception:
        return False


def _resolve_python() -> str:
    candidates = [
        _REPO.parent / ".venv" / "Scripts" / "python.exe",   # C:\Users\cgard\TAR\.venv
        _REPO / ".venv" / "Scripts" / "python.exe",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return sys.executable


def main() -> int:
    if not _AUTOSTART_FLAG.exists():
        _log("autostart flag absent — supervisor idle (no action).")
        return 0
    if _watchdog_alive():
        return 0  # healthy; stay quiet to keep the log clean

    python = _resolve_python()
    cmd = [python, str(_REPO / "tar_living_research.py"), "--platform"]
    _log(f"watchdog not alive — relaunching platform: {cmd}")
    flags = (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    try:
        subprocess.Popen(
            cmd, cwd=str(_REPO),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        _log("platform relaunch dispatched.")
        return 0
    except Exception as exc:  # pragma: no cover
        _log(f"relaunch FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
