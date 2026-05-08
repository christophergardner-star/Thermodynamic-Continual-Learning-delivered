"""
TAR watchdog.

Keeps the key long-running TAR services alive:
  - living research daemon
  - queue maintainer
  - dashboard

The watchdog is designed to be launched at logon so it can restore TAR after
reboots, while also restarting services if they crash mid-run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import ctypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_storage import ensure_workspace_layout, preferred_python, storage_env

_REPO = Path(__file__).resolve().parent
_DETACHED_FLAGS = (
    getattr(subprocess, "DETACHED_PROCESS", 0)
    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
)


@dataclass(frozen=True)
class ServiceConfig:
    service_id: str
    label: str
    args: list[str]
    log_name: str
    health_mode: str
    state_file: str = ""
    health_url: str = ""
    stale_after_s: float = 0.0
    restart_cooldown_s: float = 20.0
    process_match: str = ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts() -> str:
    return _utc_now().isoformat()


def _state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "watchdog_state.json"


def _lock_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "watchdog.lock.json"


def _log_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "logs" / "watchdog.log"


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            process_query_limited_information = 0x1000
            handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            pass
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError as exc:
        winerr = getattr(exc, "winerror", None)
        if winerr == 5:
            return True
        return False
    return True


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_s(value: str) -> float | None:
    ts = _parse_ts(value)
    if ts is None:
        return None
    return max(0.0, (_utc_now() - ts).total_seconds())


def _running_experiment_worker(workspace: Path) -> dict[str, Any] | None:
    proc_path = workspace / "tar_state" / "process_registry.json"
    proc_data = _json_load(proc_path)
    for pid_str, meta in proc_data.items():
        try:
            pid = int(pid_str)
        except (TypeError, ValueError):
            continue
        if pid and _pid_exists(pid):
            return {
                "experiment_id": meta.get("experiment_id", ""),
                "pid": pid,
                "source": "process_registry",
            }

    queue_path = workspace / "tar_state" / "experiment_queue.json"
    data = _json_load(queue_path)
    for rec in data.get("experiments", []):
        if rec.get("status") != "running":
            continue
        pid = int(rec.get("pid") or 0)
        if pid and _pid_exists(pid):
            return {
                "experiment_id": rec.get("id", ""),
                "pid": pid,
                "source": "experiment_queue",
            }
    return None


class TARWatchdog:
    def __init__(self, workspace: Path, *, poll_interval_s: float = 15.0):
        self.workspace = ensure_workspace_layout(workspace, repo_root=_REPO)
        self.repo = _REPO
        self.python = preferred_python(_REPO)
        self.poll_interval_s = poll_interval_s
        self.services = self._build_services()
        self.state: dict[str, Any] = _json_load(_state_path(self.workspace))
        self._ensure_single_instance()

    def _build_services(self) -> list[ServiceConfig]:
        return [
            ServiceConfig(
                service_id="living_research_daemon",
                label="Living Research Daemon",
                args=[
                    str(self.python),
                    str(self.repo / "tar_living_research.py"),
                    "--daemon",
                    "--poll-interval-s",
                    "30",
                ],
                log_name="watchdog-living-research.log",
                health_mode="living_daemon",
                state_file="living_research_daemon.json",
                restart_cooldown_s=30.0,
                process_match="tar_living_research.py --daemon",
            ),
            ServiceConfig(
                service_id="queue_maintainer",
                label="Queue Maintainer",
                args=[
                    str(self.python),
                    str(self.repo / "tar_living_research.py"),
                    "--queue-maintainer",
                    "--poll-interval-s",
                    "30",
                ],
                log_name="watchdog-queue-maintainer.log",
                health_mode="state_file",
                state_file="queue_maintainer_state.json",
                stale_after_s=180.0,
                restart_cooldown_s=30.0,
                process_match="tar_living_research.py --queue-maintainer",
            ),
            ServiceConfig(
                service_id="dashboard",
                label="Dashboard",
                args=[
                    str(self.python),
                    str(self.repo / "tar_dashboard.py"),
                ],
                log_name="watchdog-dashboard.log",
                health_mode="dashboard_http",
                state_file="dashboard_state.json",
                health_url=f"http://127.0.0.1:{int(os.environ.get('TAR_DASHBOARD_PORT', '7860'))}/api/status",
                stale_after_s=120.0,
                restart_cooldown_s=20.0,
                process_match="tar_dashboard.py",
            ),
        ]

    def _matching_python_pids(self, fragment: str) -> list[int]:
        if not fragment:
            return []
        command = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -eq 'python.exe' } | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            return []
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        try:
            raw = json.loads(proc.stdout)
        except Exception:
            return []
        rows = raw if isinstance(raw, list) else [raw]
        needles = fragment.lower().split()
        matches: list[int] = []
        for row in rows:
            cmdline = str((row or {}).get("CommandLine", "") or "").lower()
            pid = int((row or {}).get("ProcessId", 0) or 0)
            if pid and all(needle in cmdline for needle in needles):
                matches.append(pid)
        return sorted(set(matches))

    def _ensure_single_instance(self) -> None:
        lock_path = _lock_path(self.workspace)
        previous = _json_load(lock_path)
        prev_pid = int(previous.get("pid") or 0)
        if prev_pid and prev_pid != os.getpid() and _pid_exists(prev_pid):
            raise SystemExit(f"TAR watchdog already running with PID {prev_pid}")
        _json_write(lock_path, {
            "timestamp": _ts(),
            "pid": os.getpid(),
            "workspace": str(self.workspace),
        })

    def _log(self, msg: str) -> None:
        line = f"[{_ts()}] {msg}"
        print(line, flush=True)
        path = _log_path(self.workspace)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def _service_state_path(self, config: ServiceConfig) -> Path:
        return self.workspace / "tar_state" / config.state_file

    def _http_ok(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=4.0) as resp:
                return 200 <= int(getattr(resp, "status", 200)) < 300
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def _terminate_pid(self, pid: int) -> None:
        if not _pid_exists(pid):
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            pass

    def _spawn(self, config: ServiceConfig) -> int:
        env = storage_env(self.workspace)
        env.update({
            "PYTHONUNBUFFERED": "1",
            "TAR_AUTOSTART_SOURCE": "watchdog",
            "TAR_WATCHDOG_PID": str(os.getpid()),
        })
        log_dir = self.workspace / "tar_state" / "logs"
        stem = Path(config.log_name).stem
        suffix = Path(config.log_name).suffix or ".log"
        log_dir.mkdir(parents=True, exist_ok=True)
        proc = None
        last_error: OSError | None = None
        for attempt in range(4):
            stamp = _utc_now().strftime("%Y%m%d-%H%M%S")
            candidate = (
                log_dir / f"{stem}-{stamp}{suffix}"
                if attempt == 0 else
                log_dir / f"{stem}-{stamp}-pid{os.getpid()}-{attempt}{suffix}"
            )
            try:
                with candidate.open("ab") as log_fh:
                    proc = subprocess.Popen(
                        config.args,
                        cwd=str(self.repo),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        creationflags=_DETACHED_FLAGS,
                    )
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.1)
        if proc is None:
            assert last_error is not None
            raise last_error
        return int(proc.pid)

    def _assess(self, config: ServiceConfig, previous: dict[str, Any]) -> dict[str, Any]:
        tracked_pid = int(previous.get("tracked_pid") or 0)
        state = _json_load(self._service_state_path(config)) if config.state_file else {}
        state_pid = int(state.get("pid") or 0)
        state_age = _age_s(str(state.get("timestamp", "")))
        running_worker = _running_experiment_worker(self.workspace) if config.health_mode == "living_daemon" else None
        worker_pid = int((running_worker or {}).get("pid") or 0)
        matched_pids = self._matching_python_pids(config.process_match)
        matched_pid = matched_pids[0] if matched_pids else 0
        matched_pid_set = set(matched_pids)

        pid_candidates: list[int] = []
        # Prefer the freshest runtime identity over older tracked watchdog memory.
        for pid in (worker_pid, state_pid, matched_pid, tracked_pid):
            if pid and pid not in pid_candidates and _pid_exists(pid):
                pid_candidates.append(pid)
        adopted_pid = pid_candidates[0] if pid_candidates else 0
        state_pid_matches_role = bool(state_pid and state_pid in matched_pid_set)
        tracked_pid_matches_role = bool(tracked_pid and tracked_pid in matched_pid_set)
        adopted_pid_matches_role = bool(adopted_pid and adopted_pid in matched_pid_set)

        healthy = False
        reason = "not_running"

        if config.health_mode == "dashboard_http":
            http_ok = self._http_ok(config.health_url)
            healthy = http_ok
            if http_ok:
                reason = "http_ok"
            elif adopted_pid:
                reason = "http_unhealthy"
        elif config.health_mode == "state_file":
            fresh_state = config.stale_after_s <= 0 or state_age is None or state_age <= config.stale_after_s
            if state_pid and not state_pid_matches_role and matched_pids:
                adopted_pid = matched_pid or 0
                reason = "state_process_mismatch"
            elif state_pid and not state_pid_matches_role and not matched_pids:
                adopted_pid = 0
                reason = "missing_process_match"
            if adopted_pid and fresh_state and (adopted_pid_matches_role or state_pid_matches_role):
                healthy = True
                reason = "state_pid_alive"
            elif adopted_pid and state_age is not None and state_age > config.stale_after_s:
                reason = f"stale_state:{state_age:.0f}s"
        elif config.health_mode == "living_daemon":
            if worker_pid and _pid_exists(worker_pid):
                healthy = True
                adopted_pid = worker_pid
                reason = f"worker_active:{running_worker['experiment_id']}"
            elif matched_pid and _pid_exists(matched_pid):
                healthy = True
                adopted_pid = matched_pid
                reason = "daemon_process_match"
            elif adopted_pid:
                healthy = True
                reason = "daemon_pid_alive"
        if not healthy and config.health_mode != "dashboard_http" and matched_pid and _pid_exists(matched_pid):
            adopted_pid = matched_pid
            healthy = True
            reason = "process_match"

        return {
            "service_id": config.service_id,
            "label": config.label,
            "healthy": healthy,
            "reason": reason,
            "tracked_pid": adopted_pid or tracked_pid,
            "state_pid": state_pid,
            "matched_pid": matched_pid,
            "matched_pid_count": len(matched_pids),
            "state_pid_matches_role": state_pid_matches_role,
            "tracked_pid_matches_role": tracked_pid_matches_role,
            "state_age_s": round(state_age, 1) if state_age is not None else None,
            "state_status": state.get("status", ""),
            "last_checked": _ts(),
            "worker_active": running_worker or {},
        }

    def _restart_allowed(self, previous: dict[str, Any], cooldown_s: float) -> bool:
        age = _age_s(str(previous.get("last_started_at", "")))
        return age is None or age >= cooldown_s

    def _ensure_service(self, config: ServiceConfig, previous: dict[str, Any]) -> dict[str, Any]:
        status = self._assess(config, previous)
        if status["healthy"]:
            status["owned_by_watchdog"] = bool(previous.get("owned_by_watchdog", False))
            status["restart_count"] = int(previous.get("restart_count", 0) or 0)
            status["last_started_at"] = previous.get("last_started_at", "")
            status["last_restart_reason"] = previous.get("last_restart_reason", "")
            return status

        restart_count = int(previous.get("restart_count", 0) or 0)
        if (
            config.health_mode == "living_daemon"
            and status["worker_active"]
            and not status["tracked_pid"]
        ):
            status["deferred_restart"] = True
            status["owned_by_watchdog"] = False
            status["restart_count"] = restart_count
            status["last_started_at"] = previous.get("last_started_at", "")
            status["last_restart_reason"] = "worker still active"
            return status

        if not self._restart_allowed(previous, config.restart_cooldown_s):
            status["cooldown"] = True
            status["owned_by_watchdog"] = bool(previous.get("owned_by_watchdog", False))
            status["restart_count"] = restart_count
            status["last_started_at"] = previous.get("last_started_at", "")
            status["last_restart_reason"] = previous.get("last_restart_reason", "")
            return status

        restart_reason = str(status.get("reason", "restart"))
        old_pid = int(status.get("tracked_pid") or 0)
        if old_pid and _pid_exists(old_pid):
            self._terminate_pid(old_pid)
            time.sleep(1.0)

        new_pid = self._spawn(config)
        self._log(f"watchdog_started service={config.service_id} pid={new_pid} reason={restart_reason}")
        status.update({
            "healthy": True,
            "reason": "spawned",
            "tracked_pid": new_pid,
            "owned_by_watchdog": True,
            "restart_count": restart_count + 1,
            "last_started_at": _ts(),
            "last_restart_reason": restart_reason,
        })
        return status

    def tick(self) -> None:
        previous_services = self.state.get("services", {})
        services_state: dict[str, Any] = {}
        for config in self.services:
            previous = previous_services.get(config.service_id, {})
            services_state[config.service_id] = self._ensure_service(config, previous)

        payload = {
            "timestamp": _ts(),
            "pid": os.getpid(),
            "status": "running",
            "workspace": str(self.workspace),
            "poll_interval_s": self.poll_interval_s,
            "services": services_state,
        }
        self.state = payload
        _json_write(_state_path(self.workspace), payload)
        _json_write(_lock_path(self.workspace), {
            "timestamp": _ts(),
            "pid": os.getpid(),
            "workspace": str(self.workspace),
        })

    def run_forever(self) -> None:
        self._log(f"TAR watchdog start workspace={self.workspace}")
        while True:
            try:
                self.tick()
                time.sleep(self.poll_interval_s)
            except KeyboardInterrupt:
                self._log("TAR watchdog stopped")
                _json_write(_state_path(self.workspace), {
                    "timestamp": _ts(),
                    "pid": os.getpid(),
                    "status": "stopped",
                    "workspace": str(self.workspace),
                    "services": self.state.get("services", {}),
                })
                raise
            except Exception as exc:
                self._log(f"watchdog_error={exc}")
                time.sleep(max(10.0, self.poll_interval_s))


def main() -> None:
    workspace = ensure_workspace_layout(repo_root=_REPO)
    args = sys.argv[1:]
    poll_interval_s = 15.0
    if "--poll-interval-s" in args:
        idx = args.index("--poll-interval-s")
        if idx + 1 < len(args):
            poll_interval_s = float(args[idx + 1])

    watchdog = TARWatchdog(workspace, poll_interval_s=poll_interval_s)
    if "--once" in args:
        watchdog.tick()
        print(json.dumps(_json_load(_state_path(workspace)), indent=2))
        return
    watchdog.run_forever()


if __name__ == "__main__":
    main()
