"""
Shared run-queue state helpers.

Keeps run_queue, scheduler-facing processes, and the dashboard aligned on the
same queue execution state.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def queue_state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "run_queue_state.json"


def load_queue_state(workspace: Path) -> dict[str, Any]:
    path = queue_state_path(workspace)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_queue_state(workspace: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    data["timestamp"] = _ts()
    path = queue_state_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
    return data


def update_queue_state(
    workspace: Path,
    *,
    queue_name: str,
    status: str,
    current_step: str = "",
    step_index: int = 0,
    step_total: int = 0,
    active_script: str = "",
    active_pid: int = 0,
    message: str = "",
    last_returncode: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    previous = load_queue_state(workspace)
    payload: dict[str, Any] = {
        "queue_name": queue_name,
        "status": status,
        "current_step": current_step,
        "step_index": step_index,
        "step_total": step_total,
        "active_script": active_script,
        "active_pid": active_pid,
        "message": message,
        "last_returncode": last_returncode,
        "started_at": previous.get("started_at") if previous.get("queue_name") == queue_name else _ts(),
    }
    if status in {"complete", "failed", "idle"}:
        payload["finished_at"] = _ts()
    else:
        payload["finished_at"] = previous.get("finished_at", "")
    if extra:
        payload.update(dict(extra))
    return write_queue_state(workspace, payload)


def queue_step_env(
    base_env: Mapping[str, str],
    *,
    queue_name: str,
    current_step: str,
    step_index: int,
    step_total: int,
    active_script: str,
) -> dict[str, str]:
    env = dict(base_env)
    env.update({
        "TAR_QUEUE_NAME": queue_name,
        "TAR_QUEUE_STEP": current_step,
        "TAR_QUEUE_STEP_INDEX": str(step_index),
        "TAR_QUEUE_STEP_TOTAL": str(step_total),
        "TAR_QUEUE_SCRIPT": active_script,
    })
    return env


def heartbeat_from_env(workspace: Path, *, status: str, message: str = "") -> dict[str, Any] | None:
    queue_name = os.environ.get("TAR_QUEUE_NAME", "").strip()
    if not queue_name:
        return None
    step_name = os.environ.get("TAR_QUEUE_STEP", "").strip()
    step_index = int(os.environ.get("TAR_QUEUE_STEP_INDEX", "0") or 0)
    step_total = int(os.environ.get("TAR_QUEUE_STEP_TOTAL", "0") or 0)
    active_script = os.environ.get("TAR_QUEUE_SCRIPT", "").strip()
    return update_queue_state(
        workspace,
        queue_name=queue_name,
        status=status,
        current_step=step_name,
        step_index=step_index,
        step_total=step_total,
        active_script=active_script,
        active_pid=os.getpid(),
        message=message,
    )
