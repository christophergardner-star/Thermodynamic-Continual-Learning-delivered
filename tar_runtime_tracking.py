"""
Helpers for tracking live legacy/runtime processes in process_registry.json.

The orchestrator already writes its own running experiments. These helpers are
used by standalone scripts so they can publish the same dashboard-visible shape
without needing to know anything about the orchestrator internals.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _registry_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "process_registry.json"


def read_process_registry(workspace: Path) -> dict[str, Any]:
    path = _registry_path(workspace)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_process_registry(workspace: Path, registry: dict[str, Any]) -> None:
    path = _registry_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def upsert_process_entry(
    workspace: Path,
    experiment_id: str,
    stage: str,
    pid: int | None = None,
    owner: str = "legacy-script",
    **metadata: Any,
) -> None:
    proc_id = int(pid or os.getpid())
    registry = read_process_registry(workspace)
    current = registry.get(str(proc_id), {})
    if not isinstance(current, dict):
        current = {}
    current.update({
        "experiment_id": experiment_id,
        "stage": stage,
        "owner": owner,
        **metadata,
    })
    registry[str(proc_id)] = current
    write_process_registry(workspace, registry)


def update_stage(
    workspace: Path,
    experiment_id: str,
    stage: str,
    pid: int | None = None,
    **metadata: Any,
) -> None:
    upsert_process_entry(
        workspace,
        experiment_id=experiment_id,
        stage=stage,
        pid=pid,
        **metadata,
    )


def remove_process_entry(workspace: Path, pid: int | None = None) -> None:
    proc_id = int(pid or os.getpid())
    registry = read_process_registry(workspace)
    registry.pop(str(proc_id), None)
    write_process_registry(workspace, registry)


@contextmanager
def tracked_process(
    workspace: Path,
    experiment_id: str,
    stage: str = "running",
    owner: str = "legacy-script",
    **metadata: Any,
) -> Iterator[None]:
    upsert_process_entry(
        workspace,
        experiment_id=experiment_id,
        stage=stage,
        owner=owner,
        **metadata,
    )
    try:
        yield
    finally:
        remove_process_entry(workspace)
