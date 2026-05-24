from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Iterable

try:
    import psutil as _psutil
except Exception:
    _psutil = None


ACTIVE_STATES = {
    "starting",
    "running",
    "waiting",
    "planning",
    "authoring",
    "compiling",
}
TERMINAL_STATES = {"complete", "failed", "refused", "stale", "released"}


class RuntimeLeaseError(RuntimeError):
    """Raised when TAR cannot safely acquire or update a runtime lease."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_ledger_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "runtime_ledger.json"


def runtime_ledger_view_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "runtime_ledger_view.json"


def _default_payload() -> dict[str, Any]:
    return {
        "schema": "tar_runtime_ledger_v1",
        "updated_at": _now_iso(),
        "leases": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write via temp file + os.replace — crash-safe on NTFS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ── Ledger write lock ──────────────────────────────────────────────────────────

def _ledger_lock_path(workspace: Path) -> Path:
    return runtime_ledger_path(workspace).with_suffix(".lock")


def _acquire_ledger_lock(workspace: Path, timeout_s: float = 8.0) -> IO:
    """Exclusive lock via atomic file create. Returns open handle; caller must release."""
    lock_path = _ledger_lock_path(workspace)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            h = open(lock_path, "x", encoding="utf-8")
            h.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
            h.flush()
            return h
        except FileExistsError:
            if time.monotonic() > deadline:
                # Check if lock file is stale (>30s old)
                try:
                    age = time.time() - os.stat(lock_path).st_mtime
                    if age > 30.0:
                        try:
                            os.unlink(lock_path)
                        except FileNotFoundError:
                            pass
                        continue
                except FileNotFoundError:
                    continue
                raise RuntimeLeaseError(
                    f"Ledger lock held by another process for >{timeout_s}s: {lock_path}"
                )
            time.sleep(0.05)


def _release_ledger_lock(lock_handle: IO, workspace: Path) -> None:
    lock_handle.close()
    try:
        os.unlink(_ledger_lock_path(workspace))
    except FileNotFoundError:
        pass


def _process_exists(pid: int) -> bool | None:
    if pid <= 0:
        return False
    if _psutil is None:
        return None
    try:
        proc = _psutil.Process(pid)
        return proc.is_running() and proc.status() != _psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _conflict_keyset(value: Iterable[str] | None) -> list[str]:
    ordered: list[str] = []
    for item in value or []:
        text = str(item or "").strip()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def refresh_runtime_ledger(workspace: Path) -> dict[str, Any]:
    payload = load_runtime_ledger(workspace, refresh=False)
    now = datetime.now(timezone.utc)
    changed = False
    for lease in payload.get("leases", []):
        if not isinstance(lease, dict):
            continue
        status = str(lease.get("status", "") or "")
        if status in TERMINAL_STATES:
            continue
        pid = int(lease.get("pid", 0) or 0)
        stale_timeout_s = float(lease.get("stale_timeout_s", 300.0) or 300.0)
        heartbeat_at = str(lease.get("heartbeat_at", "") or lease.get("started_at", "") or "")
        try:
            heartbeat_dt = datetime.fromisoformat(heartbeat_at.replace("Z", "+00:00"))
        except Exception:
            heartbeat_dt = now
        heartbeat_dt = heartbeat_dt.astimezone(timezone.utc) if heartbeat_dt.tzinfo else heartbeat_dt.replace(tzinfo=timezone.utc)
        process_live = _process_exists(pid)
        heartbeat_age_s = max(0.0, (now - heartbeat_dt).total_seconds())
        if process_live is False or heartbeat_age_s > stale_timeout_s:
            lease["status"] = "stale"
            lease["completed_at"] = _now_iso()
            lease["completion_reason"] = (
                f"runtime lease expired (pid_alive={process_live}, heartbeat_age_s={heartbeat_age_s:.1f})"
            )
            changed = True

    if changed:
        payload["updated_at"] = _now_iso()
        _write_json(runtime_ledger_path(workspace), payload)
    _write_runtime_view(workspace, payload)
    return payload


def load_runtime_ledger(workspace: Path, *, refresh: bool = True) -> dict[str, Any]:
    path = runtime_ledger_path(workspace)
    payload = _read_json(path) or _default_payload()
    leases = payload.get("leases", [])
    if not isinstance(leases, list):
        payload["leases"] = []
    if payload.get("schema") != "tar_runtime_ledger_v1":
        payload["schema"] = "tar_runtime_ledger_v1"
    if refresh:
        return refresh_runtime_ledger(workspace)
    return payload


def save_runtime_ledger(workspace: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    payload["schema"] = "tar_runtime_ledger_v1"
    payload["schema_version"] = "v1"
    payload["updated_at"] = _now_iso()
    if not isinstance(payload.get("leases"), list):
        payload["leases"] = []
    _write_json(runtime_ledger_path(workspace), payload)
    _write_runtime_view(workspace, payload)
    return payload


def active_runtime_leases(workspace: Path) -> list[dict[str, Any]]:
    payload = load_runtime_ledger(workspace)
    return [
        lease for lease in payload.get("leases", [])
        if isinstance(lease, dict) and str(lease.get("status", "") or "") in ACTIVE_STATES
    ]


def find_runtime_conflicts(
    workspace: Path,
    *,
    conflict_keys: Iterable[str] | None = None,
    component_id: str = "",
    experiment_id: str = "",
) -> list[dict[str, Any]]:
    target_keys = set(_conflict_keyset(conflict_keys))
    conflicts: list[dict[str, Any]] = []
    for lease in active_runtime_leases(workspace):
        lease_keys = set(_conflict_keyset(lease.get("conflict_keys", [])))
        same_component = component_id and str(lease.get("component_id", "") or "") == component_id
        same_experiment = experiment_id and str(lease.get("experiment_id", "") or "") == experiment_id
        if same_component or same_experiment or (target_keys and lease_keys.intersection(target_keys)):
            conflicts.append(lease)
    return conflicts


def acquire_runtime_lease(
    workspace: Path,
    *,
    component_id: str,
    component_kind: str,
    status: str = "running",
    conflict_keys: Iterable[str] | None = None,
    pid: int | None = None,
    experiment_id: str = "",
    manifest_id: str = "",
    manifest_path: str = "",
    paper_id: str = "",
    frontier_problem_id: str = "",
    domain_id: str = "",
    owner_component: str = "",
    source_script: str = "",
    stale_timeout_s: float = 300.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lh = _acquire_ledger_lock(workspace)
    try:
        payload = load_runtime_ledger(workspace, refresh=False)
        conflict_keys_list = _conflict_keyset(conflict_keys)
        # Conflict check inside the lock — atomic read-check-write
        target_keys = set(conflict_keys_list)
        conflicts: list[dict[str, Any]] = []
        for lease in payload.get("leases", []):
            if not isinstance(lease, dict):
                continue
            if str(lease.get("status", "") or "") not in ACTIVE_STATES:
                continue
            lease_keys = set(_conflict_keyset(lease.get("conflict_keys", [])))
            same_component = component_id and str(lease.get("component_id", "") or "") == component_id
            same_experiment = experiment_id and str(lease.get("experiment_id", "") or "") == experiment_id
            if same_component or same_experiment or (target_keys and lease_keys.intersection(target_keys)):
                conflicts.append(lease)
        if conflicts:
            conflict_summary = [
                {
                    "lease_id": str(item.get("lease_id", "") or ""),
                    "component_id": str(item.get("component_id", "") or ""),
                    "experiment_id": str(item.get("experiment_id", "") or ""),
                    "conflict_keys": list(item.get("conflict_keys", []) or []),
                    "status": str(item.get("status", "") or ""),
                }
                for item in conflicts
            ]
            raise RuntimeLeaseError(
                f"Refusing duplicate runtime lease for component='{component_id}' experiment='{experiment_id}'. "
                f"Conflicts: {conflict_summary}"
            )

        lease_id = f"{component_kind}:{component_id}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        lease = {
            "lease_id": lease_id,
            "component_id": component_id,
            "component_kind": component_kind,
            "owner_component": owner_component or component_kind,
            "status": status,
            "pid": int(pid or os.getpid()),
            "manifest_id": manifest_id,
            "manifest_path": manifest_path,
            "experiment_id": experiment_id,
            "paper_id": paper_id,
            "frontier_problem_id": frontier_problem_id,
            "domain_id": domain_id,
            "source_script": source_script,
            "started_at": _now_iso(),
            "heartbeat_at": _now_iso(),
            "stale_timeout_s": float(stale_timeout_s or 300.0),
            "conflict_keys": conflict_keys_list,
            "completion_reason": "",
            "completed_at": "",
            "extra": extra or {},
        }
        leases = payload.get("leases", [])
        leases.append(lease)
        payload["leases"] = leases
        save_runtime_ledger(workspace, payload)
        return lease
    finally:
        _release_ledger_lock(lh, workspace)


def heartbeat_runtime_lease(
    workspace: Path,
    *,
    lease_id: str,
    status: str | None = None,
    extra_patch: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    lh = _acquire_ledger_lock(workspace)
    try:
        payload = load_runtime_ledger(workspace, refresh=False)
        for lease in payload.get("leases", []):
            if not isinstance(lease, dict) or str(lease.get("lease_id", "") or "") != lease_id:
                continue
            lease["heartbeat_at"] = _now_iso()
            if status:
                lease["status"] = status
            if extra_patch:
                extra = dict(lease.get("extra", {}) or {})
                extra.update(extra_patch)
                lease["extra"] = extra
            save_runtime_ledger(workspace, payload)
            return lease
        return None
    finally:
        _release_ledger_lock(lh, workspace)


def release_runtime_lease(
    workspace: Path,
    *,
    lease_id: str,
    final_status: str,
    completion_reason: str = "",
    extra_patch: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    lh = _acquire_ledger_lock(workspace)
    try:
        payload = load_runtime_ledger(workspace, refresh=False)
        for lease in payload.get("leases", []):
            if not isinstance(lease, dict) or str(lease.get("lease_id", "") or "") != lease_id:
                continue
            lease["status"] = final_status
            lease["completed_at"] = _now_iso()
            lease["heartbeat_at"] = lease["completed_at"]
            lease["completion_reason"] = completion_reason
            if extra_patch:
                extra = dict(lease.get("extra", {}) or {})
                extra.update(extra_patch)
                lease["extra"] = extra
            save_runtime_ledger(workspace, payload)
            return lease
        return None
    finally:
        _release_ledger_lock(lh, workspace)


def _write_runtime_view(workspace: Path, payload: dict[str, Any]) -> None:
    leases = [
        lease for lease in payload.get("leases", [])
        if isinstance(lease, dict)
    ]
    active = [lease for lease in leases if str(lease.get("status", "") or "") in ACTIVE_STATES]
    view = {
        "schema": "tar_runtime_ledger_view_v1",
        "updated_at": payload.get("updated_at", _now_iso()),
        "active_count": len(active),
        "active_components": sorted({
            str(lease.get("component_id", "") or "")
            for lease in active
            if str(lease.get("component_id", "") or "")
        }),
        "active_experiment_ids": sorted({
            str(lease.get("experiment_id", "") or "")
            for lease in active
            if str(lease.get("experiment_id", "") or "")
        }),
        "active_paper_ids": sorted({
            str(lease.get("paper_id", "") or "")
            for lease in active
            if str(lease.get("paper_id", "") or "")
        }),
        "active_frontier_problem_ids": sorted({
            str(lease.get("frontier_problem_id", "") or "")
            for lease in active
            if str(lease.get("frontier_problem_id", "") or "")
        }),
        "leases": active,
    }
    _write_json(runtime_ledger_view_path(workspace), view)
