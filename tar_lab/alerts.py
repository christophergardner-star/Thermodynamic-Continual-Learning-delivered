"""
TAR Alert Bus — write alerts from any subsystem; dashboard polls /api/alerts.

Alerts are append-only to a JSONL ring-buffer file (max 500 entries).
Severity levels: INFO, WARN, ERROR, CRITICAL.
Each alert: {timestamp, severity, source, message, detail}.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_ALERT_FILENAME = "tar_alerts.jsonl"
_MAX_ALERTS = 500
_LOCK_SUFFIX = ".lock"
_LOCK_TIMEOUT = 5.0


def _alerts_path(workspace: Path) -> Path:
    return workspace / "tar_state" / _ALERT_FILENAME


def _lock_path(workspace: Path) -> Path:
    return _alerts_path(workspace).with_suffix(_LOCK_SUFFIX)


def _acquire(workspace: Path) -> Any:
    lock = _lock_path(workspace)
    deadline = time.monotonic() + _LOCK_TIMEOUT
    while True:
        try:
            h = open(lock, "x", encoding="utf-8")
            h.write(json.dumps({"pid": os.getpid()}))
            h.flush()
            return h
        except FileExistsError:
            if time.monotonic() > deadline:
                return None
            time.sleep(0.03)


def _release(handle: Any, workspace: Path) -> None:
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    try:
        _lock_path(workspace).unlink(missing_ok=True)
    except Exception:
        pass


def emit(
    workspace: Path,
    *,
    severity: str,
    source: str,
    message: str,
    detail: str = "",
) -> None:
    """Append one alert to the ring buffer. Non-fatal on any error."""
    severity = severity.upper()
    if severity not in {"INFO", "WARN", "ERROR", "CRITICAL"}:
        severity = "INFO"
    record = {
        "timestamp": _now_iso(),
        "severity": severity,
        "source": source,
        "message": message,
        "detail": detail,
    }
    path = _alerts_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = _acquire(workspace)
    try:
        lines: list[str] = []
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                lines = []
        lines.append(json.dumps(record))
        if len(lines) > _MAX_ALERTS:
            lines = lines[-_MAX_ALERTS:]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass
    finally:
        _release(handle, workspace)


def load_alerts(workspace: Path, *, limit: int = 100) -> list[dict]:
    """Return the most recent `limit` alerts, newest first."""
    path = _alerts_path(workspace)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    alerts: list[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            alerts.append(json.loads(line))
        except Exception:
            continue
        if len(alerts) >= limit:
            break
    return alerts


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
