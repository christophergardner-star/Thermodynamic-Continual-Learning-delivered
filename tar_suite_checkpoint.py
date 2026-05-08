"""
Checkpoint helpers for long-running multi-seed benchmark suites.

These helpers make Phase 16/17 resumable from the last completed seed boundary.
If no checkpoint exists yet, they can recover safe progress from the suite log.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_path(workspace: str | Path, experiment_id: str) -> Path:
    ws = Path(workspace)
    return ws / "tar_state" / "checkpoints" / f"{experiment_id}.json"


def build_suite_state(
    experiment_id: str,
    seeds: list[int],
    methods: list[str],
    per_seed: list[dict[str, Any]] | None = None,
    status: str = "running",
    source: str = "fresh",
) -> dict[str, Any]:
    rows = list(per_seed or [])
    forgetting = {method: [] for method in methods}
    accuracy = {method: [] for method in methods}
    for row in rows:
        for method in methods:
            f_key = f"{method}_forgetting"
            a_key = f"{method}_acc"
            if row.get(f_key) is not None:
                forgetting[method].append(float(row[f_key]))
            if row.get(a_key) is not None:
                accuracy[method].append(float(row[a_key]))
    return {
        "experiment_id": experiment_id,
        "status": status,
        "source": source,
        "seeds": list(seeds),
        "methods": list(methods),
        "completed_seeds": [int(row["seed"]) for row in rows if "seed" in row],
        "per_seed": rows,
        "forgetting": forgetting,
        "accuracy": accuracy,
        "last_updated": _ts(),
    }


def save_suite_state(path: Path, state: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(state)
        state["last_updated"] = _ts()
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def load_suite_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def clear_suite_state(path: Path) -> None:
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


def append_completed_seed(state: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    rows = list(state.get("per_seed", []))
    existing = {int(item["seed"]): item for item in rows if "seed" in item}
    existing[int(row["seed"])] = dict(row)
    ordered = [existing[seed] for seed in state.get("seeds", []) if seed in existing]
    return build_suite_state(
        experiment_id=state.get("experiment_id", ""),
        seeds=list(state.get("seeds", [])),
        methods=list(state.get("methods", [])),
        per_seed=ordered,
        status=state.get("status", "running"),
        source=state.get("source", "checkpoint"),
    )


def recover_suite_state_from_log(
    experiment_id: str,
    seeds: list[int],
    methods: list[str],
    log_path: Path,
) -> dict[str, Any] | None:
    if not log_path.exists():
        return None

    seed_pattern = re.compile(r"--- seed=(\d+) ---")
    result_pattern = re.compile(r"^\s*(\w+)\s+forgetting=([0-9.]+)\s+acc=([0-9.]+)")
    complete_keys = {f"{method}_forgetting" for method in methods} | {f"{method}_acc" for method in methods}

    rows: list[dict[str, Any]] = []
    current_row: dict[str, Any] | None = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        seed_match = seed_pattern.search(raw_line)
        if seed_match:
            if current_row and complete_keys.issubset(current_row.keys()):
                rows.append(current_row)
            current_row = {"seed": int(seed_match.group(1))}
            continue

        result_match = result_pattern.search(raw_line)
        if result_match and current_row is not None:
            method = result_match.group(1)
            if method in methods:
                current_row[f"{method}_forgetting"] = float(result_match.group(2))
                current_row[f"{method}_acc"] = float(result_match.group(3))

    if current_row and complete_keys.issubset(current_row.keys()):
        rows.append(current_row)

    if not rows:
        return None

    valid_rows = []
    allowed = set(seeds)
    for row in rows:
        seed = int(row.get("seed", -1))
        if seed in allowed:
            valid_rows.append(row)

    if not valid_rows:
        return None

    return build_suite_state(
        experiment_id=experiment_id,
        seeds=seeds,
        methods=methods,
        per_seed=valid_rows,
        status="stalled",
        source="log-recovery",
    )
