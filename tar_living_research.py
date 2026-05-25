"""
TAR Living Research Ecosystem
=============================

Production driver that submits the real research portfolio into the experiment
orchestrator, lets the hardware-aware scheduler run it, and then materializes
human-facing autonomous-research outputs and author state.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil as _psutil
except Exception:
    _psutil = None

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from tar_queue_bridge import heartbeat_from_env
from tar_storage import ensure_workspace_layout, resolve_workspace
from tar_experiment_orchestrator import (
    DATASET_CIFAR10,
    DATASET_CIFAR100,
    DATASET_TINYIMAGENET,
    ExperimentOrchestrator,
    ExperimentSpec,
)

_PHASE16_VRAM_BUDGET_GB = 3.2
_PHASE17_VRAM_BUDGET_GB = 3.3
_MIN_ACTIVE_EXPERIMENTS = 6
_MIN_QUEUED_EXPERIMENTS = 6
_MAX_QUEUED_EXPERIMENTS = 15


def _bounded_execution_enabled(workspace: "Path | None" = None) -> bool:
    if str(os.environ.get("TAR_ENABLE_BOUNDED_EXECUTION", "") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return True
    if workspace is not None:
        return (Path(workspace) / "tar_state" / "execution_enabled.flag").exists()
    return False
_LEGACY_DIRECTOR_EXPERIMENT_ALIASES = {
    "director-regime-detection-accuracy-01": "director-regime-detection-accuracy-regime-probe",
    "director-catastrophic-forgetting-01": "director-catastrophic-forgetting-carryover-probe",
    "director-hyperparameter-robustness-01": "director-hyperparameter-robustness-lambda-probe",
}
_LEGACY_DIRECTOR_EXPERIMENT_PREFIX_ALIASES = {
    "director-regime-detection-accuracy-": "director-regime-detection-accuracy-regime-probe",
    "director-catastrophic-forgetting-": "director-catastrophic-forgetting-carryover-probe",
    "director-hyperparameter-robustness-": "director-hyperparameter-robustness-lambda-probe",
}

_DIRECTOR_RUNTIME_METADATA_KEYS = {
    "comparison_methods",
    "external_baselines",
    "candidate_datasets",
    "candidate_backbones",
    "research_strategy",
    "internal_method_role",
}


@dataclass
class HypothesisPlan:
    name: str
    mechanism_description: str
    prediction: str
    breakthrough_criteria: dict[str, float]
    null_prediction: str
    config_overrides: dict[str, object]
    observer_class_name: str = ""
    frontier_problem_id: str = "fp-catastrophic-forgetting"
    author_paper_id: str = ""
    priority: int = 50
    seeds: list[int] = field(default_factory=lambda: [42, 0, 1, 2, 3])


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "living_research.log"


def _daemon_state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "living_research_daemon.json"


def _queue_maintainer_state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "queue_maintainer_state.json"


def _coordination_state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "research_coordination_state.json"


def _observer_state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "state_observer_state.json"


def _log(workspace: Path, msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    path = _log_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


_HEARTBEAT_STALL_MINUTES = 10


def _check_worker_heartbeats(workspace: Path) -> None:
    """Warn if any running worker has missed heartbeats for >10 minutes."""
    lock_dir = workspace / "tar_state" / "run_locks"
    if not lock_dir.exists():
        return
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for lock_file in lock_dir.glob("*.pid"):
        try:
            payload = json.loads(lock_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("status", "")) != "running":
            continue
        hb = str(payload.get("last_heartbeat", "") or "")
        if not hb:
            continue
        try:
            hb_dt = datetime.fromisoformat(hb.replace("Z", "+00:00"))
        except Exception:
            continue
        stale_min = (now - hb_dt).total_seconds() / 60.0
        if stale_min > _HEARTBEAT_STALL_MINUTES:
            exp_id = str(payload.get("experiment_id", lock_file.stem))
            _log(
                workspace,
                f"WARN worker_stall={exp_id} last_heartbeat={hb} "
                f"age_min={stale_min:.1f}",
            )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_service_state(
    workspace: Path,
    path: Path,
    *,
    status: str,
    active_experiment_id: str = "",
    last_event: str = "",
    **extra: object,
) -> None:
    payload = {
        "timestamp": _now_iso(),
        "saved_at": _now_iso(),
        "last_tick": _now_iso(),
        "state_version": 2,
        "status": status,
        "pid": os.getpid(),
        "active_experiment_id": active_experiment_id,
        "last_event": last_event,
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _write_daemon_state(
    workspace: Path,
    *,
    status: str,
    active_experiment_id: str = "",
    last_event: str = "",
    **extra: object,
) -> None:
    _write_service_state(
        workspace,
        _daemon_state_path(workspace),
        status=status,
        active_experiment_id=active_experiment_id,
        last_event=last_event,
        **extra,
    )


def _write_queue_maintainer_state(
    workspace: Path,
    *,
    status: str,
    last_event: str = "",
    **extra: object,
) -> None:
    _write_service_state(
        workspace,
        _queue_maintainer_state_path(workspace),
        status=status,
        last_event=last_event,
        **extra,
    )


def _write_observer_state(
    workspace: Path,
    *,
    status: str,
    last_event: str = "",
    **extra: object,
) -> None:
    _write_service_state(
        workspace,
        _observer_state_path(workspace),
        status=status,
        last_event=last_event,
        **extra,
    )


def _json_load_dict(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _archive_count(workspace: Path) -> int:
    raw = _json_load_dict(workspace / "tar_state" / "experiment_archive.json")
    records = raw.get("experiments", []) if isinstance(raw, dict) else []
    return len(records) if isinstance(records, list) else 0


def _scheduler_file_summary(workspace: Path) -> dict[str, object]:
    data = _json_load_dict(workspace / "tar_state" / "scheduler_state.json")
    can_start = data.get("experiments_to_start", data.get("can_start", []))
    running_ids = data.get("running_ids", [])
    hold_reasons = data.get("hold_reasons", [])
    if not isinstance(can_start, list):
        can_start = []
    if not isinstance(running_ids, list):
        running_ids = []
    if not isinstance(hold_reasons, list):
        hold_reasons = []
    return {
        "scheduler_saved_at": str(data.get("saved_at", "") or data.get("timestamp", "") or ""),
        "scheduler_rationale": str(data.get("rationale_human_text", "") or data.get("rationale", "") or ""),
        "scheduler_can_start": [str(exp_id) for exp_id in can_start if str(exp_id or "")],
        "scheduler_running_ids": [str(exp_id) for exp_id in running_ids if str(exp_id or "")],
        "scheduler_hold_count": len(hold_reasons),
    }


def _queue_runtime_snapshot(workspace: Path, orch: ExperimentOrchestrator) -> dict[str, object]:
    ordered = list(orch._order())
    running_ids: list[str] = []
    pending_ids: list[str] = []
    stalled_ids: list[str] = []
    queued_ids: list[str] = []
    failed_ids: list[str] = []

    for spec in ordered:
        if spec.status == "running":
            running_ids.append(spec.id)
            continue
        if spec.status == "failed":
            failed_ids.append(spec.id)
            continue
        if spec.status == "pending":
            pending_ids.append(spec.id)
            if spec.stage == "stalled":
                stalled_ids.append(spec.id)
            else:
                queued_ids.append(spec.id)

    active_experiment_id = running_ids[0] if running_ids else ""
    next_experiment_id = active_experiment_id or (pending_ids[0] if pending_ids else "")
    return {
        "queue_size": len(ordered),
        "active_experiment_id": active_experiment_id,
        "next_experiment_id": next_experiment_id,
        "running_ids": running_ids,
        "pending_ids": pending_ids,
        "stalled_ids": stalled_ids,
        "queued_ids": queued_ids,
        "failed_ids": failed_ids,
        "running_count": len(running_ids),
        "pending_count": len(pending_ids),
        "stalled_count": len(stalled_ids),
        "queued_count": len(queued_ids),
        "failed_count": len(failed_ids),
        "archive_count": _archive_count(workspace),
    }


def _runtime_state_snapshot(
    workspace: Path,
    orch: ExperimentOrchestrator,
    *,
    poll_interval_s: float,
    scheduler_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    coordination = _json_load_dict(_coordination_state_path(workspace))
    snapshot = _queue_runtime_snapshot(workspace, orch)
    snapshot.update({
        "poll_interval_s": poll_interval_s,
        "coordination_saved_at": str(
            coordination.get("saved_at", "") or coordination.get("timestamp", "") or ""
        ),
        "current_paper_id": str(coordination.get("current_paper_id", "") or ""),
        "current_frontier_problem_id": str(coordination.get("current_frontier_problem_id", "") or ""),
    })
    if scheduler_summary:
        snapshot.update(scheduler_summary)
    else:
        snapshot.update(_scheduler_file_summary(workspace))
    return snapshot


def _start_state_heartbeat(writer_fn, interval_s: float) -> threading.Event:
    stop_event = threading.Event()

    def _loop() -> None:
        while not stop_event.wait(interval_s):
            try:
                writer_fn()
            except Exception:
                pass

    thread = threading.Thread(
        target=_loop,
        daemon=True,
        name=f"tar-state-heartbeat-{os.getpid()}",
    )
    thread.start()
    return stop_event


def _paper_frontier_ids(entry: dict) -> list[str]:
    ids = entry.get("frontier_problem_ids", [])
    if isinstance(ids, list):
        cleaned = [str(frontier_id) for frontier_id in ids if str(frontier_id or "")]
        if cleaned:
            return cleaned
    frontier_id = str(entry.get("frontier_problem_id", "") or "")
    return [frontier_id] if frontier_id else []


def _sync_website_research_json(workspace: Path, director_state: dict | None) -> None:
    """Write website/data/research.json from live Director state. Silent on any error."""
    import re as _re
    try:
        website_json = _REPO.parent / "website" / "data" / "research.json"
        if not website_json.parent.exists():
            return
        paths = director_state.get("active_research_paths", []) if isinstance(director_state, dict) else []
        status_map = {"pursue_now": "active", "pursue_next": "queued", "investigate": "investigating"}
        items = []
        for p in paths:
            if not isinstance(p, dict):
                continue
            title = str(p.get("title", "") or "").strip()
            if not title:
                continue
            why = str(p.get("why_this_now", "") or "")
            m = _re.search(r"(\d+) complete", why)
            exp_count = int(m.group(1)) if m else 0
            allowed_topics = list(p.get("allowed_topics", []) or [])
            description = str(allowed_topics[2]).strip() if len(allowed_topics) > 2 else ""
            items.append({
                "title": title,
                "status": status_map.get(str(p.get("status", "") or ""), "investigating"),
                "frontier_id": str(p.get("target_frontier_problem_id", "") or ""),
                "description": description[:300],
                "experiment_count": exp_count,
                "paper_title": str(p.get("target_paper_id", "") or ""),
                "evidence_strength": "moderate",
                "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            })
        out = {"research": items, "updated_at": datetime.now(timezone.utc).isoformat()}
        website_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def write_research_coordination_state(
    workspace: Path,
    orch: ExperimentOrchestrator,
    director_state: dict | None = None,
    author_state: dict | None = None,
) -> dict:
    validation_mode = _validation_mode_state(workspace)
    director_state = director_state if isinstance(director_state, dict) else _json_load_dict(
        workspace / "tar_state" / "research_director_state.json"
    )
    author_state = author_state if isinstance(author_state, dict) else _json_load_dict(
        workspace / "tar_state" / "author_state.json"
    )

    active_specs, queued_like_specs = _active_or_queued_specs(orch)
    active_paths = director_state.get("active_research_paths", []) if isinstance(director_state, dict) else []
    paper_queue = author_state.get("paper_queue", []) if isinstance(author_state, dict) else []
    current_paper = author_state.get("current_paper", {}) if isinstance(author_state, dict) else {}

    active_queue_entries = [
        entry for entry in paper_queue
        if isinstance(entry, dict) and str(entry.get("scope_status", "") or "") == "active"
    ]
    blocked_entries = [
        entry for entry in active_queue_entries
        if list(entry.get("waiting_for_experiments", []) or [])
    ]

    allowed_paper_ids = sorted({
        str(entry.get("project_id", "") or "")
        for entry in active_queue_entries
        if str(entry.get("project_id", "") or "")
    } | ({
        str(current_paper.get("project_id", "") or "")
    } if str(current_paper.get("project_id", "") or "") else set()))

    allowed_frontier_ids = sorted({
        frontier_id
        for entry in active_queue_entries
        for frontier_id in _paper_frontier_ids(entry)
    } | {
        str(path.get("target_frontier_problem_id", "") or "")
        for path in active_paths
        if isinstance(path, dict)
        and str(path.get("status", "") or "") in {"pursue_now", "incubate"}
        and str(path.get("target_frontier_problem_id", "") or "")
    })

    active_path_ids = sorted({
        str(path.get("path_id", "") or "")
        for path in active_paths
        if isinstance(path, dict)
        and str(path.get("status", "") or "") in {"pursue_now", "incubate"}
        and str(path.get("path_id", "") or "")
    })

    blocked_experiment_ids = sorted({
        str(exp_id)
        for entry in blocked_entries
        for exp_id in entry.get("waiting_for_experiments", []) or []
        if str(exp_id or "")
    })

    aligned_experiment_ids = sorted({
        spec.id for spec in active_specs
        if (
            (spec.frontier_problem_id and spec.frontier_problem_id in allowed_frontier_ids)
            or (spec.author_paper_id and spec.author_paper_id in allowed_paper_ids)
        )
    })

    next_experiment_ids = [
        spec.id for spec in sorted(
            queued_like_specs,
            key=lambda spec: (spec.priority, spec.submitted_at),
        )[:12]
    ]
    active_experiment_ids = [
        spec.id for spec in sorted(
            active_specs,
            key=lambda spec: (spec.priority, spec.submitted_at),
        )
    ]

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_paper_id": str(current_paper.get("project_id", "") or ""),
        "current_paper_title": str(current_paper.get("title", "") or ""),
        "current_paper_status": str(current_paper.get("status", "") or ""),
        "active_path_ids": active_path_ids,
        "allowed_frontier_problem_ids": allowed_frontier_ids,
        "allowed_paper_ids": allowed_paper_ids,
        "blocked_paper_ids": sorted({
            str(entry.get("project_id", "") or "")
            for entry in blocked_entries
            if str(entry.get("project_id", "") or "")
        }),
        "blocked_experiment_ids": blocked_experiment_ids,
        "active_experiment_ids": active_experiment_ids,
        "aligned_experiment_ids": aligned_experiment_ids,
        "next_experiment_ids": next_experiment_ids,
        "summary": {
            "active_experiment_count": len(active_specs),
            "queued_like_count": len(queued_like_specs),
            "active_path_count": len(active_path_ids),
            "allowed_frontier_count": len(allowed_frontier_ids),
            "allowed_paper_count": len(allowed_paper_ids),
            "blocked_paper_count": len(blocked_entries),
            "blocked_experiment_count": len(blocked_experiment_ids),
        },
        "notes": [
            "The Research Director should only seed new experiments that strengthen these active paper/frontier paths.",
            "TAR Author, the Director, and the scheduler should use this shared snapshot as the current coordination picture.",
        ],
    }
    if validation_mode.get("active"):
        primary_frontier = str(validation_mode.get("primary_frontier_problem_id", "") or "")
        primary_paper = str(validation_mode.get("primary_paper_id", "") or "")
        primary_experiment = str(validation_mode.get("primary_validation_experiment_id", "") or "")
        primary_spec = orch._specs.get(primary_experiment) if primary_experiment else None
        primary_waiting = bool(
            primary_spec is not None
            and str(getattr(primary_spec, "status", "") or "") != "complete"
        )
        payload["mode"] = "stabilisation_validation"
        payload["primary_claim"] = str(validation_mode.get("primary_claim", "") or "")
        payload["current_paper_status"] = "blocked" if primary_waiting else "ready"
        payload["current_paper_id"] = primary_paper or payload["current_paper_id"]
        payload["active_path_ids"] = ["validation-hpc-claim"]
        payload["allowed_frontier_problem_ids"] = [primary_frontier] if primary_frontier else []
        payload["allowed_paper_ids"] = [primary_paper] if primary_paper else []
        payload["blocked_paper_ids"] = [primary_paper] if primary_paper and primary_waiting else []
        payload["blocked_experiment_ids"] = [primary_experiment] if primary_experiment and primary_waiting else []
        payload["next_experiment_ids"] = [primary_experiment] if primary_experiment and primary_waiting else []
        payload["post_phase17_observability_task"] = dict(
            validation_mode.get("post_phase17_observability_task", {}) or {}
        )
        payload["summary"].update(
            {
                "active_path_count": 1,
                "allowed_frontier_count": 1 if primary_frontier else 0,
                "allowed_paper_count": 1 if primary_paper else 0,
                "blocked_paper_count": 1 if primary_paper and primary_waiting else 0,
                "blocked_experiment_count": 1 if primary_experiment and primary_waiting else 0,
            }
        )
        payload["notes"] = [
            "Stabilisation mode is active. Exploration is paused and only the single HPC claim validation lane should execute.",
            "All unrelated papers, frontiers, and Director-generated probes are secondary until the claim is verified or rejected.",
            "After Phase 17, future validation runs must emit epoch/task heartbeats, partial JSON snapshots, metric snapshots, PID/resource snapshots, and checkpoint timestamps.",
        ]
    path = _coordination_state_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    _sync_website_research_json(workspace, director_state)
    return payload


def _other_role_pids(flag: str) -> list[int]:
    if _psutil is None:
        return []
    current_pid = os.getpid()
    # On Windows, subprocess.Popen creates a parent python.exe launcher that
    # shares the same cmdline as the child script process. Excluding only
    # current_pid causes the child to detect its own parent as a "duplicate".
    # Exclude the parent PID too so same-launch pairs don't block each other.
    _excluded: set[int] = {current_pid}
    try:
        _excluded.add(_psutil.Process(current_pid).ppid())
    except Exception:
        pass
    matches: list[int] = []
    for proc in _psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(proc.info.get("pid", 0) or 0)
            if pid <= 0 or pid in _excluded:
                continue
            proc_name = str(proc.info.get("name", "") or "").lower()
            if "python" not in proc_name:
                continue
            cmdline = proc.info.get("cmdline") or []
            cmd = " ".join(str(part) for part in cmdline)
            if "tar_living_research.py" in cmd and flag in cmd:
                matches.append(pid)
        except Exception:
            continue
    return sorted(set(matches))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_phase10_baseline(workspace: Path) -> tuple[list[float], float]:
    fallback = [0.1269, 0.1294, 0.1697, 0.1007, 0.1108]
    candidates = [
        workspace / "tar_state" / "comparisons" / "phase10_baseline.json",
        _REPO / "tar_state" / "comparisons" / "phase10_baseline.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            vals = [row["tcl_forgetting"] for row in data.get("per_seed", []) if "tcl_forgetting" in row]
            if vals:
                return vals, _mean(vals)
        except Exception:
            pass
    return fallback, _mean(fallback)


def build_autonomous_plans(workspace: Path) -> list[HypothesisPlan]:
    _baseline_vals, tcl_mean = _load_phase10_baseline(workspace)
    paper_id = "tcl-autonomous-mechanism-paper"
    return [
        HypothesisPlan(
            name="deep_anchor",
            mechanism_description=(
                "Deep Anchor (DA): Longer sigma-star calibration window (50 batches instead "
                "of 20) and longer warmup guard (90 instead of 60) to reduce anchor noise."
            ),
            prediction=f"DA mean forgetting < TCL ({tcl_mean:.4f}) by >0.01, p<0.05, d>0.5.",
            breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
            null_prediction="Longer calibration provides no benefit beyond the current TCL anchor.",
            config_overrides={
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_reset_on_task_boundary": True,
            },
            observer_class_name="DeepAnchorObserver",
            frontier_problem_id="fp-regime-detection-accuracy",
            author_paper_id=paper_id,
            priority=30,
        ),
        HypothesisPlan(
            name="graduated_penalty",
            mechanism_description=(
                "Graduated Penalty (GP): Scale the TCL anchor penalty continuously with depth "
                "into the ordered regime instead of using a binary threshold."
            ),
            prediction=f"GP mean forgetting < TCL ({tcl_mean:.4f}), p<0.05, d>0.5.",
            breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
            null_prediction="Binary penalty is already sufficient; smooth scaling adds no benefit.",
            config_overrides={
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_reset_on_task_boundary": True,
            },
            observer_class_name="GraduatedPenaltyObserver",
            frontier_problem_id="fp-hyperparameter-robustness",
            author_paper_id=paper_id,
            priority=31,
        ),
        HypothesisPlan(
            name="strict_consolidation",
            mechanism_description=(
                "Strict Consolidation (SC): Narrow the TCL critical band so ordered and "
                "disordered regimes are classified more decisively."
            ),
            prediction=f"SC mean forgetting < TCL ({tcl_mean:.4f}), p<0.05, d>0.5.",
            breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
            null_prediction="Tighter thresholds over-constrain plasticity and provide no net gain.",
            config_overrides={
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_reset_on_task_boundary": True,
            },
            observer_class_name="StrictConsolidationObserver",
            frontier_problem_id="fp-regime-detection-accuracy",
            author_paper_id=paper_id,
            priority=32,
        ),
        HypothesisPlan(
            name="thermal_carryover",
            mechanism_description=(
                "Thermal Carry-Over (TCO): Carry sigma-star across task boundaries instead of "
                "restarting thermal calibration from scratch on every task."
            ),
            prediction=f"TCO mean forgetting < TCL ({tcl_mean:.4f}), p<0.05, d>0.5.",
            breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
            null_prediction="Inter-task carry-over over-constrains new-task learning.",
            config_overrides={
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_reset_on_task_boundary": False,
            },
            observer_class_name="CarryoverAnchorObserver",
            frontier_problem_id="fp-catastrophic-forgetting",
            author_paper_id=paper_id,
            priority=33,
        ),
        HypothesisPlan(
            name="high_penalty_conservative",
            mechanism_description=(
                "High-Penalty Conservative (HPC): Increase TCL penalty strength and reduce "
                "ordered-regime learning rate scaling to maximize retention."
            ),
            prediction=f"HPC mean forgetting < {tcl_mean - 0.01:.4f}, p<0.05, d>0.5, acc>0.70.",
            breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
            null_prediction="Aggressive anchoring collapses new-task learning before reducing forgetting.",
            config_overrides={
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.05,
                "tcl_ordered_lr_scale": 0.3,
                "tcl_alpha": 0.45,
                "tcl_reset_on_task_boundary": True,
            },
            frontier_problem_id="fp-hyperparameter-robustness",
            author_paper_id=paper_id,
            priority=34,
        ),
    ]


def build_autonomous_specs(workspace: Path, only_names: set[str] | None = None) -> tuple[list[HypothesisPlan], list[ExperimentSpec]]:
    plans = build_autonomous_plans(workspace)
    specs: list[ExperimentSpec] = []
    for plan in plans:
        if only_names and plan.name not in only_names:
            continue
        specs.append(ExperimentSpec(
            id=f"ar-{plan.name}",
            name=plan.name,
            project_id=f"tcl-{plan.name.replace('_', '-')}-cifar10-v1",
            hypothesis_name=plan.name,
            dataset=DATASET_CIFAR10,
            method="tcl",
            seeds=plan.seeds,
            config_overrides=plan.config_overrides,
            priority=plan.priority,
            estimated_runtime_h=6.0,
            backbone="resnet18",
            epochs=40,
            description=plan.mechanism_description,
            tags=["autonomous_research", plan.name],
            hardware_budget={"vram_gb": 2.5, "cpu_cores": 4},
            frontier_problem_id=plan.frontier_problem_id,
            author_paper_id=plan.author_paper_id,
            observer_class_name=plan.observer_class_name,
        ))
    return plans, specs


def build_scaleup_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            id="phase16_scale_up",
            name="Phase 16 - CIFAR-100 Scale-up",
            project_id="phase16_scale_up",
            hypothesis_name="scale_up_validation",
            dataset=DATASET_CIFAR100,
            method="tcl",
            seeds=[42, 0, 1],
            config_overrides={},
            priority=10,
            estimated_runtime_h=24.0,
            backbone="resnet18",
            epochs=40,
            description=(
                "Scale-up suite testing whether TCL generalizes from Split-CIFAR-10 to "
                "Split-CIFAR-100 across TCL, EWC, and SGD."
            ),
            tags=["scale_up", "phase16"],
            # Calibrated from the empirical local GTX 1650 partial run so the
            # scheduler reflects what this machine has already proven it can resume.
            hardware_budget={"vram_gb": _PHASE16_VRAM_BUDGET_GB, "cpu_cores": 4},
            frontier_problem_id="fp-scale-up",
            author_paper_id="main_tcl_scaleup_paper",
            runner_key="phase16_scale_up_suite",
            context={
                "why": (
                    "This run tests whether the thermodynamic TCL mechanism survives the jump "
                    "from Split-CIFAR-10 to a harder 10-task Split-CIFAR-100 setting."
                ),
                "hypothesis": (
                    "If regime-aware anchoring is genuinely general rather than CIFAR-10-specific, "
                    "TCL should still beat at least one baseline on mean forgetting after scale-up."
                ),
                "projected_outcome": (
                    "Queued for the scale-up slot. TAR will compare TCL against EWC and SGD as "
                    "soon as the GPU becomes available."
                ),
                "frontier_problem": "fp-scale-up",
                "feeds_paper": "Main TCL paper - scale-up section",
                "methodology_note": (
                    "ResNet-18, 40 epochs per task, 3 seeds, methods = TCL / EWC / SGD baseline, "
                    "10 tasks x 10 classes."
                ),
            },
        ),
        ExperimentSpec(
            id="phase17_tinyimagenet",
            name="Phase 17 - TinyImageNet Scale-up",
            project_id="phase17_tinyimagenet",
            hypothesis_name="scale_up_validation",
            dataset=DATASET_TINYIMAGENET,
            method="tcl",
            seeds=[42, 0, 1],
            config_overrides={},
            priority=20,
            estimated_runtime_h=36.0,
            backbone="resnet18",
            epochs=40,
            description=(
                "Scale-up suite testing whether TCL remains competitive on Split-TinyImageNet "
                "when image size, class count, and task complexity all increase."
            ),
            tags=["scale_up", "phase17"],
            # Keep TinyImageNet serialized and slightly above Phase 16 while still
            # targeting a 4 GB-class GPU once the CIFAR-100 scale-up finishes.
            hardware_budget={"vram_gb": _PHASE17_VRAM_BUDGET_GB, "cpu_cores": 4},
            frontier_problem_id="fp-scale-up",
            author_paper_id="main_tcl_scaleup_paper",
            runner_key="phase17_tinyimagenet_suite",
            depends_on=["phase16_scale_up"],
            context={
                "why": (
                    "This run pushes TCL to a materially larger visual benchmark so TAR can test "
                    "whether the mechanism still holds when tasks, images, and class diversity all grow."
                ),
                "hypothesis": (
                    "If the thermal regime signal scales cleanly, TCL should remain directionally better "
                    "than SGD and ideally competitive with EWC even on Split-TinyImageNet."
                ),
                "projected_outcome": (
                    "Waiting behind the CIFAR-100 suite. This is the highest-cost scale-up run and "
                    "will provide the strongest evidence about frontier robustness."
                ),
                "frontier_problem": "fp-scale-up",
                "feeds_paper": "Main TCL paper - scale-up section",
                "methodology_note": (
                    "Adapted ResNet-18 for 64x64 inputs, 40 epochs per task, 3 seeds, methods = TCL / "
                    "EWC / SGD baseline, 10 tasks x 20 classes."
                ),
            },
        ),
    ]


def _preregistration_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "autonomous_research" / "preregistration.json"


def write_preregistration(workspace: Path, plans: list[HypothesisPlan]) -> None:
    prereg = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "hypotheses": [
            {
                "name": plan.name,
                "prediction": plan.prediction,
                "criteria": plan.breakthrough_criteria,
                "frontier_problem_id": plan.frontier_problem_id,
                "author_paper_id": plan.author_paper_id,
            }
            for plan in plans
        ],
    }
    path = _preregistration_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prereg, indent=2), encoding="utf-8")


def _load_all_experiment_records(workspace: Path) -> list[dict]:
    records: dict[str, dict] = {}
    for path in [
        workspace / "tar_state" / "experiment_queue.json",
        workspace / "tar_state" / "experiment_archive.json",
    ]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        experiments = data.get("experiments", []) if isinstance(data, dict) else []
        for rec in experiments if isinstance(experiments, list) else []:
            if isinstance(rec, dict) and rec.get("id"):
                records[str(rec.get("id"))] = rec
    return list(records.values())


def _is_retryable_failed_record(rec: dict[str, object]) -> bool:
    if not isinstance(rec, dict):
        return False
    status = str(rec.get("status", "") or "")
    stage = str(rec.get("stage", "") or "")
    if status not in {"failed", "skipped"} and stage != "failed":
        return False
    result_path = str(rec.get("result_path", "") or "")
    if not result_path:
        return True
    return not Path(result_path).exists()


def _resubmit_failed_archived_spec(
    orch: ExperimentOrchestrator,
    spec: ExperimentSpec,
) -> bool:
    archive_records = orch._load_archive_records()
    if not any(str(rec.get("id", "") or "") == spec.id for rec in archive_records):
        return False
    filtered = [rec for rec in archive_records if str(rec.get("id", "") or "") != spec.id]
    archive_cleaned = True
    try:
        orch._save_archive_records(filtered)
    except OSError as exc:
        archive_cleaned = False
        _log(orch.workspace, f"archive_cleanup_deferred experiment={spec.id} error={exc}")
    spec.status = "pending"
    spec.stage = "queued"
    spec.started_at = ""
    spec.completed_at = ""
    spec.result_path = ""
    spec.error = ""
    spec.pid = 0
    spec.archived_at = ""
    spec.archive_reason = ""
    orch.submit(spec)
    outcome = "archive_cleaned" if archive_cleaned else "archive_cleanup_deferred"
    _log(orch.workspace, f"requeued failed archived experiment {spec.id} ({outcome})")
    return True


def _is_terminal_experiment_record(rec: dict) -> bool:
    status = str(rec.get("status", "") or "")
    stage = str(rec.get("stage", "") or "")
    return status in {"complete", "failed", "skipped"} or stage in {"complete", "failed"}


def _active_or_queued_specs(orch: ExperimentOrchestrator) -> tuple[list[ExperimentSpec], list[ExperimentSpec]]:
    active = [
        spec for spec in orch._specs.values()
        if spec.status not in {"complete", "failed", "skipped"}
        and spec.stage not in {"complete", "failed"}
    ]
    queued_like = [
        spec for spec in active
        if spec.status == "pending"
        or spec.stage in {"planned", "queued", "stalled", "analyzing", "writing_paper"}
    ]
    return active, queued_like


def _validation_mode_state(workspace: Path) -> dict[str, Any]:
    try:
        from tar_validation_mode import load_state

        return load_state(workspace)
    except Exception:
        return {}


def _ensure_validation_mode_queue(workspace: Path, orch: ExperimentOrchestrator) -> list[str]:
    state = _validation_mode_state(workspace)
    if not state.get("active"):
        return []

    from tar_validation_mode import (
        apply_validation_suite_lock,
        build_validation_suite_spec,
        save_state,
        validation_suite_lock_payload,
    )

    submitted_ids: list[str] = []
    suite_spec = build_validation_suite_spec(
        workspace,
        min_seeds=list(state.get("min_seed_list", []) or []),
        target_seeds=list(state.get("target_seed_list", []) or []),
    )
    expected_lock = validation_suite_lock_payload(
        workspace,
        min_seeds=list(state.get("min_seed_list", []) or []),
        target_seeds=list(state.get("target_seed_list", []) or []),
    )
    if state.get("validation_suite_lock", {}).get("fingerprint") != expected_lock.get("fingerprint"):
        state["validation_suite_lock"] = expected_lock
        save_state(workspace, state)
    lock_payload = state.get("validation_suite_lock", expected_lock)
    existing = orch._specs.get(suite_spec.id)
    archived = orch._get_archived_spec(suite_spec.id)
    if existing is None:
        if archived is not None and archived.status in {"complete", "failed"}:
            return []
        orch.submit(suite_spec)
        submitted_ids.append(suite_spec.id)
    else:
        changed = False
        if apply_validation_suite_lock(existing, lock_payload):
            changed = True
        if changed:
            _log(workspace, "healed validation suite drift for claim_validation_hpc_suite")
            orch._save()

    changed = False
    for spec_id, spec in list(orch._specs.items()):
        if spec_id == suite_spec.id or spec.status == "running":
            continue
        if spec.status == "pending":
            spec.status = "skipped"
            spec.stage = "complete"
            spec.completed_at = datetime.now(timezone.utc).isoformat()
            spec.archived_at = spec.completed_at
            spec.archive_reason = "stabilisation_mode_pruned"
            orch._archive_terminal_experiment(spec, reason=spec.archive_reason)
            orch._specs.pop(spec_id, None)
            changed = True
    if changed:
        orch._save()
        orch._refresh_author_state()
        orch._write_process_registry()
    return submitted_ids


def _normalize_live_experiment_specs(orch: ExperimentOrchestrator) -> None:
    changed = False
    targets = {
        "phase16_scale_up": _PHASE16_VRAM_BUDGET_GB,
        "phase17_tinyimagenet": _PHASE17_VRAM_BUDGET_GB,
    }
    for exp_id, vram_gb in targets.items():
        spec = orch._specs.get(exp_id)
        if spec is None:
            continue
        current = float((spec.hardware_budget or {}).get("vram_gb", 0.0) or 0.0)
        if abs(current - vram_gb) > 1e-6:
            spec.hardware_budget = {
                **(spec.hardware_budget or {}),
                "vram_gb": vram_gb,
                "cpu_cores": int((spec.hardware_budget or {}).get("cpu_cores", 4) or 4),
            }
            changed = True
    if changed:
        orch._save()


def _prune_legacy_director_aliases(orch: ExperimentOrchestrator) -> bool:
    changed = False
    alias_pairs = dict(_LEGACY_DIRECTOR_EXPERIMENT_ALIASES)
    for spec_id in list(orch._specs.keys()):
        for prefix, canonical_id in _LEGACY_DIRECTOR_EXPERIMENT_PREFIX_ALIASES.items():
            if spec_id == canonical_id or not spec_id.startswith(prefix):
                continue
            suffix = spec_id[len(prefix):]
            if suffix.isdigit():
                alias_pairs[spec_id] = canonical_id
    for legacy_id, canonical_id in alias_pairs.items():
        legacy = orch._specs.get(legacy_id)
        canonical = orch._specs.get(canonical_id)
        if legacy is None or canonical is None:
            continue
        if str(getattr(legacy, "status", "") or "") == "running":
            continue
        legacy.status = "skipped"
        legacy.archived_at = legacy.archived_at or datetime.now(timezone.utc).isoformat()
        legacy.archive_reason = f"superseded_by_{canonical_id}"
        orch._archive_terminal_experiment(legacy, reason=legacy.archive_reason)
        orch._specs.pop(legacy_id, None)
        changed = True
        _log(orch.workspace, f"pruned legacy director alias {legacy_id} -> {canonical_id}")
    if changed:
        orch._save()
        orch._refresh_author_state()
    return changed


def _director_followup_profile(frontier_id: str, frontier_title: str) -> tuple[str, str, dict[str, object], str]:
    if frontier_id == "fp-regime-detection-accuracy":
        return (
            "regime-probe",
            "director_regime_probe",
            {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.015,
                "tcl_alpha": 0.55,
                "tcl_reset_on_task_boundary": True,
            },
            (
                f"Director-selected follow-up on {frontier_title}. This probe tightens thermodynamic "
                "regime control to check whether sharper calibration improves forgetting."
            ),
        )
    if frontier_id == "fp-hyperparameter-robustness":
        return (
            "lambda-probe",
            "director_lambda_probe",
            {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.02,
                "tcl_alpha": 0.50,
                "tcl_reset_on_task_boundary": True,
            },
            (
                f"Director-selected follow-up on {frontier_title}. This probe checks whether a modestly "
                "stronger anchor improves robustness without collapsing plasticity."
            ),
        )
    if frontier_id == "fp-catastrophic-forgetting":
        return (
            "carryover-probe",
            "director_carryover_probe",
            {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.50,
                "tcl_reset_on_task_boundary": False,
            },
            (
                f"Director-selected follow-up on {frontier_title}. This probe focuses on direct forgetting "
                "reduction by stabilizing the thermal controller across task boundaries."
            ),
        )
    return (
        "frontier-probe",
        "director_frontier_probe",
        {
            "tcl_governor_enabled": True,
            "tcl_penalty_lambda": 0.01,
            "tcl_alpha": 0.50,
            "tcl_reset_on_task_boundary": True,
        },
        (
            f"Director-selected frontier probe on {frontier_title}. TAR is using a low-cost TCL run to keep "
            "pressure on this active research path while stronger evidence accumulates."
        ),
    )


def _priority_from_director_directive(directive: dict[str, object], fallback: int = 50) -> int:
    try:
        rank = int(directive.get("scheduler_rank", 0) or 0)
    except Exception:
        rank = 0
    if rank > 0:
        return max(1, min(95, rank))
    try:
        score = float(directive.get("priority_score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    if score > 0:
        return max(1, min(95, int(max(1.0, 100.0 - min(score, 94.0)))))
    return fallback


def _sync_director_experiment_agenda(
    orch: ExperimentOrchestrator,
    director_state: dict,
) -> bool:
    directives = director_state.get("experiment_directives", []) if isinstance(director_state, dict) else []
    directive_by_id = {
        str(rec.get("experiment_id", "") or ""): rec
        for rec in directives
        if isinstance(rec, dict) and rec.get("experiment_id")
    }
    changed = False
    for spec in orch._specs.values():
        directive = directive_by_id.get(spec.id)
        if not directive:
            continue

        new_priority = _priority_from_director_directive(directive, fallback=int(spec.priority or 50))
        if int(spec.priority or 50) != new_priority:
            spec.priority = new_priority
            changed = True

        new_paper_id = str(directive.get("target_paper_id", "") or spec.author_paper_id or "")
        if new_paper_id and spec.author_paper_id != new_paper_id:
            spec.author_paper_id = new_paper_id
            changed = True

        new_frontier_id = str(directive.get("frontier_problem_id", "") or spec.frontier_problem_id or "")
        if new_frontier_id and spec.frontier_problem_id != new_frontier_id:
            spec.frontier_problem_id = new_frontier_id
            changed = True

        runtime = dict(spec.runtime_context or {})
        runtime_updates = {
            "director_priority_score": float(directive.get("priority_score", 0.0) or 0.0),
            "director_scheduler_rank": int(directive.get("scheduler_rank", 0) or 0),
            "director_intent": str(directive.get("scheduler_intent", "") or ""),
            "director_active_path_id": str(directive.get("active_path_id", "") or ""),
            "global_problem_statement": str(directive.get("global_problem_statement", "") or ""),
            "solution_family": str(directive.get("solution_family", "TAR/TCL/ASC") or "TAR/TCL/ASC"),
            "candidate_datasets": [str(item) for item in directive.get("candidate_datasets", []) if str(item).strip()],
            "candidate_backbones": [str(item) for item in directive.get("candidate_backbones", []) if str(item).strip()],
            "external_baselines": [str(item) for item in directive.get("external_baselines", []) if str(item).strip()],
            "comparison_methods": [str(item) for item in directive.get("comparison_methods", []) if str(item).strip()],
            "research_strategy": str(directive.get("research_strategy", "") or ""),
            "internal_method_role": str(directive.get("internal_method_role", "") or ""),
        }
        for key, value in runtime_updates.items():
            if runtime.get(key) != value:
                runtime[key] = value
                changed = True
        spec.runtime_context = runtime

        context = dict(spec.context or {})
        context_updates = {
            "frontier_problem": new_frontier_id,
            "feeds_paper": new_paper_id,
            "global_problem_statement": str(directive.get("global_problem_statement", "") or ""),
            "solution_family": str(directive.get("solution_family", "TAR/TCL/ASC") or "TAR/TCL/ASC"),
            "solution_novelty_note": str(directive.get("solution_novelty_note", "") or ""),
            "director_focus": str(directive.get("experiment_goal", "") or ""),
            "director_priority": str(directive.get("scheduler_intent", "") or ""),
            "candidate_datasets": ", ".join(str(item) for item in directive.get("candidate_datasets", []) if str(item).strip()),
            "candidate_backbones": ", ".join(str(item) for item in directive.get("candidate_backbones", []) if str(item).strip()),
            "external_baselines": ", ".join(str(item) for item in directive.get("external_baselines", []) if str(item).strip()),
            "research_strategy": str(directive.get("research_strategy", "") or ""),
            "internal_method_role": str(directive.get("internal_method_role", "") or ""),
        }
        for key, value in context_updates.items():
            if value and context.get(key) != value:
                context[key] = value
                changed = True
        spec.context = context
    if changed:
        orch._save()
    return changed


def _build_director_followup_specs(
    workspace: Path,
    director_state: dict,
    existing_records: list[dict],
    coordination_state: dict | None = None,
) -> list[ExperimentSpec]:
    experiment_directives = director_state.get("experiment_directives", []) if isinstance(director_state, dict) else []
    allowed_frontier_ids = {
        str(frontier_id)
        for frontier_id in (coordination_state or {}).get("allowed_frontier_problem_ids", [])
        if str(frontier_id or "")
    }
    allowed_paper_ids = {
        str(paper_id)
        for paper_id in (coordination_state or {}).get("allowed_paper_ids", [])
        if str(paper_id or "")
    }
    active_path_ids = {
        str(path_id)
        for path_id in (coordination_state or {}).get("active_path_ids", [])
        if str(path_id or "")
    }
    current_paper_id = str((coordination_state or {}).get("current_paper_id", "") or "")
    blocked_paper_ids = {
        str(paper_id)
        for paper_id in (coordination_state or {}).get("blocked_paper_ids", [])
        if str(paper_id or "")
    }

    specs: list[ExperimentSpec] = []
    existing_ids = {str(rec.get("id", "") or "") for rec in existing_records if isinstance(rec, dict)}
    existing_by_id: dict[str, list[dict]] = {}
    for rec in existing_records:
        if not isinstance(rec, dict):
            continue
        rec_id = str(rec.get("id", "") or "")
        if not rec_id:
            continue
        existing_by_id.setdefault(rec_id, []).append(rec)
    for directive in experiment_directives:
        if not isinstance(directive, dict):
            continue
        proposal_origin = str(directive.get("proposal_origin", "") or "")
        directive_status = str(directive.get("status", "") or "")
        scheduler_intent = str(directive.get("scheduler_intent", "") or "")
        retryable_existing = (
            proposal_origin == "queue"
            and directive_status == "failed"
            and scheduler_intent in {"retry_now", "queue_now"}
        )
        if proposal_origin not in {"director", "suite"} and not retryable_existing:
            continue
        frontier_id = str(directive.get("frontier_problem_id", "") or "")
        if proposal_origin != "suite" and allowed_frontier_ids and frontier_id and frontier_id not in allowed_frontier_ids:
            continue
        if directive_status not in {"proposed", "failed"}:
            continue

        frontier_title = str(directive.get("frontier_problem_title", frontier_id) or frontier_id)
        path_id = str(directive.get("active_path_id", "") or "")
        if active_path_ids and path_id and path_id not in active_path_ids:
            continue
        spec_id = str(directive.get("experiment_id", "") or "")
        if not spec_id:
            continue
        suppressed = False
        for prefix, canonical_id in _LEGACY_DIRECTOR_EXPERIMENT_PREFIX_ALIASES.items():
            if spec_id != canonical_id and spec_id.startswith(prefix):
                suffix = spec_id[len(prefix):]
                if suffix.isdigit() and canonical_id in existing_ids:
                    suppressed = True
                    break
        if suppressed:
            continue
        retryable_archived_failure = False
        if retryable_existing and spec_id:
            records_for_id = existing_by_id.get(spec_id, [])
            has_live_nonterminal = any(
                not _is_terminal_experiment_record(rec)
                for rec in records_for_id
                if isinstance(rec, dict)
            )
            for rec in records_for_id:
                rec_status = str(rec.get("status", "") or "")
                rec_stage = str(rec.get("stage", "") or "")
                if not has_live_nonterminal and (
                    rec_status in {"failed", "skipped"} or rec_stage == "failed"
                ):
                    retryable_archived_failure = True
                    break
        if spec_id in existing_ids and not retryable_archived_failure:
            continue

        paper_id = str(directive.get("target_paper_id", "") or "")
        if not paper_id and frontier_id:
            paper_id = f"frontier-paper-{frontier_id}"
        if proposal_origin != "suite" and allowed_paper_ids and paper_id and paper_id not in allowed_paper_ids:
            continue
        priority = _priority_from_director_directive(directive, fallback=50)
        if paper_id and paper_id == current_paper_id:
            priority = max(8, priority - 15)
        elif paper_id and paper_id in blocked_paper_ids:
            priority = max(12, priority - 10)
        context = {
            "why": str(directive.get("description", "") or directive.get("experiment_goal", "") or ""),
            "hypothesis": (
                str(directive.get("mechanism_focus", "") or "")
                or f"TAR expects this Director-selected probe to strengthen evidence on {frontier_title} "
                "without drifting away from the active ML problem it has chosen."
            ),
            "projected_outcome": (
                str(directive.get("why_now", "") or "")
                or "Queued by the Research Director in sync with TAR Author. This probe exists because it sharpens "
                "evidence for an active paper/frontier path rather than drifting into side topics."
            ),
            "frontier_problem": frontier_id,
            "feeds_paper": paper_id,
            "global_problem_statement": str(directive.get("global_problem_statement", "") or ""),
            "solution_family": str(directive.get("solution_family", "TAR/TCL/ASC") or "TAR/TCL/ASC"),
            "solution_novelty_note": str(directive.get("solution_novelty_note", "") or ""),
            "methodology_note": str(
                directive.get("description", "")
                or "Director-driven frontier experiment with explicit external baselines."
            ),
            "candidate_datasets": ", ".join(str(item) for item in directive.get("candidate_datasets", []) if str(item).strip()),
            "candidate_backbones": ", ".join(str(item) for item in directive.get("candidate_backbones", []) if str(item).strip()),
            "external_baselines": ", ".join(str(item) for item in directive.get("external_baselines", []) if str(item).strip()),
            "research_strategy": str(directive.get("research_strategy", "") or ""),
            "internal_method_role": str(directive.get("internal_method_role", "") or ""),
        }
        cfg = {
            str(key): value
            for key, value in dict(directive.get("config_overrides", {}) or {}).items()
            if str(key) not in _DIRECTOR_RUNTIME_METADATA_KEYS
        }
        specs.append(ExperimentSpec(
            id=spec_id,
            name=str(directive.get("title", "") or f"Director Follow-up - {frontier_title}"),
            project_id=f"{spec_id}-{str(directive.get('dataset', DATASET_CIFAR10)).replace('split_', '')}-v1",
            hypothesis_name=str(directive.get("hypothesis_name", "") or "director_frontier_probe"),
            dataset=str(directive.get("dataset", DATASET_CIFAR10) or DATASET_CIFAR10),
            method=str(directive.get("method", "tcl") or "tcl"),
            seeds=[int(seed) for seed in directive.get("seeds", [42, 0, 1]) or [42, 0, 1]],
            config_overrides=cfg,
            priority=priority,
            estimated_runtime_h=float(directive.get("estimated_runtime_h", 6.0) or 6.0),
            backbone=str(directive.get("backbone", "resnet18") or "resnet18"),
            epochs=int(directive.get("epochs", 40) or 40),
            description=str(directive.get("description", "") or directive.get("experiment_goal", "") or ""),
            tags=[
                "director_generated",
                frontier_id,
                path_id,
                paper_id,
            ],
            hardware_budget=dict(directive.get("hardware_budget", {"vram_gb": 2.5, "cpu_cores": 4}) or {"vram_gb": 2.5, "cpu_cores": 4}),
            frontier_problem_id=frontier_id,
            author_paper_id=paper_id,
            context=context,
            depends_on=[str(dep) for dep in directive.get("depends_on", []) or [] if str(dep or "")],
        ))
        existing_ids.add(spec_id)
    return specs


def _ensure_director_seeded_queue(
    workspace: Path,
    orch: ExperimentOrchestrator,
    *,
    coordination_state: dict | None = None,
    min_active_experiments: int = _MIN_ACTIVE_EXPERIMENTS,
    min_queued_experiments: int = _MIN_QUEUED_EXPERIMENTS,
) -> list[str]:
    validation_mode = _validation_mode_state(workspace)
    if validation_mode.get("active"):
        return _ensure_validation_mode_queue(workspace, orch)

    from tar_lab.human_review import approved_experiment_ids
    from tar_research_director import ResearchDirector

    _normalize_live_experiment_specs(orch)
    _prune_legacy_director_aliases(orch)
    director_state = ResearchDirector(workspace).update_state()
    if coordination_state is None:
        coordination_state = write_research_coordination_state(workspace, orch, director_state, None)
    _sync_director_experiment_agenda(orch, director_state)
    existing_records = _load_all_experiment_records(workspace)
    existing_ids = {
        str(rec.get("id", "") or "") for rec in existing_records
        if isinstance(rec, dict) and rec.get("id")
    }
    retryable_failed_ids = {
        str(rec.get("id", "") or "")
        for rec in existing_records
        if isinstance(rec, dict) and rec.get("id") and _is_retryable_failed_record(rec)
    }
    existing_ids.update(orch._specs.keys())

    plans, autonomous_specs = build_autonomous_specs(workspace)
    del plans
    canonical_specs = build_scaleup_specs() + autonomous_specs
    submitted_ids: list[str] = []
    _active_statuses = {"pending", "running"}
    _current_active = sum(1 for s in orch._specs.values() if s.status in _active_statuses)

    active_frontier_ids = set(
        director_state.get("summary", {}).get("active_frontier_problem_ids", [])
        if isinstance(director_state, dict) else []
    )
    approved_ids = approved_experiment_ids(workspace)
    allowed_frontier_ids = set(
        coordination_state.get("allowed_frontier_problem_ids", [])
        if isinstance(coordination_state, dict) else []
    )
    if allowed_frontier_ids:
        active_frontier_ids = (
            active_frontier_ids & allowed_frontier_ids
            if active_frontier_ids else
            allowed_frontier_ids
        )
    for spec in canonical_specs:
        if _current_active + len(submitted_ids) >= _MAX_QUEUED_EXPERIMENTS:
            break
        if spec.id in existing_ids:
            continue
        if spec.id not in approved_ids:
            continue
        if active_frontier_ids and spec.frontier_problem_id and spec.frontier_problem_id not in active_frontier_ids:
            continue
        orch.submit(spec)
        submitted_ids.append(spec.id)
        existing_ids.add(spec.id)

    frontier_directives = (
        director_state.get("frontier_directives", [])
        if isinstance(director_state, dict) else []
    )
    active_paths = (
        director_state.get("active_research_paths", [])
        if isinstance(director_state, dict) else []
    )
    followups = _build_director_followup_specs(
        workspace,
        director_state,
        existing_records + [asdict(spec) for spec in orch._specs.values()],
        coordination_state=coordination_state,
    )
    for spec in followups:
        if spec.id in existing_ids:
            if spec.id in retryable_failed_ids and _resubmit_failed_archived_spec(orch, spec):
                submitted_ids.append(spec.id)
                retryable_failed_ids.discard(spec.id)
            continue
        if _current_active + len(submitted_ids) >= _MAX_QUEUED_EXPERIMENTS:
            break
        if spec.id not in approved_ids:
            continue
        orch.submit(spec)
        submitted_ids.append(spec.id)
        existing_ids.add(spec.id)

    _sync_director_experiment_agenda(orch, director_state)

    if submitted_ids:
        _log(workspace, f"director_top_up submitted={submitted_ids}")
    return submitted_ids


def _result_from_orchestrator(spec_record: dict, plan: HypothesisPlan) -> dict | None:
    result_path = spec_record.get("result_path", "")
    if not result_path:
        return None
    path = Path(result_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    seed_results = raw.get("seed_results", [])
    forgetting = [row.get("forgetting") for row in seed_results if "forgetting" in row]
    accuracy = [row.get("accuracy") for row in seed_results if "accuracy" in row]
    return {
        "hypothesis": {
            "name": plan.name,
            "mechanism_description": plan.mechanism_description,
            "prediction": plan.prediction,
            "breakthrough_criteria": plan.breakthrough_criteria,
            "null_prediction": plan.null_prediction,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        },
        "result": {
            "hypothesis_name": plan.name,
            "seeds": raw.get("seeds", spec_record.get("seeds", [])),
            "mechanism_forgetting": forgetting,
            "baseline_forgetting": raw.get("baseline_forgetting", []),
            "mechanism_accuracy": accuracy,
            "mean_delta": raw.get("mean_delta", 0.0),
            "t_stat": raw.get("t_stat", 0.0),
            "p_val": raw.get("p_val", 1.0),
            "cohens_d": raw.get("cohens_d", 0.0),
            "n_better": raw.get("n_better", 0),
            "verdict": raw.get("verdict", "NULL"),
            "notes": raw.get("notes", ""),
            "run_at": raw.get("completed_at", datetime.now(timezone.utc).isoformat()),
        },
    }


def finalize_autonomous_results(workspace: Path, plans: list[HypothesisPlan]) -> list[dict]:
    experiment_records: dict[str, dict] = {}
    for path in [
        workspace / "tar_state" / "experiment_queue.json",
        workspace / "tar_state" / "experiment_archive.json",
    ]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        experiments = data.get("experiments", []) if isinstance(data, dict) else []
        for rec in experiments if isinstance(experiments, list) else []:
            if isinstance(rec, dict) and rec.get("id"):
                experiment_records[str(rec.get("id"))] = rec
    by_id = experiment_records
    out_dir = workspace / "tar_state" / "autonomous_research"
    out_dir.mkdir(parents=True, exist_ok=True)

    finalized: list[dict] = []
    for plan in plans:
        spec_record = by_id.get(f"ar-{plan.name}")
        if not spec_record or spec_record.get("status") != "complete":
            continue
        record = _result_from_orchestrator(spec_record, plan)
        if not record:
            continue
        (out_dir / f"{plan.name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        finalized.append(record)

    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_hypotheses_tested": len(finalized),
        "results": [
            {
                "name": rec["hypothesis"]["name"],
                "verdict": rec["result"]["verdict"],
                "mean_delta": rec["result"]["mean_delta"],
                "p_val": rec["result"]["p_val"],
                "cohens_d": rec["result"]["cohens_d"],
                "n_better": rec["result"]["n_better"],
                "notes": rec["result"]["notes"],
            }
            for rec in finalized
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return finalized


def _maybe_write_breakthrough_papers(
    workspace: Path,
    finalized: list[dict],
    written_hypotheses: set[str],
) -> None:
    from tar_author import PaperSpec, TARAuthor

    for breakthrough in finalized:
        if breakthrough["result"]["verdict"] != "BREAKTHROUGH":
            continue
        hyp_name = breakthrough["hypothesis"]["name"]
        if hyp_name in written_hypotheses:
            continue
        paper_dir = workspace / "paper" / hyp_name
        if (paper_dir / "main.tex").exists():
            written_hypotheses.add(hyp_name)
            continue
        spec = PaperSpec(
            title=(
                "Thermodynamic Continual Learning - "
                f"{hyp_name.replace('_', ' ').title()}: "
                "Autonomous Mechanism Study"
            ),
            authors=["Christopher Gardner", "TAR (Thermodynamic Autonomous Researcher)"],
            affiliation="Independent Research",
            project_id=f"autonomous_{hyp_name}",
            paper_dir=paper_dir,
            workspace=workspace,
        )
        TARAuthor(workspace=workspace).write_paper(spec)
        written_hypotheses.add(hyp_name)


def _maybe_start_ready_author_paper(
    workspace: Path,
    started_papers: set[str],
) -> dict | None:
    from tar_author import auto_start_priority_paper

    state_path = workspace / "tar_state" / "author_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            current = state.get("current_paper", {}) if isinstance(state, dict) else {}
            current_project = str(current.get("project_id", "") or "")
            if current_project and current.get("status") in {"starting", "writing"}:
                started_papers.add(current_project)
                return None
        except Exception:
            pass

    result = auto_start_priority_paper(workspace)
    if not result:
        return None
    project_id = str(result.get("project_id", "") or "")
    if project_id and project_id not in started_papers:
        started_papers.add(project_id)
    return result


def _maybe_compile_completed_author_papers(workspace: Path) -> None:
    from tar_author import ensure_completed_papers_compiled

    try:
        ensure_completed_papers_compiled(workspace)
    except Exception as exc:
        _log(workspace, f"author_compile_check_failed={exc}")


def submit_portfolio(
    workspace: Path,
    include_scaleup: bool = True,
    include_autonomous: bool = True,
    only_autonomous_names: set[str] | None = None,
) -> tuple[ExperimentOrchestrator, list[HypothesisPlan]]:
    from tar_author import configure_live_author_llm, write_planned_author_state
    from tar_evidence_ingest import ExternalEvidenceIngestor
    from tar_frontier import FrontierRegistry
    from tar_hardware_monitor import HardwareMonitor
    from tar_research_director import ResearchDirector

    workspace = ensure_workspace_layout(workspace, repo_root=_REPO)
    FrontierRegistry(workspace)
    HardwareMonitor(workspace, interval_s=5.0).start_background()
    ExternalEvidenceIngestor(workspace).ensure_state()
    configure_live_author_llm(workspace)

    orch = ExperimentOrchestrator(workspace)
    plans: list[HypothesisPlan] = []
    validation_mode = _validation_mode_state(workspace)

    if validation_mode.get("active"):
        from tar_validation_mode import build_validation_suite_spec

        suite_spec = build_validation_suite_spec(
            workspace,
            min_seeds=list(validation_mode.get("min_seed_list", []) or []),
            target_seeds=list(validation_mode.get("target_seed_list", []) or []),
        )
        orch.submit(suite_spec)
        director_state = ResearchDirector(workspace).update_state()
        author_state = write_planned_author_state(workspace)
        write_research_coordination_state(workspace, orch, director_state, author_state)
        return orch, plans

    specs: list[ExperimentSpec] = []
    if include_scaleup:
        specs.extend(build_scaleup_specs())
    if include_autonomous:
        plans, autonomous_specs = build_autonomous_specs(workspace, only_names=only_autonomous_names)
        specs.extend(autonomous_specs)
        write_preregistration(workspace, plans)

    orch.submit_many(specs)
    director_state = ResearchDirector(workspace).update_state()
    author_state = write_planned_author_state(workspace)
    coordination_state = write_research_coordination_state(workspace, orch, director_state, author_state)
    _ensure_director_seeded_queue(workspace, orch, coordination_state=coordination_state)
    author_state = write_planned_author_state(workspace)
    write_research_coordination_state(workspace, orch, director_state, author_state)
    return orch, plans


def run_portfolio(
    workspace: Path,
    include_scaleup: bool = True,
    include_autonomous: bool = True,
    only_autonomous_names: set[str] | None = None,
) -> list[dict]:
    from tar_author import write_planned_author_state
    from tar_research_director import ResearchDirector

    workspace = ensure_workspace_layout(workspace, repo_root=_REPO)
    _log(workspace, "=" * 70)
    _log(workspace, "TAR LIVING RESEARCH ECOSYSTEM START")
    _log(workspace, f"workspace={workspace}")
    orch, plans = submit_portfolio(
        workspace,
        include_scaleup=include_scaleup,
        include_autonomous=include_autonomous,
        only_autonomous_names=only_autonomous_names,
    )
    _log(workspace, f"submitted={len(orch._specs)} experiments total")
    author_state = write_planned_author_state(workspace)
    director_state = ResearchDirector(workspace).update_state()
    coordination_state = write_research_coordination_state(workspace, orch, director_state, author_state)
    _ensure_director_seeded_queue(workspace, orch, coordination_state=coordination_state)
    if not _bounded_execution_enabled(workspace):
        _log(workspace, "planner-only mode active; living research portfolio will not start execution without explicit bounded execution enablement.")
        heartbeat_from_env(workspace, status="running", message="living research planner-only mode")
        return []
    orch.run_parallel()
    heartbeat_from_env(workspace, status="running", message="living research portfolio active")

    finalized = finalize_autonomous_results(workspace, plans) if include_autonomous else []

    director_state = ResearchDirector(workspace).update_state()
    _maybe_write_breakthrough_papers(workspace, finalized, written_hypotheses=set())
    author_state = write_planned_author_state(workspace)
    write_research_coordination_state(workspace, orch, director_state, author_state)
    _maybe_start_ready_author_paper(workspace, started_papers=set())
    _maybe_compile_completed_author_papers(workspace)
    _log(workspace, f"portfolio complete; finalized_autonomous={len(finalized)}")
    heartbeat_from_env(workspace, status="running", message="living research portfolio complete")
    return finalized


def run_portfolio_daemon(
    workspace: Path,
    include_scaleup: bool = True,
    include_autonomous: bool = True,
    only_autonomous_names: set[str] | None = None,
    poll_interval_s: float = 30.0,
) -> None:
    from tar_author import write_planned_author_state
    from tar_evidence_ingest import ExternalEvidenceIngestor
    from tar_research_director import ResearchDirector
    from tar_scheduler import TARScheduler

    workspace = ensure_workspace_layout(workspace, repo_root=_REPO)
    duplicate_pids = _other_role_pids("--daemon")
    if duplicate_pids:
        _log(workspace, f"duplicate living research daemon detected; refusing second start. existing_pids={duplicate_pids}")
        _write_daemon_state(workspace, status="duplicate_blocked", last_event=f"existing daemon pids={duplicate_pids}")
        return
    _log(workspace, "=" * 70)
    _log(workspace, "TAR LIVING RESEARCH DAEMON START")
    _log(workspace, f"workspace={workspace}")
    _write_daemon_state(workspace, status="starting", last_event="daemon boot")
    orch, plans = submit_portfolio(
        workspace,
        include_scaleup=include_scaleup,
        include_autonomous=include_autonomous,
        only_autonomous_names=only_autonomous_names,
    )
    # Enable autonomous manifest generation so the daemon can satisfy RAIL 3
    # without human intervention. Each experiment gets a minimal manifest that
    # is generated, committed to git, and hash-verified before execution.
    # authorised_by is set to "TAR Director (autonomous)" — TAR does not lie.
    orch.set_autonomous(True)
    _log(workspace, f"submitted={len(orch._specs)} experiments total")
    scheduler = TARScheduler(workspace)
    evidence = ExternalEvidenceIngestor(workspace, poll_interval_s=max(900.0, poll_interval_s * 10.0))
    evidence.start_background()
    written_hypotheses: set[str] = set()
    started_papers: set[str] = set()
    director_state = ResearchDirector(workspace).update_state()
    author_state = write_planned_author_state(workspace)
    coordination_state = write_research_coordination_state(workspace, orch, director_state, author_state)
    _ensure_director_seeded_queue(workspace, orch, coordination_state=coordination_state)

    # Release leases whose PIDs are dead — uses psutil (now installed), atomic write
    def _release_dead_leases_at_startup(ws: Path) -> None:
        try:
            import psutil as _ps
        except ImportError:
            import logging as _log
            _log.getLogger(__name__).warning(
                "psutil not available — skipping startup dead-lease cleanup. "
                "Install psutil to enable duplicate daemon detection and lease cleanup."
            )
            return
        from tar_lab.runtime_ledger import load_runtime_ledger, save_runtime_ledger, ACTIVE_STATES
        try:
            payload = load_runtime_ledger(ws, refresh=False)
        except Exception:
            return
        changed = False
        now_iso = datetime.now(timezone.utc).isoformat()
        for lease in payload.get("leases", []):
            if not isinstance(lease, dict):
                continue
            if str(lease.get("status", "")) not in ACTIVE_STATES:
                continue
            pid = int(lease.get("pid", 0) or 0)
            if pid <= 0:
                continue
            try:
                alive = _ps.pid_exists(pid)
            except Exception:
                alive = True  # conservative: assume alive on error
            if not alive:
                lease["status"] = "released"
                lease["completion_reason"] = "stale_pid_dead_at_daemon_startup"
                lease["completed_at"] = now_iso
                lease["pid"] = 0
                changed = True
        if changed:
            payload["updated_at"] = now_iso
            save_runtime_ledger(ws, payload)

    _release_dead_leases_at_startup(workspace)

    daemon_status = {"status": "starting", "last_event": "daemon boot"}
    execution_enabled = _bounded_execution_enabled()
    heartbeat_interval_s = max(15.0, min(60.0, poll_interval_s))

    def _scheduler_summary() -> dict[str, object]:
        pending = orch.get_pending()
        running = orch.get_running()
        decision = scheduler.decide(pending_specs=pending, running_specs=running)
        return {
            "scheduler_saved_at": decision.timestamp,
            "scheduler_rationale": decision.rationale,
            "scheduler_can_start": list(decision.can_start),
            "scheduler_running_ids": list(decision.running_ids),
            "scheduler_hold_count": len(decision.hold_reasons),
        }

    def _emit_daemon_state(*, refresh_scheduler: bool) -> None:
        scheduler_summary = _scheduler_summary() if refresh_scheduler else None
        snapshot = _runtime_state_snapshot(
            workspace,
            orch,
            poll_interval_s=poll_interval_s,
            scheduler_summary=scheduler_summary,
        )
        active_experiment_id = str(snapshot.get("active_experiment_id", "") or "")
        last_event = str(daemon_status.get("last_event", "") or "")
        if active_experiment_id and last_event in {"daemon boot", "scheduler poll", "sleeping"}:
            last_event = f"monitoring {active_experiment_id}"
        payload = dict(snapshot)
        payload.pop("active_experiment_id", None)
        _write_daemon_state(
            workspace,
            status=str(daemon_status.get("status", "running") or "running"),
            active_experiment_id=active_experiment_id,
            last_event=last_event,
            **payload,
        )

    _emit_daemon_state(refresh_scheduler=True)
    heartbeat_stop = _start_state_heartbeat(
        lambda: _emit_daemon_state(refresh_scheduler=True),
        heartbeat_interval_s,
    )

    while True:
        try:
            daemon_status.update({"status": "running", "last_event": "scheduler poll"})
            _emit_daemon_state(refresh_scheduler=True)
            heartbeat_from_env(workspace, status="running", message="living research daemon polling")
            director_state = ResearchDirector(workspace).update_state()
            _check_worker_heartbeats(workspace)
            try:
                author_state = write_planned_author_state(workspace)
            except Exception as _auth_exc:
                _log(workspace, f"author_warn={_auth_exc}")
                author_state = {}
            coordination_state = write_research_coordination_state(workspace, orch, director_state, author_state)
            _ensure_director_seeded_queue(workspace, orch, coordination_state=coordination_state)
            if not _bounded_execution_enabled(workspace):
                daemon_status.update({
                    "status": "planning_only",
                    "last_event": "awaiting human-approved manifest execution",
                })
                _emit_daemon_state(refresh_scheduler=True)
                heartbeat_from_env(workspace, status="running", message="living research daemon planning-only mode")
                time.sleep(poll_interval_s)
                continue
            result = orch.run_scheduled_once(scheduler=scheduler)
            finalized = finalize_autonomous_results(workspace, plans) if include_autonomous else []
            director_state = ResearchDirector(workspace).update_state()
            try:
                _maybe_write_breakthrough_papers(workspace, finalized, written_hypotheses)
                author_state = write_planned_author_state(workspace)
                write_research_coordination_state(workspace, orch, director_state, author_state)
                paper_start = _maybe_start_ready_author_paper(workspace, started_papers)
                _maybe_compile_completed_author_papers(workspace)
            except Exception as _auth_exc:
                _log(workspace, f"author_warn={_auth_exc}")
                paper_start = None
            if paper_start is not None:
                _log(
                    workspace,
                    f"author_started={paper_start.get('project_id','')} readiness={paper_start.get('readiness','')}",
                )
            if result is not None:
                daemon_status.update({
                    "status": "running",
                    "last_event": f"completed {result.experiment_id} verdict={result.verdict}",
                })
                _emit_daemon_state(refresh_scheduler=True)
                _log(
                    workspace,
                    f"completed={result.experiment_id} verdict={result.verdict} "
                    f"forgetting={result.mean_forgetting:.4f}",
                )
                continue
            daemon_status.update({"status": "running", "last_event": "sleeping"})
            _emit_daemon_state(refresh_scheduler=False)
            time.sleep(poll_interval_s)
        except KeyboardInterrupt:
            _log(workspace, "TAR LIVING RESEARCH DAEMON STOPPED")
            heartbeat_stop.set()
            evidence.stop()
            daemon_status.update({"status": "stopped", "last_event": "keyboard interrupt"})
            _emit_daemon_state(refresh_scheduler=False)
            heartbeat_from_env(workspace, status="failed", message="living research daemon stopped")
            raise
        except Exception as exc:
            _log(workspace, f"daemon_error={exc}")
            daemon_status.update({"status": "error", "last_event": str(exc)})
            _emit_daemon_state(refresh_scheduler=False)
            heartbeat_from_env(workspace, status="failed", message=str(exc))
            time.sleep(max(10.0, poll_interval_s))


def run_queue_maintainer_daemon(
    workspace: Path,
    *,
    poll_interval_s: float = 30.0,
) -> None:
    from tar_author import write_planned_author_state
    from tar_research_director import ResearchDirector
    from tar_scheduler import TARScheduler

    workspace = ensure_workspace_layout(workspace, repo_root=_REPO)
    duplicate_pids = _other_role_pids("--queue-maintainer")
    if duplicate_pids:
        _log(workspace, f"duplicate queue maintainer detected; refusing second start. existing_pids={duplicate_pids}")
        _write_queue_maintainer_state(workspace, status="duplicate_blocked", last_event=f"existing queue maintainer pids={duplicate_pids}")
        return
    _log(workspace, "=" * 70)
    _log(workspace, "TAR QUEUE MAINTAINER START")
    _log(workspace, f"workspace={workspace}")
    _write_queue_maintainer_state(workspace, status="starting", last_event="queue maintainer boot")
    orch = ExperimentOrchestrator(workspace)
    scheduler = TARScheduler(workspace)
    maintainer_status = {"status": "starting", "last_event": "queue maintainer boot"}

    def _scheduler_summary() -> dict[str, object]:
        decision = scheduler.decide(
            pending_specs=orch.get_pending(),
            running_specs=orch.get_running(),
        )
        return {
            "scheduler_saved_at": decision.timestamp,
            "scheduler_rationale": decision.rationale,
            "scheduler_can_start": list(decision.can_start),
            "scheduler_running_ids": list(decision.running_ids),
            "scheduler_hold_count": len(decision.hold_reasons),
        }

    def _emit_queue_maintainer_state() -> None:
        scheduler_summary = _scheduler_summary()
        snapshot = _runtime_state_snapshot(
            workspace,
            orch,
            poll_interval_s=poll_interval_s,
            scheduler_summary=scheduler_summary,
        )
        _write_queue_maintainer_state(
            workspace,
            status=str(maintainer_status.get("status", "running") or "running"),
            last_event=str(maintainer_status.get("last_event", "") or ""),
            **snapshot,
        )

    _emit_queue_maintainer_state()

    while True:
        try:
            maintainer_status.update({"status": "running", "last_event": "reconcile queue"})
            _emit_queue_maintainer_state()
            orch.reconcile_runtime_state()
            director_state = ResearchDirector(workspace).update_state()
            author_state = write_planned_author_state(workspace)
            coordination_state = write_research_coordination_state(workspace, orch, director_state, author_state)
            submitted_ids = _ensure_director_seeded_queue(workspace, orch, coordination_state=coordination_state)
            author_state = write_planned_author_state(workspace)
            write_research_coordination_state(workspace, orch, director_state, author_state)
            last_event = (
                f"submitted {len(submitted_ids)} experiments"
                if submitted_ids else
                "queue synced"
            )
            maintainer_status.update({"status": "running", "last_event": last_event})
            _emit_queue_maintainer_state()
            time.sleep(poll_interval_s)
        except KeyboardInterrupt:
            _log(workspace, "TAR QUEUE MAINTAINER STOPPED")
            maintainer_status.update({"status": "stopped", "last_event": "keyboard interrupt"})
            _emit_queue_maintainer_state()
            raise
        except Exception as exc:
            _log(workspace, f"queue_maintainer_error={exc}")
            maintainer_status.update({"status": "error", "last_event": str(exc)})
            _emit_queue_maintainer_state()
            time.sleep(max(10.0, poll_interval_s))


def run_state_observer(
    workspace: Path,
    *,
    poll_interval_s: float = 30.0,
) -> None:
    from tar_scheduler import TARScheduler

    workspace = ensure_workspace_layout(workspace, repo_root=_REPO)
    duplicate_pids = _other_role_pids("--state-observer")
    if duplicate_pids:
        _log(workspace, f"duplicate state observer detected; refusing second start. existing_pids={duplicate_pids}")
        _write_observer_state(workspace, status="duplicate_blocked", last_event=f"existing state observer pids={duplicate_pids}")
        return
    _log(workspace, "=" * 70)
    _log(workspace, "TAR STATE OBSERVER START")
    _log(workspace, f"workspace={workspace}")
    scheduler = TARScheduler(workspace)

    while True:
        try:
            orch = ExperimentOrchestrator(workspace)
            orch.reconcile_runtime_state()
            decision = scheduler.decide(
                pending_specs=orch.get_pending(),
                running_specs=orch.get_running(),
            )
            snapshot = _runtime_state_snapshot(
                workspace,
                orch,
                poll_interval_s=poll_interval_s,
                scheduler_summary={
                    "scheduler_saved_at": decision.timestamp,
                    "scheduler_rationale": decision.rationale,
                    "scheduler_can_start": list(decision.can_start),
                    "scheduler_running_ids": list(decision.running_ids),
                    "scheduler_hold_count": len(decision.hold_reasons),
                },
            )
            status = "running" if int(snapshot.get("running_count", 0) or 0) > 0 else "idle"
            active_experiment_id = str(snapshot.get("active_experiment_id", "") or "")
            last_event = (
                f"observed active experiment: {active_experiment_id}"
                if active_experiment_id else
                "observed idle queue"
            )
            _write_observer_state(
                workspace,
                status=status,
                last_event=last_event,
                observer_pid=os.getpid(),
                observer_mode="state_observer",
                **snapshot,
            )
            time.sleep(poll_interval_s)
        except KeyboardInterrupt:
            _log(workspace, "TAR STATE OBSERVER STOPPED")
            raise
        except Exception as exc:
            _log(workspace, f"state_observer_error={exc}")
            time.sleep(max(10.0, poll_interval_s))


def main() -> None:
    workspace = ensure_workspace_layout(resolve_workspace(_REPO), repo_root=_REPO)
    args = sys.argv[1:]
    arg_set = set(args)
    include_scaleup = "--autonomous-only" not in arg_set
    include_autonomous = "--scaleup-only" not in arg_set
    daemon = "--daemon" in arg_set
    queue_maintainer = "--queue-maintainer" in arg_set
    state_observer = "--state-observer" in arg_set
    poll_interval_s = 30.0
    if "--poll-interval-s" in args:
        idx = args.index("--poll-interval-s")
        if idx + 1 < len(args):
            poll_interval_s = float(args[idx + 1])

    if queue_maintainer:
        run_queue_maintainer_daemon(
            workspace,
            poll_interval_s=poll_interval_s,
        )
        return

    if state_observer:
        run_state_observer(
            workspace,
            poll_interval_s=poll_interval_s,
        )
        return

    if daemon:
        run_portfolio_daemon(
            workspace,
            include_scaleup=include_scaleup,
            include_autonomous=include_autonomous,
            poll_interval_s=poll_interval_s,
        )
        return

    run_portfolio(
        workspace,
        include_scaleup=include_scaleup,
        include_autonomous=include_autonomous,
    )


if __name__ == "__main__":
    main()
