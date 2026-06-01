"""
TAR Research Dashboard — Living Ecosystem UI
=============================================
Full-stack web dashboard for the TAR autonomous research system.

Shows: hardware gauges, all experiments (live/queued/planned, clickable modal),
frontier problems, TAR Author status, scheduler rationale, phase results,
papers, project registry, and breakthrough alerts.

Run:  python tar_dashboard.py
Open: http://localhost:7860
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from flask import Flask, Response, abort, jsonify, request, send_file
from tar_lab.human_review import (
    answer_human_question,
    load_human_review_state,
    record_review_decision,
)
from tar_lab.validation import build_validation_state, load_validation_state
from tar_storage import ensure_workspace_layout
from tar_lab.result_artifacts import read_advisory_verdict, read_statistics, iter_canonical_comparison_records

try:
    import psutil as _psutil
except Exception:
    _psutil = None

# ── workspace ─────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent
_WS   = ensure_workspace_layout(repo_root=_REPO)
_LOGS = _WS / "tar_state" / "logs"
_COMP = _WS / "tar_state" / "comparisons"
_AR   = _WS / "tar_state" / "autonomous_research"
PORT  = int(os.environ.get("TAR_DASHBOARD_PORT", 7860))


# ── helpers ───────────────────────────────────────────────────────────────────
def _jload(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _dashboard_state_path() -> Path:
    return _WS / "tar_state" / "dashboard_state.json"


def _dashboard_html() -> str:
    html_path = _REPO / "tar_dashboard_live.html"
    try:
        return html_path.read_text(encoding="utf-8")
    except Exception:
        return _HTML


def _write_dashboard_state(status: str = "running", last_event: str = "") -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pid": os.getpid(),
        "port": PORT,
        "workspace": str(_WS),
        "last_event": last_event,
    }
    path = _dashboard_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _start_dashboard_heartbeat() -> None:
    def _loop() -> None:
        while True:
            _write_dashboard_state(status="running", last_event="http server healthy")
            time.sleep(15.0)

    _write_dashboard_state(status="starting", last_event="dashboard boot")
    thread = threading.Thread(target=_loop, daemon=True, name="dashboard-heartbeat")
    thread.start()


def _queue_state() -> dict[str, Any]:
    data = _jload(_WS / "tar_state" / "run_queue_state.json") or {}
    return data if isinstance(data, dict) else {}


def _state_file(name: str) -> Path:
    return _WS / "tar_state" / name


def _json_state(name: str) -> dict[str, Any]:
    data = _jload(_state_file(name)) or {}
    return data if isinstance(data, dict) else {}


def _fresh_state(name: str, max_age_s: float = 180.0) -> dict[str, Any]:
    path = _state_file(name)
    data = _json_state(name)
    if not data or not path.exists() or _age_s(path) > max_age_s:
        return {}
    return data


def _queue_step_status_from_experiment(exp_id: str) -> str | None:
    def _status_from_record(rec: dict[str, Any]) -> str | None:
        if str(rec.get("id", "") or "") != exp_id:
            return None
        stage = str(rec.get("stage", "") or rec.get("status", "") or "")
        result_path = str(rec.get("result_path", "") or "")
        if result_path:
            path = Path(result_path)
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
                verdict = str(raw.get("verdict", "") or "").upper()
                status_hint = str(raw.get("status", "") or "").upper()
                if verdict == "ERROR" or status_hint == "ERROR":
                    return "failed"
                if raw:
                    return "complete"
        if stage == "running":
            return "running"
        if stage in {"stalled", "queued", "planned", "pending", "analyzing", "writing_paper"}:
            return "pending"
        if stage == "complete":
            return "complete"
        if stage == "failed":
            return "failed"
        return None

    for state_name in ("experiment_queue.json", "experiment_archive.json"):
        queue = _jload(_WS / "tar_state" / state_name) or {}
        experiments = queue.get("experiments", []) if isinstance(queue, dict) else []
        for rec in experiments if isinstance(experiments, list) else []:
            status = _status_from_record(rec) if isinstance(rec, dict) else None
            if status:
                return status

    direct_result = _WS / "tar_state" / "experiments" / exp_id / "result.json"
    if direct_result.exists():
        try:
            raw = json.loads(direct_result.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        verdict = str(raw.get("verdict", "") or "").upper()
        status_hint = str(raw.get("status", "") or "").upper()
        return "failed" if verdict == "ERROR" or status_hint == "ERROR" else "complete"
    return None


def _tail(path: Path, n: int = 120) -> list[str]:
    try:
        buf: deque[str] = deque(maxlen=n)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                buf.append(line.rstrip())
        return list(buf)
    except Exception:
        return []


def _age_s(path: Path) -> float:
    try:
        return datetime.now().timestamp() - path.stat().st_mtime
    except Exception:
        return 999_999.0


def _parse_iso_dt(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_age_text(seconds: float | int | None) -> str:
    try:
        total = max(0, int(float(seconds or 0)))
    except Exception:
        return ""
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _pid_started_for_spec(pid: int, started_at: str, tolerance_s: float = 300.0) -> bool | None:
    if pid <= 0 or _psutil is None:
        return None
    started_dt = _parse_iso_dt(started_at)
    if started_dt is None:
        return None
    try:
        proc = _psutil.Process(pid)
        created_at = datetime.fromtimestamp(proc.create_time(), tz=timezone.utc)
    except Exception:
        return False
    return created_at.timestamp() + tolerance_s >= started_dt.timestamp()


def _resolve_logs() -> Path:
    return _LOGS if _LOGS.exists() else (_REPO / "tar_state" / "logs")


def _pick_active_log() -> tuple[Path | None, str]:
    log_dir = _resolve_logs()
    candidates = [
        (_WS / "tar_state" / "living_research.log", "Living Research"),
        (log_dir / "phase16.log",    "Phase 16 — CIFAR-100"),
        (log_dir / "phase17.log",    "Phase 17 — TinyImageNet"),
        (log_dir / "tar_author.log", "TAR-Author"),
        (log_dir / "resume_ar.log",  "Autonomous Research Resume"),
        (_WS / "tar_state" / "autonomous_research_resume.log", "AR Resume (detail)"),
        (_WS / "tar_state" / "autonomous_research.log",        "Autonomous Research"),
    ]
    best_path, best_label, best_age = None, "—", 999_999.0
    for p, label in candidates:
        if p.exists():
            age = _age_s(p)
            if age < best_age:
                best_path, best_label, best_age = p, label, age
    return best_path, best_label


def _parse_training_progress(lines: list[str]) -> dict:
    seed = method = None
    tasks_done = 0
    accs: list[str] = []
    seed_markers: list[int] = []
    method_summaries: list[dict[str, Any]] = []
    for line in lines:
        m = re.search(r"--- seed=(\d+) ---", line)
        if m:
            seed = int(m.group(1)); tasks_done = 0; accs = []
            seed_markers.append(seed)
        m = re.search(r"running (\w+)\.\.\.", line)
        if m:
            method = m.group(1)
        m = re.search(r"after task (\d+):", line)
        if m:
            tasks_done = int(m.group(1)) + 1
        m = re.search(r"acc on seen tasks = \[(.+?)\]", line)
        if m:
            accs = [v.strip().strip("'") for v in m.group(1).split(",")]
        m = re.search(r"^\s*(\w+)\s+forgetting=([0-9.]+)\s+acc=([0-9.]+)", line)
        if m and seed is not None:
            method_summaries.append({
                "seed": seed,
                "method": m.group(1),
                "forgetting": float(m.group(2)),
                "accuracy": float(m.group(3)),
            })
    return {"current_seed": seed, "current_method": method,
            "tasks_done": tasks_done, "latest_accs": accs,
            "seed_markers": seed_markers, "method_summaries": method_summaries}


def _latest_matching_log(pattern: str) -> Path | None:
    log_dir = _resolve_logs()
    matches = sorted(
        log_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    return matches[0] if matches else None


def _suite_checkpoint_progress(experiment_id: str, seeds: list[int]) -> dict[str, Any]:
    if not experiment_id:
        return {}
    try:
        from tar_suite_checkpoint import checkpoint_path, load_suite_state
    except Exception:
        return {}

    state = load_suite_state(checkpoint_path(_WS, experiment_id))
    if not state:
        return {}

    completed = [int(seed) for seed in state.get("completed_seeds", []) if str(seed).strip()]
    next_seed = next((seed for seed in seeds if seed not in completed), None)
    forgetting = list((state.get("forgetting", {}) or {}).get("tcl", []) or [])
    return {
        "seeds_done": len(completed),
        "seeds_total": len(seeds),
        "tasks_done": 10 if completed else 0,
        "latest_accs": [],
        "forgetting_so_far": forgetting[:],
        "current_seed": next_seed,
        "current_method": "tcl" if next_seed is not None else None,
        "last_checkpoint_at": str(state.get("last_updated", "") or ""),
        "checkpoint_source": str(state.get("source", "") or ""),
    }


def _queue_steps(log_dir: Path) -> list[dict]:
    steps = [
        {"n": 1, "label": "Living Research Daemon",       "log": "living_research.log"},
        {"n": 2, "label": "Phase 16 — Split-CIFAR-100",   "log": "phase16.log"},
        {"n": 3, "label": "Phase 17 — TinyImageNet",      "log": "phase17.log"},
        {"n": 4, "label": "TAR-Author — paper work",      "log": "tar_author.log"},
        {"n": 5, "label": "Queue Maintainer — autonomous sync", "log": "resume_ar.log"},
    ]
    q3_lines = _tail(log_dir / "queue3_run.log", 200) if (log_dir / "queue3_run.log").exists() else []
    for step in steps:
        log_path = log_dir / step["log"]
        n = step["n"]
        ok_marker = (
            "[OK] Living research" if n == 1 else
            "[OK] TAR-Author" if n == 4 else
            "[OK] Autonomous" if n == 5 else
            f"[OK] Phase {n - 1}"
        )
        err_marker = (
            "[ERROR] Living research" if n == 1 else
            f"[ERROR] Phase {n - 1}" if n in {2, 3} else
            ""
        )
        done_ok    = any(ok_marker[:15] in l for l in q3_lines)
        done_err   = bool(err_marker) and any(err_marker in l for l in q3_lines)
        if not log_path.exists():
            step["status"] = "pending"
        elif done_ok:
            step["status"] = "complete"
        elif done_err:
            step["status"] = "failed"
        elif _age_s(log_path) < 300:
            step["status"] = "running"
        else:
            step["status"] = "stale"
        step["size_kb"] = int(log_path.stat().st_size / 1024) if log_path.exists() else 0

    queue_state = _queue_state()
    living_state = _fresh_state("living_research_daemon.json")
    maint_state = _fresh_state("queue_maintainer_state.json")
    author_state = _fresh_state("author_state.json")

    if living_state:
        status = str(living_state.get("status", "") or "")
        steps[0]["status"] = (
            "running" if status in {"running", "starting", "waiting"}
            else "failed" if status in {"error", "failed"}
            else "stale"
        )
    elif queue_state.get("queue_name") == "queue2":
        status = str(queue_state.get("status", "") or "")
        if status in {"running", "starting", "waiting"}:
            steps[0]["status"] = "running"
        elif status == "failed":
            steps[0]["status"] = "failed"

    phase16_status = _queue_step_status_from_experiment("phase16_scale_up")
    if phase16_status:
        steps[1]["status"] = phase16_status

    phase17_status = _queue_step_status_from_experiment("phase17_tinyimagenet")
    if phase17_status:
        steps[2]["status"] = phase17_status

    if author_state:
        current = author_state.get("current_paper", {}) if isinstance(author_state, dict) else {}
        author_status = str(current.get("status", "") or "")
        compile_status = str(current.get("compile_status", "") or "")
        if compile_status in {"compiling", "pdf_pending"} or author_status in {"writing", "revising", "starting"}:
            steps[3]["status"] = "running"
        elif author_status in {"revision_failed"} or compile_status in {"failed", "revision_failed"}:
            steps[3]["status"] = "failed"
        elif author_status in {"complete", "done"} and compile_status in {"compiled", "draft_compiled", "complete"}:
            steps[3]["status"] = "complete"
        elif current.get("project_id"):
            steps[3]["status"] = "pending"

    if maint_state:
        status = str(maint_state.get("status", "") or "")
        steps[4]["status"] = (
            "running" if status in {"running", "starting", "waiting"}
            else "failed" if status in {"error", "failed"}
            else "stale"
        )
    elif queue_state.get("queue_name") == "queue1":
        status = str(queue_state.get("status", "") or "")
        if status in {"running", "starting", "waiting"}:
            steps[4]["status"] = "running"
        elif status == "failed":
            steps[4]["status"] = "failed"

    return steps


def _serve_path_for(abs_path: str) -> str:
    if not abs_path:
        return ""
    p = Path(abs_path)
    for root in [_REPO / "paper", _WS / "paper"]:
        try:
            return str(p.relative_to(root)).replace("\\", "/")
        except ValueError:
            pass
    return ""


def _director_state(force_refresh: bool = False) -> dict:
    path = _WS / "tar_state" / "research_director_state.json"
    data = _jload(path) or {}
    if data and not force_refresh and _age_s(path) <= 60:
        return data if isinstance(data, dict) else {}
    try:
        from tar_research_director import ResearchDirector

        data = ResearchDirector(_WS).update_state()
    except Exception:
        data = data if isinstance(data, dict) else {}
    return data if isinstance(data, dict) else {}


def _human_review_payload() -> dict[str, Any]:
    state = load_human_review_state(_WS)
    try:
        from tar_validation_mode import load_state as _load_vs
        vs = _load_vs(_WS) or {}
        state["stabilisation_active"] = bool(vs.get("active"))
    except Exception:
        state["stabilisation_active"] = False
    return state


def _validation_payload() -> dict[str, Any]:
    state = load_validation_state(_WS)
    if not state:
        try:
            state = build_validation_state(_WS, persist=False)
        except Exception as exc:
            return {
                "summary": {
                    "trusted_publication_allowed": 0,
                    "limited_scope": 0,
                    "missing_env": 0,
                    "quarantined": 0,
                },
                "results": [],
                "error": str(exc),
            }
    return state if isinstance(state, dict) else {}


def _author_state_payload(force_refresh: bool = False) -> dict[str, Any]:
    state_path = _WS / "tar_state" / "author_state.json"
    state = _jload(state_path) or {}
    needs_refresh = force_refresh or not state or _age_s(state_path) > 60
    if needs_refresh:
        try:
            _director_state(force_refresh=force_refresh)
            from tar_author import write_planned_author_state

            state = write_planned_author_state(_WS)
        except Exception:
            state = state or {}
    return state if isinstance(state, dict) else {}


def _run_paper_revision_async(project_id: str, reason: str) -> None:
    try:
        from tar_author import run_paper_revision, stabilisation_authoring_override
        with stabilisation_authoring_override(_WS, reason):
            run_paper_revision(_WS, project_id, reason, request_first=False)
    except Exception as exc:
        try:
            from tar_author import _load_paper_plan, _write_paper_plan

            plan_path = _WS / "paper" / project_id / "paper_plan.json"
            plan = _load_paper_plan(plan_path)
            if plan:
                plan["status"] = "revision_failed"
                plan["compile_status"] = "revision_failed"
                plan["last_revision_error"] = str(exc)
                plan["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_paper_plan(plan_path, plan)
        except Exception:
            pass


def _result_evidence_payload(path: Path) -> dict[str, Any] | None:
    data = _jload(path) or {}
    if not isinstance(data, dict):
        return None

    # RAIL 5: prefer the new statistics / advisory_verdict blocks; fall back
    # to legacy flat fields so old result files still work.
    stats = read_statistics(data)
    advisory = read_advisory_verdict(data)
    verdict = str(advisory.get("label", "") or data.get("status", "") or "").upper()
    notes   = str(advisory.get("notes", "") or data.get("summary", "") or "").strip()

    mean_delta = stats.get("mean_delta", data.get("mean_delta"))
    p_val      = stats.get("p_val",      data.get("p_val"))
    cohens_d   = stats.get("cohens_d",   data.get("cohens_d"))

    label = path.parent.name if path.name == "result.json" else path.stem

    if mean_delta is None:
        pairwise = stats.get("pairwise") or data.get("pairwise", {})
        if isinstance(pairwise, dict):
            ewc = pairwise.get("ewc", {}) if isinstance(pairwise.get("ewc", {}), dict) else {}
            mean_delta = ewc.get("mean_delta")
            p_val = p_val if p_val is not None else ewc.get("p_val") or data.get("p_value_vs_strong_baseline")
            cohens_d = cohens_d if cohens_d is not None else ewc.get("cohens_d") or data.get("effect_size_vs_strong_baseline")

    evidence_strength = "weak"
    try:
        if verdict == "BREAKTHROUGH" or (
            mean_delta is not None and p_val is not None and float(mean_delta) < 0 and float(p_val) < 0.05
        ):
            evidence_strength = "strong"
        elif verdict == "DIRECTIONAL" or (mean_delta is not None and float(mean_delta) < 0):
            evidence_strength = "moderate"
    except Exception:
        evidence_strength = "moderate" if verdict in {"BREAKTHROUGH", "DIRECTIONAL"} else "weak"

    return {
        "label": label.replace("_", " ").replace("-", " ").title(),
        "result_path": str(path),
        "verdict": verdict,                          # advisory label
        "verdict_is_advisory": advisory.get("label_is_advisory_only", False),
        "mean_delta": mean_delta,
        "p_val": p_val,
        "cohens_d": cohens_d,
        "notes": notes[:220],
        "evidence_strength": evidence_strength,
    }


def _frontier_with_directives() -> list[dict[str, Any]]:
    data = _jload(_WS / "tar_state" / "frontier_problems.json") or {}
    problems = data.get("problems", []) if isinstance(data, dict) else []
    director = _director_state()
    directives = {
        str(rec.get("problem_id", "") or ""): rec
        for rec in director.get("frontier_directives", [])
        if rec.get("problem_id")
    }
    evidence_directives = {
        str(rec.get("frontier_problem_id", "") or ""): rec
        for rec in director.get("evidence_directives", [])
        if rec.get("frontier_problem_id")
    }
    enriched: list[dict[str, Any]] = []
    for problem in problems:
        entry = dict(problem)
        frontier_id = str(problem.get("id", "") or "")
        directive = directives.get(frontier_id, {})
        evidence = evidence_directives.get(frontier_id, {})
        linked_result_paths = [
            str(path)
            for path in evidence.get("linked_result_paths", [])
            if str(path).strip()
        ]
        result_evidence: list[dict[str, Any]] = []
        for raw_path in linked_result_paths[:8]:
            payload = _result_evidence_payload(Path(raw_path))
            if payload:
                result_evidence.append(payload)
        breakthrough_evidence = [
            rec for rec in result_evidence
            if rec.get("evidence_strength") in {"strong", "moderate"}
        ]
        linked_experiment_ids = [
            str(exp_id)
            for exp_id in directive.get("linked_experiment_ids", [])
            if str(exp_id).strip()
        ]
        linked_paper_ids = [
            str(paper_id)
            for paper_id in directive.get("linked_paper_ids", [])
            if str(paper_id).strip()
        ]
        truth_status = str(directive.get("truth_status", "") or entry.get("truth_status", "") or "weak")
        waiting_on = [
            str(exp_id)
            for exp_id in directive.get("waiting_on_experiment_ids", [])
            if str(exp_id).strip()
        ]
        if waiting_on:
            dynamic_status = "active"
        elif truth_status == "validated":
            dynamic_status = "publishing"
        elif linked_experiment_ids or result_evidence:
            dynamic_status = "active"
        else:
            dynamic_status = "exploring"
        entry.update({
            "status": dynamic_status,
            "priority_score": directive.get("priority_score"),
            "truth_status": truth_status,
            "evidence_strength": directive.get("evidence_strength"),
            "next_action": directive.get("next_action"),
            "why_now": directive.get("why_now"),
            "waiting_on_experiment_ids": waiting_on,
            "linked_experiment_ids": linked_experiment_ids,
            "linked_paper_ids": linked_paper_ids,
            "experiments_linked": linked_experiment_ids,
            "papers_linked": linked_paper_ids,
            "breakthroughs_found": len([rec for rec in result_evidence if rec.get("evidence_strength") == "strong"]),
            "evidence_notes": directive.get("evidence_notes", []),
            "verification_standard": evidence.get("verification_standard", ""),
            "linked_result_paths": linked_result_paths,
            "result_evidence": result_evidence,
            "breakthrough_evidence": breakthrough_evidence,
            "scheduler_rank": directive.get("scheduler_rank"),
            "author_rank": directive.get("author_rank"),
        })
        enriched.append(entry)
    return enriched


def _queue_experiments() -> list[dict[str, Any]]:
    data = _jload(_WS / "tar_state" / "experiment_queue.json") or {}
    if not isinstance(data, dict):
        return []
    experiments = data.get("experiments", [])
    return experiments if isinstance(experiments, list) else []


def _status_queue_steps(runtime_experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def normalize_status(rec: dict[str, Any]) -> str:
        stage = str(rec.get("stage", "") or rec.get("status", "") or "pending")
        if stage in {"queued", "planned"}:
            return "pending"
        if stage in {"running", "failed", "complete", "stalled", "pending"}:
            return stage
        return "pending"

    def progress_suffix(rec: dict[str, Any]) -> str:
        progress = rec.get("progress", {}) if isinstance(rec.get("progress", {}), dict) else {}
        seeds_total = int(progress.get("seeds_total", 0) or 0)
        seeds_done = int(progress.get("seeds_done", 0) or 0)
        if seeds_total > 0:
            return f" ({seeds_done}/{seeds_total} seeds)"
        tasks_done = int(progress.get("tasks_done", 0) or 0)
        if tasks_done > 0:
            return f" (task {tasks_done})"
        return ""

    visible = [rec for rec in runtime_experiments if normalize_status(rec) != "complete"]
    if not visible:
        visible = runtime_experiments[:]
    visible.sort(
        key=lambda rec: (
            {"running": 0, "stalled": 1, "pending": 2, "failed": 3, "complete": 4}.get(normalize_status(rec), 5),
            float(rec.get("priority", 999.0) or 999.0),
            str(rec.get("name", "") or rec.get("id", "")),
        )
    )

    steps: list[dict[str, Any]] = []
    for idx, rec in enumerate(visible[:8], start=1):
        result_path = Path(str(rec.get("result_path", "") or "")) if rec.get("result_path") else None
        steps.append({
            "n": idx,
            "label": f"{str(rec.get('name', '') or rec.get('id', 'experiment'))}{progress_suffix(rec)}",
            "status": normalize_status(rec),
            "size_kb": int(result_path.stat().st_size / 1024) if result_path and result_path.exists() else 0,
        })
    return steps


_PHASE_LIVE_META: dict[int, dict[str, Any]] = {
    16: {
        "id": "phase16_scale_up",
        "name": "Phase 16 - CIFAR-100 Scale-up",
        "dataset": "split_cifar100",
        "log": "phase16.log",
        "result": "phase16_scale_up.json",
        "step": 2,
        "priority": 16,
        "seeds": [42, 0, 1],
        "runtime_h": 24.0,
        "hardware_budget": {"vram_gb": 3.2, "cpu_cores": 4},
        "frontier_problem_id": "fp-scale-up",
        "feeds_paper": "Main TCL paper - scale-up section",
        "why": (
            "This run tests whether the thermodynamic TCL mechanism survives the jump from "
            "Split-CIFAR-10 to a harder 10-task Split-CIFAR-100 setting."
        ),
        "hypothesis": (
            "If regime-aware anchoring is genuinely general rather than CIFAR-10-specific, "
            "TCL should still beat at least one baseline on mean forgetting after scale-up."
        ),
        "methodology": (
            "ResNet-18, 40 epochs per task, 3 seeds, methods = TCL / EWC / SGD baseline, "
            "10 tasks x 10 classes."
        ),
        "queued_projection": (
            "Queued behind the current hardware slot. Once active, TAR will recompute whether "
            "the CIFAR-10 signal generalizes to a harder 100-class benchmark."
        ),
    },
    17: {
        "id": "phase17_tinyimagenet",
        "name": "Phase 17 - TinyImageNet Scale-up",
        "dataset": "split_tinyimagenet",
        "log": "phase17.log",
        "result": "phase17_tinyimagenet.json",
        "step": 3,
        "priority": 17,
        "seeds": [42, 0, 1],
        "runtime_h": 36.0,
        "hardware_budget": {"vram_gb": 3.3, "cpu_cores": 4},
        "frontier_problem_id": "fp-scale-up",
        "feeds_paper": "Main TCL paper - scale-up section",
        "why": (
            "This run pushes TCL to a materially larger visual benchmark so TAR can test "
            "whether the mechanism still holds when tasks, images, and class diversity all grow."
        ),
        "hypothesis": (
            "If the thermal regime signal scales cleanly, TCL should remain directionally better "
            "than SGD and ideally competitive with EWC even on Split-TinyImageNet."
        ),
        "methodology": (
            "Adapted ResNet-18 for 64x64 inputs, 40 epochs per task, 3 seeds, methods = TCL / "
            "EWC / SGD baseline, 10 tasks x 20 classes."
        ),
        "queued_projection": (
            "Waiting for Phase 16 to release the GPU slot. This is the highest-cost scale-up run "
            "and will provide the strongest evidence about frontier robustness."
        ),
    },
}


def _phase_progress(
    log_path: Path,
    seeds_total: int,
    *,
    experiment_id: str = "",
    seeds: list[int] | None = None,
    running: bool = False,
) -> dict:
    source_path: Path | None = None
    if log_path.exists():
        parsed = _parse_training_progress(_tail(log_path, 600))
        source_path = log_path
    else:
        parsed = {}
        if experiment_id:
            trace_paths = [
                _latest_matching_log("watchdog-living-research-*.log"),
                _WS / "tar_state" / "living_research.log",
                _WS / "tar_state" / "experiment_orchestrator.log",
            ]
            exp_stub = {"id": experiment_id, "name": ""}
            for trace_path in trace_paths:
                if trace_path is None or not trace_path.exists():
                    continue
                trace_lines = _filtered_experiment_trace(trace_path, exp_stub, limit=320)
                candidate = _parse_training_progress(trace_lines)
                if (
                    candidate.get("tasks_done", 0)
                    or candidate.get("seed_markers")
                    or candidate.get("method_summaries")
                ):
                    parsed = candidate
                    source_path = trace_path
                    parsed["log_fallback_source"] = str(trace_path)
                    break
        if not parsed:
            parsed = {
                "current_seed": None,
                "current_method": None,
                "tasks_done": 0,
                "latest_accs": [],
                "seed_markers": [],
                "method_summaries": [],
            }
    seed_markers = parsed.get("seed_markers", [])
    current_seed = parsed.get("current_seed")
    seeds_done = len(seed_markers)
    if current_seed is not None and seeds_done > 0:
        seeds_done -= 1
    checkpoint_progress = _suite_checkpoint_progress(experiment_id, list(seeds or []))
    parsed.update({
        "seeds_done": max(0, min(seeds_done, seeds_total)),
        "seeds_total": seeds_total,
    })
    if checkpoint_progress:
        parsed["seeds_done"] = max(
            int(parsed.get("seeds_done", 0) or 0),
            int(checkpoint_progress.get("seeds_done", 0) or 0),
        )
        if not parsed.get("forgetting_so_far"):
            parsed["forgetting_so_far"] = list(checkpoint_progress.get("forgetting_so_far", []) or [])
        if not parsed.get("current_seed") and checkpoint_progress.get("current_seed") is not None:
            parsed["current_seed"] = checkpoint_progress.get("current_seed")
            parsed["current_method"] = checkpoint_progress.get("current_method")
        if not parsed.get("last_checkpoint_at") and checkpoint_progress.get("last_checkpoint_at"):
            parsed["last_checkpoint_at"] = checkpoint_progress.get("last_checkpoint_at")
        if not parsed.get("checkpoint_source") and checkpoint_progress.get("checkpoint_source"):
            parsed["checkpoint_source"] = checkpoint_progress.get("checkpoint_source")
    if running and not parsed.get("current_seed") and int(parsed.get("seeds_done", 0) or 0) <= 0:
        first_seed = (seeds or [None])[0]
        parsed["current_seed"] = first_seed
        parsed["current_method"] = parsed.get("current_method") or "tcl"
        parsed["initializing"] = True
    if source_path and source_path.exists():
        parsed["last_visible_log_source"] = str(source_path)
        parsed["last_visible_log_at"] = datetime.fromtimestamp(
            source_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        parsed["last_visible_log_age_s"] = _age_s(source_path)
    return parsed


def _merge_phase_progress(parsed: dict, runtime: dict | None) -> dict:
    merged = dict(parsed or {})
    runtime = runtime if isinstance(runtime, dict) else {}
    if not runtime:
        return merged

    parsed_method = str(merged.get("current_method", "") or "")
    runtime_method = str(runtime.get("current_method", "") or runtime.get("method", "") or "")
    prefer_parsed_method = bool(merged.get("log_fallback_source"))

    runtime_tasks = int(runtime.get("tasks_done", 0) or 0)
    parsed_tasks = int(merged.get("tasks_done", 0) or 0)
    if not prefer_parsed_method and runtime_tasks >= parsed_tasks:
        merged["tasks_done"] = runtime_tasks
        if runtime.get("latest_accs"):
            merged["latest_accs"] = list(runtime.get("latest_accs", []) or [])

    runtime_seeds_done = int(runtime.get("seeds_done", 0) or 0)
    parsed_seeds_done = int(merged.get("seeds_done", 0) or 0)
    if runtime_seeds_done >= parsed_seeds_done:
        merged["seeds_done"] = runtime_seeds_done

    runtime_seeds_total = int(runtime.get("seeds_total", 0) or 0)
    if runtime_seeds_total:
        merged["seeds_total"] = runtime_seeds_total

    if runtime.get("current_seed") is not None:
        merged["current_seed"] = runtime.get("current_seed")
    elif runtime.get("seed") is not None and merged.get("current_seed") is None:
        merged["current_seed"] = runtime.get("seed")

    if not prefer_parsed_method and runtime.get("current_method"):
        merged["current_method"] = runtime.get("current_method")
    elif not prefer_parsed_method and runtime.get("method"):
        merged["current_method"] = runtime.get("method")

    if runtime.get("method_summaries"):
        merged["method_summaries"] = list(runtime.get("method_summaries", []) or [])
    if runtime.get("seed_markers"):
        merged["seed_markers"] = list(runtime.get("seed_markers", []) or [])

    for key in ("last_checkpoint_at", "tcl_forgetting_so_far", "forgetting_so_far"):
        if runtime.get(key):
            merged[key] = runtime.get(key)

    if runtime_tasks > 0 or runtime.get("latest_accs"):
        merged["initializing"] = False
    return merged


def _annotate_live_progress(
    exp_id: str,
    pid: int,
    status: str,
    progress: dict | None,
) -> dict[str, Any]:
    merged = dict(progress or {})
    hardware = _fresh_state("hardware_state.json", max_age_s=120.0)
    gpu = hardware.get("gpu", {}) if isinstance(hardware, dict) else {}
    hw_processes = hardware.get("processes", []) if isinstance(hardware.get("processes", []), list) else []
    process_entry = next(
        (
            proc for proc in hw_processes
            if int(proc.get("pid", 0) or 0) == int(pid or 0)
        ),
        {},
    )
    registry = _json_state("process_registry.json")
    registry_meta = registry.get(str(pid), {}) if isinstance(registry, dict) else {}
    process_attached = (
        bool(process_entry)
        and str(process_entry.get("experiment_id", "") or "") == exp_id
    ) or (
        bool(registry_meta)
        and str(registry_meta.get("experiment_id", "") or "") == exp_id
    )
    gpu_util = float(gpu.get("utilization_pct", 0.0) or 0.0)
    gpu_vram_used = float(gpu.get("vram_used_gb", 0.0) or 0.0)
    live_compute_active = bool(
        str(status or "") == "running"
        and int(pid or 0) > 0
        and process_attached
        and (gpu_util >= 20.0 or gpu_vram_used > 0.25)
    )
    merged["live_process_attached"] = bool(process_attached)
    merged["live_compute_active"] = live_compute_active
    merged["hardware_timestamp"] = str(hardware.get("timestamp", "") or "")
    merged["gpu_utilization_pct"] = gpu_util
    merged["gpu_vram_used_gb"] = gpu_vram_used
    merged["gpu_vram_total_gb"] = float(gpu.get("vram_total_gb", 0.0) or 0.0)

    source_text = str(
        merged.get("last_visible_log_source", "")
        or merged.get("log_fallback_source", "")
        or ""
    ).strip()
    if source_text:
        source_path = Path(source_text)
        if source_path.exists():
            last_visible_dt = datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc)
            merged["last_visible_log_at"] = last_visible_dt.isoformat()
            age_s = _age_s(source_path)
            merged["last_visible_log_age_s"] = age_s
            merged["last_visible_log_age_text"] = _fmt_age_text(age_s)

    if live_compute_active:
        heartbeat_bits = [
            f"GPU {gpu_util:.0f}%",
            f"VRAM {gpu_vram_used:.2f}/{float(gpu.get('vram_total_gb', 0.0) or 0.0):.1f} GB",
        ]
        if merged.get("last_visible_log_age_text"):
            merged["live_compute_note"] = (
                "Compute heartbeat is still active ("
                + ", ".join(heartbeat_bits)
                + f"). No new task-boundary log lines for {merged['last_visible_log_age_text']}."
            )
        else:
            merged["live_compute_note"] = (
                "Compute heartbeat is still active ("
                + ", ".join(heartbeat_bits)
                + ")."
            )
    elif str(status or "") == "running" and int(pid or 0) > 0:
        merged["live_compute_note"] = "Run is marked live, but no strong compute heartbeat is visible in the latest hardware snapshot."
    return merged


def _phase_projection(meta: dict[str, Any], status: str, progress: dict, result: dict) -> str:
    if status == "running":
        parts: list[str] = []
        seeds_done = progress.get("seeds_done", 0)
        current_seed = progress.get("current_seed")
        current_method = progress.get("current_method") or "current method"
        tasks_done = progress.get("tasks_done", 0)
        if progress.get("initializing"):
            parts.append(
                "The run is live, but no completed task checkpoint has been written yet. "
                "TAR is likely loading the dataset or warming up the first seed."
            )
        elif current_seed is not None:
            parts.append(
                f"Seed {seeds_done + 1}/{progress.get('seeds_total', len(meta['seeds']))} is active."
            )
        if tasks_done:
            parts.append(f"Latest visible progress is task {tasks_done}/10 for {current_method}.")
        elif current_seed is not None:
            parts.append(f"Waiting for the first task checkpoint on {current_method}.")
        if progress.get("live_compute_note"):
            parts.append(str(progress.get("live_compute_note")))
        summaries = progress.get("method_summaries", [])
        if summaries:
            per_method: dict[str, list[float]] = {}
            for item in summaries:
                per_method.setdefault(item["method"], []).append(item["forgetting"])
            means = ", ".join(
                f"{method} {sum(vals)/len(vals):.4f}"
                for method, vals in sorted(per_method.items())
                if vals
            )
            if means:
                parts.append(f"Visible seed-level mean forgetting so far: {means}.")
        parts.append("Final verdict will update automatically when the run writes its new result file.")
        return " ".join(parts)
    if status == "stalled":
        parts: list[str] = ["This run appears to have stalled after partial visible progress."]
        seeds_done = progress.get("seeds_done", 0)
        tasks_done = progress.get("tasks_done", 0)
        current_seed = progress.get("current_seed")
        current_method = progress.get("current_method") or "current method"
        if current_seed is not None:
            parts.append(
                f"Last visible work reached seed {seeds_done + 1}/{progress.get('seeds_total', len(meta['seeds']))}."
            )
        if tasks_done:
            parts.append(f"The latest logged point was task {tasks_done}/10 for {current_method}.")
        elif progress.get("last_checkpoint_at"):
            parts.append(f"Last checkpoint update was {progress.get('last_checkpoint_at')}.")
        parts.append("The dashboard should treat this as interrupted until the run is resumed or restarted.")
        return " ".join(parts)
    if status == "failed":
        return result.get("error") or result.get("verdict") or "Run failed."
    if status == "complete":
        return result.get("verdict") or "Run complete."
    return meta["queued_projection"]


def _synthetic_phase_experiments() -> list[dict[str, Any]]:
    log_dir = _resolve_logs()
    queue_steps = {step["n"]: step for step in _queue_steps(log_dir)}
    queued_by_id = {
        rec.get("id", ""): rec
        for rec in _queue_experiments()
        if rec.get("id")
    }
    experiments: list[dict[str, Any]] = []
    for phase_num, meta in _PHASE_LIVE_META.items():
        queued = queued_by_id.get(meta["id"])
        if queued:
            merged = dict(queued)
            log_path = log_dir / meta["log"]
            step = queue_steps.get(meta["step"], {})
            step_status = step.get("status", "pending")
            runtime_status = merged.get("status", "pending")
            runtime_stage = merged.get("stage", runtime_status)
            progress = _phase_progress(
                log_path,
                len(meta["seeds"]),
                experiment_id=meta["id"],
                seeds=list(meta["seeds"]),
                running=runtime_stage == "running",
            )
            progress = _merge_phase_progress(progress, merged.get("progress"))
            progress = _annotate_live_progress(meta["id"], int(merged.get("pid", 0) or 0), runtime_stage, progress)
            prior_status = queue_steps.get(meta["step"] - 1, {}).get("status", "")
            has_partial_progress = (
                progress.get("seeds_done", 0) > 0
                or progress.get("tasks_done", 0) > 0
            )

            if runtime_stage not in {"complete", "failed", "running", "stalled", "queued", "planned"}:
                if log_path.exists() and step_status in {"running", "stale"}:
                    if step_status == "running":
                        runtime_status, runtime_stage = "running", "running"
                    elif has_partial_progress:
                        runtime_status, runtime_stage = "stalled", "stalled"
                elif step_status == "pending":
                    runtime_status = "pending"
                    runtime_stage = "queued" if prior_status in {"running", "stale", "pending"} else "planned"

            merged["status"] = runtime_status
            merged["stage"] = runtime_stage
            merged["progress"] = progress
            merged["_phase_num"] = phase_num
            merged.setdefault("priority", meta["priority"])
            merged.setdefault("dataset", meta["dataset"])
            merged.setdefault("estimated_runtime_h", meta["runtime_h"])
            merged.setdefault("hardware_budget", meta["hardware_budget"])
            merged.setdefault("frontier_problem_id", meta["frontier_problem_id"])
            merged.setdefault("author_paper_id", meta["feeds_paper"])
            merged["context"] = {
                "why": meta["why"],
                "hypothesis": meta["hypothesis"],
                "projected_outcome": _phase_projection(meta, runtime_stage, progress, {}),
                "frontier_problem": meta["frontier_problem_id"],
                "feeds_paper": meta["feeds_paper"],
                "methodology_note": meta["methodology"],
            }
            experiments.append(merged)
            continue

        log_path = log_dir / meta["log"]
        result_path = _COMP / meta["result"]
        result = _jload(result_path) if result_path.exists() else {}
        result = result if isinstance(result, dict) else {}
        step = queue_steps.get(meta["step"], {})
        step_status = step.get("status", "pending")
        log_newer = log_path.exists() and (
            not result_path.exists() or log_path.stat().st_mtime > result_path.stat().st_mtime
        )

        if log_newer and step_status in {"running", "stale"}:
            status, stage = "running", "running"
        elif step_status == "running":
            status, stage = "running", "running"
        elif step_status == "pending":
            prior_status = queue_steps.get(meta["step"] - 1, {}).get("status", "")
            status = "pending"
            stage = "queued" if prior_status in {"running", "stale", "pending"} else "planned"
        elif step_status == "complete":
            status, stage = "complete", "complete"
        elif step_status == "failed":
            status, stage = "failed", "failed"
        elif result.get("status") == "ERROR" or "ERROR" in str(result.get("verdict", "")):
            status, stage = "failed", "failed"
        elif result_path.exists():
            status, stage = "complete", "complete"
        else:
            status, stage = "pending", "planned"

        progress = _phase_progress(
            log_path,
            len(meta["seeds"]),
            experiment_id=meta["id"],
            seeds=list(meta["seeds"]),
            running=status == "running",
        )
        has_partial_progress = (
            progress.get("seeds_done", 0) > 0
            or progress.get("tasks_done", 0) > 0
        )
        if status == "complete":
            progress["seeds_done"] = len(result.get("per_seed", meta["seeds"]))
            progress["seeds_total"] = len(meta["seeds"])
        elif step_status == "stale" and has_partial_progress:
            status, stage = "stalled", "stalled"
        projected = _phase_projection(meta, status, progress, result)
        completed_at = result.get("completed_at", "")[:16] if stage in {"complete", "failed"} else ""
        result_path_str = str(result_path) if result_path.exists() and stage in {"complete", "failed"} else ""
        experiments.append({
            "id": meta["id"],
            "name": meta["name"],
            "status": status,
            "stage": stage,
            "priority": meta["priority"],
            "dataset": meta["dataset"],
            "method": "tcl, ewc, sgd_baseline",
            "seeds": meta["seeds"],
            "estimated_runtime_h": meta["runtime_h"],
            "hardware_budget": meta["hardware_budget"],
            "frontier_problem_id": meta["frontier_problem_id"],
            "progress": progress,
            "author_paper_id": meta["feeds_paper"],
            "context": {
                "why": meta["why"],
                "hypothesis": meta["hypothesis"],
                "projected_outcome": projected,
                "frontier_problem": meta["frontier_problem_id"],
                "feeds_paper": meta["feeds_paper"],
                "methodology_note": meta["methodology"],
            },
            "started_at": "",
            "completed_at": completed_at,
            "result_path": result_path_str,
            "_phase_num": phase_num,
        })
    return experiments


def _runtime_experiment_records() -> list[dict[str, Any]]:
    phase_ids = {meta["id"] for meta in _PHASE_LIVE_META.values()}
    queue_records = _queue_experiments()
    runtime_phases = {rec["id"]: rec for rec in _synthetic_phase_experiments()}
    experiments: list[dict[str, Any]] = []
    living_state = _fresh_state("living_research_daemon.json")
    active_runtime_experiment_id = str(living_state.get("active_experiment_id", "") or "")

    for rec in queue_records:
        rec_id = rec.get("id", "")
        if rec_id in runtime_phases:
            experiments.append(runtime_phases[rec_id])
        else:
            entry = dict(rec)
            status = str(entry.get("status", "") or "")
            stage = str(entry.get("stage", status) or status)
            progress = entry.get("progress", {}) if isinstance(entry.get("progress", {}), dict) else {}
            progress = _annotate_live_progress(rec_id, int(entry.get("pid", 0) or 0), stage or status, progress)
            entry["progress"] = progress
            if status == "running" or stage == "running":
                pid = int(entry.get("pid", 0) or 0)
                pid_matches = _pid_started_for_spec(pid, str(entry.get("started_at", "") or ""))
                daemon_owns_this = bool(active_runtime_experiment_id and active_runtime_experiment_id == rec_id)
                if (
                    (active_runtime_experiment_id and active_runtime_experiment_id != rec_id)
                    or (not daemon_owns_this and pid_matches is False)
                ):
                    entry["status"] = "pending"
                    entry["stage"] = "stalled"
                    entry["pid"] = 0
                    context = dict(entry.get("context", {}) or {})
                    if (
                        int(progress.get("seeds_done", 0) or 0) <= 0
                        and int(progress.get("tasks_done", 0) or 0) <= 0
                        and not list(progress.get("forgetting_so_far", []) or [])
                    ):
                        context["projected_outcome"] = (
                            "This run appears to have stalled before producing a completed first seed. "
                            "TAR should resume or restart it before treating it as active."
                        )
                    else:
                        context["projected_outcome"] = (
                            "This run appears to have stalled after partial progress. "
                            "TAR should resume or restart it from the last safe boundary."
                        )
                    entry["context"] = context
                else:
                    context = dict(entry.get("context", {}) or {})
                    seeds_done = int(progress.get("seeds_done", 0) or 0)
                    seeds_total = int(progress.get("seeds_total", 0) or len(entry.get("seeds", []) or []))
                    tasks_done = int(progress.get("tasks_done", 0) or 0)
                    if seeds_done <= 0 and tasks_done <= 0:
                        context["projected_outcome"] = (
                            f"0/{seeds_total or '?'} seeds complete - live run in progress. "
                            "TAR is awaiting the first completed seed before projecting the outcome. "
                            + str(progress.get("live_compute_note", "") or "")
                        )
                    else:
                        context["projected_outcome"] = (
                            f"{seeds_done}/{seeds_total or '?'} seeds complete - live run in progress. "
                            "Projected outcome will tighten as new seed results are written. "
                            + str(progress.get("live_compute_note", "") or "")
                        )
                    entry["context"] = context
            experiments.append(entry)

    seen_ids = {rec.get("id", "") for rec in experiments}
    for phase_id, rec in runtime_phases.items():
        if phase_id not in seen_ids:
            experiments.append(rec)

    # Inject rerun chain steps (Phase 12/13 managed outside the queue system)
    chain_state = _jload(_WS / "tar_state" / "rerun_chain_state.json") or {}
    for chain_step in (chain_state.get("steps") or []):
        step_status = str(chain_step.get("status", "pending") or "pending")
        if step_status == "complete":
            continue
        step_id = f"{chain_step.get('logical_name', '')}__rerun"
        if step_id in seen_ids:
            continue
        phase_num = int(chain_step.get("phase", 0) or 0)
        log_p = Path(str(chain_step.get("log", "") or ""))
        # parse live progress from log for running steps
        progress: dict[str, Any] = {}
        if step_status == "running" and log_p.exists():
            raw = _parse_training_progress(_tail(log_p, 400))
            _sm = raw.get("seed_markers") or []
            _cur = raw.get("current_seed")
            sd = max(0, len(_sm) - (1 if _cur is not None and _sm else 0))
            # parse best lambda from log lines
            best_lam: dict | None = None
            for _ln in _tail(log_p, 400):
                _lm = re.search(r"(?:EWC|SI)\s+[^\s]*=\s*\S+\s+forgetting=([0-9.]+)\s+acc=([0-9.]+)\s+JAF=([0-9.]+)", _ln)
                if _lm:
                    _forg, _acc, _jaf = float(_lm.group(1)), float(_lm.group(2)), float(_lm.group(3))
                    if best_lam is None or _jaf > best_lam["jaf"]:
                        best_lam = {"forgetting": _forg, "acc": _acc, "jaf": _jaf}
            note = f"Best so far: acc={best_lam['acc']:.3f} forg={best_lam['forgetting']:.3f}" if best_lam else ""
            progress = {"seeds_done": sd, "seeds_total": 5, "live_compute_note": note}
        stage = step_status if step_status in {"running", "pending"} else "queued"
        if step_status == "pending":
            stage = "queued"
        projected = (
            (f"{progress.get('seeds_done', 0)}/5 seeds done — rerun live. " + str(progress.get("live_compute_note", ""))).strip()
            if step_status == "running"
            else f"Queued — waiting for prerequisite ({chain_step.get('wait_for', '')}) to land in canonical index."
        )
        experiments.append({
            "id":               step_id,
            "name":             f"Phase {phase_num} Rerun — {chain_step.get('logical_name', '').replace('_', ' ').title()}",
            "dataset":          "split_cifar10",
            "status":           "running" if step_status == "running" else "pending",
            "stage":            stage,
            "priority":         phase_num,
            "submitted_at":     "",
            "hardware_budget":  {"vram_gb": 2.5, "cpu_cores": 4},
            "tags":             ["rerun", "manifest_gated", f"phase{phase_num}"],
            "progress":         progress,
            "context": {
                "why": f"Controlled rerun of Phase {phase_num} under RAIL-3 manifest gate with corrected reference numbers from Phase 10 controlled rerun.",
                "projected_outcome": projected,
                "frontier_problem": "",
                "feeds_paper": "All papers — Phase reruns provide trusted canonical results.",
                "methodology_note": f"Manifest: {chain_step.get('manifest_id', '')}. Log: {log_p.name if log_p.name else '—'}.",
            },
            "pid": 0,
            "_phase_num": phase_num,
        })
        seen_ids.add(step_id)

    experiments.sort(key=lambda rec: (rec.get("priority", 50), rec.get("submitted_at", "")))
    return experiments


def _experiment_live_snapshot(exp: dict[str, Any]) -> dict[str, Any]:
    exp_id = str(exp.get("id", "") or "")
    progress = exp.get("progress", {}) if isinstance(exp.get("progress", {}), dict) else {}
    progress = _annotate_live_progress(
        exp_id,
        int(exp.get("pid", 0) or 0),
        str(exp.get("stage", "") or exp.get("status", "") or ""),
        progress,
    )
    runtime_context = exp.get("runtime_context", {}) if isinstance(exp.get("runtime_context", {}), dict) else {}
    checkpoint = _suite_checkpoint_progress(
        exp_id,
        [int(seed) for seed in exp.get("seeds", []) if str(seed).strip()],
    )
    hardware = _fresh_state("hardware_state.json", max_age_s=120.0)
    process_registry = _json_state("process_registry.json")
    pid = int(exp.get("pid", 0) or 0)
    pid_meta = process_registry.get(str(pid), {}) if isinstance(process_registry, dict) else {}
    lines = [
        f"experiment_id={exp_id}",
        f"status={exp.get('status', '')} stage={exp.get('stage', '')}",
        f"started_at={exp.get('started_at', '') or '(unknown)'} pid={pid or '(none)'}",
    ]
    if exp.get("depends_on"):
        lines.append(f"depends_on={', '.join(str(dep) for dep in exp.get('depends_on', []) if str(dep))}")
    if runtime_context.get("current_runtime_dir"):
        lines.append(f"runtime_dir={runtime_context.get('current_runtime_dir')}")
    if runtime_context.get("manifest_path"):
        lines.append(f"manifest={runtime_context.get('manifest_path')}")
    if pid_meta:
        lines.append(f"process_registry_owner={pid_meta.get('owner', '')} stage={pid_meta.get('stage', '')}")
    tasks_done = int(progress.get("tasks_done", 0) or 0)
    latest_accs = list(progress.get("latest_accs", []) or [])
    method_summaries = list(progress.get("method_summaries", []) or [])

    if checkpoint:
        completed = int(checkpoint.get("seeds_done", 0) or 0)
        total = int(checkpoint.get("seeds_total", 0) or len(exp.get("seeds", []) or []))
        lines.append(
            f"checkpoint={checkpoint.get('checkpoint_source', 'checkpoint')} updated={checkpoint.get('last_checkpoint_at', '') or '(unknown)'}"
        )
        lines.append(f"completed_seeds={completed}/{total} next_seed={checkpoint.get('current_seed') or '(none)'}")
        forgetting = checkpoint.get("forgetting_so_far", []) or []
        if forgetting:
            lines.append(
                "tcl_forgetting_so_far="
                + ", ".join(f"{float(val):.4f}" for val in forgetting[:6])
            )
    elif exp.get("status") == "running" and tasks_done <= 0 and not method_summaries:
        lines.append("No completed suite checkpoint yet; likely dataset load / first-seed warmup.")
    if tasks_done:
        lines.append(f"tasks_done={tasks_done}/10")
    if latest_accs:
        lines.append(f"latest_accs={', '.join(str(val) for val in latest_accs)}")
    if method_summaries:
        last_summary = method_summaries[-1]
        lines.append(
            f"latest_method_summary={last_summary.get('method')} forgetting={float(last_summary.get('forgetting', 0.0)):.4f} acc={float(last_summary.get('accuracy', 0.0)):.4f}"
        )
    if progress.get("last_visible_log_at"):
        lines.append(f"last_visible_log_at={progress.get('last_visible_log_at')}")
    if progress.get("live_compute_note"):
        lines.append(str(progress.get("live_compute_note")))
    gpu = hardware.get("gpu", {}) if isinstance(hardware, dict) else {}
    ram = hardware.get("ram", {}) if isinstance(hardware, dict) else {}
    if gpu or ram:
        gpu_util = gpu.get("utilization_pct", "—")
        gpu_vram = f"{float(gpu.get('vram_used_gb', 0.0) or 0.0):.2f}/{float(gpu.get('vram_total_gb', 0.0) or 0.0):.1f}GB"
        ram_used = f"{float(ram.get('used_gb', 0.0) or 0.0):.1f}/{float(ram.get('total_gb', 0.0) or 0.0):.1f}GB"
        lines.append(f"hardware_snapshot gpu={gpu_util}% vram={gpu_vram} ram={ram_used}")
    mtime = checkpoint.get("last_checkpoint_at", "") or hardware.get("timestamp", "") or _now_iso()
    return {
        "source_id": "live-snapshot",
        "label": "Live progress snapshot",
        "path": f"state://experiment/{exp_id}/snapshot",
        "mtime": str(mtime),
        "lines": lines,
        "open_log_name": "",
    }


def _experiment_log_sources(exp: dict[str, Any]) -> list[dict[str, Any]]:
    log_dir = _resolve_logs()
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_file_source(
        source_id: str,
        label: str,
        path: Path | None,
        *,
        open_log_name: str = "",
        kind: str = "file",
    ) -> None:
        if path is None or not path.exists():
            return
        key = f"{kind}:{path}"
        if key in seen:
            return
        seen.add(key)
        sources.append({
            "source_id": source_id,
            "label": label,
            "path": str(path),
            "kind": kind,
            "open_log_name": open_log_name,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S"),
        })

    snapshot = _experiment_live_snapshot(exp)
    sources.append({
        "source_id": snapshot["source_id"],
        "label": snapshot["label"],
        "path": snapshot["path"],
        "kind": "snapshot",
        "open_log_name": "",
        "mtime": str(snapshot.get("mtime", "") or ""),
    })

    phase_num = exp.get("_phase_num")
    if phase_num == 16 or str(exp.get("id", "") or "") == "phase16_scale_up":
        add_file_source("phase16", "Phase 16 live log", log_dir / "phase16.log", open_log_name="phase16")
    if phase_num == 17 or str(exp.get("id", "") or "") == "phase17_tinyimagenet":
        add_file_source("phase17", "Phase 17 live log", log_dir / "phase17.log", open_log_name="phase17")

    runtime_context = exp.get("runtime_context", {}) if isinstance(exp.get("runtime_context", {}), dict) else {}
    runtime_dir = Path(str(runtime_context.get("current_runtime_dir", "") or "")) if runtime_context.get("current_runtime_dir") else None
    if runtime_dir and runtime_dir.exists():
        add_file_source("runtime-bootstrap", "Runtime bootstrap", runtime_dir / "bootstrap.log")
        runtime_logs_dir = runtime_dir / "logs"
        if runtime_logs_dir.exists():
            runtime_logs = sorted(
                runtime_logs_dir.glob("*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for idx, path in enumerate(runtime_logs[:4]):
                add_file_source(
                    f"runtime-log-{idx}",
                    f"Runtime log - {path.stem}",
                    path,
                )

    add_file_source(
        "watchdog-living-research",
        "Watchdog living-research stdout",
        _latest_matching_log("watchdog-living-research-*.log"),
        kind="filtered",
    )
    add_file_source(
        "living-research",
        "Living Research trace",
        _WS / "tar_state" / "living_research.log",
        open_log_name="living_research",
    )
    add_file_source(
        "orchestrator-trace",
        "Orchestrator trace",
        _WS / "tar_state" / "experiment_orchestrator.log",
        kind="filtered",
    )
    return sources


def _parse_log_line_utc(line: str) -> datetime | None:
    """Extract the UTC datetime embedded in a log line like '[2026-05-26 05:35:51 UTC] ...'."""
    m = _LOG_LINE_TS_RE.match(line.strip())
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _preflight_run_start(exp: dict) -> datetime | None:
    """
    Return the prepared_at timestamp from the current run's preflight.json.
    Falls back to started_at from the experiment queue entry.
    Returns None if neither is available (caller skips filtering).
    """
    rc = exp.get("runtime_context") if isinstance(exp.get("runtime_context"), dict) else {}
    runtime_dir = (rc or {}).get("current_runtime_dir", "")
    if runtime_dir:
        try:
            pf = json.loads(
                (Path(runtime_dir) / "manifests" / "preflight.json")
                .read_text(encoding="utf-8")
            )
            ts = pf.get("prepared_at", "")
            if ts:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            pass
    try:
        ts = str(exp.get("started_at", "") or "")
        if ts:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        pass
    return None


def _filtered_experiment_trace(path: Path, exp: dict[str, Any], limit: int = 160) -> list[str]:
    lines = _tail(path, 600) if path.exists() else []
    exp_id = str(exp.get("id", "") or "")
    exp_name = str(exp.get("name", "") or "")
    match_indexes = [
        idx for idx, line in enumerate(lines)
        if (exp_id and exp_id in line) or (exp_name and exp_name in line)
    ]
    if match_indexes:
        start_idx = match_indexes[-1]
        for idx in range(match_indexes[-1], -1, -1):
            text = lines[idx]
            if (
                (exp_id and exp_id in text)
                or (exp_name and exp_name in text)
                or "[execute]" in text
            ):
                start_idx = idx
        block = lines[start_idx:]
        # Drop lines from prior runs: any timestamped line before this run's prepared_at
        # is stale data from a previous execution of the same experiment ID.
        run_start = _preflight_run_start(exp)
        if run_start:
            filtered = []
            for line in block:
                ts = _parse_log_line_utc(line)
                if ts is None or ts >= run_start:
                    filtered.append(line)
            block = filtered
        if block:
            return block[-limit:]
    return lines[-min(limit, len(lines)):] if lines else ["(no experiment trace yet)"]


def _resolve_experiment_log(exp: dict[str, Any], source_id: str = "") -> dict[str, Any]:
    sources = _experiment_log_sources(exp)
    source = None
    if source_id:
        source = next((rec for rec in sources if rec.get("source_id") == source_id), None)
    if source is None and sources:
        running = str(exp.get("status", "") or exp.get("stage", "") or "") == "running"
        preferred_source_ids = (
            ("phase17", "phase16", "runtime-log-0", "runtime-bootstrap", "live-snapshot", "watchdog-living-research", "living-research", "orchestrator-trace")
            if not running else
            ("phase17", "phase16", "runtime-log-0", "runtime-bootstrap", "live-snapshot", "watchdog-living-research", "living-research", "orchestrator-trace")
        )
        if running and not any(rec.get("source_id") in {"phase17", "phase16", "runtime-log-0"} for rec in sources):
            preferred_source_ids = (
                "live-snapshot",
                "watchdog-living-research",
                "living-research",
                "orchestrator-trace",
            )
        for preferred_id in preferred_source_ids:
            source = next((rec for rec in sources if rec.get("source_id") == preferred_id), None)
            if source is not None:
                break
        if source is None:
            source = sources[0]

    if not source:
        return {
            "source_id": "",
            "label": "No experiment log available yet",
            "path": "",
            "mtime": "",
            "lines": ["(no experiment log available yet)"],
            "open_log_name": "",
            "sources": [],
        }

    if str(source.get("kind", "")) == "snapshot":
        snapshot = _experiment_live_snapshot(exp)
        snapshot["sources"] = sources
        return snapshot

    path = Path(str(source.get("path", "") or ""))
    if str(source.get("kind", "file")) == "filtered":
        lines = _filtered_experiment_trace(path, exp)
    else:
        lines = _tail(path, 160) if path.exists() else ["(log file not found)"]
    return {
        "source_id": str(source.get("source_id", "") or ""),
        "label": str(source.get("label", "") or "Experiment log"),
        "path": str(path),
        "mtime": str(source.get("mtime", "") or ""),
        "lines": lines,
        "open_log_name": str(source.get("open_log_name", "") or ""),
        "sources": sources,
    }


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route("/")
def index():
    return Response(_dashboard_html(), mimetype="text/html")


# ── hardware ──────────────────────────────────────────────────────────────────
@app.route("/api/hardware")
def api_hardware():
    hw_path = _WS / "tar_state" / "hardware_state.json"
    data = _jload(hw_path) or {}
    needs_refresh = (
        not data
        or _age_s(hw_path) > 15
        or (data.get("ram", {}) or {}).get("total_gb", 0) <= 0
    )
    if needs_refresh:
        try:
            from tar_hardware_monitor import take_snapshot, write_snapshot
            data = take_snapshot(_WS)
            write_snapshot(_WS, data)
        except Exception:
            data = data or {}
    if not data:
        # Fallback: basic torch query
        gpu: dict = {"available": False, "name": "", "utilization_pct": 0,
                     "vram_used_gb": 0, "vram_total_gb": 0, "temperature_c": 0}
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                used = torch.cuda.memory_allocated(0) / 1e9
                gpu = {"available": True, "name": p.name,
                       "vram_used_gb": round(used, 2),
                       "vram_total_gb": round(p.total_memory / 1e9, 2),
                       "utilization_pct": 0, "temperature_c": 0}
        except Exception:
            pass
        data = {"gpu": gpu, "cpu": {}, "ram": {}, "processes": [],
                "timestamp": datetime.now(timezone.utc).isoformat()}
    return jsonify(data)


# ── status ────────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    log_dir = _resolve_logs()
    queue_state = _queue_state()
    runtime_exps = _runtime_experiment_records()
    steps = _status_queue_steps(runtime_exps)
    running_exp = next((e for e in runtime_exps if e.get("stage") == "running"), None)
    stalled_exp = next((e for e in runtime_exps if e.get("stage") == "stalled"), None)
    queued_exp = next((e for e in runtime_exps if e.get("stage") in {"queued", "planned"} or e.get("status") == "pending"), None)
    running = next((s for s in steps if s["status"] == "running"), None)
    active_path, active_label = _pick_active_log()
    progress = {}
    if active_path:
        progress = _parse_training_progress(_tail(active_path, 300))
    all_done = all(s["status"] in ("complete", "stale") for s in steps)
    return jsonify({
        "workspace":   str(_WS),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "queue_steps": steps,
        "active_step": (
            running_exp["name"] if running_exp
            else f"No active experiment — next resume: {stalled_exp['name']}" if stalled_exp
            else f"No active experiment — next queued: {queued_exp['name']}" if queued_exp
            else queue_state.get("current_step") if queue_state.get("status") in {"starting", "waiting", "running"} and queue_state.get("current_step")
            else running["label"] if running
            else active_label if active_path and _age_s(active_path) < 900
            else ("complete" if all_done else "idle")
        ),
        "active_log":  active_label,
        "progress":    progress,
        "queue_state": queue_state,
    })


# ── log endpoints ─────────────────────────────────────────────────────────────
@app.route("/api/log")
@app.route("/api/log/<name>")
def api_log(name: str = ""):
    log_dir = _resolve_logs()
    if name:
        path = log_dir / f"{name}.log"
        if not path.exists():
            path = _WS / "tar_state" / f"{name}.log"
        label = name
    else:
        path, label = _pick_active_log()
    lines = _tail(path, 120) if path and path.exists() else ["(no log yet)"]
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S") if path and path.exists() else ""
    return jsonify({"label": label, "lines": lines, "mtime": mtime, "path": str(path) if path else ""})


@app.route("/api/logs")
def api_logs():
    log_dir = _resolve_logs()
    logs = []
    extra = [_WS / "tar_state" / "autonomous_research_resume.log",
             _WS / "tar_state" / "autonomous_research.log"]
    if log_dir.exists():
        for p in sorted(log_dir.glob("*.log")):
            logs.append({"name": p.stem, "size_kb": int(p.stat().st_size / 1024),
                         "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")})
    for p in extra:
        if p.exists():
            logs.append({"name": p.stem, "size_kb": int(p.stat().st_size / 1024),
                         "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")})
    return jsonify(logs)


# ── phases ────────────────────────────────────────────────────────────────────
@app.route("/api/phases")
def api_phases():
    return jsonify(_phase_result_records())


def _phase_result_records() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    runtime_by_phase = {
        rec["_phase_num"]: rec
        for rec in _runtime_experiment_records()
        if "_phase_num" in rec
    }
    if not _COMP.exists():
        return results

    # Build canonical-index map: trusted result path keyed by phase number.
    # Index is the authoritative source; the glob below only handles phases
    # not yet in the index (e.g. 14, 15, 16, 17).
    canonical_by_phase: dict[int, Path] = {}
    canonical_by_phase_ts: dict[int, str] = {}
    try:
        for rec in iter_canonical_comparison_records(_WS):
            pnum  = rec.get("phase_number")
            rpath = rec.get("result_path", "")
            if pnum is None or not rpath:
                continue
            p = Path(str(rpath))
            if not p.exists():
                continue
            pnum = int(pnum)
            ts = str(rec.get("created_at", "") or "")
            # Latest created_at wins — actual reruns supersede earlier recomputations
            if ts > canonical_by_phase_ts.get(pnum, ""):
                canonical_by_phase[pnum] = p
                canonical_by_phase_ts[pnum] = ts
    except Exception:
        pass

    def _make_record(p: Path, num: int) -> dict[str, Any]:
        data     = _jload(p) or {}
        agg      = data.get("aggregate", {})
        tcl      = agg.get("tcl", {}) or agg.get("full_tcl", {})
        pw       = data.get("pairwise", {})
        vs_e     = pw.get("ewc", {})
        evidence = _result_evidence_payload(p) or {}
        is_err   = data.get("status") == "ERROR" or "ERROR" in data.get("verdict", "")
        is_bk    = (data.get("verdict_key") == "BREAKTHROUGH"
                    or data.get("external_breakthrough_candidate") is True)
        return {
            "phase":          num,
            "label":          p.stem,
            "title":          f"Phase {num}",
            "program":        "phase program",
            "source_kind":    "phase",
            "status":         "error" if is_err else "breakthrough" if is_bk else "complete",
            "dataset":        data.get("dataset") or data.get("benchmark", "split_cifar10"),
            "verdict_key":    data.get("verdict_key", ""),
            "verdict":        evidence.get("verdict") or (data.get("verdict") or data.get("summary") or data.get("error", ""))[:120],
            "tcl_forgetting": tcl.get("forgetting_mean"),
            "tcl_accuracy":   tcl.get("acc_mean"),
            "mean_delta":     evidence.get("mean_delta"),
            "vs_ewc_p":       evidence.get("p_val") or vs_e.get("p_val") or data.get("p_value_vs_strong_baseline"),
            "vs_ewc_d":       evidence.get("cohens_d") or vs_e.get("cohens_d") or data.get("effect_size_vs_strong_baseline"),
            "notes":          evidence.get("notes", ""),
            "result_path":    evidence.get("result_path", str(p)),
            "completed_at":   str(data.get("completed_at", "") or "")[:19],
        }

    # 1. Trusted canonical results (from index)
    for pnum, p in sorted(canonical_by_phase.items()):
        results.append(_make_record(p, pnum))

    # 2. Non-indexed phase files (phases 14+, not yet in canonical index)
    indexed_phases = set(canonical_by_phase.keys())
    for p in sorted(_COMP.glob("phase*.json")):
        m = re.search(r"phase(\d+)", p.stem)
        if not m:
            continue
        num = int(m.group(1))
        if num in indexed_phases:
            continue  # already covered by canonical
        results.append(_make_record(p, num))

    # 3. Overlay live/running runtime status where applicable
    for r in results:
        runtime = runtime_by_phase.get(r["phase"])
        if runtime and runtime.get("stage") in {"running", "queued", "planned", "stalled"}:
            r["status"]  = runtime["stage"]
            r["verdict"] = runtime["context"]["projected_outcome"][:120]
            r["title"]   = runtime.get("name", r["title"])
            r["program"] = "live phase"

    # 3b. active_rerun.json overlay — if the rerun chain is running a phase right
    #     now, force that phase's status to "running" regardless of what the
    #     canonical index says (the index may still point to corrected-stats
    #     from the previous step).
    active_rerun = _jload(_WS / "tar_state" / "active_rerun.json") or {}
    active_rerun_phase = int(active_rerun.get("phase", 0) or 0)
    if active_rerun_phase:
        log_path_ar = Path(str(active_rerun.get("log", "") or ""))
        progress_ar = _parse_training_progress(_tail(log_path_ar, 400)) if log_path_ar.exists() else {}
        # _parse_training_progress returns seed_markers, not seeds_done — compute it here
        _sm = progress_ar.get("seed_markers") or []
        _cur = progress_ar.get("current_seed")
        seeds_done_ar = max(0, len(_sm) - (1 if _cur is not None and _sm else 0))
        seeds_total_ar = 5
        # build a short live-results snippet from completed seeds
        _msums = progress_ar.get("method_summaries") or []
        _best_lambda: dict | None = None
        # Phase 12/13 log: EWC/SI lines parsed into method_summaries; pick best JAF per seed
        # also parse lambda lines directly from tail if method_summaries is empty
        if not _msums and log_path_ar.exists():
            for _ln in _tail(log_path_ar, 400):
                _lm = re.search(r"EWC\s+λ=\s*(\d+)\s+forgetting=([0-9.]+)\s+acc=([0-9.]+)\s+JAF=([0-9.]+)", _ln)
                if _lm:
                    _lam, _forg, _acc, _jaf = int(_lm.group(1)), float(_lm.group(2)), float(_lm.group(3)), float(_lm.group(4))
                    if _best_lambda is None or _jaf > _best_lambda["jaf"]:
                        _best_lambda = {"lambda": _lam, "forgetting": _forg, "acc": _acc, "jaf": _jaf}
        live_note_ar = ""
        if _best_lambda:
            live_note_ar = f"Best so far: λ={_best_lambda['lambda']} acc={_best_lambda['acc']:.3f} forg={_best_lambda['forgetting']:.3f}"
        verdict_ar = f"{seeds_done_ar}/{seeds_total_ar} seeds done — rerun live. {live_note_ar}".strip()
        for r in results:
            if r["phase"] == active_rerun_phase:
                r["status"]      = "running"
                r["verdict_key"] = "RUNNING"
                r["verdict"]     = verdict_ar[:140]
                r["title"]       = f"Phase {active_rerun_phase} — Rerun Live"
                r["program"]     = "live phase"
                r["notes"]       = (
                    f"Rerun chain active (manifest: {active_rerun.get('manifest_id', '')}). "
                    f"Log: {log_path_ar.name or '—'}. "
                    + (f"Seeds complete: {seeds_done_ar}/{seeds_total_ar}." if seeds_done_ar else "")
                )
                break

    # 4. RERUN_NEEDED: phases quarantined to _untrusted/ with no canonical result yet.
    active_rerun_phase = int(active_rerun.get("phase", 0) or 0)

    seen_phases = {r["phase"] for r in results}
    untrusted_dir = _COMP / "_untrusted"
    if untrusted_dir.exists():
        for p in sorted(untrusted_dir.glob("phase*.json")):
            m = re.search(r"phase(\d+)", p.stem)
            if not m:
                continue
            num = int(m.group(1))
            if num in seen_phases:
                continue
            runtime = runtime_by_phase.get(num)
            if runtime and runtime.get("stage") in {"running", "queued", "planned", "stalled"}:
                results.append({
                    "phase":          num,
                    "label":          runtime["id"],
                    "title":          runtime.get("name", f"Phase {num}"),
                    "program":        "live phase",
                    "source_kind":    "phase",
                    "status":         runtime["stage"],
                    "dataset":        runtime.get("dataset", "split_cifar10"),
                    "verdict_key":    "",
                    "verdict":        runtime["context"]["projected_outcome"][:120],
                    "tcl_forgetting": None,
                    "tcl_accuracy":   None,
                    "mean_delta":     None,
                    "vs_ewc_p":       None,
                    "vs_ewc_d":       None,
                    "notes":          runtime.get("context", {}).get("why", "")[:220],
                    "result_path":    runtime.get("result_path", ""),
                    "completed_at":   "",
                })
            elif num == active_rerun_phase:
                # Standalone chain script is running this phase right now
                log_path = Path(str(active_rerun.get("log", "") or ""))
                progress = _parse_training_progress(_tail(log_path, 200)) if log_path.exists() else {}
                seeds_done  = int(progress.get("seeds_done",  0) or 0)
                seeds_total = int(progress.get("seeds_total", 5) or 5)
                live_note   = str(progress.get("live_compute_note", "") or "")
                verdict_live = (
                    f"{seeds_done}/{seeds_total} seeds complete — live rerun in progress. {live_note}"
                ).strip()
                results.append({
                    "phase":          num,
                    "label":          f"phase{num}_rerun_running",
                    "title":          f"Phase {num} — Rerun Running",
                    "program":        "live phase",
                    "source_kind":    "phase",
                    "status":         "running",
                    "dataset":        "split_cifar10",
                    "verdict_key":    "RUNNING",
                    "verdict":        verdict_live[:120],
                    "tcl_forgetting": None,
                    "tcl_accuracy":   None,
                    "mean_delta":     None,
                    "vs_ewc_p":       None,
                    "vs_ewc_d":       None,
                    "notes":          f"Rerun chain active (manifest: {active_rerun.get('manifest_id','')}). Log: {log_path.name if log_path.name else '—'}",
                    "result_path":    str(p),
                    "completed_at":   "",
                })
            else:
                results.append({
                    "phase":          num,
                    "label":          f"phase{num}_rerun_needed",
                    "title":          f"Phase {num} — Rerun Needed",
                    "program":        "rerun_needed",
                    "source_kind":    "phase",
                    "status":         "rerun_needed",
                    "dataset":        "split_cifar10",
                    "verdict_key":    "RERUN_NEEDED",
                    "verdict":        "Untrusted result quarantined. Rerun pending.",
                    "tcl_forgetting": None,
                    "tcl_accuracy":   None,
                    "mean_delta":     None,
                    "vs_ewc_p":       None,
                    "vs_ewc_d":       None,
                    "notes":          "Original result quarantined to _untrusted/ — ran against corrupted Phase 10 baseline. Rerun queued.",
                    "result_path":    str(p),
                    "completed_at":   "",
                })
            seen_phases.add(num)

    # 5. Runtime-only entries (running/queued with no result file of any kind)
    seen_phases = {r["phase"] for r in results}
    for phase_num, runtime in runtime_by_phase.items():
        if phase_num in seen_phases:
            continue
        if runtime.get("stage") not in {"running", "queued", "planned", "stalled"}:
            continue
        results.append({
            "phase":          phase_num,
            "label":          runtime["id"],
            "title":          runtime.get("name", runtime["id"]),
            "program":        "live phase",
            "source_kind":    "phase",
            "status":         runtime["stage"],
            "dataset":        runtime["dataset"],
            "verdict_key":    "",
            "verdict":        runtime["context"]["projected_outcome"][:120],
            "tcl_forgetting": None,
            "tcl_accuracy":   None,
            "mean_delta":     None,
            "vs_ewc_p":       None,
            "vs_ewc_d":       None,
            "notes":          runtime.get("context", {}).get("why", "")[:220],
            "result_path":    runtime.get("result_path", ""),
            "completed_at":   "",
        })

    results.sort(key=lambda x: x["phase"])
    return results


def _experiment_program_label(exp_id: str, rec: dict[str, Any]) -> str:
    if exp_id.startswith("director-"):
        return "director"
    if exp_id.startswith("ar-"):
        return "autonomous"
    if exp_id.startswith("phase"):
        return "scale-up"
    runner_key = str(rec.get("runner_key", "") or "")
    if runner_key:
        return runner_key.replace("_", " ")
    return "experiment"


def _experiment_result_records() -> list[dict[str, Any]]:
    try:
        from tar_experiment_library import build_experiment_library
    except Exception:
        return []

    library = build_experiment_library(_WS)
    records = library.get("experiments", []) if isinstance(library, dict) else []
    records = records if isinstance(records, list) else []
    phase_rows = {rec["phase"]: rec for rec in _phase_result_records() if "phase" in rec}
    rows: list[dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        exp_id = str(rec.get("experiment_id", "") or "")
        if not exp_id:
            continue
        summary = rec.get("result_summary", {}) if isinstance(rec.get("result_summary", {}), dict) else {}
        status = str(rec.get("stage", "") or rec.get("status", "") or "")
        has_result = bool(summary) or bool(rec.get("result_path"))
        if not has_result and status not in {"running", "stalled"}:
            continue

        phase_match = re.match(r"phase(\d+)", exp_id)
        if phase_match:
            phase_num = int(phase_match.group(1))
            merged = phase_rows.get(phase_num)
            if merged is not None:
                merged["title"] = str(rec.get("name", "") or merged.get("title", f"Phase {phase_num}"))
                merged["program"] = "scale-up"
                merged["result_path"] = str(rec.get("result_path", "") or merged.get("result_path", ""))
                merged["completed_at"] = str(summary.get("completed_at", "") or merged.get("completed_at", ""))
                merged["mean_delta"] = summary.get("mean_delta", merged.get("mean_delta"))
                merged["vs_ewc_p"] = summary.get("p_val", merged.get("vs_ewc_p"))
                merged["vs_ewc_d"] = summary.get("cohens_d", merged.get("vs_ewc_d"))
                merged["notes"] = str(summary.get("notes", "") or merged.get("notes", "") or "")
                if status in {"running", "stalled"}:
                    merged["status"] = status
                    merged["verdict"] = str((rec.get("context") or {}).get("projected_outcome", "") or merged.get("verdict", ""))
                elif summary.get("verdict"):
                    merged["verdict"] = str(summary.get("verdict", "") or merged.get("verdict", ""))
                phase_rows[phase_num] = merged
                continue

        rows.append({
            "id": exp_id,
            "title": str(rec.get("name", "") or exp_id.replace("_", " ").replace("-", " ").title()),
            "program": _experiment_program_label(exp_id, rec),
            "source_kind": "experiment",
            "status": status or str(rec.get("status", "") or ""),
            "dataset": str(rec.get("dataset", "") or ""),
            "verdict": str(summary.get("verdict", "") or status or rec.get("status", "") or "").upper(),
            "tcl_forgetting": summary.get("mean_forgetting"),
            "tcl_accuracy": summary.get("mean_accuracy"),
            "mean_delta": summary.get("mean_delta"),
            "vs_ewc_p": summary.get("p_val"),
            "vs_ewc_d": summary.get("cohens_d"),
            "notes": str(summary.get("notes", "") or (rec.get("context", {}) or {}).get("projected_outcome", "") or "")[:220],
            "result_path": str(rec.get("result_path", "") or ""),
            "completed_at": str(summary.get("completed_at", "") or rec.get("completed_at", "") or "")[:19],
        })

    return rows


@app.route("/api/results")
def api_results():
    rows = _phase_result_records()
    rows.extend(_experiment_result_records())
    unique: dict[str, dict[str, Any]] = {}
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        rec_id = str(rec.get("id", "") or rec.get("label", "") or rec.get("title", "") or "")
        if not rec_id:
            continue
        unique[rec_id] = rec

    status_rank = {
        "running": 0,
        "stalled": 1,
        "breakthrough": 2,
        "complete": 3,
        "failed": 4,
        "error": 4,
        "queued": 5,
        "planned": 6,
        "pending": 6,
    }
    ordered = sorted(
        unique.values(),
        key=lambda rec: (
            status_rank.get(str(rec.get("status", "") or "").lower(), 99),
            str(rec.get("completed_at", "") or ""),
            str(rec.get("title", "") or ""),
        ),
        reverse=False,
    )
    ordered.sort(
        key=lambda rec: (
            status_rank.get(str(rec.get("status", "") or "").lower(), 99),
            str(rec.get("completed_at", "") or ""),
        ),
    )
    return jsonify({
        "total": len(ordered),
        "running": sum(1 for rec in ordered if str(rec.get("status", "") or "") == "running"),
        "breakthroughs": sum(1 for rec in ordered if "BREAKTHROUGH" in str(rec.get("verdict", "") or "")),
        "complete": sum(1 for rec in ordered if str(rec.get("status", "") or "") in {"complete", "breakthrough"}),
        "failed": sum(1 for rec in ordered if str(rec.get("status", "") or "") in {"failed", "error"}),
        "results": ordered,
    })


# ── experiments ───────────────────────────────────────────────────────────────
@app.route("/api/experiments")
def api_experiments():
    data = _jload(_WS / "tar_state" / "experiment_queue.json") or {}
    data = data if isinstance(data, dict) else {}
    experiments = [
        e for e in _runtime_experiment_records()
        if str(e.get("status", "") or "") not in {"complete", "failed", "skipped"}
        and str(e.get("stage", "") or "") not in {"complete", "failed"}
    ]
    archive = _jload(_WS / "tar_state" / "experiment_archive.json") or {}
    archived_experiments = archive.get("experiments", []) if isinstance(archive, dict) else []
    archived_experiments = archived_experiments if isinstance(archived_experiments, list) else []
    done_count = sum(
        1 for e in archived_experiments
        if str(e.get("status", "") or "") == "complete"
        or str(e.get("stage", "") or "") == "complete"
    )
    pending_count = sum(
        1 for e in experiments
        if str(e.get("status", "") or "") == "pending"
        or str(e.get("stage", "") or "") in {"planned", "queued", "stalled", "analyzing", "writing_paper"}
    )
    return jsonify({
        "saved_at": data.get("saved_at", ""),
        "total":    len(experiments),
        "pending":  pending_count,
        "running":  sum(1 for e in experiments if e.get("status") == "running"),
        "complete": done_count,
        "failed":   sum(
            1 for e in archived_experiments
            if str(e.get("status", "") or "") == "failed"
            or str(e.get("stage", "") or "") == "failed"
        ),
        "experiments": [
            {
                "id":               e.get("id", ""),
                "name":             e.get("name", ""),
                "status":           e.get("status", ""),
                "stage":            e.get("stage", e.get("status", "")),
                "priority":         e.get("priority", 50),
                "dataset":          e.get("dataset", ""),
                "method":           e.get("method", ""),
                "seeds":            e.get("seeds", []),
                "est_h":            e.get("estimated_runtime_h", 0),
                "hardware_budget":  e.get("hardware_budget", {}),
                "frontier_problem_id": e.get("frontier_problem_id", ""),
                "progress":         e.get("progress", {}),
                "context_why":      (e.get("context") or {}).get("why", "")[:120],
                "projected_outcome": (e.get("context") or {}).get("projected_outcome", ""),
                "started_at":       e.get("started_at", "")[:16],
                "completed_at":     e.get("completed_at", "")[:16],
                "result_path":      e.get("result_path", ""),
            }
            for e in experiments
        ],
    })


@app.route("/api/experiment/<exp_id>")
def api_experiment_detail(exp_id: str):
    """Full detail for modal — spec + context + live progress + result if done."""
    for e in _runtime_experiment_records():
        if e["id"] == exp_id:
            result = None
            rp = e.get("result_path", "")
            if rp and Path(rp).exists() and e.get("stage") in {"complete", "failed"}:
                result = _jload(Path(rp))
            exp_dir = _WS / "tar_state" / "experiments" / exp_id / "result.json"
            if not result and exp_dir.exists() and e.get("_phase_num") is None:
                result = _jload(exp_dir)
            return jsonify({"spec": e, "result": result})
    return jsonify({"spec": None, "result": None}), 404


@app.route("/api/experiment/<exp_id>/log")
def api_experiment_log(exp_id: str):
    source_id = str(request.args.get("source", "") or "")
    for e in _runtime_experiment_records():
        if e["id"] == exp_id:
            return jsonify(_resolve_experiment_log(e, source_id))
    return jsonify({
        "source_id": "",
        "label": "Unknown experiment",
        "path": "",
        "mtime": "",
        "lines": ["(experiment not found)"],
        "open_log_name": "",
        "sources": [],
    }), 404


# ── autonomous research ───────────────────────────────────────────────────────
@app.route("/api/autonomous")
def api_autonomous():
    results = []
    prereg = _jload(_AR / "preregistration.json") or {}
    registered = {h["name"]: h for h in prereg.get("hypotheses", [])}
    for name in ["deep_anchor", "graduated_penalty", "strict_consolidation",
                 "thermal_carryover", "high_penalty_conservative"]:
        entry: dict[str, Any] = {
            "name": name, "status": "pending", "verdict": "",
            "mean_forgetting": None, "mean_delta": None,
            "p_val": None, "cohens_d": None, "n_better": None, "seeds_done": 0,
        }
        if name in registered:
            entry["prediction"] = registered[name].get("prediction", "")[:80]
        partial = _jload(_AR / f"{name}_partial.json")
        if partial:
            done = partial.get("seeds_completed", [])
            entry["seeds_done"] = len(done)
            entry["status"] = "partial"
            agg = partial.get("partial_aggregate", {})
            entry["mean_forgetting"] = agg.get("forgetting_mean")
            entry["mean_delta"] = partial.get("partial_delta_vs_tcl")
        full = _jload(_AR / f"{name}.json")
        if full:
            res = full.get("result", {})
            forg_list = res.get("mechanism_forgetting", [])
            entry.update({
                "status":          "complete",
                "verdict":         res.get("verdict", ""),
                "mean_forgetting": (sum(forg_list) / len(forg_list)) if forg_list else None,
                "mean_delta":      res.get("mean_delta"),
                "p_val":           res.get("p_val"),
                "cohens_d":        res.get("cohens_d"),
                "n_better":        res.get("n_better"),
                "seeds_done":      len(res.get("seeds", [])),
            })
        results.append(entry)
    summary = _jload(_AR / "summary.json") or {}
    return jsonify({"hypotheses": results, "summary": summary})


# ── frontier problems ─────────────────────────────────────────────────────────
@app.route("/api/frontier")
def api_frontier():
    return jsonify(_frontier_with_directives())


@app.route("/api/research_director")
def api_research_director():
    force_refresh = str(request.args.get("refresh", "") or "").strip().lower() in {"1", "true", "yes"}
    return jsonify(_director_state(force_refresh=force_refresh))


@app.route("/api/human_review")
def api_human_review():
    return jsonify(_human_review_payload())


@app.route("/api/human_review/decision/<path:review_id>", methods=["POST"])
def api_human_review_decision(review_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        updated = record_review_decision(
            _WS,
            review_id=review_id,
            decision=str(payload.get("decision", "") or ""),
            human_notes=str(payload.get("human_notes", "") or ""),
            build_manifest_authorised=bool(payload.get("build_manifest_authorised", False)),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if updated is None:
        return jsonify({"error": f"Unknown review id: {review_id}"}), 404
    return jsonify({"ok": True, "updated": updated, "state": _human_review_payload()})


@app.route("/api/human_review/question/<path:question_id>/recommend", methods=["POST"])
def api_human_review_recommend(question_id: str):
    """Ask Claude to explain TAR's recommended answer for a human review question."""
    from tar_lab.llm_bridge import _api_key as _get_api_key
    api_key = _get_api_key()
    if not api_key:
        return jsonify({"reasoning": None, "error": "ANTHROPIC_API_KEY not configured"}), 503

    # Find the question in human review state
    review_state = load_human_review_state(_WS)
    questions = review_state.get("questions", []) if isinstance(review_state, dict) else []
    question = next((q for q in questions if q.get("question_id") == question_id), None)
    if not question:
        return jsonify({"reasoning": None, "error": f"Question not found: {question_id}"}), 404

    recommended = str(question.get("recommended_default", "") or "")
    question_text = str(question.get("question_text", "") or "")
    question_type = str(question.get("question_type", "") or "")
    options = list(question.get("options", []) or [])
    frontier_id = str(question.get("frontier_problem_id", "") or "")
    why_blocks = str(question.get("why_this_blocks_progress", "") or "")

    # Pull relevant context: experiment queue state for the linked paper/frontier
    eq = _jload(_WS / "tar_state" / "experiment_queue.json") or {}
    experiments = eq.get("experiments", []) if isinstance(eq, dict) else []
    pending_count = sum(1 for e in experiments if e.get("status") == "pending")
    running = next((e for e in experiments if e.get("status") == "running"), None)
    running_desc = f"{running.get('id', '?')} (est. {running.get('estimated_runtime_h', '?')}h)" if running else "none"

    author_state = _jload(_WS / "tar_state" / "author_state.json") or {}
    paper_queue = list(author_state.get("paper_queue", []) or [])
    paper_id_hint = question_id.split(":")[-2] if ":" in question_id else ""
    paper = next((p for p in paper_queue if p.get("project_id", "") == paper_id_hint), None)
    paper_status = paper.get("status", "unknown") if paper else "unknown"

    system = (
        "You are TAR (Thermodynamic Active Research), an autonomous ML research system. "
        "Explain your recommended answer to the human question below. "
        "RULES: (1) Only cite facts from the context provided — no speculation. "
        "(2) State the recommended option clearly. "
        "(3) Explain the specific factual reason for it in 2-3 sentences. "
        "(4) Do not suggest actions beyond the listed options. "
        "Plain English, no markdown headers."
    )
    prompt = (
        f"Question type: {question_type}\n"
        f"Question: {question_text}\n"
        f"Options: {', '.join(options)}\n"
        f"My recommended answer: {recommended or '(none set)'}\n"
        f"Why this blocks progress: {why_blocks}\n"
        f"Frontier: {frontier_id}\n"
        f"Paper status: {paper_status}\n"
        f"Currently running: {running_desc}\n"
        f"Experiments pending in queue: {pending_count}\n\n"
        "Explain concisely why you recommend your answer, grounded only in the facts above."
    )
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return jsonify({"reasoning": msg.content[0].text.strip(), "recommended": recommended, "error": None})
    except Exception as exc:
        return jsonify({"reasoning": None, "recommended": recommended, "error": str(exc)}), 500


@app.route("/api/human_review/question/<path:question_id>/answer", methods=["POST"])
def api_human_review_answer(question_id: str):
    payload = request.get_json(silent=True) or {}
    updated = answer_human_question(
        _WS,
        question_id=question_id,
        answer=str(payload.get("answer", "") or ""),
        answer_notes=str(payload.get("answer_notes", "") or ""),
    )
    if updated is None:
        return jsonify({"error": f"Unknown question id: {question_id}"}), 404
    return jsonify({"ok": True, "updated": updated, "state": _human_review_payload()})


@app.route("/api/validation")
def api_validation():
    return jsonify(_validation_payload())


@app.route("/api/alerts")
def api_alerts():
    try:
        from tar_lab.alerts import load_alerts
        limit = min(int(request.args.get("limit", 100) or 100), 500)
        return jsonify({"alerts": load_alerts(_WS, limit=limit)})
    except Exception as exc:
        return jsonify({"alerts": [], "error": str(exc)})


@app.route("/api/coordination")
def api_coordination():
    return jsonify(_jload(_WS / "tar_state" / "research_coordination_state.json") or {})


@app.route("/api/literature")
def api_literature():
    payload = _jload(_WS / "tar_state" / "literature" / "evidence_ingest_state.json") or {}
    try:
        from tar_evidence_ingest import normalize_literature_payload

        normalized = normalize_literature_payload(payload)
        return jsonify(normalized)
    except Exception:
        return jsonify(payload)


# ── scheduler ─────────────────────────────────────────────────────────────────
@app.route("/api/scheduler")
def _live_gpu_fields() -> dict:
    """Read GPU stats directly from hardware_state.json — always current."""
    hw = _jload(_WS / "tar_state" / "hardware_state.json") or {}
    gpu = hw.get("gpu", {}) if isinstance(hw, dict) else {}
    return {
        "gpu_name":     str(gpu.get("name", "") or ""),
        "gpu_util_pct": int(gpu.get("utilization_pct", 0) or 0),
        "vram_used_gb": float(gpu.get("vram_used_gb", 0.0) or 0.0),
        "vram_total_gb": float(gpu.get("vram_total_gb", 0.0) or 0.0),
        "gpu_temp_c":   int(gpu.get("temperature_c", 0) or 0),
    }


def api_scheduler():
    queue = _runtime_experiment_records()
    state_path = _WS / "tar_state" / "scheduler_state.json"
    state = _jload(state_path) or {}
    # Always patch live GPU fields regardless of cache path taken
    gpu_fields = _live_gpu_fields()
    needs_refresh = (
        not state
        or any(rec.get("status") == "running" for rec in queue)
        or _age_s(state_path) > 30
    )
    if state and not needs_refresh:
        return jsonify({**state, **gpu_fields})
    try:
        from tar_scheduler import TARScheduler

        pending = [SimpleNamespace(**rec) for rec in queue if rec.get("status") == "pending"]
        running = [SimpleNamespace(**rec) for rec in queue if rec.get("status") == "running"]
        sch = TARScheduler(_WS)
        decision = sch.decide(pending_specs=pending, running_specs=running)
        hw = decision.hardware_used
        state = {
            "timestamp": decision.timestamp,
            "rationale": decision.rationale,
            "can_start": decision.can_start,
            "running_ids": decision.running_ids,
            "hardware_used": decision.hardware_used,
            "hardware_available": decision.hardware_available,
            "hold_reasons": [
                {
                    "experiment_id": reason.experiment_id,
                    "experiment_name": reason.experiment_name,
                    "reason": reason.reason,
                }
                for reason in decision.hold_reasons
            ],
            # flat GPU fields the dashboard JS expects
            "gpu_name":     getattr(hw, "gpu_name", ""),
            "gpu_util_pct": getattr(hw, "gpu_util_pct", 0),
            "vram_used_gb": getattr(hw, "vram_used_gb", 0.0),
            "vram_total_gb": getattr(hw, "vram_total_gb", 0.0),
            "gpu_temp_c":   getattr(hw, "gpu_temp_c", 0),
        }
    except Exception:
        state = {}
    return jsonify({**state, **gpu_fields})


# ── author ────────────────────────────────────────────────────────────────────
@app.route("/api/author")
def api_author():
    force_refresh = str(request.args.get("refresh", "") or "").strip().lower() in {"1", "true", "yes"}
    return jsonify(_author_state_payload(force_refresh=force_refresh))


@app.route("/api/experiments/inject", methods=["POST"])
def api_experiment_inject():
    payload = request.get_json(silent=True) or {}
    try:
        from tar_experiment_orchestrator import ExperimentOrchestrator, ExperimentSpec

        seeds_raw = payload.get("seeds", [42, 0, 1])
        if isinstance(seeds_raw, str):
            seeds = [int(part.strip()) for part in seeds_raw.split(",") if part.strip()]
        elif isinstance(seeds_raw, list):
            seeds = [int(part) for part in seeds_raw]
        else:
            seeds = [42, 0, 1]

        overrides = payload.get("config_overrides", {})
        if isinstance(overrides, str):
            overrides = json.loads(overrides or "{}")
        if not isinstance(overrides, dict):
            overrides = {}

        name = str(payload.get("name", "")).strip() or f"human_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        dataset = str(payload.get("dataset", "split_cifar10")).strip()
        hypothesis_name = str(payload.get("hypothesis_name", name)).strip()
        method = str(payload.get("method", "tcl")).strip()
        project_id = str(payload.get("project_id", name.replace(" ", "_"))).strip()

        spec = ExperimentSpec(
            name=name,
            project_id=project_id,
            hypothesis_name=hypothesis_name,
            dataset=dataset,
            method=method,
            seeds=seeds,
            config_overrides=overrides,
            priority=int(payload.get("priority", 50)),
            estimated_runtime_h=float(payload.get("estimated_runtime_h", 6.0)),
            description=str(payload.get("description", "")).strip(),
            frontier_problem_id=str(payload.get("frontier_problem_id", "")).strip(),
            author_paper_id=str(payload.get("author_paper_id", "")).strip(),
            observer_class_name=str(payload.get("observer_class_name", "")).strip(),
            depends_on=payload.get("depends_on", []) if isinstance(payload.get("depends_on", []), list) else [],
        )
        orch = ExperimentOrchestrator(_WS)
        created = orch.submit(spec)
        return jsonify({"ok": True, "experiment_id": created.id, "name": created.name})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/processes")
def api_processes():
    hw_path = _WS / "tar_state" / "hardware_state.json"
    hw = _jload(hw_path) or {}
    needs_refresh = (
        not hw
        or _age_s(hw_path) > 15
        or not hw.get("processes")
    )
    if needs_refresh:
        try:
            from tar_hardware_monitor import take_snapshot, write_snapshot
            hw = take_snapshot(_WS)
            write_snapshot(_WS, hw)
        except Exception:
            hw = hw or {}
    registry = _jload(_WS / "tar_state" / "process_registry.json") or {}
    return jsonify({
        "timestamp": hw.get("timestamp", ""),
        "processes": hw.get("processes", []),
        "registry": registry,
    })


# ── registry ──────────────────────────────────────────────────────────────────
@app.route("/api/registry")
def api_registry():
    data = _jload(_WS / "tar_state" / "project_registry.json") or {}
    projects = [
        {
            "id":           pid,
            "name":         rec.get("name", ""),
            "field":        rec.get("field", ""),
            "subfield":     rec.get("subfield", ""),
            "status":       rec.get("status", ""),
            "dataset":      rec.get("dataset", ""),
            "phase_source": rec.get("phase_source", ""),
            "has_pdf":      bool(rec.get("paper_pdf", "")),
            "has_tex":      bool(rec.get("paper_dir", "")) and (Path(rec.get("paper_dir", "")) / "main.tex").exists(),
            "readiness":    rec.get("readiness", ""),
            "paper_status": rec.get("paper_status", ""),
            "director_truth_status": rec.get("director_truth_status", ""),
            "director_recommendation": rec.get("director_recommendation", "")[:140],
            "waiting_for_experiment_ids": rec.get("waiting_for_experiment_ids", []),
            "frontier_problem_ids": rec.get("frontier_problem_ids", []),
            "director_priority_score": rec.get("director_priority_score", 0.0),
            "abstract":     rec.get("abstract", "")[:100],
            "created_at":   rec.get("created_at", "")[:16],
        }
        for pid, rec in data.items()
    ]
    status_rank = {"running": 0, "planned": 1, "pending": 1, "complete": 2, "failed": 3, "archived": 4}
    projects.sort(key=lambda x: (
        status_rank.get(x["status"], 9),
        -float(x.get("director_priority_score", 0.0) or 0.0),
        x["created_at"],
    ))
    return jsonify(projects)


# ── papers ────────────────────────────────────────────────────────────────────
@app.route("/api/papers")
def api_papers():
    papers = []
    log_path = _WS / "tar_state" / "papers" / "papers.jsonl"
    seen_ids: set[str] = set()
    seen_serve: set[str] = set()
    author_state = _jload(_WS / "tar_state" / "author_state.json") or {}
    queue_entries = author_state.get("paper_queue", []) if isinstance(author_state, dict) else []
    queue_by_id = {
        str(entry.get("project_id", "") or ""): entry
        for entry in queue_entries
        if isinstance(entry, dict) and entry.get("project_id")
    }
    current_paper = author_state.get("current_paper", {}) if isinstance(author_state, dict) else {}
    registry_data = _jload(_WS / "tar_state" / "project_registry.json") or {}

    def _resolve_paper_artifacts(project_id: str, rec: dict | None = None) -> tuple[str, str]:
        rec = rec or {}
        queue_entry = queue_by_id.get(project_id, {})
        registry_rec = registry_data.get(project_id, {}) if isinstance(registry_data, dict) else {}

        candidates_pdf: list[Path] = []
        candidates_tex: list[Path] = []

        for raw in (
            rec.get("pdf_path"),
            registry_rec.get("paper_pdf"),
        ):
            if raw:
                candidates_pdf.append(Path(str(raw)))
        for raw in (
            rec.get("tex_path"),
        ):
            if raw:
                candidates_tex.append(Path(str(raw)))

        def _named_pdf_candidates(paper_dir: Path, pid: str) -> list[Path]:
            safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", pid)[:80]
            return [paper_dir / f"{safe}.pdf", paper_dir / "main.pdf"]

        for raw_dir in (
            queue_entry.get("paper_dir"),
            registry_rec.get("paper_dir"),
        ):
            if raw_dir:
                paper_dir = Path(str(raw_dir))
                candidates_pdf.extend(_named_pdf_candidates(paper_dir, project_id))
                candidates_tex.append(paper_dir / "main.tex")

        if isinstance(current_paper, dict) and str(current_paper.get("project_id", "") or "") == project_id:
            raw_dir = current_paper.get("paper_dir")
            if raw_dir:
                paper_dir = Path(str(raw_dir))
                candidates_pdf.extend(_named_pdf_candidates(paper_dir, project_id))
                candidates_tex.append(paper_dir / "main.tex")
            raw_tex = current_paper.get("tex_path")
            if raw_tex:
                candidates_tex.append(Path(str(raw_tex)))

        pdf = next((str(path) for path in candidates_pdf if path.exists()), "")
        tex = next((str(path) for path in candidates_tex if path.exists()), "")
        return pdf, tex

    if log_path.exists():
        recs_by_id: dict[str, dict] = {}
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                pid = rec.get("project_id", "")
                if not pid:
                    continue
                if rec.get("pdf_path") or pid not in recs_by_id:
                    recs_by_id[pid] = rec
            except Exception:
                pass
        for pid, rec in recs_by_id.items():
            seen_ids.add(pid)
            pdf, tex = _resolve_paper_artifacts(pid, rec)
            sp  = _serve_path_for(pdf)
            if sp:
                seen_serve.add(sp)
            papers.append({
                "project_id":      pid,
                "title":           rec.get("title", pid),
                "verdict":         rec.get("verdict", ""),
                "status":          "draft_compiled" if pdf else "planning",
                "has_pdf":         bool(pdf),
                "serve_pdf":       sp,
                "serve_tex":       _serve_path_for(tex),
                "generated_at":    rec.get("generated_at", "")[:16],
                "mean_forgetting": rec.get("mean_forgetting"),
                "mean_delta":      rec.get("mean_delta"),
                "p_val":           rec.get("p_val"),
                "progress_pct":    100 if pdf else 96,
                "progress_label":  "PDF compiled" if pdf else "Draft complete",
                "compile_status":  rec.get("compile_status", "draft_compiled" if pdf else ""),
            })

    for root in [_REPO / "paper", _WS / "paper"]:
        if not root.exists():
            continue
        # Prefer named PDFs (project_id.pdf) over main.pdf; skip main.pdf if named exists
        # Exclude archive folders
        _seen_paper_dirs: set[Path] = set()
        for pdf in sorted(root.rglob("*.pdf")):
            if any(part.startswith("_archive") for part in pdf.parts):
                continue
            if pdf.parent in _seen_paper_dirs:
                continue
            proj = pdf.parent.name
            tex = pdf.parent / "main.tex"
            pid = f"paper-{proj}"
            if pid in seen_ids:
                continue
            # Prefer a named PDF (not main.pdf) if one exists alongside main.pdf
            named_candidates = [p for p in pdf.parent.glob("*.pdf") if p.name != "main.pdf"]
            best_pdf = named_candidates[0] if named_candidates else pdf
            rel = str(best_pdf.relative_to(root)).replace("\\", "/")
            if rel in seen_serve:
                continue
            _seen_paper_dirs.add(pdf.parent)
            seen_ids.add(pid)
            seen_serve.add(rel)
            papers.append({
                "project_id":      pid,
                "title":           proj.replace("-", " ").replace("_", " ").title(),
                "verdict":         "",
                "status":          "draft_compiled",
                "has_pdf":         True,
                "serve_pdf":       rel,
                "serve_tex":       str(tex.relative_to(root)).replace("\\", "/") if tex.exists() else "",
                "generated_at":    datetime.fromtimestamp(best_pdf.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "mean_forgetting": None, "mean_delta": None, "p_val": None,
                "progress_pct":    100,
                "progress_label":  "PDF compiled",
                "compile_status":  "draft_compiled",
            })

    if isinstance(registry_data, dict):
        for pid, rec in registry_data.items():
            if pid in seen_ids:
                continue
            paper_dir = Path(str(rec.get("paper_dir", "") or "")) if rec.get("paper_dir") else None
            tex_path = paper_dir / "main.tex" if paper_dir else None
            pdf_path = Path(str(rec.get("paper_pdf", "") or "")) if rec.get("paper_pdf") else (paper_dir / "main.pdf" if paper_dir else None)
            serve_pdf = _serve_path_for(str(pdf_path)) if pdf_path and pdf_path.exists() else ""
            serve_tex = _serve_path_for(str(tex_path)) if tex_path and tex_path.exists() else ""
            # Only show registry-sourced entries that have a compiled PDF on disk
            if not serve_pdf:
                continue
            if serve_pdf:
                seen_serve.add(serve_pdf)
            if serve_tex:
                seen_serve.add(serve_tex)
            seen_ids.add(pid)
            papers.append({
                "project_id":      pid,
                "title":           rec.get("name", pid),
                "verdict":         "",
                "status":          rec.get("status", "planned"),
                "plan_status":     rec.get("paper_status", rec.get("readiness", rec.get("status", "planned"))),
                "readiness":       rec.get("readiness", ""),
                "truth_status":    rec.get("director_truth_status", "weak"),
                "recommendation":  rec.get("director_recommendation", ""),
                "has_pdf":         bool(pdf_path and pdf_path.exists()),
                "serve_pdf":       serve_pdf,
                "serve_tex":       serve_tex,
                "generated_at":    (rec.get("updated_at") or rec.get("created_at") or "")[:16],
                "mean_forgetting": None,
                "mean_delta":      None,
                "p_val":           None,
                "progress_pct":    None,
                "progress_label":  "",
                "compile_status":  "",
                "waiting_for_experiments": [],
                "complete_count":  0,
                "running_count":   0,
                "pending_count":   0,
                "total_experiments": 0,
            })

    for paper in papers:
        pid = str(paper.get("project_id", "") or "")
        queue_entry = queue_by_id.get(pid, {})
        if isinstance(current_paper, dict) and str(current_paper.get("project_id", "") or "") == pid:
            queue_entry = {**queue_entry, **current_paper}
        progress = queue_entry.get("paper_progress") or queue_entry.get("progress") or current_paper.get("progress", {})
        if isinstance(progress, dict):
            if progress.get("pct") is not None:
                paper["progress_pct"] = progress.get("pct")
            if progress.get("label"):
                paper["progress_label"] = progress.get("label")
            if progress.get("compile_status"):
                paper["compile_status"] = progress.get("compile_status")
        if queue_entry.get("status"):
            paper["plan_status"] = queue_entry.get("status")
        if isinstance(current_paper, dict) and str(current_paper.get("project_id", "") or "") == pid and current_paper.get("status"):
            paper["plan_status"] = current_paper.get("status")
        if queue_entry.get("readiness") and not paper.get("readiness"):
            paper["readiness"] = queue_entry.get("readiness")
        if queue_entry.get("director_recommendation") and not paper.get("recommendation"):
            paper["recommendation"] = queue_entry.get("director_recommendation")
        if queue_entry.get("truth_status") and not paper.get("truth_status"):
            paper["truth_status"] = queue_entry.get("truth_status")
        if queue_entry.get("revision_reason"):
            paper["revision_reason"] = queue_entry.get("revision_reason")
        waits = queue_entry.get("waiting_for_experiments", [])
        if isinstance(waits, list):
            paper["waiting_for_experiments"] = [str(item) for item in waits if str(item)]
        for key in ("complete_count", "running_count", "pending_count"):
            if queue_entry.get(key) is not None:
                try:
                    paper[key] = int(queue_entry.get(key) or 0)
                except Exception:
                    pass
        progress_counts = queue_entry.get("progress", {})
        if isinstance(progress_counts, dict) and progress_counts.get("total") is not None:
            try:
                paper["total_experiments"] = int(progress_counts.get("total") or 0)
            except Exception:
                paper["total_experiments"] = 0
        else:
            paper["total_experiments"] = int(paper.get("complete_count", 0) or 0) + int(paper.get("running_count", 0) or 0) + int(paper.get("pending_count", 0) or 0)
        if paper.get("waiting_for_experiments"):
            status_now = str(paper.get("plan_status", "") or "")
            readiness_now = str(paper.get("readiness", "") or "")
            if status_now in {"planned", "ready", ""} or readiness_now in {"hold", "experiment_first", "blocked"}:
                paper["plan_status"] = "blocked"
        effective_plan_status = str(paper.get("plan_status", "") or "")
        compile_status = str(paper.get("compile_status", "") or "")
        if effective_plan_status and effective_plan_status != "published":
            paper["status"] = effective_plan_status
        if paper.get("waiting_for_experiments"):
            paper["status"] = "blocked"
        elif compile_status == "draft_compiled":
            paper["status"] = "draft_compiled"

    status_rank = {"running": 0, "blocked": 1, "planned": 2, "pending": 2, "draft_compiled": 3, "complete": 4, "failed": 5}
    papers.sort(key=lambda x: (
        status_rank.get(str(x.get("status", "")), 9),
        x.get("generated_at", ""),
    ), reverse=False)
    return jsonify(papers)


@app.route("/api/paper/return-to-author/<project_id>", methods=["POST"])
def api_return_paper_to_author(project_id: str):
    project_id = str(project_id or "").strip()
    if not project_id:
        abort(400)

    payload = request.get_json(silent=True) or {}
    reason = str(payload.get("reason", "") or "").strip()

    try:
        from tar_author import request_paper_revision

        result = request_paper_revision(_WS, project_id, reason=reason)
    except Exception as exc:
        return jsonify({
            "ok": False,
            "project_id": project_id,
            "error": str(exc),
        }), 500

    if not result:
        abort(404)

    worker = threading.Thread(
        target=_run_paper_revision_async,
        args=(project_id, reason),
        daemon=True,
        name=f"paper-revision-{project_id[:20]}",
    )
    worker.start()
    return jsonify({
        "ok": True,
        "project_id": project_id,
        "status": "revision_requested",
        "reason": reason,
        "background_started": True,
    })


# ── website publication ───────────────────────────────────────────────────────
_PUBLISH_ROOT        = _REPO.parent
_STAGED_JSON         = _PUBLISH_ROOT / "staged_papers.json"
_WEBSITE_JSON        = _PUBLISH_ROOT / "website" / "data" / "papers.json"
_WEBSITE_RESEARCH_JSON = _PUBLISH_ROOT / "website" / "data" / "research.json"


@app.route("/api/website_papers")
def api_website_papers():
    staged = []
    live   = []
    if _STAGED_JSON.exists():
        try:
            data = json.loads(_STAGED_JSON.read_text(encoding="utf-8"))
            staged = data.get("staged", [])
        except Exception:
            pass
    if _WEBSITE_JSON.exists():
        try:
            data = json.loads(_WEBSITE_JSON.read_text(encoding="utf-8"))
            live = data.get("papers", [])
        except Exception:
            pass
    return jsonify({"staged": staged, "live": live})


@app.route("/api/website/sync_research", methods=["POST", "GET"])
def api_website_sync_research():
    """Regenerate website/data/research.json from live Director state."""
    import re as _re
    try:
        director_state = _jload(_WS / "tar_state" / "research_director_state.json") or {}
        paths = director_state.get("active_research_paths", [])
        if not isinstance(paths, list):
            paths = []

        status_map = {"pursue_now": "active", "pursue_next": "queued", "investigate": "investigating"}
        evidence_map = {"strong": "strong", "moderate": "moderate", "weak": "weak"}

        items = []
        for path in paths:
            if not isinstance(path, dict):
                continue
            title = str(path.get("title", "") or "").strip()
            if not title:
                continue
            why = str(path.get("why_this_now", "") or "")
            m = _re.search(r"(\d+) complete", why)
            exp_count = int(m.group(1)) if m else 0
            allowed_topics = list(path.get("allowed_topics", []) or [])
            description = str(allowed_topics[2]).strip() if len(allowed_topics) > 2 else ""
            items.append({
                "title": title,
                "status": status_map.get(str(path.get("status", "") or ""), "investigating"),
                "frontier_id": str(path.get("target_frontier_problem_id", "") or ""),
                "description": description[:300],
                "experiment_count": exp_count,
                "paper_title": str(path.get("target_paper_id", "") or ""),
                "evidence_strength": evidence_map.get(
                    str(path.get("evidence_strength", "") or ""), "moderate"
                ),
                "updated": datetime.utcnow().strftime("%Y-%m-%d"),
            })

        out = {"research": items, "updated_at": datetime.utcnow().isoformat() + "Z"}
        _WEBSITE_RESEARCH_JSON.write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return jsonify({"ok": True, "count": len(items)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/website/approve/<paper_id>", methods=["POST"])
def api_website_approve(paper_id: str):
    paper_id = str(paper_id or "").strip()
    if not paper_id:
        return jsonify({"ok": False, "error": "paper_id required"}), 400
    try:
        import sys as _sys
        if str(_PUBLISH_ROOT) not in _sys.path:
            _sys.path.insert(0, str(_PUBLISH_ROOT))
        from publish_paper import approve_paper_headless
        result = approve_paper_headless(paper_id)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


# ── breakthroughs ─────────────────────────────────────────────────────────────
@app.route("/api/breakthroughs")
def api_breakthroughs():
    bks = []
    seen_bk_ids: set[str] = set()

    if _AR.exists():
        for name in ["deep_anchor", "graduated_penalty", "strict_consolidation",
                     "thermal_carryover", "high_penalty_conservative"]:
            full = _jload(_AR / f"{name}.json")
            if not full:
                continue
            res = full.get("result", {})
            if res.get("verdict") != "BREAKTHROUGH":
                continue
            bid = f"ar-{name}"
            if bid in seen_bk_ids:
                continue
            seen_bk_ids.add(bid)
            forg_list = res.get("mechanism_forgetting", [])
            paper_pdf = _REPO / "paper" / name / "main.pdf"
            bks.append({
                "source": "autonomous_research",
                "project_id": f"tcl-{name.replace('_','-')}-cifar10-v1",
                "name": name.replace("_", " ").title(),
                "verdict": "BREAKTHROUGH",
                "mean_delta": res.get("mean_delta"),
                "p_val": res.get("p_val"),
                "cohens_d": res.get("cohens_d"),
                "n_better": res.get("n_better"),
                "mean_forgetting": sum(forg_list)/max(len(forg_list),1) if forg_list else None,
                "dataset": "split_cifar10",
                "serve_pdf": _serve_path_for(str(paper_pdf)) if paper_pdf.exists() else "",
                "notes": res.get("notes", ""),
                "found_at": res.get("run_at", "")[:16],
            })

    if _COMP.exists():
        covered_search_ids: set[str] = set()
        for ph_path in sorted(_COMP.glob("phase*.json")):
            ph_data = _jload(ph_path) or {}
            if ph_data.get("external_breakthrough_candidate") or ph_data.get("verdict_key") == "BREAKTHROUGH":
                sid = ph_data.get("search_id", "")
                if sid:
                    covered_search_ids.add(sid)

        for p in sorted(_COMP.glob("phase*.json")):
            data = _jload(p) or {}
            is_bk = (data.get("verdict_key") == "BREAKTHROUGH"
                     or data.get("external_breakthrough_candidate") is True)
            if not is_bk:
                continue
            m = re.search(r"phase(\d+)", p.stem)
            num = int(m.group(1)) if m else 0
            bid = f"phase{num}"
            if bid in seen_bk_ids:
                continue
            seen_bk_ids.add(bid)
            best_cand = max(data.get("candidates", []),
                            key=lambda c: -c.get("mean_forgetting", 9), default={})
            bks.append({
                "source": "phase_result",
                "project_id": bid,
                "name": f"Phase {num} — {data.get('benchmark') or data.get('dataset','')}",
                "verdict": "BREAKTHROUGH",
                "mean_delta": (data.get("pairwise", {}).get("ewc", {}).get("mean_delta")
                               or best_cand.get("delta_vs_sgd")),
                "p_val": (data.get("pairwise", {}).get("ewc", {}).get("p_val")
                          or data.get("p_value_vs_strong_baseline")),
                "cohens_d": (data.get("pairwise", {}).get("ewc", {}).get("cohens_d")
                             or data.get("effect_size_vs_strong_baseline")),
                "n_better": data.get("pairwise", {}).get("ewc", {}).get("n_tcl_better"),
                "mean_forgetting": ((data.get("aggregate", {}).get("tcl", {}) or {}).get("forgetting_mean")
                                    or best_cand.get("mean_forgetting")),
                "dataset": data.get("benchmark") or data.get("dataset", ""),
                "serve_pdf": "",
                "notes": (data.get("verdict") or data.get("summary") or "")[:140],
                "found_at": (data.get("completed_at") or data.get("created_at") or "")[:16],
            })

        seen_cand_keys: set[tuple] = set()
        for p in sorted(_COMP.glob("external-breakthrough-*.json")):
            data = _jload(p) or {}
            p_val_raw = data.get("p_value") or data.get("p_val")
            if p_val_raw is None or float(p_val_raw) >= 0.05:
                continue
            search_id = p.stem.replace("external-breakthrough-", "")
            if search_id in covered_search_ids:
                continue
            search_data = _jload(_COMP / f"{search_id}.json") or {}
            cand_name = search_data.get("best_candidate_name", "")
            benchmark = data.get("benchmark", "")
            cand_key = (cand_name, benchmark)
            if cand_key in seen_cand_keys:
                continue
            seen_cand_keys.add(cand_key)
            bid = data.get("assessment_id", p.stem)
            if bid in seen_bk_ids:
                continue
            seen_bk_ids.add(bid)
            best_cand = max(search_data.get("candidates", []),
                            key=lambda c: -c.get("mean_forgetting", 9), default={})
            bks.append({
                "source": "external_breakthrough",
                "project_id": data.get("problem_id") or bid,
                "name": (cand_name or bid).replace("_", " ").replace("-", " ").title(),
                "verdict": "BREAKTHROUGH",
                "mean_delta": best_cand.get("delta_vs_sgd") or search_data.get("best_delta_vs_strong_baseline"),
                "p_val": p_val_raw,
                "cohens_d": data.get("effect_size") or search_data.get("effect_size_vs_strong_baseline"),
                "n_better": None,
                "mean_forgetting": best_cand.get("mean_forgetting"),
                "dataset": benchmark,
                "serve_pdf": "",
                "notes": (data.get("summary") or "")[:140],
                "found_at": data.get("created_at", "")[:16],
            })

    bks.sort(key=lambda x: x.get("found_at", ""), reverse=True)
    for idx, bk in enumerate(bks, start=1):
        found_at = str(bk.get("found_at", "") or "")
        project_id = str(bk.get("project_id", "") or f"bk-{idx}")
        bk["notification_id"] = f"{bk.get('source', 'bk')}::{project_id}::{found_at}"
        bk["summary"] = (
            f"{bk.get('name', 'Breakthrough')} on {str(bk.get('dataset', '')).replace('split_', '')} "
            f"with p={bk.get('p_val', '—')} and d={bk.get('cohens_d', '—')}."
        )
        bk["evidence_strength"] = "strong" if bk.get("p_val") not in (None, "") else "moderate"
    return jsonify({"count": len(bks), "breakthroughs": bks})


# ── GPT narration + live replication data ────────────────────────────────────
_NARRATION_CACHE: dict = {}
_NARRATION_TTL = 90  # seconds — avoid hammering the API


def _find_replication_obs_root() -> "Path | None":
    val_root = _WS / "tar_state" / "validation"
    if not val_root.exists():
        return None
    candidates = sorted(
        [d for d in val_root.iterdir() if d.is_dir() and "hpc_claim_validation" in d.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        obs = c / "outputs" / "replication_suite" / "observability" / "per_method"
        if obs.exists():
            return obs
    return None


_EPOCH_SEED_RE = re.compile(
    r"seed=(\d+)\s+\[(\w+)\]\s+forgetting=([\d.]+)", re.IGNORECASE
)
_LOG_LINE_TS_RE = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\]')


def _parse_epoch_log_seed_progress(running: dict) -> dict:
    """
    Parse the epoch log for any running director experiment to count completed
    seeds and extract per-seed forgetting values.

    Director experiments don't write seeds_done back to experiment_queue.json,
    so we read the terminal lines directly.  Works for any experiment:
      1. Prefers the experiment's own runtime log directory (most specific).
      2. Falls back to the watchdog log, filtered to lines after the most recent
         occurrence of the experiment ID (so old experiments don't bleed through).

    Returns {seeds_done, forgetting_so_far, seeds_seen}.
    A seed is counted as done when the primary method (TCL, or most frequent)
    has reported a forgetting value for it.
    """
    exp_id = str(running.get("id", "") or "")

    # 1. Try experiment-specific runtime log first
    runtime_ctx = running.get("runtime_context", {}) if isinstance(running.get("runtime_context"), dict) else {}
    runtime_dir = Path(str(runtime_ctx.get("current_runtime_dir", "") or "")) if runtime_ctx.get("current_runtime_dir") else None
    exp_lines: list[str] = []
    if runtime_dir and runtime_dir.exists():
        runtime_logs_dir = runtime_dir / "logs"
        if runtime_logs_dir.exists():
            candidates = sorted(runtime_logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            for lp in candidates[:3]:
                exp_lines.extend(_tail(lp, 400))
            if exp_lines:
                return _extract_seed_progress_from_lines(exp_lines)

    # 2. Fall back to watchdog / living_research log, trimmed to after this exp_id
    for log_path in [
        _latest_matching_log("watchdog-living-research-*.log"),
        _WS / "tar_state" / "living_research.log",
    ]:
        if log_path is None or not log_path.exists():
            continue
        all_lines = _tail(log_path, 1000)
        if exp_id:
            # Find the last line that mentions this experiment, start reading from there
            anchor = -1
            for idx, line in enumerate(all_lines):
                if exp_id in line:
                    anchor = idx
            trimmed = all_lines[anchor:] if anchor >= 0 else all_lines[-600:]
        else:
            trimmed = all_lines[-600:]
        result = _extract_seed_progress_from_lines(trimmed)
        if result["seeds_done"] > 0:
            return result

    return {"seeds_done": 0, "forgetting_so_far": [], "seeds_seen": 0}


def _extract_seed_progress_from_lines(lines: list[str]) -> dict:
    """Parse seed=N [method] forgetting=X lines from a list of log strings."""
    seed_results: dict[int, dict[str, float]] = {}
    for line in lines:
        m = _EPOCH_SEED_RE.search(line)
        if m:
            seed_num   = int(m.group(1))
            method     = m.group(2).lower()
            forgetting = float(m.group(3))
            seed_results.setdefault(seed_num, {})[method] = forgetting

    if not seed_results:
        return {"seeds_done": 0, "forgetting_so_far": [], "seeds_seen": 0}

    # Primary method: prefer tcl, else the most frequently seen
    all_methods: dict[str, int] = {}
    for methods in seed_results.values():
        for meth in methods:
            all_methods[meth] = all_methods.get(meth, 0) + 1
    primary = "tcl" if "tcl" in all_methods else max(all_methods, key=lambda k: all_methods[k])

    completed = sorted(seed for seed, ms in seed_results.items() if primary in ms)
    return {
        "seeds_done": len(completed),
        "forgetting_so_far": [seed_results[s][primary] for s in completed],
        "seeds_seen": len(seed_results),
    }


def _build_narration_context() -> dict:
    eq = _jload(_WS / "tar_state" / "experiment_queue.json") or {}
    experiments = eq.get("experiments", []) if isinstance(eq, dict) else []
    running = next((e for e in experiments if e.get("status") == "running"), None)
    if not running:
        return {
            "prompt": None,
            "fallback": "TAR is currently idle — no active experiment is running.",
            "confidence": 0, "focus": "idle",
            "seed_trail": [], "seeds_total": 0, "threshold": 0.1161,
        }

    exp_id = str(running.get("id", "") or "")
    is_validation_suite = running.get("method", "") == "validation_suite" or "claim_validation" in exp_id

    if is_validation_suite:
        # ── HPC replication suite narration ───────────────────────────────────
        prog = running.get("progress", {})
        ctx = running.get("context", {})
        cfg = running.get("config_overrides", {})
        forgetting_so_far = prog.get("forgetting_so_far", [])
        threshold = 0.1175
        seeds_passing = sum(1 for f in forgetting_so_far if f < threshold)
        confidence = int(100 * seeds_passing / len(forgetting_so_far)) if forgetting_so_far else 0
        hpc_agg: dict = {}; tcl_agg: dict = {}; hpc_n = 0; tcl_n = 0
        obs = _find_replication_obs_root()
        if obs:
            hd = _jload(obs / "high_penalty_conservative.json") or {}
            td = _jload(obs / "tcl_baseline.json") or {}
            hpc_agg = hd.get("aggregate_so_far", {}); hpc_n = hd.get("completed_seed_count", 0)
            tcl_agg = td.get("aggregate_so_far", {}); tcl_n = td.get("completed_seed_count", 0)
        hf = hpc_agg.get("forgetting_mean"); ha = hpc_agg.get("acc_mean"); tf = tcl_agg.get("forgetting_mean")
        claim = (cfg.get("target_claim") or ctx.get("hypothesis") or running.get("description", ""))[:130]
        cur_method = (prog.get("current_method") or "").replace("_", " ")
        cur_seed = prog.get("current_seed", "unknown")
        sd = prog.get("seeds_done", 0); st = len(running.get("seeds", [])) or 10
        md = prog.get("methods_done", 0); mt = prog.get("methods_total", 60)
        trail_fmt = ", ".join(f"{f:.4f}" for f in forgetting_so_far)
        prompt = (
            "You are TAR (Thermodynamic Active Research), an autonomous ML research system. "
            "You are running a multi-method replication suite on Split-CIFAR-10 with ResNet-18 comparing "
            "HPC, TCL, EWC, SI, and SGD. Speak in first person as a thoughtful researcher. "
            "Be specific with numbers. 2-3 sentences, plain conversational English, no markdown.\n\n"
            f"State:\n"
            f"- Seeds complete: {sd}/{st} · Method-runs complete: {md}/{mt}\n"
            f"- Now running: method={cur_method or 'unknown'}, seed={cur_seed}\n"
            f"- Claim: \"{claim}\"\n"
            f"- Pre-registered threshold: HPC forgetting must be < {threshold}\n"
            f"- HPC so far (n={hpc_n}): "
            f"forgetting={round(hf, 4) if hf is not None else 'N/A'}, "
            f"acc={round(ha * 100, 1) if ha is not None else 'N/A'}%\n"
            f"- TCL baseline (n={tcl_n}): forgetting={round(tf, 4) if tf is not None else 'N/A'}\n"
            f"- Per-seed HPC forgetting trail: [{trail_fmt or 'none yet'}]\n"
            f"- Seeds passing threshold: {seeds_passing}/{len(forgetting_so_far)}\n"
        )
        return {
            "prompt": prompt, "confidence": confidence,
            "focus": f"method={cur_method or '?'} · seed={cur_seed}",
            "seed_trail": forgetting_so_far, "seeds_total": st,
            "threshold": threshold, "seeds_done": sd,
        }

    # ── Scale-up suite narration ───────────────────────────────────────────────
    is_scale_up_suite = (
        running.get("runner_key") in {"phase16_scale_up_suite", "phase17_tinyimagenet_suite"}
        or running.get("hypothesis_name") == "scale_up_validation"
        or "scale_up" in (running.get("tags") or [])
    )

    if is_scale_up_suite:
        import datetime as _dt
        prog = running.get("progress", {})
        seeds_done = int(prog.get("seeds_done", 0) or 0)
        seeds_total = int(prog.get("seeds_total", 3) or 3)
        forgetting_so_far = list(prog.get("forgetting_so_far") or [])
        dataset = str(running.get("dataset", "Tiny-ImageNet") or "Tiny-ImageNet")
        description = str(running.get("description", "") or "")[:200]
        started_str = str(running.get("started_at", "") or "")
        elapsed_h = 0.0
        est_h = float(running.get("estimated_runtime_h", 36.0) or 36.0)
        if started_str:
            try:
                started_dt = _dt.datetime.fromisoformat(started_str.replace("Z", "+00:00"))
                elapsed_h = (_dt.datetime.now(_dt.timezone.utc) - started_dt).total_seconds() / 3600
            except Exception:
                pass
        remaining_h = max(0.0, est_h - elapsed_h)
        mean_f = (sum(forgetting_so_far) / len(forgetting_so_far)) if forgetting_so_far else None
        trail_fmt = ", ".join(f"{f:.4f}" for f in forgetting_so_far)
        confidence = round(100 * seeds_done / seeds_total) if seeds_total else 0
        prompt = (
            "You are TAR (Thermodynamic Active Research), an autonomous ML research system. "
            "Speak in first person as a thoughtful researcher. "
            "2-3 sentences, plain conversational English, no markdown, no bullet points.\n\n"
            f"You are running a large-scale validation suite ({exp_id}) on {dataset} "
            f"to test whether your thermodynamic continual learning approach scales beyond Split-CIFAR-10.\n"
            f"Seeds complete: {seeds_done}/{seeds_total}\n"
            f"Per-seed forgetting so far: [{trail_fmt or 'none yet'}]\n"
            f"Mean forgetting so far: {round(mean_f, 4) if mean_f is not None else 'N/A'}\n"
            f"Experiment: {description}\n"
            f"Estimated time remaining: ~{remaining_h:.0f}h\n"
            "Narrate what you are observing about scale-up performance and why this matters."
        )
        return {
            "prompt": prompt,
            "confidence": confidence,
            "focus": f"scale-up · {dataset} · {seeds_done}/{seeds_total} seeds",
            "seed_trail": forgetting_so_far,
            "seeds_total": seeds_total,
            "threshold": 0.0,
            "seeds_done": seeds_done,
        }

    # ── Director / probe experiment narration ─────────────────────────────────
    sched = _jload(_WS / "tar_state" / "scheduler_state.json") or {}
    gpu_name  = str(sched.get("gpu_name", "GPU") or "GPU")
    gpu_util  = sched.get("gpu_util_pct", 0) or 0
    vram_used = sched.get("vram_used_gb", 0) or 0
    vram_tot  = sched.get("vram_total_gb", 0) or 0
    gpu_temp  = sched.get("gpu_temp_c", 0) or 0
    rationale = str(sched.get("rationale", "") or "")[:300]

    frontier_id    = str(running.get("frontier_problem_id", "") or running.get("director_active_path_id", ""))
    ctx_block      = running.get("context", {}) if isinstance(running.get("context"), dict) else {}
    # Use human-readable description, not internal director tags
    frontier_title = str(
        running.get("description", "")
        or ctx_block.get("why", "")
        or running.get("name", "")
        or exp_id
    ).replace("Director-selected ", "").replace("Director Follow-up - ", "").strip()
    # Use actual hypothesis text, not internal key names like "director_sigma_probe"
    hypothesis = str(
        ctx_block.get("hypothesis", "")
        or running.get("description", "")
        or ""
    )
    dataset        = str(running.get("dataset", "") or "")
    method         = str(running.get("method", "") or "").upper()
    seeds          = running.get("seeds", []) or []
    seeds_total    = len(seeds)
    epochs         = running.get("epochs", 0) or 0
    queued_count   = sum(1 for e in experiments if e.get("status") == "pending")

    # Pick up live seed progress — first try the structured progress dict,
    # then fall back to parsing the epoch log directly (director experiments
    # don't write seeds_done back to experiment_queue.json).
    prog           = running.get("progress", {}) or {}
    seeds_done     = int(prog.get("seeds_done", 0) or 0)
    forgetting_so_far = list(prog.get("forgetting_so_far") or [])
    if not seeds_total and prog.get("seeds_total"):
        seeds_total = int(prog.get("seeds_total", 0))

    if seeds_done == 0:
        _log_progress = _parse_epoch_log_seed_progress(running)
        if _log_progress["seeds_done"] > 0:
            seeds_done        = _log_progress["seeds_done"]
            forgetting_so_far = _log_progress["forgetting_so_far"]
            if not seeds_total and _log_progress.get("seeds_seen"):
                seeds_total = max(seeds_total, _log_progress["seeds_seen"])

    confidence = round(100 * seeds_done / seeds_total) if seeds_total else 0

    trail_note = ""
    if forgetting_so_far:
        mean_f = sum(forgetting_so_far) / len(forgetting_so_far)
        trail_note = (
            f"Seeds complete: {seeds_done}/{seeds_total} · "
            f"Forgetting per seed: {[round(f, 4) for f in forgetting_so_far]} · "
            f"Mean forgetting: {mean_f:.4f}\n"
        )

    prompt = (
        "You are TAR (Thermodynamic Active Research), an autonomous ML research system. "
        "Speak in first person as a thoughtful researcher. "
        "2-3 sentences, plain conversational English, no markdown, no bullet points.\n\n"
        "IMPORTANT: TAR is the research SYSTEM running this experiment. "
        f"The SUBJECT being studied is the {method or 'ML'} algorithm — do not confuse TAR's "
        "own components (director, scheduler, orchestrator) with what is being tested.\n\n"
        f"Research question: {frontier_title}\n"
        f"Hypothesis: {hypothesis or 'probe experiment'}\n"
        f"Dataset: {dataset or 'unknown'} · Method under study: {method or 'unknown'} · "
        f"Seeds: {seeds_total} · Epochs per seed: {epochs}\n"
        f"{trail_note}"
        f"GPU: {gpu_name} at {gpu_util}% utilisation, "
        f"{vram_used:.1f}/{vram_tot:.1f} GB VRAM, {gpu_temp}°C\n"
        f"{queued_count} experiment(s) queued behind this one.\n"
        "Narrate what you are doing right now and why it matters to the research. "
        "Refer to the method being studied by name, not as 'the director' or TAR's own systems."
    )
    focus_label = f"{method or exp_id} · {dataset}" if dataset else exp_id
    return {
        "prompt": prompt, "confidence": confidence,
        "focus": focus_label,
        "seed_trail": forgetting_so_far, "seeds_total": seeds_total,
        "threshold": 0.0, "seeds_done": seeds_done,
    }


@app.route("/api/narrate")
def api_narrate():
    global _NARRATION_CACHE
    now = time.time()
    force = str(request.args.get("refresh", "") or "").strip().lower() in {"1", "true"}
    if not force and _NARRATION_CACHE.get("text") and (now - float(_NARRATION_CACHE.get("ts", 0))) < _NARRATION_TTL:
        return jsonify({**_NARRATION_CACHE, "cached": True})
    ctx = _build_narration_context()
    base: dict = {
        "confidence": ctx["confidence"], "focus": ctx["focus"],
        "seed_trail": ctx["seed_trail"], "seeds_total": ctx["seeds_total"],
        "threshold": ctx["threshold"], "seeds_done": ctx.get("seeds_done", 0),
        "ts": now, "cached": False,
    }
    if ctx.get("prompt") is None:
        _NARRATION_CACHE = {**base, "text": ctx["fallback"], "error": False}
        return jsonify(_NARRATION_CACHE)
    text = _llm_narrate(ctx["prompt"], ctx)
    _NARRATION_CACHE = {**base, "text": text, "error": False}
    return jsonify(_NARRATION_CACHE)


def _llm_narrate(prompt: str, ctx: dict) -> str:
    """Try Anthropic → template fallback for narration."""
    # ── Anthropic ──────────────────────────────────────────────────────────────
    from tar_lab.llm_bridge import _api_key as _get_bridge_key
    api_key = _get_bridge_key()
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=160,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception:
            pass
    # ── Template fallback ──────────────────────────────────────────────────────
    trail     = ctx.get("seed_trail", [])
    threshold = ctx.get("threshold", 0.0)
    focus     = ctx.get("focus", "current experiment")
    st        = ctx.get("seeds_total", 0)

    if trail and threshold:
        # Validation suite template
        sd      = len(trail)
        passing = sum(1 for f in trail if f < threshold)
        mean_f  = sum(trail) / sd
        pct     = round((1 - mean_f / threshold) * 100)
        lines = [
            f"I'm {sd}/{st} seeds into the multi-method replication study, currently running {focus}.",
            f"So far, {passing}/{sd} seeds pass the pre-registered forgetting threshold of {threshold} "
            f"(mean HPC forgetting = {mean_f:.4f} — {pct}% below the ceiling).",
        ]
    elif trail:
        # Scale-up suite template (no threshold gate)
        sd     = len(trail)
        mean_f = sum(trail) / sd
        lines = [
            f"I'm {sd}/{st} seeds into the scale-up validation on {focus}.",
            f"Mean forgetting so far: {mean_f:.4f} across {sd} completed seed{'s' if sd != 1 else ''}.",
        ]
    else:
        # Director probe template
        lines = [
            f"I'm currently running {focus}.",
            f"This is a director-generated probe advancing one of my active research frontiers.",
            f"{st} seeds queued for this experiment." if st else "",
        ]
    return " ".join(l for l in lines if l)


@app.route("/api/replication")
def api_replication():
    obs = _find_replication_obs_root()
    if not obs:
        return jsonify({"available": False, "methods": {}, "is_live": False})

    # Check if a validation suite is currently running (data is live) or historical
    eq = _jload(_WS / "tar_state" / "experiment_queue.json") or {}
    experiments = eq.get("experiments", []) if isinstance(eq, dict) else []
    running = next((e for e in experiments if e.get("status") == "running"), None)
    is_live = bool(
        running and (
            running.get("method") == "validation_suite"
            or "claim_validation" in str(running.get("id", ""))
        )
    )
    running_exp_id = str(running.get("id", "")) if running else ""

    method_keys = [
        "sgd_baseline", "ewc_lambda_100", "ewc_lambda_1000",
        "si_c_0_01", "tcl_baseline", "high_penalty_conservative",
    ]
    out: dict[str, Any] = {
        "available": True, "methods": {},
        "is_live": is_live,
        "current_experiment": running_exp_id,
    }
    for m in method_keys:
        d = _jload(obs / f"{m}.json")
        if not d:
            continue
        agg = d.get("aggregate_so_far", {})
        out["methods"][m] = {
            "n": d.get("completed_seed_count", 0),
            "forgetting_mean": agg.get("forgetting_mean"),
            "forgetting_std": agg.get("forgetting_std"),
            "acc_mean": agg.get("acc_mean"),
            "acc_std": agg.get("acc_std"),
            "jaf_mean": agg.get("jaf_mean"),
            "collapse": agg.get("collapse_summary", {}),
            "seeds": [
                {
                    "seed": r.get("seed"),
                    "forgetting": r.get("mean_forgetting"),
                    "acc": r.get("final_mean_accuracy"),
                    "jaf": r.get("jaf"),
                    "per_task_acc": r.get("per_task_accuracy", []),
                    "per_task_forgetting": r.get("per_task_forgetting", []),
                }
                for r in d.get("rows", [])
            ],
        }
    return jsonify(out)


@app.route("/api/findings")
def api_findings():
    """Return all cached LLM findings memos and failure diagnoses."""
    cache_dir = _WS / "tar_state" / "llm_cache"
    findings: list[dict] = []
    failures: list[dict] = []
    if cache_dir.exists():
        for path in sorted(cache_dir.glob("findings_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = _jload(path)
            if isinstance(data, dict) and data.get("content"):
                findings.append({
                    "experiment_id": data.get("experiment_id", path.stem.removeprefix("findings_")),
                    "content": data["content"],
                    "written_at": data.get("written_at", 0),
                })
        for path in sorted(cache_dir.glob("failure_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = _jload(path)
            if isinstance(data, dict) and data.get("content"):
                failures.append({
                    "experiment_id": data.get("experiment_id", path.stem.removeprefix("failure_")),
                    "content": data["content"],
                    "error_text": data.get("error_text", ""),
                    "written_at": data.get("written_at", 0),
                })
    return jsonify({
        "findings_memos": findings,
        "failure_diagnoses": failures,
        "total_findings": len(findings),
        "total_failures": len(failures),
    })


@app.route("/api/llm_insights")
def api_llm_insights():
    """Return Claude-generated insights from the research director (synthesis, evals, claim checks)."""
    director_data = _jload(_WS / "tar_state" / "research_director_state.json") or {}
    llm_insights = director_data.get("llm_insights", {})

    cache_dir = _WS / "tar_state" / "llm_cache"
    scheduler_rationale: str = ""
    if cache_dir.exists():
        candidates = sorted(cache_dir.glob("scheduler_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates[:1]:
            data = _jload(path)
            if isinstance(data, dict):
                scheduler_rationale = str(data.get("content", "") or "")

    # Load active gap signals from the director priority overlay (written by LLM feedback loop)
    import time as _time
    overlay_raw = _jload(_WS / "tar_state" / "director_priority_overlay.json") or {}
    _48H = 48 * 3600
    now_ts = _time.time()
    gap_signals = []
    for fid, entry in overlay_raw.items():
        if isinstance(entry, dict):
            try:
                if now_ts - float(entry.get("written_ts", 0)) > _48H:
                    continue
            except Exception:
                pass
            signals = entry.get("gap_signals", [])
            delta = entry.get("priority_delta", 0)
            if signals or delta:
                gap_signals.append({
                    "frontier_id": fid,
                    "signals": signals,
                    "priority_delta": round(float(delta), 2),
                    "updated_at": entry.get("updated_at", ""),
                })

    return jsonify({
        "frontier_syntheses": llm_insights.get("frontier_syntheses", []),
        "experiment_evaluations": llm_insights.get("experiment_evaluations", []),
        "claim_verifications": llm_insights.get("claim_verifications", []),
        "scheduler_rationale": scheduler_rationale,
        "gap_signals": gap_signals,
        "available": bool(llm_insights or scheduler_rationale or gap_signals),
    })


def _build_tar_intelligence_context() -> str:
    """Build a rich plain-text summary of TAR state for Claude Q&A."""
    eq = _jload(_WS / "tar_state" / "experiment_queue.json") or {}
    experiments = eq.get("experiments", []) if isinstance(eq, dict) else []
    running = next((e for e in experiments if e.get("status") == "running"), None)
    pending = [e for e in experiments if e.get("status") == "pending"]

    hw = _jload(_WS / "tar_state" / "hardware_state.json") or {}
    gpu = hw.get("gpu", {})

    frontier_data = _jload(_WS / "tar_state" / "frontier_problems.json") or {}
    frontiers = frontier_data.get("problems", []) or []

    author_state = _jload(_WS / "tar_state" / "author_state.json") or {}
    papers = list(author_state.get("paper_queue", []) or [])

    lines: list[str] = ["# TAR System State\n"]

    if running:
        exp_id = running.get("id", "unknown")
        prog = running.get("progress", {}) or {}
        seeds_done = prog.get("seeds_done", 0)
        seeds_total = prog.get("seeds_total", 0)
        forgetting = list(prog.get("forgetting_so_far") or [])
        started = running.get("started_at", "")
        est_h = float(running.get("estimated_runtime_h", 0) or 0)
        elapsed_h = 0.0
        if started:
            try:
                import datetime as _dt
                started_dt = _dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
                elapsed_h = (_dt.datetime.now(_dt.timezone.utc) - started_dt).total_seconds() / 3600
            except Exception:
                pass
        remaining_h = max(0.0, est_h - elapsed_h)
        lines.append(f"## Currently Running: {exp_id}")
        lines.append(f"Dataset: {running.get('dataset', 'unknown')}  Method: {running.get('method', 'unknown')}")
        if seeds_total:
            lines.append(f"Seeds: {seeds_done}/{seeds_total} done")
        if forgetting:
            mean_f = sum(forgetting) / len(forgetting)
            lines.append(f"Forgetting per seed: {[round(f, 4) for f in forgetting]}")
            lines.append(f"Mean forgetting: {mean_f:.4f}")
        if est_h:
            lines.append(f"Elapsed: {elapsed_h:.1f}h  Remaining: ~{remaining_h:.0f}h")
        lines.append(f"GPU: {gpu.get('utilization_pct', 0)}% util, "
                     f"{gpu.get('vram_used_gb', 0):.1f}/{gpu.get('vram_total_gb', 0):.1f} GB VRAM, "
                     f"{gpu.get('temperature_c', 0)}°C")
        ctx_why = (running.get("context", {}) or {}).get("why", "")
        if ctx_why:
            lines.append(f"Why: {str(ctx_why)[:200]}")
        lines.append("")
    else:
        lines.append("## Currently Running: idle\n")

    if pending:
        lines.append(f"## Queue ({len(pending)} pending)")
        for e in pending[:8]:
            lines.append(f"  - {e.get('id', '?')} ({e.get('dataset', '?')})")
        lines.append("")

    cache_dir = _WS / "tar_state" / "llm_cache"
    if cache_dir.exists():
        recent = sorted(cache_dir.glob("findings_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        if recent:
            lines.append("## Recent Completed Experiments")
            for p in recent:
                d = _jload(p) or {}
                lines.append(f"  - {d.get('experiment_id', p.stem)}: {str(d.get('content', ''))[:180]}")
            lines.append("")

    if papers:
        lines.append("## Paper Pipeline")
        for paper in papers[:5]:
            pid = paper.get("project_id") or paper.get("paper_id", "?")
            lines.append(f"  - {pid}: {paper.get('status', '?')}")
        lines.append("")

    active_frontiers = [f for f in frontiers if f.get("status") not in {"done", "closed"}]
    if active_frontiers:
        lines.append("## Active Research Frontiers")
        for f in active_frontiers[:5]:
            lines.append(f"  - {f.get('id', '?')}: {f.get('title', '')[:80]}")
        lines.append("")

    return "\n".join(lines)


@app.route("/api/tar_intelligence/ask", methods=["POST"])
def api_tar_intelligence_ask():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "") or "").strip()
    if not question:
        return jsonify({"answer": None, "error": "No question provided"}), 400
    if len(question) > 600:
        return jsonify({"answer": None, "error": "Question too long (max 600 chars)"}), 400

    from tar_lab.llm_bridge import _api_key as _get_api_key
    api_key = _get_api_key()
    if not api_key:
        return jsonify({"answer": None, "error": "ANTHROPIC_API_KEY not configured — Claude Q&A unavailable"}), 503

    context = _build_tar_intelligence_context()
    system = (
        "You are TAR (Thermodynamic Active Research), an autonomous ML research system. "
        "Answer the user's question strictly using the system state provided. "
        "RULES: (1) Only cite numbers and facts that appear in the context above — never invent, estimate, "
        "or extrapolate beyond what the data shows. (2) If a fact is not in the context, say so explicitly "
        "(e.g. 'that data isn't available yet'). (3) Do not speculate about future results. "
        "(4) Do not make claims about experiment outcomes that haven't been measured. "
        "Speak in first person as a researcher. 3-5 sentences, plain English, no markdown headers."
    )
    prompt = f"{context}\n---\n\nUser question: {question}"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return jsonify({"answer": msg.content[0].text.strip(), "error": None})
    except Exception as exc:
        return jsonify({"answer": None, "error": str(exc)}), 500


@app.route("/api/experiment_history")
def api_experiment_history():
    """Complete experiment history with per-seed results, grouped by paper."""
    archive = _jload(_WS / "tar_state" / "experiment_archive.json") or {}
    arc_exps = list(archive.get("experiments", []) or [])

    queue = _jload(_WS / "tar_state" / "experiment_queue.json") or {}
    q_exps = list(queue.get("experiments", []) or [])

    seen: set[str] = set()
    all_exps: list[dict] = []
    for exp in arc_exps + q_exps:
        eid = str(exp.get("id") or exp.get("experiment_id") or exp.get("project_id") or "")
        if eid and eid not in seen:
            seen.add(eid)
            all_exps.append(exp)

    exp_results_dir = _WS / "tar_state" / "experiments"
    for exp in all_exps:
        eid = str(exp.get("id") or exp.get("experiment_id") or exp.get("project_id") or "")
        if eid:
            result_path = exp_results_dir / eid / "result.json"
            if result_path.exists():
                exp["result"] = _jload(result_path) or {}

    cache_dir = _WS / "tar_state" / "llm_cache"
    findings_by_id: dict[str, str] = {}
    if cache_dir.exists():
        for path in cache_dir.glob("findings_*.json"):
            data = _jload(path) or {}
            eid = str(data.get("experiment_id") or path.stem.removeprefix("findings_"))
            if eid:
                findings_by_id[eid] = str(data.get("content", "") or "")

    for exp in all_exps:
        eid = str(exp.get("id") or exp.get("experiment_id") or exp.get("project_id") or "")
        if eid in findings_by_id:
            exp["findings_memo"] = findings_by_id[eid]

    all_exps.sort(
        key=lambda e: str(e.get("completed_at") or e.get("started_at") or e.get("submitted_at") or ""),
        reverse=True,
    )

    groups: dict[str, list] = {}
    for exp in all_exps:
        paper = str(exp.get("author_paper_id") or exp.get("paper_id") or "unassigned")
        groups.setdefault(paper, []).append(exp)

    verdicts: dict[str, int] = {"CONFIRMED": 0, "DIRECTIONAL": 0, "NULL": 0, "ADVERSE": 0}
    for exp in all_exps:
        v = str((exp.get("result") or {}).get("verdict") or "")
        if v in verdicts:
            verdicts[v] += 1

    return jsonify({
        "experiments": all_exps,
        "groups": [{"paper_id": k, "experiments": v} for k, v in groups.items()],
        "total": len(all_exps),
        "complete": sum(1 for e in all_exps if str(e.get("status") or "") == "complete"),
        "running": sum(1 for e in all_exps if str(e.get("status") or "") == "running"),
        "pending": sum(1 for e in all_exps if str(e.get("status") or "") == "pending"),
        "verdicts": verdicts,
    })


# ── RunPod cloud GPU control ──────────────────────────────────────────────────

@app.route("/api/runpod/status")
def api_runpod_status():
    import os
    try:
        from tar_runpod_executor import load_runpod_config, is_runpod_enabled
    except ImportError:
        return jsonify({"mode": "unavailable", "error": "tar_runpod_executor not installed"})

    config     = load_runpod_config(_WS)
    enabled    = is_runpod_enabled(_WS)
    api_key    = bool(os.environ.get("RUNPOD_API_KEY", ""))

    suspended_path = _WS / "tar_state" / "runpod_suspended.flag"
    state_path     = _WS / "tar_state" / "runpod_state.json"

    suspended_data = {}
    if suspended_path.exists():
        try:
            suspended_data = json.loads(suspended_path.read_text(encoding="utf-8"))
        except Exception:
            suspended_data = {"reason": "unknown"}

    pod_state: dict = {}
    if state_path.exists():
        try:
            pod_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    active_pod_id = str(pod_state.get("active_pod_id", "") or "")
    uptime_h      = 0.0
    cost_est      = 0.0

    # Determine mode
    if suspended_data:
        mode = "suspended"
    elif active_pod_id and enabled:
        mode = "active"
    elif enabled:
        mode = "enabled"
    else:
        mode = "disabled"

    # Try live pod uptime from RunPod API (best-effort, non-blocking)
    if active_pod_id and api_key:
        try:
            import runpod as _rp
            _rp.api_key = os.environ["RUNPOD_API_KEY"]
            info = _rp.get_pod(active_pod_id)
            runtime = info.get("runtime") or {}
            uptime_s = float(runtime.get("uptimeInSeconds") or 0)
            uptime_h = round(uptime_s / 3600, 2)
            cost_est = round(uptime_h * 0.44, 2)  # conservative $0.44/hr est
        except Exception:
            pass

    # Cost warning
    cost_warning: dict = {}
    warn_path = _WS / "tar_state" / "runpod_cost_warning.json"
    if warn_path.exists():
        try:
            cw = json.loads(warn_path.read_text(encoding="utf-8"))
            # Only show warning if it's for the current active pod
            if not active_pod_id or cw.get("pod_id") == active_pod_id:
                cost_warning = cw
        except Exception:
            pass

    return jsonify({
        "mode":               mode,
        "api_key_set":        api_key,
        "pod_id":             active_pod_id,
        "gpu_type":           str(pod_state.get("gpu_type", "") or ""),
        "experiment_id":      str(pod_state.get("experiment_id", "") or ""),
        "seeds_done":         int(pod_state.get("seeds_done_snapshot", 0) or 0),
        "seeds_total":        int(pod_state.get("seeds_total", 0) or 0),
        "uptime_h":           uptime_h,
        "cost_est_usd":       cost_est,
        "min_vram_gb":        float(config.get("min_vram_gb", 24)),
        "max_cost_per_hour":  float(config.get("max_cost_per_hour", 2.0)),
        "max_experiment_cost_usd": float(config.get("max_experiment_cost_usd", 10.0)),
        "threshold_runtime_h": float(config.get("threshold_runtime_h", 12)),
        "suspended_reason":   str(suspended_data.get("reason", "") or ""),
        "suspended_at":       str(suspended_data.get("suspended_at", "") or ""),
        "volume_id":          str(config.get("volume_id", "") or ""),
        "cost_warning":       cost_warning,
    })


@app.route("/api/runpod/enable", methods=["POST"])
def api_runpod_enable():
    import os
    if not os.environ.get("RUNPOD_API_KEY"):
        return jsonify({"ok": False, "error": "RUNPOD_API_KEY environment variable not set"}), 400
    try:
        flag = _WS / "tar_state" / "runpod_enabled.flag"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(json.dumps({"enabled_at": _utcnow_iso()}), encoding="utf-8")
        suspended = _WS / "tar_state" / "runpod_suspended.flag"
        suspended.unlink(missing_ok=True)
        return jsonify({"ok": True, "mode": "enabled"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/runpod/disable", methods=["POST"])
def api_runpod_disable():
    try:
        (_WS / "tar_state" / "runpod_enabled.flag").unlink(missing_ok=True)
        return jsonify({"ok": True, "mode": "disabled"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/runpod/kill", methods=["POST"])
def api_runpod_kill():
    import os
    killed_pod = ""
    try:
        (_WS / "tar_state" / "runpod_enabled.flag").unlink(missing_ok=True)
        state_path = _WS / "tar_state" / "runpod_state.json"
        if state_path.exists():
            try:
                pod_state  = json.loads(state_path.read_text(encoding="utf-8"))
                pod_id     = str(pod_state.get("active_pod_id", "") or "")
                if pod_id and os.environ.get("RUNPOD_API_KEY"):
                    import runpod as _rp
                    _rp.api_key = os.environ["RUNPOD_API_KEY"]
                    _rp.terminate_pod(pod_id)
                    killed_pod = pod_id
            except Exception:
                pass
            state_path.unlink(missing_ok=True)
        return jsonify({"ok": True, "mode": "disabled", "killed_pod": killed_pod})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── paper file serving ────────────────────────────────────────────────────────
@app.route("/serve/paper/<path:relpath>")
def serve_paper(relpath: str):
    import mimetypes
    for root in [_REPO / "paper", _WS / "paper"]:
        candidate = (root / relpath).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.exists() and candidate.is_file():
            mime = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            return send_file(str(candidate), mimetype=mime, as_attachment=False)
    return abort(404)


# ── hardware monitor startup ──────────────────────────────────────────────────
def _start_hardware_monitor() -> None:
    try:
        from tar_hardware_monitor import HardwareMonitor
        HardwareMonitor(_WS, interval_s=5.0).start_background()
    except Exception as exc:
        print(f"[Dashboard] Hardware monitor unavailable: {exc}", flush=True)


def _seed_frontier() -> None:
    """Ensure frontier_problems.json exists with default problems seeded."""
    fp_path = _WS / "tar_state" / "frontier_problems.json"
    if fp_path.exists():
        return
    try:
        from tar_frontier import FrontierRegistry
        FrontierRegistry(_WS)
        print("[Dashboard] Frontier problems seeded.", flush=True)
    except Exception as exc:
        print(f"[Dashboard] Frontier seed unavailable: {exc}", flush=True)


# ── HTML / CSS / JS ───────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TAR Research Dashboard</title>
<style>
:root{
  --bg:#060b14;--surface:#0c1522;--border:#1a2840;--border2:#0f1e30;
  --text:#c8d8e8;--text2:#5a7a9a;--text3:#334a62;
  --green:#34d399;--blue:#60a5fa;--amber:#fbbf24;--red:#f87171;
  --pink:#f472b6;--purple:#a78bfa;--teal:#2dd4bf;--sky:#38bdf8;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'JetBrains Mono',Consolas,monospace;font-size:12px;min-height:100vh;overflow-x:hidden}
a{color:var(--sky);text-decoration:none}a:hover{text-decoration:underline}

/* ── layout ── */
.page{display:flex;flex-direction:column;min-height:100vh;padding:12px;gap:10px}
.header{display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:10px 14px;background:rgba(12,21,34,.92);border:1px solid var(--border);border-radius:10px;flex-shrink:0;position:sticky;top:8px;z-index:20;backdrop-filter:blur(10px)}
.bk-banner{background:#16060e;border:1px solid #7c2d4a;border-radius:8px;padding:8px 14px;flex-shrink:0;animation:bkglow 2s ease-in-out infinite alternate;display:none}
@keyframes bkglow{from{box-shadow:0 0 6px #7c2d4a44}to{box-shadow:0 0 18px #f472b666}}
.dashboard-toolbar{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;padding:10px 12px;background:linear-gradient(180deg,#0d1726,#0a1220);border:1px solid var(--border);border-radius:10px}
.toolbar-group{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.toolbar-label{color:var(--text2);font-size:.58rem;letter-spacing:.08em;text-transform:uppercase}
.toolbar-btn{background:#0a1930;border:1px solid #20385f;color:var(--text);cursor:pointer;padding:6px 10px;border-radius:999px;font-size:.62rem;line-height:1}
.toolbar-btn:hover{background:#122340}
.toolbar-btn.is-primary{color:var(--sky);border-color:#1f4b80}
.main-grid{display:grid;grid-template-columns:minmax(280px,1fr) minmax(620px,1.75fr) minmax(320px,1fr);gap:10px;align-items:start;flex:1;min-height:auto;overflow:visible}

/* ── cards ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;display:flex;flex-direction:column;overflow:hidden;min-height:0;box-shadow:0 12px 34px #02060d33}
.col{display:flex;flex-direction:column;gap:10px;min-height:0;overflow:visible}
.col::-webkit-scrollbar{width:3px}.col::-webkit-scrollbar-thumb{background:var(--border)}
.hdr{background:#090f1a;border-bottom:1px solid var(--border2);padding:8px 78px 8px 10px;font-size:.6rem;letter-spacing:.09em;text-transform:uppercase;color:var(--text2);display:flex;justify-content:space-between;align-items:center;flex-shrink:0;position:relative}
.body{padding:10px;overflow-y:auto;flex:1}
.body::-webkit-scrollbar{width:3px}.body::-webkit-scrollbar-thumb{background:var(--border)}
.card.is-collapsed > :not(.hdr){display:none}
.panel-tools{position:absolute;right:8px;top:50%;transform:translateY(-50%);display:flex;gap:4px}
.panel-btn{background:#0b1730;border:1px solid #1f3a61;color:var(--text2);cursor:pointer;padding:2px 8px;border-radius:999px;font-size:.58rem;line-height:1.3}
.panel-btn:hover{background:#102140;color:var(--text)}
.panel-focus-backdrop{position:fixed;inset:0;background:#02060dcc;backdrop-filter:blur(3px);z-index:90;display:none}
.panel-focus-backdrop.open{display:block}
.card.panel-focus{position:fixed !important;inset:18px;z-index:95;max-height:calc(100vh - 36px);overflow:auto;border-color:#32527f;box-shadow:0 24px 90px #000b}
.card.panel-focus .hdr{position:sticky;top:0;z-index:2}
.panel-focus-open{overflow:hidden}

/* ── hardware gauges ── */
.gauge-row{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.gauge-label{color:var(--text2);font-size:.58rem;width:36px;flex-shrink:0}
.gauge-bar{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.gauge-fill{height:100%;border-radius:3px;transition:width .5s ease}
.gauge-val{color:var(--text);font-size:.6rem;width:52px;text-align:right;flex-shrink:0}
.g-green{background:var(--green)}.g-amber{background:var(--amber)}.g-red{background:var(--red)}

/* ── pills ── */
.pill{display:inline-block;padding:1px 6px;border-radius:9999px;font-size:.55rem;font-weight:700;letter-spacing:.05em;white-space:nowrap}
.s-run {background:#0a1f12;color:#34d399;border:1px solid #065f46}
.s-done{background:#0a1526;color:#60a5fa;border:1px solid #1e40af}
.s-fail{background:#1f0a0a;color:#f87171;border:1px solid #7f1d1d}
.s-pend{background:#0f1520;color:#475569;border:1px solid #1e2d3d}
.s-part{background:#1a1005;color:#fbbf24;border:1px solid #78350f}
.s-bk  {background:#1a0814;color:#f472b6;border:1px solid #831843}
.s-plan{background:#0d1425;color:#818cf8;border:1px solid #3730a3}
.s-queue{background:#0f1825;color:#a78bfa;border:1px solid #5b21b6}
.s-analyzing{background:#0f1825;color:#34d399;border:1px solid #065f46}
.s-writing{background:#0a1f12;color:#2dd4bf;border:1px solid #0f766e}

/* ── log ── */
.log-box{font-size:.65rem;line-height:1.5;overflow-y:auto;flex:1;background:#04080f;padding:8px 10px;white-space:pre-wrap;word-break:break-all;color:#7fb3d3}
.log-box::-webkit-scrollbar{width:3px}.log-box::-webkit-scrollbar-thumb{background:var(--border)}

/* ── tables ── */
table{width:100%;border-collapse:collapse}
th{color:var(--text2);text-align:left;padding:3px 6px;border-bottom:1px solid var(--border2);font-size:.58rem}
td{padding:3px 6px;border-bottom:1px solid var(--border2);font-size:.62rem}
tr:hover td{background:#0a1220}

/* ── experiment rows ── */
.exp-row{padding:7px 8px;border-bottom:1px solid var(--border2);cursor:pointer;transition:background .15s}
.exp-row:hover{background:#0a1525}
.exp-name{color:var(--text);font-size:.7rem;font-weight:600}
.exp-meta{color:var(--text2);font-size:.58rem;margin-top:2px}
.exp-progress{margin-top:4px;height:3px;background:var(--border);border-radius:2px}
.exp-progress-fill{height:100%;background:var(--green);border-radius:2px;transition:width .5s}

/* ── modal ── */
.modal-overlay{position:fixed;inset:0;background:#000000cc;z-index:100;display:none;align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal{background:#0c1522;border:1px solid #2a4060;border-radius:10px;width:min(980px,92vw);max-height:90vh;overflow-y:auto;padding:0}
.modal-header{background:#090f1a;padding:12px 16px;border-bottom:1px solid var(--border);border-radius:10px 10px 0 0;display:flex;justify-content:space-between;align-items:flex-start}
.modal-body{padding:16px}
.modal-section{margin-bottom:16px}
.modal-section-title{font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:var(--text2);margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid var(--border2)}
.modal-text{color:var(--text);font-size:.7rem;line-height:1.7}
.modal-close{background:none;border:1px solid var(--border);color:var(--text2);cursor:pointer;padding:2px 8px;border-radius:4px;font-size:.65rem}
.modal-close:hover{background:var(--border);color:var(--text)}
.action-btn{background:#0b1730;border:1px solid #1f3a61;color:var(--sky);cursor:pointer;padding:2px 8px;border-radius:4px;font-size:.58rem}
.action-btn:hover{background:#102140}
.bk-item{padding:6px 0;border-bottom:1px solid #3b0764;cursor:pointer}
.bk-item:hover{background:#1a0b16}
.bk-item.expanded{background:#160a14}
.bk-dismiss{background:none;border:1px solid #7c2d4a;color:var(--pink);cursor:pointer;padding:1px 6px;border-radius:9999px;font-size:.55rem}
.bk-dismiss:hover{background:#3b0a1f}
.mini-note{color:var(--text3);font-size:.58rem;line-height:1.5}
.director-row{padding:5px 0;border-bottom:1px solid var(--border2)}
.learned-row{padding:7px 8px;border-bottom:1px solid var(--border2);cursor:pointer;transition:background .18s ease}
.learned-row:hover{background:#0c1523}
.learned-row.expanded{background:#0a1220}
.learned-head{display:flex;align-items:center;gap:8px}
.learned-title{color:var(--text);font-size:.68rem;font-weight:600;flex:1;min-width:0}
.learned-meta{color:var(--text2);font-size:.58rem;white-space:nowrap}
.learned-body{margin-top:7px;font-size:.6rem;color:var(--text2);line-height:1.55}
.learned-summary{color:var(--text);font-size:.6rem;line-height:1.55}
.learned-subtitle{color:var(--text2);font-size:.56rem;letter-spacing:.06em;text-transform:uppercase;margin:7px 0 3px}
.learned-chip{display:inline-block;padding:1px 6px;border-radius:999px;border:1px solid #21405d;background:#0a1524;color:var(--sky);font-size:.55rem;margin:2px 4px 0 0}
.learned-claim{margin-top:4px;color:var(--text2)}
.learned-source{color:var(--text3);font-size:.58rem}
.learned-topline{padding:8px;border-bottom:1px solid var(--border2);background:#0a1320}
.director-title{color:var(--sky);font-size:.66rem;font-weight:600}
.director-meta{color:var(--text2);font-size:.58rem;margin-top:2px}
.director-action{color:var(--teal);font-size:.6rem;margin-top:3px;line-height:1.45}
.result-row{padding:7px 8px;border-bottom:1px solid var(--border2)}
.result-title{color:var(--text);font-size:.67rem;font-weight:600}
.result-meta{color:var(--text2);font-size:.58rem;margin-top:2px}
.result-metrics{color:var(--text3);font-size:.58rem;margin-top:3px;line-height:1.45}
.result-note{color:var(--text2);font-size:.58rem;margin-top:3px;line-height:1.5}

/* ── animations ── */
.bk-flash{animation:bkf .8s ease-in-out infinite alternate}
@keyframes bkf{from{color:#f472b6}to{color:#ffffff}}
.pulse{animation:pu 2s infinite}
@keyframes pu{0%,100%{opacity:1}50%{opacity:.35}}

/* ── frontier ── */
.fp-card{padding:7px 8px;border-bottom:1px solid var(--border2)}
.fp-title{color:var(--sky);font-size:.68rem;font-weight:600}
.fp-meta{color:var(--text2);font-size:.58rem;margin-top:2px;display:flex;gap:8px}

/* ── misc ── */
select{background:#090f1a;color:var(--text2);border:1px solid var(--border);border-radius:3px;padding:1px 4px;font-size:.6rem}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.page.comfortable #proc-panel{max-height:32vh !important}
.page.comfortable #scheduler-panel{max-height:24vh !important}
.page.comfortable #director-panel{max-height:34vh !important}
.page.comfortable #exp-panel{max-height:58vh !important}
.page.comfortable #fp-panel{max-height:38vh !important}
.page.comfortable #phases-panel{max-height:42vh !important}
.page.comfortable #papers-panel{max-height:36vh !important}
.page.comfortable .log-box{min-height:420px;max-height:68vh}
.page.dense #proc-panel{max-height:140px !important}
.page.dense #scheduler-panel{max-height:100px !important}
.page.dense #director-panel{max-height:160px !important}
.page.dense #exp-panel{max-height:340px !important}
.page.dense #fp-panel{max-height:160px !important}
.page.dense #phases-panel{max-height:200px !important}
.page.dense #papers-panel{max-height:200px !important}
.page.dense .log-box{min-height:260px;max-height:50vh}
@media (max-width: 1520px){
  .main-grid{grid-template-columns:minmax(260px,0.95fr) minmax(540px,1.55fr) minmax(300px,0.95fr)}
}
@media (max-width: 1220px){
  .main-grid{grid-template-columns:1fr}
  .header{position:static}
}
</style>
</head>
<body>
<div class="page">

<!-- ── HEADER with hardware gauges ─────────────────────────────────────────── -->
<div class="header">
  <span style="color:var(--sky);font-weight:700;font-size:.9rem;flex-shrink:0">TAR</span>
  <span style="color:var(--text);font-weight:600;font-size:.8rem;flex-shrink:0">Research Ecosystem</span>
  <div style="display:flex;flex-direction:column;gap:2px;flex:1;max-width:560px;margin:0 12px">
    <div class="gauge-row">
      <span class="gauge-label">GPU</span>
      <div class="gauge-bar"><div class="gauge-fill g-green" id="g-gpu-util" style="width:0%"></div></div>
      <span class="gauge-val" id="gv-gpu-util">—</span>
      <span class="gauge-label" style="width:28px">VRAM</span>
      <div class="gauge-bar"><div class="gauge-fill g-green" id="g-vram" style="width:0%"></div></div>
      <span class="gauge-val" id="gv-vram">—</span>
      <span id="gv-gpu-temp" style="color:var(--text2);font-size:.6rem;width:44px;text-align:right">—</span>
    </div>
    <div class="gauge-row">
      <span class="gauge-label">CPU</span>
      <div class="gauge-bar"><div class="gauge-fill g-green" id="g-cpu" style="width:0%"></div></div>
      <span class="gauge-val" id="gv-cpu">—</span>
      <span class="gauge-label" style="width:28px">RAM</span>
      <div class="gauge-bar"><div class="gauge-fill g-green" id="g-ram" style="width:0%"></div></div>
      <span class="gauge-val" id="gv-ram">—</span>
      <span id="gv-cpu-temp" style="color:var(--text2);font-size:.6rem;width:44px;text-align:right">—</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;flex-shrink:0">
    <span id="active-step-label" style="color:var(--amber);font-size:.65rem"></span>
    <span id="last-update" style="color:var(--text3);font-size:.58rem"></span>
    <span class="pill pulse" style="background:#0a1f10;color:#34d399;border:1px solid #166534">● LIVE</span>
  </div>
</div>

<!-- ── BREAKTHROUGH BANNER ────────────────────────────────────────────────── -->
<div class="dashboard-toolbar">
  <div class="toolbar-group">
    <span class="toolbar-label">View</span>
    <button class="toolbar-btn is-primary" id="density-btn" onclick="toggleDensity()">Dense view</button>
    <button class="toolbar-btn" onclick="expandAllPanels()">Expand all</button>
    <button class="toolbar-btn" onclick="collapseAllPanels()">Collapse all</button>
    <button class="toolbar-btn" id="restore-panel-btn" onclick="unfocusPanel()" style="display:none">Restore focused panel</button>
  </div>
  <div class="toolbar-group">
    <span class="toolbar-label">Jump</span>
    <button class="toolbar-btn" onclick="scrollToPanel('experiments-card')">Experiments</button>
    <button class="toolbar-btn" onclick="scrollToPanel('frontier-card')">Frontier</button>
    <button class="toolbar-btn" onclick="scrollToPanel('litmem-card')">Learning</button>
    <button class="toolbar-btn" onclick="scrollToPanel('log-card')">Log</button>
    <button class="toolbar-btn" onclick="scrollToPanel('phases-card')">Results</button>
    <button class="toolbar-btn" onclick="scrollToPanel('papers-card')">Papers</button>
  </div>
</div>

<div class="bk-banner" id="bk-banner">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
    <span style="font-size:.9rem;color:var(--pink)" class="bk-flash">⚡</span>
    <span style="color:var(--pink);font-weight:700;font-size:.75rem;letter-spacing:.06em">BREAKTHROUGHS DETECTED</span>
    <button class="action-btn" id="bk-reset-btn" onclick="resetBreakthroughDismissals()" style="display:none">Show hidden</button>
    <span id="bk-count" class="pill s-bk" style="margin-left:auto"></span>
  </div>
  <div id="bk-list"></div>
</div>

<!-- ── MAIN GRID ─────────────────────────────────────────────────────────── -->
<div class="main-grid" style="flex:1;min-height:0">

  <!-- LEFT COLUMN -->
  <div class="col">

    <!-- Queue steps -->
    <div class="card" id="queue-card" data-panel-title="Run Queue" style="flex-shrink:0">
      <div class="hdr">Run Queue</div>
      <div class="body" id="queue-panel" style="padding:6px 8px"></div>
    </div>

    <!-- Processes -->
    <div class="card" id="processes-card" data-panel-title="Processes" style="flex-shrink:0">
      <div class="hdr"><span>Processes</span><span id="proc-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div class="body" id="proc-panel" style="padding:4px 6px;max-height:140px"></div>
    </div>

    <!-- Scheduler -->
    <div class="card" id="scheduler-card" data-panel-title="Scheduler" style="flex-shrink:0">
      <div class="hdr">Scheduler</div>
      <div class="body" id="scheduler-panel" style="font-size:.62rem;color:var(--text2);line-height:1.5;max-height:180px;overflow-y:auto"></div>
    </div>

    <!-- Research Director -->
    <div class="card" id="director-card" data-panel-title="Research Director" style="flex-shrink:0">
      <div class="hdr"><span>Research Director</span><span id="director-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div class="body" id="director-panel" style="padding:4px 6px;max-height:280px;overflow-y:auto"></div>
    </div>

    <!-- Human Review -->
    <div class="card" id="human-review-card" data-panel-title="Human Review" style="flex-shrink:0">
      <div class="hdr"><span>Human Review</span><span id="review-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div class="body" id="review-panel" style="padding:4px 6px;max-height:280px;overflow-y:auto"></div>
    </div>

    <!-- Validation -->
    <div class="card" id="validation-card" data-panel-title="Validation Board" style="flex-shrink:0">
      <div class="hdr"><span>Validation Board</span><span id="validation-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div class="body" id="validation-panel" style="padding:4px 6px;max-height:220px;overflow-y:auto"></div>
    </div>

    <!-- TAR Author -->
    <div class="card" id="author-card" data-panel-title="TAR Author">
      <div class="hdr">TAR Author</div>
      <div class="body" id="author-panel" style="font-size:.65rem"></div>
    </div>

    <!-- Autonomous Research -->
    <div class="card" id="autonomous-card" data-panel-title="Autonomous Research">
      <div class="hdr"><span>Autonomous Research</span><span id="ar-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div id="ar-panel" class="body" style="padding:4px 6px"></div>
    </div>

  </div>

  <!-- CENTRE COLUMN -->
  <div style="display:flex;flex-direction:column;gap:8px;min-height:0">

    <!-- Experiments panel -->
    <div class="card" id="experiments-card" data-panel-title="Experiments" style="flex:0 0 auto">
      <div class="hdr">
        <span>Experiments</span>
        <div style="display:flex;gap:8px;align-items:center">
          <button class="action-btn" onclick="openInjectModal()">Inject</button>
          <span id="exp-summary" style="color:var(--text2);font-size:.58rem"></span>
        </div>
      </div>
      <div id="exp-panel" style="overflow-y:auto;max-height:340px"></div>
    </div>

    <!-- Frontier Problems -->
    <div class="card" id="frontier-card" data-panel-title="Frontier Research Problems" style="flex:0 0 auto">
      <div class="hdr"><span>Frontier Research Problems</span><span id="fp-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div id="fp-panel" style="overflow-y:auto;max-height:260px"></div>
    </div>

    <!-- Learned literature memory -->
    <div class="card" id="litmem-card" data-panel-title="Learned Literature Memory" style="flex:0 0 auto">
      <div class="hdr"><span>Learned Literature Memory</span><span id="litmem-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div id="litmem-panel" class="body" style="overflow-y:auto;max-height:280px"></div>
    </div>

    <!-- Live Log -->
    <div class="card" id="log-card" data-panel-title="Live Log" style="flex:1;min-height:0;display:flex;flex-direction:column">
      <div class="hdr" style="flex-shrink:0">
        <span>Live Log — <span id="log-label" style="color:var(--sky)">auto</span></span>
        <div style="display:flex;align-items:center;gap:6px">
          <select id="log-select" onchange="onLogSelect(this.value)"><option value="">auto</option><option value="experiment_orchestrator" selected>experiment_orchestrator</option></select>
          <span id="log-mtime" style="color:var(--text3);font-size:.58rem"></span>
        </div>
      </div>
      <div class="log-box" id="log-box">Waiting for logs…</div>
    </div>

  </div>

  <!-- RIGHT COLUMN -->
  <div class="col">

    <!-- Experiment results -->
    <div class="card" id="phases-card" data-panel-title="Experiment Results" style="flex-shrink:0">
      <div class="hdr"><span>Experiment Results</span><span id="results-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div id="phases-panel" class="body" style="padding:2px 4px;max-height:200px"></div>
    </div>

    <!-- Papers -->
    <div class="card" id="papers-card" data-panel-title="Papers" style="flex-shrink:0">
      <div class="hdr"><span>Papers</span><span id="papers-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div id="papers-panel" class="body" style="max-height:200px"></div>
    </div>

    <!-- Project Registry -->
    <div class="card" id="registry-card" data-panel-title="Project Registry">
      <div class="hdr"><span>Project Registry</span><span id="reg-count" style="color:var(--text3);font-size:.58rem"></span></div>
      <div id="registry-panel" class="body" style="padding:4px 6px"></div>
    </div>

  </div>
</div>
</div>

<!-- ── EXPERIMENT DETAIL MODAL ────────────────────────────────────────────── -->
<div class="panel-focus-backdrop" id="panel-focus-backdrop" onclick="unfocusPanel()"></div>
<div class="modal-overlay" id="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <div class="modal-header">
      <div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
          <span id="m-stage-pill"></span>
          <span id="m-name" style="color:var(--text);font-weight:700;font-size:.85rem"></span>
        </div>
        <div id="m-meta" style="color:var(--text2);font-size:.62rem"></div>
      </div>
      <button class="modal-close" onclick="closeModal()">✕ close</button>
    </div>
    <div class="modal-body">
      <div class="modal-section">
        <div class="modal-section-title">Why We're Running This</div>
        <div class="modal-text" id="m-why"></div>
      </div>
      <div class="modal-section">
        <div class="modal-section-title">Scientific Hypothesis</div>
        <div class="modal-text" id="m-hypothesis"></div>
      </div>
      <div class="modal-section" id="m-progress-section">
        <div class="modal-section-title">Live Progress</div>
        <div id="m-progress-bar" style="height:4px;background:var(--border);border-radius:2px;margin-bottom:6px">
          <div id="m-progress-fill" style="height:100%;background:var(--green);border-radius:2px;width:0%;transition:width .5s"></div>
        </div>
        <div class="modal-text" id="m-progress-text"></div>
      </div>
      <div class="modal-section">
        <div class="modal-section-title">Projected Outcome</div>
        <div class="modal-text" id="m-projected"></div>
      </div>
      <div class="modal-section" id="m-live-log-section">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
          <div class="modal-section-title" style="margin-bottom:0">Live Logs</div>
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
            <select id="m-log-source" onchange="changeExperimentLogSource(this.value)" style="background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 6px;font-size:.6rem;min-width:180px"></select>
            <button class="action-btn" id="m-open-log-btn" onclick="openExperimentLogPanel()">Open in Live Log panel</button>
            <span id="m-log-mtime" style="color:var(--text3);font-size:.58rem"></span>
          </div>
        </div>
        <div id="m-log-path" style="color:var(--text3);font-size:.56rem;margin-top:6px;word-break:break-all"></div>
        <div class="log-box" id="m-log-box" style="max-height:220px;margin-top:8px">(loading experiment log…)</div>
      </div>
      <div class="modal-section" id="m-result-section" style="display:none">
        <div class="modal-section-title">Final Result</div>
        <div class="modal-text" id="m-result-text"></div>
      </div>
      <div style="display:flex;gap:16px;margin-top:4px">
        <div style="flex:1">
          <div class="modal-section-title" style="margin-bottom:4px">Frontier Problem</div>
          <div id="m-frontier" style="color:var(--sky);font-size:.65rem"></div>
        </div>
        <div style="flex:1">
          <div class="modal-section-title" style="margin-bottom:4px">Feeds Paper</div>
          <div id="m-paper" style="color:var(--teal);font-size:.65rem"></div>
        </div>
      </div>
      <div class="modal-section" style="margin-top:12px">
        <div class="modal-section-title">Hardware Allocation</div>
        <div id="m-hardware" style="color:var(--text2);font-size:.65rem;display:flex;gap:16px;flex-wrap:wrap"></div>
      </div>
      <div class="modal-section" id="m-methodology-section">
        <div class="modal-section-title">Methodology</div>
        <div class="modal-text" id="m-methodology" style="color:var(--text2)"></div>
      </div>
    </div>
  </div>
</div>

<div class="modal-overlay" id="inject-modal-overlay" onclick="if(event.target===this)closeInjectModal()">
  <div class="modal" style="width:min(820px,92vw)">
    <div class="modal-header">
      <div>
        <div style="color:var(--text);font-weight:700;font-size:.85rem">Inject Experiment</div>
        <div style="color:var(--text2);font-size:.62rem">Submit a human-authored experiment directly into the orchestrator queue.</div>
      </div>
      <button class="modal-close" onclick="closeInjectModal()">x close</button>
    </div>
    <div class="modal-body">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Name</div>
          <input id="inj-name" style="width:100%;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px" value="human_experiment">
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Project Id</div>
          <input id="inj-project" style="width:100%;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px" value="human_experiment">
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Dataset</div>
          <select id="inj-dataset" style="width:100%;padding:6px">
            <option value="split_cifar10">split_cifar10</option>
            <option value="split_cifar100">split_cifar100</option>
            <option value="split_tinyimagenet">split_tinyimagenet</option>
          </select>
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Method</div>
          <select id="inj-method" style="width:100%;padding:6px">
            <option value="tcl">tcl</option>
            <option value="ewc">ewc</option>
            <option value="sgd_baseline">sgd_baseline</option>
          </select>
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Hypothesis</div>
          <input id="inj-hypothesis" style="width:100%;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px" value="human_hypothesis">
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Seeds</div>
          <input id="inj-seeds" style="width:100%;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px" value="42,0,1">
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Priority</div>
          <input id="inj-priority" type="number" style="width:100%;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px" value="40">
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Runtime Hours</div>
          <input id="inj-runtime" type="number" step="0.5" style="width:100%;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px" value="6">
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Frontier Problem</div>
          <input id="inj-frontier" style="width:100%;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px" value="fp-catastrophic-forgetting">
        </div>
        <div>
          <div class="modal-section-title" style="margin-bottom:4px">Paper Id</div>
          <input id="inj-paper" style="width:100%;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px" value="">
        </div>
      </div>
      <div class="modal-section" style="margin-top:12px">
        <div class="modal-section-title">Description</div>
        <textarea id="inj-description" style="width:100%;min-height:70px;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px">Human-injected experiment.</textarea>
      </div>
      <div class="modal-section">
        <div class="modal-section-title">Config Overrides JSON</div>
        <textarea id="inj-overrides" style="width:100%;min-height:90px;background:#09111d;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px">{}</textarea>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div id="inj-status" style="color:var(--text2);font-size:.62rem"></div>
        <button class="action-btn" onclick="submitInject()">Submit To Queue</button>
      </div>
    </div>
  </div>
</div>

<script>
// ── utils ──────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt  = (v, d=4) => v==null?'—':Number(v).toFixed(d);
const fmtP = v => v==null?'—':v<0.001?'<.001':Number(v).toFixed(3);
const pct  = v => v==null?'—':Math.round(Number(v)*100)+'%';
const delta = v => {
  if(v==null)return'—';
  const n=Number(v), col=n<0?'#34d399':n>0?'#f87171':'#94a3b8';
  return`<span style="color:${col}">${n>=0?'+':''}${n.toFixed(4)}</span>`;
};
function pill(s,label){
  const map={running:'s-run',complete:'s-done',failed:'s-fail',error:'s-fail',
    pending:'s-pend',stale:'s-pend',partial:'s-part',
    planned:'s-plan',queued:'s-queue',analyzing:'s-analyzing',writing_paper:'s-writing',
    stalled:'s-part',blocked:'s-part',hold:'s-pend',ready:'s-run',
    revision_requested:'s-analyzing',revising:'s-writing',revision_failed:'s-fail',
    BREAKTHROUGH:'s-bk',DIRECTIONAL:'s-run',NULL:'s-pend',ADVERSE:'s-fail',
    breakthrough:'s-bk',
  };
  return`<span class="pill ${map[s]||'s-pend'}">${label||s}</span>`;
}
function gaugeColor(pct){
  if(pct>=95)return'g-red';
  if(pct>=80)return'g-amber';
  return'g-green';
}
function setGauge(barId,valId,pct,label){
  const bar=$(`g-${barId}`), val=$(`gv-${barId}`);
  if(!bar)return;
  bar.style.width=Math.min(pct,100)+'%';
  bar.className='gauge-fill '+gaugeColor(pct);
  if(val)val.textContent=label||pct+'%';
}
function progressHtml(progress,color='var(--teal)'){
  if(!progress||progress.pct==null)return'';
  const pct=Math.max(0,Math.min(100,Number(progress.pct)||0));
  const label=progress.label||`${pct}%`;
  return `<div style="margin-top:6px">
    <div style="height:5px;background:var(--border);border-radius:999px;overflow:hidden">
      <div style="height:100%;width:${pct}%;background:${color};border-radius:999px;transition:width .35s ease"></div>
    </div>
    <div style="color:var(--text2);font-size:.58rem;margin-top:3px">${label}</div>
  </div>`;
}
const _DENSITY_KEY='tar_density_v2';
let _focusedPanelId='';
function currentDensity(){
  try{return localStorage.getItem(_DENSITY_KEY)||'comfortable';}catch{return'comfortable';}
}
function applyDensity(mode){
  const page=document.querySelector('.page');
  if(!page)return;
  page.classList.remove('comfortable','dense');
  page.classList.add(mode);
  const btn=$('density-btn');
  if(btn)btn.textContent=mode==='comfortable'?'Dense view':'Comfortable view';
  try{localStorage.setItem(_DENSITY_KEY,mode);}catch{}
}
function toggleDensity(){
  applyDensity(currentDensity()==='comfortable'?'dense':'comfortable');
}
function updateRestorePanelButton(){
  const btn=$('restore-panel-btn');
  if(btn)btn.style.display=_focusedPanelId?'':'none';
}
function updatePanelToolState(card){
  const collapseBtn=card.querySelector('.panel-btn-collapse');
  const focusBtn=card.querySelector('.panel-btn-focus');
  if(collapseBtn)collapseBtn.textContent=card.classList.contains('is-collapsed')?'+':'-';
  if(focusBtn)focusBtn.textContent=card.classList.contains('panel-focus')?'Restore':'Focus';
}
function scrollToPanel(id){
  const card=$(id);
  if(!card)return;
  card.classList.remove('is-collapsed');
  updatePanelToolState(card);
  card.scrollIntoView({behavior:'smooth',block:'start'});
}
function togglePanelCollapse(id){
  const card=$(id);
  if(!card||card.classList.contains('panel-focus'))return;
  card.classList.toggle('is-collapsed');
  updatePanelToolState(card);
}
function focusPanel(id){
  const card=$(id);
  if(!card)return;
  if(_focusedPanelId&&_focusedPanelId!==id)unfocusPanel();
  card.classList.remove('is-collapsed');
  card.classList.add('panel-focus');
  _focusedPanelId=id;
  $('panel-focus-backdrop')?.classList.add('open');
  document.body.classList.add('panel-focus-open');
  updatePanelToolState(card);
  updateRestorePanelButton();
}
function unfocusPanel(){
  if(!_focusedPanelId)return;
  const card=$(_focusedPanelId);
  if(card){
    card.classList.remove('panel-focus');
    updatePanelToolState(card);
  }
  _focusedPanelId='';
  $('panel-focus-backdrop')?.classList.remove('open');
  document.body.classList.remove('panel-focus-open');
  updateRestorePanelButton();
}
function togglePanelFocus(id){
  const card=$(id);
  if(!card)return;
  if(card.classList.contains('panel-focus'))unfocusPanel();
  else focusPanel(id);
}
function expandAllPanels(){
  document.querySelectorAll('.card[data-panel-title]').forEach(card=>{
    card.classList.remove('is-collapsed');
    updatePanelToolState(card);
  });
}
function collapseAllPanels(){
  document.querySelectorAll('.card[data-panel-title]').forEach(card=>{
    if(card.id!==_focusedPanelId)card.classList.add('is-collapsed');
    updatePanelToolState(card);
  });
}
function enhancePanels(){
  document.querySelectorAll('.card[data-panel-title]').forEach((card,idx)=>{
    if(!card.id)card.id=`panel-card-${idx}`;
    if(card.dataset.enhanced==='1')return;
    const hdr=card.querySelector('.hdr');
    if(!hdr)return;
    const tools=document.createElement('div');
    tools.className='panel-tools';
    tools.innerHTML=`<button class="panel-btn panel-btn-collapse" title="Collapse panel" onclick="event.stopPropagation();togglePanelCollapse('${card.id}')">-</button><button class="panel-btn panel-btn-focus" title="Focus panel" onclick="event.stopPropagation();togglePanelFocus('${card.id}')">Focus</button>`;
    hdr.appendChild(tools);
    hdr.addEventListener('dblclick',()=>togglePanelFocus(card.id));
    card.dataset.enhanced='1';
    updatePanelToolState(card);
  });
  applyDensity(currentDensity());
  updateRestorePanelButton();
}
const _BK_DISMISSED_KEY='tar_bk_dismissed_v1';
const _BK_SEEN_KEY='tar_bk_seen_v1';
let _breakthroughs=[];
let _expandedBreakthroughs=new Set();
let _expandedLearnedDomains=new Set();
let _expandedFrontiers=new Set();
let _learnedLiteraturePayload={};
let _frontierPayload=[];
let _coordinationPayload={};
let _directorPayload={};
let _statusPayload={};
let _schedulerPayload={};
let _expandedResultRows=new Set();
function loadLocalSet(key){
  try{return new Set(JSON.parse(localStorage.getItem(key)||'[]'));}catch{return new Set();}
}
function saveLocalSet(key,setObj){
  localStorage.setItem(key, JSON.stringify(Array.from(setObj)));
}
function markBreakthroughSeen(id){
  const seen=loadLocalSet(_BK_SEEN_KEY);
  seen.add(id);
  saveLocalSet(_BK_SEEN_KEY, seen);
}
function dismissBreakthrough(ev,id){
  ev.stopPropagation();
  const dismissed=loadLocalSet(_BK_DISMISSED_KEY);
  dismissed.add(id);
  saveLocalSet(_BK_DISMISSED_KEY, dismissed);
  updateBreakthroughs({breakthroughs:_breakthroughs});
}
function resetBreakthroughDismissals(){
  localStorage.removeItem(_BK_DISMISSED_KEY);
  updateBreakthroughs({breakthroughs:_breakthroughs});
}
function toggleBreakthrough(id){
  const expanded=new Set(_expandedBreakthroughs);
  if(expanded.has(id))expanded.delete(id);
  else expanded.add(id);
  _expandedBreakthroughs=expanded;
  markBreakthroughSeen(id);
  updateBreakthroughs({breakthroughs:_breakthroughs});
}
function toggleLearnedDomain(id){
  const expanded=new Set(_expandedLearnedDomains);
  if(expanded.has(id))expanded.delete(id);
  else expanded.add(id);
  _expandedLearnedDomains=expanded;
  updateLearnedLiterature(_learnedLiteraturePayload);
}
function toggleFrontierCard(id){
  const expanded=new Set(_expandedFrontiers);
  if(expanded.has(id))expanded.delete(id);
  else expanded.add(id);
  _expandedFrontiers=expanded;
  if(Array.isArray(_frontierPayload)){
    updateFrontier(_frontierPayload);
  }
}

// ── Hardware ──────────────────────────────────────────────────────────────────
function updateHardware(d){
  const gpu=d.gpu||{}, cpu=d.cpu||{}, ram=d.ram||{};
  const gpuPct=gpu.utilization_pct||0;
  const vramPct=gpu.vram_total_gb>0?Math.round(gpu.vram_used_gb/gpu.vram_total_gb*100):0;
  const cpuPct=cpu.utilization_pct||0;
  const ramPct=ram.percent||0;
  setGauge('gpu-util','gpu-util',gpuPct,gpuPct+'%');
  setGauge('vram','vram',vramPct,`${(gpu.vram_used_gb||0).toFixed(1)}/${(gpu.vram_total_gb||0).toFixed(0)}GB`);
  setGauge('cpu','cpu',cpuPct,cpuPct+'%');
  setGauge('ram','ram',ramPct,`${(ram.used_gb||0).toFixed(1)}/${(ram.total_gb||0).toFixed(0)}GB`);
  const gtEl=$('gv-gpu-temp');
  if(gtEl&&gpu.temperature_c){
    const tc=gpu.temperature_c;
    const tc_col=tc>85?'#f87171':tc>70?'#fbbf24':'#5a7a9a';
    gtEl.innerHTML=`<span style="color:${tc_col}">${tc}°C ${gpu.name?'GPU':''}</span>`;
  }
  const ctEl=$('gv-cpu-temp');
  if(ctEl&&cpu.temperature_c)ctEl.textContent=cpu.temperature_c+'°C CPU';
  // Processes
  const procs=(d.processes||[]).filter(p=>p.cpu_pct>0.5||p.ram_gb>0.1);
  $('proc-count').textContent=procs.length+'';
  $('proc-panel').innerHTML=procs.length?procs.map(p=>{
    const expTag=p.experiment_id?`<span style="color:var(--teal)">[${p.experiment_id.slice(0,8)}]</span> `:'';
    return`<div style="padding:2px 0;border-bottom:1px solid var(--border2)">
      ${expTag}<span style="color:var(--text);font-size:.62rem">${p.cmd_short.slice(-45)}</span>
      <span style="color:var(--text2);font-size:.58rem;float:right">CPU ${p.cpu_pct.toFixed(0)}% RAM ${p.ram_gb.toFixed(1)}GB VRAM ${(p.vram_gb||0).toFixed(1)}GB</span>
    </div>`;
  }).join(''):'<span style="color:var(--text3);font-size:.62rem">No active processes</span>';
}

// ── Status / Queue ─────────────────────────────────────────────────────────────
function updateStatus(d){
  _statusPayload=d||{};
  $('active-step-label').textContent=d.active_step||'';
  $('last-update').textContent='↻ '+new Date(d.timestamp).toLocaleTimeString();
  const icons={running:'▶ ',complete:'✓ ',failed:'✗ ',pending:'· ',stalled:'~ ',stale:'~ '};
  const cols={running:'#34d399',complete:'#60a5fa',failed:'#f87171',pending:'#334a62',stalled:'#92400e',stale:'#92400e'};
  const baseQueue=(d.queue_steps||[]).map(s=>{
    const ic=icons[s.status]||'· ', c=cols[s.status]||'#334a62';
    const kb=s.size_kb>0?`<span style="color:var(--text3);font-size:.58rem">${s.size_kb}kb</span>`:'';
    return`<div style="display:flex;align-items:center;gap:5px;padding:4px 0;border-bottom:1px solid var(--border2)">
      <span style="color:${c};font-weight:700;width:12px">${ic}</span>
      <span style="color:${c};flex:1;font-size:.68rem">${s.label}</span>${kb}</div>`;
  }).join('')||'—';
  const directorAgenda=((_directorPayload.experiment_directives||[]).filter(rec=>{
    const status=String(rec.status||rec.scheduler_intent||'');
    return !['complete','archive','archived','done'].includes(status);
  }).slice(0,5)).map(rec=>{
    const dataset=rec.dataset?String(rec.dataset).replace('split_',''):'experiment';
    const targetPaper=rec.target_paper_id?` · ${rec.target_paper_id}`:'';
    return `<div style="padding:4px 0;border-bottom:1px solid var(--border2)">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.scheduler_intent||rec.status, rec.scheduler_intent||rec.status)}
        <span style="color:var(--text);font-size:.64rem;flex:1">${rec.title||rec.experiment_id}</span>
      </div>
      <div class="mini-note">rank ${rec.scheduler_rank||'—'} · ${dataset}${targetPaper}</div>
    </div>`;
  }).join('');
  $('queue-panel').innerHTML=baseQueue + (directorAgenda
    ? `<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Director agenda</div>${directorAgenda}</div>`
    : '');
}

// ── Experiments ───────────────────────────────────────────────────────────────
function updateExperiments(data){
  const exps=data.experiments||[];
  const running=exps.filter(e=>(e.stage||e.status)==='running');
  const queuedLike=exps.filter(e=>['stalled','queued','pending','planned','analyzing','writing_paper'].includes(e.stage||e.status));
  const done=exps.filter(e=>['complete','failed','skipped'].includes(e.status)||['complete','failed'].includes(e.stage));
  $('exp-summary').textContent=`${running.length} running  ${queuedLike.length} queued  ${done.length} done  total=${data.total||0}`;
  if(!exps.length){$('exp-panel').innerHTML='<div style="padding:12px;color:var(--text3);font-size:.65rem">No experiments in queue</div>';return;}
  const stageRank={
    running:0,
    stalled:1,
    queued:2,
    pending:2,
    planned:3,
    analyzing:4,
    writing_paper:5,
    complete:6,
    failed:7,
    skipped:8,
  };
  const ordered=[...exps].sort((a,b)=>{
    const aStage=a.stage||a.status||'';
    const bStage=b.stage||b.status||'';
    const aRank=stageRank.hasOwnProperty(aStage)?stageRank[aStage]:99;
    const bRank=stageRank.hasOwnProperty(bStage)?stageRank[bStage]:99;
    if(aRank!==bRank)return aRank-bRank;
    const aPriority=Number(a.priority??50);
    const bPriority=Number(b.priority??50);
    if(aPriority!==bPriority)return aPriority-bPriority;
    const aTs=a.started_at||a.submitted_at||'';
    const bTs=b.started_at||b.submitted_at||'';
    return aTs.localeCompare(bTs);
  });
  $('exp-panel').innerHTML=ordered.map(e=>{
    const prog=e.progress||{};
    const seedsDone=prog.seeds_done||0, seedsTotal=prog.seeds_total||e.seeds?.length||5;
    const progPct=seedsTotal>0?Math.round(seedsDone/seedsTotal*100):0;
    const vram=(e.hardware_budget||{}).vram_gb||0;
    const ds=(e.dataset||'').replace('split_','');
    return`<div class="exp-row" onclick="showModal('${e.id}')">
      <div style="display:flex;align-items:center;gap:6px">
        ${pill(e.stage||e.status)}
        <span class="exp-name">${e.name}</span>
        <span style="color:var(--text2);font-size:.58rem;margin-left:auto">${ds}${vram?'  '+vram+'GB':''}</span>
      </div>
      <div class="exp-meta">${e.context_why?e.context_why.slice(0,90)+'…':e.method+' · seeds='+JSON.stringify(e.seeds)}</div>
      ${['running','stalled'].includes(e.stage||e.status)?`
      <div style="display:flex;align-items:center;gap:6px;margin-top:4px">
        <div class="exp-progress" style="flex:1"><div class="exp-progress-fill" style="width:${progPct}%"></div></div>
        <span style="color:var(--text2);font-size:.58rem">${seedsDone}/${seedsTotal} seeds</span>
      </div>
      ${e.projected_outcome?`<div style="color:var(--teal);font-size:.6rem;margin-top:2px">${e.projected_outcome}</div>`:''}
      `:''}
    </div>`;
  }).join('');
}

// ── Modal ──────────────────────────────────────────────────────────────────────
let _currentExpId='';
let _currentExpLogSource='';
let _currentExpOpenLogName='';
function showModal(expId){
  _currentExpId=expId;
  $('modal-overlay').classList.add('open');
  fetch(`/api/experiment/${expId}`).then(r=>r.json()).then(renderModal).catch(()=>{});
  loadExperimentLog(expId,_currentExpLogSource);
}
function closeModal(){
  $('modal-overlay').classList.remove('open');
  _currentExpId='';
  _currentExpLogSource='';
  _currentExpOpenLogName='';
}
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){
    unfocusPanel();
    closeModal();
    closeInjectModal();
  }
});

function openInjectModal(){
  $('inj-status').textContent='';
  $('inject-modal-overlay').classList.add('open');
}
function closeInjectModal(){
  $('inject-modal-overlay').classList.remove('open');
}
async function submitInject(){
  const payload={
    name:$('inj-name').value.trim(),
    project_id:$('inj-project').value.trim(),
    dataset:$('inj-dataset').value,
    method:$('inj-method').value,
    hypothesis_name:$('inj-hypothesis').value.trim(),
    seeds:$('inj-seeds').value.trim(),
    priority:Number($('inj-priority').value||50),
    estimated_runtime_h:Number($('inj-runtime').value||6),
    frontier_problem_id:$('inj-frontier').value.trim(),
    author_paper_id:$('inj-paper').value.trim(),
    description:$('inj-description').value.trim(),
    config_overrides:$('inj-overrides').value.trim(),
  };
  $('inj-status').textContent='Submitting...';
  try{
    const res=await fetch('/api/experiments/inject',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload),
    });
    const data=await res.json();
    if(!res.ok||!data.ok){
      throw new Error(data.error||'Submit failed');
    }
    $('inj-status').textContent=`Queued ${data.name} (${data.experiment_id})`;
    refresh();
    setTimeout(closeInjectModal,800);
  }catch(err){
    $('inj-status').textContent=`Error: ${err.message}`;
  }
}

function loadExperimentLog(expId,source=''){
  const qs=source?`?source=${encodeURIComponent(source)}`:'';
  fetch(`/api/experiment/${expId}/log${qs}`)
    .then(r=>r.json())
    .then(renderExperimentLog)
    .catch(()=>{
      $('m-log-box').textContent='(unable to load experiment log)';
    });
}

function changeExperimentLogSource(source){
  _currentExpLogSource=source||'';
  if(_currentExpId)loadExperimentLog(_currentExpId,_currentExpLogSource);
}

function renderExperimentLog(d){
  const sel=$('m-log-source');
  const sources=d.sources||[];
  const selected=d.source_id||_currentExpLogSource||'';
  sel.innerHTML=sources.map(src=>
    `<option value="${src.source_id}" ${src.source_id===selected?'selected':''}>${src.label}</option>`
  ).join('') || '<option value="">No log source</option>';
  _currentExpLogSource=sel.value||selected||'';
  _currentExpOpenLogName=d.open_log_name||'';
  $('m-log-mtime').textContent=d.mtime?`updated ${d.mtime}`:'';
  $('m-log-path').textContent=d.path||'';
  $('m-log-box').textContent=(d.lines||['(no experiment log yet)']).join('\n');
  const btn=$('m-open-log-btn');
  btn.disabled=!_currentExpOpenLogName;
  btn.style.opacity=_currentExpOpenLogName?'1':'.6';
}

function openExperimentLogPanel(){
  focusPanel('log-card');
  if(!_currentExpOpenLogName)return;
  const sel=$('log-select');
  if(sel && !Array.from(sel.options).some(o=>o.value===_currentExpOpenLogName)){
    const opt=document.createElement('option');
    opt.value=_currentExpOpenLogName;
    opt.textContent=_currentExpOpenLogName;
    sel.appendChild(opt);
  }
  if(sel)sel.value=_currentExpOpenLogName;
  onLogSelect(_currentExpOpenLogName);
}

function renderModal(d){
  const spec=d.spec||{}, res=d.result||null;
  const ctx=spec.context||{};
  const prog=spec.progress||{};
  const budget=spec.hardware_budget||{};

  $('m-name').textContent=spec.name||'Experiment';
  $('m-stage-pill').innerHTML=pill(spec.stage||spec.status);
  $('m-meta').textContent=`${(spec.dataset||'').replace('split_','')}  ·  ${spec.method||''}  ·  seeds ${JSON.stringify(spec.seeds||[])}  ·  priority ${spec.priority||'—'}  ·  est ${(spec.estimated_runtime_h||0).toFixed(1)}h`;
  $('m-why').textContent=ctx.why||spec.description||'No context available.';
  $('m-hypothesis').textContent=ctx.hypothesis||'—';

  const seedsDone=prog.seeds_done||0, seedsTotal=prog.seeds_total||spec.seeds?.length||5;
  const progPct=seedsTotal>0?Math.round(seedsDone/seedsTotal*100):0;
  $('m-progress-fill').style.width=progPct+'%';
  const fSoFar=(prog.forgetting_so_far||[]).map(v=>v.toFixed(4)).join(', ');
  const currentSeed=prog.current_seed!=null?` · current seed ${prog.current_seed}`:'';
  const currentMethod=prog.current_method?` · method ${prog.current_method}`:'';
  const checkpointAt=prog.last_checkpoint_at?` · checkpoint ${prog.last_checkpoint_at}`:'';
  const initializing=prog.initializing?' · live warmup / awaiting first checkpoint':'';
  const liveCompute=prog.live_compute_note?` · ${prog.live_compute_note}`:'';
  $('m-progress-text').innerHTML=
    `Seed <strong>${seedsDone}</strong>/${seedsTotal} complete`+
    (fSoFar?` · forgetting so far: [${fSoFar}]`:'')+
    (prog.tasks_done?` · task ${prog.tasks_done}/10`:'')+
    currentSeed+currentMethod+checkpointAt+initializing+liveCompute;

  $('m-projected').textContent=ctx.projected_outcome||'Awaiting first seed results.';
  $('m-frontier').textContent=ctx.frontier_problem||spec.frontier_problem_id||'—';
  $('m-paper').textContent=ctx.feeds_paper||spec.author_paper_id||'—';
  $('m-methodology').textContent=ctx.methodology_note||'—';

  // Result
  if(res){
    $('m-result-section').style.display='';
    $('m-result-text').innerHTML=
      `Verdict: ${pill(res.verdict)}  `+
      `forgetting=${fmt(res.mean_forgetting)}  `+
      `delta=${delta(res.mean_delta)}  `+
      `p=${fmtP(res.p_val)}  d=${fmt(res.cohens_d,2)}  `+
      `${res.n_better||'—'}/${(res.seeds||res.seed_results||[]).length} seeds better`;
  } else {
    $('m-result-section').style.display='none';
  }

  // Hardware
  $('m-hardware').innerHTML=
    `<span>VRAM budget: <strong>${budget.vram_gb||'?'} GB</strong></span>`+
    `<span>CPU cores: <strong>${budget.cpu_cores||'?'}</strong></span>`+
    (spec.pid?`<span>PID: <strong>${spec.pid}</strong></span>`:'');
}

// ── Frontier ──────────────────────────────────────────────────────────────────
function updateFrontier(problems){
  _frontierPayload=problems||[];
  $('fp-count').textContent=problems.length+'';
  if(!problems.length){$('fp-panel').innerHTML='<div style="padding:8px;color:var(--text3)">Seeding frontier problems…</div>';return;}
  const statusPill={active:'s-run',publishing:'s-bk',exploring:'s-pend',complete:'s-done'};
  $('fp-panel').innerHTML=problems.map(p=>{
    const id=String(p.id||p.problem_id||p.title||'frontier');
    const expanded=_expandedFrontiers.has(id);
    const bk=p.breakthroughs_found?`<span class="pill s-bk" style="margin-left:6px">${p.breakthroughs_found} BK</span>`:'';
    const evidence=p.evidence_strength?`<span class="pill ${p.truth_status==='validated'?'s-run':p.truth_status==='supported'?'s-analyzing':'s-pend'}">${p.evidence_strength}</span>`:'';
    const next=p.next_action?`<div class="director-action">${p.next_action}</div>`:'';
    const why=p.why_now?`<div class="mini-note" style="margin-top:4px">${p.why_now}</div>`:'';
    const notes=(p.evidence_notes||[]).map(note=>`<div class="learned-claim">• ${note}</div>`).join('');
    const waits=(p.waiting_on_experiment_ids||[]).length?`<div class="mini-note" style="margin-top:4px;color:var(--amber)">Waiting on: ${(p.waiting_on_experiment_ids||[]).join(', ')}</div>`:'';
    const linked=(p.linked_experiment_ids||[]).length?`<div class="mini-note" style="margin-top:4px">Experiments: ${(p.linked_experiment_ids||[]).join(', ')}</div>`:'';
    const standard=p.verification_standard?`<div class="learned-subtitle">Verification standard</div><div class="mini-note">${p.verification_standard}</div>`:'';
    const results=(p.result_evidence||[]).map(rec=>{
      const deltaText=rec.mean_delta!=null?`Δ=${Number(rec.mean_delta)>=0?'+':''}${Number(rec.mean_delta).toFixed(4)}`:'Δ=—';
      const pText=rec.p_val!=null?`p=${Number(rec.p_val)<0.001?'<.001':Number(rec.p_val).toFixed(3)}`:'p=—';
      const dText=rec.cohens_d!=null?`d=${Number(rec.cohens_d).toFixed(2)}`:'d=—';
      return `<div class="learned-claim"><span style="color:var(--teal)">${rec.label||'result'}</span> · ${(rec.verdict||'').toLowerCase()||'result'} · ${deltaText} · ${pText} · ${dText}<br><span style="color:var(--text3)">${rec.result_path||''}</span>${rec.notes?`<br><span style="color:var(--text2)">${rec.notes}</span>`:''}</div>`;
    }).join('');
    return`<div class="fp-card ${expanded?'expanded':''}" onclick="toggleFrontierCard('${id.replace(/'/g,'\\\'')}')">
      <div style="display:flex;align-items:center;gap:6px">
        <span class="pill ${statusPill[p.status]||'s-pend'}">${p.status}</span>
        <span class="fp-title">${p.title}</span>${bk}${evidence}
      </div>
      <div class="fp-meta">
        <span>${(p.domain||'').replace(/_/g,' ')}</span>
        <span>${p.experiments_linked?.length||0} experiments</span>
        <span>${p.papers_linked?.length||0} papers</span>
      </div>
      ${next}
      ${why}
      ${expanded?`<div class="learned-body" style="margin-top:6px">
        ${waits}
        ${linked}
        ${notes?`<div class="learned-subtitle">Evidence notes</div>${notes}`:''}
        ${standard}
        ${(p.result_evidence||[]).length?`<div class="learned-subtitle">Result evidence</div>${results}`:''}
      </div>`:''}
    </div>`;
  }).join('');
}

// ── Scheduler ─────────────────────────────────────────────────────────────────
function updateScheduler(data){
  const coordination=_coordinationPayload||{};
  const canStart=(data.experiments_to_start||data.can_start||[]);
  const running=(data.running_ids||[]).map(id=>`<div class="learned-claim">• running: ${id}</div>`).join('');
  const next=data.next_experiment_id?`<div class="mini-note" style="margin-top:4px">Next clear start: <span style="color:var(--teal)">${data.next_experiment_id}</span></div>`:'';
  const starts=canStart.length?`<div class="mini-note" style="margin-top:4px">Can start now: ${canStart.join(', ')}</div>`:'';
  const holds=(data.hold_reasons||[]).slice(0,5).map(rec=>`<div class="learned-claim">• ${rec.experiment_id||rec.experiment_name}: ${rec.reason||''}</div>`).join('');
  const papers=(coordination.allowed_paper_ids||[]).length?`<div class="mini-note" style="margin-top:4px">Active papers: ${(coordination.allowed_paper_ids||[]).join(', ')}</div>`:'';
  $('scheduler-panel').innerHTML=`
    <div>${data.rationale||'No scheduler data yet.'}</div>
    ${next}
    ${starts}
    ${papers}
    ${running?`<div class="learned-subtitle">Running</div>${running}`:''}
    ${holds?`<div class="learned-subtitle">Hold reasons</div>${holds}`:''}
  `;
}

function updateCoordination(data){
  _coordinationPayload=data||{};
}

function _deprecatedUpdateDirectorStage1(data){
  _directorPayload=data||{};
  if(_statusPayload&&Object.keys(_statusPayload).length)updateStatus(_statusPayload);
  const directives=data.frontier_directives||[];
  const experimentDirectives=data.experiment_directives||[];
  const activePaths=data.active_research_paths||[];
  const domains=data.knowledge_domains||[];
  $('director-count').textContent=`${directives.length}/${domains.length}`;
  if(!directives.length){
    $('director-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Director is building the frontier map…</span>';
    return;
  }
  const experimentAgenda=experimentDirectives.slice(0,8).map(rec=>{
    const blocked=(rec.blocked_by_experiment_ids||[]).length?` · blocked ${(rec.blocked_by_experiment_ids||[]).length}`:'';
    const dataset=rec.dataset?String(rec.dataset).replace('split_',''):'experiment';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.scheduler_intent||rec.status, rec.scheduler_intent||rec.status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'—'} · ${dataset} · score ${rec.priority_score||'—'}${blocked}</div>
      <div class="director-action">${rec.experiment_goal||rec.mechanism_focus||rec.why_now||''}</div>
    </div>`;
  }).join('');
  const frontierPriority=directives.map(rec=>{
    const wait=(rec.waiting_on_experiment_ids||[]).length?` · wait ${(rec.waiting_on_experiment_ids||[]).length}`:'';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.truth_status, rec.truth_status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'—'} · score ${rec.priority_score||'—'}${wait}</div>
      <div class="director-action">${rec.next_action||''}</div>
    </div>`;
  }).join('');
  const pathHtml=activePaths.slice(0,5).map(path=>
    `<div class="mini-note" style="margin-top:3px"><span style="color:var(--teal)">${path.title}</span> · ${path.status} · score ${path.priority_score||'—'}</div>`
  ).join('');
  const weak=domains.filter(d=>d.status==='candidate'||d.status==='seed').map(d=>
    `<div class="mini-note"><span style="color:var(--amber)">${d.label}</span> · ${d.next_action}</div>`
  ).join('');
  $('director-panel').innerHTML=
    (experimentAgenda?`<div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Planned experiments</div>${experimentAgenda}`:'')+
    `<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Frontier priorities</div>${frontierPriority}</div>`+
    (pathHtml?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Active research paths</div>${pathHtml}</div>`:'')+
    (weak?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Weak / expanding domains</div>${weak}</div>`:'');
}

// ── TAR Author ────────────────────────────────────────────────────────────────
function updateAuthor(data){
  const cp=data.current_paper||{};
  const log=(data.activity_log||[]);
  if(!cp.project_id){$('author-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Idle — waiting for experiments to complete.</span>';return;}
  const sectionIcon={writing:'✍',done:'✓',starting:'▶',complete:'✓',waiting:'⏳',revision_requested:'↺',revising:'✍',revision_failed:'⚠'};
  const ic=sectionIcon[cp.status]||'·';
  const waitHtml=cp.waiting_for_experiments?.length?
    `<div style="color:var(--amber);font-size:.6rem;margin-top:3px">⏳ waiting for: ${cp.waiting_for_experiments.join(', ')}</div>`:'';
  const logHtml=log.map(l=>`<div style="color:var(--text3);font-size:.58rem">${l.timestamp?.slice(11,19)||''} ${l.action} ${l.section||''}</div>`).join('');
  const suggestionHtml=(data.suggested_frontier_papers||[]).map(p=>
    `<div class="mini-note" style="margin-top:3px"><span style="color:var(--teal)">${p.title||p.paper_id}</span> · ${p.recommendation||p.readiness||''}</div>`
  ).join('');
  $('author-panel').innerHTML=`
    <div style="color:var(--text);font-size:.68rem;font-weight:600;margin-bottom:4px">${ic} ${cp.title||'—'}</div>
    <div style="color:var(--text2);font-size:.62rem">Section: <span style="color:var(--teal)">${cp.section_in_progress||'—'}</span></div>
    <div style="color:var(--text2);font-size:.6rem;margin-top:2px">Done: ${(cp.sections_complete||[]).join(', ')||'—'}</div>
    ${cp.director_recommendation?`<div class="mini-note" style="margin-top:3px">${cp.director_recommendation}</div>`:''}
    ${waitHtml}
    ${suggestionHtml?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Next frontier papers</div>${suggestionHtml}</div>`:''}
    <div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px">${logHtml}</div>
  `;
}

function _deprecatedUpdateDirectorStage2(data){
  const directives=data.frontier_directives||[];
  const experimentDirectives=data.experiment_directives||[];
  const domains=data.knowledge_domains||[];
  $('director-count').textContent=`${directives.length}/${domains.length}`;
  const experimentAgendaLegacy=experimentDirectives.slice(0,8).map(rec=>{
    const blocked=(rec.blocked_by_experiment_ids||[]).length?` Â· blocked ${(rec.blocked_by_experiment_ids||[]).length}`:'';
    const dataset=rec.dataset?rec.dataset.replace('split_',''):'experiment';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.scheduler_intent||rec.status, rec.scheduler_intent||rec.status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'â€”'} Â· ${dataset} Â· score ${rec.priority_score||'â€”'}${blocked}</div>
      <div class="director-action">${rec.experiment_goal||rec.mechanism_focus||rec.why_now||''}</div>
    </div>`;
  }).join('');
  if(!directives.length){
    $('director-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Director is building the frontier map…</span>';
    return;
  }
  const experimentAgenda=experimentDirectives.slice(0,8).map(rec=>{
    const blocked=(rec.blocked_by_experiment_ids||[]).length?` · blocked ${(rec.blocked_by_experiment_ids||[]).length}`:'';
    const dataset=rec.dataset?rec.dataset.replace('split_',''):'experiment';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.scheduler_intent||rec.status, rec.scheduler_intent||rec.status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'—'} · ${dataset} · score ${rec.priority_score||'—'}${blocked}</div>
      <div class="director-action">${rec.experiment_goal||rec.mechanism_focus||rec.why_now||''}</div>
    </div>`;
  }).join('');
  const top=directives.map(rec=>{
    const wait=(rec.waiting_on_experiment_ids||[]).length?` · wait ${(rec.waiting_on_experiment_ids||[]).length}`:'';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.truth_status, rec.truth_status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'—'} · score ${rec.priority_score||'—'}${wait}</div>
      <div class="director-action">${rec.next_action||''}</div>
    </div>`;
  }).join('');
  const weak=domains.filter(d=>d.status==='candidate'||d.status==='seed').map(d=>
    `<div class="mini-note"><span style="color:var(--amber)">${d.label}</span> · ${d.next_action}</div>`
  ).join('');
  $('director-panel').innerHTML=top + (weak?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Weak / expanding domains</div>${weak}</div>`:'');
}

// ── Log ────────────────────────────────────────────────────────────────────────
let _selLog='experiment_orchestrator';
function onLogSelect(v){_selLog=v;loadLog(v);}
function renderLogLines(lines){
  const box=$('log-box');
  const atBottom=box.scrollHeight-box.clientHeight<=box.scrollTop+60;
  box.innerHTML='';
  for(const raw of lines){
    const span=document.createElement('span');
    if(/seed=\S+.*epoch\s+\d+/i.test(raw)){
      span.style.cssText='color:#4ade80;font-weight:600';
      span.textContent=raw;
    } else if(/error|traceback|exception|failed/i.test(raw)){
      span.style.color='#f87171';
      span.textContent=raw;
    } else if(/\[execute\]|\[manifest\]|\[preflight\]/i.test(raw)){
      span.style.color='#38bdf8';
      span.textContent=raw;
    } else {
      span.textContent=raw;
    }
    box.appendChild(span);
    box.appendChild(document.createTextNode('\n'));
  }
  if(atBottom)box.scrollTop=box.scrollHeight;
}
function loadLog(name){
  const n=name||_selLog||'';
  fetch(n?`/api/log/${n}`:'/api/log')
    .then(r=>r.json()).then(d=>{
      $('log-label').textContent=d.label||'—';
      $('log-mtime').textContent=d.mtime||'';
      renderLogLines(d.lines||[]);
    }).catch(()=>{});
}
function refreshLogList(){
  fetch('/api/logs').then(r=>r.json()).then(logs=>{
    const sel=$('log-select');
    const have=new Set(Array.from(sel.options).map(o=>o.value));
    for(const l of logs){
      if(!have.has(l.name)){
        const o=document.createElement('option');
        o.value=l.name;o.textContent=`${l.name} (${l.size_kb}kb)`;
        sel.appendChild(o);
      }
    }
  }).catch(()=>{});
}

// ── Phases ────────────────────────────────────────────────────────────────────
function updatePhases(phases){
  if(!phases.length){$('phases-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">No results yet</span>';return;}
  const vkShort={OUTCOME_A:'A✓',OUTCOME_B:'B✓',OUTCOME_C:'C',ALL_DEGENERATE:'DEG',
    PARTIAL_RECOVERY:'P-REC',FULL_RECOVERY:'REC',BREAKTHROUGH:'BK!'};
  $('phases-panel').innerHTML=`<table><thead><tr><th>Phase</th><th>Forg</th><th>p</th><th>d</th><th>Result</th></tr></thead><tbody>`+
  phases.map(p=>{
    const isBk=p.status==='breakthrough';
    const vs=vkShort[p.verdict_key]||(p.status==='error'?'ERR':p.verdict_key||'—');
    const ds=(p.dataset||'').replace('split_','');
    return`<tr>
      <td style="color:var(--text2)">P${p.phase} <span style="color:var(--text3);font-size:.55rem">${ds}</span></td>
      <td style="color:var(--sky)">${fmt(p.tcl_forgetting)}</td>
      <td>${fmtP(p.vs_ewc_p)}</td>
      <td>${fmt(p.vs_ewc_d,2)}</td>
      <td ${isBk?'class="bk-flash"':''}>${vs}</td>
    </tr>`;
  }).join('')+'</tbody></table>';
}

function updateResults(payload){
  const rows=payload.results||[];
  $('results-count').textContent=`${payload.running||0} live · ${payload.breakthroughs||0} BK · ${payload.total||0} total`;
  if(!rows.length){$('phases-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">No experiment results yet</span>';return;}
  $('phases-panel').innerHTML=rows.map(rec=>{
    const verdict=String(rec.verdict||rec.status||'').toUpperCase();
    const verdictLabel=verdict||'—';
    const dataset=String(rec.dataset||'').replace('split_','');
    const program=String(rec.program||rec.source_kind||'experiment');
    const metaBits=[program,dataset].filter(Boolean).join(' · ');
    const metrics=[
      rec.tcl_forgetting!=null?`forg ${fmt(rec.tcl_forgetting)}`:'',
      rec.mean_delta!=null?`Δ ${delta(rec.mean_delta)}`:'',
      rec.vs_ewc_p!=null?`p ${fmtP(rec.vs_ewc_p)}`:'',
      rec.vs_ewc_d!=null?`d ${fmt(rec.vs_ewc_d,2)}`:''
    ].filter(Boolean).join(' · ');
    const path=rec.result_path?`<div class="result-note" style="color:var(--text3)">path: ${rec.result_path}</div>`:'';
    return `<div class="result-row">
      <div style="display:flex;align-items:center;gap:6px">
        ${pill(rec.status||verdictLabel, rec.status||verdictLabel)}
        <span class="result-title">${rec.title||rec.label||rec.id||'Result'}</span>
        ${verdictLabel?`<span class="${verdictLabel.includes('BREAKTHROUGH')?'bk-flash':''}" style="color:${verdictLabel.includes('BREAKTHROUGH')?'var(--pink)':'var(--text2)'};font-size:.58rem;margin-left:auto">${verdictLabel}</span>`:''}
      </div>
      <div class="result-meta">${metaBits || 'experiment result'}${rec.completed_at?` · ${rec.completed_at}`:''}</div>
      ${metrics?`<div class="result-metrics">${metrics}</div>`:''}
      ${rec.notes?`<div class="result-note">${rec.notes}</div>`:''}
      ${path}
    </div>`;
  }).join('');
}

// ── Registry ──────────────────────────────────────────────────────────────────
function updateRegistry(projects){
  $('reg-count').textContent=projects.length+'';
  if(!projects.length){$('registry-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Populated after TAR-Author runs</span>';return;}
  $('registry-panel').innerHTML=projects.map(p=>{
    const pdfMark=p.has_pdf?'<span style="color:var(--green);font-size:.55rem"> PDF</span>':'';
    const texMark=p.has_tex?'<span style="color:var(--sky);font-size:.55rem"> TEX</span>':'';
    const waitCount=(p.waiting_for_experiment_ids||[]).length;
    return`<div style="padding:4px 0;border-bottom:1px solid var(--border2)">
      <div style="display:flex;align-items:center;gap:4px">
        ${pill(p.status)}
        <span style="color:var(--sky);font-size:.65rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.name}</span>${pdfMark}${texMark}
      </div>
      <div style="color:var(--text3);font-size:.58rem;margin-top:1px">${p.field}/${p.subfield} · ${p.dataset} · ${p.paper_status||p.readiness||'planned'}</div>
      ${p.director_recommendation?`<div style="color:var(--text2);font-size:.58rem;margin-top:2px;line-height:1.45">${p.director_recommendation}</div>`:''}
      ${waitCount?`<div style="color:var(--amber);font-size:.56rem;margin-top:2px">waiting on ${waitCount} experiment${waitCount>1?'s':''}</div>`:''}
    </div>`;
  }).join('');
}

// ── Autonomous Research ───────────────────────────────────────────────────────
function updateAR(data){
  const hyps=data.hypotheses||[];
  const done=hyps.filter(h=>h.status==='complete').length;
  const bk=hyps.filter(h=>h.verdict==='BREAKTHROUGH').length;
  $('ar-count').textContent=`${done}/${hyps.length}${bk?' · '+bk+' BK':''}`;
  $('ar-panel').innerHTML=hyps.map(h=>{
    const v=h.verdict||h.status;
    const isBk=h.verdict==='BREAKTHROUGH';
    const n=h.name.replace(/_/g,' ');
    return`<div style="padding:4px 0;border-bottom:1px solid var(--border2)">
      <div style="display:flex;align-items:center;gap:4px">
        ${pill(v)}
        <span style="${isBk?'animation:bkf .8s ease-in-out infinite alternate':'color:var(--text)'};font-size:.68rem;flex:1">${n}</span>
        <span style="color:var(--text3);font-size:.58rem">${h.seeds_done||0}/5</span>
      </div>
      <div style="color:var(--text3);font-size:.6rem;margin-top:1px">
        forg=${fmt(h.mean_forgetting)} Δ=${delta(h.mean_delta)} p=${fmtP(h.p_val)}
      </div>
    </div>`;
  }).join('')||'<span style="color:var(--text3);font-size:.62rem">Not started</span>';
}

// ── Breakthroughs ─────────────────────────────────────────────────────────────
function updateBreakthroughs(data){
  const bks=data.breakthroughs||[];
  _breakthroughs=bks;
  const banner=$('bk-banner');
  const dismissed=loadLocalSet(_BK_DISMISSED_KEY);
  const seen=loadLocalSet(_BK_SEEN_KEY);
  const visible=bks.filter(b=>!dismissed.has(b.notification_id||''));
  $('bk-reset-btn').style.display=dismissed.size?'':'none';
  if(!visible.length){banner.style.display='none';return;}
  banner.style.display='block';
  $('bk-count').textContent=`${visible.length} breakthrough${visible.length>1?'s':''}`;
  const srcLabel={autonomous_research:'AR',phase_result:'Phase',external_breakthrough:'Ext'};
  $('bk-list').innerHTML=visible.map(b=>{
    const id=b.notification_id||b.project_id||b.name;
    const expanded=_expandedBreakthroughs.has(id);
    const isNew=!seen.has(id);
    const d=b.mean_delta!=null?(Number(b.mean_delta)>=0?'+':'')+Number(b.mean_delta).toFixed(4):'—';
    const dCol=b.mean_delta!=null&&Number(b.mean_delta)<0?'#34d399':'#f87171';
    const pdfLink=b.serve_pdf?`<a class="paper-link" href="/serve/paper/${b.serve_pdf}" target="_blank" style="color:var(--sky);font-size:.6rem">[PDF]</a>`:'';
    return`<div class="bk-item ${expanded?'expanded':''}" onclick="toggleBreakthrough('${id.replace(/'/g,'\\\'')}')">
      <div style="display:flex;align-items:center;gap:6px">
        <span style="color:var(--pink);font-weight:700;font-size:.72rem" class="bk-flash">${b.name}</span>
        <span style="color:var(--text2);font-size:.58rem">${srcLabel[b.source]||b.source}</span>
        ${isNew?'<span class="pill s-bk">new</span>':''}
        ${pdfLink}
        <button class="bk-dismiss" onclick="dismissBreakthrough(event,'${id.replace(/'/g,'\\\'')}')">hide</button>
      </div>
      <div style="font-size:.6rem;color:var(--text2);margin-top:1px">
        <span style="color:var(--sky)">${(b.dataset||'').replace('split_','')}</span>
        &nbsp;Δ=<span style="color:${dCol}">${d}</span>
        &nbsp;p=${b.p_val!=null?(Number(b.p_val)<0.001?'<.001':Number(b.p_val).toFixed(3)):'—'}
        &nbsp;d=${b.cohens_d!=null?Number(b.cohens_d).toFixed(2):'—'}
        ${b.found_at?'&nbsp;'+b.found_at:''}
      </div>
      ${expanded?`<div style="font-size:.6rem;color:var(--text);margin-top:5px;line-height:1.55">${b.summary||''}</div>`:''}
      ${expanded&&b.notes?`<div style="font-size:.58rem;color:#f9a8d4;margin-top:3px;line-height:1.55">${b.notes}</div>`:''}
      ${!expanded&&b.notes?`<div style="font-size:.58rem;color:#9d174d;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${b.notes.slice(0,100)}</div>`:''}
    </div>`;
  }).join('');
}

// ── Papers ────────────────────────────────────────────────────────────────────
function updatePapers(papers){
  $('papers-count').textContent=papers.length+'';
  if(!papers.length){$('papers-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Papers appear after each run</span>';return;}
  const vColors={BREAKTHROUGH:'#f472b6',DIRECTIONAL:'#34d399',NULL:'#64748b',ADVERSE:'#f87171'};
  $('papers-panel').innerHTML=papers.map(p=>{
    const vCol=vColors[p.verdict]||'#64748b';
    const vPill=p.verdict?`<span class="pill" style="background:#0f1929;color:${vCol};border:1px solid ${vCol}44;font-size:.5rem">${p.verdict}</span> `:'';
    const effectiveStatus=(p.plan_status&&p.plan_status!=='published')?p.plan_status:p.status;
    const sPill=!p.verdict&&effectiveStatus?`${pill(effectiveStatus,p.plan_status||effectiveStatus)} `:'';
    const pdfLink=p.has_pdf&&p.serve_pdf?`<a href="/serve/paper/${p.serve_pdf}" target="_blank" style="color:var(--sky);font-size:.6rem">PDF</a>`:'<span style="color:var(--text3);font-size:.6rem">no PDF</span>';
    const texLink=p.serve_tex?`<a href="/serve/paper/${p.serve_tex}" target="_blank" style="color:var(--text2);font-size:.6rem">LaTeX</a>`:'';
    const dFmt=p.mean_delta!=null?(Number(p.mean_delta)>=0?'+':'')+Number(p.mean_delta).toFixed(4):'';
    const dCol=p.mean_delta!=null&&Number(p.mean_delta)<0?'#34d399':'#f87171';
    return`<div style="padding:5px 0;border-bottom:1px solid var(--border2)">
      <div style="display:flex;align-items:center;gap:3px;overflow:hidden">
        ${vPill}${sPill}<span style="color:var(--text);font-size:.67rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.title}">${p.title}</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:2px">
        ${pdfLink} ${texLink}
        ${dFmt?`<span style="color:${dCol};font-size:.6rem">Δ${dFmt}</span>`:''}
        <span style="color:var(--text3);font-size:.55rem;margin-left:auto">${p.generated_at||''}</span>
      </div>
      ${p.recommendation?`<div style="color:var(--text2);font-size:.58rem;margin-top:2px;line-height:1.45">${p.recommendation}</div>`:''}
    </div>`;
  }).join('');
}

// ── Poll loop ─────────────────────────────────────────────────────────────────
function updateAuthor(data){
  const cp=data.current_paper||{};
  const log=(data.activity_log||[]);
  if(!cp.project_id){$('author-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Idle — waiting for experiments to complete.</span>';return;}
  const sectionIcon={writing:'✍',done:'✓',starting:'▶',complete:'✓',waiting:'⏳',revision_requested:'↺',revising:'✍',revision_failed:'⚠'};
  const ic=sectionIcon[cp.status]||'·';
  const progressBlock=progressHtml(cp.progress||{}, 'var(--green)');
  const waitHtml=cp.waiting_for_experiments?.length?
    `<div style="color:var(--amber);font-size:.6rem;margin-top:3px">⏳ waiting for: ${cp.waiting_for_experiments.join(', ')}</div>`:'';
  const logHtml=log.map(l=>`<div style="color:var(--text3);font-size:.58rem">${l.timestamp?.slice(11,19)||''} ${l.action} ${l.section||''}</div>`).join('');
  const suggestionHtml=(data.suggested_frontier_papers||[]).map(p=>
    `<div class="mini-note" style="margin-top:3px"><span style="color:var(--teal)">${p.title||p.paper_id}</span> · ${p.recommendation||p.readiness||''}</div>`
  ).join('');
  const evidenceHtml=(data.evidence_directives||[]).slice(0,3).map(task=>
    `<div class="mini-note" style="margin-top:3px"><span style="color:var(--amber)">${task.frontier_title||task.task_id}</span> · ${task.status} · ${task.next_action||''}</div>`
  ).join('');
  $('author-panel').innerHTML=`
    <div style="color:var(--text);font-size:.68rem;font-weight:600;margin-bottom:4px">${ic} ${cp.title||'—'}</div>
    <div style="color:var(--text2);font-size:.62rem">Section: <span style="color:var(--teal)">${cp.section_in_progress||'—'}</span></div>
    <div style="color:var(--text2);font-size:.6rem;margin-top:2px">Done: ${(cp.sections_complete||[]).join(', ')||'—'}</div>
    ${progressBlock}
    ${cp.director_recommendation?`<div class="mini-note" style="margin-top:3px">${cp.director_recommendation}</div>`:''}
    ${cp.compile_status?`<div class="mini-note" style="margin-top:3px">compile: ${cp.compile_status}</div>`:''}
    ${waitHtml}
    ${suggestionHtml?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Next frontier papers</div>${suggestionHtml}</div>`:''}
    ${evidenceHtml?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Evidence tasks</div>${evidenceHtml}</div>`:''}
    <div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px">${logHtml}</div>
  `;
}

function updateAuthor(data){
  const cp=data.current_paper||{};
  const queueRaw=Array.isArray(data.paper_queue)?data.paper_queue:[];
  const queue=[...queueRaw].sort((a,b)=>(Number(a.priority??999)-Number(b.priority??999)));
  const log=(data.activity_log||[]);
  const panel=$('author-panel');
  if(!panel)return;

  const paperProgress=p=>p?.paper_progress||p?.progress||{};
  const waits=p=>Array.isArray(p?.waiting_for_experiments)?p.waiting_for_experiments.filter(Boolean):[];
  const isPublished=p=>{
    if(!p)return false;
    const prog=paperProgress(p);
    const compileState=String(p.compile_status||prog.compile_status||'').toLowerCase();
    const stage=String(prog.stage||p.status||'').toLowerCase();
    if(waits(p).length)return false;
    if(['blocked','hold','waiting','revision_requested','revising','revision_failed'].includes(String(p.plan_status||p.status||'').toLowerCase()))return false;
    if(compileState==='draft_compiled')return false;
    return !!(compileState==='published' || stage==='published' || stage==='complete');
  };
  const isBlocked=p=>{
    if(!p)return false;
    if(waits(p).length)return true;
    const status=String(p.status||'').toLowerCase();
    const readiness=String(p.readiness||'').toLowerCase();
    return ['blocked','hold','waiting'].includes(status) || ['blocked','hold','experiment_first'].includes(readiness);
  };
  const isActionable=p=>{
    if(!p || isPublished(p) || waits(p).length)return false;
    const status=String(p.status||'').toLowerCase();
    const readiness=String(p.readiness||'').toLowerCase();
    return ['writing','revising','starting','ready','planning'].includes(status) || ['write_now','outline_now','ready'].includes(readiness);
  };
  const progressLabel=p=>{
    const prog=paperProgress(p);
    const total=Number(prog.total ?? p.total_experiments ?? p.experiment_ids?.length ?? 0);
    const complete=Number(prog.complete ?? p.complete_count ?? 0);
    const running=Number(prog.running ?? p.running_count ?? 0);
    const pending=Number(prog.pending ?? p.pending_count ?? 0);
    if(total>0)return `${complete}/${total} experiments complete`;
    if(running>0 || pending>0)return `${running} running, ${pending} pending`;
    return '';
  };
  const renderWaits=p=>{
    const items=waits(p);
    if(!items.length)return '';
    return `<div style="color:var(--amber);font-size:.6rem;margin-top:3px">waiting for: ${items.join(', ')}</div>`;
  };

  const unpublished=queue.filter(p=>!isPublished(p));
  const actionableTarget=unpublished.find(isActionable)||null;
  const currentTarget=actionableTarget||unpublished[0]||(!isPublished(cp)&&cp.project_id?cp:null);
  const blockedPapers=unpublished.filter(p=>p.project_id!==currentTarget?.project_id && isBlocked(p));
  const lastPublished=(isPublished(cp)&&cp.project_id)?cp:(queue.find(isPublished)||null);

  if(!currentTarget && !lastPublished && !queue.length){
    panel.innerHTML='<span style="color:var(--text3);font-size:.62rem">Idle - waiting for experiments or a new paper directive.</span>';
    return;
  }

  const statusLabel=String(currentTarget?.status||currentTarget?.readiness||'planned').toLowerCase();
  const pillStatusMap={writing:'running',revising:'running',starting:'running',ready:'running',complete:'complete',published:'complete',draft_compiled:'queued',blocked:'blocked',hold:'queued',planning:'planned'};
  const targetPill=currentTarget?pill(pillStatusMap[statusLabel]||statusLabel, statusLabel):'';
  const targetProgress=currentTarget?progressHtml(paperProgress(currentTarget), 'var(--green)'):'';
  const targetMeta=currentTarget?progressLabel(currentTarget):'';
  const logHtml=log.slice(-4).map(l=>`<div style="color:var(--text3);font-size:.58rem">${l.timestamp?.slice(11,19)||''} ${l.action} ${l.section||''}</div>`).join('');
  const suggestionHtml=(data.suggested_frontier_papers||[]).map(p=>
    `<div class="mini-note" style="margin-top:3px"><span style="color:var(--teal)">${p.title||p.paper_id}</span> - ${p.recommendation||p.readiness||''}</div>`
  ).join('');
  const evidenceHtml=(data.evidence_directives||[]).slice(0,3).map(task=>
    `<div class="mini-note" style="margin-top:3px"><span style="color:var(--amber)">${task.frontier_title||task.task_id}</span> - ${task.status} - ${task.next_action||''}</div>`
  ).join('');
  const blockedHtml=blockedPapers.slice(0,3).map(p=>`
    <div class="mini-note" style="margin-top:4px">
      <span style="color:var(--text)">${p.title||p.project_id}</span>
      <span style="color:var(--text3)"> - ${progressLabel(p)||'awaiting experiments'}</span>
      ${renderWaits(p)}
    </div>
  `).join('');

  panel.innerHTML=`
    ${currentTarget?`
      <div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Current writing target</div>
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
        ${targetPill}
        <div style="color:var(--text);font-size:.68rem;font-weight:600;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${currentTarget.title||currentTarget.project_id||'-'}</div>
      </div>
      <div style="color:var(--text2);font-size:.6rem;margin-top:3px">Section: <span style="color:var(--teal)">${cp.section_in_progress||currentTarget.section_in_progress||'planning'}</span></div>
      ${targetMeta?`<div style="color:var(--text2);font-size:.6rem;margin-top:2px">${targetMeta}</div>`:''}
      ${targetProgress}
      ${currentTarget.director_recommendation?`<div class="mini-note" style="margin-top:3px">${currentTarget.director_recommendation}</div>`:''}
      ${currentTarget.compile_status?`<div class="mini-note" style="margin-top:3px">compile: ${currentTarget.compile_status}</div>`:''}
      ${renderWaits(currentTarget)}
    `:`
      <div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Current writing target</div>
      <div style="color:var(--text3);font-size:.62rem">No paper is ready for drafting right now. The next meaningful work is blocked on experiments.</div>
    `}
    ${lastPublished && (!currentTarget || lastPublished.project_id!==currentTarget.project_id)?`
      <div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px">
        <div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Last validated / published</div>
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
          ${pill('complete','published')}
          <div style="color:var(--sky);font-size:.64rem">${lastPublished.title||lastPublished.project_id}</div>
        </div>
      </div>
    `:''}
    ${blockedHtml?`
      <div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px">
        <div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Blocked next papers</div>
        ${blockedHtml}
      </div>
    `:''}
    ${suggestionHtml?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Next frontier papers</div>${suggestionHtml}</div>`:''}
    ${evidenceHtml?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:2px">Evidence tasks</div>${evidenceHtml}</div>`:''}
    ${logHtml?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px">${logHtml}</div>`:''}
  `;
}

function updateLearnedLiterature(data){
  const panel=$('litmem-panel');
  const countEl=$('litmem-count');
  if(!panel||!countEl)return;
  _learnedLiteraturePayload=data||{};
  const learned=(data&&data.learned_knowledge)||{};
  const summary=learned.summary||{};
  const domains=Array.isArray(learned.domains)?learned.domains:[];
  const mastered=Array.isArray(learned.mastered_domains)?learned.mastered_domains:[];
  countEl.textContent=`${domains.length} domains`;
  if(!domains.length){
    panel.innerHTML='<div style="padding:8px;color:var(--text3);font-size:.62rem">TAR has not written learned-literature memory yet.</div>';
    return;
  }
  const timestamp=learned.timestamp||data?.timestamp||'';
  const topSummary=`<div class="learned-topline">
    <div class="mini-note">domains ${summary.domain_count||domains.length} · claims ${summary.claim_count||0} · connected topics ${summary.connected_topic_count||0} · sources ${summary.source_count||0} · mastered ${summary.mastered_count||0}</div>
    <div class="mini-note" style="margin-top:2px">Treat this as harvested working memory, not mastered knowledge.</div>
    ${timestamp?`<div class="mini-note" style="margin-top:2px">last learned refresh ${timestamp}</div>`:''}
  </div>`;
  const masteredSummary=mastered.length
    ? `<div class="learned-topline" style="margin-top:8px">
        <div class="learned-subtitle">Mastered knowledge</div>
        ${mastered.map(item=>`<div class="learned-claim"><span style="color:var(--green)">${item.label||item.domain_id}</span> · learning ${Number(item.learning_score||0).toFixed(1)} · confidence ${Number(item.learning_confidence_score||0).toFixed(1)}<br><span style="color:var(--text2)">${item.mastery_reason||''}</span></div>`).join('')}
      </div>`
    : `<div class="learned-topline" style="margin-top:8px"><div class="learned-subtitle">Mastered knowledge</div><div class="mini-note">No domains are mastered yet. TAR has working memory in some areas, but it is not claiming mastery.</div></div>`;
  const rows=domains.map((domain, idx)=>{
    const id=String(domain.domain_id||domain.label||`domain-${idx}`);
    const expanded=_expandedLearnedDomains.has(id);
    const learning=Number(domain.learning_score||0).toFixed(1);
    const confidence=Number(domain.learning_confidence_score||domain.learning_score||0).toFixed(1);
    const diversity=Number(domain.source_diversity_score||0).toFixed(1);
    const confidenceBand=String(domain.learning_confidence_band||'provisional');
    const maturityLabel=String(domain.learning_maturity_label||'harvested memory');
    const masteryStatus=String(domain.mastery_status||'emerging');
    const masteryLabel=masteryStatus==='mastered'?'mastered':masteryStatus==='working_memory'?'working memory':'emerging';
    const sourceMix=(domain.source_mix&&typeof domain.source_mix==='object')?domain.source_mix:{};
    const sourceItems=Object.entries(sourceMix)
      .sort((a,b)=>Number(b[1]||0)-Number(a[1]||0))
      .map(([name,count])=>`<span class="learned-source">${name} ${count}</span>`)
      .join(' · ');
    const claims=(domain.claim_fragments||[]).map(item=>`<div class="learned-claim">• ${item}</div>`).join('');
    const topics=(domain.connected_topics||[]).map(item=>`<span class="learned-chip">${item}</span>`).join('');
    const terms=(domain.trusted_terms||[]).map(item=>`<span class="learned-chip">${item}</span>`).join('');
    const titles=(domain.top_verified_titles||[]).map(item=>`<div class="learned-claim">• ${item}</div>`).join('');
    const uncertainty=(domain.uncertainty_flags||[]).map(item=>`<span class="learned-chip" style="border-color:#f59e0b55;color:#fbbf24">${String(item).replace(/_/g,' ')}</span>`).join('');
    return `<div class="learned-row ${expanded?'expanded':''}" onclick="toggleLearnedDomain('${id.replace(/'/g,'\\\'')}')">
      <div class="learned-head">
        ${pill(masteryStatus==='mastered'?'complete':masteryStatus==='working_memory'?'running':'planned', masteryLabel)}
        <span class="learned-title" title="${domain.label||id}">${domain.label||id}</span>
        <span class="learned-meta">learning ${learning} · confidence ${confidence} · diversity ${diversity}</span>
      </div>
      <div class="mini-note" style="margin-top:4px">${domain.learned_summary||'No learned summary yet.'}</div>
      ${expanded?`<div class="learned-body">
        <div class="learned-summary">Confidence band: <span style="color:var(--teal)">${confidenceBand}</span> · ${maturityLabel}</div>
        ${domain.mastery_reason?`<div class="learned-summary">Mastery status: <span style="color:${masteryStatus==='mastered'?'var(--green)':'var(--amber)'}">${masteryLabel}</span> · ${domain.mastery_reason}</div>`:''}
        <div class="learned-summary">${domain.learned_summary||''}</div>
        ${claims?`<div class="learned-subtitle">What TAR thinks it learned</div>${claims}`:''}
        ${uncertainty?`<div class="learned-subtitle">Uncertainty flags</div><div>${uncertainty}</div>`:''}
        ${topics?`<div class="learned-subtitle">Connected topics</div><div>${topics}</div>`:''}
        ${terms?`<div class="learned-subtitle">Trusted terms</div><div>${terms}</div>`:''}
        ${titles?`<div class="learned-subtitle">Top verified papers</div>${titles}`:''}
        ${sourceItems?`<div class="learned-subtitle">Source mix</div><div>${sourceItems}</div>`:''}
      </div>`:''}
    </div>`;
  }).join('');
  panel.innerHTML=topSummary+masteredSummary+rows;
}

function _deprecatedUpdateDirectorStage3(data){
  const directives=data.frontier_directives||[];
  const experimentDirectives=data.experiment_directives||[];
  const domains=data.knowledge_domains||[];
  const evidenceTasks=data.evidence_directives||[];
  const paths=data.active_research_paths||[];
  const summary=data.summary||{};
  const expanding=domains.filter(d=>d.expansion_status==='active_expansion'||d.expansion_status==='stabilizing');
  const experimentAgenda=experimentDirectives.slice(0,8).map(rec=>{
    const blocked=(rec.blocked_by_experiment_ids||[]).length?` Â· blocked ${(rec.blocked_by_experiment_ids||[]).length}`:'';
    const dataset=rec.dataset?rec.dataset.replace('split_',''):'experiment';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.scheduler_intent||rec.status, rec.scheduler_intent||rec.status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'â€”'} Â· ${dataset} Â· score ${rec.priority_score||'â€”'}${blocked}</div>
      <div class="director-action">${rec.experiment_goal||rec.mechanism_focus||rec.why_now||''}</div>
    </div>`;
  }).join('');
  const litSummary=(summary.literature_total_papers||summary.external_verified_sources||summary.last_literature_sync)?
    `<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px">
      <div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Literature harvest</div>
      <div class="mini-note">papers ${summary.literature_total_papers||0} · verified sources ${summary.external_verified_sources||0}</div>
      ${summary.last_literature_sync?`<div class="mini-note">last sync ${summary.last_literature_sync}</div>`:''}
    </div>`:'';
  $('director-count').textContent=`${directives.length}F · ${experimentDirectives.length}E · ${domains.length}D`;
  if(!directives.length){
    $('director-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Director is building the frontier map…</span>';
    return;
  }
  const top=directives.map(rec=>{
    const wait=(rec.waiting_on_experiment_ids||[]).length?` · wait ${(rec.waiting_on_experiment_ids||[]).length}`:'';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.truth_status, rec.truth_status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'—'} · score ${rec.priority_score||'—'}${wait}</div>
      <div class="director-action">${rec.next_action||''}</div>
    </div>`;
  }).join('');
  const evidence=evidenceTasks.slice(0,6).map(task=>
    `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(task.status, task.status)}
        <span class="director-title">${task.frontier_title||task.task_id}</span>
      </div>
      <div class="director-meta">${task.task_type||'evidence'} · score ${task.priority_score||'—'}</div>
      <div class="director-action">${task.next_action||task.verification_standard||''}</div>
    </div>`
  ).join('');
  const domainExpansion=expanding.slice(0,6).map(d=>
    `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(d.expansion_status||d.status, d.expansion_status||d.status)}
        <span class="director-title">${d.label}</span>
      </div>
      <div class="director-meta">truth ${d.truth_of_knowledge_score||0} · proficiency ${d.domain_proficiency_score||0} · expand ${d.active_expansion_score||0}</div>
      <div class="director-action">${d.expansion_goal||d.next_action||''}</div>
    </div>`
  ).join('');
  const weak=domains.filter(d=>d.status==='candidate'||d.status==='seed').slice(0,6).map(d=>
    `<div class="mini-note"><span style="color:var(--amber)">${d.label}</span> · truth ${d.truth_of_knowledge_score||0} · proficiency ${d.domain_proficiency_score||0} · ${d.next_action}</div>`
  ).join('');
  const activePaths=paths.slice(0,6).map(path=>
    `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(path.status||'planned', path.status||'planned')}
        <span class="director-title">${path.title}</span>
      </div>
      <div class="director-meta">${path.path_kind||'path'} Â· ${path.domain_label||path.domain_id||'domain'} Â· score ${path.priority_score||'â€”'}</div>
      <div class="director-action">${path.problem_statement||path.why_this_now||''}</div>
    </div>`
  ).join('');
  $('director-panel').innerHTML=
    litSummary +
    (activePaths?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Chosen research paths</div>${activePaths}</div>`:'') +
    (experimentAgenda?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Active experiment agenda</div>${experimentAgenda}</div>`:'') +
    top +
    (evidence?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Verified evidence tasks</div>${evidence}</div>`:'') +
    (domainExpansion?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Active domain expansion</div>${domainExpansion}</div>`:'') +
    (weak?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Weak / expanding domains</div>${weak}</div>`:'');
}

function returnPaperToAuthor(projectId){
  const reason=window.prompt('What should TAR Author fix in this paper? Leave blank to just reopen it for revision.','');
  if(reason===null)return;
  fetch(`/api/paper/return-to-author/${encodeURIComponent(projectId)}`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({reason})
  })
  .then(r=>r.json().then(data=>({ok:r.ok,data})))
  .then(({ok,data})=>{
    if(!ok){
      window.alert(data?.error||'Failed to return paper to TAR Author.');
      return;
    }
    refresh();
  })
  .catch(()=>window.alert('Failed to return paper to TAR Author.'));
}

function updatePapers(papers){
  $('papers-count').textContent=papers.length+'';
  if(!papers.length){$('papers-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Papers appear after each run</span>';return;}
  const vColors={BREAKTHROUGH:'#f472b6',DIRECTIONAL:'#34d399',NULL:'#64748b',ADVERSE:'#f87171'};
  $('papers-panel').innerHTML=papers.map(p=>{
    const vCol=vColors[p.verdict]||'#64748b';
    const vPill=p.verdict?`<span class="pill" style="background:#0f1929;color:${vCol};border:1px solid ${vCol}44;font-size:.5rem">${p.verdict}</span> `:'';
    const effectiveStatus=(p.plan_status&&p.plan_status!=='published')?p.plan_status:p.status;
    const sPill=!p.verdict&&effectiveStatus?`${pill(effectiveStatus,p.plan_status||effectiveStatus)} `:'';
    const pdfLink=p.has_pdf&&p.serve_pdf?`<a href="/serve/paper/${p.serve_pdf}" target="_blank" style="color:var(--sky);font-size:.6rem">PDF</a>`:'<span style="color:var(--text3);font-size:.6rem">no PDF</span>';
    const texLink=p.serve_tex?`<a href="/serve/paper/${p.serve_tex}" target="_blank" style="color:var(--text2);font-size:.6rem">LaTeX</a>`:'';
    const dFmt=p.mean_delta!=null?(Number(p.mean_delta)>=0?'+':'')+Number(p.mean_delta).toFixed(4):'';
    const dCol=p.mean_delta!=null&&Number(p.mean_delta)<0?'#34d399':'#f87171';
    const progressBlock=progressHtml({pct:p.progress_pct,label:p.progress_label,compile_status:p.compile_status}, p.has_pdf?'var(--green)':'var(--sky)');
    const canReturn=Boolean(p.project_id)&&(p.compile_status==='draft_compiled'||p.status==='draft_compiled'||p.status==='complete'||p.compile_status==='published');
    const returnBtn=canReturn?`<button class="action-btn" onclick="event.stopPropagation();returnPaperToAuthor('${p.project_id}')">Return to Author</button>`:'';
    const waits=Array.isArray(p.waiting_for_experiments)?p.waiting_for_experiments:[];
    const expCounts=(p.total_experiments||p.complete_count||p.running_count||p.pending_count)?
      `<span style="color:var(--text3);font-size:.55rem">${p.complete_count||0}/${p.total_experiments||0} exp complete</span>`:'';
    return`<div style="padding:6px 0;border-bottom:1px solid var(--border2)">
      <div style="display:flex;align-items:center;gap:3px;overflow:hidden">
        ${vPill}${sPill}<span style="color:var(--text);font-size:.67rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.title}">${p.title}</span>
      </div>
      ${progressBlock}
      <div style="display:flex;align-items:center;gap:8px;margin-top:4px">
        ${pdfLink} ${texLink}
        ${dFmt?`<span style="color:${dCol};font-size:.6rem">Δ${dFmt}</span>`:''}
        ${expCounts}
        ${p.compile_status?`<span style="color:var(--text3);font-size:.55rem">${p.compile_status}</span>`:''}
        ${returnBtn}
        <span style="color:var(--text3);font-size:.55rem;margin-left:auto">${p.generated_at||''}</span>
      </div>
      ${waits.length?`<div style="color:var(--amber);font-size:.58rem;margin-top:3px;line-height:1.45">Blocked on: ${waits.join(', ')}</div>`:''}
      ${p.revision_reason?`<div style="color:var(--amber);font-size:.58rem;margin-top:3px;line-height:1.45">Revision request: ${p.revision_reason}</div>`:''}
      ${p.recommendation?`<div style="color:var(--text2);font-size:.58rem;margin-top:2px;line-height:1.45">${p.recommendation}</div>`:''}
    </div>`;
  }).join('');
}

function updateCoordination(data){
  _coordinationPayload=data||{};
  if(_schedulerPayload&&Object.keys(_schedulerPayload).length)updateScheduler(_schedulerPayload);
}

function updateScheduler(data){
  _schedulerPayload=data||{};
  const coordination=_coordinationPayload||{};
  const canStart=(data.experiments_to_start||data.can_start||[]);
  const runningIds=(data.running_ids||[]);
  const holdRows=(data.hold_reasons||[]).slice(0,8);
  const running=runningIds.map(id=>`<div class="learned-claim">- running: ${id}</div>`).join('');
  const next=data.next_experiment_id?`<div class="mini-note" style="margin-top:4px">Next clear start: <span style="color:var(--teal)">${data.next_experiment_id}</span></div>`:'';
  const starts=canStart.length?`<div class="mini-note" style="margin-top:4px">Can start now: ${canStart.join(', ')}</div>`:'';
  const holds=holdRows.map(rec=>`<div class="learned-claim">- ${rec.experiment_id||rec.experiment_name}: ${rec.reason||''}</div>`).join('');
  const papers=(coordination.allowed_paper_ids||[]).length?`<div class="mini-note" style="margin-top:4px">Active papers: ${(coordination.allowed_paper_ids||[]).join(', ')}</div>`:'';
  const agenda=((_directorPayload.experiment_directives||[]).filter(rec=>{
    const status=String(rec.status||rec.scheduler_intent||'').toLowerCase();
    return !['complete','done','failed','archived','archive'].includes(status);
  }).slice(0,6)).map(rec=>{
    const expId=String(rec.experiment_id||rec.id||rec.title||'');
    const dataset=rec.dataset?String(rec.dataset).replace('split_',''):'experiment';
    const hold=holdRows.find(item=>String(item.experiment_id||item.experiment_name||'')===expId);
    const isRunning=runningIds.includes(expId);
    const isNext=String(data.next_experiment_id||'')===expId;
    let trace='queued behind higher-ranked work';
    let traceState=rec.scheduler_intent||rec.status||'queued';
    if(isRunning){
      trace='active on the current hardware slot';
      traceState='running';
    }else if(hold){
      trace=hold.reason||trace;
      traceState='hold';
    }else if(isNext){
      trace='next ranked experiment once the current slot frees';
      traceState='next';
    }else if(canStart.includes(expId)){
      trace='runnable now under current hardware limits';
      traceState='ready';
    }
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(traceState, traceState)}
        <span class="director-title">${rec.title||expId}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'-'} · ${dataset} · priority ${rec.priority_score||'-'}</div>
      <div class="director-action">${trace}</div>
    </div>`;
  }).join('');
  $('scheduler-panel').innerHTML=`
    <div>${data.rationale||'No scheduler data yet.'}</div>
    ${next}
    ${starts}
    ${papers}
    ${agenda?`<div class="learned-subtitle">Decision trace</div>${agenda}`:''}
    ${running?`<div class="learned-subtitle">Running</div>${running}`:''}
    ${holds?`<div class="learned-subtitle">Hold reasons</div>${holds}`:''}
  `;
}

function toggleResultRow(rowId){
  if(_expandedResultRows.has(rowId))_expandedResultRows.delete(rowId);
  else _expandedResultRows.add(rowId);
  updateResults(window.__lastResultsPayload||{});
}

function updateResults(payload){
  window.__lastResultsPayload=payload||{};
  const rows=payload.results||[];
  $('results-count').textContent=`${payload.running||0} live · ${payload.breakthroughs||0} BK · ${payload.total||0} total`;
  if(!rows.length){
    $('phases-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">No experiment results yet</span>';
    return;
  }
  $('phases-panel').innerHTML=rows.map(rec=>{
    const rowId=String(rec.id||rec.label||rec.title||'result');
    const expanded=_expandedResultRows.has(rowId);
    const verdict=String(rec.verdict||rec.status||'').toUpperCase();
    const verdictLabel=verdict||'-';
    const dataset=String(rec.dataset||'').replace('split_','');
    const program=String(rec.program||rec.source_kind||'experiment');
    const metaBits=[program,dataset].filter(Boolean).join(' · ');
    const metrics=[
      rec.tcl_forgetting!=null?`forg ${fmt(rec.tcl_forgetting)}`:'',
      rec.mean_delta!=null?`delta ${delta(rec.mean_delta)}`:'',
      rec.vs_ewc_p!=null?`p ${fmtP(rec.vs_ewc_p)}`:'',
      rec.vs_ewc_d!=null?`d ${fmt(rec.vs_ewc_d,2)}`:''
    ].filter(Boolean).join(' · ');
    const path=rec.result_path?`<div class="result-note" style="color:var(--text3)">path: ${rec.result_path}</div>`:'';
    const experimentButton=rec.id?`<button class="action-btn" onclick="event.stopPropagation();showModal('${String(rec.id).replace(/'/g,"\\\\'")}')">Experiment</button>`:'';
    return `<div class="result-row ${expanded?'expanded':''}" onclick="toggleResultRow('${rowId.replace(/'/g,"\\\\'")}')">
      <div style="display:flex;align-items:center;gap:6px">
        ${pill(rec.status||verdictLabel, rec.status||verdictLabel)}
        <span class="result-title">${rec.title||rec.label||rec.id||'Result'}</span>
        ${verdictLabel?`<span class="${verdictLabel.includes('BREAKTHROUGH')?'bk-flash':''}" style="color:${verdictLabel.includes('BREAKTHROUGH')?'var(--pink)':'var(--text2)'};font-size:.58rem;margin-left:auto">${verdictLabel}</span>`:''}
      </div>
      <div class="result-meta">${metaBits||'experiment result'}${rec.completed_at?` · ${rec.completed_at}`:''}</div>
      ${metrics?`<div class="result-metrics">${metrics}</div>`:''}
      ${expanded?`
        ${rec.notes?`<div class="result-note">${rec.notes}</div>`:''}
        ${path}
        ${experimentButton?`<div style="display:flex;gap:6px;align-items:center;margin-top:6px">${experimentButton}</div>`:''}
      `:''}
    </div>`;
  }).join('');
}

function updateDirector(data){
  _directorPayload=data||{};
  if(_statusPayload&&Object.keys(_statusPayload).length)updateStatus(_statusPayload);
  if(_schedulerPayload&&Object.keys(_schedulerPayload).length)updateScheduler(_schedulerPayload);
  const directives=data.frontier_directives||[];
  const experimentDirectives=data.experiment_directives||[];
  const domains=data.knowledge_domains||[];
  const evidenceTasks=data.evidence_directives||[];
  const paths=data.active_research_paths||[];
  const summary=data.summary||{};
  const expanding=domains.filter(d=>d.expansion_status==='active_expansion'||d.expansion_status==='stabilizing');
  $('director-count').textContent=`${directives.length}F · ${experimentDirectives.length}E · ${domains.length}D`;
  if(!directives.length){
    $('director-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Director is building the frontier map...</span>';
    return;
  }
  const litSummary=(summary.literature_total_papers||summary.external_verified_sources||summary.last_literature_sync)?
    `<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px">
      <div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Literature harvest</div>
      <div class="mini-note">papers ${summary.literature_total_papers||0} · verified sources ${summary.external_verified_sources||0}</div>
      ${summary.last_literature_sync?`<div class="mini-note">last sync ${summary.last_literature_sync}</div>`:''}
    </div>`:'';
  const activePaths=paths.slice(0,8).map(path=>
    `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(path.status||'planned', path.status||'planned')}
        <span class="director-title">${path.title}</span>
      </div>
      <div class="director-meta">${path.path_kind||'path'} · ${path.domain_label||path.domain_id||'domain'} · score ${path.priority_score||'-'}</div>
      <div class="director-action">${path.problem_statement||path.why_this_now||''}</div>
    </div>`
  ).join('');
  const experimentAgenda=experimentDirectives.slice(0,10).map(rec=>{
    const blocked=(rec.blocked_by_experiment_ids||[]).length?` · blocked ${(rec.blocked_by_experiment_ids||[]).length}`:'';
    const dataset=rec.dataset?String(rec.dataset).replace('split_',''):'experiment';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.scheduler_intent||rec.status, rec.scheduler_intent||rec.status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'-'} · ${dataset} · score ${rec.priority_score||'-'}${blocked}</div>
      <div class="director-action">${rec.experiment_goal||rec.mechanism_focus||rec.why_now||''}</div>
    </div>`;
  }).join('');
  const top=directives.map(rec=>{
    const wait=(rec.waiting_on_experiment_ids||[]).length?` · wait ${(rec.waiting_on_experiment_ids||[]).length}`:'';
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.truth_status, rec.truth_status)}
        <span class="director-title">${rec.title}</span>
      </div>
      <div class="director-meta">rank ${rec.scheduler_rank||'-'} · score ${rec.priority_score||'-'}${wait}</div>
      <div class="director-action">${rec.next_action||''}</div>
    </div>`;
  }).join('');
  const evidence=evidenceTasks.slice(0,6).map(task=>
    `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(task.status, task.status)}
        <span class="director-title">${task.frontier_title||task.task_id}</span>
      </div>
      <div class="director-meta">${task.task_type||'evidence'} · score ${task.priority_score||'-'}</div>
      <div class="director-action">${task.next_action||task.verification_standard||''}</div>
    </div>`
  ).join('');
  const domainExpansion=expanding.slice(0,6).map(d=>
    `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(d.expansion_status||d.status, d.expansion_status||d.status)}
        <span class="director-title">${d.label}</span>
      </div>
      <div class="director-meta">truth ${d.truth_of_knowledge_score||0} · proficiency ${d.domain_proficiency_score||0} · expand ${d.active_expansion_score||0}</div>
      <div class="director-action">${d.expansion_goal||d.next_action||''}</div>
    </div>`
  ).join('');
  $('director-panel').innerHTML=
    litSummary +
    (activePaths?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Chosen research paths</div>${activePaths}</div>`:'') +
    (experimentAgenda?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Active experiment agenda</div>${experimentAgenda}</div>`:'') +
    top +
    (evidence?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Verified evidence tasks</div>${evidence}</div>`:'') +
    (domainExpansion?`<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px"><div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Active domain expansion</div>${domainExpansion}</div>`:'');
}

async function reviewDecision(reviewId, decision, buildManifest=false){
  const needsNote=['request_revision','reject','pause_this_frontier'].includes(decision);
  const human_notes=needsNote?(window.prompt('Optional note for this review action:','')||''):'';
  const res=await fetch(`/api/human_review/decision/${encodeURIComponent(reviewId)}`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({decision,human_notes,build_manifest_authorised:buildManifest})
  });
  if(!res.ok){
    const data=await res.json().catch(()=>({}));
    window.alert(data.error||'Failed to save review decision.');
    return;
  }
  refresh();
}

async function answerHumanReviewQuestion(questionId, answer){
  const answer_notes=window.prompt('Optional note for this answer:','')||'';
  const res=await fetch(`/api/human_review/question/${encodeURIComponent(questionId)}/answer`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({answer,answer_notes})
  });
  if(!res.ok){
    const data=await res.json().catch(()=>({}));
    window.alert(data.error||'Failed to save answer.');
    return;
  }
  refresh();
}

function updateHumanReview(data){
  const proposals=data.proposals||[];
  const questions=data.questions||[];
  const claims=data.claim_reviews||[];
  $('review-count').textContent=`${proposals.length}P · ${questions.length}Q · ${claims.length}C`;
  if(!proposals.length && !questions.length && !claims.length){
    $('review-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">No items awaiting human review.</span>';
    return;
  }
  const proposalHtml=proposals.slice(0,8).map(item=>`
    <div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(item.status||'awaiting_human_review', item.status||'awaiting_human_review')}
        <span class="director-title">${item.title||item.experiment_id}</span>
      </div>
      <div class="director-meta">${item.frontier_problem_id||'frontier'} · ${item.dataset||'dataset'} · ${item.method||'method'}</div>
      <div class="director-action">${item.experiment_goal||item.why_now||''}</div>
      ${(item.manifest_path||'') ? `<div class="mini-note">Manifest prepared: ${item.manifest_path}</div>` : ``}
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">
        <button class="action-btn" onclick="reviewDecision('${String(item.review_id).replace(/'/g,"\\\\'")}','approve')">Approve</button>
        <button class="action-btn" onclick="reviewDecision('${String(item.review_id).replace(/'/g,"\\\\'")}','approve_and_build_manifest',true)">Approve + Manifest</button>
        <button class="action-btn" onclick="reviewDecision('${String(item.review_id).replace(/'/g,"\\\\'")}','request_revision')">Revision</button>
        <button class="action-btn" onclick="reviewDecision('${String(item.review_id).replace(/'/g,"\\\\'")}','reject')">Reject</button>
      </div>
    </div>`).join('');
  const claimHtml=claims.slice(0,6).map(item=>`
    <div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(item.status||'awaiting_human_review', item.status||'awaiting_human_review')}
        <span class="director-title">${item.title||item.paper_id}</span>
      </div>
      <div class="director-meta">${item.truth_status||'weak'} · ${item.readiness||'planned'}</div>
      <div class="director-action">${item.recommendation||''}</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">
        <button class="action-btn" onclick="reviewDecision('${String(item.review_id).replace(/'/g,"\\\\'")}','approve_claim_scope')">Approve Claim Scope</button>
        <button class="action-btn" onclick="reviewDecision('${String(item.review_id).replace(/'/g,"\\\\'")}','approve_paper_rewrite')">Approve Rewrite</button>
        <button class="action-btn" onclick="reviewDecision('${String(item.review_id).replace(/'/g,"\\\\'")}','hold_pending_more_evidence')">Hold</button>
      </div>
    </div>`).join('');
  const questionHtml=questions.slice(0,8).map(item=>`
    <div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(item.status||'awaiting_human_answer', item.question_type||'question')}
        <span class="director-title">${item.frontier_problem_id||item.component||'question'}</span>
      </div>
      <div class="director-action">${item.question_text||''}</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">
        ${(item.options||[]).slice(0,3).map(opt=>`<button class="action-btn" onclick="answerHumanReviewQuestion('${String(item.question_id).replace(/'/g,"\\\\'")}','${String(opt).replace(/'/g,"\\\\'")}')">${String(opt).replace(/_/g,' ')}</button>`).join('')}
      </div>
    </div>`).join('');
  $('review-panel').innerHTML=
    (proposalHtml?`<div style="color:var(--text2);font-size:.58rem;margin-bottom:3px">Experiment proposals</div>${proposalHtml}`:'')+
    (claimHtml?`<div style="margin-top:8px;border-top:1px solid var(--border2);padding-top:6px;color:var(--text2);font-size:.58rem;margin-bottom:3px">Claim and rewrite approvals</div>${claimHtml}`:'')+
    (questionHtml?`<div style="margin-top:8px;border-top:1px solid var(--border2);padding-top:6px;color:var(--text2);font-size:.58rem;margin-bottom:3px">Questions TAR needs answered</div>${questionHtml}`:'');
}

function updateValidationBoard(data){
  const summary=data.summary||{};
  const results=data.results||[];
  $('validation-count').textContent=`${summary.trusted_publication_allowed||0} trusted · ${summary.limited_scope||0} limited`;
  if(!results.length){
    $('validation-panel').innerHTML='<span style="color:var(--text3);font-size:.62rem">Validation state will appear after the first audit pass.</span>';
    return;
  }
  const rows=results.slice(0,10).map(rec=>{
    const trust=rec.trust||{};
    const issues=(rec.issues||[]).join('; ');
    return `<div class="director-row">
      <div style="display:flex;align-items:center;gap:5px">
        ${pill(rec.ok?'complete':'blocked', trust.trust_tier||'validation')}
        <span class="director-title">${trust.logical_name||rec.logical_name||'result'}</span>
      </div>
      <div class="director-meta">${trust.provenance_status||'unknown'} · publication ${trust.publication_allowed?'allowed':'blocked'}</div>
      <div class="director-action">${issues||trust.basis||''}</div>
    </div>`;
  }).join('');
  $('validation-panel').innerHTML=
    `<div class="mini-note">Trusted for publication: ${summary.trusted_publication_allowed||0}</div>`+
    `<div class="mini-note">Limited scope: ${summary.limited_scope||0} · Missing env: ${summary.missing_env||0} · Quarantined: ${summary.quarantined||0}</div>`+
    `<div style="margin-top:6px;border-top:1px solid var(--border2);padding-top:4px">${rows}</div>`;
}

function refresh(){
  fetch('/api/hardware').then(r=>r.json()).then(updateHardware).catch(()=>{});
  fetch('/api/status').then(r=>r.json()).then(updateStatus).catch(()=>{});
  fetch('/api/experiments').then(r=>r.json()).then(updateExperiments).catch(()=>{});
  fetch('/api/results').then(r=>r.json()).then(updateResults).catch(()=>{});
  fetch('/api/registry').then(r=>r.json()).then(updateRegistry).catch(()=>{});
  fetch('/api/autonomous').then(r=>r.json()).then(updateAR).catch(()=>{});
  fetch('/api/breakthroughs').then(r=>r.json()).then(updateBreakthroughs).catch(()=>{});
  fetch('/api/papers').then(r=>r.json()).then(updatePapers).catch(()=>{});
  fetch('/api/frontier').then(r=>r.json()).then(updateFrontier).catch(()=>{});
  fetch('/api/research_director').then(r=>r.json()).then(updateDirector).catch(()=>{});
  fetch('/api/human_review').then(r=>r.json()).then(updateHumanReview).catch(()=>{});
  fetch('/api/validation').then(r=>r.json()).then(updateValidationBoard).catch(()=>{});
  fetch('/api/literature').then(r=>r.json()).then(updateLearnedLiterature).catch(()=>{});
  fetch('/api/coordination').then(r=>r.json()).then(updateCoordination).catch(()=>{});
  fetch('/api/scheduler').then(r=>r.json()).then(updateScheduler).catch(()=>{});
  fetch('/api/author').then(r=>r.json()).then(updateAuthor).catch(()=>{});
  refreshLogList();
  loadLog(_selLog);
  // Refresh modal if open
  if(_currentExpId){
    fetch(`/api/experiment/${_currentExpId}`).then(r=>r.json()).then(renderModal).catch(()=>{});
    loadExperimentLog(_currentExpId,_currentExpLogSource);
  }
}

enhancePanels();
refresh();
setInterval(refresh,5000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    _seed_frontier()
    _start_hardware_monitor()
    _start_dashboard_heartbeat()
    print(f"\n{'='*60}")
    print(f"  TAR Research Dashboard")
    print(f"  Workspace : {_WS}")
    print(f"  Open      : http://localhost:{PORT}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
