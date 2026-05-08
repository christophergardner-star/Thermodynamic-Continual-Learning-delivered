"""
TAR Experiment Library
======================

Builds an author-friendly evidence index over the orchestrator queue and saved
experiment artifacts so TAR-Author can reliably discover what evidence exists
for each paper, frontier problem, and project.

Storage: {workspace}/tar_state/experiment_library.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jload(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _result_payload(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    if isinstance(raw.get("result"), dict):
        return raw["result"]
    return raw


def _result_summary(raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = _result_payload(raw)
    if not payload:
        return {}
    return {
        "verdict": payload.get("verdict", ""),
        "mean_forgetting": payload.get("mean_forgetting"),
        "std_forgetting": payload.get("std_forgetting"),
        "mean_accuracy": payload.get("mean_accuracy"),
        "std_accuracy": payload.get("std_accuracy"),
        "mean_delta": payload.get("mean_delta"),
        "p_val": payload.get("p_val"),
        "cohens_d": payload.get("cohens_d"),
        "n_better": payload.get("n_better"),
        "completed_at": payload.get("completed_at", ""),
        "notes": payload.get("notes", ""),
        "seed_results": payload.get("seed_results", []),
    }


def _load_experiment_records(workspace: Path) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    queue_path = workspace / "tar_state" / "experiment_queue.json"
    queue_data = _jload(queue_path)
    queue_experiments = queue_data.get("experiments", []) if isinstance(queue_data, dict) else []
    for exp in queue_experiments if isinstance(queue_experiments, list) else []:
        exp_id = str(exp.get("id", "") or "")
        if exp_id:
            records[exp_id] = dict(exp)

    archive_path = workspace / "tar_state" / "experiment_archive.json"
    archive_data = _jload(archive_path)
    archive_experiments = archive_data.get("experiments", []) if isinstance(archive_data, dict) else []
    for exp in archive_experiments if isinstance(archive_experiments, list) else []:
        exp_id = str(exp.get("id", "") or "")
        if exp_id and exp_id not in records:
            records[exp_id] = dict(exp)

    return list(records.values())


def build_experiment_library(workspace: Path) -> dict[str, Any]:
    experiments = _load_experiment_records(workspace)

    records: list[dict[str, Any]] = []
    by_paper: dict[str, dict[str, Any]] = {}

    for exp in experiments:
        exp_id = str(exp.get("id", "") or "")
        if not exp_id:
            continue
        result_path = str(exp.get("result_path", "") or "")
        result_data = _jload(Path(result_path)) if result_path and Path(result_path).exists() else None
        summary = _result_summary(result_data)
        archive_dir = workspace / "tar_state" / "experiments" / exp_id
        spec_path = archive_dir / "spec.json"
        paper_id = str(exp.get("author_paper_id") or exp.get("project_id") or "unassigned-paper")
        frontier_id = str(exp.get("frontier_problem_id", "") or "")

        record = {
            "experiment_id": exp_id,
            "name": exp.get("name", ""),
            "project_id": exp.get("project_id", ""),
            "paper_id": paper_id,
            "frontier_problem_id": frontier_id,
            "dataset": exp.get("dataset", ""),
            "method": exp.get("method", ""),
            "status": exp.get("status", ""),
            "stage": exp.get("stage", ""),
            "priority": exp.get("priority", 50),
            "submitted_at": exp.get("submitted_at", ""),
            "started_at": exp.get("started_at", ""),
            "completed_at": exp.get("completed_at", ""),
            "archived_at": exp.get("archived_at", ""),
            "archive_reason": exp.get("archive_reason", ""),
            "is_archived": bool(exp.get("archived_at", "")),
            "progress": exp.get("progress", {}),
            "context": exp.get("context", {}),
            "description": exp.get("description", ""),
            "runner_key": exp.get("runner_key", ""),
            "result_path": result_path,
            "spec_path": str(spec_path) if spec_path.exists() else "",
            "archive_dir": str(archive_dir),
            "result_summary": summary,
        }
        records.append(record)

        group = by_paper.setdefault(paper_id, {
            "paper_id": paper_id,
            "title": (exp.get("context") or {}).get("feeds_paper") or paper_id.replace("_", " ").replace("-", " ").title(),
            "frontier_problem_ids": [],
            "experiment_ids": [],
            "result_paths": [],
            "completed_count": 0,
            "running_count": 0,
            "pending_count": 0,
            "archived_count": 0,
            "experiments": [],
        })
        group["experiments"].append(record)
        group["experiment_ids"].append(exp_id)
        if frontier_id and frontier_id not in group["frontier_problem_ids"]:
            group["frontier_problem_ids"].append(frontier_id)
        if result_path and result_path not in group["result_paths"]:
            group["result_paths"].append(result_path)
        if record["is_archived"]:
            group["archived_count"] += 1
        status = str(exp.get("status", "") or "")
        stage = str(exp.get("stage", "") or "")
        if status == "complete" or stage == "complete":
            group["completed_count"] += 1
        elif status == "running" or stage == "running":
            group["running_count"] += 1
        else:
            group["pending_count"] += 1

    payload = {
        "timestamp": _now_iso(),
        "experiment_count": len(records),
        "archived_experiment_count": sum(1 for rec in records if rec.get("is_archived")),
        "active_experiment_count": sum(1 for rec in records if not rec.get("is_archived")),
        "paper_count": len(by_paper),
        "experiments": sorted(
            records,
            key=lambda rec: (
                rec.get("is_archived", False),
                rec.get("status") != "running",
                rec.get("status") != "complete",
                rec.get("priority", 50),
                rec.get("name", ""),
            ),
        ),
        "papers": sorted(
            by_paper.values(),
            key=lambda rec: (
                rec["pending_count"] == 0 and rec["running_count"] == 0 and rec["completed_count"] == 0,
                -rec["completed_count"],
                -rec["running_count"],
                rec["title"],
            ),
        ),
    }
    return payload


def save_experiment_library(workspace: Path) -> dict[str, Any]:
    payload = build_experiment_library(workspace)
    path = workspace / "tar_state" / "experiment_library.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_experiment_library(workspace: Path, refresh: bool = False) -> dict[str, Any]:
    path = workspace / "tar_state" / "experiment_library.json"
    if refresh or not path.exists():
        return save_experiment_library(workspace)
    data = _jload(path)
    if isinstance(data, dict):
        return data
    return save_experiment_library(workspace)


def get_paper_evidence(workspace: Path, paper_id: str, refresh: bool = False) -> dict[str, Any]:
    library = load_experiment_library(workspace, refresh=refresh)
    for paper in library.get("papers", []):
        if str(paper.get("paper_id", "") or "") == paper_id:
            return paper
    return {
        "paper_id": paper_id,
        "title": paper_id.replace("_", " ").replace("-", " ").title(),
        "frontier_problem_ids": [],
        "experiment_ids": [],
        "result_paths": [],
        "completed_count": 0,
        "running_count": 0,
        "pending_count": 0,
        "experiments": [],
    }
