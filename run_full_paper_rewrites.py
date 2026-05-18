from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tar_author import request_full_paper_rewrite, run_paper_revision, write_planned_author_state
from tar_project_registry import ProjectRegistry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    workspace = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered")
    from tar_validation_mode import _read_stabilisation_state_strict
    from tar_lab.errors import (
        StabilisationGateStateUnreadableError,
        StabilisationGateCategoricalBlockError,
    )
    stab = _read_stabilisation_state_strict(workspace)
    if bool(stab.get("active")):
        raise StabilisationGateCategoricalBlockError(
            "run_full_paper_rewrites.py is a Class-B mass-rewrite script. "
            "Categorically blocked during stabilisation. No override path exists."
        )
    audit_dir = workspace / "tar_state" / "stat_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    reason = (
        "Full rewrite required. Rebuild every paper from current validated evidence only; "
        "remove unsupported claims, do not mix domains, include figures and diagrams generated "
        "from validated results, and do not target a fixed page limit."
    )

    summary: dict[str, object] = {
        "started_at": _now_iso(),
        "workspace": str(workspace),
        "reason": reason,
        "requested": [],
        "targets": [],
        "results": [],
    }

    requested = request_full_paper_rewrite(workspace, reason)
    summary["requested"] = requested

    author_state = write_planned_author_state(workspace)
    queue_ids = [
        str(entry.get("project_id", "") or "")
        for entry in author_state.get("paper_queue", [])
        if isinstance(entry, dict) and str(entry.get("project_id", "") or "")
    ]
    registry_ids = [
        project.id
        for project in ProjectRegistry(workspace).list_all()
        if project.project_type == "paper"
    ]
    targets = []
    for project_id in queue_ids + registry_ids:
        if project_id and project_id not in targets:
            targets.append(project_id)
    summary["targets"] = targets

    for project_id in targets:
        started_at = _now_iso()
        try:
            result = run_paper_revision(
                workspace=workspace,
                project_id=project_id,
                reason=reason,
                request_first=False,
            )
            summary["results"].append({
                "project_id": project_id,
                "started_at": started_at,
                "finished_at": _now_iso(),
                "ok": result is not None,
                "result": result,
            })
        except Exception as exc:  # pragma: no cover - operational runner
            summary["results"].append({
                "project_id": project_id,
                "started_at": started_at,
                "finished_at": _now_iso(),
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    summary["finished_at"] = _now_iso()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = audit_dir / f"full_paper_rewrite_run__{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
