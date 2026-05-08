from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from tar_author import write_planned_author_state
from tar_experiment_orchestrator import ExperimentOrchestrator
from tar_living_research import _ensure_validation_mode_queue, write_research_coordination_state
from tar_research_director import ResearchDirector
from tar_storage import ensure_workspace_layout, resolve_workspace
from tar_validation_mode import DEFAULT_MIN_SEEDS, DEFAULT_TARGET_SEEDS, activate_validation_mode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activate TAR stabilisation and HPC validation mode.")
    parser.add_argument("--workspace", default="", help="Workspace root. Defaults to resolved TAR workspace.")
    parser.add_argument("--watch", action="store_true", help="Continuously re-assert the validation queue.")
    parser.add_argument("--interval-s", type=float, default=30.0, help="Watch interval in seconds.")
    return parser.parse_args()


def _workspace_from_args(raw: str) -> Path:
    if raw.strip():
        return ensure_workspace_layout(Path(raw.strip()), repo_root=_REPO)
    return ensure_workspace_layout(resolve_workspace(_REPO), repo_root=_REPO)


def _activate_once(workspace: Path) -> dict:
    state = activate_validation_mode(
        workspace,
        _REPO,
        min_seeds=list(DEFAULT_MIN_SEEDS),
        target_seeds=list(DEFAULT_TARGET_SEEDS),
        allow_current_run_to_finish=True,
    )
    orch = ExperimentOrchestrator(workspace)
    orch.reconcile_runtime_state()
    submitted = _ensure_validation_mode_queue(workspace, orch)
    director_state = ResearchDirector(workspace).update_state()
    author_state = write_planned_author_state(workspace)
    coordination = write_research_coordination_state(workspace, orch, director_state, author_state)
    return {
        "mode_state": state,
        "submitted": submitted,
        "coordination": coordination,
        "running_ids": [spec.id for spec in orch.get_running()],
        "pending_ids": [spec.id for spec in orch.get_pending()],
    }


def main() -> None:
    args = _parse_args()
    workspace = _workspace_from_args(args.workspace)
    summary = _activate_once(workspace)
    print(json.dumps(summary, indent=2))
    if not args.watch:
        return
    while True:
        time.sleep(max(5.0, float(args.interval_s)))
        summary = _activate_once(workspace)
        print(json.dumps({
            "timestamp": summary["mode_state"].get("activated_at", ""),
            "running_ids": summary["running_ids"],
            "pending_ids": summary["pending_ids"],
        }, indent=2))


if __name__ == "__main__":
    main()
