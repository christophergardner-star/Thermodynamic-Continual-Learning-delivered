"""
Workspace-local experiment worker.

Used when the orchestrator prepares a non-default Python environment for a run.
The worker executes one experiment inside that interpreter while sharing the
same queue/result state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from tar_experiment_orchestrator import ExperimentOrchestrator
from tar_storage import ensure_workspace_layout, resolve_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single TAR experiment inside a prepared interpreter.")
    parser.add_argument("--workspace", default=str(resolve_workspace(_REPO)))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = ensure_workspace_layout(Path(args.workspace), repo_root=_REPO)
    orch = ExperimentOrchestrator(workspace)
    result = orch.execute_by_id(
        args.experiment_id,
        skip_preflight=args.skip_preflight,
        force_in_process=True,
    )
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
