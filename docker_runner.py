"""
Docker experiment runner — executed inside the tar-experiment container.

Loads an ExperimentSpec from --spec-file, runs it through the orchestrator's
dispatch logic, and writes the ExperimentResult to --result-file.

RAIL 3 note: the manifest gate (authorization check) is enforced by the outer
orchestrator before it starts this container. _dispatch() is called directly
here to avoid re-entering the gate from inside Docker.

Usage (managed by ExperimentOrchestrator._run_in_docker — do not invoke manually):
    python docker_runner.py \\
        --workspace /workspace \\
        --spec-file  /workspace/tar_state/docker_runs/<id>/spec.json \\
        --result-file /workspace/tar_state/docker_runs/<id>/result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from tar_experiment_orchestrator import ExperimentOrchestrator, ExperimentResult, ExperimentSpec
from tar_storage import ensure_workspace_layout


def _spec_from_dict(d: dict) -> ExperimentSpec:
    """Reconstruct ExperimentSpec from a plain dict (produced by dataclasses.asdict)."""
    valid_keys = {f.name for f in fields(ExperimentSpec)}
    return ExperimentSpec(**{k: v for k, v in d.items() if k in valid_keys})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TAR Docker experiment runner")
    p.add_argument("--workspace", required=True, help="Absolute path to workspace root")
    p.add_argument("--spec-file", required=True, help="Path to ExperimentSpec JSON")
    p.add_argument("--result-file", required=True, help="Path to write ExperimentResult JSON")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    workspace = ensure_workspace_layout(Path(args.workspace), repo_root=_REPO)

    spec_raw = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
    spec = _spec_from_dict(spec_raw)

    print(
        f"[docker_runner] id={spec.id}  dataset={spec.dataset}"
        f"  method={spec.method}  seeds={spec.seeds}",
        flush=True,
    )

    # Instantiate the orchestrator so runner methods have access to self.workspace.
    # _dispatch() does not check the manifest gate — authorization was already
    # verified by the outer orchestrator before this container was launched.
    orch = ExperimentOrchestrator(workspace)
    result: ExperimentResult = orch._dispatch(spec)

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    print(
        f"[docker_runner] Complete. verdict={result.verdict}"
        f"  mean_forgetting={result.mean_forgetting:.4f}  p_val={result.p_val:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
