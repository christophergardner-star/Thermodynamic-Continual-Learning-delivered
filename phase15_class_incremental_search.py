from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.manifest import load_and_verify_manifest, ManifestGateError, write_refuse_note
from tar_lab.result_artifacts import collect_environment_snapshot, write_canonical_comparison_result


WORKSPACE = Path(__file__).resolve().parent
PROBLEM_ID = "phase15-class-incremental-tcl-search"
SEEDS = [42, 0, 1]
BACKBONE = "resnet18"
EPOCHS = 40


def _require_manifest() -> None:
    manifest_path_str = str(os.environ.get("TAR_MANIFEST_PATH", "") or "").strip()
    if not manifest_path_str:
        print("REFUSED: TAR_MANIFEST_PATH not set. Set it to the path of the signed manifest and re-run.", flush=True)
        raise SystemExit(1)
    manifest_path = Path(manifest_path_str)
    if not manifest_path.is_absolute():
        manifest_path = WORKSPACE / manifest_path
    try:
        manifest = load_and_verify_manifest(manifest_path, WORKSPACE)
        for experiment_id in ("phase15_class_incremental_search", "phase15-class-incremental-rerun"):
            try:
                manifest.assert_experiment_authorised(experiment_id)
                print(f"[RAIL 3] Manifest gate: OK ({manifest.manifest_id})", flush=True)
                return
            except ManifestGateError:
                continue
        raise ManifestGateError(
            "Manifest does not authorise any accepted Phase 15 execution id "
            "('phase15_class_incremental_search', 'phase15-class-incremental-rerun')."
        )
    except ManifestGateError as exc:
        write_refuse_note(
            WORKSPACE,
            component="phase15_class_incremental_search",
            reason=str(exc),
            experiment_id="phase15_class_incremental_search",
            manifest_path=str(manifest_path),
        )
        print(f"REFUSED: {exc}", flush=True)
        raise SystemExit(1)


def main() -> int:
    _require_manifest()
    orchestrator = TAROrchestrator(str(WORKSPACE))
    try:
        run_started_at = datetime.utcnow().isoformat()
        workspace_path = Path(orchestrator.workspace)
        print("=" * 70)
        print("Phase 15 - TCL Mechanism Search (class-incremental Split-CIFAR-10)")
        print(f"workspace={workspace_path}")
        print(f"problem_id={PROBLEM_ID}")
        print(f"backbone={BACKBONE} epochs={EPOCHS} seeds={SEEDS}")
        print(run_started_at)
        print("=" * 70, flush=True)

        result = orchestrator.run_tcl_class_incremental_mechanism_search(
            problem_id=PROBLEM_ID,
            seeds=SEEDS,
            backbone=BACKBONE,
            train_epochs_per_task=EPOCHS,
        )
        artifact_path = workspace_path / "tar_state" / "comparisons" / f"{result.search_id}.json"

        phase_summary = {**result.model_dump(), "phase": 15}
        completed_at = datetime.utcnow().isoformat()
        env_payload = collect_environment_snapshot(
            repo_root=WORKSPACE,
            workspace=workspace_path,
            config={
                "suite": "phase15_class_incremental_search",
                "problem_id": PROBLEM_ID,
                "seeds": SEEDS,
                "backbone": BACKBONE,
                "train_epochs_per_task": EPOCHS,
            },
            trigger="manual_script",
            source_script=Path(__file__).name,
            run_started_at=run_started_at,
            run_ended_at=completed_at,
            extra={
                "logical_name": "phase15_class_incremental_search",
                "search_id": result.search_id,
                "underlying_artifact_path": str(artifact_path),
            },
        )
        comparison_artifacts = write_canonical_comparison_result(
            workspace=workspace_path,
            logical_name="phase15_class_incremental_search",
            payload={**phase_summary, "completed_at": completed_at},
            env_payload=env_payload,
            phase_number=15,
            source_script=Path(__file__).name,
        )

        print(result.summary)
        print(f"best_candidate={result.best_candidate_name}")
        print(
            f"delta_vs_{result.strong_baseline_method}="
            f"{result.best_delta_vs_strong_baseline:+.4f} "
            f"p={result.p_value_vs_strong_baseline:.4f} "
            f"d={result.effect_size_vs_strong_baseline:.3f}"
        )
        print(f"publishability_status={result.publishability_status}")
        print(f"external_breakthrough_candidate={result.external_breakthrough_candidate}")
        print(f"artifact={artifact_path}")
        print(f"phase_summary={comparison_artifacts['result_path']}")
        print(f"phase_summary_env={comparison_artifacts['env_path']}")
        if artifact_path.exists():
            print(json.loads(artifact_path.read_text(encoding='utf-8')).get("summary", ""))
        return 0
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
