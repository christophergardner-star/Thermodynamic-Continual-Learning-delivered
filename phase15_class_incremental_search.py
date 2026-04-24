from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tar_lab.orchestrator import TAROrchestrator


WORKSPACE = Path(__file__).resolve().parent
PROBLEM_ID = "phase15-class-incremental-tcl-search"
SEEDS = [42, 0, 1]
BACKBONE = "resnet18"
EPOCHS = 40


def main() -> int:
    orchestrator = TAROrchestrator(str(WORKSPACE))
    try:
        print("=" * 70)
        print("Phase 15 - TCL Mechanism Search (class-incremental Split-CIFAR-10)")
        print(f"workspace={WORKSPACE}")
        print(f"problem_id={PROBLEM_ID}")
        print(f"backbone={BACKBONE} epochs={EPOCHS} seeds={SEEDS}")
        print(datetime.utcnow().isoformat())
        print("=" * 70, flush=True)

        result = orchestrator.run_tcl_class_incremental_mechanism_search(
            problem_id=PROBLEM_ID,
            seeds=SEEDS,
            backbone=BACKBONE,
            train_epochs_per_task=EPOCHS,
        )
        artifact_path = WORKSPACE / "tar_state" / "comparisons" / f"{result.search_id}.json"
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
        if artifact_path.exists():
            print(json.loads(artifact_path.read_text(encoding='utf-8')).get("summary", ""))
        return 0
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
