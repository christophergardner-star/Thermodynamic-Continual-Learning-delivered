"""
Merge Phase 11 ablation data (4 conditions) with the 3-condition run to produce
the final 7-condition mechanistic ablation result.

Usage:
    python merge_mechanistic_ablation.py <new_3_conditions_json>

Output:
    tar_state/comparisons/mechanistic_ablation_7condition_final.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_TAR_STATE = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state")
sys.path.insert(0, str(_REPO))

PHASE11_FILE = _TAR_STATE / "comparisons" / "phase11_ablation__20260511T113318Z.json"
OUTPUT_FILE = _TAR_STATE / "comparisons" / "mechanistic_ablation_7condition_final.json"
BONFERRONI_K = 6
BONFERRONI_THRESHOLD = 0.05 / BONFERRONI_K  # ~0.00833

# Phase 11 uses 'sgd' internally; canonical name is 'sgd_baseline'
PHASE11_NAME_MAP = {
    "sgd": "sgd_baseline",
    "governor_only": "governor_only",
    "penalty_only": "penalty_only",
    "full_tcl": "full_tcl",
}
EXPECTED_CONDITIONS = {
    "sgd_baseline", "penalty_only", "governor_only", "full_tcl",
    "anchor_frozen_init", "warmup_batches_60", "ewc_best_lambda",
}


def _extract_phase11_per_condition(p11: dict) -> dict[str, list[float]]:
    """Extract per-condition forgetting lists from Phase 11 flat per_seed list."""
    per_seed_list: list[dict] = p11["per_seed"]  # [{seed, sgd_forgetting, ...}]
    result: dict[str, list[float]] = {}
    for p11_name, canonical in PHASE11_NAME_MAP.items():
        key = f"{p11_name}_forgetting"
        vals = [row[key] for row in per_seed_list if key in row]
        if vals:
            result[canonical] = vals
    return result


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5


def main(new_file: Path) -> None:
    from tar_lab.stat_utils import compare_methods_paired

    p11 = json.loads(PHASE11_FILE.read_text())
    new_data = json.loads(new_file.read_text())

    # Extract per-condition forgetting lists
    per_condition: dict[str, list[float]] = {}

    # From Phase 11 (4 conditions)
    per_condition.update(_extract_phase11_per_condition(p11))

    # From new 3-condition run
    new_raw: dict = new_data.get("per_condition_forgetting_raw", {})
    for cond, vals in new_raw.items():
        if cond in per_condition:
            print(f"WARNING: {cond!r} already present from Phase 11 — keeping Phase 11 data")
        else:
            per_condition[cond] = vals

    missing = EXPECTED_CONDITIONS - set(per_condition.keys())
    if missing:
        raise ValueError(f"Missing conditions after merge: {missing}")

    seeds = p11["seeds"]

    # Aggregate stats per condition
    aggregate = {
        cond: {
            "forgetting_mean": _mean(vals),
            "forgetting_std": _std(vals),
            "n_seeds": len(vals),
        }
        for cond, vals in per_condition.items()
    }

    # Pairwise comparisons vs full_tcl (Bonferroni k=6)
    reference = "full_tcl"
    comparators = [c for c in EXPECTED_CONDITIONS if c != reference]
    pairwise = []
    for comp in sorted(comparators):
        ref_vals = per_condition[reference]
        comp_vals = per_condition[comp]
        result = compare_methods_paired(ref_vals, comp_vals)
        p_raw = result.p_nonparametric
        significant = bool(p_raw < BONFERRONI_THRESHOLD) if p_raw == p_raw else False  # NaN guard
        pairwise.append({
            "comparison": f"{reference}_vs_{comp}",
            "p_wilcoxon": p_raw,
            "bonferroni_k": BONFERRONI_K,
            "bonferroni_threshold": BONFERRONI_THRESHOLD,
            "bonferroni_significant": significant,
            "cohens_d": result.cohens_d,
            "mean_full_tcl": result.mean_a,
            "mean_comparator": result.mean_b,
        })

    output = {
        "experiment_id": "mechanistic_ablation_7condition_final",
        "merged_from": [str(PHASE11_FILE), str(new_file)],
        "conditions": sorted(per_condition.keys()),
        "seeds": seeds,
        "bonferroni_k": BONFERRONI_K,
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
        "per_condition_forgetting_raw": per_condition,
        "aggregate": aggregate,
        "pairwise": pairwise,
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nWritten: {OUTPUT_FILE}")
    print(f"Conditions: {sorted(per_condition.keys())}")
    print(f"\nAggregate forgetting (mean ± std):")
    for cond in sorted(aggregate.keys()):
        a = aggregate[cond]
        print(f"  {cond}: {a['forgetting_mean']:.4f} ± {a['forgetting_std']:.4f}")
    print(f"\nPairwise vs {reference} (Bonferroni threshold={BONFERRONI_THRESHOLD:.4f}):")
    for p in pairwise:
        sig = "SIGNIFICANT" if p["bonferroni_significant"] else "ns"
        print(f"  {p['comparison']}: p={p['p_wilcoxon']:.4f} [{sig}]  d={p['cohens_d']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <new_3_conditions_json>")
        sys.exit(1)
    main(Path(sys.argv[1]))
