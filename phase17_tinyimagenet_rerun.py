"""
Phase 17 TinyImageNet Rerun — 5-seed, 7-method replacement
===========================================================

This script replaces phase17_tinyimagenet.py (old: 3 seeds, 3 methods, no
pre-registration requirement) with a Phase 2 PhD-rehabilitation-standard run:

  * 5 seeds  (SEEDS = [42, 0, 1, 2, 3])
  * 7 methods: tcl, ewc_generic, si_generic, sgd_generic,
               der_plus_plus, lwf, agem
  * Full Phase 1 metric suite: forgetting, accuracy, BWT, FWT,
    intransigence_index, ECE trajectory, per-seed detail
  * Pairwise statistical comparisons with Bonferroni correction (k=4)
    and Bayesian evidence

Pre-requisites (MUST be satisfied before running)
--------------------------------------------------
1. E:\\TAR\\...\\tar_state\\preregistrations\\phase17_rerun.json must exist.
   Create it with run_hyperparameter_selection.py or by hand before running.
2. E:\\TAR\\...\\tar_state\\execution_enabled.flag must exist.
   Create it to authorise autonomous-mode execution.
3. Hyperparameter values for DER++, LwF, A-GEM should be updated from
   hyperparameter_selection.json (Task 2.6) before running; the defaults
   below are conservative starting points from Phase 12/13 sweeps.

Hyperparameter notes
--------------------
EWC   lambda=1000  — Phase 12 sweep optimum (p=0.318 vs TCL at lambda=1000;
                     lambda=100 gave p=0.019, i.e. worse for EWC).
SI    c=0.01        — Phase 13 sweep: c=0.1 causes universal collapse (all 5
                     seeds → 0.500 accuracy). c=0.01 is the locked HPC value.
DER++ mem_size=200  — placeholder; update from hyperparameter_selection.json.
LwF   alpha=0.5, T=2.0  — literature defaults; update from Task 2.6 sweep.
A-GEM mem_size=200  — literature default.

Usage
-----
    python phase17_tinyimagenet_rerun.py

The script writes a single JSON result to:
    E:\\TAR\\...\\tar_state\\comparisons\\phase17_tinyimagenet_rerun_<timestamp>.json

Do NOT commit result files — they are tracked by the experiment registry.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any tar_lab imports
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

_ARROW_CACHE = (
    _REPO / "dataset_artifacts" / "tinyimagenet"
    / "Maysee___tiny-imagenet" / "default" / "0.0.0"
    / "5a77092c28e51558c5586e9c5eb71a7e17a5e43f"
)


def _load_tinyimagenet_from_arrow():
    """Load TinyImageNet from local Arrow cache, bypassing HF lock-file issues."""
    import datasets as hf_datasets
    if not _ARROW_CACHE.exists():
        raise FileNotFoundError(
            f"TinyImageNet Arrow cache not found at {_ARROW_CACHE}. "
            "Run phase17_tinyimagenet.py once to populate the cache."
        )
    train_ds = hf_datasets.Dataset.from_file(
        str(_ARROW_CACHE / "tiny-imagenet-train.arrow")
    )
    val_ds = hf_datasets.Dataset.from_file(
        str(_ARROW_CACHE / "tiny-imagenet-valid.arrow")
    )
    train_items = [(row["image"], int(row["label"])) for row in train_ds]
    val_items   = [(row["image"], int(row["label"])) for row in val_ds]
    print(f"  Loaded {len(train_items)} train / {len(val_items)} val from Arrow cache",
          flush=True)
    return train_items, val_items


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS    = [42, 0, 1, 2, 3]
BACKBONE = "resnet18"
DATASET  = "split_tinyimagenet"
N_TASKS  = 20           # split_tinyimagenet: 20 tasks × 10 classes = 200 classes
EPOCHS   = 40
DATA_ROOT = str(_REPO / "dataset_artifacts")

# Hyperparameters locked by Task 2.6 hyperparameter selection.
# EWC and SI values from Phase 12/13 sweeps (confirmed pre-hoc).
# DER++, LwF values are literature defaults — update from
# hyperparameter_selection.json (Task 2.6) before final paper run.
# NOTE: Run run_hyperparameter_selection.py BEFORE this script.
METHODS_CONFIG: dict[str, dict] = {
    # Registry key renamed tcl->tcl_canonical (truth-lock key-collision fix,
    # commit 33ad4a6): same canonical TCLMethod; only the lookup key changed.
    "tcl_canonical": {
        "tcl_penalty_lambda": 1.0,
        "tcl_ema_beta":       0.99,
        "tcl_governor_enabled": False,
    },
    "ewc_generic": {
        "ewc_lambda": 1000.0,   # Phase 12 optimum
    },
    "si_generic": {
        "si_c":  0.01,          # Phase 13 optimum (c=0.1 collapses on CIFAR)
        "si_xi": 0.001,
    },
    "sgd_generic": {},
    "der_plus_plus": {
        "der_mem_size": 200,    # Update from hyperparameter_selection.json
    },
    "lwf": {
        "lwf_alpha":       0.5,    # Update from hyperparameter_selection.json
        "lwf_temperature": 2.0,
    },
    "agem": {
        "agem_mem_size": 200,
    },
}

OUTPUT_DIR   = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state\comparisons")
RESULT_ID    = f"phase17_tinyimagenet_rerun_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
PREREG_FILE  = Path(
    r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state\preregistrations\phase17_rerun.json"
)
EXEC_FLAG    = Path(
    r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state\execution_enabled.flag"
)

# Bonferroni family: tcl vs ewc, tcl vs sgd, tcl vs der_plus_plus, tcl vs lwf
BONFERRONI_K = 4

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def _preflight() -> None:
    """Abort early if pre-registration or execution flag are missing."""
    if not PREREG_FILE.exists():
        print(
            f"\nABORTED: Pre-registration file not found:\n  {PREREG_FILE}\n"
            "Create it (or run run_hyperparameter_selection.py) before executing "
            "this Phase 2 experiment.",
            flush=True,
        )
        sys.exit(1)

    if not EXEC_FLAG.exists():
        print(
            f"\nABORTED: execution_enabled.flag not found:\n  {EXEC_FLAG}\n"
            "The system must be explicitly enabled before autonomous-mode "
            "experiments can run.\n"
            "Create the flag file to authorise execution.",
            flush=True,
        )
        sys.exit(1)

    print(f"[preflight] Pre-registration: OK  ({PREREG_FILE.name})", flush=True)
    print(f"[preflight] Execution flag:   OK  ({EXEC_FLAG.name})", flush=True)

# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------

def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _std(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _build_method_aggregate(
    method_name: str,
    seed_results: list[dict],
    forgetting_list: list[float],
    accuracy_list: list[float],
) -> dict:
    """Compute per-method aggregate statistics from seed_results."""
    from tar_lab.stat_utils import compute_ci95

    ci = compute_ci95(forgetting_list)

    bwt_vals          = [r["bwt"]                 for r in seed_results]
    fwt_vals          = [r["fwt"]                 for r in seed_results]
    intrans_vals      = [r["intransigence_index"]  for r in seed_results]

    return {
        "seed_results":     seed_results,
        "mean_forgetting":  round(ci.mean, 6),
        "std_forgetting":   round(ci.std,  6),
        "ci95_forgetting":  [round(ci.ci_low, 6), round(ci.ci_high, 6)],
        "ci_method":        ci.ci_method,
        "mean_accuracy":    round(_mean(accuracy_list), 6),
        "mean_bwt":         round(_mean(bwt_vals),      6),
        "mean_fwt":         round(_mean(fwt_vals),      6),
        "mean_intransigence": round(_mean(intrans_vals), 6),
    }


def _direction_label(mean_tcl: float, mean_other: float) -> str:
    if abs(mean_tcl - mean_other) < 1e-6:
        return "tied"
    return "tcl_better" if mean_tcl < mean_other else "other_better"


def _build_comparison(
    label: str,
    tcl_forgetting: list[float],
    other_forgetting: list[float],
    bonferroni_k: int,
) -> dict:
    """Run paired stats for tcl vs one comparator; return serialisable dict."""
    from tar_lab.stat_utils import (
        compare_methods_paired,
        apply_bonferroni_to_comparison,
        bayesian_evidence,
    )

    comp = compare_methods_paired(tcl_forgetting, other_forgetting, alternative="less")
    comp = apply_bonferroni_to_comparison(comp, k=bonferroni_k)

    deltas = [t - o for t, o in zip(tcl_forgetting, other_forgetting)]
    bayes  = bayesian_evidence(deltas)

    return {
        "label":                  label,
        "wilcoxon_p":             round(comp.p_nonparametric, 6),
        "wilcoxon_statistic":     round(comp.statistic_nonparametric, 6),
        "paired_t_p":             round(comp.p_parametric,    6),
        "cohens_d":               round(comp.cohens_d,        6),
        "mean_tcl_forgetting":    round(comp.mean_a,          6),
        "mean_other_forgetting":  round(comp.mean_b,          6),
        "mean_delta":             round(comp.mean_delta,      6),
        "bonferroni_k":           bonferroni_k,
        "bonferroni_threshold":   round(comp.bonferroni_threshold, 6),
        "bonferroni_significant": comp.bonferroni_significant,
        "bayesian_p_tcl_better":  round(bayes.posterior_p_better,   6),
        "bayesian_mean_delta":    round(bayes.posterior_mean_delta,  6),
        "bayesian_ci95":          [round(bayes.credible_interval_95[0], 6),
                                   round(bayes.credible_interval_95[1], 6)],
        "bayesian_interpretation": bayes.interpretation,
        "direction":              _direction_label(comp.mean_a, comp.mean_b),
    }


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------

def _honest_verdict(
    tcl_agg: dict,
    comparisons: dict,
) -> str:
    """
    Produce a brief, factual summary of what the results show.

    Mirrors the framing from project_hpc_result_framing.md:
    - Report magnitude honestly (not just direction).
    - Do not over-claim; use hedged language where n=5 limits power.
    """
    lines = []
    tcl_f  = tcl_agg["mean_forgetting"]
    tcl_a  = tcl_agg["mean_accuracy"]
    lines.append(
        f"TCL: mean_forgetting={tcl_f:.4f}, mean_accuracy={tcl_a:.4f} "
        f"(n={len(SEEDS)} seeds, {EPOCHS} epochs/task, {N_TASKS} tasks, {DATASET})"
    )
    for cname, cdata in comparisons.items():
        sig  = cdata["bonferroni_significant"]
        d    = cdata["cohens_d"]
        dirn = cdata["direction"]
        wp   = cdata["wilcoxon_p"]
        lines.append(
            f"  {cname}: direction={dirn}, wilcoxon_p={wp:.4f}, "
            f"cohens_d={d:.3f}, bonferroni_significant={sig}"
        )
    return " | ".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _preflight()

    from tar_lab.generic_cl_runner import run_generic_benchmark

    print(f"\n{'='*72}")
    print(f"Phase 17 TinyImageNet Rerun — PhD Rehabilitation Plan Phase 2")
    print(f"dataset={DATASET}  backbone={BACKBONE}  epochs={EPOCHS}  seeds={SEEDS}")
    print(f"methods={list(METHODS_CONFIG)}")
    print(f"result_id={RESULT_ID}")
    print(f"{'='*72}", flush=True)

    import phase17_tinyimagenet as _p17
    print("[TinyImageNet] Loading dataset from Arrow cache...", flush=True)
    _ti_train_items, _ti_val_items = _load_tinyimagenet_from_arrow()

    method_results: dict[str, dict] = {}

    for method_name, config_overrides in METHODS_CONFIG.items():
        print(f"\n[run] method={method_name}  config={config_overrides}", flush=True)

        seed_results_all: list = []
        forgetting_list: list[float] = []
        accuracy_list: list[float] = []
        for seed in SEEDS:
            from torch.utils.data import DataLoader
            _subsets_train, _subsets_test = _p17._build_tinyimagenet_tasks(
                seed, _ti_train_items, _ti_val_items, backbone=BACKBONE
            )
            prebuilt_train = [DataLoader(ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False) for ds in _subsets_train]
            prebuilt_test  = [DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False) for ds in _subsets_test]
            sr, fl, al = run_generic_benchmark(
                dataset_name        = DATASET,
                backbone_name       = BACKBONE,
                method_name         = method_name,
                seeds               = [seed],
                epochs              = EPOCHS,
                config_overrides    = config_overrides,
                data_root           = DATA_ROOT,
                log_fn              = print,
                prebuilt_task_train = prebuilt_train,
                prebuilt_task_test  = prebuilt_test,
            )
            seed_results_all.extend(sr)
            forgetting_list.extend(fl)
            accuracy_list.extend(al)

        seed_results = seed_results_all

        method_results[method_name] = _build_method_aggregate(
            method_name, seed_results, forgetting_list, accuracy_list
        )

        mf = method_results[method_name]["mean_forgetting"]
        ma = method_results[method_name]["mean_accuracy"]
        print(
            f"[done] method={method_name}  mean_forgetting={mf:.4f}"
            f"  mean_accuracy={ma:.4f}",
            flush=True,
        )

    # ---- Statistical comparisons ----------------------------------------
    print("\n[stats] Computing pairwise comparisons ...", flush=True)

    tcl_f = [r["forgetting"] for r in method_results["tcl"]["seed_results"]]

    comparisons: dict[str, dict] = {}

    for label, cmp_method in [
        ("tcl_vs_ewc",         "ewc_generic"),
        ("tcl_vs_sgd",         "sgd_generic"),
        ("tcl_vs_der_plus_plus", "der_plus_plus"),
        ("tcl_vs_lwf",         "lwf"),
    ]:
        other_f = [r["forgetting"] for r in method_results[cmp_method]["seed_results"]]
        comparisons[label] = _build_comparison(label, tcl_f, other_f, BONFERRONI_K)
        print(
            f"  {label}: wilcoxon_p={comparisons[label]['wilcoxon_p']:.4f}"
            f"  bonferroni_significant={comparisons[label]['bonferroni_significant']}",
            flush=True,
        )

    # ---- Assemble output JSON -------------------------------------------
    completed_at = datetime.now(timezone.utc).isoformat()

    output: dict[str, Any] = {
        "result_id":            RESULT_ID,
        "dataset":              DATASET,
        "backbone":             BACKBONE,
        "seeds":                SEEDS,
        "epochs":               EPOCHS,
        "n_tasks":              N_TASKS,
        "methods_config":       METHODS_CONFIG,
        "preregistration_file": str(PREREG_FILE),
        "completed_at":         completed_at,
        "method_results":       method_results,
        "comparisons":          comparisons,
        "trust_tier":           "trusted_rerun",
        "gate_a_passed":        True,   # pre-registration checked at startup
        "gate_b_passed":        True,   # n_seeds=5 >= 5
        "honest_verdict":       _honest_verdict(method_results["tcl"], comparisons),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{RESULT_ID}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n[done] Results written to: {out_path}", flush=True)
    print(f"[done] Honest verdict: {output['honest_verdict'][:200]}", flush=True)


if __name__ == "__main__":
    main()
