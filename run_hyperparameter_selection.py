"""
Phase 2 Task 2.6 — Fair Joint Hyperparameter Selection

Runs all baseline methods with each hyperparameter value on the held-out
validation split (Split-CIFAR-10, seed=999, 20 epochs). Selects the best
configuration per method by lowest mean_forgetting subject to accuracy > threshold.
Writes tar_state/hyperparameter_selection.json with locked values.

This script MUST be run and its output committed BEFORE running any
confirmatory experiments (phase16_rerun, phase17_rerun, run_hpc_replication).

Usage: python run_hyperparameter_selection.py [--dry-run]
  --dry-run: Print what would be run without executing (for verification)
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path setup — must come before any tar_lab import
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))

from tar_lab.generic_cl_runner import run_generic_benchmark  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALIDATION_SEED      = 999
VALIDATION_DATASET   = "split_cifar10"
BACKBONE             = "resnet18"
VALIDATION_EPOCHS    = 20    # Faster than full 40; sufficient for relative ranking
DATA_ROOT            = str(_REPO_ROOT / "dataset_artifacts")

# TAR_STATE lives on E:\ in production; fall back to a local subdir if absent
_TAR_STATE_CANDIDATES = [
    Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state"),
    _REPO_ROOT.parent / "tar_state",
    _REPO_ROOT / "tar_state",
]
TAR_STATE = next((p for p in _TAR_STATE_CANDIDATES if p.exists()), _REPO_ROOT / "tar_state")

OUTPUT_JSON = TAR_STATE / "hyperparameter_selection.json"
EXECUTION_FLAG = TAR_STATE / "execution_enabled.flag"

# ---------------------------------------------------------------------------
# Hyperparameter sweeps per method
# ---------------------------------------------------------------------------

SWEEPS: dict[str, list[dict]] = {
    "ewc_generic": [
        {"ewc_lambda": 10.0},
        {"ewc_lambda": 100.0},
        {"ewc_lambda": 1000.0},
        {"ewc_lambda": 5000.0},
    ],
    "si_generic": [
        {"si_c": 0.001, "si_xi": 0.001},
        {"si_c": 0.01,  "si_xi": 0.001},
        {"si_c": 0.1,   "si_xi": 0.001},
        {"si_c": 1.0,   "si_xi": 0.001},
    ],
    "der_plus_plus": [
        {"der_mem_size": 100},
        {"der_mem_size": 200},
        {"der_mem_size": 500},
    ],
    "lwf": [
        {"lwf_alpha": 0.3, "lwf_temperature": 1.0},
        {"lwf_alpha": 0.3, "lwf_temperature": 2.0},
        {"lwf_alpha": 0.5, "lwf_temperature": 1.0},
        {"lwf_alpha": 0.5, "lwf_temperature": 2.0},
        {"lwf_alpha": 1.0, "lwf_temperature": 1.0},
        {"lwf_alpha": 1.0, "lwf_temperature": 2.0},
    ],
    # TCL: fixed — do not tune post-hoc
    # SGD: no hyperparameters to tune
}

# TCL reference configuration
# Registry key renamed tcl->tcl_canonical (truth-lock key-collision fix, 33ad4a6):
# same built-in canonical TCLMethod as before; only the lookup key changed.
TCL_METHOD_KEY = "tcl_canonical"
TCL_REFERENCE_CONFIG: dict = {
    "tcl_penalty_lambda": 1.0,
    "tcl_ema_beta":       0.99,
    "tcl_governor_enabled": False,   # governor disabled: isolates elastic regularization
}

# Selection criterion: lowest mean_forgetting where mean_accuracy >= threshold
ACCURACY_THRESHOLDS: dict[str, float] = {
    "ewc_generic":   0.60,   # Must achieve >60% mean accuracy (not collapse)
    "si_generic":    0.50,   # SI can collapse to 0.500 — reject those configs
    "der_plus_plus": 0.55,
    "lwf":           0.55,
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _preflight(dry_run: bool) -> None:
    """Abort early if safety rails are violated."""
    if not dry_run:
        if not EXECUTION_FLAG.exists():
            print(
                f"[ABORT] execution_enabled.flag not found at:\n"
                f"  {EXECUTION_FLAG}\n"
                f"Create this file to permit GPU execution.",
                flush=True,
            )
            sys.exit(1)

    if OUTPUT_JSON.exists():
        print(
            f"[ABORT] hyperparameter_selection.json already exists at:\n"
            f"  {OUTPUT_JSON}\n"
            f"Selection is LOCKED — do not re-run. Delete the file manually only "
            f"if you are certain you need to redo the selection.",
            flush=True,
        )
        sys.exit(1)

    print(f"[preflight] validation_seed   = {VALIDATION_SEED}", flush=True)
    print(f"[preflight] validation_dataset = {VALIDATION_DATASET}", flush=True)
    print(f"[preflight] backbone           = {BACKBONE}", flush=True)
    print(f"[preflight] validation_epochs  = {VALIDATION_EPOCHS}", flush=True)
    print(f"[preflight] data_root          = {DATA_ROOT}", flush=True)
    print(f"[preflight] tar_state          = {TAR_STATE}", flush=True)
    print(f"[preflight] output_json        = {OUTPUT_JSON}", flush=True)

# ---------------------------------------------------------------------------
# Dry-run print
# ---------------------------------------------------------------------------

def _print_dry_run() -> None:
    """Print what would be run without executing anything."""
    total_runs = 0
    print("\n=== DRY RUN — hyperparameter sweep plan ===\n")

    # TCL reference
    print(f"[TCL reference] method={TCL_METHOD_KEY!r}  config={TCL_REFERENCE_CONFIG}")
    total_runs += 1

    # Method sweeps
    for method, configs in SWEEPS.items():
        threshold = ACCURACY_THRESHOLDS.get(method, "n/a")
        print(f"\n[{method}] accuracy_threshold={threshold}  ({len(configs)} configs)")
        for i, cfg in enumerate(configs):
            print(f"  [{i}] {cfg}")
            total_runs += 1

    print(f"\nTotal runs: {total_runs}")
    print(f"Output would be written to: {OUTPUT_JSON}")
    print("\n=== END DRY RUN ===")

# ---------------------------------------------------------------------------
# Single benchmark run helper
# ---------------------------------------------------------------------------

def _run_one(
    method_name: str,
    config_overrides: dict,
) -> tuple[float, float]:
    """
    Run run_generic_benchmark for a single method + config on the validation seed.

    Returns (mean_forgetting, mean_accuracy).
    Raises on method-not-found or training failure — caller logs and handles.
    """
    seed_results, forgetting_list, accuracy_list = run_generic_benchmark(
        dataset_name     = VALIDATION_DATASET,
        backbone_name    = BACKBONE,
        method_name      = method_name,
        seeds            = [VALIDATION_SEED],
        epochs           = VALIDATION_EPOCHS,
        config_overrides = config_overrides,
        data_root        = DATA_ROOT,
        log_fn           = print,
    )

    mean_forgetting = forgetting_list[0] if forgetting_list else float("nan")
    mean_accuracy   = accuracy_list[0]   if accuracy_list   else float("nan")
    return mean_forgetting, mean_accuracy

# ---------------------------------------------------------------------------
# Best-config selection
# ---------------------------------------------------------------------------

def _select_best(
    method: str,
    sweep_results: list[dict],
) -> dict:
    """
    From sweep_results, select the config with lowest mean_forgetting
    subject to mean_accuracy >= ACCURACY_THRESHOLDS[method].

    Returns the winning entry (with "selected": True already set by caller)
    or a "collapsed" sentinel if no config passes the threshold.
    """
    threshold = ACCURACY_THRESHOLDS.get(method, 0.0)
    passing   = [r for r in sweep_results if r["mean_accuracy"] >= threshold]

    if not passing:
        return {
            "status":  "collapsed",
            "note":    (
                f"No config achieved mean_accuracy >= {threshold:.2f}. "
                f"All {len(sweep_results)} configs below threshold. "
                f"Best accuracy was "
                f"{max(r['mean_accuracy'] for r in sweep_results):.4f}."
            ),
            "all_configs_tried": [r["config"] for r in sweep_results],
        }

    best = min(passing, key=lambda r: r["mean_forgetting"])
    result = dict(best["config"])
    result["mean_forgetting"] = round(best["mean_forgetting"], 6)
    result["mean_accuracy"]   = round(best["mean_accuracy"],   6)
    return result

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 2.6 — Fair joint hyperparameter selection on seed=999 validation split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing (for verification).",
    )
    args = parser.parse_args()

    _preflight(dry_run=args.dry_run)

    if args.dry_run:
        _print_dry_run()
        return

    # ------------------------------------------------------------------
    # 1. TCL reference run
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print(f"[TCL reference] method={TCL_METHOD_KEY!r}  config={TCL_REFERENCE_CONFIG}", flush=True)
    print("=" * 60, flush=True)

    tcl_reference: dict
    try:
        tcl_forg, tcl_acc = _run_one(TCL_METHOD_KEY, TCL_REFERENCE_CONFIG)
        tcl_reference = {
            "config":           TCL_REFERENCE_CONFIG,
            "mean_forgetting":  round(tcl_forg, 6),
            "mean_accuracy":    round(tcl_acc,  6),
        }
        print(
            f"[TCL reference] forgetting={tcl_forg:.4f}  accuracy={tcl_acc:.4f}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[TCL reference] FAILED: {exc}\n"
            f"  TCL method key '{TCL_METHOD_KEY}' may not be registered in METHOD_REGISTRY.\n"
            f"  Ensure tar_state/synthesized_methods/tcl.py exists or the method is built-in.\n"
            f"  Continuing with remaining methods — tcl_reference will be marked as failed.",
            flush=True,
        )
        tcl_reference = {
            "config":  TCL_REFERENCE_CONFIG,
            "status":  "failed",
            "error":   str(exc),
        }

    # ------------------------------------------------------------------
    # 2. Sweep each method
    # ------------------------------------------------------------------
    full_sweep_results:  dict[str, list[dict]] = {}
    selected:            dict[str, dict]       = {}

    for method, configs in SWEEPS.items():
        threshold = ACCURACY_THRESHOLDS.get(method, 0.0)
        print(f"\n{'=' * 60}", flush=True)
        print(
            f"[sweep] method={method!r}  "
            f"n_configs={len(configs)}  "
            f"accuracy_threshold={threshold:.2f}",
            flush=True,
        )
        print("=" * 60, flush=True)

        method_results: list[dict] = []

        for i, cfg in enumerate(configs):
            print(
                f"\n[{method}] config {i + 1}/{len(configs)}: {cfg}",
                flush=True,
            )
            try:
                forg, acc = _run_one(method, cfg)
                entry = {
                    "config":           cfg,
                    "mean_forgetting":  round(forg, 6),
                    "mean_accuracy":    round(acc,  6),
                    "selected":         False,       # overwritten below for winner
                }
                print(
                    f"[{method}] forgetting={forg:.4f}  accuracy={acc:.4f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{method}] config {cfg} FAILED: {exc}", flush=True)
                entry = {
                    "config":  cfg,
                    "status":  "failed",
                    "error":   str(exc),
                    "selected": False,
                }
            method_results.append(entry)

        full_sweep_results[method] = method_results

        # Select best from this method's valid (non-failed) results
        valid_results = [r for r in method_results if "mean_forgetting" in r]
        if valid_results:
            best_entry = _select_best(method, valid_results)
            selected[method] = best_entry

            # Mark the winning entry in full_sweep_results
            if best_entry.get("status") != "collapsed":
                for r in method_results:
                    if r.get("config") == best_entry.get(
                        # rebuild the config dict from best_entry keys
                        "config",
                        {k: best_entry[k] for k in best_entry
                         if k not in {"mean_forgetting", "mean_accuracy", "selected"}},
                    ):
                        r["selected"] = True
                        break
        else:
            selected[method] = {
                "status": "all_failed",
                "note":   f"All {len(method_results)} configs raised exceptions.",
            }

        print(f"\n[{method}] selected: {selected[method]}", flush=True)

    # ------------------------------------------------------------------
    # 3. Write locked output JSON
    # ------------------------------------------------------------------
    output: dict = {
        "locked_at":          datetime.now(tz=timezone.utc).isoformat(),
        "validation_seed":    VALIDATION_SEED,
        "validation_dataset": VALIDATION_DATASET,
        "validation_epochs":  VALIDATION_EPOCHS,
        "note": (
            "LOCKED — do not re-run. "
            "Commit this file before running confirmatory experiments."
        ),

        "tcl_reference":      tcl_reference,
        "selected":           selected,
        "full_sweep_results": full_sweep_results,

        "provenance": (
            "Generated by run_hyperparameter_selection.py "
            "(Task 2.6, TAR PhD Rehabilitation Plan)"
        ),
    }

    TAR_STATE.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n[done] hyperparameter_selection.json written to:\n  {OUTPUT_JSON}", flush=True)
    print(
        "[next] Commit hyperparameter_selection.json BEFORE running any "
        "confirmatory experiments (phase16_rerun, phase17_rerun, run_hpc_replication).",
        flush=True,
    )


if __name__ == "__main__":
    main()
