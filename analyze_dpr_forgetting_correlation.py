"""
Phase 3 Task 3.5 — D_PR–Forgetting Correlation Analysis

Tests the hypothesis that D_PR compression at task boundaries correlates
with catastrophic forgetting.

Two modes:
  Mode 1: LIVE — runs new experiments with compute_dpr=True and records
          per-task D_PR alongside forgetting (requires GPU + execution_enabled.flag)
  Mode 2: ANALYSIS — reads existing comparison JSON files that contain
          'dpr_per_task' field (from run_generic_benchmark with compute_dpr=True)

Usage:
  python analyze_dpr_forgetting_correlation.py --mode analysis  # uses existing JSON
  python analyze_dpr_forgetting_correlation.py --mode live      # runs new experiments
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent
_TAR_STATE = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state")
_COMPARISONS_DIR = _TAR_STATE / "comparisons"
_STAT_AUDIT_DIR  = _TAR_STATE / "stat_audit"
_OUTPUT_PATH     = _STAT_AUDIT_DIR / "dpr_forgetting_correlation.json"
_EXEC_FLAG       = _TAR_STATE / "execution_enabled.flag"
_DATA_ROOT       = str(_REPO / "dataset_artifacts")

# ---------------------------------------------------------------------------
# Live-mode experiment configuration
# ---------------------------------------------------------------------------

# Registry key renamed tcl->tcl_canonical (truth-lock key-collision fix, 33ad4a6)
LIVE_METHODS  = ["tcl_canonical", "ewc_generic", "sgd_generic", "der_plus_plus"]
LIVE_SEEDS    = [42, 0, 1, 2, 3]
LIVE_DATASET  = "split_cifar10"
LIVE_EPOCHS   = 40
LIVE_BACKBONE = "resnet18"

# Default config overrides for each method (mirrors phase10 / phase16 rerun defaults)
_METHOD_CONFIGS: dict[str, dict] = {
    "tcl_canonical": {"ewc_lambda": 400.0, "alpha": 0.5},
    "ewc_generic":  {"ewc_lambda": 1000.0},
    "sgd_generic":  {},
    "der_plus_plus": {"der_mem_size": 200, "der_alpha": 0.1, "der_beta": 0.5},
}

# ---------------------------------------------------------------------------
# Spearman correlation (pure Python — no scipy required for this sub-task,
# but scipy is preferred if available)
# ---------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Return fractional (average) ranks for a list of floats."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Compute Spearman rank correlation and two-tailed p-value.

    Falls back to scipy.stats.spearmanr when available (recommended),
    otherwise uses the exact rank-correlation formula with t-approximation.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    n = len(x)
    if n < 3:
        return (float("nan"), float("nan"))

    try:
        from scipy.stats import spearmanr as _scipy_spearman
        result = _scipy_spearman(x, y)
        # scipy ≥ 1.9 returns a SpearmanrResult; older versions return (rho, p)
        rho = float(result.statistic if hasattr(result, "statistic") else result[0])
        p   = float(result.pvalue    if hasattr(result, "pvalue")    else result[1])
        return (rho, p)
    except ImportError:
        pass

    # Pure-Python fallback: rank-correlation + t-approximation
    rx = _rank(x)
    ry = _rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_rx = sum((v - mean_rx) ** 2 for v in rx)
    var_ry = sum((v - mean_ry) ** 2 for v in ry)
    denom = math.sqrt(var_rx * var_ry)
    rho = cov / denom if denom > 1e-12 else 0.0
    # t-statistic with df = n - 2
    df = n - 2
    t_stat = rho * math.sqrt(df) / math.sqrt(max(1.0 - rho ** 2, 1e-12))
    # two-tailed p approximation via normal when df is large, else note imprecision
    try:
        from scipy.stats import t as _t_dist
        p = float(2.0 * _t_dist.sf(abs(t_stat), df))
    except ImportError:
        # Very rough normal approximation — flag as imprecise
        z = abs(t_stat)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return (rho, p)


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def _decision(rho: float, p: float) -> str:
    """Return the three-way decision label."""
    if math.isnan(rho) or math.isnan(p):
        return "INSUFFICIENT_EVIDENCE"
    abs_rho = abs(rho)
    if abs_rho > 0.5 and p < 0.05:
        return "D_PR_IS_VALID_SIGNAL"
    if abs_rho > 0.5 and p >= 0.05:
        return "INSUFFICIENT_EVIDENCE"
    return "D_PR_NOT_PREDICTIVE"


def _verdict_text(decision: str) -> str:
    if decision == "D_PR_IS_VALID_SIGNAL":
        return (
            "D_PR provides a valid diagnostic for forgetting risk. "
            "D_PR compression at task boundaries reliably predicts "
            "higher subsequent forgetting (|rho|>0.5, p<0.05)."
        )
    if decision == "INSUFFICIENT_EVIDENCE":
        return (
            "Insufficient evidence to confirm or deny D_PR as a forgetting "
            "predictor. More data needed — run --mode live to collect D_PR "
            "observations from fresh experiments."
        )
    return (
        "D_PR does not reliably predict forgetting. "
        "The correlation between D_PR compression and task forgetting "
        "is weak (|rho|<=0.5) across the observations collected."
    )


# ---------------------------------------------------------------------------
# Analysis mode helpers
# ---------------------------------------------------------------------------

def _iter_comparison_jsons(comparisons_dir: Path) -> list[Path]:
    """
    Return all non-env JSON files directly under comparisons_dir
    (non-recursive — subdirectories like _untrusted/ are excluded).
    """
    return sorted(
        p for p in comparisons_dir.glob("*.json")
        if not p.name.endswith("_env.json")
    )


def _load_seed_records_from_file(path: Path) -> list[dict]:
    """
    Extract (method, seed, dpr_per_task, forgetting_per_task) records from
    a comparison JSON that stores seed_results.

    The file must contain at least one seed entry with both 'dpr_per_task'
    (non-empty) and 'forgetting_per_task' (non-empty).

    Returns a list of dicts:
        {method, seed, dpr_per_task: list[float], forgetting_per_task: list[float]}
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []

    records: list[dict] = []

    # --- Format A: top-level seed_results list (comptest / run_generic output) ---
    method_name = data.get("method") or data.get("experiment_id", "unknown")
    seed_results = data.get("seed_results", [])
    for entry in seed_results:
        if not isinstance(entry, dict):
            continue
        dpr = entry.get("dpr_per_task", [])
        fgt = entry.get("forgetting_per_task", [])
        if not dpr or not fgt:
            continue
        records.append({
            "method":             str(method_name),
            "seed":               entry.get("seed", "?"),
            "dpr_per_task":       [float(v) for v in dpr],
            "forgetting_per_task": [float(v) for v in fgt],
            "source_file":        path.name,
        })

    # --- Format B: per-seed data nested under each method key (phase10 style) ---
    # These older files do not carry dpr_per_task, so we skip them.
    # (If a future rerun adds dpr_per_task to per_seed entries, add handling here.)

    return records


def _check_existing(comparisons_dir: Path) -> None:
    """Print a summary of how many comparison JSON files have dpr_per_task."""
    all_jsons = _iter_comparison_jsons(comparisons_dir)
    with_dpr: list[str] = []
    without_dpr: list[str] = []

    for path in all_jsons:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        found = False
        for entry in data.get("seed_results", []):
            if isinstance(entry, dict) and entry.get("dpr_per_task"):
                found = True
                break
        if found:
            with_dpr.append(path.name)
        else:
            without_dpr.append(path.name)

    total = len(all_jsons)
    print(f"\n=== D_PR field audit: {comparisons_dir} ===")
    print(f"Total comparison JSON files scanned : {total}")
    print(f"Files WITH dpr_per_task populated   : {len(with_dpr)}")
    print(f"Files WITHOUT dpr_per_task           : {len(without_dpr)}")

    if with_dpr:
        print("\nFiles that carry dpr_per_task:")
        for name in with_dpr:
            print(f"  [OK]  {name}")
    else:
        print(
            "\nNo files carry dpr_per_task yet.\n"
            "This is expected if Phase 2 reruns with compute_dpr=True have not\n"
            "yet completed.  To produce observations:\n"
            "\n"
            "  python analyze_dpr_forgetting_correlation.py --mode live\n"
            "\n"
            "(Requires GPU and execution_enabled.flag)"
        )


def run_analysis_mode(comparisons_dir: Path) -> dict:
    """
    Load all comparison JSONs, extract (dpr_drop, forgetting) pairs, and
    compute Spearman correlation.

    Returns the output JSON dict ready for serialisation.
    """
    all_jsons = _iter_comparison_jsons(comparisons_dir)
    all_records: list[dict] = []

    for path in all_jsons:
        all_records.extend(_load_seed_records_from_file(path))

    if not all_records:
        print(
            "\n[ANALYSIS MODE] No comparison JSON files contain 'dpr_per_task' data.\n"
            "\n"
            "This is expected if Phase 2 reruns (phase16_cifar100_rerun.py,\n"
            "phase17_tinyimagenet_rerun.py) have not yet been run with\n"
            "compute_dpr=True, or if the comptest / integration-test runs were\n"
            "executed without that flag.\n"
            "\n"
            "=== To collect D_PR data ===\n"
            "\n"
            "Option A — Run this script in LIVE mode (GPU required):\n"
            "\n"
            "    1. Ensure execution_enabled.flag exists:\n"
            f"       {_EXEC_FLAG}\n"
            "\n"
            "    2. Run:\n"
            "       python analyze_dpr_forgetting_correlation.py --mode live\n"
            "\n"
            "    This runs split_cifar10 × {tcl, ewc_generic, sgd_generic,\n"
            "    der_plus_plus} × 5 seeds with compute_dpr=True and produces\n"
            "    sufficient observations for the Spearman test.\n"
            "\n"
            "Option B — Re-run existing phase scripts with compute_dpr=True:\n"
            "\n"
            "    Set compute_dpr=True in phase16_cifar100_rerun.py and\n"
            "    phase17_tinyimagenet_rerun.py, then re-run.  The resulting\n"
            "    comparison JSONs will be picked up automatically by --mode\n"
            "    analysis on the next invocation.\n"
        )
        return {}

    # --- Build flat (dpr_drop, forgetting) pairs across all (method, seed, task) ---
    #
    # For a run with T tasks:
    #   dpr_per_task has T values  (D_PR after task 0, 1, ..., T-1 finishes)
    #   forgetting_per_task has T-1 values (forgetting of tasks 0..T-2)
    #
    # D_PR drop at task boundary t→t+1:
    #   dpr_drop[t] = dpr_per_task[t] - dpr_per_task[t+1]
    #
    # This is paired with forgetting_per_task[t] (forgetting of task t after
    # all subsequent tasks have been trained).
    #
    # A positive dpr_drop (compression) paired with high forgetting supports
    # the hypothesis (expected rho > 0).

    method_data: dict[str, list[tuple[float, float]]] = {}
    all_pairs_x: list[float] = []   # dpr_drop
    all_pairs_y: list[float] = []   # forgetting

    for rec in all_records:
        dpr = rec["dpr_per_task"]          # length T
        fgt = rec["forgetting_per_task"]   # length T-1
        method = rec["method"]

        if len(dpr) < 2 or len(fgt) < 1:
            continue

        n_boundary = min(len(dpr) - 1, len(fgt))
        for t in range(n_boundary):
            drop = dpr[t] - dpr[t + 1]
            forgetting_t = fgt[t]
            all_pairs_x.append(drop)
            all_pairs_y.append(forgetting_t)
            if method not in method_data:
                method_data[method] = []
            method_data[method].append((drop, forgetting_t))

    n_obs = len(all_pairs_x)
    print(f"\n[ANALYSIS MODE] Collected {n_obs} (dpr_drop, forgetting) observations "
          f"from {len(all_records)} (method, seed) traces.")

    if n_obs < 3:
        print(
            f"\nInsufficient observations (n={n_obs} < 3) for Spearman correlation.\n"
            "Run --mode live to collect more data."
        )
        return {}

    # --- Overall Spearman ---
    rho, p = _spearman(all_pairs_x, all_pairs_y)
    decision = _decision(rho, p)

    # --- Per-method Spearman ---
    per_method_rho: dict[str, dict] = {}
    for method, pairs in method_data.items():
        if len(pairs) < 3:
            per_method_rho[method] = {
                "rho": None,
                "p":   None,
                "n":   len(pairs),
                "note": "insufficient_observations",
            }
            continue
        mx = [p[0] for p in pairs]
        my = [p[1] for p in pairs]
        mr, mp = _spearman(mx, my)
        per_method_rho[method] = {
            "rho": round(mr, 4) if not math.isnan(mr) else None,
            "p":   round(mp, 4) if not math.isnan(mp) else None,
            "n":   len(pairs),
        }

    return {
        "analysis_type":    "dpr_forgetting_spearman",
        "conducted_at":     datetime.now(tz=timezone.utc).isoformat(),
        "mode":             "analysis",
        "n_observations":   n_obs,
        "spearman_rho":     round(rho, 4) if not math.isnan(rho) else None,
        "p_value":          round(p,   4) if not math.isnan(p)   else None,
        "n_method_seed_task_triples": n_obs,
        "decision":         decision,
        "per_method_rho":   per_method_rho,
        "theoretical_connection": (
            "TCL's elastic penalty resists D_PR compression by protecting "
            "parameters in the task-T principal subspace, preserving covariance "
            "structure of task-T activations. This is the mechanism linking the "
            "penalty to D_PR stability."
        ),
        "verdict": _verdict_text(decision),
    }


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def run_live_mode() -> dict:
    """
    Run fresh experiments with compute_dpr=True and collect D_PR observations.

    Requires:
      - execution_enabled.flag at _EXEC_FLAG
      - GPU (CUDA device) — falls back to CPU but warns
      - tar_lab.generic_cl_runner.run_generic_benchmark importable from _REPO
    """
    # Gate: execution_enabled.flag
    if not _EXEC_FLAG.exists():
        print(
            f"\nABORTED: execution_enabled.flag not found:\n  {_EXEC_FLAG}\n"
            "\nCreate this file to authorise live-mode execution:\n"
            f"    New-Item -ItemType File '{_EXEC_FLAG}'\n"
        )
        sys.exit(1)

    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cpu":
        print(
            "\nWARNING: No CUDA GPU detected. Running on CPU.\n"
            "Live mode with resnet18 × 5 seeds × 4 methods × 40 epochs will be\n"
            "very slow.  Interrupt (Ctrl-C) and run on a GPU machine, or reduce\n"
            "LIVE_EPOCHS / LIVE_SEEDS in this script.\n"
        )

    # Ensure _REPO is on sys.path so tar_lab imports work
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    from tar_lab.generic_cl_runner import run_generic_benchmark  # type: ignore

    all_records: list[dict] = []
    live_log_lines: list[str] = []

    def _log(msg: str) -> None:
        print(msg, flush=True)
        live_log_lines.append(msg)

    for method in LIVE_METHODS:
        config = _METHOD_CONFIGS.get(method, {})
        _log(f"\n[LIVE] method={method}  seeds={LIVE_SEEDS}  "
             f"dataset={LIVE_DATASET}  epochs={LIVE_EPOCHS}")
        try:
            seed_results, _, _ = run_generic_benchmark(
                dataset_name     = LIVE_DATASET,
                backbone_name    = LIVE_BACKBONE,
                method_name      = method,
                seeds            = LIVE_SEEDS,
                epochs           = LIVE_EPOCHS,
                config_overrides = config,
                data_root        = _DATA_ROOT,
                log_fn           = _log,
                compute_dpr      = True,
            )
        except Exception as exc:
            _log(f"[LIVE] ERROR running {method}: {exc}")
            continue

        for sr in seed_results:
            dpr = sr.get("dpr_per_task", [])
            fgt = sr.get("forgetting_per_task", [])
            if not dpr or not fgt:
                _log(f"[LIVE] WARNING: seed={sr.get('seed')} method={method} "
                     "missing dpr_per_task or forgetting_per_task — skipping")
                continue
            all_records.append({
                "method":              method,
                "seed":                sr["seed"],
                "dpr_per_task":        dpr,
                "forgetting_per_task": fgt,
                "source_file":         "live",
            })

    if not all_records:
        print("\n[LIVE MODE] No records collected.  All methods failed or "
              "returned empty dpr_per_task.  Check tar_lab installation.\n")
        return {}

    # Save raw live records alongside output for auditability
    live_raw_path = _STAT_AUDIT_DIR / "dpr_live_raw_records.json"
    _STAT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(live_raw_path, "w", encoding="utf-8") as fh:
        json.dump(all_records, fh, indent=2)
    print(f"\n[LIVE MODE] Raw records saved to: {live_raw_path}")

    # --- Build pairs ---
    method_data: dict[str, list[tuple[float, float]]] = {}
    all_pairs_x: list[float] = []
    all_pairs_y: list[float] = []

    for rec in all_records:
        dpr = rec["dpr_per_task"]
        fgt = rec["forgetting_per_task"]
        method = rec["method"]

        if len(dpr) < 2 or len(fgt) < 1:
            continue

        n_boundary = min(len(dpr) - 1, len(fgt))
        for t in range(n_boundary):
            drop = dpr[t] - dpr[t + 1]
            forgetting_t = fgt[t]
            all_pairs_x.append(drop)
            all_pairs_y.append(forgetting_t)
            if method not in method_data:
                method_data[method] = []
            method_data[method].append((drop, forgetting_t))

    n_obs = len(all_pairs_x)
    print(f"\n[LIVE MODE] Collected {n_obs} (dpr_drop, forgetting) observations.")

    if n_obs < 3:
        print(f"Insufficient observations (n={n_obs}).  All further methods failed.")
        return {}

    rho, p = _spearman(all_pairs_x, all_pairs_y)
    decision = _decision(rho, p)

    per_method_rho: dict[str, dict] = {}
    for method, pairs in method_data.items():
        if len(pairs) < 3:
            per_method_rho[method] = {
                "rho": None, "p": None, "n": len(pairs),
                "note": "insufficient_observations",
            }
            continue
        mx = [p[0] for p in pairs]
        my = [p[1] for p in pairs]
        mr, mp = _spearman(mx, my)
        per_method_rho[method] = {
            "rho": round(mr, 4) if not math.isnan(mr) else None,
            "p":   round(mp, 4) if not math.isnan(mp) else None,
            "n":   len(pairs),
        }

    return {
        "analysis_type":    "dpr_forgetting_spearman",
        "conducted_at":     datetime.now(tz=timezone.utc).isoformat(),
        "mode":             "live",
        "n_observations":   n_obs,
        "spearman_rho":     round(rho, 4) if not math.isnan(rho) else None,
        "p_value":          round(p,   4) if not math.isnan(p)   else None,
        "n_method_seed_task_triples": n_obs,
        "decision":         decision,
        "per_method_rho":   per_method_rho,
        "theoretical_connection": (
            "TCL's elastic penalty resists D_PR compression by protecting "
            "parameters in the task-T principal subspace, preserving covariance "
            "structure of task-T activations. This is the mechanism linking the "
            "penalty to D_PR stability."
        ),
        "verdict": _verdict_text(decision),
        "live_log": live_log_lines,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 Task 3.5 — D_PR–Forgetting Correlation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["analysis", "live"],
        default="analysis",
        help=(
            "analysis: read existing comparison JSONs (no GPU needed). "
            "live: run new experiments with compute_dpr=True (GPU + flag required)."
        ),
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help=(
            "Scan tar_state/comparisons/ and report how many files have "
            "dpr_per_task populated, then exit."
        ),
    )
    parser.add_argument(
        "--comparisons-dir",
        type=Path,
        default=_COMPARISONS_DIR,
        help=f"Override comparisons directory (default: {_COMPARISONS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_PATH,
        help=f"Output JSON path (default: {_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    # --- --check-existing ---
    if args.check_existing:
        _check_existing(args.comparisons_dir)
        return

    # --- Run the selected mode ---
    if args.mode == "analysis":
        result = run_analysis_mode(args.comparisons_dir)
    else:
        result = run_live_mode()

    if not result:
        # run_analysis_mode / run_live_mode already printed guidance
        sys.exit(0)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("D_PR–Forgetting Correlation Results")
    print("=" * 60)
    print(f"  Mode              : {result.get('mode', args.mode)}")
    print(f"  Observations      : {result['n_observations']}")
    rho_display = result.get("spearman_rho")
    p_display   = result.get("p_value")
    print(f"  Spearman rho      : {rho_display}")
    print(f"  p-value           : {p_display}")
    print(f"  Decision          : {result['decision']}")
    print(f"  Verdict           : {result['verdict']}")
    print()
    if result.get("per_method_rho"):
        print("  Per-method breakdown:")
        for method, stats in result["per_method_rho"].items():
            r = stats.get("rho", "N/A")
            pv = stats.get("p", "N/A")
            n = stats.get("n", 0)
            note = stats.get("note", "")
            note_str = f"  [{note}]" if note else ""
            print(f"    {method:<20}  rho={r}  p={pv}  n={n}{note_str}")
    print("=" * 60)

    # --- Write output JSON ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nOutput written to: {args.output}")

    # --- Post-decision guidance ---
    if result["decision"] == "INSUFFICIENT_EVIDENCE":
        print(
            "\nNext step: run live mode to collect fresh D_PR observations:\n"
            "  1. Create execution_enabled.flag:\n"
            f"     New-Item -ItemType File '{_EXEC_FLAG}'\n"
            "  2. Run:\n"
            "     python analyze_dpr_forgetting_correlation.py --mode live\n"
        )
    elif result["decision"] == "D_PR_IS_VALID_SIGNAL":
        print(
            "\nD_PR is a valid forgetting signal.  This supports the theoretical\n"
            "connection between TCL's elastic penalty and representation stability.\n"
            "Record this result in the honest evidence inventory (Phase 0.5).\n"
        )
    elif result["decision"] == "D_PR_NOT_PREDICTIVE":
        print(
            "\nD_PR is not reliably predictive of forgetting.  This is a negative\n"
            "theoretical result — record it honestly in the evidence inventory.\n"
            "It does NOT invalidate TCL's empirical forgetting advantage (Phase 10),\n"
            "but it does weaken the thermodynamic mechanistic framing.\n"
        )


if __name__ == "__main__":
    main()
