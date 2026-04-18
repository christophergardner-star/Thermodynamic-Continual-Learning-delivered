"""
Phase 8B scouting pass — single seed, epochs_per_task in {5, 15, 40}.

Distinguishes three hypotheses:
  A) Volume        — ordered% shifts smoothly with epoch count and appears
                     in the convergence window (final half of each task).
  B) Amortised     — ordered% grows with global step count but always in
                     early epochs of tasks, not convergence window.
  C) Calibration   — rho stays stuck well above 0.9 regardless of epochs.
                     sigma/sigma_star shows regime thresholds need revisiting.

Key measurement: per-epoch-within-task regime and raw sigma/sigma_star.
Trace file is read immediately after each run (file is overwritten per run).
"""
import sys, json, math
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, "/workspace/Thermodynamic-Continual-Learning-delivered")
workspace = "/workspace/Thermodynamic-Continual-Learning-delivered"

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark

SEED         = 42
EPOCH_COUNTS = [5, 15, 40]
TRACE_FILE   = Path(workspace) / "tar_state" / "cl_traces" / f"tcl_{SEED}.json"

def mean(v): return sum(v) / len(v) if v else float("nan")


def read_and_analyse_trace(epochs: int) -> dict:
    """Read trace immediately after a run, return structured analysis."""
    if not TRACE_FILE.exists():
        return {}
    t = json.loads(TRACE_FILE.read_text())
    et = t.get("epoch_trace", [])
    anchor_d = t.get("anchor_effective_dimensionality", 0.0)
    alpha    = t.get("alpha", "?")

    n_tasks = max((e["task"] for e in et), default=0) + 1
    task_data = []

    for task_idx in range(n_tasks):
        task_epochs = [e for e in et if e["task"] == task_idx]
        if not task_epochs:
            continue
        n_ep    = len(task_epochs)
        split   = max(n_ep // 2, 1)
        early   = task_epochs[:split]
        conv    = task_epochs[split:]

        # where within the task does ordered first appear?
        first_ordered_ep = next(
            (e["epoch"] for e in task_epochs
             if e["regime_pct"].get("ordered", 0) > 0.01),
            None
        )
        # is first ordered in convergence window (second half of task)?
        in_conv_window = (
            first_ordered_ep is not None and
            first_ordered_ep >= task_epochs[split]["epoch"]
            if conv else False
        )

        task_data.append({
            "task": task_idx,
            "n_epochs_run": n_ep,
            "early_ordered_pct":  round(mean([e["regime_pct"].get("ordered", 0) for e in early]), 4),
            "conv_ordered_pct":   round(mean([e["regime_pct"].get("ordered", 0) for e in conv]),  4),
            "overall_ordered_pct":round(mean([e["regime_pct"].get("ordered", 0) for e in task_epochs]), 4),
            "first_ordered_epoch": first_ordered_ep,
            "first_ordered_in_conv_window": in_conv_window,
            # rho trajectory across task epochs (mean_rho per epoch)
            "rho_per_epoch": [round(e["mean_rho"], 4) for e in task_epochs if "mean_rho" in e],
            # rho_to_ordered: distance from 0.9 threshold (negative = crossed)
            "rho_to_ordered_final": round(task_epochs[-1].get("rho_to_ordered", float("nan")), 4)
                if task_epochs and "rho_to_ordered" in task_epochs[-1] else None,
            # sigma/sigma_star at final epoch (calibration diagnostic)
            "final_sigma":      round(task_epochs[-1].get("mean_sigma", float("nan")), 8)
                if task_epochs and "mean_sigma" in task_epochs[-1] else None,
            "final_sigma_star": round(task_epochs[-1].get("mean_sigma_star", float("nan")), 8)
                if task_epochs and "mean_sigma_star" in task_epochs[-1] else None,
        })

    # global ordered fraction
    all_ordered = mean([e["regime_pct"].get("ordered", 0) for e in et])
    all_disord  = mean([e["regime_pct"].get("disordered", 0) for e in et])

    return {
        "epochs_per_task": epochs,
        "alpha": alpha,
        "anchor_d": anchor_d,
        "overall_ordered_pct": round(all_ordered, 4),
        "overall_disordered_pct": round(all_disord, 4),
        "task_data": task_data,
    }


# ── main loop: run each epoch count, stash trace immediately ─────────────────
run_results  = {}   # epochs -> {tcl_forgetting, sgd_forgetting}
trace_analyses = {}  # epochs -> analyse dict

for epochs in EPOCH_COUNTS:
    print(f"\n{'='*60}")
    print(f"epochs_per_task={epochs}  seed={SEED}  {datetime.utcnow().isoformat()}")
    print(f"{'='*60}", flush=True)

    cfg = ContinualLearningBenchmarkConfig(seed=SEED, train_epochs_per_task=epochs)

    print("[TCL]", flush=True)
    tcl = run_split_cifar10_benchmark(cfg, method="tcl", workspace=workspace)
    print(f"  forgetting={tcl.mean_forgetting:.4f}  acc={tcl.final_mean_accuracy:.4f}",
          flush=True)

    # READ TRACE NOW before next run overwrites it
    analysis = read_and_analyse_trace(epochs)
    trace_analyses[epochs] = analysis

    print("[SGD]", flush=True)
    sgd = run_split_cifar10_benchmark(cfg, method="sgd_baseline", workspace=workspace)
    print(f"  forgetting={sgd.mean_forgetting:.4f}  acc={sgd.final_mean_accuracy:.4f}",
          flush=True)

    run_results[epochs] = {
        "tcl_forgetting": tcl.mean_forgetting,
        "tcl_accuracy":   tcl.final_mean_accuracy,
        "sgd_forgetting": sgd.mean_forgetting,
        "sgd_accuracy":   sgd.final_mean_accuracy,
        "delta":          tcl.mean_forgetting - sgd.mean_forgetting,
    }


# ── regime diagnostic print ───────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("REGIME DIAGNOSTIC — PER EPOCH COUNT")
print(f"{'='*60}")

for epochs in EPOCH_COUNTS:
    a = trace_analyses.get(epochs, {})
    if not a:
        print(f"\nepochs={epochs}: no trace")
        continue
    print(f"\n── epochs_per_task={epochs}  "
          f"ordered={a['overall_ordered_pct']:.1%}  "
          f"disordered={a['overall_disordered_pct']:.1%}  "
          f"anchor_d={a['anchor_d']:.3f}")
    for td in a["task_data"]:
        conv_arrow = ("↑" if td["conv_ordered_pct"] > td["early_ordered_pct"] + 0.01
                      else "·")
        in_conv = "conv✓" if td["first_ordered_in_conv_window"] else (
                  "early" if td["first_ordered_epoch"] is not None else "none")
        rho_final = f"rho_to_ord={td['rho_to_ordered_final']:.3f}" \
            if td["rho_to_ordered_final"] is not None else ""
        sig_ratio = ""
        if td["final_sigma"] and td["final_sigma_star"]:
            ratio = td["final_sigma"] / max(td["final_sigma_star"], 1e-12)
            sig_ratio = f"  σ/σ*={ratio:.3f}"
        print(f"   task {td['task']:d}: "
              f"early={td['early_ordered_pct']:.1%}  "
              f"conv={td['conv_ordered_pct']:.1%} {conv_arrow}  "
              f"first_ordered={td['first_ordered_epoch']} ({in_conv})  "
              f"{rho_final}{sig_ratio}")
    # rho trajectory for task 0 (shows whether rho drops as task converges)
    if a["task_data"]:
        t0 = a["task_data"][0]
        print(f"   task 0 rho trajectory: {t0['rho_per_epoch']}")


# ── forgetting table ──────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("FORGETTING vs EPOCH COUNT")
print(f"{'='*60}")
print(f"{'epochs':>8}  {'TCL':>8}  {'SGD':>8}  {'delta':>8}  ordered%  verdict")
for epochs in EPOCH_COUNTS:
    if epochs not in run_results:
        continue
    r   = run_results[epochs]
    op  = trace_analyses.get(epochs, {}).get("overall_ordered_pct", float("nan"))
    mrk = ("TCL>" if r["delta"] < -0.005
           else "SGD>" if r["delta"] > 0.005 else "~=")
    print(f"{epochs:>8}  {r['tcl_forgetting']:>8.4f}  {r['sgd_forgetting']:>8.4f}  "
          f"{r['delta']:>+8.4f}  {op:.1%}     {mrk}")


# ── hypothesis verdict ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("HYPOTHESIS VERDICT")
print(f"{'='*60}")

ord_pcts = [trace_analyses[e]["overall_ordered_pct"]
            for e in EPOCH_COUNTS if e in trace_analyses]
ordered_grows      = (len(ord_pcts) >= 2 and ord_pcts[-1] > ord_pcts[0] + 0.05)
ordered_grows_any  = (len(ord_pcts) >= 2 and ord_pcts[-1] > ord_pcts[0] + 0.01)

# any task shows ordered in convergence window?
conv_signal = any(
    td["first_ordered_in_conv_window"]
    for e in EPOCH_COUNTS if e in trace_analyses
    for td in trace_analyses[e]["task_data"]
)

# calibration check: rho_to_ordered stuck > 0.15 at end of task 0 even at 40ep
rho_stuck_at_40 = False
if 40 in trace_analyses and trace_analyses[40]["task_data"]:
    t0_final_rho = trace_analyses[40]["task_data"][0].get("rho_to_ordered_final")
    if t0_final_rho is not None and t0_final_rho > 0.15:
        rho_stuck_at_40 = True

delta_40 = run_results.get(40, {}).get("delta", float("nan"))
delta_15 = run_results.get(15, {}).get("delta", float("nan"))

if rho_stuck_at_40 and not ordered_grows_any:
    verdict = ("CALIBRATION — rho stays >0.15 above threshold even at 40 epochs. "
               "Volume alone won't work. Recommend alpha=0.3 or lower rho thresholds.")
elif ordered_grows and conv_signal and delta_40 < -0.005:
    verdict = ("VOLUME CONFIRMED — ordered% grows, appears in convergence window, "
               "TCL beats SGD at 40 epochs. Run full 5-seed pass at winning epoch count.")
elif ordered_grows and not conv_signal:
    verdict = ("AMORTISED — ordered% grows with epochs but does NOT appear in convergence "
               "window. Governor responds to global training volume, not per-task convergence. "
               "Theory needs rework.")
elif ordered_grows and conv_signal and not math.isnan(delta_40) and delta_40 >= -0.005:
    target_ep = 15 if delta_15 < -0.005 else 40
    verdict = (f"VOLUME PARTIAL — ordered% grows, convergence window signal present, "
               f"but no forgetting separation yet. "
               f"Recommend multiplier tuning (try 0.5x→0.3x brake) at {target_ep} epochs.")
elif ordered_grows_any:
    verdict = ("WEAK VOLUME — marginal ordered% growth. Inspect rho trajectories above "
               "to determine if threshold adjustment is needed alongside more epochs.")
else:
    verdict = "INCONCLUSIVE — inspect per-task tables and rho trajectories above."

print(f"\n{verdict}")

# ── recommendation for next step ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("RECOMMENDED NEXT STEP")
print(f"{'='*60}")
if "VOLUME CONFIRMED" in verdict:
    winning_ep = next((e for e in [15, 40] if run_results.get(e, {}).get("delta", 1) < -0.005), 40)
    print(f"\nFull 5-seed run at epochs_per_task={winning_ep}:")
    print(f"  Methods: TCL + SGD (5 seeds each) — primary comparison")
    print(f"  Spot-check: EWC + SI at 2 seeds to confirm they don't degrade")
    print(f"  Estimated pod time: ~{winning_ep * 5 * 2 * 3.5 / 60:.1f}h (TCL+SGD x5) "
          f"+ ~{winning_ep * 2 * 2 * 3.5 / 60:.1f}h (EWC+SI x2 spot)")
elif "CALIBRATION" in verdict:
    print(f"\nAlpha sweep on single seed at epochs_per_task=15:")
    print(f"  Try alpha in {{0.1, 0.2, 0.3}} — lower alpha makes sigma_star smaller,")
    print(f"  rho drops below 0.9 earlier in training.")
    print(f"  Read rho_to_ordered_final for task 0 to find alpha where threshold is crossed.")
elif "MULTIPLIER" in verdict or "PARTIAL" in verdict:
    print(f"\nMultiplier sweep: fix epochs_per_task=15 or 40, vary brake multiplier:")
    print(f"  Try ordered_brake in {{0.5, 0.3, 0.2}} — stronger consolidation signal.")
    print(f"  Run TCL+SGD x1 seed each to see if forgetting delta inverts.")
else:
    print(f"\nInspect the per-task rho trajectories above and decide.")

# ── write result ──────────────────────────────────────────────────────────────
out = Path(workspace) / "tar_state" / "comparisons" / "phase8b_scout.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "seed": SEED,
    "epoch_counts": EPOCH_COUNTS,
    "run_results": run_results,
    "trace_analyses": {str(k): v for k, v in trace_analyses.items()},
    "verdict": verdict,
    "completed_at": datetime.utcnow().isoformat(),
}, indent=2, default=str))
print(f"\nResult written: {out}")
print(f"[{datetime.utcnow().isoformat()}] Scouting pass complete")
