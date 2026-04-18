"""
Phase 8A validation run.

Runs TCL x5 seeds and SGD x5 seeds with the fixed integration.
After each TCL run, reads the trace and checks three things:
  1. Regime transitions at task boundaries (reset_for_new_task working?)
  2. dimensionality_ratio live from task 1 onward (anchor working?)
  3. "ordered" phase actually reached (alpha=0.5 calibration working?)

Prints a clear verdict on each check plus final benchmark numbers.
"""
import sys, json, math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace/Thermodynamic-Continual-Learning-delivered")
workspace = "/workspace/Thermodynamic-Continual-Learning-delivered"

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark

SEEDS = [42, 123, 456, 789, 1337]

def mean(v): return sum(v) / len(v) if v else float("nan")
def std(v):
    if len(v) < 2: return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

# ── run TCL ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Phase 8A Validation — {datetime.utcnow().isoformat()}")
print(f"{'='*60}")

tcl_results = []
for seed in SEEDS:
    print(f"\n[TCL] seed={seed}", flush=True)
    cfg = ContinualLearningBenchmarkConfig(seed=seed)
    result = run_split_cifar10_benchmark(cfg, method="tcl", workspace=workspace)
    tcl_results.append(result)
    print(f"  forgetting={result.mean_forgetting:.4f}  acc={result.final_mean_accuracy:.4f}",
          flush=True)

# ── analyse traces ────────────────────────────────────────────────────────────
trace_dir = Path(workspace) / "tar_state" / "cl_traces"

print(f"\n{'='*60}")
print("TRACE ANALYSIS")
print(f"{'='*60}")

check1_pass = []   # regime transitions at task boundaries
check2_pass = []   # dimensionality_ratio live from task 1
check3_pass = []   # "ordered" phase reached at least once

for seed in SEEDS:
    tf = trace_dir / f"tcl_{seed}.json"
    if not tf.exists():
        print(f"\n[seed={seed}] NO TRACE FILE")
        continue
    t = json.loads(tf.read_text())
    summaries = t.get("task_summaries", [])
    epoch_trace = t.get("epoch_trace", [])
    anchor_d = t.get("anchor_effective_dimensionality", 0.0)
    alpha = t.get("alpha", "?")
    final_regime = t.get("final_regime", "?")

    print(f"\n── seed={seed}  alpha={alpha}  anchor_d={anchor_d:.3f}  final={final_regime}")

    # CHECK 1: regime transitions at task boundaries
    # Look at dominant_regime in the first epoch of each task vs the last
    # epoch of the previous task. If reset is working, task 2 epoch 0 should
    # differ from task 1 epoch 4 (or at least cycle through transient states).
    boundary_transitions = []
    for task_idx in range(1, len(summaries)):
        prev = summaries[task_idx - 1]
        curr = summaries[task_idx]
        changed = prev["dominant_regime"] != curr["dominant_regime"]
        boundary_transitions.append({
            "task": task_idx,
            "prev_regime": prev["dominant_regime"],
            "curr_regime": curr["dominant_regime"],
            "changed": changed,
        })
    # also check first-epoch vs last-epoch within each task
    task_internal_transitions = []
    for task_idx in range(len(summaries)):
        task_epochs = [e for e in epoch_trace if e["task"] == task_idx]
        if len(task_epochs) >= 2:
            first = task_epochs[0]["dominant_regime"]
            last  = task_epochs[-1]["dominant_regime"]
            task_internal_transitions.append({
                "task": task_idx, "first_epoch": first, "last_epoch": last,
                "changed": first != last,
            })

    any_boundary_change = any(b["changed"] for b in boundary_transitions)
    any_internal_change = any(b["changed"] for b in task_internal_transitions)
    check1_pass.append(any_boundary_change or any_internal_change)

    print(f"  CHECK 1 — regime transitions:")
    for b in boundary_transitions:
        marker = "✓" if b["changed"] else "·"
        print(f"    {marker} task {b['task']-1}→{b['task']}: "
              f"{b['prev_regime']} → {b['curr_regime']}")
    for b in task_internal_transitions:
        marker = "✓" if b["changed"] else "·"
        print(f"    {marker} task {b['task']} internal: "
              f"{b['first_epoch']} → {b['last_epoch']}")
    print(f"  → {'PASS: transitions observed' if check1_pass[-1] else 'FAIL: stuck, no transitions'}")

    # CHECK 2: dimensionality_ratio live from task 1
    # anchor_effective_dimensionality should be > 0 (set after task 0)
    # and mean_lr should vary across tasks (non-constant governor output)
    anchor_live = anchor_d > 0.0
    lrs = [s["mean_lr"] for s in summaries]
    lr_varies = max(lrs) - min(lrs) > 1e-5 if lrs else False
    check2_pass.append(anchor_live)
    print(f"  CHECK 2 — dimensionality_ratio live:")
    print(f"    anchor_effective_dimensionality = {anchor_d:.4f} "
          f"{'(LIVE ✓)' if anchor_live else '(DEAD — still 0.0 ✗)'}")
    print(f"    mean LR per task: {[round(lr,6) for lr in lrs]}")
    print(f"    LR varies across tasks: {'yes ✓' if lr_varies else 'no (governor constant)'}")
    print(f"  → {'PASS' if check2_pass[-1] else 'FAIL: anchor not set'}")

    # CHECK 3: "ordered" phase reached at any point
    all_regimes_seen = set()
    for entry in epoch_trace:
        all_regimes_seen.update(entry["regime_pct"].keys())
    ordered_reached = "ordered" in all_regimes_seen
    check3_pass.append(ordered_reached)
    print(f"  CHECK 3 — 'ordered' phase reached:")
    print(f"    regimes seen across all epochs: {sorted(all_regimes_seen)}")
    # show per-task dominant regime
    for s in summaries:
        pct_str = "  ".join(f"{r}={v:.0%}" for r, v in s["regime_pct"].items())
        print(f"    task {s['task']}: dominant={s['dominant_regime']}  [{pct_str}]")
    print(f"  → {'PASS: ordered reached ✓' if ordered_reached else 'FAIL: never ordered — sigma_star still miscalibrated'}")

# ── run SGD baseline ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SGD BASELINE (comparison)")
print(f"{'='*60}")
sgd_results = []
for seed in SEEDS:
    print(f"[SGD] seed={seed}", flush=True)
    cfg = ContinualLearningBenchmarkConfig(seed=seed)
    result = run_split_cifar10_benchmark(cfg, method="sgd_baseline", workspace=workspace)
    sgd_results.append(result)
    print(f"  forgetting={result.mean_forgetting:.4f}  acc={result.final_mean_accuracy:.4f}",
          flush=True)

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")

tcl_forg  = [r.mean_forgetting        for r in tcl_results]
tcl_acc   = [r.final_mean_accuracy    for r in tcl_results]
sgd_forg  = [r.mean_forgetting        for r in sgd_results]
sgd_acc   = [r.final_mean_accuracy    for r in sgd_results]

print(f"\nBenchmark numbers (5 seeds):")
print(f"  TCL  forgetting={mean(tcl_forg):.4f}±{std(tcl_forg):.4f}  "
      f"acc={mean(tcl_acc):.4f}")
print(f"  SGD  forgetting={mean(sgd_forg):.4f}±{std(sgd_forg):.4f}  "
      f"acc={mean(sgd_acc):.4f}")
delta_forg = mean(tcl_forg) - mean(sgd_forg)
print(f"\n  TCL vs SGD forgetting delta: {delta_forg:+.4f} "
      f"({'TCL better' if delta_forg < 0 else 'TCL worse' if delta_forg > 0 else 'equal'})")

print(f"\nCheck results across 5 seeds:")
print(f"  CHECK 1 (regime transitions): "
      f"{sum(check1_pass)}/5 seeds show transitions")
print(f"  CHECK 2 (anchor live):        "
      f"{sum(check2_pass)}/5 seeds have anchor > 0")
print(f"  CHECK 3 (ordered reached):    "
      f"{sum(check3_pass)}/5 seeds reach 'ordered' phase")

if all(check3_pass) and delta_forg < -0.01:
    verdict = "GOVERNOR WORKING AND HELPING — ordered phase reached, TCL beats SGD"
elif all(check3_pass) and abs(delta_forg) <= 0.01:
    verdict = "GOVERNOR WORKING BUT NOT SUFFICIENT — ordered reached, no separation from SGD → epoch sweep next"
elif not any(check3_pass):
    verdict = "GOVERNOR STILL STUCK — ordered never reached → check alpha further or window size"
else:
    verdict = "PARTIAL — some seeds reach ordered, results mixed → investigate per-seed traces"

print(f"\nVERDICT: {verdict}")
print(f"\n[{datetime.utcnow().isoformat()}] Validation complete")
