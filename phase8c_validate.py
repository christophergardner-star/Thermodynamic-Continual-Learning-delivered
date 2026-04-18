"""
Phase 8C validation — fixed sigma_star anchor.

Checks three things that prove the fix is working:
  1. ANCHOR FROZEN — sigma_star_anchor is constant across epochs within a task
     (not co-contracting with sigma).  Measured via std(mean_sigma_star per epoch).
  2. RHO DESCENDS — mean_rho shows downward trend within at least 2 tasks.
     Previously rho was flat or increasing; any systematic decrease is the signal.
  3. ORDERED IN CONVERGENCE WINDOW — ordered% > 0 in the second half of at least
     1 task.  Previously ordered was < 3% and randomly scattered.

Run settings: seed=42, epochs_per_task=15 (the epoch count where the 8B trace
showed sigma collapsing for tasks 3/4 — those tasks should now clearly reach
ordered with the fixed anchor).

After confirming the fix works, run TCL+SGD x5 seeds at 15 epochs for the
benchmark comparison.
"""
import sys, json, math
from pathlib import Path
from datetime import datetime
from statistics import median, stdev

sys.path.insert(0, "/workspace/Thermodynamic-Continual-Learning-delivered")
workspace = "/workspace/Thermodynamic-Continual-Learning-delivered"

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark

SEED          = 42
EPOCHS        = 15
TRACE_FILE    = Path(workspace) / "tar_state" / "cl_traces" / f"tcl_{SEED}.json"


def mean(v): return sum(v) / len(v) if v else float("nan")


def analyse_trace() -> dict:
    if not TRACE_FILE.exists():
        return {}
    t  = json.loads(TRACE_FILE.read_text())
    et = t.get("epoch_trace", [])
    n_tasks = max((e["task"] for e in et), default=0) + 1

    task_data = []
    for task_idx in range(n_tasks):
        task_ep = [e for e in et if e["task"] == task_idx]
        if not task_ep:
            continue
        n_ep   = len(task_ep)
        split  = max(n_ep // 2, 1)
        early  = task_ep[:split]
        conv   = task_ep[split:]

        rhos        = [e.get("mean_rho", float("nan")) for e in task_ep]
        sigmas      = [e.get("mean_sigma", float("nan")) for e in task_ep]
        sigma_stars = [e.get("mean_sigma_star", float("nan")) for e in task_ep]
        ordered     = [e["regime_pct"].get("ordered", 0) for e in task_ep]

        # CHECK 1: anchor frozen?  sigma_star should be ~constant within task.
        valid_ss = [s for s in sigma_stars if not math.isnan(s) and s > 0]
        ss_std   = stdev(valid_ss) if len(valid_ss) > 1 else float("nan")
        ss_mean  = mean(valid_ss) if valid_ss else float("nan")
        ss_cv    = ss_std / max(ss_mean, 1e-12)  # coefficient of variation

        # CHECK 2: rho descending?  Compare last-third mean vs first-third mean.
        third = max(n_ep // 3, 1)
        early_rho_mean = mean([r for r in rhos[:third] if not math.isnan(r)])
        late_rho_mean  = mean([r for r in rhos[-third:] if not math.isnan(r)])
        rho_descends   = (not math.isnan(early_rho_mean) and
                          not math.isnan(late_rho_mean) and
                          late_rho_mean < early_rho_mean - 0.1)

        # CHECK 3: ordered in convergence window?
        conv_ordered = mean([e["regime_pct"].get("ordered", 0) for e in conv])
        early_ordered = mean([e["regime_pct"].get("ordered", 0) for e in early])
        ordered_in_conv = conv_ordered > 0.02  # > 2% in convergence half

        task_data.append({
            "task":            task_idx,
            "n_epochs":        n_ep,
            "rho_per_epoch":   [round(r, 4) for r in rhos],
            "sigma_per_epoch": [round(s, 8) for s in sigmas if not math.isnan(s)],
            "sigma_star_per_epoch": [round(s, 8) for s in sigma_stars if not math.isnan(s)],
            "sigma_star_cv":   round(ss_cv, 4),   # near 0 = frozen, large = co-tracking
            "rho_early_mean":  round(early_rho_mean, 4),
            "rho_late_mean":   round(late_rho_mean, 4),
            "rho_descends":    rho_descends,
            "early_ordered":   round(early_ordered, 4),
            "conv_ordered":    round(conv_ordered, 4),
            "ordered_in_conv_window": ordered_in_conv,
        })

    return {
        "alpha":     t.get("alpha"),
        "anchor_d":  t.get("anchor_effective_dimensionality", 0.0),
        "task_data": task_data,
    }


# ── run ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Phase 8C Validation — fixed sigma_star anchor")
print(f"seed={SEED}  epochs_per_task={EPOCHS}  {datetime.utcnow().isoformat()}")
print(f"{'='*60}", flush=True)

cfg = ContinualLearningBenchmarkConfig(seed=SEED, train_epochs_per_task=EPOCHS)

print("[TCL]", flush=True)
tcl = run_split_cifar10_benchmark(cfg, method="tcl", workspace=workspace)
print(f"  forgetting={tcl.mean_forgetting:.4f}  acc={tcl.final_mean_accuracy:.4f}",
      flush=True)

analysis = analyse_trace()

print("[SGD]", flush=True)
sgd = run_split_cifar10_benchmark(cfg, method="sgd_baseline", workspace=workspace)
print(f"  forgetting={sgd.mean_forgetting:.4f}  acc={sgd.final_mean_accuracy:.4f}",
      flush=True)


# ── check 1: anchor frozen ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CHECK 1: sigma_star_anchor FROZEN within each task")
print(f"{'='*60}")
print("(sigma_star CV ≈ 0.0 means frozen; > 0.1 means still co-tracking)\n")

check1_pass = []
for td in analysis.get("task_data", []):
    cv   = td["sigma_star_cv"]
    ok   = cv < 0.05
    check1_pass.append(ok)
    mark = "✓" if ok else "✗"
    print(f"  {mark} task {td['task']}: sigma_star CV={cv:.4f}  "
          f"sigma_star/epoch = {td['sigma_star_per_epoch']}")

n_frozen = sum(check1_pass)
print(f"\n  → {n_frozen}/{len(check1_pass)} tasks have frozen anchor  "
      f"({'PASS' if n_frozen >= 3 else 'FAIL'})")


# ── check 2: rho descends ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CHECK 2: rho DESCENDS within tasks (early rho > late rho)")
print(f"{'='*60}")
print("(Previously rho was flat or increasing — any drop is signal)\n")

check2_pass = []
for td in analysis.get("task_data", []):
    ok   = td["rho_descends"]
    check2_pass.append(ok)
    mark = "✓" if ok else "·"
    delta = td["rho_late_mean"] - td["rho_early_mean"]
    print(f"  {mark} task {td['task']}: "
          f"early_rho={td['rho_early_mean']:.3f}  late_rho={td['rho_late_mean']:.3f}  "
          f"delta={delta:+.3f}  {'↓ descends' if ok else '→ flat/rising'}")
    print(f"       rho trajectory: {td['rho_per_epoch']}")

n_desc = sum(check2_pass)
print(f"\n  → {n_desc}/{len(check2_pass)} tasks show rho descending  "
      f"({'PASS' if n_desc >= 2 else 'FAIL — fix not working as expected'})")


# ── check 3: ordered in convergence window ────────────────────────────────────
print(f"\n{'='*60}")
print("CHECK 3: ordered% in CONVERGENCE WINDOW (final half of task epochs)")
print(f"{'='*60}")
print("(Previously ordered was < 3% and random — need > 2% in conv window)\n")

check3_pass = []
for td in analysis.get("task_data", []):
    ok   = td["ordered_in_conv_window"]
    check3_pass.append(ok)
    mark = "✓" if ok else "·"
    print(f"  {mark} task {td['task']}: "
          f"early_ordered={td['early_ordered']:.1%}  "
          f"conv_ordered={td['conv_ordered']:.1%}  "
          f"{'IN CONV WINDOW' if ok else 'not in conv window'}")

n_conv = sum(check3_pass)
print(f"\n  → {n_conv}/{len(check3_pass)} tasks show ordered in convergence window  "
      f"({'PASS' if n_conv >= 1 else 'PARTIAL — may need more epochs or lower alpha'})")


# ── benchmark comparison ──────────────────────────────────────────────────────
delta = tcl.mean_forgetting - sgd.mean_forgetting
print(f"\n{'='*60}")
print("BENCHMARK (single seed, indicative only)")
print(f"{'='*60}")
print(f"  TCL  forgetting={tcl.mean_forgetting:.4f}  acc={tcl.final_mean_accuracy:.4f}")
print(f"  SGD  forgetting={sgd.mean_forgetting:.4f}  acc={sgd.final_mean_accuracy:.4f}")
print(f"  delta={delta:+.4f}  ({'TCL better' if delta < 0 else 'SGD better' if delta > 0 else 'equal'})")


# ── overall verdict ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")

if n_frozen >= 3 and n_desc >= 2 and n_conv >= 1:
    verdict = ("ANCHOR FIX WORKING — sigma_star frozen, rho descends, ordered fires "
               "in convergence window.  Proceed to full 5-seed benchmark at 15 epochs.")
elif n_frozen >= 3 and n_desc >= 1 and n_conv == 0:
    verdict = ("ANCHOR FROZEN, RHO MOVING but ordered not yet in convergence window.  "
               "Try alpha=0.3 or epochs_per_task=20 to reach threshold.")
elif n_frozen < 3:
    verdict = ("ANCHOR NOT FROZEN — sigma_star still co-tracking.  "
               "Check reset_for_new_task() and sigma_star_anchor logic.")
else:
    verdict = ("PARTIAL — inspect per-task rho trajectories above.")

print(f"\n{verdict}")


# ── write result ──────────────────────────────────────────────────────────────
out = Path(workspace) / "tar_state" / "comparisons" / "phase8c_validate.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "seed": SEED,
    "epochs_per_task": EPOCHS,
    "tcl_forgetting": tcl.mean_forgetting,
    "sgd_forgetting": sgd.mean_forgetting,
    "delta": delta,
    "trace_analysis": analysis,
    "checks": {
        "anchor_frozen": {"n_pass": n_frozen, "total": len(check1_pass)},
        "rho_descends":  {"n_pass": n_desc,   "total": len(check2_pass)},
        "ordered_in_conv_window": {"n_pass": n_conv, "total": len(check3_pass)},
    },
    "verdict": verdict,
    "completed_at": datetime.utcnow().isoformat(),
}, indent=2, default=str))
print(f"\nResult written: {out}")
print(f"[{datetime.utcnow().isoformat()}] Validation complete")
