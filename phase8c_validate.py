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
        # Separated into two sub-questions per analysis:
        #   3a. Does rho reach a minimum below the ordered threshold (0.9)?
        #       This is whether the anchor calibration allows ordered at all.
        #   3b. Does ordered% appear in the convergence window (final half)?
        #       This is whether ordered fires at the right time in the task.
        conv_ordered  = mean([e["regime_pct"].get("ordered", 0) for e in conv])
        early_ordered = mean([e["regime_pct"].get("ordered", 0) for e in early])
        ordered_in_conv = conv_ordered > 0.02  # > 2% in convergence half

        valid_rhos = [(i, r) for i, r in enumerate(rhos) if not math.isnan(r)]
        if valid_rhos:
            rho_min_val = min(r for _, r in valid_rhos)
            rho_min_ep  = valid_rhos[min(range(len(valid_rhos)), key=lambda k: valid_rhos[k][1])][0]
        else:
            rho_min_val, rho_min_ep = float("nan"), None
        # sigma drop ratio: final sigma vs anchor (sigma_star_anchor / alpha gives the anchor sigma)
        # With fixed anchor, sigma_star ≈ constant = sigma_star_anchor_value
        anchor_sigma_est = valid_ss[0] / max(t.get("alpha", 0.5), 1e-12) if valid_ss else float("nan")
        final_sigma = sigmas[-1] if sigmas else float("nan")
        sigma_drop_ratio = (final_sigma / max(anchor_sigma_est, 1e-12)
                            if not (math.isnan(final_sigma) or math.isnan(anchor_sigma_est)) else float("nan"))

        task_data.append({
            "task":            task_idx,
            "n_epochs":        n_ep,
            "rho_per_epoch":   [round(r, 4) for r in rhos],
            "sigma_per_epoch": [round(s, 8) for s in sigmas if not math.isnan(s)],
            "sigma_star_per_epoch": [round(s, 8) for s in sigma_stars if not math.isnan(s)],
            "sigma_star_cv":   round(ss_cv, 4),     # near 0 = frozen, large = co-tracking
            "rho_early_mean":  round(early_rho_mean, 4),
            "rho_late_mean":   round(late_rho_mean, 4),
            "rho_descends":    rho_descends,
            # CHECK 3a: does rho reach ordered threshold?
            "rho_min":         round(rho_min_val, 4),
            "rho_min_epoch":   rho_min_ep,
            "rho_min_below_ordered": rho_min_val < 0.9 if not math.isnan(rho_min_val) else False,
            "sigma_drop_ratio": round(sigma_drop_ratio, 4),  # < 0.5 means sigma halved vs anchor
            # CHECK 3b: does ordered fire at the right time?
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


# ── check 3: ordered in convergence window (two sub-questions) ───────────────
print(f"\n{'='*60}")
print("CHECK 3a: rho reaches ordered threshold (rho_min < 0.9)")
print("CHECK 3b: ordered% in CONVERGENCE WINDOW (final half of task epochs)")
print(f"{'='*60}")
print("(3a = anchor calibration allows ordered at all)")
print("(3b = ordered fires at the right time, not just early)\n")

check3a_pass = []
check3_pass  = []
for td in analysis.get("task_data", []):
    ok3a = td.get("rho_min_below_ordered", False)
    ok3b = td["ordered_in_conv_window"]
    check3a_pass.append(ok3a)
    check3_pass.append(ok3b)
    mark3a = "✓" if ok3a else "·"
    mark3b = "✓" if ok3b else "·"
    drop = td.get("sigma_drop_ratio", float("nan"))
    drop_str = f"σ_drop={drop:.2f}x" if not math.isnan(drop) else ""
    print(f"  {mark3a} task {td['task']} (3a): "
          f"rho_min={td.get('rho_min', float('nan')):.3f} at epoch {td.get('rho_min_epoch')}  "
          f"{drop_str}")
    print(f"  {mark3b} task {td['task']} (3b): "
          f"early_ordered={td['early_ordered']:.1%}  conv_ordered={td['conv_ordered']:.1%}  "
          f"{'IN CONV WINDOW' if ok3b else 'not in conv window'}")

n_3a   = sum(check3a_pass)
n_conv = sum(check3_pass)
print(f"\n  → 3a: {n_3a}/{len(check3a_pass)} tasks reach rho < 0.9  "
      f"({'threshold reachable' if n_3a >= 1 else 'THRESHOLD UNREACHABLE — check alpha or anchor_n'})")
print(f"  → 3b: {n_conv}/{len(check3_pass)} tasks show ordered in conv window  "
      f"({'timing correct' if n_conv >= 1 else 'timing wrong — check window definition or increase epochs'})")


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

if n_frozen >= 3 and n_desc >= 2 and n_3a >= 1 and n_conv >= 1:
    verdict = ("ANCHOR FIX WORKING — sigma_star frozen, rho descends, threshold reachable, "
               "ordered fires in convergence window.  Proceed to full 5-seed benchmark at 15 epochs.")
elif n_frozen >= 3 and n_desc >= 2 and n_3a >= 1 and n_conv == 0:
    verdict = ("THRESHOLD REACHABLE but ordered timing wrong — ordered fires in early epochs only.  "
               "Check whether 'convergence window' definition (final half) is too conservative; "
               "report rho_min_epoch to see when sigma actually collapses.")
elif n_frozen >= 3 and n_desc >= 2 and n_3a == 0:
    verdict = ("RHO DESCENDS but never crosses threshold — alpha=0.5 too conservative.  "
               "Try alpha=0.3 (ordered when sigma drops to 30% of anchor) or increase epochs.")
elif n_frozen >= 3 and n_desc < 2:
    verdict = ("ANCHOR FROZEN but rho not descending in most tasks — sigma not falling below anchor.  "
               "Check sigma trajectory: if sigma rises after anchor window, possible easy-batch "
               "anchor (check DataLoader shuffle) or task gradient dynamics mismatch.")
elif n_frozen < 3:
    verdict = ("ANCHOR NOT FROZEN — sigma_star still co-tracking.  "
               "Check reset_for_new_task() and sigma_star_anchor logic.")
else:
    verdict = ("PARTIAL — inspect per-task rho trajectories and 3a/3b sub-checks above.")

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
        "anchor_frozen":           {"n_pass": n_frozen, "total": len(check1_pass)},
        "rho_descends":            {"n_pass": n_desc,   "total": len(check2_pass)},
        "rho_min_below_threshold": {"n_pass": n_3a,     "total": len(check3a_pass)},
        "ordered_in_conv_window":  {"n_pass": n_conv,   "total": len(check3_pass)},
    },
    "verdict": verdict,
    "completed_at": datetime.utcnow().isoformat(),
}, indent=2, default=str))
print(f"\nResult written: {out}")
print(f"[{datetime.utcnow().isoformat()}] Validation complete")
