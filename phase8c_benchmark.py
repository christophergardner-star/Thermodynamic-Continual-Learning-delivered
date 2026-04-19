"""
Phase 8C five-seed benchmark.
Pre-registered outcomes (from docs/phase8_roadmap.md):
  Outcome A: delta > 0.01, p < 0.05, d > 0.5  — clean win
  Outcome B: directional delta OR lower variance OR better accuracy tradeoff
  Outcome C: no SGD separation → weight penalty needed
"""
import sys, json, math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace/Thermodynamic-Continual-Learning-delivered")
workspace = "/workspace/Thermodynamic-Continual-Learning-delivered"

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark

SEEDS  = [42, 0, 1, 2, 3]
EPOCHS = 15


def mean(v):   return sum(v) / len(v)
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))


# ── paired t-test (one-sample t on deltas) ────────────────────────────────────
def paired_t(deltas):
    n = len(deltas)
    m = mean(deltas)
    s = std(deltas)
    if s < 1e-12 or n < 2:
        return float("nan"), float("nan")
    t = m / (s / math.sqrt(n))
    try:
        from scipy import stats as _stats
        p = float(_stats.ttest_1samp(deltas, 0).pvalue)
    except ImportError:
        p = float("nan")
    return t, p


def cohens_d(deltas):
    return mean(deltas) / max(std(deltas), 1e-12)


# ── run ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Phase 8C Five-Seed Benchmark")
print(f"seeds={SEEDS}  epochs_per_task={EPOCHS}")
print(f"{datetime.utcnow().isoformat()}")
print(f"{'='*60}", flush=True)

results = []
for seed in SEEDS:
    print(f"\n--- seed={seed} ---", flush=True)
    cfg = ContinualLearningBenchmarkConfig(seed=seed, train_epochs_per_task=EPOCHS)

    tcl = run_split_cifar10_benchmark(cfg, method="tcl", workspace=workspace)
    print(f"  TCL  forgetting={tcl.mean_forgetting:.4f}  acc={tcl.final_mean_accuracy:.4f}", flush=True)

    sgd = run_split_cifar10_benchmark(cfg, method="sgd_baseline", workspace=workspace)
    print(f"  SGD  forgetting={sgd.mean_forgetting:.4f}  acc={sgd.final_mean_accuracy:.4f}", flush=True)

    delta = tcl.mean_forgetting - sgd.mean_forgetting
    print(f"  delta={delta:+.4f}  ({'TCL better' if delta < 0 else 'SGD better'})", flush=True)

    results.append({
        "seed":            seed,
        "tcl_forgetting":  tcl.mean_forgetting,
        "sgd_forgetting":  sgd.mean_forgetting,
        "tcl_acc":         tcl.final_mean_accuracy,
        "sgd_acc":         sgd.final_mean_accuracy,
        "delta":           delta,
    })


# ── aggregate ─────────────────────────────────────────────────────────────────
deltas    = [r["delta"] for r in results]
tcl_forg  = [r["tcl_forgetting"] for r in results]
sgd_forg  = [r["sgd_forgetting"] for r in results]
tcl_acc   = [r["tcl_acc"] for r in results]
sgd_acc   = [r["sgd_acc"] for r in results]

mean_delta = mean(deltas)
t_stat, p_val = paired_t(deltas)
d_stat = cohens_d(deltas)
n_tcl_better = sum(1 for d in deltas if d < 0)

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'seed':>6}  {'TCL_forg':>10}  {'SGD_forg':>10}  {'delta':>8}  {'winner':>8}")
for r in results:
    w = "TCL" if r["delta"] < 0 else "SGD"
    print(f"  {r['seed']:>4}  {r['tcl_forgetting']:>10.4f}  {r['sgd_forgetting']:>10.4f}  {r['delta']:>+8.4f}  {w:>8}")

print(f"\n  mean_delta={mean_delta:+.4f}  ({n_tcl_better}/{len(SEEDS)} seeds TCL better)")
print(f"  TCL  forgetting={mean(tcl_forg):.4f}±{std(tcl_forg):.4f}  acc={mean(tcl_acc):.4f}±{std(tcl_acc):.4f}")
print(f"  SGD  forgetting={mean(sgd_forg):.4f}±{std(sgd_forg):.4f}  acc={mean(sgd_acc):.4f}±{std(sgd_acc):.4f}")
print(f"  t={t_stat:.3f}  p={p_val:.4f}  Cohen's d={d_stat:.3f}")


# ── verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"OUTCOME VERDICT")
print(f"{'='*60}")

outcome_a = (mean_delta < -0.01 and
             (math.isnan(p_val) or p_val < 0.05) and
             d_stat > 0.5)
outcome_b = (mean_delta < 0 or
             std(tcl_forg) < std(sgd_forg) or
             mean(tcl_acc) > mean(sgd_acc))

if outcome_a:
    verdict = ("OUTCOME A — clean win.  TCL reduces forgetting by >{:.1f}pp, "
               "p<0.05, d>{:.2f}.  Proceed to Phase 8D (class-incremental).").format(
               abs(mean_delta) * 100, d_stat)
elif outcome_b:
    verdict = ("OUTCOME B — partial result.  TCL shows directional improvement "
               "(mean_delta={:+.4f}) but does not meet all Outcome A thresholds.  "
               "Interpret as: fixed anchor improves TCL but effect size needs "
               "weight penalty (Phase 9) to amplify.").format(mean_delta)
else:
    verdict = ("OUTCOME C — no separation.  Anchor fix correct mechanistically "
               "but insufficient alone.  Next: dimensionality-weighted L2 "
               "weight penalty term.")

print(f"\n{verdict}")


# ── write result ──────────────────────────────────────────────────────────────
out = Path(workspace) / "tar_state" / "comparisons" / "phase8c_benchmark.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "seeds":       SEEDS,
    "epochs":      EPOCHS,
    "results":     results,
    "aggregate": {
        "mean_delta":  mean_delta,
        "t_stat":      t_stat,
        "p_val":       p_val,
        "cohens_d":    d_stat,
        "n_tcl_better": n_tcl_better,
        "tcl_forgetting_mean": mean(tcl_forg),
        "tcl_forgetting_std":  std(tcl_forg),
        "sgd_forgetting_mean": mean(sgd_forg),
        "sgd_forgetting_std":  std(sgd_forg),
    },
    "verdict": verdict,
    "completed_at": datetime.utcnow().isoformat(),
}, indent=2, default=str))
print(f"\nResult written: {out}")
print(f"[{datetime.utcnow().isoformat()}] Benchmark complete")
