"""
Phase 8C scale-up benchmark — ResNet-18 backbone.

Hypothesis: the anchor fix shows noisy results on the tiny model (~189K params)
because gradient noise dominates the thermodynamic signal.  A larger backbone
(ResNet-18, ~11M params) gives the governor more thermal mass to work with and
should produce a tighter, more significant delta.

Same 5 seeds, same 15 epochs, same pre-registered criteria as phase8c_benchmark.py.
Only change: backbone="resnet18".
"""
import sys, json, math
from pathlib import Path
from datetime import datetime
from scipy import stats as _scipy_stats

sys.path.insert(0, "/workspace/Thermodynamic-Continual-Learning-delivered")
workspace = "/workspace/Thermodynamic-Continual-Learning-delivered"

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark

SEEDS    = [42, 0, 1, 2, 3]
BACKBONE = "resnet18"
# ResNet-18 needs more epochs to converge than the tiny model.
# At 15 epochs it is badly underfitted and the governor misreads
# random-initialisation noise as thermal stability.
EPOCHS   = 40 if BACKBONE == "resnet18" else 15


def mean(v):  return sum(v) / len(v)
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))


print(f"\n{'='*60}")
print(f"Phase 8C Scale-Up Benchmark — {BACKBONE}")
print(f"seeds={SEEDS}  epochs_per_task={EPOCHS}")
print(f"{datetime.utcnow().isoformat()}")
print(f"{'='*60}", flush=True)

results = []
for seed in SEEDS:
    print(f"\n--- seed={seed} ---", flush=True)
    cfg = ContinualLearningBenchmarkConfig(seed=seed, train_epochs_per_task=EPOCHS)

    tcl = run_split_cifar10_benchmark(cfg, method="tcl", workspace=workspace, backbone=BACKBONE)
    print(f"  TCL  forgetting={tcl.mean_forgetting:.4f}  acc={tcl.final_mean_accuracy:.4f}", flush=True)

    sgd = run_split_cifar10_benchmark(cfg, method="sgd_baseline", workspace=workspace, backbone=BACKBONE)
    print(f"  SGD  forgetting={sgd.mean_forgetting:.4f}  acc={sgd.final_mean_accuracy:.4f}", flush=True)

    delta = tcl.mean_forgetting - sgd.mean_forgetting
    print(f"  delta={delta:+.4f}  ({'TCL better' if delta < 0 else 'SGD better'})", flush=True)

    results.append({
        "seed": seed,
        "tcl_forgetting": tcl.mean_forgetting,
        "sgd_forgetting": sgd.mean_forgetting,
        "tcl_acc":        tcl.final_mean_accuracy,
        "sgd_acc":        sgd.final_mean_accuracy,
        "delta":          delta,
    })

# ── aggregate ─────────────────────────────────────────────────────────────────
deltas   = [r["delta"] for r in results]
tcl_forg = [r["tcl_forgetting"] for r in results]
sgd_forg = [r["sgd_forgetting"] for r in results]
tcl_acc  = [r["tcl_acc"] for r in results]
sgd_acc  = [r["sgd_acc"] for r in results]

t_stat, p_val = _scipy_stats.ttest_1samp(deltas, 0)
mean_delta = mean(deltas)
d_stat = mean_delta / max(std(deltas), 1e-12)
n_tcl_better = sum(1 for d in deltas if d < 0)

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY — {BACKBONE}")
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

outcome_a = mean_delta < -0.01 and p_val < 0.05 and d_stat > 0.5
outcome_b = mean_delta < 0 or std(tcl_forg) < std(sgd_forg) or mean(tcl_acc) > mean(sgd_acc)

if outcome_a:
    verdict = (f"OUTCOME A — clean win on {BACKBONE}.  "
               f"TCL reduces forgetting by {abs(mean_delta)*100:.1f}pp, "
               f"p={p_val:.4f}, d={d_stat:.2f}.  Scale-up hypothesis confirmed.")
elif outcome_b:
    verdict = (f"OUTCOME B on {BACKBONE}.  Directional improvement (mean_delta={mean_delta:+.4f}) "
               f"but not all Outcome A thresholds met (p={p_val:.4f}, d={d_stat:.2f}).  "
               f"Signal stronger than tiny model but weight penalty still needed.")
else:
    verdict = (f"OUTCOME C on {BACKBONE}.  No separation even at scale.  "
               f"Mechanism needs revisiting.")

print(f"\n{verdict}")

# ── tiny model comparison line ────────────────────────────────────────────────
print(f"\n  [tiny model ref]  mean_delta=-0.0059  p≈0.68  d≈0.20  (3/5 TCL better)")
print(f"  [{BACKBONE}]  mean_delta={mean_delta:+.4f}  p={p_val:.4f}  d={d_stat:.3f}  ({n_tcl_better}/5 TCL better)")

# ── write result ──────────────────────────────────────────────────────────────
out = Path(workspace) / "tar_state" / "comparisons" / "phase8c_scale.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "backbone": BACKBONE, "seeds": SEEDS, "epochs": EPOCHS,
    "results": results,
    "aggregate": {
        "mean_delta": mean_delta, "t_stat": float(t_stat),
        "p_val": float(p_val), "cohens_d": d_stat,
        "n_tcl_better": n_tcl_better,
        "tcl_forgetting_mean": mean(tcl_forg), "tcl_forgetting_std": std(tcl_forg),
        "sgd_forgetting_mean": mean(sgd_forg), "sgd_forgetting_std": std(sgd_forg),
    },
    "verdict": verdict,
    "completed_at": datetime.utcnow().isoformat(),
}, indent=2, default=str))
print(f"\nResult written: {out}")
print(f"[{datetime.utcnow().isoformat()}] Scale benchmark complete")
