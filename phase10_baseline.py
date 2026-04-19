"""
Phase 10 — Full 4-way baseline comparison.

Methods: TCL vs EWC vs SI vs SGD
Backbone: ResNet-18, 40 epochs/task, 5 seeds
Dataset: Split-CIFAR-10 (task-incremental, 5 tasks)

Pre-registered outcomes:
  Outcome A: TCL mean_forgetting < EWC AND p < 0.05 AND d > 0.5  — beats strong baseline
  Outcome B: TCL directional improvement vs EWC OR clearly beats SGD
  Outcome C: no improvement over EWC → revisit mechanism

EWC lambda=100 (selected Phase 7 sweep).
SI c=0.1, xi=0.001 (schema defaults).
"""
import sys
import json
import math
from pathlib import Path
from datetime import datetime
from scipy import stats as _scipy_stats

sys.path.insert(0, "/workspace/Thermodynamic-Continual-Learning-delivered")
workspace = "/workspace/Thermodynamic-Continual-Learning-delivered"

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark

SEEDS   = [42, 0, 1, 2, 3]
BACKBONE = "resnet18"
EPOCHS  = 40
METHODS = ["tcl", "ewc", "si", "sgd_baseline"]


def mean(v):  return sum(v) / len(v)
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))


print(f"\n{'='*70}")
print(f"Phase 10 — Full 4-Way Baseline Comparison")
print(f"backbone={BACKBONE}  epochs={EPOCHS}  seeds={SEEDS}")
print(f"methods={METHODS}")
print(f"{datetime.utcnow().isoformat()}")
print(f"{'='*70}", flush=True)

per_seed = []
forgetting = {m: [] for m in METHODS}
accuracy   = {m: [] for m in METHODS}

for seed in SEEDS:
    print(f"\n--- seed={seed} ---", flush=True)
    cfg = ContinualLearningBenchmarkConfig(
        seed=seed,
        train_epochs_per_task=EPOCHS,
        ewc_lambda=100.0,
    )
    row = {"seed": seed}
    for method in METHODS:
        r = run_split_cifar10_benchmark(cfg, method=method, workspace=workspace, backbone=BACKBONE)
        row[f"{method}_forgetting"] = r.mean_forgetting
        row[f"{method}_acc"]        = r.final_mean_accuracy
        forgetting[method].append(r.mean_forgetting)
        accuracy[method].append(r.final_mean_accuracy)
        print(f"  {method:12s}  forgetting={r.mean_forgetting:.4f}  acc={r.final_mean_accuracy:.4f}", flush=True)
    per_seed.append(row)


# ── aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")

print(f"{'seed':>6}  " + "  ".join(f"{m:>14}" for m in METHODS))
for row in per_seed:
    vals = "  ".join(f"{row[f'{m}_forgetting']:>14.4f}" for m in METHODS)
    print(f"  {row['seed']:>4}  {vals}")

print()
agg = {}
for method in METHODS:
    agg[method] = {
        "forgetting_mean": mean(forgetting[method]),
        "forgetting_std":  std(forgetting[method]),
        "acc_mean":        mean(accuracy[method]),
        "acc_std":         std(accuracy[method]),
    }
    print(f"  {method:12s}  "
          f"forgetting={agg[method]['forgetting_mean']:.4f}±{agg[method]['forgetting_std']:.4f}  "
          f"acc={agg[method]['acc_mean']:.4f}±{agg[method]['acc_std']:.4f}")


# ── pairwise TCL vs each baseline ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"PAIRWISE: TCL vs each baseline (paired t-test on forgetting deltas)")
print(f"{'='*70}")

tcl_forg = forgetting["tcl"]
pairwise = {}
for baseline in ["ewc", "si", "sgd_baseline"]:
    deltas = [tcl - b for tcl, b in zip(tcl_forg, forgetting[baseline])]
    t_stat, p_val = _scipy_stats.ttest_1samp(deltas, 0)
    d_stat = abs(mean(deltas)) / max(std(deltas), 1e-12)
    n_tcl_better = sum(1 for d in deltas if d < 0)
    pairwise[baseline] = {
        "mean_delta": mean(deltas),
        "t_stat":     float(t_stat),
        "p_val":      float(p_val),
        "cohens_d":   d_stat,
        "n_tcl_better": n_tcl_better,
    }
    direction = "TCL better" if mean(deltas) < 0 else "TCL worse"
    print(f"  TCL vs {baseline:12s}: delta={mean(deltas):+.4f}  "
          f"p={p_val:.4f}  d={d_stat:.3f}  {n_tcl_better}/5  ({direction})")


# ── verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"OUTCOME VERDICT")
print(f"{'='*70}")

vs_ewc = pairwise["ewc"]
vs_sgd = pairwise["sgd_baseline"]

outcome_a = (
    vs_ewc["mean_delta"] < -0.01
    and vs_ewc["p_val"] < 0.05
    and vs_ewc["cohens_d"] > 0.5
)
outcome_b = vs_ewc["mean_delta"] < 0 or vs_sgd["mean_delta"] < -0.05

if outcome_a:
    verdict = (
        f"OUTCOME A — TCL beats EWC (strong baseline). "
        f"delta={vs_ewc['mean_delta']:+.4f}, p={vs_ewc['p_val']:.4f}, d={vs_ewc['cohens_d']:.2f}. "
        f"Publishable result."
    )
elif outcome_b:
    verdict = (
        f"OUTCOME B — TCL directionally better than EWC (delta={vs_ewc['mean_delta']:+.4f}) "
        f"or clearly better than SGD (delta={vs_sgd['mean_delta']:+.4f}). "
        f"Strong result vs SGD but not Outcome A vs EWC. "
        f"Publishable with honest framing — thermodynamic regularisation vs weight-importance methods."
    )
else:
    verdict = (
        f"OUTCOME C — TCL does not improve over EWC (delta={vs_ewc['mean_delta']:+.4f}). "
        f"Weight penalty approach insufficient at this scale vs importance-weighted methods. "
        f"Consider Fisher-weighted scaling for the L2 penalty."
    )

print(f"\n{verdict}")


# ── write result ──────────────────────────────────────────────────────────────
out = Path(workspace) / "tar_state" / "comparisons" / "phase10_baseline.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "backbone":     BACKBONE,
    "epochs":       EPOCHS,
    "seeds":        SEEDS,
    "methods":      METHODS,
    "per_seed":     per_seed,
    "aggregate":    agg,
    "pairwise":     pairwise,
    "verdict":      verdict,
    "completed_at": datetime.utcnow().isoformat(),
}, indent=2, default=str))

print(f"\nResult written: {out}")
print(f"[{datetime.utcnow().isoformat()}] Phase 10 baseline complete")
