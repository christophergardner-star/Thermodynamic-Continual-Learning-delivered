"""
Phase 11 — TCL ablation study.

Four conditions isolate which components of TCL are load-bearing:

  sgd           — fine-tuning, no anti-forgetting mechanism (baseline)
  governor_only — thermodynamic governor (LR adjustment) only; no L2 penalty
  penalty_only  — D_PR-fixed L2 weight anchor only; no governor, no LR adjustment
                  (D_PR=1.0: penalty = lambda_tcl * L2_dist, unscaled)
  full_tcl      — governor + D_PR-scaled L2 penalty (Phase 10 configuration)

Backbone: ResNet-18, 40 epochs/task, 5 seeds {42, 0, 1, 2, 3}.
Dataset:  Split-CIFAR-10 (task-incremental, 5 tasks).

Pre-registered TCL hyperparameters (unchanged from Phase 10):
  lambda_tcl=0.01, alpha=0.5, anchor window 20 batches.

Outcome criteria (pre-registered):
  COMPONENT_VERDICT: full_tcl > both ablations > sgd        — both components load-bearing
  PENALTY_DOMINANT:  penalty_only ≈ full_tcl >> governor_only — penalty is the real contribution
  GOVERNOR_DOMINANT: governor_only ≈ full_tcl >> penalty_only  — governor is the real contribution
  NEITHER_HELPS:     all ablations ≈ sgd                       — revisit mechanism
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

SEEDS    = [42, 0, 1, 2, 3]
BACKBONE = "resnet18"
EPOCHS   = 40

# Each entry: (label, method_string, config_overrides)
CONDITIONS = [
    ("sgd",           "sgd_baseline",    {}),
    ("governor_only", "tcl",             {"tcl_penalty_lambda": 0.0}),
    ("penalty_only",  "tcl_penalty_only", {}),
    ("full_tcl",      "tcl",             {}),
]
LABELS = [c[0] for c in CONDITIONS]


def mean(v):  return sum(v) / len(v)
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
def jaf(acc, forg):  return acc * (1.0 - forg)


print(f"\n{'='*70}")
print(f"Phase 11 — TCL Ablation Study")
print(f"backbone={BACKBONE}  epochs={EPOCHS}  seeds={SEEDS}")
print(f"conditions={LABELS}")
print(f"{datetime.utcnow().isoformat()}")
print(f"{'='*70}", flush=True)

per_seed   = []
forgetting = {lbl: [] for lbl in LABELS}
accuracy   = {lbl: [] for lbl in LABELS}

for seed in SEEDS:
    print(f"\n--- seed={seed} ---", flush=True)
    row = {"seed": seed}
    for label, method, overrides in CONDITIONS:
        cfg = ContinualLearningBenchmarkConfig(
            seed=seed,
            train_epochs_per_task=EPOCHS,
            ewc_lambda=100.0,
            **overrides,
        )
        r = run_split_cifar10_benchmark(cfg, method=method, workspace=workspace, backbone=BACKBONE)
        row[f"{label}_forgetting"] = r.mean_forgetting
        row[f"{label}_acc"]        = r.final_mean_accuracy
        forgetting[label].append(r.mean_forgetting)
        accuracy[label].append(r.final_mean_accuracy)
        j = jaf(r.final_mean_accuracy, r.mean_forgetting)
        print(
            f"  {label:16s}  forgetting={r.mean_forgetting:.4f}"
            f"  acc={r.final_mean_accuracy:.4f}  JAF={j:.4f}",
            flush=True,
        )
    per_seed.append(row)


# ── aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"AGGREGATE (mean ± std)")
print(f"{'='*70}")

agg = {}
for label in LABELS:
    f_vals = forgetting[label]
    a_vals = accuracy[label]
    j_vals = [jaf(a, f) for a, f in zip(a_vals, f_vals)]
    agg[label] = {
        "forgetting_mean": mean(f_vals),
        "forgetting_std":  std(f_vals),
        "acc_mean":        mean(a_vals),
        "acc_std":         std(a_vals),
        "jaf_mean":        mean(j_vals),
        "jaf_std":         std(j_vals),
    }
    print(
        f"  {label:16s}  "
        f"F={agg[label]['forgetting_mean']:.4f}±{agg[label]['forgetting_std']:.4f}  "
        f"A={agg[label]['acc_mean']:.4f}±{agg[label]['acc_std']:.4f}  "
        f"JAF={agg[label]['jaf_mean']:.4f}±{agg[label]['jaf_std']:.4f}"
    )


# ── pairwise: full_tcl vs each condition ─────────────────────────────────────
print(f"\n{'='*70}")
print(f"PAIRWISE: full_tcl vs each condition (paired t-test on forgetting deltas)")
print(f"{'='*70}")

tcl_forg = forgetting["full_tcl"]
pairwise = {}
for other in ["sgd", "governor_only", "penalty_only"]:
    deltas = [tcl - b for tcl, b in zip(tcl_forg, forgetting[other])]
    t_stat, p_val = _scipy_stats.ttest_1samp(deltas, 0)
    d_stat = abs(mean(deltas)) / max(std(deltas), 1e-12)
    n_tcl_better = sum(1 for d in deltas if d < 0)
    pairwise[other] = {
        "mean_delta": mean(deltas),
        "t_stat":     float(t_stat),
        "p_val":      float(p_val),
        "cohens_d":   d_stat,
        "n_full_tcl_better": n_tcl_better,
    }
    direction = "full_tcl better" if mean(deltas) < 0 else "full_tcl worse"
    print(
        f"  full_tcl vs {other:16s}: delta={mean(deltas):+.4f}  "
        f"p={p_val:.4f}  d={d_stat:.3f}  {n_tcl_better}/5  ({direction})"
    )

# Also report governor_only vs sgd and penalty_only vs sgd
print()
print(f"  Component vs SGD baseline:")
for component in ["governor_only", "penalty_only"]:
    deltas = [c - s for c, s in zip(forgetting[component], forgetting["sgd"])]
    t_stat, p_val = _scipy_stats.ttest_1samp(deltas, 0)
    d_stat = abs(mean(deltas)) / max(std(deltas), 1e-12)
    n_better = sum(1 for d in deltas if d < 0)
    direction = "better" if mean(deltas) < 0 else "worse"
    print(
        f"    {component:16s} vs sgd: delta={mean(deltas):+.4f}  "
        f"p={p_val:.4f}  d={d_stat:.3f}  {n_better}/5  ({direction})"
    )


# ── verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"COMPONENT VERDICT")
print(f"{'='*70}")

vs_sgd      = pairwise["sgd"]
vs_gov      = pairwise["governor_only"]
vs_pen      = pairwise["penalty_only"]

full_beats_sgd = vs_sgd["mean_delta"] < -0.01 and vs_sgd["p_val"] < 0.05
full_beats_gov = vs_gov["mean_delta"] < -0.01 and vs_gov["p_val"] < 0.05
full_beats_pen = vs_pen["mean_delta"] < -0.01 and vs_pen["p_val"] < 0.05

gov_delta_vs_sgd = mean([c - s for c, s in zip(forgetting["governor_only"], forgetting["sgd"])])
pen_delta_vs_sgd = mean([c - s for c, s in zip(forgetting["penalty_only"], forgetting["sgd"])])

gov_helps = gov_delta_vs_sgd < -0.01
pen_helps = pen_delta_vs_sgd < -0.01

if full_beats_sgd and full_beats_gov and full_beats_pen and gov_helps and pen_helps:
    verdict_key = "BOTH_LOAD_BEARING"
    verdict = (
        f"BOTH COMPONENTS LOAD-BEARING. "
        f"full_tcl beats sgd (d={vs_sgd['p_val']:.4f}), governor_only (d={vs_gov['cohens_d']:.2f}), "
        f"and penalty_only (d={vs_pen['cohens_d']:.2f}). "
        f"Both governor and penalty contribute independently. "
        f"Contribution 1 (TCL method) is mechanistically supported."
    )
elif full_beats_sgd and pen_helps and not gov_helps:
    verdict_key = "PENALTY_DOMINANT"
    verdict = (
        f"PENALTY IS THE PRIMARY COMPONENT. "
        f"penalty_only beats sgd (delta={pen_delta_vs_sgd:+.4f}) but governor_only does not "
        f"(delta={gov_delta_vs_sgd:+.4f}). "
        f"The D_PR-weighted L2 anchor is load-bearing; governor LR adjustment is secondary. "
        f"Paper framing: emphasise penalty as the mechanistic contribution."
    )
elif full_beats_sgd and gov_helps and not pen_helps:
    verdict_key = "GOVERNOR_DOMINANT"
    verdict = (
        f"GOVERNOR IS THE PRIMARY COMPONENT. "
        f"governor_only beats sgd (delta={gov_delta_vs_sgd:+.4f}) but penalty_only does not "
        f"(delta={pen_delta_vs_sgd:+.4f}). "
        f"Regime-adaptive LR adjustment is load-bearing; L2 penalty at lambda=0.01 with D_PR=1.0 is secondary. "
        f"Paper framing: emphasise governor as the mechanistic contribution."
    )
elif full_beats_sgd and not gov_helps and not pen_helps:
    verdict_key = "SYNERGY_REQUIRED"
    verdict = (
        f"SYNERGY: neither component alone beats SGD, but together they do. "
        f"governor_only delta={gov_delta_vs_sgd:+.4f}, penalty_only delta={pen_delta_vs_sgd:+.4f}. "
        f"Interaction effect. Paper framing: both components are necessary; synergy is the contribution."
    )
else:
    verdict_key = "NEITHER_HELPS"
    verdict = (
        f"NEITHER COMPONENT HELPS over SGD. "
        f"full_tcl delta vs sgd={vs_sgd['mean_delta']:+.4f} (p={vs_sgd['p_val']:.4f}). "
        f"Revisit mechanism — Phase 10 result may not replicate under ablation conditions."
    )

print(f"\n{verdict_key}: {verdict}")


# ── write result ──────────────────────────────────────────────────────────────
out = Path(workspace) / "tar_state" / "comparisons" / "phase11_ablation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "backbone":     BACKBONE,
    "epochs":       EPOCHS,
    "seeds":        SEEDS,
    "conditions":   LABELS,
    "per_seed":     per_seed,
    "aggregate":    agg,
    "pairwise":     pairwise,
    "verdict_key":  verdict_key,
    "verdict":      verdict,
    "completed_at": datetime.utcnow().isoformat(),
}, indent=2, default=str))

print(f"\nResult written: {out}")
print(f"[{datetime.utcnow().isoformat()}] Phase 11 ablation complete")
