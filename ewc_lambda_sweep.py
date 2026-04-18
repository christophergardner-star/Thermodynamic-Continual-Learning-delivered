"""
EWC lambda sweep: lambda in {10, 100, 1000} x 5 seeds.
lambda=100 runs already exist from ewc_fix_run.py.
This script runs lambda=10 and lambda=1000, then selects
the best lambda and produces the final comparison result.
"""
import sys, json, collections, math
from pathlib import Path
from datetime import datetime
from uuid import uuid4

sys.path.insert(0, "/workspace/Thermodynamic-Continual-Learning-delivered")
workspace = "/workspace/Thermodynamic-Continual-Learning-delivered"

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark

SEEDS = [42, 123, 456, 789, 1337]
SWEEP_LAMBDAS = [10.0, 1000.0]   # 100.0 already done

runs_dir = Path(workspace) / "tar_state" / "comparisons" / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)

# ── run lambda=10 and lambda=1000 ────────────────────────────────────────────
new_ewc_results = {}   # lambda -> list[result]

for lam in SWEEP_LAMBDAS:
    new_ewc_results[lam] = []
    print(f"\n[{datetime.utcnow().isoformat()}] EWC sweep lambda={lam}", flush=True)
    for seed in SEEDS:
        print(f"  seed={seed}", flush=True)
        cfg = ContinualLearningBenchmarkConfig(seed=seed, ewc_lambda=lam)
        result = run_split_cifar10_benchmark(cfg, method="ewc", workspace=workspace)
        tag = f"ewc_lam{int(lam)}"
        out = runs_dir / f"{tag}_{result.benchmark_id}.json"
        d = result.model_dump()
        d["ewc_lambda_sweep_value"] = lam    # extra annotation
        out.write_text(json.dumps(d, indent=2))
        new_ewc_results[lam].append(result)
        print(f"    forgetting={result.mean_forgetting:.4f}  acc={result.final_mean_accuracy:.4f}", flush=True)

# ── collect lambda=100 results from runs dir ──────────────────────────────────
lambda100_runs = []
for f in runs_dir.glob("ewc_fixed_*.json"):
    try:
        d = json.loads(f.read_text())
        if d.get("method") == "ewc":
            lambda100_runs.append(d)
    except Exception:
        pass

# ── summarise each lambda ─────────────────────────────────────────────────────
def mean(vals): return sum(vals) / len(vals) if vals else float("nan")
def std(vals):
    if len(vals) < 2: return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

lambda_stats = {}

# lambda=100
forg100 = [r["mean_forgetting"] for r in lambda100_runs]
acc100  = [r["final_mean_accuracy"] for r in lambda100_runs]
lambda_stats[100.0] = {"forgetting_mean": mean(forg100), "forgetting_std": std(forg100),
                        "acc_mean": mean(acc100), "n": len(forg100)}

for lam in SWEEP_LAMBDAS:
    runs = new_ewc_results[lam]
    forg = [r.mean_forgetting for r in runs]
    acc  = [r.final_mean_accuracy for r in runs]
    lambda_stats[lam] = {"forgetting_mean": mean(forg), "forgetting_std": std(forg),
                          "acc_mean": mean(acc), "n": len(forg)}

print("\n=== LAMBDA SWEEP SUMMARY ===")
print(f"{'lambda':>8}  {'forgetting':>12}  {'acc':>8}  n")
for lam in sorted(lambda_stats):
    s = lambda_stats[lam]
    print(f"{lam:>8.0f}  {s['forgetting_mean']:.4f}±{s['forgetting_std']:.4f}  {s['acc_mean']:.4f}  {s['n']}")

# ── select best lambda: lowest mean_forgetting, acc > 0.6 ────────────────────
eligible = {lam: s for lam, s in lambda_stats.items()
            if s["acc_mean"] > 0.6 and s["n"] == 5}
if not eligible:
    eligible = lambda_stats   # fallback: use all
best_lambda = min(eligible, key=lambda lam: eligible[lam]["forgetting_mean"])
print(f"\nSelected best lambda: {best_lambda} "
      f"(forgetting={eligible[best_lambda]['forgetting_mean']:.4f}, "
      f"acc={eligible[best_lambda]['acc_mean']:.4f})")

# ── build canonical dict for final comparison ─────────────────────────────────
all_run_files = list(runs_dir.glob("*.json"))
method_seed_best = collections.defaultdict(list)

for f in all_run_files:
    try:
        d = json.loads(f.read_text())
        method = d.get("method")
        seed   = d.get("seed")
        if not method or seed is None:
            continue
        if method == "ewc":
            continue   # handled separately below
        method_seed_best[(method, seed)].append((f.stat().st_mtime, d))
    except Exception:
        pass

# pick most-recent non-EWC run per (method, seed)
canonical = {}
for (method, seed), entries in method_seed_best.items():
    entries.sort(key=lambda x: x[0], reverse=True)
    canonical[(method, seed)] = entries[0][1]

# add best-lambda EWC runs
if best_lambda == 100.0:
    for r in lambda100_runs:
        canonical[("ewc", r["seed"])] = r
else:
    for r in new_ewc_results[best_lambda]:
        canonical[("ewc", r.seed)] = r.model_dump()

# ── statistical helpers ───────────────────────────────────────────────────────
def mann_whitney_u(a, b):
    na, nb = len(a), len(b)
    u = sum(1 for x in a for y in b if x < y) + 0.5 * sum(1 for x in a for y in b if x == y)
    mu = na * nb / 2
    su = max(((na * nb * (na + nb + 1)) / 12) ** 0.5, 1e-8)
    z  = (u - mu) / su
    p  = float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))
    return float(u), float(p)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / max(na - 1, 1)
    vb = sum((x - mb) ** 2 for x in b) / max(nb - 1, 1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return (ma - mb) / max(pooled, 1e-8)

PRIMARY = "mean_forgetting"
METHODS = ["tcl", "ewc", "si", "sgd_baseline"]

method_values = {}
for m in METHODS:
    runs = [canonical[(m, s)] for s in SEEDS if (m, s) in canonical]
    method_values[m] = [r[PRIMARY] for r in runs]

method_means = {m: {PRIMARY: mean(vals),
                     "final_mean_accuracy": mean([canonical[(m, s)]["final_mean_accuracy"]
                                                  for s in SEEDS if (m, s) in canonical])}
                for m, vals in method_values.items()}
method_stds = {m: {PRIMARY: std(vals)} for m, vals in method_values.items()}

pairwise_pvalues = {}
pairwise_effect_sizes = {}
baselines = ["ewc", "si", "sgd_baseline"]
for baseline in baselines:
    key = f"tcl_vs_{baseline}"
    tcl_v = method_values.get("tcl", [])
    b_v   = method_values.get(baseline, [])
    if tcl_v and b_v:
        _, p = mann_whitney_u(tcl_v, b_v)
        d    = cohens_d(tcl_v, b_v)
        pairwise_pvalues[key]       = round(p, 4)
        pairwise_effect_sizes[key]  = round(d, 4)

tcl_better = all(pairwise_pvalues.get(f"tcl_vs_{b}", 1.0) < 0.05
                 and method_means["tcl"][PRIMARY] < method_means[b][PRIMARY]
                 for b in baselines)
tcl_worse  = any(pairwise_pvalues.get(f"tcl_vs_{b}", 1.0) < 0.05
                 and method_means["tcl"][PRIMARY] > method_means[b][PRIMARY]
                 for b in baselines)

lines = []
for baseline in baselines:
    key  = f"tcl_vs_{baseline}"
    tm   = method_means["tcl"][PRIMARY]
    bm   = method_means[baseline][PRIMARY]
    p    = pairwise_pvalues.get(key, 1.0)
    d    = pairwise_effect_sizes.get(key, 0.0)
    sig  = ("significantly better" if p < 0.05 and tm < bm
            else "significantly worse"    if p < 0.05 and tm > bm
            else "not significantly different")
    lines.append(f"TCL vs {baseline}: mean_forgetting {tm:.3f} vs {bm:.3f} "
                 f"(p={p:.3f}, d={d:.3f}) — {sig}.")

summary = ("Overall: TCL significantly reduces forgetting across all baselines."
           if tcl_better else
           "Overall: TCL shows significantly worse forgetting on at least one baseline."
           if tcl_worse else
           "Overall: TCL does not significantly differ from baselines on primary metric.")
honest_assessment = " ".join(lines) + " " + summary

print(f"\n=== FINAL HONEST ASSESSMENT (EWC lambda={best_lambda}) ===")
print(honest_assessment)
print("\nPer-method summary:")
for m in METHODS:
    vals = method_values.get(m, [])
    accs = [canonical[(m, s)]["final_mean_accuracy"] for s in SEEDS if (m, s) in canonical]
    print(f"  {m} (n={len(vals)}): forgetting={mean(vals):.4f}±{std(vals):.4f}  acc={mean(accs):.4f}")

# ── write sweep result ────────────────────────────────────────────────────────
result_id = uuid4().hex
sweep_out = Path(workspace) / "tar_state" / "comparisons" / f"lambda_sweep_{result_id}.json"

sweep_data = {
    "result_id": result_id,
    "sweep_type": "ewc_lambda_sweep",
    "lambdas_tested": sorted(lambda_stats.keys()),
    "best_lambda": best_lambda,
    "lambda_stats": {str(int(k)): v for k, v in lambda_stats.items()},
    "selection_criterion": "lowest_mean_forgetting_with_acc_gt_0.6",
    "completed_at": datetime.utcnow().isoformat(),
    "correction_note": (
        f"EWC lambda swept over {{10, 100, 1000}} x 5 seeds. "
        f"Original lambda=5000 over-constrained the 85K-param network (all tasks at 0.5 accuracy). "
        f"Best lambda selected: {best_lambda}."
    ),
    "method_means": method_means,
    "method_stds": method_stds,
    "pairwise_pvalues": pairwise_pvalues,
    "pairwise_effect_sizes": pairwise_effect_sizes,
    "tcl_is_significantly_better": tcl_better,
    "tcl_is_significantly_worse": tcl_worse,
    "honest_assessment": honest_assessment,
}
sweep_out.write_text(json.dumps(sweep_data, indent=2))
print(f"\nSweep result written: {sweep_out}")
print(f"result_id: {result_id}")
print(f"[{datetime.utcnow().isoformat()}] Lambda sweep complete")
