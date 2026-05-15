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
import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Any
from scipy import stats as _scipy_stats

_repo = str(Path(__file__).resolve().parent)
sys.path.insert(0, _repo)
from tar_storage import ensure_workspace_layout, resolve_workspace
workspace = str(ensure_workspace_layout(resolve_workspace(Path(_repo)), repo_root=Path(_repo)))

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark
from tar_lab.result_artifacts import collect_environment_snapshot, wrap_verdict_separation, write_canonical_comparison_result
from tar_lab.manifest import load_and_verify_manifest, ManifestGateError, write_refuse_note

SEEDS   = [42, 0, 1, 2, 3]
BACKBONE = "resnet18"
EPOCHS  = 40
METHODS = ["tcl", "ewc", "si", "sgd_baseline"]
BASE_CFG = ContinualLearningBenchmarkConfig(
    seed=SEEDS[0],
    train_epochs_per_task=EPOCHS,
    ewc_lambda=100.0,
)


def mean(v):  return sum(v) / len(v)
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))


def _require_manifest() -> tuple[Path, Any]:
    manifest_path_str = os.environ.get("TAR_MANIFEST_PATH", "")
    if not manifest_path_str:
        print("REFUSED: TAR_MANIFEST_PATH not set. Set it to the path of the signed manifest and re-run.", flush=True)
        sys.exit(1)
    manifest_path = Path(manifest_path_str)
    if not manifest_path.is_absolute():
        manifest_path = Path(_repo) / manifest_path
    try:
        manifest = load_and_verify_manifest(manifest_path, Path(_repo))
        for experiment_id in ("phase10_baseline", "phase10-baseline-rerun", "phase10-controlled-rerun"):
            try:
                manifest.assert_experiment_authorised(experiment_id)
                print(f"[RAIL 3] Manifest gate: OK ({manifest.manifest_id})", flush=True)
                return manifest_path, manifest
            except ManifestGateError:
                continue
        raise ManifestGateError(
            "Manifest does not authorise any accepted Phase 10 execution id "
            "('phase10_baseline', 'phase10-baseline-rerun', 'phase10-controlled-rerun')."
        )
    except ManifestGateError as exc:
        write_refuse_note(
            Path(workspace),
            component="phase10_baseline",
            reason=str(exc),
            experiment_id="phase10_baseline",
            manifest_path=str(manifest_path),
        )
        print(f"REFUSED: {exc}", flush=True)
        sys.exit(1)


_manifest_path, _manifest = _require_manifest()


run_started_at = datetime.utcnow().isoformat()
print(f"\n{'='*70}")
print(f"Phase 10 — Full 4-Way Baseline Comparison")
print(f"backbone={BACKBONE}  epochs={EPOCHS}  seeds={SEEDS}")
print(f"methods={METHODS}")
print(f"{run_started_at}")
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
completed_at = datetime.utcnow().isoformat()
payload = {
    "backbone":     BACKBONE,
    "epochs":       EPOCHS,
    "seeds":        SEEDS,
    "methods":      METHODS,
    "per_seed":     per_seed,
    "aggregate":    agg,
    "pairwise":     pairwise,
    "verdict":      verdict,
    "completed_at": completed_at,
}
env_payload = collect_environment_snapshot(
    repo_root=Path(_repo),
    workspace=Path(workspace),
    config={
        "suite": "phase10_baseline",
        "base_benchmark_config": BASE_CFG.model_dump(mode="json"),
        "methods": METHODS,
        "backbone": BACKBONE,
        "epochs": EPOCHS,
        "per_method_overrides": {
            "tcl": {},
            "ewc": {"ewc_lambda": 100.0},
            "si": {"si_c": BASE_CFG.si_c, "si_xi": BASE_CFG.si_xi},
            "sgd_baseline": {},
        },
    },
    trigger="manual_script",
    source_script=Path(__file__).name,
    run_started_at=run_started_at,
    run_ended_at=completed_at,
    extra={"logical_name": "phase10_baseline"},
)
artifacts = write_canonical_comparison_result(
    workspace=Path(workspace),
    logical_name="phase10_baseline",
    payload=wrap_verdict_separation(payload),
    env_payload=env_payload,
    phase_number=10,
    source_script=Path(__file__).name,
)

print(f"\nResult written: {artifacts['result_path']}")
print(f"Env snapshot: {artifacts['env_path']}")
print(f"Index updated: {artifacts['index_path']}")
print(f"[{completed_at}] Phase 10 baseline complete")
