"""
Phase 13 — SI robustness sweep (minimal).

Reviewer-defence experiment: scopes the SI degeneracy claim.
SI at default hyperparameters (c=0.1, ξ=0.001) produced acc=0.500 on all
5 seeds in Phase 10. This sweep tests c∈{0.01, 0.1, 0.5} with ξ=0.001
fixed across 3 seeds to establish whether non-degenerate SI configurations
exist on this benchmark.

Backbone: ResNet-18, 40 epochs/task, 3 seeds {42, 0, 1}.
Dataset:  Split-CIFAR-10 (task-incremental, 5 tasks).
ξ=0.001 fixed (published default). c is the damping/regularisation scale.

Phase 10 reference (c=0.1, ξ=0.001, all 5 seeds): acc=0.500 on every seed.

Pre-registered outcome criteria:
  ALL_DEGENERATE:    acc=0.500 on all tested configurations — SI degeneracy
                     claim holds unconditionally for this benchmark.
  PARTIAL_RECOVERY:  at least one (c, ξ) produces acc > 0.55 — SI can be
                     non-degenerate; claim must be scoped to default params.
  FULL_RECOVERY:     all tested c values produce acc > 0.55 — default params
                     specifically are pathological; broader claim unsupported.

Collapse threshold: acc ≤ 0.55 (safely above 0.500 chance, allowing small
numerical variation while still indicating degenerate behaviour).
"""
import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime
from scipy import stats as _scipy_stats

_repo = str(Path(__file__).resolve().parent)
sys.path.insert(0, _repo)
from tar_storage import ensure_workspace_layout, resolve_workspace
workspace = str(ensure_workspace_layout(resolve_workspace(Path(_repo)), repo_root=Path(_repo)))

from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark
from tar_lab.result_artifacts import collect_environment_snapshot, write_canonical_comparison_result
from tar_lab.manifest import load_and_verify_manifest, ManifestGateError, write_refuse_note

SEEDS    = [42, 0, 1]
BACKBONE = "resnet18"
EPOCHS   = 40
XI_FIXED = 0.001
C_VALUES = [0.01, 0.1, 0.5]   # c=0.1 is Phase 10 default
BASE_CFG = ContinualLearningBenchmarkConfig(
    seed=SEEDS[0],
    train_epochs_per_task=EPOCHS,
    si_xi=XI_FIXED,
)

# Phase 10 reference: SI c=0.1, ξ=0.001, all 5 seeds → acc=0.500 exactly
# TCL reference (seeds 42/0/1) from phase10_controlled_rerun_20260509T132155Z.json
PHASE10_TCL_FORG = [0.08780, 0.10620, 0.09570]
PHASE10_TCL_ACC  = [0.7767, 0.7644, 0.7826]

COLLAPSE_THRESHOLD = 0.55


def mean(v):  return sum(v) / len(v)
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
def jaf(acc, forg):  return acc * (1.0 - forg)
def collapsed(acc):  return acc <= COLLAPSE_THRESHOLD


# ── RAIL 3: manifest gate ─────────────────────────────────────────────────────
_manifest_path_str = os.environ.get("TAR_MANIFEST_PATH", "")
if not _manifest_path_str:
    print("REFUSED: TAR_MANIFEST_PATH not set. Set it to the path of the signed manifest and re-run.", flush=True)
    sys.exit(1)
_manifest_path = Path(_manifest_path_str)
if not _manifest_path.is_absolute():
    _manifest_path = Path(_repo) / _manifest_path
try:
    _manifest = load_and_verify_manifest(_manifest_path, Path(_repo))
    _manifest.assert_experiment_authorised("phase13-si-sweep-rerun-20260511")
except ManifestGateError as _e:
    write_refuse_note(
        Path(workspace),
        component="phase13_si_sweep",
        reason=str(_e),
        experiment_id="phase13-si-sweep-rerun-20260511",
        manifest_path=str(_manifest_path),
    )
    print(f"REFUSED: {_e}", flush=True)
    sys.exit(1)
print(f"[RAIL 3] Manifest gate: OK ({_manifest.manifest_id})", flush=True)

run_started_at = datetime.utcnow().isoformat()
print(f"\n{'='*70}")
print(f"Phase 13 — SI Robustness Sweep")
print(f"backbone={BACKBONE}  epochs={EPOCHS}  seeds={SEEDS}")
print(f"c_values={C_VALUES}  xi_fixed={XI_FIXED}")
print(f"collapse_threshold={COLLAPSE_THRESHOLD}")
print(f"{run_started_at}")
print(f"{'='*70}", flush=True)

# results[c] = {"forgetting": [...], "accuracy": [...]}
results = {c: {"forgetting": [], "accuracy": []} for c in C_VALUES}
per_seed = []

for seed in SEEDS:
    print(f"\n--- seed={seed} ---", flush=True)
    row = {"seed": seed}
    for c in C_VALUES:
        cfg = ContinualLearningBenchmarkConfig(
            seed=seed,
            train_epochs_per_task=EPOCHS,
            si_c=c,
            si_xi=XI_FIXED,
        )
        r = run_split_cifar10_benchmark(cfg, method="si", workspace=workspace, backbone=BACKBONE)
        results[c]["forgetting"].append(r.mean_forgetting)
        results[c]["accuracy"].append(r.final_mean_accuracy)
        j = jaf(r.final_mean_accuracy, r.mean_forgetting)
        col = "*** COLLAPSE ***" if collapsed(r.final_mean_accuracy) else ""
        row[f"si_c{c}_forgetting"] = r.mean_forgetting
        row[f"si_c{c}_acc"]        = r.final_mean_accuracy
        print(
            f"  SI c={c}  ξ={XI_FIXED}  forgetting={r.mean_forgetting:.4f}"
            f"  acc={r.final_mean_accuracy:.4f}  JAF={j:.4f}  {col}",
            flush=True,
        )
    per_seed.append(row)


# ── aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"AGGREGATE (mean ± std) — 3 seeds {{42, 0, 1}}")
print(f"{'='*70}")

agg = {}
for c in C_VALUES:
    f_vals = results[c]["forgetting"]
    a_vals = results[c]["accuracy"]
    j_vals = [jaf(a, f) for a, f in zip(a_vals, f_vals)]
    n_col  = sum(1 for a in a_vals if collapsed(a))
    agg[c] = {
        "forgetting_mean": mean(f_vals),
        "forgetting_std":  std(f_vals),
        "acc_mean":        mean(a_vals),
        "acc_std":         std(a_vals),
        "jaf_mean":        mean(j_vals),
        "jaf_std":         std(j_vals),
        "n_collapsed":     n_col,
    }
    col_flag = f"  [{n_col}/3 collapsed]" if n_col > 0 else ""
    print(
        f"  SI c={c}  "
        f"F={agg[c]['forgetting_mean']:.4f}±{agg[c]['forgetting_std']:.4f}  "
        f"A={agg[c]['acc_mean']:.4f}±{agg[c]['acc_std']:.4f}  "
        f"JAF={agg[c]['jaf_mean']:.4f}±{agg[c]['jaf_std']:.4f}"
        f"{col_flag}"
    )

print(f"\n  TCL ref (seeds 42/0/1):")
tcl_j = [jaf(a, f) for a, f in zip(PHASE10_TCL_ACC, PHASE10_TCL_FORG)]
print(
    f"  TCL  F={mean(PHASE10_TCL_FORG):.4f}±{std(PHASE10_TCL_FORG):.4f}  "
    f"A={mean(PHASE10_TCL_ACC):.4f}±{std(PHASE10_TCL_ACC):.4f}  "
    f"JAF={mean(tcl_j):.4f}±{std(tcl_j):.4f}"
)


# ── collapse summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"COLLAPSE SUMMARY (acc ≤ {COLLAPSE_THRESHOLD})")
print(f"{'='*70}")
for c in C_VALUES:
    n_col = agg[c]["n_collapsed"]
    seeds_col = [
        SEEDS[i] for i, a in enumerate(results[c]["accuracy"]) if collapsed(a)
    ]
    print(f"  c={c}:  {n_col}/3 collapsed  seeds={seeds_col}")

any_non_degenerate = any(
    not collapsed(a)
    for c in C_VALUES
    for a in results[c]["accuracy"]
)
all_degenerate = not any_non_degenerate


# ── verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SI ROBUSTNESS VERDICT")
print(f"{'='*70}")

# Count non-degenerate configs
non_degen_configs = [
    (c, seed, results[c]["accuracy"][i])
    for c in C_VALUES
    for i, seed in enumerate(SEEDS)
    if not collapsed(results[c]["accuracy"][i])
]

if all_degenerate:
    verdict_key = "ALL_DEGENERATE"
    verdict = (
        f"All tested SI configurations collapse to acc ≤ {COLLAPSE_THRESHOLD}. "
        f"SI degeneracy claim holds across c∈{{{','.join(str(c) for c in C_VALUES)}}}, "
        f"ξ={XI_FIXED}. The collapse is not specific to the published default (c=0.1)."
    )
elif len(non_degen_configs) < 3:
    # Some but not all non-degenerate
    verdict_key = "PARTIAL_RECOVERY"
    verdict = (
        f"Some SI configurations avoid collapse: "
        f"{[(c, s) for c, s, _ in non_degen_configs]}. "
        f"SI degeneracy claim must be scoped to specific hyperparameter settings. "
        f"Non-degenerate configurations exist on this benchmark."
    )
else:
    # Most or all non-degenerate
    verdict_key = "FULL_RECOVERY"
    verdict = (
        f"Most tested SI configurations avoid collapse. "
        f"{len(non_degen_configs)}/{len(C_VALUES)*len(SEEDS)} seed-config pairs non-degenerate. "
        f"Degeneracy is specific to certain (c, ξ) combinations; "
        f"the claim must be narrowed to the published default hyperparameters."
    )

print(f"\n{verdict_key}: {verdict}")


# ── write result ──────────────────────────────────────────────────────────────
completed_at = datetime.utcnow().isoformat()
payload = {
    "backbone":           BACKBONE,
    "epochs":             EPOCHS,
    "seeds":              SEEDS,
    "c_values":           C_VALUES,
    "xi_fixed":           XI_FIXED,
    "collapse_threshold": COLLAPSE_THRESHOLD,
    "phase10_tcl_ref": {
        "seeds":              SEEDS,
        "forgetting_per_seed": PHASE10_TCL_FORG,
        "acc_per_seed":        PHASE10_TCL_ACC,
    },
    "per_seed":    per_seed,
    "aggregate":   {str(k): v for k, v in agg.items()},
    "verdict_key": verdict_key,
    "verdict":     verdict,
    "completed_at": completed_at,
}
env_payload = collect_environment_snapshot(
    repo_root=Path(_repo),
    workspace=Path(workspace),
    config={
        "suite": "phase13_si_sweep",
        "base_benchmark_config": BASE_CFG.model_dump(mode="json"),
        "c_values": C_VALUES,
        "xi_fixed": XI_FIXED,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "backbone": BACKBONE,
        "epochs": EPOCHS,
    },
    trigger="manual_script",
    source_script=Path(__file__).name,
    run_started_at=run_started_at,
    run_ended_at=completed_at,
    manifest_path=str(_manifest._path),
    manifest_hash=_manifest.content_hash,
    extra={"logical_name": "phase13_si_sweep", "manifest_id": _manifest.manifest_id},
)
artifacts = write_canonical_comparison_result(
    workspace=Path(workspace),
    logical_name="phase13_si_sweep",
    payload=payload,
    env_payload=env_payload,
    phase_number=13,
    source_script=Path(__file__).name,
)

print(f"\nResult written: {artifacts['result_path']}")
print(f"Env snapshot: {artifacts['env_path']}")
print(f"Index updated: {artifacts['index_path']}")
print(f"[{completed_at}] Phase 13 SI sweep complete")
