"""
Phase 18 — Canonical / Full TCL head-to-head at the protocol of record.

WHY THIS EXISTS
---------------
Every previously published "TCL" number uses method="tcl", which is a D_PR-scaled
*uniform* L2 anchor to the last task (+ a regime-observer LR scaler) — NOT the
per-element gradient-energy importance method described in tcl.py / the paper.
The genuine algorithm (method="tcl_canonical") and the full intended algorithm
(method="tcl_full" = canonical importance AND regime-observer LR control) have
never had a full-protocol head-to-head on record. Phase 18 fixes that: it is the
first time the *named* algorithm is the *measured* algorithm.

Methods compared (same backbone, class order, seeds, epochs, optimizer):
  sgd_baseline, ewc, si, tcl (published proxy, reference), tcl_canonical, tcl_full

Protocol of record (held constant vs phase10_controlled_rerun, TCL-in-suite 0.1161):
  ResNet-18, 40 epochs/task, seeds [42,0,1,2,3], fixed class order [[0,1]..[8,9]],
  SGD lr=0.01 mom=0.9 wd=1e-4, ewc_lambda=100, si_c=0.01, si_xi=0.001.

Pre-registered primary family (Holm–Bonferroni, k=6, one-tailed Wilcoxon "less"):
  {tcl_canonical, tcl_full} x {ewc, si, sgd_baseline}  on mean_forgetting.
Secondary/exploratory (NOT in the correction family):
  tcl_full vs tcl_canonical; {tcl_canonical, tcl_full} vs tcl (proxy).

POWER: n=5 detects only d>=~1.5 at alpha=0.05 (paired, one-tailed). n=5 is a
DIRECTIONAL PILOT only; a confirmatory canonical-vs-EWC claim requires n=10-20
(see the pre-registration). Do not over-claim from 5 seeds.

EXECUTION: GPU-gated via the RAIL-3 manifest (TAR_MANIFEST_PATH). ~2.8 h/seed for
the CL suite — run on RunPod or strictly AFTER the local HPC replication finishes;
do NOT contend for the GPU with a live run.

Optional lambda sweep (canonical penalty is untuned):
  set TAR_PHASE18_LAMBDA_SWEEP=1 to sweep tcl_canonical_lambda in {0.001,0.01,0.1,1.0}
  for tcl_canonical and tcl_full (multiplies the TCL portion ~4x).
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

SEEDS    = [42, 0, 1, 2, 3]
BACKBONE = "resnet18"
EPOCHS   = 40
# Headline methods. "tcl" (proxy) included as the same-seed reference so the proxy
# vs canonical vs full contrast is direct rather than cross-referenced from phase10.
METHODS  = ["sgd_baseline", "ewc", "si", "tcl", "tcl_canonical", "tcl_full"]
HEADLINE_CANON_LAMBDA = 0.01
LAMBDA_SWEEP = [0.001, 0.01, 0.1, 1.0]
RUN_LAMBDA_SWEEP = os.environ.get("TAR_PHASE18_LAMBDA_SWEEP", "") in {"1", "true", "yes", "on"}

# Pre-registered primary comparison family (Holm–Bonferroni over k = len(PRIMARY)).
PRIMARY = [
    ("tcl_canonical", "ewc"), ("tcl_canonical", "si"), ("tcl_canonical", "sgd_baseline"),
    ("tcl_full",      "ewc"), ("tcl_full",      "si"), ("tcl_full",      "sgd_baseline"),
]
ALPHA = 0.05


def mean(v):  return sum(v) / len(v)
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))


def _base_cfg(seed: int, canon_lambda: float) -> ContinualLearningBenchmarkConfig:
    return ContinualLearningBenchmarkConfig(
        seed=seed,
        train_epochs_per_task=EPOCHS,
        ewc_lambda=100.0,        # Phase-12 best for the protocol of record
        si_c=0.01,               # Phase-13 best; schema default 0.1 collapses on CIFAR-10
        si_xi=0.001,
        tcl_governor_enabled=True,   # required so "tcl" and "tcl_full" get the regime observer
        tcl_canonical_lambda=canon_lambda,
    )


def _paired_one_tailed(deltas: list[float]) -> dict:
    """One-tailed paired test that `method` has LOWER forgetting (deltas < 0)."""
    m = mean(deltas)
    d = abs(m) / max(std(deltas), 1e-12)
    n_better = sum(1 for x in deltas if x < 0)
    # Wilcoxon signed-rank (primary, non-parametric); fall back gracefully.
    try:
        if all(abs(x) < 1e-12 for x in deltas):
            raise ValueError("all-zero deltas")
        w_stat, p_w = _scipy_stats.wilcoxon(deltas, alternative="less")
        p_nonparametric = float(p_w)
    except Exception:
        p_nonparametric = float("nan")
    # Paired t (secondary).
    try:
        t_stat, p_two = _scipy_stats.ttest_1samp(deltas, 0.0)
        p_t = float(p_two) / 2.0 if mean(deltas) < 0 else 1.0 - float(p_two) / 2.0
    except Exception:
        t_stat, p_t = float("nan"), float("nan")
    return {
        "mean_delta": m, "cohens_d": d, "n_better": n_better, "n": len(deltas),
        "p_wilcoxon_1t": p_nonparametric, "t_stat": float(t_stat), "p_t_1t": p_t,
    }


def _holm_bonferroni(pairs_p: list[tuple[str, float]], alpha: float) -> dict:
    """Holm step-down. Returns {label: {p, adj_threshold, significant}}."""
    ranked = sorted(
        [(lbl, p) for lbl, p in pairs_p if not math.isnan(p)],
        key=lambda kv: kv[1],
    )
    k = len(pairs_p)
    out: dict[str, dict] = {}
    still_significant = True
    for i, (lbl, p) in enumerate(ranked):
        thr = alpha / (k - i)
        sig = still_significant and (p <= thr)
        if not sig:
            still_significant = False
        out[lbl] = {"p": p, "adj_threshold": thr, "significant": sig}
    for lbl, p in pairs_p:
        if lbl not in out:
            out[lbl] = {"p": p, "adj_threshold": float("nan"), "significant": False}
    return out


def _require_manifest() -> tuple[Path, Any]:
    manifest_path_str = os.environ.get("TAR_MANIFEST_PATH", "")
    if not manifest_path_str:
        print("REFUSED: TAR_MANIFEST_PATH not set. Phase 18 is GPU-gated; run on RunPod or "
              "AFTER the local HPC replication completes. Set TAR_MANIFEST_PATH to a signed "
              "manifest authorising 'phase18_tcl_canonical_fullprotocol' and re-run.", flush=True)
        sys.exit(1)
    manifest_path = Path(manifest_path_str)
    if not manifest_path.is_absolute():
        manifest_path = Path(_repo) / manifest_path
    try:
        manifest = load_and_verify_manifest(manifest_path, Path(_repo))
        for experiment_id in ("phase18_tcl_canonical_fullprotocol", "phase18-tcl-canonical-fullprotocol"):
            try:
                manifest.assert_experiment_authorised(experiment_id)
                print(f"[RAIL 3] Manifest gate: OK ({manifest.manifest_id})", flush=True)
                return manifest_path, manifest
            except ManifestGateError:
                continue
        raise ManifestGateError(
            "Manifest does not authorise 'phase18_tcl_canonical_fullprotocol'."
        )
    except ManifestGateError as exc:
        write_refuse_note(
            Path(workspace),
            component="phase18_tcl_canonical_fullprotocol",
            reason=str(exc),
            experiment_id="phase18_tcl_canonical_fullprotocol",
            manifest_path=str(manifest_path),
        )
        print(f"REFUSED: {exc}", flush=True)
        sys.exit(1)


_manifest_path, _manifest = _require_manifest()

run_started_at = datetime.utcnow().isoformat()
print(f"\n{'='*70}")
print(f"Phase 18 — Canonical / Full TCL head-to-head (protocol of record)")
print(f"backbone={BACKBONE}  epochs={EPOCHS}  seeds={SEEDS}")
print(f"methods={METHODS}  canon_lambda(headline)={HEADLINE_CANON_LAMBDA}")
print(f"lambda_sweep={'ON ' + str(LAMBDA_SWEEP) if RUN_LAMBDA_SWEEP else 'OFF'}")
print(f"{run_started_at}")
print(f"{'='*70}", flush=True)

per_seed: list[dict] = []
forgetting: dict[str, list[float]] = {m: [] for m in METHODS}
accuracy:   dict[str, list[float]] = {m: [] for m in METHODS}

for seed in SEEDS:
    print(f"\n--- seed={seed} ---", flush=True)
    cfg = _base_cfg(seed, HEADLINE_CANON_LAMBDA)
    row = {"seed": seed}
    for method in METHODS:
        r = run_split_cifar10_benchmark(cfg, method=method, workspace=workspace, backbone=BACKBONE)
        row[f"{method}_forgetting"] = r.mean_forgetting
        row[f"{method}_acc"]        = r.final_mean_accuracy
        forgetting[method].append(r.mean_forgetting)
        accuracy[method].append(r.final_mean_accuracy)
        print(f"  {method:14s}  forgetting={r.mean_forgetting:.4f}  acc={r.final_mean_accuracy:.4f}", flush=True)
    per_seed.append(row)

# ── aggregate ─────────────────────────────────────────────────────────────────
agg = {
    m: {
        "forgetting_mean": mean(forgetting[m]), "forgetting_std": std(forgetting[m]),
        "acc_mean": mean(accuracy[m]), "acc_std": std(accuracy[m]),
    }
    for m in METHODS
}
print(f"\n{'='*70}\nAGGREGATE (mean_forgetting, lower is better)\n{'='*70}")
for m in METHODS:
    print(f"  {m:14s}  forgetting={agg[m]['forgetting_mean']:.4f}±{agg[m]['forgetting_std']:.4f}  "
          f"acc={agg[m]['acc_mean']:.4f}±{agg[m]['acc_std']:.4f}")

# ── primary comparisons (Holm–Bonferroni, k=6) ────────────────────────────────
primary_stats: dict[str, dict] = {}
pairs_p: list[tuple[str, float]] = []
for method, baseline in PRIMARY:
    deltas = [a - b for a, b in zip(forgetting[method], forgetting[baseline])]
    s = _paired_one_tailed(deltas)
    label = f"{method}_vs_{baseline}"
    primary_stats[label] = s
    pairs_p.append((label, s["p_wilcoxon_1t"]))
holm = _holm_bonferroni(pairs_p, ALPHA)
for label, s in primary_stats.items():
    s["holm"] = holm.get(label, {})

print(f"\n{'='*70}\nPRIMARY (Holm–Bonferroni, k={len(PRIMARY)}, one-tailed Wilcoxon 'less')\n{'='*70}")
for label, s in primary_stats.items():
    h = s["holm"]
    mark = "SIG" if h.get("significant") else "ns "
    print(f"  [{mark}] {label:30s} delta={s['mean_delta']:+.4f}  p={s['p_wilcoxon_1t']:.4f}  "
          f"d={s['cohens_d']:.2f}  {s['n_better']}/{s['n']}  thr={h.get('adj_threshold', float('nan')):.4f}")

# ── secondary / exploratory (not in the correction family) ────────────────────
secondary: dict[str, dict] = {}
for method, ref in [("tcl_full", "tcl_canonical"), ("tcl_canonical", "tcl"), ("tcl_full", "tcl")]:
    deltas = [a - b for a, b in zip(forgetting[method], forgetting[ref])]
    secondary[f"{method}_vs_{ref}"] = _paired_one_tailed(deltas)

# ── optional lambda sweep ─────────────────────────────────────────────────────
lambda_sweep_results: dict = {}
if RUN_LAMBDA_SWEEP:
    print(f"\n{'='*70}\nLAMBDA SWEEP (tcl_canonical, tcl_full)\n{'='*70}")
    for lam in LAMBDA_SWEEP:
        if abs(lam - HEADLINE_CANON_LAMBDA) < 1e-12:
            continue  # already in headline
        for m in ("tcl_canonical", "tcl_full"):
            vals: list[float] = []
            for seed in SEEDS:
                cfg = _base_cfg(seed, lam)
                r = run_split_cifar10_benchmark(cfg, method=m, workspace=workspace, backbone=BACKBONE)
                vals.append(r.mean_forgetting)
            lambda_sweep_results[f"{m}@{lam}"] = {"forgetting_mean": mean(vals), "forgetting_std": std(vals), "values": vals}
            print(f"  {m:14s} lambda={lam:<6}  forgetting={mean(vals):.4f}±{std(vals):.4f}", flush=True)

# ── verdict (honest, Bonferroni-aware) ────────────────────────────────────────
canon_vs_ewc = primary_stats["tcl_canonical_vs_ewc"]
full_vs_ewc  = primary_stats["tcl_full_vs_ewc"]
def _beats_ewc(s):  # Holm-significant AND directional AND non-trivial effect
    return s["holm"].get("significant") and s["mean_delta"] < -0.01 and s["cohens_d"] > 0.5
if _beats_ewc(full_vs_ewc) or _beats_ewc(canon_vs_ewc):
    who = "tcl_full" if _beats_ewc(full_vs_ewc) else "tcl_canonical"
    verdict = (f"OUTCOME A — {who} beats EWC at full protocol with Holm-corrected significance. "
               f"The named TCL algorithm (not the L2 proxy) is competitive. NOTE n={len(SEEDS)} pilot; "
               f"confirm at pre-registered n=10-20 before publication.")
elif full_vs_ewc["mean_delta"] < 0 or canon_vs_ewc["mean_delta"] < 0:
    verdict = (f"OUTCOME B — directional only: tcl_full delta_vs_ewc={full_vs_ewc['mean_delta']:+.4f} "
               f"(p={full_vs_ewc['p_wilcoxon_1t']:.4f}), tcl_canonical={canon_vs_ewc['mean_delta']:+.4f}. "
               f"Not Holm-significant at n={len(SEEDS)}. Scale seeds before any claim.")
else:
    verdict = (f"OUTCOME C — the canonical/full TCL does NOT beat EWC at full protocol "
               f"(tcl_full delta_vs_ewc={full_vs_ewc['mean_delta']:+.4f}). The thermodynamic mechanism, "
               f"as implemented, is not superior to importance-weighted EWC here. Report honestly.")
print(f"\n{'='*70}\nVERDICT\n{'='*70}\n{verdict}")

# ── write result ──────────────────────────────────────────────────────────────
completed_at = datetime.utcnow().isoformat()
payload = {
    "backbone": BACKBONE, "epochs": EPOCHS, "seeds": SEEDS, "methods": METHODS,
    "headline_canonical_lambda": HEADLINE_CANON_LAMBDA,
    "per_seed": per_seed, "aggregate": agg,
    "primary": primary_stats, "secondary": secondary,
    "lambda_sweep": lambda_sweep_results,
    "correction": {"method": "holm_bonferroni", "k": len(PRIMARY), "alpha": ALPHA},
    "verdict": verdict, "completed_at": completed_at,
    "power_note": "n=5 directional pilot; confirmatory claim requires n=10-20 (see preregistration).",
}
env_payload = collect_environment_snapshot(
    repo_root=Path(_repo),
    workspace=Path(workspace),
    config={
        "suite": "phase18_tcl_canonical_fullprotocol",
        "base_benchmark_config": _base_cfg(SEEDS[0], HEADLINE_CANON_LAMBDA).model_dump(mode="json"),
        "methods": METHODS, "backbone": BACKBONE, "epochs": EPOCHS,
        "primary_family": [f"{m}_vs_{b}" for m, b in PRIMARY],
        "lambda_sweep": LAMBDA_SWEEP if RUN_LAMBDA_SWEEP else [],
    },
    trigger="manual_script",
    source_script=Path(__file__).name,
    run_started_at=run_started_at,
    run_ended_at=completed_at,
    extra={"logical_name": "phase18_tcl_canonical_fullprotocol"},
)
artifacts = write_canonical_comparison_result(
    workspace=Path(workspace),
    logical_name="phase18_tcl_canonical_fullprotocol",
    payload=wrap_verdict_separation(payload),
    env_payload=env_payload,
    phase_number=18,
    source_script=Path(__file__).name,
)
print(f"\nResult written: {artifacts['result_path']}")
print(f"Env snapshot: {artifacts['env_path']}")
print(f"[{completed_at}] Phase 18 complete")
