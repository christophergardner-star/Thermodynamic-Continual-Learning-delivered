"""
Phase 2 Task 2.1 -- Pre-registered HPC Replication (n=20 new seeds)

Replicates the high_penalty_conservative (HPC) finding from the autonomous
research suite. The original result (p=0.018, d=1.727) emerged from a
5-hypothesis simultaneous battery at uncorrected alpha=0.05 -- false discovery
rate ~22.6%. This confirmatory replication uses 20 NEW seeds [9..28] as a
single pre-registered hypothesis test with no multiple-comparison correction.

Pre-registration: tar_state/preregistrations/hpc_replication.json
Original result:  tar_state/autonomous_research/high_penalty_conservative.json

SPRT boundary is checked after every 4 seeds using Wald's sequential test.
Early stopping is possible: if decision='accept_H1' or 'accept_H0', halt.

Usage: python run_hpc_replication.py [--dry-run]
  --dry-run: Print what would be run without executing.

Requires: execution_enabled.flag to exist in tar_state/
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent / "Thermodynamic-Continual-Learning-delivered"
_TAR_STATE = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state")

sys.path.insert(0, str(_REPO))

# ── Experiment configuration ──────────────────────────────────────────────────

REPLICATION_SEEDS = list(range(9, 29))   # [9, 10, ..., 28] — 20 fresh seeds
ORIGINAL_SEEDS = [42, 0, 1, 2, 3]        # already run — DO NOT re-run
DATASET = "split_cifar10"
BACKBONE = "resnet18"
EPOCHS = 40
N_TASKS = 5
CLASSES_PER_TASK = 2          # 5 tasks × 2 classes = 10 classes (CIFAR-10)
BATCH_SIZE = 64
BASE_LR = 0.01

HPC_CONFIG = {
    "tcl_penalty_lambda": 0.05,      # 5× normal 0.01
    "tcl_ordered_lr_scale": 0.3,
    "tcl_alpha": 0.45,
    "tcl_reset_on_task_boundary": True,
    "tcl_governor_enabled": False,   # Governor disabled (Phase 0.10)
}

BASELINE_CONFIG = {
    "tcl_penalty_lambda": 0.01,
    "tcl_ordered_lr_scale": 0.5,
    "tcl_alpha": 0.5,
    "tcl_reset_on_task_boundary": True,
    "tcl_governor_enabled": False,
}

SPRT_ALPHA = 0.05
SPRT_BETA = 0.10
SPRT_CHECK_INTERVAL = 4   # Check boundary every N seeds

PREREG_FILE = _TAR_STATE / "preregistrations" / "hpc_replication.json"
EXEC_FLAG = _TAR_STATE / "execution_enabled.flag"
CHECKPOINT_FILE = _TAR_STATE / "comparisons" / "hpc_replication_checkpoint.json"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# ── Prerequisite checks ───────────────────────────────────────────────────────

def _check_prerequisites(dry_run: bool) -> None:
    """Abort with a clear message if prerequisites are not met."""
    missing = []
    if not PREREG_FILE.exists():
        missing.append(f"Pre-registration file not found: {PREREG_FILE}")
    if not EXEC_FLAG.exists():
        missing.append(f"Execution flag not found: {EXEC_FLAG}")
    if missing:
        print("ABORTED — prerequisites not met:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        sys.exit(1)
    if dry_run:
        print("[DRY RUN] Prerequisites OK. Would run the following:", flush=True)
        print(f"  Seeds:    {REPLICATION_SEEDS}", flush=True)
        print(f"  Dataset:  {DATASET}  backbone={BACKBONE}  epochs={EPOCHS}", flush=True)
        print(f"  HPC:      lambda={HPC_CONFIG['tcl_penalty_lambda']}", flush=True)
        print(f"  Baseline: lambda={BASELINE_CONFIG['tcl_penalty_lambda']}", flush=True)
        print(f"  SPRT: alpha={SPRT_ALPHA}  beta={SPRT_BETA}  interval={SPRT_CHECK_INTERVAL}", flush=True)
        sys.exit(0)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _build_cifar10_tasks(seed: int):
    """
    Build Split-CIFAR-10: 5 tasks, each containing 2 of CIFAR-10's 10 classes.

    Returns (train_subsets, test_subsets) where each element is a list of 5
    torch.utils.data.Subset objects with remapped labels [0, 1].
    """
    import torch
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import Subset

    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    cache_dir = str(_REPO.parent / "dataset_artifacts" / "cifar10")
    train_full = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                               download=True, transform=train_tf)
    test_full = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                              download=True, transform=test_tf)

    rng = random.Random(seed)
    all_classes = list(range(10))
    rng.shuffle(all_classes)
    class_order = [all_classes[i * CLASSES_PER_TASK:(i + 1) * CLASSES_PER_TASK]
                   for i in range(N_TASKS)]

    # Wrap dataset to remap labels to [0, CLASSES_PER_TASK)
    class _RemappedSubset:
        """Dataset view that remaps class labels to task-local indices."""
        def __init__(self, base_dataset, indices, label_map):
            self._base = base_dataset
            self._indices = indices
            self._label_map = label_map

        def __len__(self):
            return len(self._indices)

        def __getitem__(self, idx):
            img, label = self._base[self._indices[idx]]
            return img, self._label_map[int(label)]

    train_subsets = []
    test_subsets = []
    train_targets = [int(y) for y in train_full.targets]
    test_targets = [int(y) for y in test_full.targets]

    for task_classes in class_order:
        label_map = {orig: local for local, orig in enumerate(task_classes)}
        task_set = set(task_classes)
        tr_idx = [i for i, y in enumerate(train_targets) if y in task_set]
        te_idx = [i for i, y in enumerate(test_targets) if y in task_set]
        train_subsets.append(_RemappedSubset(train_full, tr_idx, label_map))
        test_subsets.append(_RemappedSubset(test_full, te_idx, label_map))

    return train_subsets, test_subsets


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_model(device):
    """
    Build a ResNet-18 trunk + per-task linear heads for Split-CIFAR-10.

    ResNet-18 is modified for 32x32 inputs: conv1 becomes 3x3 stride-1,
    maxpool is removed (Identity), matching standard CL practice.
    """
    import torch.nn as nn
    import torchvision.models as models

    class _ResNet18Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            rn = models.resnet18(weights=None)
            rn.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            rn.maxpool = nn.Identity()
            self.features = nn.Sequential(*list(rn.children())[:-1])
            self.feat_dim = 512

        def forward(self, x):
            return self.features(x).flatten(1)

    trunk = _ResNet18Trunk().to(device)
    heads = nn.ModuleList([
        nn.Linear(trunk.feat_dim, CLASSES_PER_TASK)
        for _ in range(N_TASKS)
    ]).to(device)
    return trunk, heads


# ── Training loop ─────────────────────────────────────────────────────────────

def _run_one_config(
    seed: int,
    config: dict,
    train_subsets: list,
    test_subsets: list,
    label: str,
) -> float:
    """
    Train TCL on Split-CIFAR-10 for one seed with the given config.

    Returns average forgetting across tasks 0..N_TASKS-2 (forgetting is
    undefined for the last task since there is no subsequent task to forget).

    The training loop:
      - Per-task: fresh ThermalImportance, accumulate after every backward()
      - After each task: memory.commit() snapshots weights + importance
      - For tasks > 0: TCLRegularizer.penalty() is added and backpropagated
        separately (task-loss backward first, then reg backward)
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from tcl import ThermalImportance, ThermalMemory, TCLRegularizer

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk, heads = _build_model(device)

    penalty_lambda = config["tcl_penalty_lambda"]
    ordered_lr_scale = config["tcl_ordered_lr_scale"]

    memory = ThermalMemory(max_tasks=N_TASKS, task_decay=1.0, anneal_rate=1.0)

    # accuracy_matrix[train_t][eval_t] = accuracy after training task train_t,
    # evaluated on task eval_t's test set.
    accuracy_matrix: Dict[int, Dict[int, float]] = {}

    for train_t in range(N_TASKS):
        importance = ThermalImportance(trunk, ema_beta=0.99)
        regularizer = TCLRegularizer(memory, lambda_tcl=penalty_lambda)

        all_params = list(trunk.parameters()) + list(heads[train_t].parameters())
        lr = BASE_LR * ordered_lr_scale
        optimizer = torch.optim.SGD(all_params, lr=lr, momentum=0.9,
                                    weight_decay=1e-4)

        loader = DataLoader(train_subsets[train_t], batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)

        for epoch in range(EPOCHS):
            trunk.train()
            heads[train_t].train()
            epoch_loss = 0.0
            n_batches = 0

            for bx, by in loader:
                bx = bx.to(device)
                by = torch.tensor([int(y) for y in by], dtype=torch.long,
                                  device=device)

                optimizer.zero_grad()

                # Forward + task loss
                reps = trunk(bx)
                logits = heads[train_t](reps)
                task_loss = F.cross_entropy(logits, by)
                task_loss.backward()

                # Accumulate importance from task-specific gradients BEFORE step
                importance.accumulate(trunk)

                # Forgetting protection: elastic penalty (only for tasks > 0)
                if memory.num_tasks > 0:
                    reg_loss = regularizer.penalty(trunk, device=device)
                    reg_loss.backward()

                optimizer.step()
                memory.anneal_all()

                epoch_loss += float(task_loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  [{label}] seed={seed} task={train_t+1}/{N_TASKS}"
                    f" epoch={epoch+1}/{EPOCHS} loss={avg_loss:.4f}",
                    flush=True,
                )

        # Commit task to memory after full training
        memory.commit(trunk, importance, task_id=train_t)

        # Optionally reset optimizer state on task boundary
        if config.get("tcl_reset_on_task_boundary", True):
            # Rebuild optimizer — effectively resets momentum buffers
            del optimizer
            optimizer = torch.optim.SGD(
                list(trunk.parameters()) + list(heads[train_t].parameters()),
                lr=lr, momentum=0.9, weight_decay=1e-4,
            )

        # Evaluate all seen tasks after this task completes
        trunk.eval()
        row: Dict[int, float] = {}
        for eval_t in range(N_TASKS):
            eval_loader = DataLoader(test_subsets[eval_t], batch_size=256,
                                     shuffle=False, num_workers=0)
            correct = 0
            total = 0
            with torch.no_grad():
                for bx, by in eval_loader:
                    bx = bx.to(device)
                    by_t = torch.tensor([int(y) for y in by], dtype=torch.long,
                                        device=device)
                    logits = heads[eval_t](trunk(bx))
                    correct += int((logits.argmax(1) == by_t).sum())
                    total += by_t.size(0)
            row[eval_t] = correct / max(total, 1)
        accuracy_matrix[train_t] = row

        seen_accs = [f"{accuracy_matrix[train_t][t]:.3f}"
                     for t in range(train_t + 1)]
        print(f"  [{label}] seed={seed} after task {train_t}: accs={seen_accs}",
              flush=True)

    # Compute average forgetting: mean over tasks 0..N_TASKS-2
    # Forgetting_t = peak_acc_t - final_acc_t
    forgetting_per_task = []
    for t in range(N_TASKS - 1):
        peak = max(accuracy_matrix[step][t]
                   for step in range(t, N_TASKS))
        final = accuracy_matrix[N_TASKS - 1][t]
        forgetting_per_task.append(max(0.0, peak - final))

    avg_forgetting = sum(forgetting_per_task) / max(len(forgetting_per_task), 1)

    # Free GPU memory
    del trunk, heads, memory, importance, regularizer
    try:
        import torch as _torch
        _torch.cuda.empty_cache()
    except Exception:
        pass

    return avg_forgetting


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"per_seed_results": [], "sprt_log": [], "seeds_run": 0}


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(CHECKPOINT_FILE)


# ── Statistics helpers ────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _import_stat_utils():
    """Import stat_utils from tar_lab, inserting the repo root onto sys.path."""
    import importlib
    # Ensure the tar_lab package is importable
    tar_lab_parent = str(_REPO)
    if tar_lab_parent not in sys.path:
        sys.path.insert(0, tar_lab_parent)
    return importlib.import_module("tar_lab.stat_utils")


# ── Final analysis ────────────────────────────────────────────────────────────

def _run_final_analysis(
    per_seed_results: List[dict],
    sprt_log: List[dict],
    sprt_final_decision: str,
) -> dict:
    """
    Run compare_methods_paired, bayesian_evidence, power_analysis on all
    completed seeds. Determine REPLICATION_SUCCESS / FALSE_POSITIVE / BORDERLINE.
    """
    su = _import_stat_utils()

    hpc_vals = [r["hpc_forgetting"] for r in per_seed_results]
    baseline_vals = [r["baseline_forgetting"] for r in per_seed_results]
    deltas = [r["delta"] for r in per_seed_results]  # hpc - baseline
    n = len(hpc_vals)

    # Paired comparison: HPC (a) vs baseline (b), alternative='less'
    # H1: HPC forgetting < baseline forgetting
    cmp = su.compare_methods_paired(hpc_vals, baseline_vals, alternative="less")

    # Bayesian evidence
    bayes = su.bayesian_evidence(deltas)

    # Power analysis using observed Cohen's d
    power = su.power_analysis(abs(cmp.cohens_d), n_seeds=n, alpha=SPRT_ALPHA)

    wilcoxon_p = cmp.p_nonparametric
    cohens_d = cmp.cohens_d

    # Verdict logic
    # Success criterion: p < 0.05 AND |d| >= 0.5 (medium effect)
    # Failure criterion: p > 0.10 at n=20
    # Borderline: 0.05 <= p <= 0.10, or p < 0.05 but |d| < 0.5
    if wilcoxon_p < 0.05 and abs(cohens_d) >= 0.5:
        verdict = "REPLICATION_SUCCESS"
        honest_detail = (
            f"Confirmatory replication succeeded: Wilcoxon p={wilcoxon_p:.4f} < 0.05 "
            f"and Cohen's d={cohens_d:.3f} >= 0.5 (medium effect). "
            f"HPC reduces forgetting vs baseline (n={n} seeds, single pre-registered test, "
            f"Bonferroni k=1). "
            f"Bayesian P(HPC better)={bayes.posterior_p_better:.3f}."
        )
    elif wilcoxon_p > 0.10 and n >= 20:
        verdict = "FALSE_POSITIVE"
        honest_detail = (
            f"Original finding does not replicate: Wilcoxon p={wilcoxon_p:.4f} > 0.10 "
            f"at n={n} seeds. The original result (p=0.018, d=1.727) from a "
            f"5-hypothesis battery (FDR ~22.6%) appears to have been a false positive. "
            f"Cohen's d={cohens_d:.3f}."
        )
    elif wilcoxon_p < 0.05 and abs(cohens_d) < 0.5:
        verdict = "BORDERLINE"
        honest_detail = (
            f"Statistically significant (p={wilcoxon_p:.4f}) but small effect "
            f"(d={cohens_d:.3f} < 0.5). Effect may be real but practically negligible. "
            f"n={n} seeds."
        )
    else:
        verdict = "BORDERLINE"
        honest_detail = (
            f"Inconclusive at n={n}: p={wilcoxon_p:.4f} (threshold 0.05/0.10), "
            f"d={cohens_d:.3f}. Neither confirms nor refutes the original finding."
        )

    analysis = {
        "hpc_forgetting_mean": _mean(hpc_vals),
        "hpc_forgetting_std": _std(hpc_vals),
        "baseline_forgetting_mean": _mean(baseline_vals),
        "baseline_forgetting_std": _std(baseline_vals),
        "mean_delta": _mean(deltas),
        "wilcoxon_p": wilcoxon_p,
        "wilcoxon_statistic": cmp.statistic_nonparametric,
        "paired_t_p": cmp.p_parametric,
        "cohens_d": cohens_d,
        "bonferroni_k": 1,
        "bayesian_p_hpc_better": bayes.posterior_p_better,
        "bayesian_posterior_mean_delta": bayes.posterior_mean_delta,
        "bayesian_ci_95": list(bayes.credible_interval_95),
        "achieved_power": power.achieved_power,
        "seeds_needed_80pct": power.seeds_needed_80pct,
        "verdict": verdict,
        "success_criterion": "p<0.05 AND d>=0.5",
        "failure_criterion": "p>0.10 at n=20",
        "honest_verdict_detail": honest_detail,
        "sprt_final_decision": sprt_final_decision,
    }
    return analysis


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Task 2.1 -- Pre-registered HPC Replication (n=20)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without executing.")
    args = parser.parse_args()

    # 1. Prerequisites
    _check_prerequisites(dry_run=args.dry_run)

    su = _import_stat_utils()

    # 2. Load pre-registration for reference (informational only at runtime)
    with open(PREREG_FILE, "r", encoding="utf-8") as f:
        prereg = json.load(f)
    print(f"[HPC REPLICATION] Pre-registration loaded: {PREREG_FILE}", flush=True)
    print(f"[HPC REPLICATION] Starting n=20 seed replication.", flush=True)
    print(f"  Seeds: {REPLICATION_SEEDS}", flush=True)
    print(f"  HPC lambda={HPC_CONFIG['tcl_penalty_lambda']}  "
          f"Baseline lambda={BASELINE_CONFIG['tcl_penalty_lambda']}", flush=True)
    print(f"  SPRT check every {SPRT_CHECK_INTERVAL} seeds "
          f"(alpha={SPRT_ALPHA}, beta={SPRT_BETA})", flush=True)
    print("=" * 70, flush=True)

    # 3. Resume from checkpoint if available
    state = _load_checkpoint()
    completed_seeds = {r["seed"] for r in state["per_seed_results"]}
    per_seed_results: List[dict] = list(state["per_seed_results"])
    sprt_log: List[dict] = list(state["sprt_log"])
    sprt_final_decision = "continue"

    if completed_seeds:
        print(f"[RESUME] Found {len(completed_seeds)} completed seeds: "
              f"{sorted(completed_seeds)}", flush=True)

    # 4. Seed loop
    for seed in REPLICATION_SEEDS:
        if seed in completed_seeds:
            print(f"\n--- seed={seed} --- [SKIPPED: already in checkpoint]",
                  flush=True)
            continue

        # Check if SPRT already stopped
        if sprt_final_decision in ("accept_H1", "accept_H0"):
            print(f"\n[SPRT] Early stop decision={sprt_final_decision} already reached. "
                  f"Skipping remaining seeds.", flush=True)
            break

        print(f"\n{'='*60}", flush=True)
        print(f"--- seed={seed} ---", flush=True)

        # Load dataset for this seed
        print(f"  Building Split-CIFAR-10 tasks (seed={seed})...", flush=True)
        train_subsets, test_subsets = _build_cifar10_tasks(seed)

        # Run baseline
        print(f"  Running BASELINE (lambda={BASELINE_CONFIG['tcl_penalty_lambda']})...",
              flush=True)
        baseline_forgetting = _run_one_config(
            seed, BASELINE_CONFIG, train_subsets, test_subsets, label="BASELINE"
        )

        # Run HPC
        print(f"  Running HPC (lambda={HPC_CONFIG['tcl_penalty_lambda']})...",
              flush=True)
        hpc_forgetting = _run_one_config(
            seed, HPC_CONFIG, train_subsets, test_subsets, label="HPC"
        )

        delta = hpc_forgetting - baseline_forgetting
        seed_result = {
            "seed": seed,
            "hpc_forgetting": hpc_forgetting,
            "baseline_forgetting": baseline_forgetting,
            "delta": delta,
        }
        per_seed_results.append(seed_result)
        completed_seeds.add(seed)

        print(
            f"  seed={seed}  hpc_forgetting={hpc_forgetting:.4f}  "
            f"baseline_forgetting={baseline_forgetting:.4f}  delta={delta:+.4f}",
            flush=True,
        )

        # 5. Write checkpoint after every seed
        state = {
            "per_seed_results": per_seed_results,
            "sprt_log": sprt_log,
            "seeds_run": len(per_seed_results),
        }
        _save_checkpoint(state)

        # 6. SPRT check every SPRT_CHECK_INTERVAL seeds
        n_run = len(per_seed_results)
        if n_run % SPRT_CHECK_INTERVAL == 0 and n_run > 0:
            # Compute a per-seed p-value proxy: sign(delta) mapped to a
            # Bernoulli p-value.  A proper sequential test requires p-values;
            # here we use a sign test approximation: if delta < 0 (HPC better),
            # that is one-tailed evidence. We feed a t-test p-value computed
            # from the running sample.
            hpc_vals = [r["hpc_forgetting"] for r in per_seed_results]
            baseline_vals = [r["baseline_forgetting"] for r in per_seed_results]

            if n_run >= 2:
                cmp_sprt = su.compare_methods_paired(
                    hpc_vals, baseline_vals, alternative="less"
                )
                # Build list of per-seed indicator p-values for SPRT
                # Each seed contributes: p_indicator = alpha if evidence for H1,
                # else (1-alpha). We approximate using running paired p-values
                # at each checkpoint. For SPRT we feed the running p-values
                # accumulated in batches.
                batch_p_values = []
                for i in range(1, n_run + 1):
                    if i < 2:
                        batch_p_values.append(0.5)
                        continue
                    sub_hpc = hpc_vals[:i]
                    sub_base = baseline_vals[:i]
                    try:
                        sub_cmp = su.compare_methods_paired(
                            sub_hpc, sub_base, alternative="less"
                        )
                        batch_p_values.append(sub_cmp.p_nonparametric)
                    except Exception:
                        batch_p_values.append(0.5)

                sprt_result = su.sprt_boundary(
                    n_seeds_run=n_run,
                    p_values_so_far=batch_p_values,
                    alpha=SPRT_ALPHA,
                    beta=SPRT_BETA,
                )
                sprt_entry = {
                    "n_seeds": n_run,
                    "log_lr": sprt_result.log_lr,
                    "decision": sprt_result.decision,
                    "A": sprt_result.A,
                    "B": sprt_result.B,
                    "running_wilcoxon_p": cmp_sprt.p_nonparametric,
                    "running_cohens_d": cmp_sprt.cohens_d,
                }
                sprt_log.append(sprt_entry)

                print(
                    f"\n  [SPRT @ n={n_run}] log_LR={sprt_result.log_lr:.3f}  "
                    f"A={sprt_result.A:.3f}  B={sprt_result.B:.3f}  "
                    f"decision={sprt_result.decision}  "
                    f"wilcoxon_p={cmp_sprt.p_nonparametric:.4f}  "
                    f"d={cmp_sprt.cohens_d:.3f}",
                    flush=True,
                )

                sprt_final_decision = sprt_result.decision
                if sprt_final_decision in ("accept_H1", "accept_H0"):
                    print(
                        f"\n[SPRT] EARLY STOP at n={n_run}: "
                        f"decision={sprt_final_decision}",
                        flush=True,
                    )
                    # Update checkpoint with SPRT log
                    state["sprt_log"] = sprt_log
                    _save_checkpoint(state)
                    break

    # 7. Final analysis
    print("\n" + "=" * 70, flush=True)
    print("[HPC REPLICATION] Running final analysis...", flush=True)

    if len(per_seed_results) < 2:
        print("ERROR: Fewer than 2 seeds completed. Cannot run analysis.", flush=True)
        sys.exit(1)

    analysis = _run_final_analysis(per_seed_results, sprt_log, sprt_final_decision)

    # 8. Write output JSON
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = _TAR_STATE / "comparisons" / f"hpc_replication_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "result_id": f"hpc_replication_{timestamp}",
        "preregistration_file": str(PREREG_FILE),
        "replication_seeds": REPLICATION_SEEDS,
        "original_seeds": ORIGINAL_SEEDS,
        "seeds_run": len(per_seed_results),
        "dataset": DATASET,
        "backbone": BACKBONE,
        "epochs": EPOCHS,
        "n_tasks": N_TASKS,
        "hpc_config": HPC_CONFIG,
        "baseline_config": BASELINE_CONFIG,
        "sprt_alpha": SPRT_ALPHA,
        "sprt_beta": SPRT_BETA,
        "sprt_check_interval": SPRT_CHECK_INTERVAL,
        "sprt_log": sprt_log,
        "per_seed_results": per_seed_results,
        "hpc_forgetting_mean": analysis["hpc_forgetting_mean"],
        "hpc_forgetting_std": analysis["hpc_forgetting_std"],
        "baseline_forgetting_mean": analysis["baseline_forgetting_mean"],
        "wilcoxon_p": analysis["wilcoxon_p"],
        "cohens_d": analysis["cohens_d"],
        "bonferroni_k": 1,
        "bayesian_p_hpc_better": analysis["bayesian_p_hpc_better"],
        "sprt_final_decision": analysis["sprt_final_decision"],
        "verdict": analysis["verdict"],
        "success_criterion": analysis["success_criterion"],
        "failure_criterion": analysis["failure_criterion"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "honest_verdict_detail": analysis["honest_verdict_detail"],
        # Extended fields
        "wilcoxon_statistic": analysis["wilcoxon_statistic"],
        "paired_t_p": analysis["paired_t_p"],
        "mean_delta": analysis["mean_delta"],
        "bayesian_posterior_mean_delta": analysis["bayesian_posterior_mean_delta"],
        "bayesian_ci_95": analysis["bayesian_ci_95"],
        "achieved_power": analysis["achieved_power"],
        "seeds_needed_80pct": analysis["seeds_needed_80pct"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # 9. Print summary
    print(f"\n[HPC REPLICATION] COMPLETE", flush=True)
    print(f"  Seeds run:       {len(per_seed_results)}", flush=True)
    print(f"  HPC forgetting:  {analysis['hpc_forgetting_mean']:.4f} "
          f"± {analysis['hpc_forgetting_std']:.4f}", flush=True)
    print(f"  Base forgetting: {analysis['baseline_forgetting_mean']:.4f} "
          f"± {analysis['baseline_forgetting_std']:.4f}", flush=True)
    print(f"  Wilcoxon p:      {analysis['wilcoxon_p']:.4f}", flush=True)
    print(f"  Cohen's d:       {analysis['cohens_d']:.3f}", flush=True)
    print(f"  Bayesian P(HPC better): {analysis['bayesian_p_hpc_better']:.3f}",
          flush=True)
    print(f"  SPRT decision:   {analysis['sprt_final_decision']}", flush=True)
    print(f"\n  VERDICT: {analysis['verdict']}", flush=True)
    print(f"  {analysis['honest_verdict_detail']}", flush=True)
    print(f"\n  Output written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
