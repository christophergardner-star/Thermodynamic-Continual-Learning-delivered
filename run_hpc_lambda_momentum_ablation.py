"""
Phase 3 Task 3.4 -- HPC Lambda vs LR-Scale Ablation

Determines whether HPC's forgetting reduction comes from:
  (a) Higher penalty lambda (0.05 vs 0.01)
  (b) Conservative LR scaling (ordered_lr_scale=0.3 vs 0.5)
  (c) Both in combination

Depends on: HPC replication (Task 2.1) showing REPLICATION_SUCCESS.
Pre-registration: tar_state/preregistrations/hpc_lambda_momentum_ablation.json

4 conditions, 5 seeds each, Split-CIFAR-10, ResNet-18.
Primary comparison: hpc vs tcl_high_lambda (p<0.0167 → LR scaling matters)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# ── Path setup ─────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent
TAR_STATE = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state")

sys.path.insert(0, str(_REPO))

# ── Experiment configuration ───────────────────────────────────────────────────

SEEDS = [42, 0, 1, 2, 3]
BONFERRONI_K = 3
BONFERRONI_THRESHOLD = 0.05 / BONFERRONI_K   # 0.0167

CONDITIONS: Dict[str, dict] = {
    "tcl_baseline": {
        "tcl_penalty_lambda": 0.01,
        "tcl_ordered_lr_scale": 0.5,
        "tcl_ema_beta": 0.99,
    },
    "tcl_high_lambda": {
        "tcl_penalty_lambda": 0.05,
        "tcl_ordered_lr_scale": 0.5,
        "tcl_ema_beta": 0.99,
    },
    "tcl_conservative_lr": {
        "tcl_penalty_lambda": 0.01,
        "tcl_ordered_lr_scale": 0.3,
        "tcl_ema_beta": 0.99,
    },
    "hpc": {
        "tcl_penalty_lambda": 0.05,
        "tcl_ordered_lr_scale": 0.3,
        "tcl_ema_beta": 0.45,   # original HPC config from spec.json
    },
}

DATASET = "split_cifar10"
BACKBONE = "resnet18"
EPOCHS = 40
N_TASKS = 5
CLASSES_PER_TASK = 2        # 5 tasks × 2 classes = 10 classes (CIFAR-10)
BATCH_SIZE = 64
BASE_LR = 0.01

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

PREREG_FILE = TAR_STATE / "preregistrations" / "hpc_lambda_momentum_ablation.json"
EXEC_FLAG = TAR_STATE / "execution_enabled.flag"
CHECKPOINT_FILE = TAR_STATE / "comparisons" / "hpc_lambda_momentum_ablation_checkpoint.json"


# ── Prerequisite checks ────────────────────────────────────────────────────────

def _check_prerequisites(dry_run: bool) -> None:
    """Abort with a clear message if prerequisites are not met."""
    missing = []

    if not PREREG_FILE.exists():
        missing.append(f"Pre-registration file not found: {PREREG_FILE}")
    if not EXEC_FLAG.exists():
        missing.append(f"Execution flag not found: {EXEC_FLAG}")

    if missing:
        print("ABORTED -- prerequisites not met:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        sys.exit(1)

    # Check HPC replication result
    hpc_replication_results = list(
        (TAR_STATE / "comparisons").glob("hpc_replication_*.json")
    )
    if not hpc_replication_results:
        print(
            "ERROR: HPC replication (Task 2.1) not complete. "
            "Run run_hpc_replication.py first.",
            flush=True,
        )
        sys.exit(1)

    latest = max(hpc_replication_results, key=lambda p: p.stat().st_mtime)
    result = json.loads(latest.read_text(encoding="utf-8"))
    if result.get("verdict") != "REPLICATION_SUCCESS":
        print(
            f"ERROR: HPC replication verdict={result.get('verdict')}. "
            "Ablation requires REPLICATION_SUCCESS.",
            flush=True,
        )
        sys.exit(1)
    print(f"HPC replication confirmed: {result.get('verdict')}  ({latest.name})",
          flush=True)

    if dry_run:
        print("\n[DRY RUN] Prerequisites OK. Would run the following:", flush=True)
        print(f"  Seeds:      {SEEDS}", flush=True)
        print(f"  Conditions: {list(CONDITIONS.keys())}", flush=True)
        print(f"  Dataset:    {DATASET}  backbone={BACKBONE}  epochs={EPOCHS}",
              flush=True)
        print(f"  Bonferroni: k={BONFERRONI_K}  threshold={BONFERRONI_THRESHOLD:.4f}",
              flush=True)
        print(f"  Total runs: {len(SEEDS) * len(CONDITIONS)}", flush=True)
        sys.exit(0)


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _build_cifar10_tasks(seed: int):
    """
    Build Split-CIFAR-10: 5 tasks, each containing 2 of CIFAR-10's 10 classes.

    Returns (train_subsets, test_subsets) where each element is a list of 5
    dataset views with remapped labels [0, 1].
    """
    import torchvision
    import torchvision.transforms as T

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
    train_full = torchvision.datasets.CIFAR10(
        root=cache_dir, train=True, download=True, transform=train_tf
    )
    test_full = torchvision.datasets.CIFAR10(
        root=cache_dir, train=False, download=True, transform=test_tf
    )

    rng = random.Random(seed)
    all_classes = list(range(10))
    rng.shuffle(all_classes)
    class_order = [
        all_classes[i * CLASSES_PER_TASK:(i + 1) * CLASSES_PER_TASK]
        for i in range(N_TASKS)
    ]

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


# ── Model helpers ──────────────────────────────────────────────────────────────

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


# ── Training loop ──────────────────────────────────────────────────────────────

def _run_one_condition(
    seed: int,
    condition_name: str,
    config: dict,
    train_subsets: list,
    test_subsets: list,
) -> float:
    """
    Train TCL on Split-CIFAR-10 for one seed under the given condition config.

    Returns average forgetting across tasks 0..N_TASKS-2.

    Notes on ordered_lr_scale:
      The governor (Phase 0.10) is disabled by default, so the "ordered" regime
      never triggers.  We apply ordered_lr_scale unconditionally as a static
      learning-rate multiplier so the ablation can test its effect explicitly,
      consistent with how the original HPC runs were configured.

    Training loop:
      - Per-task: fresh ThermalImportance; importance accumulated after every
        backward pass.
      - After each task: memory.commit() snapshots weights + importance.
      - For tasks > 0: TCLRegularizer.penalty() provides elastic forgetting
        protection (separate backward after task loss backward).
    """
    import numpy as np
    import torch
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
    ema_beta = config["tcl_ema_beta"]

    memory = ThermalMemory(max_tasks=N_TASKS, task_decay=1.0, anneal_rate=1.0)

    # accuracy_matrix[train_t][eval_t] = accuracy after training task train_t,
    # evaluated on task eval_t's test set.
    accuracy_matrix: Dict[int, Dict[int, float]] = {}

    for train_t in range(N_TASKS):
        importance = ThermalImportance(trunk, ema_beta=ema_beta)
        regularizer = TCLRegularizer(memory, lambda_tcl=penalty_lambda)

        all_params = list(trunk.parameters()) + list(heads[train_t].parameters())
        # ordered_lr_scale applied unconditionally as a static LR multiplier
        lr = BASE_LR * ordered_lr_scale
        optimizer = torch.optim.SGD(
            all_params, lr=lr, momentum=0.9, weight_decay=1e-4
        )

        loader = DataLoader(
            train_subsets[train_t], batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0,
        )

        for epoch in range(EPOCHS):
            trunk.train()
            heads[train_t].train()
            epoch_loss = 0.0
            n_batches = 0

            for bx, by in loader:
                bx = bx.to(device)
                by = torch.tensor(
                    [int(y) for y in by], dtype=torch.long, device=device
                )

                optimizer.zero_grad()

                # Forward + task loss
                reps = trunk(bx)
                logits = heads[train_t](reps)
                task_loss = F.cross_entropy(logits, by)
                task_loss.backward()

                # Accumulate importance from task-specific gradients BEFORE step
                importance.accumulate(trunk)

                # Forgetting protection: elastic penalty (tasks > 0 only)
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
                    f"  [{condition_name}] seed={seed} task={train_t+1}/{N_TASKS}"
                    f" epoch={epoch+1}/{EPOCHS} loss={avg_loss:.4f}",
                    flush=True,
                )

        # Commit task to memory after full training
        memory.commit(trunk, importance, task_id=train_t)

        # Reset optimizer state on task boundary (standard CL practice)
        del optimizer
        optimizer = torch.optim.SGD(
            list(trunk.parameters()) + list(heads[train_t].parameters()),
            lr=lr, momentum=0.9, weight_decay=1e-4,
        )

        # Evaluate all seen tasks after this task completes
        trunk.eval()
        row: Dict[int, float] = {}
        for eval_t in range(N_TASKS):
            eval_loader = DataLoader(
                test_subsets[eval_t], batch_size=256, shuffle=False, num_workers=0
            )
            correct = 0
            total = 0
            with torch.no_grad():
                for bx, by in eval_loader:
                    bx = bx.to(device)
                    by_t = torch.tensor(
                        [int(y) for y in by], dtype=torch.long, device=device
                    )
                    logits = heads[eval_t](trunk(bx))
                    correct += int((logits.argmax(1) == by_t).sum())
                    total += by_t.size(0)
            row[eval_t] = correct / max(total, 1)
        accuracy_matrix[train_t] = row

        seen_accs = [
            f"{accuracy_matrix[train_t][t]:.3f}" for t in range(train_t + 1)
        ]
        print(
            f"  [{condition_name}] seed={seed} after task {train_t}: "
            f"accs={seen_accs}",
            flush=True,
        )

    # Compute average forgetting: mean over tasks 0..N_TASKS-2
    # Forgetting_t = peak_acc_t - final_acc_t
    forgetting_per_task = []
    for t in range(N_TASKS - 1):
        peak = max(accuracy_matrix[step][t] for step in range(t, N_TASKS))
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


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"per_seed_results": [], "conditions_complete": {}}


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_FILE)


# ── Statistics helpers ─────────────────────────────────────────────────────────

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
    tar_lab_parent = str(_REPO)
    if tar_lab_parent not in sys.path:
        sys.path.insert(0, tar_lab_parent)
    return importlib.import_module("tar_lab.stat_utils")


# ── Pairwise comparison helper ─────────────────────────────────────────────────

def _compare_pair(
    su,
    a_vals: List[float],
    b_vals: List[float],
    label_a: str,
    label_b: str,
) -> dict:
    """
    Run a pre-registered paired comparison (Wilcoxon signed-rank, one-tailed
    alternative='less': H1 = a_vals < b_vals, i.e. a forgets less than b).
    """
    cmp = su.compare_methods_paired(a_vals, b_vals, alternative="less")
    p = cmp.p_nonparametric
    d = cmp.cohens_d
    sig = bool(p < BONFERRONI_THRESHOLD)
    return {
        "label_a": label_a,
        "label_b": label_b,
        "wilcoxon_p": p,
        "wilcoxon_statistic": cmp.statistic_nonparametric,
        "cohens_d": d,
        "paired_t_p": cmp.p_parametric,
        "significant_bonferroni": sig,
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
        "mean_a": _mean(a_vals),
        "mean_b": _mean(b_vals),
        "mean_delta": _mean([a - b for a, b in zip(a_vals, b_vals)]),
    }


# ── Decision logic ─────────────────────────────────────────────────────────────

def _determine_decision(comparisons: dict) -> Tuple[str, str]:
    """
    Apply the pre-registered decision rules to produce a verdict and rationale.

    Comparisons dict keys (mirrors decision_rules in pre-registration):
      primary:      hpc vs tcl_high_lambda
      lambda_iso:   tcl_high_lambda vs tcl_baseline
      lr_iso:       tcl_conservative_lr vs tcl_baseline

    Decision rules (from pre-registration):
      LAMBDA_IS_MECHANISM:
        Primary p > 0.05  →  LR scaling adds nothing on top of high lambda.
        HPC is effectively 'TCL with better lambda.'
      LR_SCALE_IS_MECHANISM:
        Primary p < BONFERRONI_THRESHOLD  →  LR scaling provides additional
        benefit beyond lambda alone.  HPC is a new combination.
      SYNERGISTIC:
        Both lambda_iso AND lr_iso individually beat baseline (p < threshold),
        AND hpc beats both (primary p < threshold).  Lambda and LR scaling
        interact synergistically.
      AMBIGUOUS:
        None of the above conditions clearly met (e.g. neither isolation
        condition beats baseline, but primary is significant).
    """
    primary = comparisons["primary_hpc_vs_high_lambda"]
    lambda_iso = comparisons["lambda_iso_high_lambda_vs_baseline"]
    lr_iso = comparisons["lr_iso_conservative_lr_vs_baseline"]

    primary_sig = primary["significant_bonferroni"]
    lambda_iso_sig = lambda_iso["significant_bonferroni"]
    lr_iso_sig = lr_iso["significant_bonferroni"]

    # Primary p-value (not Bonferroni-adjusted threshold — using raw p < 0.05
    # to evaluate the null of "no additional benefit from LR scaling")
    primary_p_raw = primary["wilcoxon_p"]

    if primary_sig and lambda_iso_sig and lr_iso_sig:
        decision = "SYNERGISTIC"
        rationale = (
            f"Both lambda isolation (p={lambda_iso['wilcoxon_p']:.4f}) and LR "
            f"isolation (p={lr_iso['wilcoxon_p']:.4f}) independently beat "
            f"baseline, AND HPC beats tcl_high_lambda "
            f"(p={primary['wilcoxon_p']:.4f} < {BONFERRONI_THRESHOLD:.4f}). "
            "Lambda and conservative LR scaling interact synergistically."
        )
    elif primary_sig:
        decision = "LR_SCALE_IS_MECHANISM"
        rationale = (
            f"HPC significantly beats tcl_high_lambda "
            f"(p={primary['wilcoxon_p']:.4f} < {BONFERRONI_THRESHOLD:.4f}, "
            f"d={primary['cohens_d']:.3f}). "
            "Conservative LR scaling contributes additional forgetting reduction "
            "beyond higher lambda alone. HPC is a novel combination."
        )
    elif primary_p_raw > 0.05:
        decision = "LAMBDA_IS_MECHANISM"
        rationale = (
            f"No significant difference between HPC and tcl_high_lambda "
            f"(p={primary['wilcoxon_p']:.4f} > 0.05). "
            "Conservative LR scaling (ordered_lr_scale=0.3) adds nothing on top "
            "of high lambda. HPC's forgetting reduction is attributable to "
            "lambda=0.05, not to LR schedule. "
            f"Lambda isolation: p={lambda_iso['wilcoxon_p']:.4f}, "
            f"d={lambda_iso['cohens_d']:.3f}."
        )
    else:
        decision = "AMBIGUOUS"
        rationale = (
            f"Primary comparison borderline (p={primary['wilcoxon_p']:.4f}, "
            f"threshold={BONFERRONI_THRESHOLD:.4f}). "
            f"Lambda isolation p={lambda_iso['wilcoxon_p']:.4f}, "
            f"LR isolation p={lr_iso['wilcoxon_p']:.4f}. "
            "Cannot attribute HPC benefit to a single mechanism with confidence. "
            "Consider n >= 10 per condition for a definitive answer."
        )

    return decision, rationale


# ── Final analysis ─────────────────────────────────────────────────────────────

def _run_final_analysis(per_seed_results: List[dict]) -> dict:
    """
    Compute per-condition summaries, all pairwise comparisons, bayesian
    evidence, and apply the pre-registered decision rules.
    """
    su = _import_stat_utils()

    # Group forgetting values by condition
    condition_forgetting: Dict[str, List[float]] = {c: [] for c in CONDITIONS}
    for r in per_seed_results:
        condition_forgetting[r["condition"]].append(r["forgetting"])

    # Per-condition summaries
    summaries = {}
    for cname, vals in condition_forgetting.items():
        summaries[cname] = {
            "n": len(vals),
            "mean_forgetting": _mean(vals),
            "std_forgetting": _std(vals),
            "values": vals,
        }

    hpc = condition_forgetting["hpc"]
    baseline = condition_forgetting["tcl_baseline"]
    high_lambda = condition_forgetting["tcl_high_lambda"]
    conservative_lr = condition_forgetting["tcl_conservative_lr"]

    # Three pre-registered pairwise comparisons (Bonferroni k=3)
    # All one-tailed: H1 = left condition forgets LESS than right
    comparisons = {
        "primary_hpc_vs_high_lambda": _compare_pair(
            su, hpc, high_lambda, "hpc", "tcl_high_lambda"
        ),
        "lambda_iso_high_lambda_vs_baseline": _compare_pair(
            su, high_lambda, baseline, "tcl_high_lambda", "tcl_baseline"
        ),
        "lr_iso_conservative_lr_vs_baseline": _compare_pair(
            su, conservative_lr, baseline, "tcl_conservative_lr", "tcl_baseline"
        ),
    }

    # Bayesian evidence for primary comparison (HPC vs high_lambda)
    primary_deltas = [a - b for a, b in zip(hpc, high_lambda)]
    bayes_primary = su.bayesian_evidence(primary_deltas)

    # Bayesian evidence for lambda isolation (high_lambda vs baseline)
    lambda_deltas = [a - b for a, b in zip(high_lambda, baseline)]
    bayes_lambda = su.bayesian_evidence(lambda_deltas)

    # Power analysis on primary comparison
    n = len(hpc)
    power_primary = su.power_analysis(
        abs(comparisons["primary_hpc_vs_high_lambda"]["cohens_d"]),
        n_seeds=n,
        alpha=0.05,
    )

    # Apply decision rules
    decision, rationale = _determine_decision(comparisons)

    return {
        "condition_summaries": summaries,
        "comparisons": comparisons,
        "decision": decision,
        "decision_rationale": rationale,
        "primary_comparison_hpc_vs_high_lambda": {
            "p": comparisons["primary_hpc_vs_high_lambda"]["wilcoxon_p"],
            "d": comparisons["primary_hpc_vs_high_lambda"]["cohens_d"],
            "significant": comparisons["primary_hpc_vs_high_lambda"]["significant_bonferroni"],
        },
        "bayesian_primary": {
            "posterior_p_hpc_better": bayes_primary.posterior_p_better,
            "posterior_mean_delta": bayes_primary.posterior_mean_delta,
            "credible_interval_95": list(bayes_primary.credible_interval_95),
        },
        "bayesian_lambda_iso": {
            "posterior_p_high_lambda_better": bayes_lambda.posterior_p_better,
            "posterior_mean_delta": bayes_lambda.posterior_mean_delta,
            "credible_interval_95": list(bayes_lambda.credible_interval_95),
        },
        "achieved_power_primary": power_primary.achieved_power,
        "seeds_needed_80pct_primary": power_primary.seeds_needed_80pct,
    }


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 Task 3.4 -- HPC Lambda vs LR-Scale Ablation (n=5 seeds, 4 conditions)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing.",
    )
    args = parser.parse_args()

    # 1. Prerequisites (includes HPC replication check and dry-run exit)
    _check_prerequisites(dry_run=args.dry_run)

    # 2. Load pre-registration for reference
    prereg = json.loads(PREREG_FILE.read_text(encoding="utf-8"))
    print(f"[ABLATION] Pre-registration loaded: {PREREG_FILE}", flush=True)
    print(f"[ABLATION] Hypothesis: {prereg.get('hypothesis', '(see file)')}", flush=True)
    print(f"[ABLATION] 4 conditions × {len(SEEDS)} seeds = {4 * len(SEEDS)} total runs",
          flush=True)
    print(f"  Conditions: {list(CONDITIONS.keys())}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Bonferroni threshold: {BONFERRONI_THRESHOLD:.4f}", flush=True)
    print("=" * 70, flush=True)

    # 3. Resume from checkpoint if available
    state = _load_checkpoint()
    per_seed_results: List[dict] = list(state.get("per_seed_results", []))
    completed = {
        (r["condition"], r["seed"]) for r in per_seed_results
    }

    if completed:
        print(f"[RESUME] Found {len(completed)} completed (condition, seed) pairs.",
              flush=True)

    # 4. Main loop: iterate conditions × seeds
    for condition_name, config in CONDITIONS.items():
        for seed in SEEDS:
            key = (condition_name, seed)
            if key in completed:
                print(
                    f"\n--- {condition_name} seed={seed} --- "
                    "[SKIPPED: already in checkpoint]",
                    flush=True,
                )
                continue

            print(f"\n{'='*60}", flush=True)
            print(
                f"--- condition={condition_name}  seed={seed} ---",
                flush=True,
            )
            print(
                f"  lambda={config['tcl_penalty_lambda']}  "
                f"ordered_lr_scale={config['tcl_ordered_lr_scale']}  "
                f"ema_beta={config['tcl_ema_beta']}",
                flush=True,
            )

            # Build dataset (seed-specific class order)
            print(f"  Building Split-CIFAR-10 tasks (seed={seed})...", flush=True)
            train_subsets, test_subsets = _build_cifar10_tasks(seed)

            # Run training
            forgetting = _run_one_condition(
                seed=seed,
                condition_name=condition_name,
                config=config,
                train_subsets=train_subsets,
                test_subsets=test_subsets,
            )

            result = {
                "condition": condition_name,
                "seed": seed,
                "forgetting": forgetting,
                "config": config,
            }
            per_seed_results.append(result)
            completed.add(key)

            print(
                f"  DONE: condition={condition_name}  seed={seed}  "
                f"forgetting={forgetting:.4f}",
                flush=True,
            )

            # Save checkpoint after every run
            state = {
                "per_seed_results": per_seed_results,
                "conditions_complete": {
                    c: sum(1 for r in per_seed_results if r["condition"] == c)
                    for c in CONDITIONS
                },
            }
            _save_checkpoint(state)

    # 5. Final analysis
    print("\n" + "=" * 70, flush=True)
    print("[ABLATION] Running final analysis...", flush=True)

    if len(per_seed_results) < len(CONDITIONS) * 2:
        print(
            f"ERROR: Only {len(per_seed_results)} runs completed; "
            "need at least 2 per condition. Cannot run analysis.",
            flush=True,
        )
        sys.exit(1)

    analysis = _run_final_analysis(per_seed_results)

    # 6. Write output JSON
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        TAR_STATE / "comparisons" / f"hpc_lambda_momentum_ablation_{timestamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "result_id": f"hpc_lambda_momentum_ablation_{timestamp}",
        "preregistration_file": str(PREREG_FILE),
        "experiment_id": "hpc_lambda_momentum_ablation_4condition",
        "seeds": SEEDS,
        "conditions": {
            name: {
                "config": cfg,
                "mean_forgetting": analysis["condition_summaries"][name]["mean_forgetting"],
                "std_forgetting": analysis["condition_summaries"][name]["std_forgetting"],
                "n": analysis["condition_summaries"][name]["n"],
                "values": analysis["condition_summaries"][name]["values"],
            }
            for name, cfg in CONDITIONS.items()
        },
        "bonferroni_k": BONFERRONI_K,
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
        "dataset": DATASET,
        "backbone": BACKBONE,
        "epochs": EPOCHS,
        "n_tasks": N_TASKS,
        # Primary output fields (required by spec)
        "decision": analysis["decision"],
        "decision_rationale": analysis["decision_rationale"],
        "primary_comparison_hpc_vs_high_lambda": analysis["primary_comparison_hpc_vs_high_lambda"],
        # Extended comparisons
        "comparisons": analysis["comparisons"],
        "bayesian_primary": analysis["bayesian_primary"],
        "bayesian_lambda_iso": analysis["bayesian_lambda_iso"],
        "achieved_power_primary": analysis["achieved_power_primary"],
        "seeds_needed_80pct_primary": analysis["seeds_needed_80pct_primary"],
        # Raw per-seed results
        "per_seed_results": per_seed_results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # 7. Print summary
    print(f"\n[ABLATION] COMPLETE", flush=True)
    print("  Condition forgetting (mean ± std):", flush=True)
    for cname in CONDITIONS:
        s = analysis["condition_summaries"][cname]
        print(
            f"    {cname:<22}  {s['mean_forgetting']:.4f} ± {s['std_forgetting']:.4f}"
            f"  (n={s['n']})",
            flush=True,
        )
    print(flush=True)
    cmp_primary = analysis["comparisons"]["primary_hpc_vs_high_lambda"]
    print(
        f"  Primary (hpc vs tcl_high_lambda): "
        f"p={cmp_primary['wilcoxon_p']:.4f}  "
        f"d={cmp_primary['cohens_d']:.3f}  "
        f"sig={cmp_primary['significant_bonferroni']}",
        flush=True,
    )
    cmp_lambda = analysis["comparisons"]["lambda_iso_high_lambda_vs_baseline"]
    print(
        f"  Lambda isolation (high_lambda vs baseline): "
        f"p={cmp_lambda['wilcoxon_p']:.4f}  "
        f"d={cmp_lambda['cohens_d']:.3f}",
        flush=True,
    )
    cmp_lr = analysis["comparisons"]["lr_iso_conservative_lr_vs_baseline"]
    print(
        f"  LR isolation (conservative_lr vs baseline): "
        f"p={cmp_lr['wilcoxon_p']:.4f}  "
        f"d={cmp_lr['cohens_d']:.3f}",
        flush=True,
    )
    print(f"\n  DECISION: {analysis['decision']}", flush=True)
    print(f"  {analysis['decision_rationale']}", flush=True)
    print(f"\n  Output written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
