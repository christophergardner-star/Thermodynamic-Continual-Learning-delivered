"""
Phase 3 Task 3.1 — Definitive 7-Condition Mechanistic Ablation

Determines WHAT in TCL drives forgetting reduction and WHY.

Conditions:
  sgd_baseline        — no protection (positive control)
  penalty_only        — gradient-EMA penalty, no governor
  governor_only       — governor LR adjustment, no penalty
  full_tcl            — penalty + governor (standard)
  anchor_frozen_init  — penalty, but ThermalImportance NOT reset between tasks
  warmup_batches_60   — full_tcl with warmup_batches=60 in observer
  ewc_best_lambda     — EWC lambda=1000 (external comparator)

Pre-registration: tar_state/preregistrations/mechanistic_ablation_7condition.json
Primary comparison: penalty_only vs sgd_baseline (Bonferroni k=6, threshold=0.0083)

NOTE ON PHASE 11 DATA: Phase 11 ran conditions sgd, governor_only, penalty_only,
full_tcl with seeds [42,0,1,2,3]. If those results are used, only
anchor_frozen_init, warmup_batches_60, ewc_best_lambda need new runs.
See: tar_state/comparisons/phase11_ablation__20260511T113318Z.json

Usage: python run_mechanistic_ablation.py [--dry-run] [--conditions CON1,CON2,...]
  --dry-run: Print config without running
  --conditions: Comma-separated subset of conditions to run (default: all 7)
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
_REPO = Path(__file__).resolve().parent
_TAR_STATE = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state")

sys.path.insert(0, str(_REPO))

# ── Experiment configuration ──────────────────────────────────────────────────

SEEDS = [42, 0, 1, 2, 3]
DATASET = "split_cifar10"
BACKBONE = "resnet18"
EPOCHS = 40
N_TASKS = 5
CLASSES_PER_TASK = 2          # 5 tasks × 2 classes = 10 classes (CIFAR-10)
BATCH_SIZE = 64
BASE_LR = 0.01

DATA_ROOT = str(Path(__file__).parent / "dataset_artifacts")

# Bonferroni family across 6 secondary comparisons
BONFERRONI_K = 6
BONFERRONI_THRESHOLD = 0.05 / BONFERRONI_K  # 0.0083

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# ── Condition configurations ──────────────────────────────────────────────────

CONDITION_CONFIGS: Dict[str, dict] = {
    "sgd_baseline": {
        "tcl_penalty_lambda": 0.0,
        "use_governor": False,
        "description": "No regularization — positive control",
    },
    "penalty_only": {
        "tcl_penalty_lambda": 0.01,
        "tcl_ema_beta": 0.99,
        "use_governor": False,
        "reset_importance_per_task": True,
        "description": "Gradient-EMA elastic penalty, no governor",
    },
    "governor_only": {
        "tcl_penalty_lambda": 0.0,
        "use_governor": True,
        "warmup_batches": 0,
        "description": "Governor LR adjustment only, no penalty",
    },
    "full_tcl": {
        "tcl_penalty_lambda": 0.01,
        "tcl_ema_beta": 0.99,
        "use_governor": True,
        "warmup_batches": 0,
        "description": "Full TCL: penalty + governor",
    },
    "anchor_frozen_init": {
        "tcl_penalty_lambda": 0.01,
        "tcl_ema_beta": 0.99,
        "use_governor": False,
        "reset_importance_per_task": False,  # KEY: no reset between tasks
        "description": "Penalty only, importance accumulates across all tasks (no per-task reset)",
    },
    "warmup_batches_60": {
        "tcl_penalty_lambda": 0.01,
        "tcl_ema_beta": 0.99,
        "use_governor": True,
        "warmup_batches": 60,  # KEY: 60-batch warmup before anchor lock
        "description": "Full TCL with warmup_batches=60 (Path B governor test)",
    },
    "ewc_best_lambda": {
        "method_override": "ewc_generic",   # use method_registry EWC
        "ewc_lambda": 1000.0,
        "description": "EWC lambda=1000 (Phase 12 optimum) — external comparator",
    },
}

# Conditions driven entirely by the direct training loop (not method_registry)
_DIRECT_CONDITIONS = {
    "sgd_baseline", "penalty_only", "governor_only",
    "full_tcl", "anchor_frozen_init", "warmup_batches_60",
}

# Conditions driven via run_generic_benchmark() / method_registry
_REGISTRY_CONDITIONS = {"ewc_best_lambda"}

PREREG_FILE = _TAR_STATE / "preregistrations" / "mechanistic_ablation_7condition.json"
EXEC_FLAG   = _TAR_STATE / "execution_enabled.flag"
CHECKPOINT_FILE = _TAR_STATE / "comparisons" / "mechanistic_ablation_checkpoint.json"

# Phase 11 data file (may be reused for 4 conditions)
PHASE11_FILE = _TAR_STATE / "comparisons" / "phase11_ablation__20260511T113318Z.json"

# Phase 11 condition name → our canonical condition name
_PHASE11_MAP = {
    "sgd":          "sgd_baseline",
    "governor_only": "governor_only",
    "penalty_only":  "penalty_only",
    "full_tcl":      "full_tcl",
}


# ── Prerequisite checks ───────────────────────────────────────────────────────

def _check_prerequisites(dry_run: bool, conditions_to_run: List[str]) -> None:
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
        print(f"  Seeds:      {SEEDS}", flush=True)
        print(f"  Dataset:    {DATASET}  backbone={BACKBONE}  epochs={EPOCHS}", flush=True)
        print(f"  Conditions: {conditions_to_run}", flush=True)
        print(f"  Bonferroni: k={BONFERRONI_K}  threshold={BONFERRONI_THRESHOLD:.4f}", flush=True)
        print(f"  Data root:  {DATA_ROOT}", flush=True)
        print(f"  Output dir: {_TAR_STATE / 'comparisons'}", flush=True)
        print("\n  Condition details:", flush=True)
        for cond in conditions_to_run:
            cfg = CONDITION_CONFIGS[cond]
            print(f"    {cond}: {cfg['description']}", flush=True)
        sys.exit(0)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _build_cifar10_tasks(seed: int):
    """
    Build Split-CIFAR-10: 5 tasks, each containing 2 of CIFAR-10's 10 classes.

    Returns (train_subsets, test_subsets) — each a list of 5 dataset objects
    with labels remapped to task-local [0, 1].
    """
    import torch
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

    cache_dir = str(Path(DATA_ROOT) / "cifar10")
    train_full = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                               download=True, transform=train_tf)
    test_full  = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                               download=True, transform=test_tf)

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
    test_subsets  = []
    train_targets = [int(y) for y in train_full.targets]
    test_targets  = [int(y) for y in test_full.targets]

    for task_classes in class_order:
        label_map = {orig: local for local, orig in enumerate(task_classes)}
        task_set  = set(task_classes)
        tr_idx = [i for i, y in enumerate(train_targets) if y in task_set]
        te_idx = [i for i, y in enumerate(test_targets)  if y in task_set]
        train_subsets.append(_RemappedSubset(train_full, tr_idx, label_map))
        test_subsets.append( _RemappedSubset(test_full,  te_idx, label_map))

    return train_subsets, test_subsets


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_model(device):
    """
    Build a ResNet-18 trunk + per-task linear heads for Split-CIFAR-10.

    ResNet-18 is modified for 32×32 inputs: conv1 → 3×3 stride-1,
    maxpool → Identity, matching standard CL practice on CIFAR-scale.
    """
    import torch.nn as nn
    import torchvision.models as models

    class _ResNet18Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            rn = models.resnet18(weights=None)
            rn.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            rn.maxpool = nn.Identity()
            self.features = nn.Sequential(*list(rn.children())[:-1])
            self.feat_dim  = 512

        def forward(self, x):
            return self.features(x).flatten(1)

    import torch.nn as nn
    trunk = _ResNet18Trunk().to(device)
    heads = nn.ModuleList([
        nn.Linear(trunk.feat_dim, CLASSES_PER_TASK)
        for _ in range(N_TASKS)
    ]).to(device)
    return trunk, heads


# ── Governor LR modulation ────────────────────────────────────────────────────

def _apply_governor(optimizer, regime: str) -> None:
    """
    Modulate optimizer learning rate based on thermodynamic regime.

    Regime semantics from ActivationThermoObserver.current_regime:
      "ordered"    — sigma < 0.9 × sigma_star  → converging, scale LR down
      "disordered" — sigma > 1.1 × sigma_star  → unstable, scale LR up
      "critical"   — near equilibrium → no change
      "unknown"    — observer not ready → no change

    LR scaling factors match the Phase 11 full_tcl configuration.
    """
    if regime == "ordered":
        scale = 0.5       # converging: reduce step size
    elif regime == "disordered":
        scale = 1.5       # unstable: larger steps to escape
    else:
        return            # critical / unknown: leave LR unchanged

    for pg in optimizer.param_groups:
        pg["lr"] = BASE_LR * scale


# ── Direct training loop (TCL conditions) ─────────────────────────────────────

def _run_direct_condition(
    seed: int,
    config: dict,
    condition_name: str,
    train_subsets: list,
    test_subsets: list,
) -> float:
    """
    Train one TCL-family condition on Split-CIFAR-10 for one seed.

    Returns average forgetting across tasks 0..N_TASKS-2 (forgetting is
    undefined for the last task since there is no subsequent task to forget).

    Training loop per task:
      - For penalty conditions: ThermalImportance accumulates after every
        backward(); TCLRegularizer.penalty() applied for tasks > 0.
      - For governor conditions: ActivationThermoObserver.step() called after
        every optimizer.step(); LR modulated by regime.
      - anchor_frozen_init: single ThermalImportance object created once
        before the task loop (not per-task); accumulates gradients from ALL
        tasks; committed after each task as normal.
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

    penalty_lambda          = config["tcl_penalty_lambda"]
    ema_beta                = config.get("tcl_ema_beta", 0.99)
    use_governor            = config.get("use_governor", False)
    warmup_batches          = config.get("warmup_batches", 0)
    reset_importance        = config.get("reset_importance_per_task", True)

    memory = ThermalMemory(max_tasks=N_TASKS, task_decay=1.0, anneal_rate=1.0)

    # For governor conditions, create a single observer for the lifetime of
    # this seed. It is reset at each task boundary via reset_for_new_task().
    observer = None
    if use_governor:
        from tar_lab.thermoobserver import ActivationThermoObserver
        observer = ActivationThermoObserver(
            trunk,
            warmup_batches=warmup_batches,
        )

    # anchor_frozen_init: create importance ONCE — not per task.
    # It accumulates gradient energy across ALL tasks from random init.
    frozen_importance: Optional[ThermalImportance] = None
    if not reset_importance:
        frozen_importance = ThermalImportance(trunk, ema_beta=ema_beta)

    # accuracy_matrix[train_t][eval_t] = accuracy after training task train_t,
    # evaluated on task eval_t's test set.
    accuracy_matrix: Dict[int, Dict[int, float]] = {}

    for train_t in range(N_TASKS):
        # Per-task importance (reset_importance=True) or accumulated (False)
        if reset_importance:
            importance = ThermalImportance(trunk, ema_beta=ema_beta)
        else:
            # anchor_frozen_init: continue accumulating into the same object
            importance = frozen_importance  # type: ignore[assignment]

        regularizer = TCLRegularizer(memory, lambda_tcl=penalty_lambda)

        all_params = list(trunk.parameters()) + list(heads[train_t].parameters())
        optimizer  = torch.optim.SGD(
            all_params, lr=BASE_LR, momentum=0.9, weight_decay=1e-4
        )

        loader = torch.utils.data.DataLoader(
            train_subsets[train_t], batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0,
        )

        # Reset per-task observer state at task boundary
        if observer is not None:
            observer.reset_for_new_task()

        for epoch in range(EPOCHS):
            trunk.train()
            heads[train_t].train()
            epoch_loss = 0.0
            n_batches  = 0

            for bx, by in loader:
                bx = bx.to(device)
                by = torch.tensor(
                    [int(y) for y in by], dtype=torch.long, device=device
                )

                optimizer.zero_grad()

                # Forward + task loss
                reps   = trunk(bx)
                logits = heads[train_t](reps)
                task_loss = F.cross_entropy(logits, by)
                task_loss.backward()

                # Accumulate importance from task-specific gradients BEFORE step
                importance.accumulate(trunk)

                # Forgetting protection: elastic penalty (only for tasks > 0)
                if memory.num_tasks > 0 and penalty_lambda > 0.0:
                    reg_loss = regularizer.penalty(trunk, device=device)
                    reg_loss.backward()

                optimizer.step()
                memory.anneal_all()

                # Governor: observe thermodynamic regime, modulate LR
                if observer is not None:
                    snapshot = observer.step(optimizer)
                    regime   = observer.current_regime
                    _apply_governor(optimizer, regime)

                epoch_loss += float(task_loss.item())
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  [{condition_name}] seed={seed} task={train_t+1}/{N_TASKS}"
                    f" epoch={epoch+1}/{EPOCHS} loss={avg_loss:.4f}",
                    flush=True,
                )

        # Commit task to memory after full training
        memory.commit(trunk, importance, task_id=train_t)

        # Evaluate all seen tasks after this task completes
        trunk.eval()
        row: Dict[int, float] = {}
        for eval_t in range(N_TASKS):
            eval_loader = torch.utils.data.DataLoader(
                test_subsets[eval_t], batch_size=256,
                shuffle=False, num_workers=0,
            )
            correct = 0
            total   = 0
            with torch.no_grad():
                for bx, by in eval_loader:
                    bx   = bx.to(device)
                    by_t = torch.tensor(
                        [int(y) for y in by], dtype=torch.long, device=device
                    )
                    preds   = heads[eval_t](trunk(bx)).argmax(1)
                    correct += int((preds == by_t).sum())
                    total   += by_t.size(0)
            row[eval_t] = correct / max(total, 1)
        accuracy_matrix[train_t] = row

        seen_accs = [
            f"{accuracy_matrix[train_t][t]:.3f}" for t in range(train_t + 1)
        ]
        print(
            f"  [{condition_name}] seed={seed} after task {train_t}: accs={seen_accs}",
            flush=True,
        )

    # Close observer hooks to avoid memory leaks
    if observer is not None:
        observer.close()

    # Compute average forgetting: mean over tasks 0..N_TASKS-2
    # Forgetting_t = peak_acc_t (across all subsequent evaluations) - final_acc_t
    forgetting_per_task = []
    for t in range(N_TASKS - 1):
        peak  = max(accuracy_matrix[step][t] for step in range(t, N_TASKS))
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


# ── EWC via generic_cl_runner ─────────────────────────────────────────────────

def _run_ewc_condition(seed: int, config: dict) -> float:
    """
    Run EWC at ewc_lambda=1000 via generic_cl_runner.run_generic_benchmark().

    Uses method_registry "ewc_generic" to match the Phase 12 optimum.
    Returns mean forgetting for this seed.
    """
    from tar_lab.generic_cl_runner import run_generic_benchmark

    method_name = config["method_override"]      # "ewc_generic"
    ewc_lambda  = config["ewc_lambda"]           # 1000.0

    seed_results, forgetting_list, _ = run_generic_benchmark(
        dataset_name     = DATASET,
        backbone_name    = BACKBONE,
        method_name      = method_name,
        seeds            = [seed],
        epochs           = EPOCHS,
        config_overrides = {
            "ewc_lambda":  ewc_lambda,
            "lr":          BASE_LR,
            "batch_size":  BATCH_SIZE,
        },
        data_root        = DATA_ROOT,
        log_fn           = lambda msg: print(
            f"  [ewc_best_lambda] seed={seed} {msg}", flush=True
        ),
    )

    return forgetting_list[0] if forgetting_list else float("nan")


# ── Condition dispatcher ──────────────────────────────────────────────────────

def _run_one_condition_seed(
    condition_name: str,
    seed: int,
    train_subsets: list,
    test_subsets: list,
) -> float:
    """Run a single (condition, seed) pair and return mean forgetting."""
    config = CONDITION_CONFIGS[condition_name]

    if condition_name in _REGISTRY_CONDITIONS:
        return _run_ewc_condition(seed, config)
    else:
        return _run_direct_condition(
            seed, config, condition_name, train_subsets, test_subsets
        )


# ── Phase 11 data loading ─────────────────────────────────────────────────────

def _try_load_phase11_results() -> Optional[Dict[str, List[float]]]:
    """
    Attempt to load Phase 11 forgetting values for the 4 conditions that
    Phase 11 already ran with seeds [42, 0, 1, 2, 3].

    Returns dict {condition_name: [forgetting_seed42, ..., forgetting_seed3]}
    or None if the file is absent or not usable.

    Only valid if Phase 11 seeds match SEEDS exactly.
    """
    if not PHASE11_FILE.exists():
        return None
    try:
        with open(PHASE11_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # Validate that Phase 11 seeds match our registered seeds
    p11_seeds = data.get("seeds", [])
    if list(p11_seeds) != list(SEEDS):
        print(
            f"[PHASE11] Seed mismatch: Phase 11 seeds={p11_seeds}, "
            f"required={SEEDS}. Skipping Phase 11 data reuse.",
            flush=True,
        )
        return None

    # Validate epochs and backbone match
    if data.get("epochs") != EPOCHS or data.get("backbone") != BACKBONE:
        print(
            f"[PHASE11] Config mismatch (epochs or backbone). Skipping.",
            flush=True,
        )
        return None

    per_seed = data.get("per_seed", [])
    if len(per_seed) != len(SEEDS):
        return None

    results: Dict[str, List[float]] = {}
    for our_name in ("sgd_baseline", "governor_only", "penalty_only", "full_tcl"):
        # Find Phase 11 key: the reverse of _PHASE11_MAP
        p11_key = next(
            (k for k, v in _PHASE11_MAP.items() if v == our_name), None
        )
        if p11_key is None:
            continue
        forgetting_key = f"{p11_key}_forgetting"
        vals = [row[forgetting_key] for row in per_seed if forgetting_key in row]
        if len(vals) == len(SEEDS):
            results[our_name] = vals

    return results if results else None


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"per_condition_per_seed": {}, "conditions_complete": []}


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
    """Import stat_utils, inserting the repo root onto sys.path."""
    import importlib
    tar_lab_parent = str(_REPO)
    if tar_lab_parent not in sys.path:
        sys.path.insert(0, tar_lab_parent)
    return importlib.import_module("tar_lab.stat_utils")


def _compare_pair(
    su,
    vals_a: List[float],
    vals_b: List[float],
    label_a: str,
    label_b: str,
    threshold: float = BONFERRONI_THRESHOLD,
    alternative: str = "less",   # H1: a < b (a has less forgetting)
) -> dict:
    """
    Run paired Wilcoxon and compute Cohen's d for vals_a vs vals_b.
    Returns a flat dict suitable for JSON output.
    """
    try:
        cmp = su.compare_methods_paired(vals_a, vals_b, alternative=alternative)
        p_val    = cmp.p_nonparametric
        cohens_d = cmp.cohens_d
        bonf_sig = bool(p_val < threshold)
        direction = (
            f"{label_a}_better"
            if _mean(vals_a) < _mean(vals_b)
            else f"{label_b}_better"
        )
        return {
            "p_wilcoxon":           round(p_val, 6),
            "cohens_d":             round(cohens_d, 4),
            "bonferroni_threshold": threshold,
            "bonferroni_significant": bonf_sig,
            "direction":            direction,
            f"mean_{label_a}":      round(_mean(vals_a), 6),
            f"mean_{label_b}":      round(_mean(vals_b), 6),
            "n":                    len(vals_a),
            "note":                 (
                f"Wilcoxon signed-rank, one-tailed (H1: {label_a} < {label_b}), "
                f"Bonferroni k={BONFERRONI_K}"
            ),
        }
    except Exception as exc:
        return {
            "error":       str(exc),
            "p_wilcoxon":  float("nan"),
            "cohens_d":    float("nan"),
            "bonferroni_significant": False,
            "direction":   "unavailable",
        }


# ── Mechanistic verdict ───────────────────────────────────────────────────────

def _derive_verdict(
    primary: dict,
    secondary: dict,
    per_condition_mean: Dict[str, float],
) -> str:
    """
    Derive mechanistic verdict from primary and secondary comparison results.

    PENALTY_DOMINANT: penalty_only beats sgd_baseline (p < 0.0083) AND
        governor adds no significant improvement (full_tcl vs penalty_only
        p >= 0.0083 OR full_tcl is not better).
    GOVERNOR_CONTRIBUTES: penalty_only beats sgd_baseline AND full_tcl
        significantly beats penalty_only.
    AMBIGUOUS: primary comparison fails or results contradict the
        a priori hypothesis.
    """
    primary_sig = primary.get("penalty_only_vs_sgd", {}).get(
        "bonferroni_significant", False
    )
    penalty_better = primary.get("penalty_only_vs_sgd", {}).get(
        "direction", ""
    ) == "penalty_only_better"

    governor_sig = secondary.get("full_tcl_vs_penalty_only", {}).get(
        "bonferroni_significant", False
    )
    governor_better = secondary.get("full_tcl_vs_penalty_only", {}).get(
        "direction", ""
    ) == "full_tcl_better"

    if not primary_sig or not penalty_better:
        return "AMBIGUOUS"
    if governor_sig and governor_better:
        return "GOVERNOR_CONTRIBUTES"
    return "PENALTY_DOMINANT"


# ── Final analysis ────────────────────────────────────────────────────────────

def _run_final_analysis(
    all_results: Dict[str, List[float]],
    conditions_used: List[str],
) -> dict:
    """
    Run all pre-registered comparisons and derive mechanistic verdict.

    all_results: {condition_name: [forgetting_seed0, ..., forgetting_seed4]}
    conditions_used: conditions actually present in all_results
    """
    su = _import_stat_utils()

    per_condition_mean = {
        cond: round(_mean(vals), 6)
        for cond, vals in all_results.items()
    }
    per_condition_std = {
        cond: round(_std(vals), 6)
        for cond, vals in all_results.items()
    }

    # ── Primary comparison: penalty_only vs sgd_baseline ──────────────────
    primary: dict = {}
    if "penalty_only" in all_results and "sgd_baseline" in all_results:
        primary["penalty_only_vs_sgd"] = _compare_pair(
            su,
            all_results["penalty_only"],
            all_results["sgd_baseline"],
            label_a="penalty_only",
            label_b="sgd_baseline",
            threshold=BONFERRONI_THRESHOLD,
            alternative="less",
        )
    else:
        primary["penalty_only_vs_sgd"] = {"error": "one or both conditions missing"}

    # ── Secondary comparisons (Bonferroni k=6 applied to all) ─────────────
    secondary: dict = {}

    # 1. full_tcl vs penalty_only — governor marginal contribution
    if "full_tcl" in all_results and "penalty_only" in all_results:
        secondary["full_tcl_vs_penalty_only"] = _compare_pair(
            su,
            all_results["full_tcl"],
            all_results["penalty_only"],
            label_a="full_tcl",
            label_b="penalty_only",
            threshold=BONFERRONI_THRESHOLD,
            alternative="less",
        )

    # 2. anchor_frozen_init vs penalty_only — per-task anchor value
    if "anchor_frozen_init" in all_results and "penalty_only" in all_results:
        secondary["anchor_frozen_vs_penalty_only"] = _compare_pair(
            su,
            all_results["anchor_frozen_init"],
            all_results["penalty_only"],
            label_a="anchor_frozen_init",
            label_b="penalty_only",
            threshold=BONFERRONI_THRESHOLD,
            alternative="less",
        )

    # 3. warmup_batches_60 vs governor_only — whether warmup fixes governor
    if "warmup_batches_60" in all_results and "governor_only" in all_results:
        secondary["warmup60_vs_governor_only"] = _compare_pair(
            su,
            all_results["warmup_batches_60"],
            all_results["governor_only"],
            label_a="warmup_batches_60",
            label_b="governor_only",
            threshold=BONFERRONI_THRESHOLD,
            alternative="less",
        )

    # 4. penalty_only vs ewc_best_lambda — paper's comparative claim
    if "penalty_only" in all_results and "ewc_best_lambda" in all_results:
        secondary["penalty_only_vs_ewc"] = _compare_pair(
            su,
            all_results["penalty_only"],
            all_results["ewc_best_lambda"],
            label_a="penalty_only",
            label_b="ewc_best_lambda",
            threshold=BONFERRONI_THRESHOLD,
            alternative="less",
        )

    # ── Mechanistic verdict ────────────────────────────────────────────────
    verdict = _derive_verdict(primary, secondary, per_condition_mean)

    return {
        "primary_comparison":       primary,
        "secondary_comparisons":    secondary,
        "mechanistic_verdict":      verdict,
        "per_condition_mean_forgetting": per_condition_mean,
        "per_condition_std_forgetting":  per_condition_std,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 Task 3.1 — Definitive 7-Condition Mechanistic Ablation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config without running.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of conditions to run. "
            "Default: all 7. "
            f"Choices: {','.join(CONDITION_CONFIGS)}"
        ),
    )
    args = parser.parse_args()

    # Parse conditions
    all_condition_names = list(CONDITION_CONFIGS)
    if args.conditions:
        requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in requested if c not in CONDITION_CONFIGS]
        if unknown:
            print(f"ERROR: Unknown conditions: {unknown}", flush=True)
            print(f"  Valid: {all_condition_names}", flush=True)
            sys.exit(1)
        conditions_to_run = requested
    else:
        conditions_to_run = all_condition_names

    # 1. Prerequisites
    _check_prerequisites(dry_run=args.dry_run, conditions_to_run=conditions_to_run)

    # 2. Load pre-registration (informational)
    with open(PREREG_FILE, "r", encoding="utf-8") as f:
        prereg = json.load(f)
    print(f"[MECH ABLATION] Pre-registration loaded: {PREREG_FILE}", flush=True)
    print(f"[MECH ABLATION] Conditions to run: {conditions_to_run}", flush=True)
    print(f"[MECH ABLATION] Seeds: {SEEDS}", flush=True)
    print(f"[MECH ABLATION] Bonferroni k={BONFERRONI_K}  threshold={BONFERRONI_THRESHOLD:.4f}",
          flush=True)
    print("=" * 70, flush=True)

    # 3. Attempt to reuse Phase 11 data for 4 conditions
    phase11_results = _try_load_phase11_results()
    if phase11_results:
        reused = sorted(phase11_results)
        print(
            f"[PHASE11 REUSE] Loaded {len(reused)} conditions from Phase 11: {reused}",
            flush=True,
        )
        print(
            f"  Only {[c for c in conditions_to_run if c not in phase11_results]} "
            f"will require new runs.",
            flush=True,
        )
    else:
        print("[PHASE11 REUSE] Phase 11 data not available or not compatible; "
              "all requested conditions will run fresh.", flush=True)
        phase11_results = {}

    # 4. Resume from checkpoint
    ckpt = _load_checkpoint()
    # all_results: condition → list of per-seed forgetting values (in SEEDS order)
    all_results: Dict[str, List[float]] = {}

    # Pre-populate from Phase 11 (for conditions covered by it)
    for cond, vals in phase11_results.items():
        if cond in conditions_to_run:
            all_results[cond] = list(vals)
            print(
                f"  [PHASE11] {cond}: {[round(v, 4) for v in vals]} "
                f"(mean={round(_mean(vals), 4)})",
                flush=True,
            )

    # Pre-populate from checkpoint (overrides Phase 11 if both present)
    ckpt_data = ckpt.get("per_condition_per_seed", {})
    for cond in conditions_to_run:
        if cond in ckpt_data:
            # Checkpoint stores {seed: forgetting} — align to SEEDS order
            seed_map = ckpt_data[cond]
            vals = [seed_map[str(s)] for s in SEEDS if str(s) in seed_map]
            if len(vals) == len(SEEDS):
                all_results[cond] = vals
                print(
                    f"  [CHECKPOINT] {cond}: {[round(v, 4) for v in vals]} "
                    f"(mean={round(_mean(vals), 4)})",
                    flush=True,
                )

    # 5. Run conditions that still need work
    needs_run = [c for c in conditions_to_run if c not in all_results]
    print(f"\n[MECH ABLATION] Conditions needing new runs: {needs_run}", flush=True)

    for condition_name in needs_run:
        print(f"\n{'='*60}", flush=True)
        print(f"[CONDITION] {condition_name}", flush=True)
        print(f"  {CONDITION_CONFIGS[condition_name]['description']}", flush=True)

        condition_vals: List[float] = []
        # Checkpoint: may have partial seed results
        cond_ckpt: Dict[str, float] = dict(ckpt_data.get(condition_name, {}))
        completed_seeds = {int(k): v for k, v in cond_ckpt.items()}

        for seed in SEEDS:
            if seed in completed_seeds:
                f_val = completed_seeds[seed]
                print(
                    f"  seed={seed} [SKIP: checkpoint] "
                    f"forgetting={f_val:.4f}",
                    flush=True,
                )
                condition_vals.append(f_val)
                continue

            print(f"\n  --- seed={seed} ---", flush=True)

            # Build dataset for this seed (not needed for EWC registry path)
            if condition_name in _DIRECT_CONDITIONS:
                print(f"  Building Split-CIFAR-10 tasks (seed={seed})...", flush=True)
                train_subsets, test_subsets = _build_cifar10_tasks(seed)
            else:
                train_subsets, test_subsets = [], []   # unused by EWC path

            forgetting = _run_one_condition_seed(
                condition_name, seed, train_subsets, test_subsets
            )

            print(
                f"  seed={seed} forgetting={forgetting:.4f}",
                flush=True,
            )
            condition_vals.append(forgetting)
            completed_seeds[seed] = forgetting

            # Write checkpoint after each seed
            ckpt_data[condition_name] = {str(s): v for s, v in completed_seeds.items()}
            _save_checkpoint({
                "per_condition_per_seed": ckpt_data,
                "conditions_complete": [
                    c for c in conditions_to_run if c in all_results
                ],
            })

        all_results[condition_name] = condition_vals
        print(
            f"\n  [{condition_name}] DONE: "
            f"{[round(v, 4) for v in condition_vals]} "
            f"mean={round(_mean(condition_vals), 4):.4f}",
            flush=True,
        )

    # 6. Final analysis
    print("\n" + "=" * 70, flush=True)
    print("[MECH ABLATION] Running final analysis...", flush=True)

    if len(all_results) < 2:
        print("ERROR: Fewer than 2 conditions completed. Cannot run analysis.",
              flush=True)
        sys.exit(1)

    analysis = _run_final_analysis(all_results, conditions_to_run)

    # 7. Build per-seed structured records
    per_seed_records = []
    for idx, seed in enumerate(SEEDS):
        record: dict = {"seed": seed}
        for cond in conditions_to_run:
            if cond in all_results and idx < len(all_results[cond]):
                record[f"{cond}_forgetting"] = round(all_results[cond][idx], 6)
        per_seed_records.append(record)

    # 8. Write output JSON
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = _TAR_STATE / "comparisons" / f"mechanistic_ablation_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "result_id":        f"mechanistic_ablation_{timestamp}",
        "experiment_id":    prereg.get("experiment_id", "mechanistic_ablation_7condition_phase3"),
        "preregistration_file": str(PREREG_FILE),
        "phase11_file_used": str(PHASE11_FILE) if phase11_results else None,
        "conditions_requested": conditions_to_run,
        "conditions_from_phase11": sorted(phase11_results.keys()) if phase11_results else [],
        "seeds": SEEDS,
        "dataset": DATASET,
        "backbone": BACKBONE,
        "epochs": EPOCHS,
        "n_tasks": N_TASKS,
        "base_lr": BASE_LR,
        "batch_size": BATCH_SIZE,
        "bonferroni_k": BONFERRONI_K,
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
        "condition_configs": CONDITION_CONFIGS,
        "per_seed_results": per_seed_records,
        # Analysis outputs — mirrors required schema
        "primary_comparison":            analysis["primary_comparison"],
        "secondary_comparisons":         analysis["secondary_comparisons"],
        "mechanistic_verdict":           analysis["mechanistic_verdict"],
        "per_condition_mean_forgetting": analysis["per_condition_mean_forgetting"],
        "per_condition_std_forgetting":  analysis["per_condition_std_forgetting"],
        # Raw per-condition lists for downstream re-analysis
        "per_condition_forgetting_raw":  {
            cond: [round(v, 6) for v in vals]
            for cond, vals in all_results.items()
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # 9. Print summary
    print(f"\n[MECH ABLATION] COMPLETE", flush=True)
    print(f"  Conditions run: {sorted(all_results)}", flush=True)
    print(f"\n  Per-condition mean forgetting:", flush=True)
    for cond in conditions_to_run:
        if cond in all_results:
            vals = all_results[cond]
            print(
                f"    {cond:<22s}  {_mean(vals):.4f} ± {_std(vals):.4f}",
                flush=True,
            )

    print(f"\n  PRIMARY: penalty_only vs sgd_baseline", flush=True)
    p_entry = analysis["primary_comparison"].get("penalty_only_vs_sgd", {})
    print(
        f"    p={p_entry.get('p_wilcoxon', 'N/A')}  "
        f"d={p_entry.get('cohens_d', 'N/A')}  "
        f"Bonferroni-sig={p_entry.get('bonferroni_significant', 'N/A')}  "
        f"direction={p_entry.get('direction', 'N/A')}",
        flush=True,
    )

    print(f"\n  SECONDARY comparisons (Bonferroni threshold={BONFERRONI_THRESHOLD:.4f}):",
          flush=True)
    for label, entry in analysis["secondary_comparisons"].items():
        sig = entry.get("bonferroni_significant", "N/A")
        direction = entry.get("direction", "N/A")
        p_val = entry.get("p_wilcoxon", "N/A")
        print(f"    {label}: p={p_val}  sig={sig}  direction={direction}",
              flush=True)

    print(f"\n  MECHANISTIC VERDICT: {analysis['mechanistic_verdict']}", flush=True)
    print(f"\n  Output written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
