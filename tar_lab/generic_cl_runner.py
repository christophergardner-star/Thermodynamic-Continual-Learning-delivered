"""
Generic continual-learning benchmark runner using the CLMethod plugin registry.

Supports:
  Datasets : split_cifar10 (5 tasks × 2 cls), split_cifar100 (10 tasks × 10 cls),
             split_tinyimagenet (20 tasks × 10 cls, data passed in by caller)
  Backbones: tiny_cnn, resnet18
  Methods  : any key in METHOD_REGISTRY (built-ins + LLM-synthesised)

Returns (seed_results, forgetting_list, accuracy_list) matching the format
expected by ExperimentOrchestrator._build_result().
"""
from __future__ import annotations

import math
import random
import types
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

_DATASET_N_CLASSES = {
    "split_cifar10":       10,
    "split_cifar100":     100,
    "split_tinyimagenet": 200,
}

_DATASET_TASK_SPLITS = {
    "split_cifar10":       5,   # 5 tasks × 2 classes
    "split_cifar100":     10,   # 10 tasks × 10 classes
    "split_tinyimagenet": 20,   # 20 tasks × 10 classes
}

_NORMALIZE = {
    "split_cifar10":       ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "split_cifar100":      ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "split_tinyimagenet":  ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
}

# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class _TinyCNN(nn.Module):
    """3-conv trunk; works for 32×32 (CIFAR) and 64×64 (TinyImageNet)."""

    feat_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _ResNet18Trunk(nn.Module):
    feat_dim = 512

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as _tv
        rn = _tv.resnet18(weights=None)
        self.body = nn.Sequential(*list(rn.children())[:-1])  # strip FC

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).flatten(1)


class _CLModel(nn.Module):
    """Trunk + shared linear head."""

    def __init__(self, trunk: nn.Module, n_classes: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.head  = nn.Linear(trunk.feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def _build_model(backbone_name: str, n_classes: int, device: torch.device) -> _CLModel:
    trunk: nn.Module
    if backbone_name == "resnet18":
        trunk = _ResNet18Trunk()
    else:
        trunk = _TinyCNN()
    return _CLModel(trunk, n_classes).to(device)


BACKBONE_REGISTRY: dict[str, Callable[[int, torch.device], _CLModel]] = {
    "tiny_cnn":  lambda nc, dev: _build_model("tiny_cnn",  nc, dev),
    "resnet18":  lambda nc, dev: _build_model("resnet18", nc, dev),
}

# Task 2.11 ablation: BatchNorm running statistics may leak cross-task information.
# Set reset_bn_at_task_boundary=True to isolate each task's BN statistics.
# Default=False preserves the existing behaviour (matching all prior experiments).

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _get_torchvision_dataset(dataset_name: str, data_root: str, train: bool):
    """Return a torchvision dataset (no transforms yet — applied in loader)."""
    import torchvision.datasets as _ds
    import torchvision.transforms as _T

    mean, std = _NORMALIZE[dataset_name]
    xforms = [_T.ToTensor(), _T.Normalize(mean, std)]
    if train:
        xforms = [_T.RandomHorizontalFlip(), _T.RandomCrop(32, padding=4)] + xforms
    t = _T.Compose(xforms)

    if dataset_name == "split_cifar10":
        return _ds.CIFAR10(data_root, train=train, download=True, transform=t)
    elif dataset_name == "split_cifar100":
        return _ds.CIFAR100(data_root, train=train, download=True, transform=t)
    else:
        raise ValueError(f"Use pre-built loaders for {dataset_name}")


def _class_indices(dataset, class_ids: list[int]) -> list[int]:
    """Return sample indices whose label is in class_ids."""
    targets = (
        dataset.targets
        if hasattr(dataset, "targets")
        else [int(dataset[i][1]) for i in range(len(dataset))]
    )
    id_set = set(class_ids)
    return [i for i, t in enumerate(targets) if t in id_set]


def _build_task_loaders(
    dataset_name: str,
    data_root: str,
    seed: int,
    batch_size: int,
) -> tuple[list[DataLoader], list[DataLoader], list[list[int]]]:
    """
    Build train/test loaders per task and return task class groups.
    Only for CIFAR-10 and CIFAR-100 (torchvision).
    """
    n_tasks   = _DATASET_TASK_SPLITS[dataset_name]
    n_classes = _DATASET_N_CLASSES[dataset_name]
    cls_per_task = n_classes // n_tasks

    rng = random.Random(seed)
    class_order = list(range(n_classes))
    rng.shuffle(class_order)
    task_groups = [
        class_order[i * cls_per_task:(i + 1) * cls_per_task]
        for i in range(n_tasks)
    ]

    full_train = _get_torchvision_dataset(dataset_name, data_root, train=True)
    full_test  = _get_torchvision_dataset(dataset_name, data_root, train=False)

    train_loaders, test_loaders = [], []
    for grp in task_groups:
        tr_idx = _class_indices(full_train, grp)
        te_idx = _class_indices(full_test,  grp)
        train_loaders.append(DataLoader(
            Subset(full_train, tr_idx),
            batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True,
        ))
        test_loaders.append(DataLoader(
            Subset(full_test,  te_idx),
            batch_size=256,       shuffle=False, num_workers=2, pin_memory=True,
        ))

    return train_loaders, test_loaders, task_groups

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    model.train()
    return correct / max(total, 1)


def _compute_forgetting(acc_matrix: list[list[float]]) -> float:
    """
    acc_matrix[task_idx][after_task_idx] = accuracy.
    Forgetting = mean over tasks 0..T-2 of (peak_acc - final_acc).
    Returns 0.0 if only one task was trained.
    """
    n = len(acc_matrix)
    if n <= 1:
        return 0.0
    forgetting = []
    for t in range(n - 1):          # exclude the last-trained task
        row = acc_matrix[t][t:]     # accuracies from when task t was first seen
        if not row:
            continue
        peak  = max(row)
        final = row[-1]
        forgetting.append(max(0.0, peak - final))
    return sum(forgetting) / len(forgetting) if forgetting else 0.0


def compute_transfer_metrics(
    acc_matrix: list[list[float]],
    random_baseline: float = 0.1,
) -> dict:
    """
    Compute backward transfer (BWT), forward transfer (FWT), and intransigence
    index from the accuracy matrix.

    Storage convention (ragged upper-triangular):
      acc_matrix[t] has (T - t) entries.
      acc_matrix[t][0]  = accuracy of task t immediately after training it  (diagonal).
      acc_matrix[t][-1] = accuracy of task t after all T tasks are trained  (final column).

    BWT  — Lopez-Paz & Ranzato (2017):
      BWT = (1/(T-1)) * sum_{t=0}^{T-2} (final_t - diag_t)
      Negative BWT indicates catastrophic forgetting.

    FWT  — Lopez-Paz & Ranzato (2017):
      FWT = (1/T) * sum_{t=0}^{T-1} (diag_t - random_baseline)

    Intransigence Index — Chaudhry et al. (2018):
      Fraction of tasks 0..T-2 with > 5 % forgetting at the end of training.

    Parameters
    ----------
    acc_matrix      : ragged list as produced by _run_one_seed
    random_baseline : expected accuracy of a random classifier (default 0.1 for
                      10-class tasks; adjust for other class counts)

    Returns
    -------
    dict with keys: bwt, fwt, intransigence_index, forgetting_per_task, n_tasks
    """
    T = len(acc_matrix)
    _zero = {
        "bwt": 0.0,
        "fwt": 0.0,
        "intransigence_index": 0.0,
        "forgetting_per_task": [],
        "n_tasks": T,
    }
    if T < 2:
        return _zero

    # Diagonal: acc_matrix[t][0] — accuracy right after training task t
    # Final col: acc_matrix[t][-1] — accuracy after all tasks
    diag  = [acc_matrix[t][0]  for t in range(T)]
    final = [acc_matrix[t][-1] for t in range(T)]

    # BWT: average over tasks 0..T-2 (exclude the last task — no subsequent tasks)
    bwt_terms = [final[t] - diag[t] for t in range(T - 1)]
    bwt = sum(bwt_terms) / (T - 1)

    # FWT: average over all tasks
    fwt_terms = [diag[t] - random_baseline for t in range(T)]
    fwt = sum(fwt_terms) / T

    # Forgetting per task (0..T-2) — non-negative
    forgetting_per_task = [max(0.0, diag[t] - final[t]) for t in range(T - 1)]

    # Intransigence index: fraction of old tasks with > 5 % forgetting
    n_old = len(forgetting_per_task)
    intransigence_index = (
        sum(1 for f in forgetting_per_task if f > 0.05) / max(n_old, 1)
    )

    return {
        "bwt":                round(bwt, 4),
        "fwt":                round(fwt, 4),
        "intransigence_index": round(intransigence_index, 4),
        "forgetting_per_task": [round(f, 4) for f in forgetting_per_task],
        "n_tasks":             T,
    }


@torch.no_grad()
def compute_per_task_ece(
    model: nn.Module,
    task_test_loaders: list,
    device: torch.device,
    trained_up_to_task: int,
    n_bins: int = 15,
) -> list[float]:
    """
    Compute Expected Calibration Error (ECE) for each task seen so far.

    ECE = sum_b |acc_b - conf_b| * |B_b| / N
    where B_b is the set of samples whose max-softmax confidence falls in bin b.

    Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks".

    Parameters
    ----------
    model               : trained nn.Module
    task_test_loaders   : list of DataLoader, one per task
    device              : torch.device
    trained_up_to_task  : evaluate tasks 0..trained_up_to_task (inclusive)
    n_bins              : number of uniform confidence bins over [0, 1]

    Returns
    -------
    list of ECE floats (rounded to 4 dp), one entry per evaluated task
    """
    model.eval()
    ece_list: list[float] = []
    bin_edges = [k / n_bins for k in range(n_bins + 1)]  # n_bins+1 boundaries

    for task_idx in range(trained_up_to_task + 1):
        loader = task_test_loaders[task_idx]
        all_conf:    list[float] = []
        all_correct: list[bool]  = []

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                             # (B, C)
            probs  = F.softmax(logits, dim=-1)            # (B, C)
            conf, pred = probs.max(dim=-1)                # (B,), (B,)
            correct = (pred == y)                         # (B,) bool

            all_conf.extend(conf.cpu().tolist())
            all_correct.extend(correct.cpu().tolist())

        N = len(all_conf)
        if N == 0:
            ece_list.append(0.0)
            continue

        ece = 0.0
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            # Include upper bound only for the last bin to avoid missing samples
            # with confidence exactly 1.0
            if b < n_bins - 1:
                in_bin = [i for i in range(N) if lo <= all_conf[i] < hi]
            else:
                in_bin = [i for i in range(N) if lo <= all_conf[i] <= hi]

            if not in_bin:
                continue

            acc_b  = sum(all_correct[i] for i in in_bin) / len(in_bin)
            conf_b = sum(all_conf[i]    for i in in_bin) / len(in_bin)
            ece   += abs(acc_b - conf_b) * len(in_bin) / N

        ece_list.append(round(ece, 4))

    model.train()
    return ece_list

# ---------------------------------------------------------------------------
# BatchNorm reset helper (Task 2.11 ablation)
# ---------------------------------------------------------------------------

def _reset_bn_running_stats(model: nn.Module) -> None:
    """
    Reset BatchNorm running mean and variance at task boundaries.

    BatchNorm running statistics accumulate across tasks by default.
    This creates an implicit memory: task T+1's batch normalization is
    conditioned on statistics from tasks 0..T. Resetting at task boundaries
    eliminates this confound for ablation studies.

    Reference: Lomonaco et al. (2020), "CORe50: a New Dataset and
    Benchmark for Continuous Object Recognition." — notes BN as a
    source of cross-task information leakage.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()

# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def _run_one_seed(
    seed: int,
    model: nn.Module,
    method_name: str,
    task_train_loaders: list[DataLoader],
    task_test_loaders:  list[DataLoader],
    epochs: int,
    config_obj: Any,
    device: torch.device,
    log_fn: Callable[[str], None],
    progress_callback: Callable[[dict], None] | None,
    reset_bn_at_task_boundary: bool = False,
) -> dict:
    from tar_lab.method_registry import METHOD_REGISTRY

    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Method '{method_name}' not in METHOD_REGISTRY. "
            f"Available: {sorted(METHOD_REGISTRY)}"
        )

    method = METHOD_REGISTRY[method_name](config_obj)

    lr = float(getattr(config_obj, "lr", 0.05))
    wd = float(getattr(config_obj, "weight_decay", 1e-4))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    n_tasks    = len(task_train_loaders)
    # acc_matrix[task_t][after_task_k] — only filled for k >= t
    acc_matrix: list[list[float]] = [[] for _ in range(n_tasks)]

    for task_id, train_loader in enumerate(task_train_loaders):
        method.pre_task(task_id, model, device)
        if reset_bn_at_task_boundary:
            _reset_bn_running_stats(model)

        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                out  = model(x)
                loss = F.cross_entropy(out, y) + method.regularization_loss(model)
                loss.backward()

                # aug_loss: for replay methods adds a second backward;
                # for SI reads p.grad to accumulate path integral (returns 0.0).
                aug = method.augmented_loss(model, x, y, task_id, device)
                if aug.item() != 0.0:
                    aug.backward()

                optimizer.step()

        method.post_task(task_id, model, train_loader, device)

        # Evaluate all seen tasks
        for prev_t in range(task_id + 1):
            acc = _evaluate(model, task_test_loaders[prev_t], device)
            acc_matrix[prev_t].append(acc)
            log_fn(
                f"  seed={seed}  task={task_id}  eval_task={prev_t}"
                f"  acc={acc:.4f}"
            )

        if progress_callback is not None:
            latest = [acc_matrix[t][-1] for t in range(task_id + 1)]
            progress_callback({
                "tasks_done": task_id + 1,
                "latest_accs": [f"{v:.3f}" for v in latest],
            })

    mean_forgetting = _compute_forgetting(acc_matrix)
    # Final accuracy = mean over all tasks of their last measured accuracy
    final_accs = [acc_matrix[t][-1] for t in range(n_tasks) if acc_matrix[t]]
    mean_accuracy = sum(final_accs) / len(final_accs) if final_accs else 0.0

    # --- NEW: transfer metrics and ECE ---
    transfer_metrics = compute_transfer_metrics(acc_matrix)
    ece_trajectory = compute_per_task_ece(
        model,
        task_test_loaders,
        device,
        trained_up_to_task=len(task_test_loaders) - 1,
    )

    return {
        "mean_forgetting":     mean_forgetting,
        "mean_accuracy":       mean_accuracy,
        "final_accs_per_task": final_accs,
        "acc_matrix":          [list(row) for row in acc_matrix],
        # Transfer metrics (Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018)
        "bwt":                 transfer_metrics["bwt"],
        "fwt":                 transfer_metrics["fwt"],
        "intransigence_index": transfer_metrics["intransigence_index"],
        "forgetting_per_task": transfer_metrics["forgetting_per_task"],
        # Calibration trajectory — ECE per task after final training
        "ece_trajectory":      ece_trajectory,
        # Task 2.11 ablation flag — records which variant was run
        "bn_reset_at_boundaries": reset_bn_at_task_boundary,
    }

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_generic_benchmark(
    dataset_name:    str,
    backbone_name:   str,
    method_name:     str,
    seeds:           list[int],
    epochs:          int,
    config_overrides: dict,
    data_root:       str,
    log_fn:          Callable[[str], None],
    *,
    # Caller may supply pre-built loaders (required for split_tinyimagenet)
    prebuilt_task_train: list[DataLoader] | None = None,
    prebuilt_task_test:  list[DataLoader] | None = None,
    progress_callback:   Callable[[int, dict], None] | None = None,
    reset_bn_at_task_boundary: bool = False,
) -> tuple[list[dict], list[float], list[float]]:
    """
    Run a continual-learning benchmark with a registered CLMethod.

    Returns
    -------
    seed_results   : list of per-seed dicts  {"seed", "forgetting", "accuracy"}
    forgetting_list: mean_forgetting per seed
    accuracy_list  : mean_accuracy per seed
    """
    from tar_lab.method_registry import METHOD_REGISTRY, load_generated_methods
    from pathlib import Path

    # Load any LLM-synthesised methods from the standard location
    synth_dir = Path(data_root).parent / "tar_state" / "synthesized_methods"
    load_generated_methods(synth_dir)

    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Method '{method_name}' not registered. "
            f"Known: {sorted(METHOD_REGISTRY)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config object: wrap dict in a namespace for attribute access
    cfg = types.SimpleNamespace(**config_overrides)
    # Propagate epochs so methods can read it if needed
    if not hasattr(cfg, "epochs"):
        cfg.epochs = epochs

    n_classes = _DATASET_N_CLASSES.get(dataset_name, 10)
    batch_size = int(getattr(cfg, "batch_size", 128))

    seed_results:    list[dict]  = []
    forgetting_list: list[float] = []
    accuracy_list:   list[float] = []

    for i, seed in enumerate(seeds):
        _set_seed(seed)

        if prebuilt_task_train is not None and prebuilt_task_test is not None:
            train_loaders = prebuilt_task_train
            test_loaders  = prebuilt_task_test
        else:
            train_loaders, test_loaders, _ = _build_task_loaders(
                dataset_name, data_root, seed, batch_size
            )

        model = _build_model(backbone_name, n_classes, device)

        def _cb(payload: dict, _i: int = i) -> None:
            if progress_callback is not None:
                progress_callback(_i, payload)

        log_fn(f"[generic_cl] seed={seed}  method={method_name}"
               f"  dataset={dataset_name}  backbone={backbone_name}")

        res = _run_one_seed(
            seed          = seed,
            model         = model,
            method_name   = method_name,
            task_train_loaders = train_loaders,
            task_test_loaders  = test_loaders,
            epochs        = epochs,
            config_obj    = cfg,
            device        = device,
            log_fn        = log_fn,
            progress_callback = _cb,
            reset_bn_at_task_boundary = reset_bn_at_task_boundary,
        )

        forgetting_list.append(res["mean_forgetting"])
        accuracy_list.append(res["mean_accuracy"])
        seed_results.append({
            "seed":                seed,
            "forgetting":          res["mean_forgetting"],
            "accuracy":            res["mean_accuracy"],
            # Transfer metrics
            "bwt":                 res["bwt"],
            "fwt":                 res["fwt"],
            "intransigence_index": res["intransigence_index"],
            "forgetting_per_task": res["forgetting_per_task"],
            # Calibration
            "ece_trajectory":      res["ece_trajectory"],
            # Task 2.11 ablation flag
            "bn_reset_at_boundaries": res["bn_reset_at_boundaries"],
        })

        log_fn(
            f"[generic_cl] seed={seed}  forgetting={res['mean_forgetting']:.4f}"
            f"  accuracy={res['mean_accuracy']:.4f}"
        )

        # Release CUDA memory between seeds
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return seed_results, forgetting_list, accuracy_list
