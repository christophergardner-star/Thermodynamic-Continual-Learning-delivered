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

    return {
        "mean_forgetting":     mean_forgetting,
        "mean_accuracy":       mean_accuracy,
        "final_accs_per_task": final_accs,
        "acc_matrix":          [list(row) for row in acc_matrix],
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
        )

        forgetting_list.append(res["mean_forgetting"])
        accuracy_list.append(res["mean_accuracy"])
        seed_results.append({
            "seed":      seed,
            "forgetting": res["mean_forgetting"],
            "accuracy":  res["mean_accuracy"],
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
