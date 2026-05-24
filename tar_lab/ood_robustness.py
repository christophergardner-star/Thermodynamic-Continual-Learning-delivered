"""OOD robustness benchmark.

Trains a vision model on clean data then evaluates under synthetic distribution
shift (Gaussian noise at three severities, average-pool blur). Reports:
  - clean_accuracy: accuracy on the original test set
  - mean_forgetting: clean_accuracy - mean_corrupted_accuracy (the "drop")
  - final_mean_accuracy: mean accuracy across all shift types

Methods:
  - standard   : standard SGD training, no robustness intervention
  - augmentation: adds horizontal flip + random crop during training
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OODResult:
    mean_forgetting: float       # accuracy drop under distribution shift
    final_mean_accuracy: float   # mean accuracy across shift conditions
    clean_accuracy: float
    shift_accuracies: dict[str, float] = field(default_factory=dict)


# ── benchmark entry point ─────────────────────────────────────────────────────

def run_ood_robustness_benchmark(
    dataset: str,
    method: str,
    seed: int,
    backbone: str = "resnet18",
    workspace: str = ".",
    epochs: int = 20,
) -> OODResult:
    """Train on clean data; evaluate accuracy under synthetic distribution shift."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset == "cifar10_corrupted":
        return _run_cifar10_ood(method=method, seed=seed, backbone=backbone, epochs=epochs)
    raise ValueError(f"Unknown OOD dataset: {dataset}")


def _run_cifar10_ood(method: str, seed: int, backbone: str, epochs: int) -> OODResult:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("cifar10")

    base_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    aug_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    use_aug = method == "augmentation"
    train_tfm = aug_transform if use_aug else base_transform

    X_train = torch.stack([train_tfm(ex["img"]) for ex in ds["train"]])
    y_train = torch.tensor([ex["label"] for ex in ds["train"]])
    X_test  = torch.stack([base_transform(ex["img"]) for ex in ds["test"]])
    y_test  = torch.tensor([ex["label"] for ex in ds["test"]])

    model = _build_model(backbone, n_classes=10).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce    = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            ce(model(xb), yb).backward()
            opt.step()
            opt.zero_grad()
        sched.step()

    model.eval()

    def _acc(X: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            return float((model(X.to(device)).argmax(1).cpu() == y).float().mean())

    clean_acc = _acc(X_test, y_test)

    # Synthetic corruptions — no external dataset required
    shift_accs: dict[str, float] = {}
    for noise_std, name in [(0.05, "noise_light"), (0.15, "noise_medium"), (0.30, "noise_heavy")]:
        X_noisy = (X_test + torch.randn_like(X_test) * noise_std).clamp(-2.0, 2.0)
        shift_accs[name] = _acc(X_noisy, y_test)

    import torch.nn.functional as F
    X_blur = F.avg_pool2d(X_test, kernel_size=3, stride=1, padding=1)
    shift_accs["blur"] = _acc(X_blur, y_test)

    X_bright = (X_test + 0.3).clamp(-2.0, 2.0)
    shift_accs["brightness"] = _acc(X_bright, y_test)

    mean_shift = float(np.mean(list(shift_accs.values())))
    drop = clean_acc - mean_shift

    return OODResult(
        mean_forgetting=drop,
        final_mean_accuracy=mean_shift,
        clean_accuracy=clean_acc,
        shift_accuracies=shift_accs,
    )


def _build_model(backbone: str, n_classes: int):
    import torch.nn as nn
    if backbone == "resnet18":
        import torchvision.models as tvm
        m = tvm.resnet18(weights=None, num_classes=n_classes)
        return m
    # Lightweight fallback CNN
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, n_classes),
    )
