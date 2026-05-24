"""NLP continual-learning benchmark (Split-AGNews, Split-DBpedia).

Implements sequential task training with three comparison methods:
  - sgd_baseline : fine-tune on each task in turn (catastrophic forgetting reference)
  - ewc_nlp      : Kirkpatrick et al. EWC via Fisher diagonal on TF-IDF features
  - replay_nlp   : experience replay with a fixed per-task buffer

Uses TF-IDF + PyTorch MLP so it runs on the same GPU stack as the vision runners.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NLPBenchmarkResult:
    mean_forgetting: float
    final_mean_accuracy: float
    per_task_forgetting: list[float] = field(default_factory=list)
    per_task_final_acc: list[float] = field(default_factory=list)
    per_task_initial_acc: list[float] = field(default_factory=list)


# ── data loading ──────────────────────────────────────────────────────────────

def _load_tasks(dataset: str) -> list[dict]:
    from datasets import load_dataset
    if dataset == "split_agnews":
        ds = load_dataset("ag_news")
        # 4 classes split into 2 tasks of 2 classes each
        return [
            {
                "train": [(ex["text"], ex["label"]) for ex in ds["train"] if ex["label"] in cls],
                "test":  [(ex["text"], ex["label"]) for ex in ds["test"]  if ex["label"] in cls],
                "classes": cls,
            }
            for cls in [[0, 1], [2, 3]]
        ]
    if dataset == "split_dbpedia":
        ds = load_dataset("dbpedia_14")
        # 14 classes split into 7 tasks of 2 classes — capped for compute
        return [
            {
                "train": [(ex["content"], ex["label"]) for ex in ds["train"] if ex["label"] in cls][:2000],
                "test":  [(ex["content"], ex["label"]) for ex in ds["test"]  if ex["label"] in cls][:400],
                "classes": cls,
            }
            for cls in [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]]
        ]
    raise ValueError(f"Unknown NLP CL dataset: {dataset}")


# ── benchmark entry point ─────────────────────────────────────────────────────

def run_nlp_continual_benchmark(
    dataset: str,
    method: str,
    seed: int,
    epochs_per_task: int = 5,
    replay_buffer_per_task: int = 300,
    ewc_lambda: float = 200.0,
) -> NLPBenchmarkResult:
    """Train a text MLP sequentially on tasks and measure catastrophic forgetting.

    Args:
        dataset: "split_agnews" or "split_dbpedia"
        method:  "sgd_baseline" | "ewc_nlp" | "replay_nlp"
        seed:    random seed for reproducibility
        epochs_per_task: SGD passes over each task's data
        replay_buffer_per_task: examples stored per task for replay
        ewc_lambda: EWC penalty coefficient
    """
    import torch
    import torch.nn as nn
    from sklearn.feature_extraction.text import TfidfVectorizer

    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks = _load_tasks(dataset)
    n_classes = max(lbl for task in tasks for _, lbl in task["train"]) + 1
    n_features = 10_000

    # TF-IDF fitted on training text only (no label leakage — text is not labelled info)
    all_texts = [text for task in tasks for text, _ in task["train"]]
    vec = TfidfVectorizer(max_features=n_features, sublinear_tf=True, strip_accents="unicode")
    vec.fit(all_texts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensors(pairs: list) -> tuple:
        X = torch.tensor(vec.transform([t for t, _ in pairs]).toarray(), dtype=torch.float32)
        y = torch.tensor([l for _, l in pairs], dtype=torch.long)
        return X, y

    def _new_model() -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_classes),
        ).to(device)

    def _train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
               ewc_anchors: list | None = None) -> None:
        from torch.utils.data import DataLoader, TensorDataset
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        ce = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)
        for _ in range(epochs_per_task):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = ce(model(xb), yb)
                if ewc_anchors:
                    for fisher, theta_star, param in zip(
                        ewc_anchors[0], ewc_anchors[1], model.parameters()
                    ):
                        loss = loss + (ewc_lambda / 2.0) * (fisher.to(device) * (param - theta_star.to(device)) ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _eval(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            preds = model(X.to(device)).argmax(1).cpu()
        return float((preds == y).float().mean())

    def _compute_fisher(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> list:
        model.train()
        model.zero_grad()
        sample_idx = torch.randperm(len(X))[:min(500, len(X))]
        xb, yb = X[sample_idx].to(device), y[sample_idx].to(device)
        nn.CrossEntropyLoss()(model(xb), yb).backward()
        return [(p.grad.detach().cpu() ** 2) for p in model.parameters()]

    # ── per-task upper bound (single task, no interference) ──────────────────
    init_accs: list[float] = []
    for task in tasks:
        X_tr, y_tr = _to_tensors(task["train"])
        X_te, y_te = _to_tensors(task["test"])
        m0 = _new_model()
        _train(m0, X_tr, y_tr)
        init_accs.append(_eval(m0, X_te, y_te))

    # ── sequential training with chosen method ────────────────────────────────
    model = _new_model()
    ewc_fisher: list | None = None
    ewc_theta:  list | None = None
    replay_X: list[torch.Tensor] = []
    replay_y: list[torch.Tensor] = []

    for task_idx, task in enumerate(tasks):
        X_tr, y_tr = _to_tensors(task["train"])

        if method == "replay_nlp" and replay_X:
            X_tr = torch.cat([X_tr] + replay_X)
            y_tr = torch.cat([y_tr] + replay_y)

        anchors = ([ewc_fisher, ewc_theta, list(model.parameters())]
                   if method == "ewc_nlp" and ewc_fisher is not None else None)
        # anchors format expected by _train: (fisher_list, theta_list, params)
        # restructure for zip in _train
        ewc_arg = None
        if method == "ewc_nlp" and ewc_fisher is not None:
            ewc_arg = (ewc_fisher, ewc_theta)
            # pass as (fisher_list, theta_star_list) for the zip in _train
            _train(model, X_tr, y_tr, ewc_anchors=ewc_arg)
        else:
            _train(model, X_tr, y_tr)

        # Update EWC anchors after this task
        if method == "ewc_nlp":
            X_tr_raw, y_tr_raw = _to_tensors(task["train"])
            new_fisher = _compute_fisher(model, X_tr_raw, y_tr_raw)
            new_theta  = [p.detach().cpu().clone() for p in model.parameters()]
            if ewc_fisher is None:
                ewc_fisher, ewc_theta = new_fisher, new_theta
            else:
                ewc_fisher = [f + nf for f, nf in zip(ewc_fisher, new_fisher)]
                ewc_theta  = new_theta  # anchor to most recent task

        # Update replay buffer
        if method == "replay_nlp":
            X_raw, y_raw = _to_tensors(task["train"])
            idxs = torch.randperm(len(X_raw))[:replay_buffer_per_task]
            replay_X.append(X_raw[idxs])
            replay_y.append(y_raw[idxs])

    # ── final per-task accuracy ───────────────────────────────────────────────
    final_accs: list[float] = []
    for task in tasks:
        X_te, y_te = _to_tensors(task["test"])
        final_accs.append(_eval(model, X_te, y_te))

    # Forgetting = drop on all non-final tasks (last task has no prior to forget)
    per_task_forg = [init - final for init, final in zip(init_accs[:-1], final_accs[:-1])]
    mean_forgetting = float(np.mean(per_task_forg)) if per_task_forg else 0.0

    return NLPBenchmarkResult(
        mean_forgetting=mean_forgetting,
        final_mean_accuracy=float(np.mean(final_accs)),
        per_task_forgetting=per_task_forg,
        per_task_final_acc=final_accs,
        per_task_initial_acc=init_accs,
    )
