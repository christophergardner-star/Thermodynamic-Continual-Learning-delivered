from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tar_lab.multimodal_payloads import _paired_t_stats
from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.thermoobserver import ActivationThermoObserver
from tar_optimizer_backend import build_optimizer, maybe_apply_optimizer_safety
from tar_validation_mode import (
    DEFAULT_MIN_SEEDS,
    PRIMARY_CLAIM,
    VALIDATION_BACKBONE,
    VALIDATION_BATCH_SIZE,
    VALIDATION_DATASET,
    VALIDATION_EPOCHS,
    VALIDATION_FRONTIER_ID,
    VALIDATION_METHOD_ORDER,
    VALIDATION_PAPER_ID,
    ensure_validation_method_order_exact,
    load_state,
    method_matrix,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def _jaf(accuracy: float, forgetting: float) -> float:
    return float(accuracy - forgetting)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _chance_level(num_classes: int) -> float:
    return 1.0 / float(max(num_classes, 1))


class _LabelSubset(Dataset):
    def __init__(
        self,
        dataset: Any,
        indices: list[int],
        label_map: Optional[dict[int, int]] = None,
    ) -> None:
        self._dataset = dataset
        self._indices = indices
        self._label_map = label_map or {}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self._dataset[self._indices[idx]]
        remapped = self._label_map.get(int(label), int(label))
        return image, remapped


class _CLTrunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 4 * 4, 128)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.relu(self.fc(x))


class _ResNet18Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as models

        rn = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(rn.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def _tcl_lr_adjustment(
    observer: ActivationThermoObserver,
    base_lr: float,
    *,
    ordered_scale: float,
    disordered_scale: float,
) -> float:
    regime = str(getattr(observer, "current_regime", "unknown") or "unknown")
    if regime == "ordered":
        return base_lr * ordered_scale
    if regime == "disordered":
        return base_lr * disordered_scale
    return base_lr


def _strict_classification(
    *,
    mean_delta_forgetting: float,
    p_value: float,
    effect_size: float,
    consistency_fraction: float,
    no_collapse: bool,
) -> str:
    if (
        mean_delta_forgetting < 0.0
        and p_value < 0.05
        and effect_size > 0.8
        and consistency_fraction >= 0.7
        and no_collapse
    ):
        return "BREAKTHROUGH"
    if mean_delta_forgetting < 0.0 and consistency_fraction >= 0.6 and no_collapse:
        return "DIRECTIONAL"
    if mean_delta_forgetting > 0.0 or not no_collapse:
        return "ADVERSE"
    return "NULL"


def _bool_count(records: list[dict[str, Any]], key: str) -> int:
    return sum(1 for rec in records if bool(rec.get(key)))


def _collapse_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    suspicious = any(bool(row.get("collapse_summary", {}).get("suspicious")) for row in rows)
    hidden = any(bool(row.get("collapse_summary", {}).get("hidden_collapse")) for row in rows)
    constant = any(bool(row.get("collapse_summary", {}).get("constant_prediction")) for row in rows)
    uniform = any(bool(row.get("collapse_summary", {}).get("near_uniform_outputs")) for row in rows)
    near_random = any(bool(row.get("collapse_summary", {}).get("near_random_accuracy")) for row in rows)
    return {
        "suspicious": suspicious,
        "hidden_collapse": hidden,
        "constant_prediction": constant,
        "near_uniform_outputs": uniform,
        "near_random_accuracy": near_random,
        "suspicious_seed_count": _bool_count(
            [row.get("collapse_summary", {}) for row in rows if isinstance(row.get("collapse_summary"), dict)],
            "suspicious",
        ),
    }


def _prediction_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probs = probabilities.clamp(min=1e-12)
    return -(probs * probs.log()).sum(dim=1)


def _load_split_cifar10(config: ContinualLearningBenchmarkConfig, workspace: str) -> tuple[Any, Any, list[Dataset], list[Dataset]]:
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for HPC validation suite.") from exc

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    if config.augmentation == "flip_normalize":
        train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
    else:
        train_transform = transform

    cache_dir = str(Path(workspace) / "dataset_artifacts" / "split_cifar10")
    full_train = torchvision.datasets.CIFAR10(
        root=cache_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    full_test = torchvision.datasets.CIFAR10(
        root=cache_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_targets = np.array(full_train.targets)
    test_targets = np.array(full_test.targets)
    task_train_subsets: list[Dataset] = []
    task_test_subsets: list[Dataset] = []
    class_order = config.class_order[: config.n_tasks]
    for task_classes in class_order:
        label_map = {int(label): idx for idx, label in enumerate(task_classes)}
        train_indices = [int(idx) for idx, label in enumerate(train_targets) if int(label) in label_map]
        test_indices = [int(idx) for idx, label in enumerate(test_targets) if int(label) in label_map]
        task_train_subsets.append(_LabelSubset(full_train, train_indices, label_map))
        task_test_subsets.append(_LabelSubset(full_test, test_indices, label_map))

    return full_train, full_test, task_train_subsets, task_test_subsets


def run_single_validation_method(
    *,
    workspace: str,
    seed: int,
    method_key: str,
    method_name: str,
    method: str,
    config_overrides: dict[str, Any],
    backbone: str = VALIDATION_BACKBONE,
    epochs: int = VALIDATION_EPOCHS,
    batch_size: int = VALIDATION_BATCH_SIZE,
) -> dict[str, Any]:
    cfg = ContinualLearningBenchmarkConfig(
        dataset=VALIDATION_DATASET,
        setting="task_incremental",
        seed=seed,
        train_epochs_per_task=epochs,
        batch_size=batch_size,
        **config_overrides,
    )
    _seed_everything(seed)
    _, _, task_train_subsets, task_test_subsets = _load_split_cifar10(cfg, workspace)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backbone == "resnet18":
        trunk: nn.Module = _ResNet18Trunk().to(device)
        feat_dim = 512
    else:
        trunk = _CLTrunk().to(device)
        feat_dim = 128
    heads = nn.ModuleList([nn.Linear(feat_dim, 2) for _ in range(cfg.n_tasks)]).to(device)
    all_params = list(trunk.parameters()) + list(heads.parameters())

    ewc_fisher: dict[str, torch.Tensor] = {}
    ewc_params: dict[str, torch.Tensor] = {}
    si_omega: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device="cpu") for name, param in trunk.named_parameters()
    }
    si_prev_params: dict[str, torch.Tensor] = {
        name: param.detach().cpu().clone() for name, param in trunk.named_parameters()
    }
    si_path_integral: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device="cpu") for name, param in trunk.named_parameters()
    }

    observer: ActivationThermoObserver | None = None
    if method == "tcl" and cfg.tcl_governor_enabled:
        observer = ActivationThermoObserver(
            trunk,
            stat_window_size=5,
            alpha=cfg.tcl_alpha,
            warmup_batches=60 if backbone == "resnet18" else 0,
            compute_dpr=backbone != "resnet18",
        )

    accuracy_matrix: dict[int, dict[int, float]] = {}
    confusion_by_train_task: dict[int, dict[int, list[list[int]]]] = {}
    collapse_diagnostics: dict[int, dict[int, dict[str, Any]]] = {}
    base_lr = 0.01
    tcl_trace: list[dict[str, Any]] = []
    tcl_anchor_params: dict[str, torch.Tensor] = {}
    tcl_anchor_dpr = 0.0

    for train_task_idx in range(cfg.n_tasks):
        if observer is not None and method == "tcl" and cfg.tcl_reset_on_task_boundary and train_task_idx > 0:
            observer.reset_for_new_task()

        optimizer = build_optimizer(
            all_params,
            backend=cfg.optimizer_backend,
            lr=base_lr,
            weight_decay=1e-4,
            momentum=0.9,
            workspace=workspace,
            run_label=f"hpc-validation-{method_key}-seed{seed}-task{train_task_idx}",
            config=cfg.optimizer_backend_config,
        )
        train_loader = DataLoader(task_train_subsets[train_task_idx], batch_size=cfg.batch_size, shuffle=True)
        for epoch_idx in range(cfg.train_epochs_per_task):
            trunk.train()
            heads[train_task_idx].train()
            epoch_regimes: list[str] = []
            epoch_lrs: list[float] = []
            epoch_sigmas: list[float] = []
            epoch_sigma_stars: list[float] = []
            epoch_rhos: list[float] = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                reps = trunk(batch_x)
                logits = heads[train_task_idx](reps)
                loss = F.cross_entropy(logits, batch_y)

                if method == "ewc" and ewc_fisher:
                    ewc_loss = torch.zeros((), device=device)
                    for name, param in trunk.named_parameters():
                        fisher = ewc_fisher.get(name)
                        params_ref = ewc_params.get(name)
                        if fisher is None or params_ref is None:
                            continue
                        ewc_loss = ewc_loss + (fisher.to(device) * (param - params_ref.to(device)).pow(2)).sum()
                    loss = loss + (cfg.ewc_lambda / 2.0) * ewc_loss

                if method == "si":
                    si_reg = torch.zeros((), device=device)
                    for name, param in trunk.named_parameters():
                        si_reg = si_reg + (
                            si_omega[name].to(device) * (param - si_prev_params[name].to(device)).pow(2)
                        ).sum()
                    loss = loss + cfg.si_c * si_reg

                if method == "tcl" and tcl_anchor_params and tcl_anchor_dpr > 0.0 and cfg.tcl_penalty_lambda > 0.0:
                    tcl_reg = torch.zeros((), device=device)
                    for name, param in trunk.named_parameters():
                        ref = tcl_anchor_params.get(name)
                        if ref is not None:
                            tcl_reg = tcl_reg + (param.float() - ref.to(device).float()).pow(2).sum()
                    loss = loss + cfg.tcl_penalty_lambda * tcl_anchor_dpr * tcl_reg

                optimizer.zero_grad()
                loss.backward()

                if method == "si":
                    for name, param in trunk.named_parameters():
                        if param.grad is not None:
                            si_path_integral[name] = si_path_integral[name] + (
                                -param.grad.detach() * (param.detach() - si_prev_params[name].to(device))
                            ).abs().cpu()

                if observer is not None and method == "tcl":
                    snap = observer.step(optimizer)
                    adj_lr = _tcl_lr_adjustment(
                        observer,
                        base_lr,
                        ordered_scale=cfg.tcl_ordered_lr_scale,
                        disordered_scale=cfg.tcl_disordered_lr_scale,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = adj_lr
                    epoch_regimes.append(observer.current_regime)
                    epoch_lrs.append(adj_lr)
                    if snap.layer_metrics:
                        epoch_sigmas.append(sum(lm.sigma for lm in snap.layer_metrics) / len(snap.layer_metrics))
                        epoch_sigma_stars.append(sum(lm.sigma_star for lm in snap.layer_metrics) / len(snap.layer_metrics))
                        epoch_rhos.append(snap.regime_rho)

                maybe_apply_optimizer_safety(optimizer, all_params)
                optimizer.step()

            if observer is not None and method == "tcl" and epoch_regimes:
                counts = Counter(epoch_regimes)
                total = len(epoch_regimes)
                entry: dict[str, Any] = {
                    "task": train_task_idx,
                    "epoch": epoch_idx,
                    "n_batches": total,
                    "regime_pct": {
                        regime: round(counts[regime] / total, 3)
                        for regime in ["ordered", "critical", "disordered", "unknown"]
                        if counts[regime] > 0
                    },
                    "dominant_regime": counts.most_common(1)[0][0],
                    "mean_lr": round(sum(epoch_lrs) / len(epoch_lrs), 6),
                }
                if epoch_sigmas:
                    entry["mean_sigma"] = round(sum(epoch_sigmas) / len(epoch_sigmas), 8)
                    entry["mean_sigma_star"] = round(sum(epoch_sigma_stars) / len(epoch_sigma_stars), 8)
                    entry["mean_rho"] = round(sum(epoch_rhos) / len(epoch_rhos), 4)
                tcl_trace.append(entry)

        if method == "ewc":
            trunk.eval()
            ewc_fisher_new: dict[str, torch.Tensor] = {
                name: torch.zeros_like(param, device="cpu") for name, param in trunk.named_parameters()
            }
            sample_loader = DataLoader(task_train_subsets[train_task_idx], batch_size=32, shuffle=True)
            count = 0
            for batch_x, batch_y in sample_loader:
                if count >= 100:
                    break
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                reps = trunk(batch_x)
                log_probs = F.log_softmax(heads[train_task_idx](reps), dim=1)
                for item_idx in range(batch_x.size(0)):
                    log_probs[item_idx, batch_y[item_idx]].backward(
                        retain_graph=item_idx < batch_x.size(0) - 1
                    )
                    for name, param in trunk.named_parameters():
                        if param.grad is not None:
                            ewc_fisher_new[name] = ewc_fisher_new[name] + param.grad.detach().cpu().pow(2)
                    trunk.zero_grad()
                    heads[train_task_idx].zero_grad()
                count += batch_x.size(0)
            n_samples = max(count, 1)
            for name, fisher in ewc_fisher_new.items():
                normalized = fisher / n_samples
                if name in ewc_fisher:
                    ewc_fisher[name] = ewc_fisher[name] + normalized
                else:
                    ewc_fisher[name] = normalized
            ewc_params = {name: param.detach().cpu().clone() for name, param in trunk.named_parameters()}

        if method == "si":
            for name, param in trunk.named_parameters():
                current = param.detach().cpu()
                denom = (current - si_prev_params[name].cpu()).pow(2) + cfg.si_xi
                si_omega[name] = (si_omega[name].cpu() + si_path_integral[name].cpu() / denom).clamp(min=0)
                si_path_integral[name].zero_()
            si_prev_params = {name: param.detach().cpu().clone() for name, param in trunk.named_parameters()}

        if observer is not None and method == "tcl" and train_task_idx == 0:
            observer.anchor_snapshot()

        if observer is not None and method == "tcl" and cfg.tcl_penalty_lambda > 0.0:
            tcl_anchor_params = {
                name: param.detach().cpu().clone()
                for name, param in trunk.named_parameters()
            }
            tcl_anchor_dpr = observer.anchor_effective_dimensionality

        trunk.eval()
        row: dict[int, float] = {}
        row_confusion: dict[int, list[list[int]]] = {}
        row_collapse: dict[int, dict[str, Any]] = {}
        for eval_task_idx in range(cfg.n_tasks):
            test_loader = DataLoader(task_test_subsets[eval_task_idx], batch_size=256, shuffle=False)
            correct = 0
            total = 0
            confusion = [[0, 0], [0, 0]]
            pred_counter: Counter[int] = Counter()
            entropy_total = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    reps = trunk(batch_x)
                    logits = heads[eval_task_idx](reps)
                    probs = F.softmax(logits, dim=1)
                    preds = probs.argmax(1)
                    correct += int((preds == batch_y).sum().item())
                    total += int(batch_y.size(0))
                    entropy_total += float(_prediction_entropy(probs).sum().item())
                    for truth, pred in zip(batch_y.tolist(), preds.tolist()):
                        confusion[int(truth)][int(pred)] += 1
                        pred_counter[int(pred)] += 1
            acc = correct / max(total, 1)
            chance = _chance_level(2)
            dominant_pred_fraction = (max(pred_counter.values()) / total) if total else 0.0
            mean_entropy = entropy_total / max(total, 1)
            normalized_entropy = mean_entropy / max(math.log(2), 1e-12)
            row[eval_task_idx] = acc
            row_confusion[eval_task_idx] = confusion
            row_collapse[eval_task_idx] = {
                "chance_level": chance,
                "accuracy": acc,
                "constant_prediction": dominant_pred_fraction >= 0.98,
                "dominant_prediction_fraction": dominant_pred_fraction,
                "near_uniform_outputs": normalized_entropy >= 0.98,
                "mean_entropy": mean_entropy,
                "normalized_entropy": normalized_entropy,
                "near_random_accuracy": acc <= chance + 0.05,
                "prediction_histogram": {str(k): int(v) for k, v in pred_counter.items()},
            }
        accuracy_matrix[train_task_idx] = row
        confusion_by_train_task[train_task_idx] = row_confusion
        collapse_diagnostics[train_task_idx] = row_collapse

    per_task_accuracy: list[float] = []
    per_task_forgetting: list[float] = []
    final_task_idx = cfg.n_tasks - 1
    for task_idx in range(cfg.n_tasks):
        acc_final = accuracy_matrix[final_task_idx][task_idx]
        peak = max(accuracy_matrix[t][task_idx] for t in range(task_idx, cfg.n_tasks))
        forgetting = peak - acc_final
        per_task_accuracy.append(acc_final)
        per_task_forgetting.append(forgetting)

    final_mean_accuracy = _mean(per_task_accuracy)
    mean_forgetting = _mean(per_task_forgetting)
    legacy_jaf = final_mean_accuracy * (1.0 - mean_forgetting)
    jaf = _jaf(final_mean_accuracy, mean_forgetting)

    final_task_collapse = collapse_diagnostics.get(final_task_idx, {})
    any_constant = any(bool(diag.get("constant_prediction")) for diag in final_task_collapse.values() if isinstance(diag, dict))
    any_uniform = any(bool(diag.get("near_uniform_outputs")) for diag in final_task_collapse.values() if isinstance(diag, dict))
    near_random = final_mean_accuracy <= 0.55
    hidden_collapse = mean_forgetting <= 0.05 and near_random
    suspicious = any_constant or any_uniform or near_random or hidden_collapse

    trace_path = ""
    if observer is not None:
        trace_dir = Path(workspace) / "tar_state" / "validation" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / f"{method_key}_seed{seed}.json"
        trace_data = {
            "seed": seed,
            "method_key": method_key,
            "alpha": observer.alpha,
            "final_regime": observer.current_regime,
            "anchor_effective_dimensionality": observer.anchor_effective_dimensionality,
            "epoch_trace": tcl_trace,
        }
        trace_file.write_text(json.dumps(trace_data, indent=2), encoding="utf-8")
        trace_path = str(trace_file)

    return {
        "seed": seed,
        "method_key": method_key,
        "method_name": method_name,
        "method": method,
        "config_overrides": dict(config_overrides),
        "mean_forgetting": mean_forgetting,
        "final_mean_accuracy": final_mean_accuracy,
        "jaf": jaf,
        "legacy_jaf": legacy_jaf,
        "per_task_accuracy": per_task_accuracy,
        "per_task_forgetting": per_task_forgetting,
        "final_confusion_matrices": confusion_by_train_task.get(final_task_idx, {}),
        "final_task_collapse_diagnostics": final_task_collapse,
        "collapse_summary": {
            "constant_prediction": any_constant,
            "near_uniform_outputs": any_uniform,
            "near_random_accuracy": near_random,
            "hidden_collapse": hidden_collapse,
            "suspicious": suspicious,
        },
        "thermodynamic_trace_path": trace_path,
    }


def _aggregate_method_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forgetting = [float(row["mean_forgetting"]) for row in rows]
    accuracy = [float(row["final_mean_accuracy"]) for row in rows]
    jaf = [float(row["jaf"]) for row in rows]
    return {
        "n_seeds": len(rows),
        "forgetting_mean": _mean(forgetting),
        "forgetting_std": _std(forgetting),
        "acc_mean": _mean(accuracy),
        "acc_std": _std(accuracy),
        "jaf_mean": _mean(jaf),
        "jaf_std": _std(jaf),
        "collapse_summary": _collapse_summary(rows),
    }


def _pairwise_claim(
    primary_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_by_seed = {int(row["seed"]): row for row in primary_rows}
    comparator_by_seed = {int(row["seed"]): row for row in comparator_rows}
    shared_seeds = sorted(set(primary_by_seed) & set(comparator_by_seed))
    deltas_forgetting: list[float] = []
    deltas_accuracy: list[float] = []
    deltas_jaf: list[float] = []
    for seed in shared_seeds:
        p_row = primary_by_seed[seed]
        c_row = comparator_by_seed[seed]
        deltas_forgetting.append(float(p_row["mean_forgetting"]) - float(c_row["mean_forgetting"]))
        deltas_accuracy.append(float(p_row["final_mean_accuracy"]) - float(c_row["final_mean_accuracy"]))
        deltas_jaf.append(float(p_row["jaf"]) - float(c_row["jaf"]))
    t_stat, p_value, effect_size = _paired_t_stats(deltas_forgetting)
    consistency_fraction = (
        sum(1 for delta in deltas_forgetting if delta < 0.0) / len(deltas_forgetting)
        if deltas_forgetting else 0.0
    )
    no_collapse = not _collapse_summary(primary_rows)["suspicious"]
    classification = _strict_classification(
        mean_delta_forgetting=_mean(deltas_forgetting),
        p_value=p_value,
        effect_size=effect_size,
        consistency_fraction=consistency_fraction,
        no_collapse=no_collapse,
    )
    return {
        "shared_seeds": shared_seeds,
        "mean_delta_forgetting": _mean(deltas_forgetting),
        "mean_delta_accuracy": _mean(deltas_accuracy),
        "mean_delta_jaf": _mean(deltas_jaf),
        "p_value_forgetting": p_value,
        "t_stat_forgetting": t_stat,
        "cohens_d_forgetting": effect_size,
        "consistency_fraction": consistency_fraction,
        "primary_no_collapse": no_collapse,
        "classification": classification,
    }


def _default_output_root(workspace: Path) -> Path:
    state = load_state(workspace)
    bundle_root = str(state.get("bundle_root", "") or "")
    if bundle_root:
        return Path(bundle_root) / "outputs"
    return workspace / "tar_state" / "validation" / "adhoc_outputs"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path: Path, raw: dict[str, Any]) -> None:
    primary = raw["pairwise"]["high_penalty_conservative"]
    lines = [
        "# HPC Claim Replication Report",
        "",
        f"- Generated: `{raw['completed_at']}`",
        f"- Claim: {PRIMARY_CLAIM}",
        f"- Seeds: `{raw['seeds']}`",
        "",
        "## Primary claim status",
        "",
        f"- Status: `{raw['claim_assessment']['status']}`",
        f"- Confidence: `{raw['claim_assessment']['confidence_level']}`",
        f"- Accuracy stable: `{raw['claim_assessment']['accuracy_stable']}`",
        "",
        "## Pairwise comparison summary",
        "",
        "| Comparator | Δ forgetting | Δ accuracy | Δ JAF | p | d | class |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for comp, record in primary.items():
        lines.append(
            f"| {comp} | {record['mean_delta_forgetting']:+.4f} | {record['mean_delta_accuracy']:+.4f} | "
            f"{record['mean_delta_jaf']:+.4f} | {record['p_value_forgetting']:.4f} | "
            f"{record['cohens_d_forgetting']:.3f} | {record['classification']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hpc_validation_suite(
    workspace: str,
    *,
    seeds: list[int] | None = None,
    backbone: str = VALIDATION_BACKBONE,
    epochs: int = VALIDATION_EPOCHS,
    batch_size: int = VALIDATION_BATCH_SIZE,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    output_root = _default_output_root(workspace_path)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / "replication_suite"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_list = list(seeds or DEFAULT_MIN_SEEDS)
    methods = [rec for rec in method_matrix() if rec["key"] in VALIDATION_METHOD_ORDER]
    methods.sort(key=lambda rec: VALIDATION_METHOD_ORDER.index(rec["key"]))
    ensure_validation_method_order_exact([str(rec["key"]) for rec in methods])

    per_method: dict[str, list[dict[str, Any]]] = {rec["key"]: [] for rec in methods}
    flat_rows: list[dict[str, Any]] = []
    total_method_steps = len(seed_list) * len(methods)
    completed_method_steps = 0
    completed_seed_count = 0

    for seed_index, seed in enumerate(seed_list):
        for method_index, rec in enumerate(methods):
            row = run_single_validation_method(
                workspace=workspace,
                seed=seed,
                method_key=str(rec["key"]),
                method_name=str(rec["label"]),
                method=str(rec["method"]),
                config_overrides=dict(rec["config_overrides"]),
                backbone=backbone,
                epochs=epochs,
                batch_size=batch_size,
            )
            per_method[str(rec["key"])].append(row)
            flat_rows.append(
                {
                    "seed": row["seed"],
                    "method_key": row["method_key"],
                    "method_name": row["method_name"],
                    "method": row["method"],
                    "forgetting": row["mean_forgetting"],
                    "accuracy": row["final_mean_accuracy"],
                    "jaf": row["jaf"],
                    "legacy_jaf": row["legacy_jaf"],
                    "suspicious": row["collapse_summary"]["suspicious"],
                    "constant_prediction": row["collapse_summary"]["constant_prediction"],
                    "near_uniform_outputs": row["collapse_summary"]["near_uniform_outputs"],
                    "near_random_accuracy": row["collapse_summary"]["near_random_accuracy"],
                }
            )
            completed_method_steps += 1
            if method_index == len(methods) - 1:
                completed_seed_count = seed_index + 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "seeds_done": completed_seed_count,
                        "seeds_total": len(seed_list),
                        "tasks_done": completed_method_steps,
                        "latest_accs": [f"{value:.3f}" for value in row["per_task_accuracy"]],
                        "forgetting_so_far": [
                            float(item["mean_forgetting"])
                            for item in per_method["high_penalty_conservative"]
                        ],
                        "current_seed": seed,
                        "current_method": rec["key"],
                        "methods_done": completed_method_steps,
                        "methods_total": total_method_steps,
                    }
                )

    aggregate = {
        method_key: _aggregate_method_rows(rows)
        for method_key, rows in per_method.items()
    }

    primary_pairwise = {}
    hpc_rows = per_method["high_penalty_conservative"]
    for method_key, rows in per_method.items():
        if method_key == "high_penalty_conservative":
            continue
        primary_pairwise[method_key] = _pairwise_claim(hpc_rows, rows)

    accuracy_stable = (
        aggregate["high_penalty_conservative"]["acc_mean"] > 0.70
        and not aggregate["high_penalty_conservative"]["collapse_summary"]["suspicious"]
    )
    verified = (
        primary_pairwise["tcl_baseline"]["classification"] == "BREAKTHROUGH"
        and primary_pairwise["ewc_lambda_100"]["classification"] == "BREAKTHROUGH"
        and accuracy_stable
    )
    rejected = (
        primary_pairwise["tcl_baseline"]["mean_delta_forgetting"] >= 0.0
        or primary_pairwise["ewc_lambda_100"]["mean_delta_forgetting"] >= 0.0
        or not accuracy_stable
    )
    claim_status = "VERIFIED" if verified else "REJECTED" if rejected else "INCONCLUSIVE"
    confidence_level = "high" if verified else "low" if rejected else "provisional"

    raw = {
        "created_at": _now_iso(),
        "completed_at": _now_iso(),
        "claim": PRIMARY_CLAIM,
        "problem_id": VALIDATION_FRONTIER_ID,
        "paper_id": VALIDATION_PAPER_ID,
        "dataset": VALIDATION_DATASET,
        "setting": "task_incremental",
        "backbone": backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "seeds": seed_list,
        "method_order": [rec["key"] for rec in methods],
        "per_method": per_method,
        "aggregate": aggregate,
        "pairwise": {"high_penalty_conservative": primary_pairwise},
        "claim_assessment": {
            "status": claim_status,
            "confidence_level": confidence_level,
            "holds_across_seeds": (
                min(record["consistency_fraction"] for record in primary_pairwise.values())
                if primary_pairwise else 0.0
            ),
            "accuracy_stable": accuracy_stable,
            "dataset_specific": True,
            "hyperparameter_artefact_risk": (
                "reduced"
                if primary_pairwise["ewc_lambda_1000"]["classification"] in {"BREAKTHROUGH", "DIRECTIONAL"}
                else "present"
            ),
        },
        "verdict_key": claim_status,
        "verdict": (
            "Primary claim verified on Split-CIFAR-10 under strict criteria."
            if claim_status == "VERIFIED"
            else "Primary claim rejected under strict criteria."
            if claim_status == "REJECTED"
            else "Primary claim remains provisional pending stronger replication."
        ),
        "final_claim_line": f"CLAIM {claim_status}",
    }

    json_path = run_dir / "replication_results.json"
    csv_path = run_dir / "replication_results.csv"
    confusion_path = run_dir / "confusion_matrices.json"
    claim_path = run_dir / "claim_assessment.json"
    report_path = run_dir / "replication_report.md"

    confusion_payload = {
        method_key: {
            str(row["seed"]): row["final_confusion_matrices"]
            for row in rows
        }
        for method_key, rows in per_method.items()
    }

    _write_csv(csv_path, flat_rows)
    confusion_path.write_text(json.dumps(confusion_payload, indent=2), encoding="utf-8")
    claim_path.write_text(json.dumps(raw["claim_assessment"], indent=2), encoding="utf-8")
    _write_markdown_report(report_path, raw)
    raw["artifacts"] = {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "confusion_path": str(confusion_path),
        "claim_path": str(claim_path),
        "report_path": str(report_path),
    }
    json_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(raw["final_claim_line"], flush=True)
    return raw
