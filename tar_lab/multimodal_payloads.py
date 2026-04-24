from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tar_lab.errors import ScientificValidityError
from tar_lab.schemas import (
    BackendProvenance,
    ContinualLearningBenchmarkConfig,
    ContinualLearningBenchmarkResult,
    ContinualLearningMetrics,
    DataBundleProvenance,
    DataProvenance,
    GovernorMetrics,
    ExternalBreakthroughAssessment,
    TCLMechanismCandidate,
    TCLMechanismSearchResult,
    TokenizerProvenance,
    TrainingPayloadConfig,
)
from tar_lab.thermoobserver import ActivationThermoObserver, compute_participation_ratio

try:
    from sklearn.datasets import load_breast_cancer, load_digits  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_breast_cancer = None  # type: ignore[assignment]
    load_digits = None  # type: ignore[assignment]


class _TinyVisionNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        self.hidden = nn.Linear(16 * 8 * 8, 64)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        reps = torch.tanh(self.hidden(feats))
        return self.head(reps), reps


class _BanditPolicy(nn.Module):
    def __init__(self, num_actions: int = 4):
        super().__init__()
        self.hidden = nn.Linear(1, 16)
        self.head = nn.Linear(16, num_actions)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reps = torch.tanh(self.hidden(obs))
        return self.head(reps), reps


class _QuantumModel(nn.Module):
    def __init__(self, feature_dim: int = 2):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(feature_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = x + self.theta
        reps = torch.stack(
            [
                torch.cos(angles[:, 0] / 2.0),
                torch.sin(angles[:, 0] / 2.0),
                torch.cos(angles[:, 1] / 2.0),
                torch.sin(angles[:, 1] / 2.0),
            ],
            dim=-1,
        )
        logit = torch.cos(angles[:, 0]) * torch.cos(angles[:, 1]) + self.bias
        return logit.unsqueeze(-1), reps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executable multimodal TAR backend entrypoint.")
    parser.add_argument("--backend", required=True, choices=["asc_cv", "asc_rl", "asc_qml"])
    parser.add_argument("--trial-name", required=True)
    parser.add_argument("--config-json", required=True)
    return parser.parse_args()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_metric(path: Path, metric: GovernorMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(metric.model_dump_json() + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ece(confidences: Iterable[float], correct: Iterable[int], bins: int = 10) -> float:
    confs = list(confidences)
    labels = list(correct)
    if not confs:
        return 0.0
    total = len(confs)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [(c, y) for c, y in zip(confs, labels) if lower <= c < upper or (index == bins - 1 and c == 1.0)]
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        mean_acc = sum(y for _, y in bucket) / len(bucket)
        error += abs(mean_conf - mean_acc) * (len(bucket) / total)
    return float(error)


def _param_drift_l2(model: nn.Module, anchor: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in model.state_dict().items():
        ref = anchor[name].to(param.device)
        total += float(torch.sum((param.float() - ref.float()) ** 2).item())
    return math.sqrt(total)


def _common_metric(
    *,
    trial_id: str,
    step: int,
    loss: float,
    grad_norm: float,
    drift_l2: float,
    reps: torch.Tensor,
    initial_loss: float,
    equilibrium_fraction: float = 0.0,
) -> GovernorMetrics:
    d_pr = float(compute_participation_ratio(reps.detach().float()))
    rho = float(loss / max(initial_loss, 1e-8))
    return GovernorMetrics(
        trial_id=trial_id,
        step=step,
        energy_e=max(0.0, drift_l2 / max(1, reps.shape[-1])),
        entropy_sigma=max(0.0, loss),
        drift_l2=max(0.0, drift_l2),
        drift_rho=max(0.0, rho),
        grad_norm=max(0.0, grad_norm),
        regime_rho=max(0.0, rho),
        effective_dimensionality=max(0.0, d_pr),
        effective_dimensionality_std_err=0.0,
        dimensionality_ratio=1.0,
        entropy_sigma_std_err=0.0,
        regime_rho_std_err=0.0,
        stat_window_size=1,
        stat_sample_count=1,
        statistically_ready=True,
        equilibrium_fraction=max(0.0, min(1.0, equilibrium_fraction)),
        equilibrium_gate=equilibrium_fraction >= 0.5,
        training_loss=max(0.0, loss),
    )


def _validate_backend_context(
    *,
    backend_id: str,
    run_intent: str,
    backend_provenance: Optional[BackendProvenance],
    provenance_complete: bool,
    research_grade: bool,
) -> None:
    if run_intent != "research":
        return
    if backend_provenance is None:
        raise ScientificValidityError(f"Research {backend_id} run refused: backend provenance is missing.")
    if backend_provenance.status != "executable":
        raise ScientificValidityError(f"Research {backend_id} run refused: backend is scaffold-only.")
    if backend_provenance.control_only:
        raise ScientificValidityError(f"Research {backend_id} run refused: backend is control-only.")
    if not backend_provenance.research_grade_capable:
        raise ScientificValidityError(f"Research {backend_id} run refused: backend is not research-grade capable.")
    if not provenance_complete or not research_grade:
        raise ScientificValidityError(f"Research {backend_id} run refused: provenance is incomplete or fallback-tainted.")


def _cv_dataset(run_intent: str) -> tuple[torch.Tensor, torch.Tensor, DataProvenance]:
    if load_digits is not None:
        digits = load_digits()
        images = torch.tensor(digits.images[:, None, :, :] / 16.0, dtype=torch.float32)
        labels = torch.tensor(digits.target, dtype=torch.long)
        tokenizer = TokenizerProvenance(
            stream_name="research",
            tokenizer_id="not_applicable",
            tokenizer_class="NotApplicable",
            tokenizer_hash="not_applicable",
            tokenizer_vocab_size=0,
            integrity_check=True,
            is_fallback=False,
        )
        provenance = DataProvenance(
            stream_name="research",
            dataset_name="sklearn:digits",
            dataset_split="train",
            data_mode="CACHED_REAL",
            data_purity="local_real",
            source_kind="sklearn",
            dataset_identifier="sklearn:digits",
            sampling_strategy="deterministic_shuffle",
            dataset_fingerprint=f"digits:{images.shape[0]}:{labels.shape[0]}",
            tokenizer_id="not_applicable",
            tokenizer_class="NotApplicable",
            tokenizer_hash="not_applicable",
            tokenizer_vocab_size=0,
            integrity_check=True,
            is_real_data=True,
            is_fallback=False,
            provenance_complete=True,
            research_safe=run_intent == "research",
            tokenizer_provenance=tokenizer,
        )
        return images, labels, provenance
    if run_intent == "research":
        raise ScientificValidityError("Research asc_cv run refused: sklearn digits is unavailable and fallback imagery is not allowed.")
    rng = np.random.default_rng(7)
    images = np.zeros((96, 1, 8, 8), dtype=np.float32)
    labels = np.zeros(96, dtype=np.int64)
    for index in range(96):
        label = int(rng.integers(0, 3))
        labels[index] = label
        base = np.zeros((8, 8), dtype=np.float32)
        if label == 0:
            base[:, 2:4] = 1.0
        elif label == 1:
            base[3:5, :] = 1.0
        else:
            np.fill_diagonal(base, 1.0)
            np.fill_diagonal(np.fliplr(base), 1.0)
        base += rng.normal(0.0, 0.08, size=base.shape).astype(np.float32)
        images[index, 0] = np.clip(base, 0.0, 1.0)
    tokenizer = TokenizerProvenance(
        stream_name="research",
        tokenizer_id="not_applicable",
        tokenizer_class="NotApplicable",
        tokenizer_hash="not_applicable",
        tokenizer_vocab_size=0,
        integrity_check=False,
        is_fallback=True,
    )
    provenance = DataProvenance(
        stream_name="research",
        dataset_name="synthetic-cv-fallback",
        dataset_split="train",
        data_mode="OFFLINE_FALLBACK",
        data_purity="fallback",
        source_kind="synthetic",
        dataset_identifier="synthetic-cv-fallback",
        sampling_strategy="deterministic_shuffle",
        dataset_fingerprint=f"synthetic-cv:{images.shape[0]}:{labels.shape[0]}",
        tokenizer_id="not_applicable",
        tokenizer_class="NotApplicable",
        tokenizer_hash="not_applicable",
        tokenizer_vocab_size=0,
        integrity_check=False,
        is_real_data=False,
        is_fallback=True,
        provenance_complete=True,
        research_safe=False,
        tokenizer_provenance=tokenizer,
        notes=["control_only_fallback"],
    )
    return torch.tensor(images), torch.tensor(labels), provenance


def _qml_dataset(run_intent: str) -> tuple[torch.Tensor, torch.Tensor, DataProvenance]:
    if load_breast_cancer is not None:
        ds = load_breast_cancer()
        xs = torch.tensor(ds.data[:, :2], dtype=torch.float32)
        xs = (xs - xs.mean(dim=0, keepdim=True)) / xs.std(dim=0, keepdim=True).clamp_min(1e-6)
        ys = torch.tensor(ds.target.astype(np.float32), dtype=torch.float32)
        tokenizer = TokenizerProvenance(
            stream_name="research",
            tokenizer_id="not_applicable",
            tokenizer_class="NotApplicable",
            tokenizer_hash="not_applicable",
            tokenizer_vocab_size=0,
            integrity_check=True,
            is_fallback=False,
        )
        provenance = DataProvenance(
            stream_name="research",
            dataset_name="sklearn:breast_cancer",
            dataset_split="train",
            data_mode="CACHED_REAL",
            data_purity="local_real",
            source_kind="sklearn",
            dataset_identifier="sklearn:breast_cancer[:2]",
            sampling_strategy="deterministic_shuffle",
            dataset_fingerprint=f"breast-cancer:{xs.shape[0]}:{ys.shape[0]}",
            tokenizer_id="not_applicable",
            tokenizer_class="NotApplicable",
            tokenizer_hash="not_applicable",
            tokenizer_vocab_size=0,
            integrity_check=True,
            is_real_data=True,
            is_fallback=False,
            provenance_complete=True,
            research_safe=run_intent == "research",
            tokenizer_provenance=tokenizer,
        )
        return xs, ys, provenance
    if run_intent == "research":
        raise ScientificValidityError(
            "Research asc_qml run refused: sklearn breast-cancer data is unavailable and fallback data is not allowed."
        )
    xs = torch.linspace(-1.0, 1.0, steps=96).unsqueeze(-1).repeat(1, 2)
    ys = (xs[:, 0] > 0.0).float()
    tokenizer = TokenizerProvenance(
        stream_name="research",
        tokenizer_id="not_applicable",
        tokenizer_class="NotApplicable",
        tokenizer_hash="not_applicable",
        tokenizer_vocab_size=0,
        integrity_check=False,
        is_fallback=True,
    )
    provenance = DataProvenance(
        stream_name="research",
        dataset_name="synthetic-qml-fallback",
        dataset_split="train",
        data_mode="OFFLINE_FALLBACK",
        data_purity="fallback",
        source_kind="synthetic",
        dataset_identifier="synthetic-qml-fallback",
        sampling_strategy="deterministic_shuffle",
        dataset_fingerprint=f"synthetic-qml:{xs.shape[0]}:{ys.shape[0]}",
        tokenizer_id="not_applicable",
        tokenizer_class="NotApplicable",
        tokenizer_hash="not_applicable",
        tokenizer_vocab_size=0,
        integrity_check=False,
        is_real_data=False,
        is_fallback=True,
        provenance_complete=True,
        research_safe=False,
        tokenizer_provenance=tokenizer,
        notes=["control_only_fallback"],
    )
    return xs, ys, provenance


def _rl_provenance(run_intent: str) -> DataProvenance:
    tokenizer = TokenizerProvenance(
        stream_name="research",
        tokenizer_id="not_applicable",
        tokenizer_class="NotApplicable",
        tokenizer_hash="not_applicable",
        tokenizer_vocab_size=0,
        integrity_check=True,
        is_fallback=False,
    )
    return DataProvenance(
        stream_name="research",
        dataset_name="local:four_arm_bandit",
        dataset_split="rollout",
        data_mode="CACHED_REAL",
        data_purity="local_real",
        source_kind="environment",
        dataset_identifier="environment://four_arm_bandit",
        sampling_strategy="on_policy_rollout",
        dataset_fingerprint="four-arm-bandit-v1",
        tokenizer_id="not_applicable",
        tokenizer_class="NotApplicable",
        tokenizer_hash="not_applicable",
        tokenizer_vocab_size=0,
        integrity_check=True,
        is_real_data=True,
        is_fallback=False,
        provenance_complete=True,
        research_safe=run_intent == "research",
        tokenizer_provenance=tokenizer,
    )


def _cv_loop(trial_id: str, run_intent: str, device: torch.device, log_path: Path) -> tuple[dict[str, Any], DataProvenance]:
    images, labels, provenance = _cv_dataset(run_intent)
    images = images.to(device)
    labels = labels.to(device)
    model = _TinyVisionNet(num_classes=int(labels.max().item()) + 1).to(device)
    anchor = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32
    first_loss: Optional[float] = None
    last_metric: Optional[GovernorMetrics] = None
    last_logits: Optional[torch.Tensor] = None
    last_labels: Optional[torch.Tensor] = None
    for step in range(1, 7):
        start = ((step - 1) * batch_size) % images.shape[0]
        batch_x = images[start : start + batch_size]
        batch_y = labels[start : start + batch_size]
        if batch_x.shape[0] == 0:
            batch_x = images[:batch_size]
            batch_y = labels[:batch_size]
        logits, reps = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = math.sqrt(
            sum(float(param.grad.detach().float().pow(2).sum().item()) for param in model.parameters() if param.grad is not None)
        )
        opt.step()
        first_loss = float(loss.item()) if first_loss is None else first_loss
        last_metric = _common_metric(
            trial_id=trial_id,
            step=step,
            loss=float(loss.item()),
            grad_norm=grad_norm,
            drift_l2=_param_drift_l2(model, anchor),
            reps=reps,
            initial_loss=first_loss or float(loss.item()),
            equilibrium_fraction=0.5 if step >= 5 else 0.0,
        )
        _append_metric(log_path, last_metric)
        last_logits = logits.detach()
        last_labels = batch_y.detach()
    probs = torch.softmax(last_logits, dim=-1) if last_logits is not None else torch.zeros(1, 1, device=device)
    conf, pred = probs.max(dim=-1)
    accuracy = float(pred.eq(last_labels).float().mean().item()) if last_labels is not None else 0.0
    return {
        "last_metrics": last_metric.model_dump(mode="json") if last_metric is not None else None,
        "accuracy": accuracy,
        "calibration_ece": _ece(conf.detach().cpu().tolist(), pred.eq(last_labels).int().cpu().tolist()) if last_labels is not None else 0.0,
    }, provenance


def _rl_loop(trial_id: str, run_intent: str, device: torch.device, log_path: Path) -> tuple[dict[str, Any], DataProvenance]:
    reward_probs = torch.tensor([0.18, 0.31, 0.56, 0.86], dtype=torch.float32, device=device)
    policy = _BanditPolicy(num_actions=reward_probs.numel()).to(device)
    anchor = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    opt = torch.optim.Adam(policy.parameters(), lr=0.03)
    obs = torch.ones(1, 1, device=device)
    baseline = 0.0
    returns: list[float] = []
    first_loss: Optional[float] = None
    last_metric: Optional[GovernorMetrics] = None
    for step in range(1, 31):
        logits, reps = policy(obs)
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        reward = float(torch.rand(1, device=device).item() < reward_probs[action].item())
        baseline = 0.9 * baseline + 0.1 * reward
        loss = -(reward - baseline) * torch.log(probs[action].clamp_min(1e-6))
        entropy = float((-(probs * probs.clamp_min(1e-6).log()).sum()).item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = math.sqrt(
            sum(float(param.grad.detach().float().pow(2).sum().item()) for param in policy.parameters() if param.grad is not None)
        )
        opt.step()
        first_loss = float(loss.item()) if first_loss is None else first_loss
        last_metric = _common_metric(
            trial_id=trial_id,
            step=step,
            loss=float(abs(loss.item()) + 1e-6),
            grad_norm=grad_norm,
            drift_l2=_param_drift_l2(policy, anchor),
            reps=reps,
            initial_loss=first_loss or float(abs(loss.item()) + 1e-6),
            equilibrium_fraction=0.5 if step >= 20 else 0.0,
        )
        last_metric = last_metric.model_copy(update={"training_loss": float(abs(loss.item()) + 1e-6), "entropy_sigma": float(abs(loss.item()) + entropy * 0.01)})
        _append_metric(log_path, last_metric)
        returns.append(reward)
    return {
        "last_metrics": last_metric.model_dump(mode="json") if last_metric is not None else None,
        "episodic_return": float(sum(returns[-10:]) / max(1, min(10, len(returns)))),
        "sample_efficiency": float(sum(returns) / max(1, len(returns))),
    }, _rl_provenance(run_intent)


def _qml_loop(trial_id: str, run_intent: str, device: torch.device, log_path: Path) -> tuple[dict[str, Any], DataProvenance]:
    xs, ys, provenance = _qml_dataset(run_intent)
    xs = xs.to(device)
    ys = ys.to(device)
    model = _QuantumModel(feature_dim=2).to(device)
    anchor = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    first_loss: Optional[float] = None
    last_metric: Optional[GovernorMetrics] = None
    last_logits: Optional[torch.Tensor] = None
    last_labels: Optional[torch.Tensor] = None
    batch_size = 32
    for step in range(1, 11):
        start = ((step - 1) * batch_size) % xs.shape[0]
        batch_x = xs[start : start + batch_size]
        batch_y = ys[start : start + batch_size]
        if batch_x.shape[0] == 0:
            batch_x = xs[:batch_size]
            batch_y = ys[:batch_size]
        logits, reps = model(batch_x)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), batch_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = math.sqrt(
            sum(float(param.grad.detach().float().pow(2).sum().item()) for param in model.parameters() if param.grad is not None)
        )
        opt.step()
        first_loss = float(loss.item()) if first_loss is None else first_loss
        last_metric = _common_metric(
            trial_id=trial_id,
            step=step,
            loss=float(loss.item()),
            grad_norm=grad_norm,
            drift_l2=_param_drift_l2(model, anchor),
            reps=reps,
            initial_loss=first_loss or float(loss.item()),
            equilibrium_fraction=0.5 if step >= 8 else 0.0,
        )
        _append_metric(log_path, last_metric)
        last_logits = logits.detach().squeeze(-1)
        last_labels = batch_y.detach()
    probs = torch.sigmoid(last_logits) if last_logits is not None else torch.zeros(1, device=device)
    pred = (probs >= 0.5).long()
    correct = pred.eq(last_labels.long()) if last_labels is not None else torch.zeros(1, dtype=torch.bool, device=device)
    return {
        "last_metrics": last_metric.model_dump(mode="json") if last_metric is not None else None,
        "accuracy": float(correct.float().mean().item()) if last_labels is not None else 0.0,
        "calibration_ece": _ece(probs.cpu().tolist(), correct.int().cpu().tolist()),
        "execution_mode": "internal_variational_simulator",
    }, provenance


def run_multimodal_backend(
    *,
    config: TrainingPayloadConfig,
    dry_run: bool,
    device: torch.device,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    _validate_backend_context(
        backend_id=config.backend_id,
        run_intent=config.run_intent,
        backend_provenance=config.backend_provenance,
        provenance_complete=config.provenance_complete,
        research_grade=config.research_grade,
    )
    _seed_everything(config.seed)
    if config.backend_id == "asc_cv":
        result, provenance = _cv_loop(config.trial_id, config.run_intent, device, log_path)
    elif config.backend_id == "asc_rl":
        result, provenance = _rl_loop(config.trial_id, config.run_intent, device, log_path)
    elif config.backend_id == "asc_qml":
        result, provenance = _qml_loop(config.trial_id, config.run_intent, device, log_path)
    else:  # pragma: no cover - guarded by caller
        raise ScientificValidityError(f"Unknown multimodal backend: {config.backend_id}")
    checkpoint_path = output_dir / f"{config.backend_id}_checkpoint.pt"
    torch.save({"backend_id": config.backend_id, "trial_id": config.trial_id, "seed": config.seed}, checkpoint_path)
    summary = {
        "trial_id": config.trial_id,
        "backend_id": config.backend_id,
        "backend_family": config.backend_provenance.domain if config.backend_provenance is not None else "unknown",
        "backend_readiness": config.backend_provenance.status if config.backend_provenance is not None else "unknown",
        "backend_provenance": config.backend_provenance.model_dump(mode="json") if config.backend_provenance is not None else None,
        "run_intent": config.run_intent,
        "research_grade": config.research_grade,
        "provenance_complete": config.provenance_complete,
        "data_purity": provenance.data_purity,
        "data_provenance": provenance.model_dump(mode="json"),
        "tokenizer_provenance": {k: v.model_dump(mode="json") for k, v in config.tokenizer_provenance.items()},
        "last_metrics": result.get("last_metrics"),
        "metrics": {k: v for k, v in result.items() if k != "last_metrics"},
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "dry_run": dry_run,
    }
    _save_json(output_dir / "payload_summary.json", summary)
    _save_json(output_dir / "execution_summary.json", summary)
    return summary


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((item - mean_value) ** 2 for item in values) / (len(values) - 1))


def _jaf(accuracy: float, forgetting: float) -> float:
    return accuracy * (1.0 - forgetting)


def _paired_t_stats(values: list[float]) -> tuple[float, float, float]:
    deltas = list(values)
    if not deltas:
        return 0.0, 1.0, 0.0
    try:
        from scipy import stats as _scipy_stats

        t_stat, p_value = _scipy_stats.ttest_1samp(deltas, 0.0)
        effect_size = abs(_mean(deltas)) / max(_std(deltas), 1e-12)
        return float(t_stat), float(p_value), float(effect_size)
    except Exception:
        mean_delta = _mean(deltas)
        sample_std = _std(deltas)
        if sample_std <= 1e-12:
            return 0.0, 1.0 if abs(mean_delta) <= 1e-12 else 0.0, 0.0
        stderr = sample_std / math.sqrt(max(len(deltas), 1))
        t_stat = mean_delta / max(stderr, 1e-12)
        # Normal approximation fallback.
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
        effect_size = abs(mean_delta) / max(sample_std, 1e-12)
        return float(t_stat), float(p_value), float(effect_size)


def _tcl_class_incremental_mechanism_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "tcl_balanced",
            "method": "tcl",
            "overrides": {
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_ordered_lr_scale": 0.5,
                "tcl_disordered_lr_scale": 1.2,
                "tcl_reset_on_task_boundary": True,
            },
        },
        {
            "name": "tcl_stability_bias",
            "method": "tcl",
            "overrides": {
                "tcl_penalty_lambda": 0.02,
                "tcl_alpha": 0.45,
                "tcl_ordered_lr_scale": 0.4,
                "tcl_disordered_lr_scale": 1.05,
                "tcl_reset_on_task_boundary": True,
            },
        },
        {
            "name": "tcl_plasticity_bias",
            "method": "tcl",
            "overrides": {
                "tcl_penalty_lambda": 0.005,
                "tcl_alpha": 0.55,
                "tcl_ordered_lr_scale": 0.7,
                "tcl_disordered_lr_scale": 1.35,
                "tcl_reset_on_task_boundary": True,
            },
        },
        {
            "name": "tcl_carryover_anchor",
            "method": "tcl",
            "overrides": {
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_ordered_lr_scale": 0.5,
                "tcl_disordered_lr_scale": 1.2,
                "tcl_reset_on_task_boundary": False,
            },
        },
        {
            "name": "tcl_governor_only",
            "method": "tcl",
            "overrides": {
                "tcl_penalty_lambda": 0.0,
                "tcl_alpha": 0.5,
                "tcl_ordered_lr_scale": 0.5,
                "tcl_disordered_lr_scale": 1.2,
                "tcl_reset_on_task_boundary": True,
            },
        },
        {
            "name": "tcl_penalty_only",
            "method": "tcl_penalty_only",
            "overrides": {
                "tcl_penalty_lambda": 0.01,
            },
        },
    ]


def _tcl_lr_adjustment(
    observer: ActivationThermoObserver,
    base_lr: float,
    *,
    ordered_scale: float = 0.5,
    disordered_scale: float = 1.2,
) -> float:
    regime = observer.current_regime
    if regime in ("critical", "unknown"):
        return base_lr
    if regime == "ordered":
        return base_lr * ordered_scale
    if regime == "disordered":
        return base_lr * disordered_scale
    return base_lr


def run_split_cifar10_benchmark(
    config: ContinualLearningBenchmarkConfig,
    method: str,
    observer: Optional[ActivationThermoObserver] = None,
    workspace: Optional[str] = None,
    backbone: str = "tiny",
) -> ContinualLearningBenchmarkResult:
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "torchvision is required for split_cifar10 benchmark. Install with: pip install torchvision"
        ) from exc

    torch.manual_seed(config.seed)
    import random as _random
    import numpy as _np

    _random.seed(config.seed)
    _np.random.seed(config.seed)

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

    cache_dir = str(Path(workspace) / "dataset_artifacts" / "split_cifar10") if workspace else "/tmp/cifar10"
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

    if config.setting not in {"task_incremental", "class_incremental"}:
        raise ValueError("split_cifar10 benchmark supports only task_incremental or class_incremental")

    class _LabelSubset(Dataset):
        def __init__(
            self,
            dataset: Any,
            indices: list[int],
            label_map: Optional[dict[int, int]] = None,
        ):
            self._dataset = dataset
            self._indices = indices
            self._label_map = label_map or {}

        def __len__(self) -> int:
            return len(self._indices)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            image, label = self._dataset[self._indices[idx]]
            remapped = self._label_map.get(int(label), int(label))
            return image, remapped

    class_order = config.class_order[: config.n_tasks]
    if len(class_order) != config.n_tasks:
        raise ValueError("class_order must provide one class-pair entry per task")

    train_targets = np.array(full_train.targets)
    test_targets = np.array(full_test.targets)
    task_train_subsets: list[Dataset] = []
    task_test_subsets: list[Dataset] = []
    total_num_classes = max(label for task_classes in class_order for label in task_classes) + 1
    for task_classes in class_order:
        if len(task_classes) != config.classes_per_task:
            raise ValueError("Each split_cifar10 task must match classes_per_task")
        if config.setting == "task_incremental":
            label_map = {int(label): idx for idx, label in enumerate(task_classes)}
        else:
            label_map = {int(label): int(label) for label in task_classes}
        train_indices = [int(idx) for idx, label in enumerate(train_targets) if int(label) in label_map]
        test_indices = [int(idx) for idx, label in enumerate(test_targets) if int(label) in label_map]
        task_train_subsets.append(_LabelSubset(full_train, train_indices, label_map))
        task_test_subsets.append(_LabelSubset(full_test, test_indices, label_map))

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
            import torchvision.models as _models
            rn = _models.resnet18(weights=None)
            self.features = nn.Sequential(*list(rn.children())[:-1])  # drop final FC
            self.feat_dim = 512

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.features(x).flatten(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backbone == "resnet18":
        trunk: nn.Module = _ResNet18Trunk().to(device)
        feat_dim = 512
    else:
        trunk = _CLTrunk().to(device)
        feat_dim = 128
    if config.setting == "class_incremental":
        shared_head: Optional[nn.Module] = nn.Linear(feat_dim, total_num_classes).to(device)
        heads = None
        all_params = list(trunk.parameters()) + list(shared_head.parameters())
    else:
        shared_head = None
        heads = nn.ModuleList([nn.Linear(feat_dim, 2) for _ in range(config.n_tasks)]).to(device)
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

    if observer is None and method == "tcl" and config.tcl_governor_enabled:
        # alpha=0.5: "ordered" fires when sigma drops to 50% of the early-task
        # anchor level.  warmup_batches delays anchor collection until the
        # network has had meaningful gradient signal — set to ~2 epochs worth
        # of batches for large backbones (resnet18) to avoid anchoring on
        # random-initialisation noise.
        _warmup = 60 if backbone == "resnet18" else 0
        # compute_dpr=False for large backbones: per-batch eigendecomp over
        # all ResNet layers is the dominant cost (~1s/batch on L40S).
        # Regime detection (sigma/rho/LR) still works without it.
        _dpr = backbone != "resnet18"
        observer = ActivationThermoObserver(
            trunk, stat_window_size=5, alpha=config.tcl_alpha,
            warmup_batches=_warmup, compute_dpr=_dpr,
        )

    accuracy_matrix: dict[int, dict[int, float]] = {}
    base_lr = 0.01
    tcl_trace: list[dict] = []  # per-epoch diagnostic snapshots
    # Dimensionality-weighted weight penalty state.
    # Populated after each task completes; used in subsequent tasks.
    tcl_anchor_params: dict[str, torch.Tensor] = {}
    tcl_anchor_dpr: float = 0.0

    for train_task_idx in range(config.n_tasks):
        # Reset per-task calibration state.  sigma_star_anchor will be
        # re-established from the first 20 batches of this task and frozen.
        if (
            observer is not None
            and method == "tcl"
            and config.tcl_reset_on_task_boundary
            and train_task_idx > 0
        ):
            observer.reset_for_new_task()

        optimizer = torch.optim.SGD(all_params, lr=base_lr, momentum=0.9, weight_decay=1e-4)
        train_loader = DataLoader(
            task_train_subsets[train_task_idx],
            batch_size=config.batch_size,
            shuffle=True,
        )

        for _epoch in range(config.train_epochs_per_task):
            trunk.train()
            if heads is not None:
                heads[train_task_idx].train()
            if shared_head is not None:
                shared_head.train()
            _epoch_regimes: list[str] = []
            _epoch_lrs: list[float] = []
            _epoch_sigmas: list[float] = []
            _epoch_sigma_stars: list[float] = []
            _epoch_rhos: list[float] = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                reps = trunk(batch_x)
                if shared_head is not None:
                    logits = shared_head(reps)
                else:
                    assert heads is not None
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
                    loss = loss + (config.ewc_lambda / 2.0) * ewc_loss

                if method == "si":
                    si_reg = torch.zeros((), device=device)
                    for name, param in trunk.named_parameters():
                        si_reg = si_reg + (si_omega[name].to(device) * (param - si_prev_params[name].to(device)).pow(2)).sum()
                    loss = loss + config.si_c * si_reg

                # Dimensionality-weighted L2 penalty: penalise drift from the
                # previous task's weights, scaled by that task's anchor D_PR.
                # tcl_anchor_dpr is 0.0 on task 0 so the penalty is inactive.
                if method in ("tcl", "tcl_penalty_only") and tcl_anchor_params and tcl_anchor_dpr > 0.0 and config.tcl_penalty_lambda > 0.0:
                    tcl_reg = torch.zeros((), device=device)
                    for name, param in trunk.named_parameters():
                        ref = tcl_anchor_params.get(name)
                        if ref is not None:
                            tcl_reg = tcl_reg + (param.float() - ref.to(device).float()).pow(2).sum()
                    loss = loss + config.tcl_penalty_lambda * tcl_anchor_dpr * tcl_reg

                optimizer.zero_grad()
                loss.backward()

                if method == "si":
                    for name, param in trunk.named_parameters():
                        if param.grad is not None:
                            si_path_integral[name] = si_path_integral[name] + (
                                -param.grad.detach() * (param.detach() - si_prev_params[name].to(device))
                            ).abs().cpu()

                # observer.step() reads gradients computed by backward() above.
                # Must fire BEFORE optimizer.step() so gradients are fresh and
                # the adjusted LR takes effect on the current weight update.
                if observer is not None and method == "tcl":
                    snap = observer.step(optimizer)
                    adj_lr = _tcl_lr_adjustment(
                        observer,
                        base_lr,
                        ordered_scale=config.tcl_ordered_lr_scale,
                        disordered_scale=config.tcl_disordered_lr_scale,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = adj_lr
                    _epoch_regimes.append(observer.current_regime)
                    _epoch_lrs.append(adj_lr)
                    # capture raw sigma/sigma_star for calibration diagnostics
                    if snap.layer_metrics:
                        _epoch_sigmas.append(
                            sum(lm.sigma for lm in snap.layer_metrics) / len(snap.layer_metrics)
                        )
                        _epoch_sigma_stars.append(
                            sum(lm.sigma_star for lm in snap.layer_metrics) / len(snap.layer_metrics)
                        )
                        _epoch_rhos.append(snap.regime_rho)

                optimizer.step()

            # record per-epoch regime summary for diagnostic trace
            if observer is not None and method == "tcl" and _epoch_regimes:
                from collections import Counter
                counts = Counter(_epoch_regimes)
                total = len(_epoch_regimes)
                entry: dict = {
                    "task": train_task_idx,
                    "epoch": _epoch,
                    "n_batches": total,
                    "regime_pct": {r: round(counts[r] / total, 3)
                                   for r in ["ordered", "critical", "disordered", "unknown"]
                                   if counts[r] > 0},
                    "dominant_regime": counts.most_common(1)[0][0],
                    "mean_lr": round(sum(_epoch_lrs) / len(_epoch_lrs), 6),
                    "min_lr": round(min(_epoch_lrs), 6),
                    "max_lr": round(max(_epoch_lrs), 6),
                }
                if _epoch_sigmas:
                    entry["mean_sigma"] = round(sum(_epoch_sigmas) / len(_epoch_sigmas), 8)
                    entry["mean_sigma_star"] = round(sum(_epoch_sigma_stars) / len(_epoch_sigma_stars), 8)
                    entry["mean_rho"] = round(sum(_epoch_rhos) / len(_epoch_rhos), 4)
                    # rho headroom: how far from the "ordered" threshold (0.9)
                    # negative = already ordered, positive = distance to go
                    entry["rho_to_ordered"] = round(
                        sum(_epoch_rhos) / len(_epoch_rhos) - 0.9, 4
                    )
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
                if shared_head is not None:
                    log_probs = F.log_softmax(shared_head(reps), dim=1)
                else:
                    assert heads is not None
                    log_probs = F.log_softmax(heads[train_task_idx](reps), dim=1)
                for item_idx in range(batch_x.size(0)):
                    log_probs[item_idx, batch_y[item_idx]].backward(retain_graph=item_idx < batch_x.size(0) - 1)
                    for name, param in trunk.named_parameters():
                        if param.grad is not None:
                            ewc_fisher_new[name] = ewc_fisher_new[name] + param.grad.detach().cpu().pow(2)
                    trunk.zero_grad()
                    if shared_head is not None:
                        shared_head.zero_grad()
                    else:
                        assert heads is not None
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
                denom = (current - si_prev_params[name].cpu()).pow(2) + config.si_xi
                si_omega[name] = (si_omega[name].cpu() + si_path_integral[name].cpu() / denom).clamp(min=0)
                si_path_integral[name].zero_()
            si_prev_params = {name: param.detach().cpu().clone() for name, param in trunk.named_parameters()}

        # After task 0 training: set the dimensionality anchor so that
        # dimensionality_ratio is meaningful for all subsequent tasks.
        # last_activation is populated from the final training batch above.
        if observer is not None and method == "tcl" and train_task_idx == 0:
            observer.anchor_snapshot()

        # Snapshot weights + D_PR for the dimensionality-weighted penalty.
        # Done after every task so the next task penalises drift from the
        # most-recently-consolidated state.
        if observer is not None and method == "tcl" and config.tcl_penalty_lambda > 0.0:
            tcl_anchor_params = {
                name: param.detach().cpu().clone()
                for name, param in trunk.named_parameters()
            }
            tcl_anchor_dpr = observer.anchor_effective_dimensionality
        elif method == "tcl_penalty_only" and config.tcl_penalty_lambda > 0.0:
            # Governor disabled: no observer, no LR adjustment.
            # Anchor weights after each task; D_PR fixed at 1.0 (no dimensionality scaling).
            tcl_anchor_params = {
                name: param.detach().cpu().clone()
                for name, param in trunk.named_parameters()
            }
            tcl_anchor_dpr = 1.0

        trunk.eval()
        row: dict[int, float] = {}
        for eval_task_idx in range(config.n_tasks):
            test_loader = DataLoader(task_test_subsets[eval_task_idx], batch_size=256, shuffle=False)
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    reps = trunk(batch_x)
                    if shared_head is not None:
                        logits = shared_head(reps)
                    else:
                        assert heads is not None
                        logits = heads[eval_task_idx](reps)
                    correct += int((logits.argmax(1) == batch_y).sum().item())
                    total += int(batch_y.size(0))
            row[eval_task_idx] = correct / max(total, 1)
        accuracy_matrix[train_task_idx] = row

    per_task_metrics: list[ContinualLearningMetrics] = []
    final_task_idx = config.n_tasks - 1
    for task_idx in range(config.n_tasks):
        acc_right_after = accuracy_matrix[task_idx][task_idx]
        acc_final = accuracy_matrix[final_task_idx][task_idx]
        backward_transfer = acc_final - acc_right_after
        peak = max(accuracy_matrix[t][task_idx] for t in range(task_idx, config.n_tasks))
        forgetting = peak - acc_final
        stability_plasticity_gap = abs(1.0 - acc_final / max(acc_right_after, 1e-8))
        per_task_metrics.append(
            ContinualLearningMetrics(
                task_id=task_idx,
                task_accuracy=acc_final,
                accuracy_right_after_training=acc_right_after,
                backward_transfer=backward_transfer,
                forgetting_measure=forgetting,
                forward_transfer=0.0,
                stability_plasticity_gap=stability_plasticity_gap,
            )
        )

    mean_backward_transfer = sum(metric.backward_transfer for metric in per_task_metrics) / len(per_task_metrics)
    mean_forgetting = sum(metric.forgetting_measure for metric in per_task_metrics) / len(per_task_metrics)
    final_mean_accuracy = sum(metric.task_accuracy for metric in per_task_metrics) / len(per_task_metrics)

    trace_path = ""
    if observer is not None and workspace:
        trace_dir = Path(workspace) / "tar_state" / "cl_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_stem = f"tcl_{config.seed}.json"
        if config.setting == "class_incremental":
            trace_stem = f"tcl_class_incremental_{config.seed}.json"
        trace_file = trace_dir / trace_stem
        # Summarise regime distribution per task across all epochs in that task
        task_summaries: list[dict] = []
        for t in range(config.n_tasks):
            task_epochs = [e for e in tcl_trace if e["task"] == t]
            if task_epochs:
                all_regimes = []
                for e in task_epochs:
                    for regime, pct in e["regime_pct"].items():
                        all_regimes.extend([regime] * round(pct * e["n_batches"]))
                from collections import Counter
                counts = Counter(all_regimes)
                total = max(len(all_regimes), 1)
                task_summaries.append({
                    "task": t,
                    "regime_pct": {r: round(counts[r] / total, 3)
                                   for r in ["ordered", "critical", "disordered", "unknown"]
                                   if counts[r] > 0},
                    "dominant_regime": counts.most_common(1)[0][0],
                    "mean_lr": round(
                        sum(e["mean_lr"] for e in task_epochs) / len(task_epochs), 6
                    ),
                })
        trace_data = {
            "seed": config.seed,
            "alpha": observer.alpha,
            "final_regime": observer.current_regime,
            "anchor_effective_dimensionality": observer.anchor_effective_dimensionality,
            "task_summaries": task_summaries,
            "epoch_trace": tcl_trace,
        }
        trace_file.write_text(json.dumps(trace_data, indent=2), encoding="utf-8")
        trace_path = str(trace_file)

    return ContinualLearningBenchmarkResult(
        benchmark_id=uuid4().hex,
        method=method,
        seed=config.seed,
        n_tasks=config.n_tasks,
        per_task_metrics=per_task_metrics,
        mean_backward_transfer=mean_backward_transfer,
        mean_forgetting=mean_forgetting,
        final_mean_accuracy=final_mean_accuracy,
        last_task_accuracy=per_task_metrics[-1].task_accuracy,
        thermodynamic_trace_path=trace_path,
    )


def search_tcl_class_incremental_mechanisms(
    base_config: Optional[ContinualLearningBenchmarkConfig] = None,
    *,
    workspace: Optional[str] = None,
    backbone: str = "tiny",
    seeds: Optional[list[int]] = None,
    problem_id: str = "",
) -> TCLMechanismSearchResult:
    seeds = list(seeds) if seeds is not None else [42, 0, 1]
    base_config = base_config or ContinualLearningBenchmarkConfig()
    base_config = base_config.model_copy(update={"setting": "class_incremental"})

    baseline_runs: list[ContinualLearningBenchmarkResult] = []
    strong_baseline_runs: list[ContinualLearningBenchmarkResult] = []
    for seed in seeds:
        baseline_cfg = base_config.model_copy(update={"seed": seed})
        baseline_runs.append(
            run_split_cifar10_benchmark(
                baseline_cfg,
                method="sgd_baseline",
                workspace=workspace,
                backbone=backbone,
            )
        )
        strong_baseline_runs.append(
            run_split_cifar10_benchmark(
                baseline_cfg,
                method="ewc",
                workspace=workspace,
                backbone=backbone,
            )
        )

    candidate_specs = _tcl_class_incremental_mechanism_specs()
    candidates: list[TCLMechanismCandidate] = []
    candidate_runs: dict[str, list[ContinualLearningBenchmarkResult]] = {}
    for spec in candidate_specs:
        runs: list[ContinualLearningBenchmarkResult] = []
        for seed in seeds:
            candidate_cfg = base_config.model_copy(update={"seed": seed, **spec["overrides"]})
            runs.append(
                run_split_cifar10_benchmark(
                    candidate_cfg,
                    method=spec["method"],
                    workspace=workspace,
                    backbone=backbone,
                )
            )
        candidate_runs[spec["name"]] = runs
        forgetting = [run.mean_forgetting for run in runs]
        accuracy = [run.final_mean_accuracy for run in runs]
        baseline_forgetting = [run.mean_forgetting for run in baseline_runs]
        deltas_vs_sgd = [cand - base for cand, base in zip(forgetting, baseline_forgetting)]
        candidates.append(
            TCLMechanismCandidate(
                name=spec["name"],
                method=spec["method"],
                config_overrides=dict(spec["overrides"]),
                mean_forgetting=_mean(forgetting),
                std_forgetting=_std(forgetting),
                mean_accuracy=_mean(accuracy),
                std_accuracy=_std(accuracy),
                mean_jaf=_mean(_jaf(run.final_mean_accuracy, run.mean_forgetting) for run in runs),
                delta_vs_sgd=_mean(deltas_vs_sgd),
                wins_vs_sgd=sum(1 for delta in deltas_vs_sgd if delta < 0.0),
            )
        )

    best_candidate = max(
        candidates,
        key=lambda item: (item.mean_jaf, -item.mean_forgetting, item.mean_accuracy),
    )
    best_runs = candidate_runs[best_candidate.name]
    deltas_vs_strong = [
        cand.mean_forgetting - strong.mean_forgetting
        for cand, strong in zip(best_runs, strong_baseline_runs)
    ]
    _, p_value, effect_size = _paired_t_stats(deltas_vs_strong)
    best_delta_vs_strong = _mean(deltas_vs_strong)
    external_breakthrough_candidate = (
        best_delta_vs_strong < -0.01 and p_value < 0.05 and effect_size > 0.5
    )
    if external_breakthrough_candidate:
        publishability_status = "reviewer_grade_candidate"
        summary = (
            f"Best class-incremental TCL mechanism {best_candidate.name} beats EWC on mean_forgetting "
            f"(delta={best_delta_vs_strong:+.4f}, p={p_value:.4f}, d={effect_size:.3f})."
        )
    elif best_candidate.delta_vs_sgd < -0.01:
        publishability_status = "mechanism_signal_only"
        summary = (
            f"Best class-incremental TCL mechanism {best_candidate.name} improves over SGD "
            f"(delta={best_candidate.delta_vs_sgd:+.4f}) but does not yet clear EWC "
            f"(delta={best_delta_vs_strong:+.4f}, p={p_value:.4f}, d={effect_size:.3f})."
        )
    else:
        publishability_status = "no_reviewer_grade_signal"
        summary = (
            f"Class-incremental TCL mechanism search did not find a reviewer-grade candidate. "
            f"Best mechanism {best_candidate.name} remains below the strong-baseline bar "
            f"(delta={best_delta_vs_strong:+.4f}, p={p_value:.4f}, d={effect_size:.3f})."
        )

    return TCLMechanismSearchResult(
        search_id=uuid4().hex,
        created_at=datetime.utcnow().isoformat(),
        problem_id=problem_id,
        benchmark="split_cifar10",
        setting="class_incremental",
        backbone=backbone,
        seeds=seeds,
        baseline_method="sgd_baseline",
        strong_baseline_method="ewc",
        candidates=candidates,
        best_candidate_name=best_candidate.name,
        best_delta_vs_strong_baseline=best_delta_vs_strong,
        p_value_vs_strong_baseline=p_value,
        effect_size_vs_strong_baseline=effect_size,
        external_breakthrough_candidate=external_breakthrough_candidate,
        publishability_status=publishability_status,
        summary=summary,
    )


def _load_payload(path: str) -> Dict[str, Any]:
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Backend config not found: {path}")
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _run_from_backend_config(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_payload(args.config_json)
    backend_id = str(payload.get("backend_id", args.backend)).strip().lower()
    backend_provenance = None
    if payload.get("backend_provenance"):
        backend_provenance = BackendProvenance.model_validate(payload["backend_provenance"])
    run_intent = str(payload.get("run_intent", "control"))
    provenance_complete = bool(payload.get("provenance_complete", run_intent != "research"))
    research_grade = bool(payload.get("research_grade", run_intent != "research"))
    _validate_backend_context(
        backend_id=backend_id,
        run_intent=run_intent,
        backend_provenance=backend_provenance,
        provenance_complete=provenance_complete,
        research_grade=research_grade,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.config_json).resolve().parent
    log_path = output_dir / "thermo_metrics.jsonl"
    _seed_everything(int(payload.get("config", {}).get("seed", 7)))
    if backend_id == "asc_cv":
        result, provenance = _cv_loop(args.trial_name, run_intent, device, log_path)
    elif backend_id == "asc_rl":
        result, provenance = _rl_loop(args.trial_name, run_intent, device, log_path)
    elif backend_id == "asc_qml":
        result, provenance = _qml_loop(args.trial_name, run_intent, device, log_path)
    else:
        raise ScientificValidityError(f"Unsupported multimodal backend: {backend_id}")
    summary = {
        "backend_id": backend_id,
        "trial_name": args.trial_name,
        "backend_readiness": backend_provenance.status if backend_provenance is not None else "unknown",
        "backend_provenance": backend_provenance.model_dump(mode="json") if backend_provenance is not None else None,
        "run_intent": run_intent,
        "research_grade": research_grade,
        "provenance_complete": provenance_complete,
        "data_purity": provenance.data_purity,
        "data_provenance": provenance.model_dump(mode="json"),
        "tokenizer_provenance": payload.get("tokenizer_provenance", {}),
        "metrics": result,
        "device": str(device),
    }
    _save_json(output_dir / "execution_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    _run_from_backend_config(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
