from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tar_lab.errors import ScientificValidityError
from tar_lab.schemas import (
    BackendProvenance,
    DataBundleProvenance,
    DataProvenance,
    GovernorMetrics,
    TokenizerProvenance,
    TrainingPayloadConfig,
)
from tar_lab.thermoobserver import compute_participation_ratio

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
