from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from asc_model import ASCConfig, ASCForCausalLM
from tar_lab.errors import ScientificValidityError
from tar_lab.governor import ThermodynamicGovernor
from tar_lab.multimodal_payloads import run_multimodal_backend
from tar_lab.schemas import CalibrationBin, CalibrationReport, DatasetManifest, GovernorMetrics, TrainingPayloadConfig
from tar_lab.thermoobserver import (
    ActivationThermoObserver,
    RegimeSnapshot,
    StatAccumulator,
    compute_activation_covariance,
)


class TinyAnchorNet(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAR training payload template")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _resolve_execution_device() -> tuple[torch.device, Optional[str]]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), None
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda"), None
    except Exception as exc:  # pragma: no cover - depends on host CUDA stack
        return torch.device("cpu"), str(exc)


def load_config(path: str) -> TrainingPayloadConfig:
    return TrainingPayloadConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_manifest(path: str | None) -> DatasetManifest | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _validate_payload_provenance(config: TrainingPayloadConfig) -> None:
    if config.run_intent != "research":
        return
    if config.backend_provenance is None:
        raise ScientificValidityError("Research payload execution refused: backend provenance is missing.")
    if config.backend_provenance.status != "executable":
        raise ScientificValidityError(
            f"Research payload execution refused: backend '{config.backend_id}' is scaffold-only."
        )
    if config.backend_provenance.control_only:
        raise ScientificValidityError(
            f"Research payload execution refused: backend '{config.backend_id}' is control-only."
        )
    if not config.backend_provenance.research_grade_capable:
        raise ScientificValidityError(
            f"Research payload execution refused: backend '{config.backend_id}' is not validated for research-grade execution."
        )
    if not config.provenance_complete or not config.research_grade:
        raise ScientificValidityError(
            "Research payload execution refused: provenance is incomplete or fallback-tainted."
        )


def load_or_create_anchor(model: nn.Module, anchor_path: Path) -> Dict[str, torch.Tensor]:
    expected = {name: tensor.detach().clone().cpu() for name, tensor in model.state_dict().items()}
    if anchor_path.exists():
        raw = torch.load(anchor_path, map_location="cpu")
        if isinstance(raw, dict):
            loaded = {k: v.detach().clone().cpu() for k, v in raw.items()}
            compatible = True
            for name, tensor in expected.items():
                candidate = loaded.get(name)
                if candidate is None or tuple(candidate.shape) != tuple(tensor.shape):
                    compatible = False
                    break
            if compatible:
                return {name: loaded[name] for name in expected}
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expected, anchor_path)
    return expected


def sample_batch(batch_size: int, feature_dim: int, strategy_family: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, feature_dim, device=device)
    if strategy_family == "elastic_anchor":
        y = (x[:, 0] > 0).long()
    elif strategy_family == "ou_drift_jitter":
        y = (x[:, -1] > 0).long()
    else:
        y = (x[:, : feature_dim // 2].sum(dim=1) > x[:, feature_dim // 2 :].sum(dim=1)).long()
    return x, y


def compute_metrics(
    config: TrainingPayloadConfig,
    model: nn.Module,
    anchor: Dict[str, torch.Tensor],
    step: int,
    regime: RegimeSnapshot | None = None,
    training_loss: float | None = None,
) -> GovernorMetrics:
    delta_sq_sum = 0.0
    anchor_sq_sum = 0.0
    grad_sq_sum = 0.0
    entropy_numerator = 0.0
    params = dict(model.named_parameters())

    for name, tensor in model.state_dict().items():
        current = tensor.detach().float().cpu()
        ref = anchor[name].float()
        delta = (current - ref).reshape(-1)
        delta_sq_sum += float((delta * delta).sum())
        anchor_sq_sum += float((ref.reshape(-1) * ref.reshape(-1)).sum())
        param = params.get(name)
        if param is not None and param.grad is not None:
            grad_vec = param.grad.detach().float().cpu().reshape(-1)
            grad_sq_sum += float((grad_vec * grad_vec).sum())
            entropy_numerator += float((delta.abs() * grad_vec.abs()).sum())

    drift_l2 = delta_sq_sum ** 0.5
    anchor_l2 = anchor_sq_sum ** 0.5
    entropy_sigma = entropy_numerator / (anchor_l2 + 1e-12)
    if regime is not None and regime.entropy_sigma > 0.0:
        entropy_sigma = regime.entropy_sigma
    statistically_ready = (
        regime.statistically_ready and step >= config.min_stat_steps
        if regime is not None
        else False
    )

    return GovernorMetrics(
        trial_id=config.trial_id,
        step=step,
        energy_e=delta_sq_sum,
        entropy_sigma=entropy_sigma,
        drift_l2=drift_l2,
        drift_rho=drift_l2 / (anchor_l2 + 1e-12),
        grad_norm=grad_sq_sum ** 0.5,
        regime_rho=regime.regime_rho if regime is not None else 0.0,
        effective_dimensionality=regime.effective_dimensionality if regime is not None else 0.0,
        effective_dimensionality_std_err=regime.effective_dimensionality_std_err if regime is not None else 0.0,
        dimensionality_ratio=regime.dimensionality_ratio if regime is not None else 0.0,
        entropy_sigma_std_err=regime.entropy_sigma_std_err if regime is not None else 0.0,
        regime_rho_std_err=regime.regime_rho_std_err if regime is not None else 0.0,
        stat_window_size=regime.stat_window_size if regime is not None else 0,
        stat_sample_count=regime.stat_sample_count if regime is not None else 0,
        statistically_ready=statistically_ready,
        equilibrium_fraction=regime.equilibrium_fraction if regime is not None else 0.0,
        equilibrium_gate=(regime.equilibrium_gate and statistically_ready) if regime is not None else False,
        training_loss=training_loss,
    )


def append_metric(path: Path, metrics: GovernorMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics.model_dump(mode="json")) + "\n")


def compute_calibration_report(
    confidences: List[float],
    correct: List[int],
    num_bins: int = 10,
) -> CalibrationReport:
    if not confidences:
        return CalibrationReport(ece=0.0, accuracy=0.0, mean_confidence=0.0, bins=[])

    bins: List[CalibrationBin] = []
    total = len(confidences)
    ece = 0.0
    for idx in range(num_bins):
        lower = idx / num_bins
        upper = (idx + 1) / num_bins
        if idx == num_bins - 1:
            chosen = [i for i, conf in enumerate(confidences) if lower <= conf <= upper]
        else:
            chosen = [i for i, conf in enumerate(confidences) if lower <= conf < upper]
        if chosen:
            bin_conf = sum(confidences[i] for i in chosen) / len(chosen)
            bin_acc = sum(correct[i] for i in chosen) / len(chosen)
            ece += abs(bin_conf - bin_acc) * (len(chosen) / total)
        else:
            bin_conf = 0.0
            bin_acc = 0.0
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(chosen),
                mean_confidence=bin_conf,
                accuracy=bin_acc,
            )
        )
    return CalibrationReport(
        ece=ece,
        accuracy=sum(correct) / total,
        mean_confidence=sum(confidences) / total,
        bins=bins,
    )


def _iter_manifest_rows(manifest: DatasetManifest | None) -> Iterable[dict[str, Any]]:
    if manifest is None:
        return []
    rows: list[dict[str, Any]] = []
    for shard in manifest.shards:
        path = _resolve_manifest_shard_path(manifest, shard.path, shard.container_path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_manifest_shard_path(
    manifest: DatasetManifest,
    host_path: str,
    container_path: Optional[str],
) -> Path:
    primary = Path(host_path)
    if primary.exists():
        return primary
    if container_path:
        container_candidate = Path(container_path)
        if container_candidate.exists():
            return container_candidate
    fallback_root = Path(os.environ.get("TAR_CONTAINER_DATA_DIR", "/data"))
    shard_name = PureWindowsPath(host_path).name or primary.name
    fallback = fallback_root / manifest.stream_name / shard_name
    if fallback.exists():
        return fallback
    return primary


def _load_sequences(manifest: DatasetManifest | None) -> list[list[int]]:
    sequences: list[list[int]] = []
    for row in _iter_manifest_rows(manifest):
        token_ids = [int(item) for item in row.get("token_ids", []) if int(item) >= 0]
        if len(token_ids) >= 2:
            sequences.append(token_ids)
    return sequences


def _split_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    rows = [list(sequence) for sequence in sequences]
    if not rows:
        return [], [], []
    shuffled = rows[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, shuffled, shuffled

    train_count = max(1, int(round(len(shuffled) * train_fraction)))
    val_count = int(round(len(shuffled) * val_fraction))
    test_count = int(round(len(shuffled) * test_fraction))
    total = train_count + val_count + test_count
    while total > len(shuffled):
        if test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        elif train_count > 1:
            train_count -= 1
        total = train_count + val_count + test_count
    if total < len(shuffled):
        train_count += len(shuffled) - total

    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count : train_count + val_count + test_count]
    if not val:
        val = train[:1] if train else shuffled[:1]
    if not test:
        test = val[:1] if val else train[:1]
    return train, val, test


def _infer_vocab_size(*sequence_groups: Sequence[Sequence[int]]) -> int:
    max_token = 255
    for group in sequence_groups:
        for sequence in group:
            if sequence:
                max_token = max(max_token, max(int(item) for item in sequence))
    return max(256, max_token + 2)


def _sample_lm_batch(
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    rng: random.Random,
) -> dict[str, torch.Tensor]:
    if not sequences:
        raise RuntimeError("No token sequences are available for the ASC payload.")
    chosen: list[list[int]] = []
    for _ in range(batch_size):
        sequence = list(sequences[rng.randrange(len(sequences))])
        if len(sequence) > seq_len:
            start_max = max(0, len(sequence) - seq_len)
            start = rng.randrange(start_max + 1) if start_max > 0 else 0
            sequence = sequence[start : start + seq_len]
        chosen.append(sequence)
    width = max(2, max(len(sequence) for sequence in chosen))
    input_ids = torch.zeros((batch_size, width), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long, device=device)
    labels = torch.full((batch_size, width), fill_value=-100, dtype=torch.long, device=device)
    for row_index, sequence in enumerate(chosen):
        row = torch.tensor(sequence, dtype=torch.long, device=device)
        length = row.shape[0]
        input_ids[row_index, :length] = row
        attention_mask[row_index, :length] = 1
        labels[row_index, :length] = row
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _refresh_base_activations(base_model: nn.Module, batch: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )


def _evaluate_lm_calibration(
    model: ASCForCausalLM,
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    num_batches: int = 1,
) -> CalibrationReport:
    model.eval()
    confidences: list[float] = []
    correct: list[int] = []
    rng = random.Random(17)
    with torch.no_grad():
        for _ in range(num_batches):
            batch = _sample_lm_batch(
                sequences,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                rng=rng,
            )
            logits = model.base(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = batch["input_ids"][:, 1:]
            shift_mask = batch["attention_mask"][:, 1:].bool()
            probs = torch.softmax(shift_logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            if shift_mask.any():
                confidences.extend(float(item) for item in conf[shift_mask].detach().cpu())
                correct.extend(int(item) for item in pred[shift_mask].eq(shift_labels[shift_mask]).detach().cpu())
    return compute_calibration_report(confidences, correct)


def _build_toy_anchor_reference(
    config: TrainingPayloadConfig,
    anchor: Dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float], float]:
    reference_model = TinyAnchorNet(config.feature_dim).to(device)
    reference_model.load_state_dict(anchor, strict=False)
    observer = ActivationThermoObserver(reference_model, stat_window_size=config.anchor_batches)
    accumulators: dict[str, StatAccumulator] = {}
    try:
        reference_model.eval()
        with torch.no_grad():
            for _ in range(config.anchor_batches):
                x, _ = sample_batch(config.batch_size, config.feature_dim, config.strategy_family, device)
                reference_model(x)
                for name, covariance in observer.current_covariances().items():
                    accumulators.setdefault(name, StatAccumulator(window_size=config.anchor_batches)).push(
                        covariance,
                        sigma=0.0,
                        rho=1.0,
                    )
    finally:
        observer.close()
    means = {
        name: stats.get_smoothed_metrics().effective_dimensionality
        for name, stats in accumulators.items()
        if stats.sample_count > 0
    }
    positives = sorted(value for value in means.values() if value > 0.0)
    anchor_effective_dimensionality = positives[len(positives) // 2] if positives else 0.0
    return means, anchor_effective_dimensionality


def _build_asc_anchor_reference(
    asc_config: ASCConfig,
    anchor_state: Dict[str, torch.Tensor],
    anchor_sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    num_batches: int,
) -> tuple[dict[str, float], float]:
    reference_model = ASCForCausalLM._build_backbone(asc_config).to(device)
    reference_model.load_state_dict(anchor_state, strict=False)
    observer = ActivationThermoObserver(reference_model, stat_window_size=num_batches)
    accumulators: dict[str, StatAccumulator] = {}
    rng = random.Random(29)
    try:
        reference_model.eval()
        with torch.no_grad():
            for _ in range(num_batches):
                batch = _sample_lm_batch(
                    anchor_sequences,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    rng=rng,
                )
                reference_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                for name, covariance in observer.current_covariances().items():
                    accumulators.setdefault(name, StatAccumulator(window_size=num_batches)).push(
                        covariance,
                        sigma=0.0,
                        rho=1.0,
                    )
    finally:
        observer.close()
    means = {
        name: stats.get_smoothed_metrics().effective_dimensionality
        for name, stats in accumulators.items()
        if stats.sample_count > 0
    }
    positives = sorted(value for value in means.values() if value > 0.0)
    anchor_effective_dimensionality = positives[len(positives) // 2] if positives else 0.0
    return means, anchor_effective_dimensionality


def _asc_config_from_payload(
    config: TrainingPayloadConfig,
    *,
    vocab_size: int,
    seq_len: int,
    dry_run: bool = False,
) -> ASCConfig:
    requested_base_model_name = str(
        config.notes.get("base_model_name", "deepseek-ai/deepseek-coder-1.3b-base")
    )
    base_model_name = requested_base_model_name
    if dry_run and requested_base_model_name != "__tiny_gpt2__":
        base_model_name = config.dry_run_backbone or "__tiny_gpt2__"
    warp_dim = int(config.notes.get("warp_dim", 128))
    warp_init_scale = float(config.notes.get("warp_init_scale", 0.05))
    ema_decay = float(config.notes.get("ema_decay", 0.995))
    consistency_lambda = float(config.notes.get("consistency_lambda", 0.3))
    adapter_mode = "full" if base_model_name == "__tiny_gpt2__" else config.adapter_mode
    if base_model_name in {"124M", "355M", "774M", "1558M"}:
        return ASCConfig.for_size(
            base_model_name,
            warp_dim=warp_dim,
            warp_init_scale=warp_init_scale,
            consistency_lambda=consistency_lambda,
            ema_decay=ema_decay,
            adapter_mode=adapter_mode,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
    overrides = {}
    if base_model_name == "__tiny_gpt2__":
        overrides = {
            "vocab_size": vocab_size,
            "n_positions": max(seq_len, 16),
            "n_ctx": max(seq_len, 16),
            "n_embd": int(config.notes.get("n_embd", 48)),
            "n_layer": int(config.notes.get("n_layer", 2)),
            "n_head": int(config.notes.get("n_head", 4)),
        }
    return ASCConfig(
        base_model_name=base_model_name,
        warp_dim=warp_dim,
        warp_init_scale=warp_init_scale,
        consistency_lambda=consistency_lambda,
        ema_decay=ema_decay,
        adapter_mode=adapter_mode,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        backbone_config_overrides=overrides,
    )


def _apply_layer_freeze(base_model: nn.Module, config: TrainingPayloadConfig) -> None:
    if config.strategy_family != "layer_freeze":
        return
    protected = config.protected_layers or ["transformer.wte", "transformer.h.0"]
    for name, param in base_model.named_parameters():
        if any(name.startswith(prefix) for prefix in protected):
            param.requires_grad = False


def _match_layer(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _anchor_regularizers(
    model: nn.Module,
    anchor: Dict[str, torch.Tensor],
    protected_layers: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    anchor_penalty = torch.zeros(1, device=device)
    fim_proxy_penalty = torch.zeros(1, device=device)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        anchor_tensor = anchor.get(name)
        if anchor_tensor is None:
            continue
        diff = param - anchor_tensor.to(device=device, dtype=param.dtype)
        mse = diff.pow(2).mean()
        anchor_penalty = anchor_penalty + mse
        if _match_layer(name, protected_layers):
            fim_proxy_penalty = fim_proxy_penalty + mse
    return anchor_penalty, fim_proxy_penalty


def _save_asc_checkpoint(
    output_dir: Path,
    model: ASCForCausalLM,
    asc_config: ASCConfig,
    *,
    base_optimizer: Optional[torch.optim.Optimizer] = None,
    warp_optimizer: Optional[torch.optim.Optimizer] = None,
    latest: GovernorMetrics | None,
    completed_steps: int = 0,
) -> str:
    checkpoint_path = output_dir / "asc_checkpoint.pt"
    torch.save(
        {
            "asc_config": asc_config.to_dict(),
            "base_state_dict": {k: v.detach().cpu() for k, v in model.base.state_dict().items()},
            "warp_state_dict": {k: v.detach().cpu() for k, v in model.warp.state_dict().items()},
            "target_state_dict": {k: v.detach().cpu() for k, v in model.target.state_dict().items()},
            "base_optimizer_state_dict": base_optimizer.state_dict() if base_optimizer is not None else None,
            "warp_optimizer_state_dict": warp_optimizer.state_dict() if warp_optimizer is not None else None,
            "latest_metrics": latest.model_dump(mode="json") if latest is not None else None,
            "completed_steps": completed_steps,
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def _load_asc_checkpoint(
    checkpoint_path: Path,
    *,
    model: ASCForCausalLM,
    base_optimizer: torch.optim.Optimizer,
    warp_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, Optional[GovernorMetrics]]:
    payload = torch.load(checkpoint_path, map_location=device)
    model.base.load_state_dict(payload.get("base_state_dict", {}), strict=False)
    model.warp.load_state_dict(payload.get("warp_state_dict", {}), strict=False)
    target_state = payload.get("target_state_dict")
    if isinstance(target_state, dict):
        model.target.load_state_dict(target_state, strict=False)
    base_state = payload.get("base_optimizer_state_dict")
    warp_state = payload.get("warp_optimizer_state_dict")
    if base_state:
        base_optimizer.load_state_dict(base_state)
    if warp_state:
        warp_optimizer.load_state_dict(warp_state)
    if payload.get("python_random_state") is not None:
        random.setstate(payload["python_random_state"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    latest_raw = payload.get("latest_metrics")
    latest = GovernorMetrics.model_validate(latest_raw) if latest_raw else None
    return int(payload.get("completed_steps", 0)), latest


def _run_toy_payload(
    config: TrainingPayloadConfig,
    *,
    dry_run: bool,
    device: torch.device,
    output_dir: Path,
    log_path: Path,
    anchor_manifest: DatasetManifest | None,
    research_manifest: DatasetManifest | None,
) -> dict[str, Any]:
    seed = config.seed
    torch.manual_seed(seed)
    model = TinyAnchorNet(config.feature_dim).to(device)
    if config.strategy_family == "layer_freeze":
        for name, param in model.named_parameters():
            if name.startswith("net.0"):
                param.requires_grad = False

    anchor = load_or_create_anchor(model, Path(config.anchor_path))
    anchor_reference, anchor_effective_dimensionality = _build_toy_anchor_reference(config, anchor, device)
    observer = ActivationThermoObserver(model, stat_window_size=config.stat_window_size)
    observer.set_anchor_reference(anchor_reference)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.eta)
    governor = ThermodynamicGovernor()
    criterion = nn.CrossEntropyLoss()
    steps = min(config.steps, 1) if dry_run else config.steps
    latest = None
    fail_fast_reasons: list[str] = []
    governor_action = "continue"
    disable_anchor_penalty = bool(config.notes.get("disable_anchor_penalty", False))
    disable_fim_proxy = bool(config.notes.get("disable_fim_proxy", False))
    disable_ou_jitter = bool(config.notes.get("disable_ou_jitter", False))

    try:
        for step in range(1, steps + 1):
            x, y = sample_batch(config.batch_size, config.feature_dim, config.strategy_family, device)
            optimizer.zero_grad()
            logits = model(x)
            task_loss = criterion(logits, y)
            anchor_penalty = torch.zeros(1, device=device)
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                anchor_tensor = anchor[name].to(device=device, dtype=param.dtype)
                anchor_penalty = anchor_penalty + ((param - anchor_tensor) ** 2).sum()
            total_loss = task_loss
            if not disable_anchor_penalty:
                total_loss = total_loss + config.alpha * anchor_penalty
            if not disable_fim_proxy:
                total_loss = total_loss + (config.fim_lambda * 1e-4)
            total_loss.backward()

            if config.strategy_family == "ou_drift_jitter" and not disable_ou_jitter:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.add_(torch.randn_like(param.grad) * float(config.notes.get("ou_sigma", 0.01)))

            regime = observer.step(optimizer)
            optimizer.step()
            latest = compute_metrics(
                config,
                model,
                anchor,
                step,
                regime=regime,
                training_loss=float(task_loss.item()),
            )
            append_metric(log_path, latest)

            decision = governor.evaluate(latest, config.governor_thresholds)
            governor_action = decision.action
            fail_fast_reasons = list(decision.reasons)
            if decision.action == "terminate":
                break
    finally:
        observer.close()

    calibration = _evaluate_toy_calibration(model, config, device, num_batches=config.calibration_batches)
    summary = {
        "trial_id": config.trial_id,
        "backend_id": config.backend_id,
        "backend_family": config.backend_provenance.domain if config.backend_provenance is not None else "unknown",
        "backend_readiness": config.backend_provenance.status if config.backend_provenance is not None else "unknown",
        "backend_provenance": config.backend_provenance.model_dump(mode="json") if config.backend_provenance is not None else None,
        "run_intent": config.run_intent,
        "research_grade": config.research_grade,
        "provenance_complete": config.provenance_complete,
        "device": str(device),
        "seed": seed,
        "steps": latest.step if latest is not None else 0,
        "last_metrics": latest.model_dump(mode="json") if latest else None,
        "strategy_family": config.strategy_family,
        "output_dir": str(output_dir),
        "anchor_manifest_path": config.anchor_manifest_path,
        "research_manifest_path": config.research_manifest_path,
        "anchor_records": anchor_manifest.records if anchor_manifest else 0,
        "research_records": research_manifest.records if research_manifest else 0,
        "data_purity": config.data_provenance.data_purity if config.data_provenance is not None else "unknown",
        "data_provenance": config.data_provenance.model_dump(mode="json") if config.data_provenance is not None else None,
        "tokenizer_provenance": {k: v.model_dump(mode="json") for k, v in config.tokenizer_provenance.items()},
        "anchor_effective_dimensionality": anchor_effective_dimensionality,
        "governor_action": governor_action,
        "governor_reasons": fail_fast_reasons,
        "calibration": calibration.model_dump(mode="json"),
        "ablations": {
            "disable_anchor_penalty": disable_anchor_penalty,
            "disable_fim_proxy": disable_fim_proxy,
            "disable_ou_jitter": disable_ou_jitter,
        },
        "checkpoint_path": None,
    }
    save_json(output_dir / "payload_summary.json", summary)
    return summary


def _evaluate_toy_calibration(
    model: nn.Module,
    config: TrainingPayloadConfig,
    device: torch.device,
    num_batches: int = 8,
) -> CalibrationReport:
    model.eval()
    confidences: List[float] = []
    correct: List[int] = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = sample_batch(config.batch_size, config.feature_dim, config.strategy_family, device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            confidences.extend(float(item) for item in conf.detach().cpu())
            correct.extend(int(item) for item in pred.eq(y).detach().cpu())
    return compute_calibration_report(confidences, correct)


def _run_asc_text_payload(
    config: TrainingPayloadConfig,
    *,
    dry_run: bool,
    device: torch.device,
    output_dir: Path,
    log_path: Path,
    anchor_manifest: DatasetManifest | None,
    research_manifest: DatasetManifest | None,
) -> dict[str, Any]:
    anchor_sequences = _load_sequences(anchor_manifest)
    research_sequences = _load_sequences(research_manifest)
    corpus_sequences = research_sequences or anchor_sequences
    if not corpus_sequences:
        raise RuntimeError("TAR ASC payload requires at least one tokenized manifest with sequences of length >= 2.")
    train_sequences, val_sequences, test_sequences = _split_sequences(
        corpus_sequences,
        seed=config.seed,
        train_fraction=config.train_split,
        val_fraction=config.val_split,
        test_fraction=config.test_split,
    )
    eval_sequences = val_sequences or test_sequences or train_sequences

    seed = config.seed
    random.seed(seed)
    torch.manual_seed(seed)
    seq_len = max(16, int(config.notes.get("max_seq_len", 96)))
    vocab_size = _infer_vocab_size(anchor_sequences, research_sequences)
    requested_payload_model = str(config.notes.get("base_model_name", "deepseek-ai/deepseek-coder-1.3b-base"))
    asc_config = _asc_config_from_payload(config, vocab_size=vocab_size, seq_len=seq_len, dry_run=dry_run)
    model = ASCForCausalLM(asc_config).to(device)
    _apply_layer_freeze(model.base, config)

    anchor = load_or_create_anchor(model.base, Path(config.anchor_path))
    anchor_reference, anchor_effective_dimensionality = _build_asc_anchor_reference(
        asc_config,
        anchor,
        anchor_sequences or eval_sequences or train_sequences,
        batch_size=config.batch_size,
        seq_len=seq_len,
        device=device,
        num_batches=config.anchor_batches,
    )
    observer = ActivationThermoObserver(model.base, stat_window_size=config.stat_window_size)
    observer.set_anchor_reference(anchor_reference)

    base_optimizer = torch.optim.AdamW(
        model.parameters_trainable(),
        lr=config.eta,
        weight_decay=float(config.notes.get("weight_decay", 0.01)),
    )
    warp_lr_multiplier = float(config.notes.get("warp_lr_multiplier", 3.0))
    warp_optimizer = torch.optim.AdamW(
        model.warp.parameters(),
        lr=config.eta * warp_lr_multiplier,
        weight_decay=0.0,
    )
    governor = ThermodynamicGovernor()
    rng = random.Random(seed)
    steps = min(config.steps, 3) if dry_run else config.steps
    latest = None
    resume_path = Path(config.resume_from_checkpoint or (output_dir / "asc_checkpoint.pt"))
    resumed_from_checkpoint = False
    start_step = 0
    last_task_loss = 0.0
    last_consistency_loss = 0.0
    fail_fast_reasons: list[str] = []
    governor_action = "continue"
    disable_anchor_penalty = bool(config.notes.get("disable_anchor_penalty", False))
    disable_fim_proxy = bool(config.notes.get("disable_fim_proxy", False))
    disable_ou_jitter = bool(config.notes.get("disable_ou_jitter", False))
    consistency_lambda = float(config.notes.get("consistency_lambda", asc_config.consistency_lambda))

    if resume_path.exists():
        start_step, latest = _load_asc_checkpoint(
            resume_path,
            model=model,
            base_optimizer=base_optimizer,
            warp_optimizer=warp_optimizer,
            device=device,
        )
        resumed_from_checkpoint = True

    try:
        for step in range(start_step + 1, steps + 1):
            batch = _sample_lm_batch(
                train_sequences,
                batch_size=config.batch_size,
                seq_len=seq_len,
                device=device,
                rng=rng,
            )
            base_optimizer.zero_grad(set_to_none=True)
            task_out = model.base(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            with torch.no_grad():
                clean_h = model.target(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                ).hidden_states[-1]
            consistency_out = model.base(
                inputs_embeds=model.warp(clean_h).detach(),
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            task_loss = task_out.loss
            consistency_loss = consistency_out.loss
            anchor_penalty, fim_proxy_penalty = _anchor_regularizers(
                model.base,
                anchor,
                config.protected_layers,
            )
            total_loss = task_loss + consistency_lambda * consistency_loss
            if not disable_anchor_penalty:
                total_loss = total_loss + config.alpha * anchor_penalty
            if not disable_fim_proxy:
                total_loss = total_loss + config.fim_lambda * fim_proxy_penalty
            total_loss.backward()

            if config.strategy_family == "ou_drift_jitter" and not disable_ou_jitter:
                noise_scale = float(config.notes.get("ou_sigma", max(1e-4, min(config.drift_budget, 0.01))))
                for param in model.base.parameters():
                    if param.grad is not None:
                        param.grad.add_(torch.randn_like(param.grad) * noise_scale)

            _refresh_base_activations(model.base, batch)
            regime = observer.step(base_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters_trainable(), 1.0)
            base_optimizer.step()

            warp_optimizer.zero_grad(set_to_none=True)
            warp_loss = model.base(
                inputs_embeds=model.warp(clean_h),
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            ).loss
            (-warp_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.warp.parameters(), 1.0)
            warp_optimizer.step()
            model.update_target()

            last_task_loss = float(task_loss.item())
            last_consistency_loss = float(consistency_loss.item())
            latest = compute_metrics(
                config,
                model.base,
                anchor,
                step,
                regime=regime,
                training_loss=last_task_loss,
            )
            append_metric(log_path, latest)

            decision = governor.evaluate(latest, config.governor_thresholds)
            governor_action = decision.action
            fail_fast_reasons = list(decision.reasons)
            _save_asc_checkpoint(
                output_dir,
                model,
                asc_config,
                base_optimizer=base_optimizer,
                warp_optimizer=warp_optimizer,
                latest=latest,
                completed_steps=step,
            )
            if decision.action == "terminate":
                break
    finally:
        observer.close()

    calibration = _evaluate_lm_calibration(
        model,
        test_sequences or eval_sequences or train_sequences,
        batch_size=config.batch_size,
        seq_len=seq_len,
        device=device,
        num_batches=config.calibration_batches,
    )
    checkpoint_path = _save_asc_checkpoint(
        output_dir,
        model,
        asc_config,
        base_optimizer=base_optimizer,
        warp_optimizer=warp_optimizer,
        latest=latest,
        completed_steps=latest.step if latest is not None else start_step,
    )
    summary = {
        "trial_id": config.trial_id,
        "backend_id": config.backend_id,
        "backend_family": config.backend_provenance.domain if config.backend_provenance is not None else "unknown",
        "backend_readiness": config.backend_provenance.status if config.backend_provenance is not None else "unknown",
        "backend_provenance": config.backend_provenance.model_dump(mode="json") if config.backend_provenance is not None else None,
        "run_intent": config.run_intent,
        "research_grade": config.research_grade,
        "provenance_complete": config.provenance_complete,
        "device": str(device),
        "seed": seed,
        "steps": latest.step if latest is not None else 0,
        "last_metrics": latest.model_dump(mode="json") if latest else None,
        "strategy_family": config.strategy_family,
        "output_dir": str(output_dir),
        "anchor_manifest_path": config.anchor_manifest_path,
        "research_manifest_path": config.research_manifest_path,
        "anchor_records": anchor_manifest.records if anchor_manifest else 0,
        "research_records": research_manifest.records if research_manifest else 0,
        "data_purity": config.data_provenance.data_purity if config.data_provenance is not None else "unknown",
        "data_provenance": config.data_provenance.model_dump(mode="json") if config.data_provenance is not None else None,
        "tokenizer_provenance": {k: v.model_dump(mode="json") for k, v in config.tokenizer_provenance.items()},
        "anchor_effective_dimensionality": anchor_effective_dimensionality,
        "governor_action": governor_action,
        "governor_reasons": fail_fast_reasons,
        "calibration": calibration.model_dump(mode="json"),
        "ablations": {
            "disable_anchor_penalty": disable_anchor_penalty,
            "disable_fim_proxy": disable_fim_proxy,
            "disable_ou_jitter": disable_ou_jitter,
        },
        "task_loss": last_task_loss,
        "consistency_loss": last_consistency_loss,
        "checkpoint_path": checkpoint_path,
        "requested_payload_model": requested_payload_model,
        "payload_model": asc_config.base_model_name,
        "adapter_mode": asc_config.adapter_mode,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resume_start_step": start_step,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "split_counts": {
            "train": len(train_sequences),
            "val": len(val_sequences),
            "test": len(test_sequences),
        },
        "stat_window_size": config.stat_window_size,
        "min_stat_steps": config.min_stat_steps,
    }
    save_json(output_dir / "payload_summary.json", summary)
    return summary


def run_payload(config: TrainingPayloadConfig, dry_run: bool = False) -> dict[str, Any]:
    device, device_fallback_reason = _resolve_execution_device()
    output_dir = Path(config.output_dir)
    log_path = Path(config.log_path)
    anchor_manifest = load_manifest(config.anchor_manifest_path)
    research_manifest = load_manifest(config.research_manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_payload_provenance(config)

    if config.backend_id == "toy_anchor":
        summary = _run_toy_payload(
            config,
            dry_run=dry_run,
            device=device,
            output_dir=output_dir,
            log_path=log_path,
            anchor_manifest=anchor_manifest,
            research_manifest=research_manifest,
        )
    elif config.backend_id in {"asc_cv", "asc_rl", "asc_qml"}:
        summary = run_multimodal_backend(
            config=config,
            dry_run=dry_run,
            device=device,
            output_dir=output_dir,
            log_path=log_path,
        )
    elif config.backend_id != "asc_text":
        raise ScientificValidityError(
            f"Backend '{config.backend_id}' is not a scientifically valid tar_lab.train_template execution path."
        )
    else:
        summary = _run_asc_text_payload(
            config,
            dry_run=dry_run,
            device=device,
            output_dir=output_dir,
            log_path=log_path,
            anchor_manifest=anchor_manifest,
            research_manifest=research_manifest,
        )

    summary["device"] = str(device)
    summary["device_fallback_reason"] = device_fallback_reason
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    summary = run_payload(config, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
