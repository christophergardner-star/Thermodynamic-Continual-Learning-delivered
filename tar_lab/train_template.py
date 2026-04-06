from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from tar_lab.governor import ThermodynamicGovernor
from tar_lab.schemas import CalibrationBin, CalibrationReport, DatasetManifest, GovernorMetrics, TrainingPayloadConfig
from tar_lab.thermoobserver import ActivationThermoObserver, RegimeSnapshot


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


def load_config(path: str) -> TrainingPayloadConfig:
    return TrainingPayloadConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_manifest(path: str | None) -> DatasetManifest | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_or_create_anchor(model: nn.Module, anchor_path: Path) -> Dict[str, torch.Tensor]:
    if anchor_path.exists():
        raw = torch.load(anchor_path, map_location="cpu")
        if isinstance(raw, dict):
            return {k: v.detach().clone().cpu() for k, v in raw.items()}
    anchor = {name: tensor.detach().clone().cpu() for name, tensor in model.state_dict().items()}
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(anchor, anchor_path)
    return anchor


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

    for name, tensor in model.state_dict().items():
        current = tensor.detach().float().cpu()
        ref = anchor[name].float()
        delta = (current - ref).reshape(-1)
        delta_sq_sum += float((delta * delta).sum())
        anchor_sq_sum += float((ref.reshape(-1) * ref.reshape(-1)).sum())
        param = dict(model.named_parameters()).get(name)
        if param is not None and param.grad is not None:
            grad_vec = param.grad.detach().float().cpu().reshape(-1)
            grad_sq_sum += float((grad_vec * grad_vec).sum())
            entropy_numerator += float((delta.abs() * grad_vec.abs()).sum())

    drift_l2 = delta_sq_sum ** 0.5
    anchor_l2 = anchor_sq_sum ** 0.5
    entropy_sigma = entropy_numerator / (anchor_l2 + 1e-12)
    if regime is not None and regime.entropy_sigma > 0.0:
        entropy_sigma = regime.entropy_sigma

    return GovernorMetrics(
        trial_id=config.trial_id,
        step=step,
        energy_e=delta_sq_sum,
        entropy_sigma=entropy_sigma,
        drift_l2=drift_l2,
        drift_rho=drift_l2 / (anchor_l2 + 1e-12),
        grad_norm=grad_sq_sum ** 0.5,
        effective_dimensionality=regime.effective_dimensionality if regime is not None else 0.0,
        dimensionality_ratio=regime.dimensionality_ratio if regime is not None else 0.0,
        equilibrium_fraction=regime.equilibrium_fraction if regime is not None else 0.0,
        equilibrium_gate=regime.equilibrium_gate if regime is not None else False,
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


def evaluate_calibration(
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


def build_anchor_reference(
    config: TrainingPayloadConfig,
    anchor: Dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float], float]:
    reference_model = TinyAnchorNet(config.feature_dim).to(device)
    reference_model.load_state_dict(anchor, strict=False)
    observer = ActivationThermoObserver(reference_model)
    values: dict[str, list[float]] = {}
    try:
        reference_model.eval()
        with torch.no_grad():
            for _ in range(3):
                x, _ = sample_batch(config.batch_size, config.feature_dim, config.strategy_family, device)
                reference_model(x)
                snapshot = observer.anchor_snapshot()
                for name, value in snapshot.items():
                    values.setdefault(name, []).append(value)
    finally:
        observer.close()
    means = {
        name: sum(samples) / len(samples)
        for name, samples in values.items()
        if samples
    }
    anchor_effective_dimensionality = 0.0
    positives = [value for value in means.values() if value > 0.0]
    if positives:
        positives.sort()
        anchor_effective_dimensionality = positives[len(positives) // 2]
    return means, anchor_effective_dimensionality


def run_payload(config: TrainingPayloadConfig, dry_run: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    log_path = Path(config.log_path)
    anchor_manifest = load_manifest(config.anchor_manifest_path)
    research_manifest = load_manifest(config.research_manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.notes.get("seed", 7))
    torch.manual_seed(seed)
    model = TinyAnchorNet(config.feature_dim).to(device)
    if config.strategy_family == "layer_freeze":
        for name, param in model.named_parameters():
            if name.startswith("net.0"):
                param.requires_grad = False

    anchor = load_or_create_anchor(model, Path(config.anchor_path))
    anchor_reference, anchor_effective_dimensionality = build_anchor_reference(config, anchor, device)
    observer = ActivationThermoObserver(model)
    observer.set_anchor_reference(anchor_reference)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.eta)
    governor = ThermodynamicGovernor()
    criterion = nn.CrossEntropyLoss()
    steps = min(config.steps, 3) if dry_run else config.steps
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

    calibration = evaluate_calibration(model, config, device)

    summary = {
        "trial_id": config.trial_id,
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
        "anchor_effective_dimensionality": anchor_effective_dimensionality,
        "governor_action": governor_action,
        "governor_reasons": fail_fast_reasons,
        "calibration": calibration.model_dump(mode="json"),
        "ablations": {
            "disable_anchor_penalty": disable_anchor_penalty,
            "disable_fim_proxy": disable_fim_proxy,
            "disable_ou_jitter": disable_ou_jitter,
        },
    }
    save_json(output_dir / "payload_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    summary = run_payload(config, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
