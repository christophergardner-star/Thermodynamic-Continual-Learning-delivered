from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from tar_lab.schemas import GovernorDecision, GovernorMetrics, GovernorThresholds


TensorMap = Mapping[str, torch.Tensor]


def _as_tensor_map(source: nn.Module | TensorMap) -> Dict[str, torch.Tensor]:
    if isinstance(source, nn.Module):
        return {name: tensor.detach().cpu() for name, tensor in source.state_dict().items()}
    return {name: tensor.detach().cpu() for name, tensor in source.items()}


class ThermodynamicGovernor:
    def __init__(self, thresholds: Optional[GovernorThresholds] = None):
        self.thresholds = thresholds or GovernorThresholds()

    def compute_metrics(
        self,
        trial_id: str,
        step: int,
        anchor: nn.Module | TensorMap,
        current: nn.Module | TensorMap,
        gradients: Optional[TensorMap] = None,
        gpu_temperature_c: Optional[float] = None,
        gpu_memory_temperature_c: Optional[float] = None,
        gpu_power_w: Optional[float] = None,
        regime_rho: float = 0.0,
        effective_dimensionality: Optional[float] = None,
        effective_dimensionality_std_err: float = 0.0,
        dimensionality_ratio: Optional[float] = None,
        entropy_sigma_std_err: float = 0.0,
        regime_rho_std_err: float = 0.0,
        stat_window_size: int = 0,
        stat_sample_count: int = 0,
        statistically_ready: bool = False,
        equilibrium_fraction: float = 0.0,
        equilibrium_gate: bool = False,
        training_loss: Optional[float] = None,
    ) -> GovernorMetrics:
        anchor_map = _as_tensor_map(anchor)
        current_map = _as_tensor_map(current)
        grad_map = _as_tensor_map(gradients or {})

        delta_sq_sum = 0.0
        anchor_sq_sum = 0.0
        grad_sq_sum = 0.0
        entropy_numerator = 0.0

        for name, anchor_tensor in anchor_map.items():
            current_tensor = current_map.get(name)
            if current_tensor is None:
                continue
            delta = (current_tensor.float() - anchor_tensor.float()).reshape(-1)
            anchor_vec = anchor_tensor.float().reshape(-1)
            delta_sq_sum += float((delta * delta).sum())
            anchor_sq_sum += float((anchor_vec * anchor_vec).sum())

            grad_tensor = grad_map.get(name)
            if grad_tensor is not None:
                grad_vec = grad_tensor.float().reshape(-1)
                grad_sq_sum += float((grad_vec * grad_vec).sum())
                entropy_numerator += float((delta.abs() * grad_vec.abs()).sum())

        drift_l2 = math.sqrt(max(delta_sq_sum, 0.0))
        anchor_l2 = math.sqrt(max(anchor_sq_sum, 0.0))
        grad_norm = math.sqrt(max(grad_sq_sum, 0.0))
        drift_rho = drift_l2 / (anchor_l2 + 1e-12)
        energy_e = delta_sq_sum
        entropy_sigma = entropy_numerator / (anchor_l2 + 1e-12)
        if gradients is None:
            entropy_sigma = energy_e / ((anchor_l2 * anchor_l2) + 1e-12)

        return GovernorMetrics(
            trial_id=trial_id,
            step=step,
            energy_e=energy_e,
            entropy_sigma=entropy_sigma,
            drift_l2=drift_l2,
            drift_rho=drift_rho,
            grad_norm=grad_norm,
            regime_rho=regime_rho,
            effective_dimensionality=0.0 if effective_dimensionality is None else effective_dimensionality,
            effective_dimensionality_std_err=effective_dimensionality_std_err,
            dimensionality_ratio=0.0 if dimensionality_ratio is None else dimensionality_ratio,
            entropy_sigma_std_err=entropy_sigma_std_err,
            regime_rho_std_err=regime_rho_std_err,
            stat_window_size=stat_window_size,
            stat_sample_count=stat_sample_count,
            statistically_ready=statistically_ready,
            equilibrium_fraction=equilibrium_fraction,
            equilibrium_gate=equilibrium_gate,
            training_loss=training_loss,
            gpu_temperature_c=gpu_temperature_c,
            gpu_memory_temperature_c=gpu_memory_temperature_c,
            gpu_power_w=gpu_power_w,
        )

    def evaluate(
        self,
        metrics: GovernorMetrics,
        thresholds: Optional[GovernorThresholds] = None,
    ) -> GovernorDecision:
        limits = thresholds or self.thresholds
        reasons = []
        if metrics.drift_l2 > limits.max_drift_l2:
            reasons.append("weight_drift_limit")
        if metrics.drift_rho > limits.max_drift_rho:
            reasons.append("relative_drift_limit")
        if metrics.entropy_sigma > limits.max_entropy_sigma:
            reasons.append("entropy_limit")
        if metrics.grad_norm > limits.max_grad_norm:
            reasons.append("grad_norm_limit")
        if (
            metrics.statistically_ready
            and metrics.dimensionality_ratio > 0.0
            and metrics.dimensionality_ratio < limits.min_dimensionality_ratio
            and metrics.training_loss is not None
            and metrics.training_loss <= limits.max_quenching_loss
        ):
            reasons.append("thermodynamic_quenching")
        if (
            metrics.gpu_temperature_c is not None
            and metrics.gpu_temperature_c > limits.max_gpu_temperature_c
        ):
            reasons.append("gpu_temperature_limit")
        action = "terminate" if reasons else "continue"
        return GovernorDecision(action=action, reasons=reasons, metrics=metrics)
