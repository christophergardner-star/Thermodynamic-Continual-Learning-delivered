from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn


def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _reshape_activations(activations: torch.Tensor, feature_axis: int = -1) -> torch.Tensor:
    tensor = activations.detach().float()
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    axis = feature_axis if feature_axis >= 0 else tensor.ndim + feature_axis
    if axis < 0 or axis >= tensor.ndim:
        axis = tensor.ndim - 1
    if axis != tensor.ndim - 1:
        order = [idx for idx in range(tensor.ndim) if idx != axis] + [axis]
        tensor = tensor.permute(*order)
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    return tensor.reshape(-1, tensor.shape[-1])


def compute_participation_ratio(
    activations: torch.Tensor,
    *,
    feature_axis: int = -1,
    eps: float = 1e-12,
) -> float:
    matrix = _reshape_activations(activations, feature_axis=feature_axis)
    if matrix.numel() == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return 0.0
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    if float(matrix.pow(2).sum()) <= eps:
        return 0.0

    sample_count = matrix.shape[0]
    covariance = matrix.T @ matrix / max(sample_count - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalues = torch.clamp(eigenvalues.real, min=0.0)
    trace = float(eigenvalues.sum().item())
    trace_sq = float((eigenvalues * eigenvalues).sum().item())
    if trace_sq <= eps:
        return 0.0
    return trace * trace / trace_sq


@dataclass(slots=True)
class LayerRegimeSnapshot:
    name: str
    sigma: float
    sigma_ema: float
    sigma_star: float
    fim_trace: float
    fim_rel_change: float
    effective_dimensionality: float
    equilibrium_gate: bool


@dataclass(slots=True)
class RegimeSnapshot:
    layer_metrics: List[LayerRegimeSnapshot]
    entropy_sigma: float
    effective_dimensionality: float
    dimensionality_ratio: float
    equilibrium_fraction: float
    equilibrium_gate: bool


@dataclass(slots=True)
class _ObservedGroup:
    name: str
    module: nn.Module
    params: List[nn.Parameter]
    feature_axis: int
    sigma_ema: Optional[float] = None
    sigma_star: Optional[float] = None
    sigma_window: List[float] = field(default_factory=list)
    fim_ema: Optional[float] = None
    stable_steps: int = 0
    last_activation: Optional[torch.Tensor] = None


class ActivationThermoObserver:
    def __init__(
        self,
        model: nn.Module,
        *,
        alpha: float = 1.0,
        sigma_window_size: int = 8,
        calib_start: int = 2,
        sigma_tolerance: float = 0.15,
        fim_tolerance: float = 0.10,
        equilibrium_patience: int = 2,
    ):
        self.model = model
        self.alpha = alpha
        self.sigma_window_size = sigma_window_size
        self.calib_start = calib_start
        self.sigma_tolerance = sigma_tolerance
        self.fim_tolerance = fim_tolerance
        self.equilibrium_patience = equilibrium_patience
        self._hooks: list[Any] = []
        self._groups: list[_ObservedGroup] = []
        self.anchor_dimensionality_by_group: Dict[str, float] = {}
        self.anchor_effective_dimensionality: float = 0.0
        self._register_groups()

    def _register_groups(self) -> None:
        named_modules = list(self.model.named_modules())
        for name, module in named_modules:
            if not name or any(module.children()):
                continue
            params = [param for param in module.parameters(recurse=False) if param.requires_grad]
            if not params:
                continue
            feature_axis = 1 if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) else -1
            group = _ObservedGroup(name=name, module=module, params=params, feature_axis=feature_axis)
            self._groups.append(group)
            handle = module.register_forward_hook(self._make_hook(group))
            self._hooks.append(handle)

    def _make_hook(self, group: _ObservedGroup):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _extract_tensor(output)
            if tensor is None:
                group.last_activation = None
                return
            group.last_activation = tensor.detach().float()

        return _hook

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def anchor_snapshot(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for group in self._groups:
            if group.last_activation is None:
                continue
            values[group.name] = compute_participation_ratio(
                group.last_activation,
                feature_axis=group.feature_axis,
            )
        self.set_anchor_reference(values)
        return values

    def set_anchor_reference(self, values: Dict[str, float]) -> None:
        self.anchor_dimensionality_by_group = {
            key: float(val) for key, val in values.items() if float(val) >= 0.0
        }
        anchors = [value for value in self.anchor_dimensionality_by_group.values() if value > 0.0]
        self.anchor_effective_dimensionality = float(median(anchors)) if anchors else 0.0

    def step(self, optimizer: torch.optim.Optimizer) -> RegimeSnapshot:
        layer_metrics: List[LayerRegimeSnapshot] = []

        for group in self._groups:
            grads = [param.grad for param in group.params if param.grad is not None]
            if not grads:
                continue

            lr = self._group_lr(optimizer, group.params)
            grad_norm_sq = self._norm_sq(grads)
            sigma = lr * grad_norm_sq
            fim_trace = self._fim_trace(grads)

            ema_alpha = 2.0 / (self.sigma_window_size + 1.0)
            if group.sigma_ema is None:
                group.sigma_ema = sigma
            else:
                group.sigma_ema = (1.0 - ema_alpha) * group.sigma_ema + ema_alpha * sigma

            if group.fim_ema is None:
                group.fim_ema = fim_trace
            else:
                group.fim_ema = (1.0 - ema_alpha) * group.fim_ema + ema_alpha * fim_trace

            if len(group.sigma_window) >= self.calib_start:
                group.sigma_window.append(group.sigma_ema)
                if len(group.sigma_window) > self.sigma_window_size:
                    group.sigma_window.pop(0)
                group.sigma_star = self.alpha * float(median(group.sigma_window))
            else:
                group.sigma_window.append(group.sigma_ema)

            sigma_star = group.sigma_star if group.sigma_star is not None else max(group.sigma_ema, 1e-12)
            sigma_ratio = group.sigma_ema / max(sigma_star, 1e-12)
            fim_rel_change = abs(fim_trace - group.fim_ema) / max(abs(group.fim_ema), 1e-12)
            stable = abs(sigma_ratio - 1.0) <= self.sigma_tolerance and fim_rel_change <= self.fim_tolerance
            group.stable_steps = group.stable_steps + 1 if stable else 0
            equilibrium_gate = group.stable_steps >= self.equilibrium_patience and group.sigma_star is not None

            d_pr = 0.0
            if group.last_activation is not None:
                d_pr = compute_participation_ratio(group.last_activation, feature_axis=group.feature_axis)

            layer_metrics.append(
                LayerRegimeSnapshot(
                    name=group.name,
                    sigma=float(sigma),
                    sigma_ema=float(group.sigma_ema),
                    sigma_star=float(sigma_star),
                    fim_trace=float(fim_trace),
                    fim_rel_change=float(fim_rel_change),
                    effective_dimensionality=float(d_pr),
                    equilibrium_gate=equilibrium_gate,
                )
            )

        sigma_values = [item.sigma_ema for item in layer_metrics]
        d_values = [item.effective_dimensionality for item in layer_metrics if item.effective_dimensionality > 0.0]
        eq_values = [1.0 if item.equilibrium_gate else 0.0 for item in layer_metrics]
        effective_dimensionality = float(median(d_values)) if d_values else 0.0
        equilibrium_fraction = float(sum(eq_values) / len(eq_values)) if eq_values else 0.0
        dimensionality_ratio = 0.0
        if self.anchor_effective_dimensionality > 0.0 and effective_dimensionality > 0.0:
            dimensionality_ratio = effective_dimensionality / self.anchor_effective_dimensionality

        return RegimeSnapshot(
            layer_metrics=layer_metrics,
            entropy_sigma=float(median(sigma_values)) if sigma_values else 0.0,
            effective_dimensionality=effective_dimensionality,
            dimensionality_ratio=dimensionality_ratio,
            equilibrium_fraction=equilibrium_fraction,
            equilibrium_gate=equilibrium_fraction >= 0.8 if eq_values else False,
        )

    @staticmethod
    def _group_lr(optimizer: torch.optim.Optimizer, params: Iterable[nn.Parameter]) -> float:
        group_ids = {id(param) for param in params}
        for param_group in optimizer.param_groups:
            if group_ids & {id(param) for param in param_group["params"]}:
                return float(param_group.get("_eta", param_group.get("lr", 1e-3)))
        return 1e-3

    @staticmethod
    def _norm_sq(tensors: Iterable[torch.Tensor]) -> float:
        total = 0.0
        for tensor in tensors:
            grad = tensor.detach().float()
            total += float((grad * grad).sum().item())
        return total

    @staticmethod
    def _fim_trace(tensors: Iterable[torch.Tensor]) -> float:
        total = 0.0
        for tensor in tensors:
            grad = tensor.detach().float()
            total += float(grad.pow(2).mean().item())
        return total
