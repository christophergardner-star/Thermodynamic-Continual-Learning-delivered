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
    n_features = covariance.shape[0]
    reg = covariance.diagonal().mean().clamp(min=1e-8) * 1e-4
    covariance = covariance + reg * torch.eye(n_features, dtype=covariance.dtype, device=covariance.device)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalues = torch.clamp(eigenvalues.real, min=0.0)
    trace = float(eigenvalues.sum().item())
    trace_sq = float((eigenvalues * eigenvalues).sum().item())
    if trace_sq <= eps:
        return 0.0
    return trace * trace / trace_sq


def compute_activation_covariance(
    activations: torch.Tensor,
    *,
    feature_axis: int = -1,
    eps: float = 1e-12,
) -> Optional[torch.Tensor]:
    matrix = _reshape_activations(activations, feature_axis=feature_axis)
    if matrix.numel() == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return None
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    if float(matrix.pow(2).sum()) <= eps:
        return None
    sample_count = matrix.shape[0]
    covariance = matrix.T @ matrix / max(sample_count - 1, 1)
    return covariance.detach().float().cpu()


@dataclass(slots=True)
class SmoothedStatMetrics:
    sample_count: int
    window_size: int
    entropy_sigma: float
    entropy_sigma_std_err: float
    rho: float
    rho_std_err: float
    effective_dimensionality: float
    effective_dimensionality_std_err: float
    statistically_ready: bool
    dimensionality_ratio: float


class StatAccumulator:
    def __init__(self, window_size: int = 5):
        self.window_size = max(1, int(window_size))
        self.covariances: list[torch.Tensor] = []
        self.sigma_samples: list[float] = []
        self.rho_samples: list[float] = []
        self.dpr_samples: list[float] = []

    @property
    def sample_count(self) -> int:
        return len(self.covariances)

    def push(self, covariance: Optional[torch.Tensor], *, sigma: float, rho: float) -> None:
        if covariance is None:
            return
        self.covariances.append(covariance.detach().float().cpu())
        self.sigma_samples.append(float(sigma))
        self.rho_samples.append(float(rho))
        trace = float(torch.trace(covariance).item())
        trace_sq = float((covariance * covariance).sum().item())
        d_pr = 0.0 if trace_sq <= 1e-12 else (trace * trace / trace_sq)
        self.dpr_samples.append(d_pr)
        if len(self.covariances) > self.window_size:
            self.covariances.pop(0)
            self.sigma_samples.pop(0)
            self.rho_samples.pop(0)
            self.dpr_samples.pop(0)

    def get_smoothed_metrics(self, *, anchor_effective_dimensionality: float = 0.0) -> SmoothedStatMetrics:
        if not self.covariances:
            return SmoothedStatMetrics(
                sample_count=0,
                window_size=self.window_size,
                entropy_sigma=0.0,
                entropy_sigma_std_err=0.0,
                rho=0.0,
                rho_std_err=0.0,
                effective_dimensionality=0.0,
                effective_dimensionality_std_err=0.0,
                statistically_ready=False,
                dimensionality_ratio=0.0,
            )
        mean_covariance = torch.stack(self.covariances, dim=0).mean(dim=0)
        trace = float(torch.trace(mean_covariance).item())
        trace_sq = float((mean_covariance * mean_covariance).sum().item())
        smoothed_d_pr = 0.0 if trace_sq <= 1e-12 else (trace * trace / trace_sq)
        d_pr_std_err = self._std_err(self.dpr_samples)
        sigma_mean = self._mean(self.sigma_samples)
        rho_mean = self._mean(self.rho_samples)
        sigma_std_err = self._std_err(self.sigma_samples)
        rho_std_err = self._std_err(self.rho_samples)
        dimensionality_ratio = 0.0
        if anchor_effective_dimensionality > 0.0 and smoothed_d_pr > 0.0:
            dimensionality_ratio = smoothed_d_pr / anchor_effective_dimensionality
        return SmoothedStatMetrics(
            sample_count=self.sample_count,
            window_size=self.window_size,
            entropy_sigma=sigma_mean,
            entropy_sigma_std_err=sigma_std_err,
            rho=rho_mean,
            rho_std_err=rho_std_err,
            effective_dimensionality=smoothed_d_pr,
            effective_dimensionality_std_err=d_pr_std_err,
            statistically_ready=self.sample_count >= self.window_size,
            dimensionality_ratio=dimensionality_ratio,
        )

    @staticmethod
    def _mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _std_err(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(max(variance, 0.0)) / math.sqrt(len(values))


@dataclass(slots=True)
class LayerRegimeSnapshot:
    name: str
    sigma: float
    sigma_std_err: float
    sigma_star: float
    regime_rho: float
    regime_rho_std_err: float
    fim_trace: float
    fim_rel_change: float
    effective_dimensionality: float
    effective_dimensionality_std_err: float
    dimensionality_ratio: float
    stat_sample_count: int
    stat_window_size: int
    statistically_ready: bool
    equilibrium_gate: bool


@dataclass(slots=True)
class RegimeSnapshot:
    layer_metrics: List[LayerRegimeSnapshot]
    entropy_sigma: float
    entropy_sigma_std_err: float
    regime_rho: float
    regime_rho_std_err: float
    effective_dimensionality: float
    effective_dimensionality_std_err: float
    dimensionality_ratio: float
    stat_sample_count: int
    stat_window_size: int
    statistically_ready: bool
    equilibrium_fraction: float
    equilibrium_gate: bool


@dataclass(slots=True)
class _ObservedGroup:
    name: str
    module: nn.Module
    params: List[nn.Parameter]
    feature_axis: int
    sigma_ema: Optional[float] = None
    # Rolling window: smooths current sigma estimate, reset each task.
    sigma_window: List[float] = field(default_factory=list)
    # Fixed per-task reference: set from sigma_star_anchor_n batches
    # *after* warmup_batches have elapsed, then frozen for the rest of
    # that task.  "Ordered" fires when sigma drops below alpha × anchor.
    sigma_star_anchor: Optional[float] = None
    anchor_window: List[float] = field(default_factory=list)
    anchor_set: bool = False
    # Counts batches since last reset — anchor collection is suppressed
    # until this reaches the observer's warmup_batches threshold.
    batch_counter: int = 0
    fim_ema: Optional[float] = None
    stable_steps: int = 0
    last_activation: Optional[torch.Tensor] = None
    stat_accumulator: Optional[StatAccumulator] = None


class ActivationThermoObserver:
    def __init__(
        self,
        model: nn.Module,
        *,
        alpha: float = 1.0,
        sigma_window_size: int = 8,
        sigma_star_anchor_n: int = 20,
        warmup_batches: int = 0,
        stat_window_size: int = 5,
        calib_start: int = 2,  # retained for API compat; superseded by sigma_star_anchor_n
        sigma_tolerance: float = 0.15,
        fim_tolerance: float = 0.10,
        equilibrium_patience: int = 2,
    ):
        self.model = model
        self.alpha = alpha
        self.sigma_window_size = sigma_window_size
        self.sigma_star_anchor_n = max(1, int(sigma_star_anchor_n))
        self.warmup_batches = max(0, int(warmup_batches))
        self.stat_window_size = max(1, stat_window_size)
        self.calib_start = calib_start
        self.sigma_tolerance = sigma_tolerance
        self.fim_tolerance = fim_tolerance
        self.equilibrium_patience = equilibrium_patience
        self._hooks: list[Any] = []
        self._groups: list[_ObservedGroup] = []
        self.anchor_dimensionality_by_group: Dict[str, float] = {}
        self.anchor_effective_dimensionality: float = 0.0
        self._last_snapshot: Optional[RegimeSnapshot] = None
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
            group = _ObservedGroup(
                name=name,
                module=module,
                params=params,
                feature_axis=feature_axis,
                stat_accumulator=StatAccumulator(window_size=self.stat_window_size),
            )
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

    def reset_for_new_task(self) -> None:
        """Reset per-task calibration state at each task boundary.

        Clears sigma_ema, sigma_window, sigma_star_anchor, anchor_window,
        fim_ema, stable_steps, and stat_accumulator.  sigma_star_anchor will
        be re-established from the first sigma_star_anchor_n batches of the
        new task and then frozen for that task's duration, giving a fixed
        reference temperature against which convergence is measured.

        Preserves: hooks, group structure, dimensionality anchor, last_activation.
        """
        for group in self._groups:
            group.sigma_ema = None
            group.sigma_window = []
            group.sigma_star_anchor = None
            group.anchor_window = []
            group.anchor_set = False
            group.batch_counter = 0
            group.fim_ema = None
            group.stable_steps = 0
            group.stat_accumulator = StatAccumulator(window_size=self.stat_window_size)
        self._last_snapshot = None

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

    def current_covariances(self) -> Dict[str, torch.Tensor]:
        values: Dict[str, torch.Tensor] = {}
        for group in self._groups:
            if group.last_activation is None:
                continue
            covariance = compute_activation_covariance(
                group.last_activation,
                feature_axis=group.feature_axis,
            )
            if covariance is not None:
                values[group.name] = covariance
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

            # Rolling window: smooths current sigma (noise reduction only).
            group.sigma_window.append(float(sigma))
            if len(group.sigma_window) > self.sigma_window_size:
                group.sigma_window.pop(0)

            # Fixed per-task anchor: accumulate sigma_star_anchor_n batches
            # *after* warmup_batches have elapsed, then freeze.
            # warmup_batches=0 (default) preserves original behaviour.
            # Set warmup_batches > 0 for large backbones where the first
            # batches are random-initialisation noise, not a meaningful
            # thermal baseline (e.g. warmup_batches=60 ≈ 2 epochs on
            # Split-CIFAR-10 with batch_size=32).
            group.batch_counter += 1
            if not group.anchor_set and group.batch_counter > self.warmup_batches:
                group.anchor_window.append(float(sigma))
                if len(group.anchor_window) >= self.sigma_star_anchor_n:
                    group.sigma_star_anchor = self.alpha * float(median(group.anchor_window))
                    group.anchor_set = True

            sigma_star = (
                group.sigma_star_anchor
                if group.sigma_star_anchor is not None
                else max(group.sigma_ema, 1e-12)
            )
            sigma_ratio = sigma / max(sigma_star, 1e-12)
            fim_rel_change = abs(fim_trace - group.fim_ema) / max(abs(group.fim_ema), 1e-12)
            covariance = None
            if group.last_activation is not None:
                covariance = compute_activation_covariance(group.last_activation, feature_axis=group.feature_axis)
            if group.stat_accumulator is None:
                group.stat_accumulator = StatAccumulator(window_size=self.stat_window_size)
            group.stat_accumulator.push(covariance, sigma=float(sigma), rho=float(sigma_ratio))
            smoothed = group.stat_accumulator.get_smoothed_metrics(
                anchor_effective_dimensionality=self.anchor_dimensionality_by_group.get(group.name, 0.0)
            )
            stable = (
                smoothed.statistically_ready
                and abs(smoothed.rho - 1.0) <= self.sigma_tolerance
                and fim_rel_change <= self.fim_tolerance
            )
            group.stable_steps = group.stable_steps + 1 if stable else 0
            equilibrium_gate = (
                smoothed.statistically_ready
                and group.stable_steps >= self.equilibrium_patience
                and group.sigma_star_anchor is not None
            )

            layer_metrics.append(
                LayerRegimeSnapshot(
                    name=group.name,
                    sigma=float(smoothed.entropy_sigma),
                    sigma_std_err=float(smoothed.entropy_sigma_std_err),
                    sigma_star=float(sigma_star),
                    regime_rho=float(smoothed.rho),
                    regime_rho_std_err=float(smoothed.rho_std_err),
                    fim_trace=float(fim_trace),
                    fim_rel_change=float(fim_rel_change),
                    effective_dimensionality=float(smoothed.effective_dimensionality),
                    effective_dimensionality_std_err=float(smoothed.effective_dimensionality_std_err),
                    dimensionality_ratio=float(smoothed.dimensionality_ratio),
                    stat_sample_count=smoothed.sample_count,
                    stat_window_size=smoothed.window_size,
                    statistically_ready=smoothed.statistically_ready,
                    equilibrium_gate=equilibrium_gate,
                )
            )

        sigma_values = [item.sigma for item in layer_metrics]
        sigma_errors = [item.sigma_std_err for item in layer_metrics]
        rho_values = [item.regime_rho for item in layer_metrics]
        rho_errors = [item.regime_rho_std_err for item in layer_metrics]
        d_values = [item.effective_dimensionality for item in layer_metrics if item.effective_dimensionality > 0.0]
        d_errors = [item.effective_dimensionality_std_err for item in layer_metrics if item.effective_dimensionality > 0.0]
        eq_values = [1.0 if item.equilibrium_gate else 0.0 for item in layer_metrics]
        effective_dimensionality = float(median(d_values)) if d_values else 0.0
        equilibrium_fraction = float(sum(eq_values) / len(eq_values)) if eq_values else 0.0
        dimensionality_ratio = 0.0
        if self.anchor_effective_dimensionality > 0.0 and effective_dimensionality > 0.0:
            dimensionality_ratio = effective_dimensionality / self.anchor_effective_dimensionality
        sample_counts = [item.stat_sample_count for item in layer_metrics if item.stat_sample_count > 0]
        stat_sample_count = min(sample_counts) if sample_counts else 0
        statistically_ready = stat_sample_count >= self.stat_window_size

        snapshot = RegimeSnapshot(
            layer_metrics=layer_metrics,
            entropy_sigma=float(median(sigma_values)) if sigma_values else 0.0,
            entropy_sigma_std_err=float(median(sigma_errors)) if sigma_errors else 0.0,
            regime_rho=float(median(rho_values)) if rho_values else 0.0,
            regime_rho_std_err=float(median(rho_errors)) if rho_errors else 0.0,
            effective_dimensionality=effective_dimensionality,
            effective_dimensionality_std_err=float(median(d_errors)) if d_errors else 0.0,
            dimensionality_ratio=dimensionality_ratio,
            stat_sample_count=stat_sample_count,
            stat_window_size=self.stat_window_size,
            statistically_ready=statistically_ready,
            equilibrium_fraction=equilibrium_fraction,
            equilibrium_gate=(equilibrium_fraction >= 0.8 and statistically_ready) if eq_values else False,
        )
        self._last_snapshot = snapshot
        return snapshot

    @property
    def current_regime(self) -> str:
        if self._last_snapshot is None:
            return "unknown"
        if not self._last_snapshot.statistically_ready:
            return "unknown"
        rho = self._last_snapshot.regime_rho
        if rho > 1.1:
            return "disordered"
        if rho < 0.9:
            return "ordered"
        return "critical"

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
