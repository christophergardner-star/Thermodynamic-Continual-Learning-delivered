"""
Shared observer variants for TAR research experiments.

These classes are import-safe and can be resolved by name from the experiment
queue, which lets the orchestrator run hypothesis-specific TCL variants without
hard-coding them into multiple entrypoints.
"""
from __future__ import annotations

from typing import Type

from tar_lab.thermoobserver import ActivationThermoObserver


class CarryoverAnchorObserver(ActivationThermoObserver):
    """Carry sigma-star across task boundaries instead of resetting per task."""

    def reset_for_new_task(self) -> None:
        if hasattr(self, "_sigma_star") and self._sigma_star is not None:
            self._collecting_anchor = False
            self._anchor_buffer = []
            return
        super().reset_for_new_task()


class StrictConsolidationObserver(ActivationThermoObserver):
    """Use narrower critical-band thresholds to force stronger consolidation."""

    @property
    def current_regime(self) -> str:
        rho = getattr(self, "rho", 1.0)
        if rho > 1.05:
            return "disordered"
        if rho < 0.85:
            return "ordered"
        return "critical"


class GraduatedPenaltyObserver(ActivationThermoObserver):
    """Scale the anchor penalty continuously with depth into ordered regime."""

    def _graduated_penalty_scale(self) -> float:
        rho = getattr(self, "rho", 1.0)
        if rho >= 0.9:
            return 0.0
        return (0.9 - rho) / 0.9

    def get_penalty_scale(self) -> float:
        return self._graduated_penalty_scale()


class DeepAnchorObserver(ActivationThermoObserver):
    """Longer sigma-star calibration to reduce anchor noise."""

    def __init__(self, model, **kwargs):
        kwargs.setdefault("sigma_star_anchor_n", 50)
        kwargs.setdefault("warmup_batches", 90)
        kwargs.setdefault("sigma_window_size", 12)
        kwargs.setdefault("sigma_tolerance", 0.12)
        super().__init__(model, **kwargs)


OBSERVER_REGISTRY: dict[str, Type[ActivationThermoObserver]] = {
    "CarryoverAnchorObserver": CarryoverAnchorObserver,
    "StrictConsolidationObserver": StrictConsolidationObserver,
    "GraduatedPenaltyObserver": GraduatedPenaltyObserver,
    "DeepAnchorObserver": DeepAnchorObserver,
}


def resolve_observer_class(name: str | None) -> Type[ActivationThermoObserver] | None:
    if not name:
        return None
    return OBSERVER_REGISTRY.get(name)
