"""
Forgetting-attribution probes — a faithful port of STRATA's pre-registered
"trichotomy of forgetting causes" methodology onto TAR's own continual-learning runs.

STRATA (github.com/christophergardner-star/CF) pre-registered the claim that every
diagnosable forgetting event is attributable to one of three causes, each with an
interventional EXCLUSION test (a test that PASSES rules the cause OUT):

  1. SUBSTRATE overlap   — restore the PARAMETERS changed during the interfering task to
                           their pre-task values (keeping buffers, e.g. BatchNorm running
                           stats, at post-task values). If this recovers < RECOVERY_THRESHOLD
                           of the lost accuracy, the loss was NOT carried by overwritten
                           substrate (substrate EXCLUDED). Recovery >= threshold => substrate
                           is a live cause. (Restoring params-not-buffers is what makes this
                           discriminate: a loss carried by BN-stat drift or representational
                           reorganisation will NOT recover from a param restore.)
  2. TIMESCALE collision  — restore only the TOP `TIMESCALE_TOP_FRACTION` most-changed
                           parameter ELEMENTS (the fastest/largest unprotected updates). If
                           recovery < threshold, the loss is NOT a function of unprotected
                           update speed (timescale EXCLUDED).
  3. BASIS correlation    — measure representational alignment between the forgotten task and
                           the interfering task at every layer (STRATA amendment 2:
                           covariance alignment A(X,Y)=tr(S_X S_Y)/(||S_X||_F ||S_Y||_F);
                           linear CKA also provided). If alignment < CKA_EXCLUSION at EVERY
                           layer while forgetting still occurs, the loss is NOT carried by
                           correlated bases (basis EXCLUDED). Alignment >= threshold at some
                           layer => basis correlation is a live cause.

An event is ATTRIBUTED to the cause(s) whose exclusion test FAILS. It is UNATTRIBUTED only if
all three exclusion tests pass. Soft-incompleteness: if the unattributed accuracy magnitude
exceeds INCOMPLETENESS_FRACTION of the total, the trichotomy is reported INCOMPLETE for the run
(STRATA's own pre-registered honesty clause).

This module is HARNESS-AGNOSTIC: it operates on torch state_dicts + caller-supplied closures
(an eval closure that applies parameter overrides, and per-layer activation arrays), so it can
be driven by scripts/run_forgetting_attribution.py (a self-contained demo) OR wired onto
tar_lab/generic_cl_runner outputs to diagnose TAR's real Split-CIFAR runs. It VALIDATES claims;
it does not assert them — a run may legitimately return unattributed events (trichotomy holes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

# Pre-registered thresholds (STRATA preregistration.md).
FORGETTING_REL_DROP = 0.20        # a forgetting EVENT = >=20% relative accuracy drop
RECOVERY_THRESHOLD = 0.50         # >=50% recovery => that cause is NOT excluded
TIMESCALE_TOP_FRACTION = 0.01     # top 1% most-changed parameter elements
CKA_EXCLUSION = 0.10              # alignment < 0.10 at every layer => basis excluded
INCOMPLETENESS_FRACTION = 0.30    # unattributed magnitude > 30% of total => trichotomy incomplete

CAUSES = ("substrate", "timescale", "basis")

ParamDict = "dict[str, torch.Tensor]"


# ── metrics ─────────────────────────────────────────────────────────────────────
def _center(m: np.ndarray) -> np.ndarray:
    return m - m.mean(axis=0, keepdims=True)


def covariance_alignment(x: np.ndarray, y: np.ndarray) -> float:
    """STRATA amendment-2 pairing-free basis overlap: A(X,Y)=tr(S_X S_Y)/(||S_X||_F ||S_Y||_F),
    where S_* is the (centered) feature covariance. 1.0 = identical covariance structure,
    ~0 = orthogonal representational subspaces. X: (n_x, d), Y: (n_y, d)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or x.shape[0] < 2 or y.shape[0] < 2:
        return 0.0
    xc, yc = _center(x), _center(y)
    sx = (xc.T @ xc) / (xc.shape[0] - 1)
    sy = (yc.T @ yc) / (yc.shape[0] - 1)
    denom = np.linalg.norm(sx, "fro") * np.linalg.norm(sy, "fro")
    if denom <= 1e-12:
        return 0.0
    return float(np.trace(sx @ sy) / denom)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered CKA between two activation sets (pre-amendment cross-check).
    Requires equal n (paired samples); returns 0.0 if not applicable."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return 0.0
    xc, yc = _center(x), _center(y)
    hsic = np.linalg.norm(xc.T @ yc, "fro") ** 2
    denom = np.linalg.norm(xc.T @ xc, "fro") * np.linalg.norm(yc.T @ yc, "fro")
    return float(hsic / denom) if denom > 1e-12 else 0.0


def recovery_fraction(acc_before: float, acc_after: float, acc_restored: float) -> float:
    """Fraction of the LOST accuracy that a restore recovers, clamped to [0,1].
    (restored - after) / (before - after). ~1 = full recovery, ~0 = none."""
    lost = acc_before - acc_after
    if lost <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, (acc_restored - acc_after) / lost)))


# ── parameter snapshots / restores ──────────────────────────────────────────────
def snapshot_parameters(model: torch.nn.Module) -> "dict[str, torch.Tensor]":
    """Clone the learnable PARAMETERS only (buffers like BN running stats are excluded —
    that exclusion is what lets the substrate test discriminate param-overwrite from
    buffer/representation drift)."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _changed_abs(before: "dict[str, torch.Tensor]", after: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    return {n: (after[n] - before[n]).abs() for n in before if n in after}


def full_restore_overrides(before: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    """Substrate test: every changed parameter -> its pre-task value (== the before params)."""
    return {n: t.clone() for n, t in before.items()}


def top_fraction_restore_overrides(before: "dict[str, torch.Tensor]", after: "dict[str, torch.Tensor]",
                                   fraction: float = TIMESCALE_TOP_FRACTION) -> "dict[str, torch.Tensor]":
    """Timescale test: keep AFTER values everywhere EXCEPT the top-`fraction` most-changed
    parameter elements (across the whole model), which are reverted to BEFORE values."""
    changed = _changed_abs(before, after)
    all_deltas = torch.cat([v.reshape(-1) for v in changed.values()]) if changed else torch.tensor([])
    if all_deltas.numel() == 0:
        return {n: after[n].clone() for n in after}
    k = max(1, int(all_deltas.numel() * fraction))
    thresh = torch.kthvalue(all_deltas, all_deltas.numel() - k + 1).values  # k-th largest
    overrides: "dict[str, torch.Tensor]" = {}
    for n in after:
        if n not in changed:
            overrides[n] = after[n].clone()
            continue
        mask = changed[n] >= thresh                     # top-fraction elements
        overrides[n] = torch.where(mask, before[n], after[n]).clone()
    return overrides


# ── exclusion tests ─────────────────────────────────────────────────────────────
@dataclass
class ExclusionTest:
    cause: str
    statistic: float          # recovery fraction (substrate/timescale) or max-layer alignment (basis)
    threshold: float
    excluded: bool            # True => this cause is ruled OUT for the event
    detail: str = ""


def substrate_test(acc_before: float, acc_after: float,
                   before: "dict[str, torch.Tensor]",
                   eval_with_overrides: "Callable[[dict], float]") -> ExclusionTest:
    acc_r = eval_with_overrides(full_restore_overrides(before))
    rec = recovery_fraction(acc_before, acc_after, acc_r)
    return ExclusionTest("substrate", rec, RECOVERY_THRESHOLD, rec < RECOVERY_THRESHOLD,
                         f"param-restore recovery={rec:.3f} (acc {acc_after:.3f}->{acc_r:.3f}, peak {acc_before:.3f})")


def timescale_test(acc_before: float, acc_after: float,
                   before: "dict[str, torch.Tensor]", after: "dict[str, torch.Tensor]",
                   eval_with_overrides: "Callable[[dict], float]",
                   fraction: float = TIMESCALE_TOP_FRACTION) -> ExclusionTest:
    acc_r = eval_with_overrides(top_fraction_restore_overrides(before, after, fraction))
    rec = recovery_fraction(acc_before, acc_after, acc_r)
    return ExclusionTest("timescale", rec, RECOVERY_THRESHOLD, rec < RECOVERY_THRESHOLD,
                         f"top-{fraction:.0%}-restore recovery={rec:.3f}")


def basis_test(acts_forgotten: "dict[str, np.ndarray]", acts_interfering: "dict[str, np.ndarray]",
               use_cka: bool = False) -> ExclusionTest:
    """Basis EXCLUDED iff alignment < CKA_EXCLUSION at EVERY layer (representations decorrelated
    yet forgetting occurred). Otherwise basis correlation is a live cause."""
    layers = [l for l in acts_forgotten if l in acts_interfering]
    aligns = []
    for l in layers:
        a = linear_cka(acts_forgotten[l], acts_interfering[l]) if use_cka \
            else covariance_alignment(acts_forgotten[l], acts_interfering[l])
        aligns.append(a)
    max_align = max(aligns) if aligns else 0.0
    excluded = bool(aligns) and all(a < CKA_EXCLUSION for a in aligns)
    metric = "cka" if use_cka else "cov_alignment"
    return ExclusionTest("basis", max_align, CKA_EXCLUSION, excluded,
                         f"max-layer {metric}={max_align:.3f} over {len(layers)} layers")


# ── attribution ─────────────────────────────────────────────────────────────────
@dataclass
class ForgettingEvent:
    forgotten_task: int
    after_task: int
    peak_acc: float           # best acc ever seen on forgotten_task (before interference)
    current_acc: float        # acc on forgotten_task after training after_task
    rel_drop: float


@dataclass
class EventAttribution:
    event: ForgettingEvent
    tests: "list[ExclusionTest]"
    attributed_causes: "list[str]"
    unattributed: bool


@dataclass
class RunAttribution:
    events: "list[EventAttribution]" = field(default_factory=list)
    n_events: int = 0
    n_attributed: int = 0
    unattributed_fraction: float = 0.0
    incomplete: bool = False

    def to_dict(self) -> dict:
        return {
            "n_events": self.n_events,
            "n_attributed": self.n_attributed,
            "unattributed_fraction": round(self.unattributed_fraction, 4),
            "trichotomy_incomplete": self.incomplete,
            "cause_counts": {c: sum(1 for e in self.events if c in e.attributed_causes) for c in CAUSES},
            "events": [
                {
                    "forgotten_task": e.event.forgotten_task,
                    "after_task": e.event.after_task,
                    "peak_acc": round(e.event.peak_acc, 4),
                    "current_acc": round(e.event.current_acc, 4),
                    "rel_drop": round(e.event.rel_drop, 4),
                    "attributed_causes": e.attributed_causes,
                    "unattributed": e.unattributed,
                    "tests": [{"cause": t.cause, "statistic": round(t.statistic, 4),
                               "threshold": t.threshold, "excluded": t.excluded, "detail": t.detail}
                              for t in e.tests],
                }
                for e in self.events
            ],
        }


def identify_forgetting_events(acc_matrix: "list[list[float]]",
                               rel_drop: float = FORGETTING_REL_DROP) -> "list[ForgettingEvent]":
    """acc_matrix[k][j] = accuracy on task j after training through task k (j<=k; None/NaN if
    task j not yet learned). An event = a task j whose accuracy after task k dropped by
    >= rel_drop RELATIVE to its peak (max acc over rows <= k-1... i.e. before task k)."""
    events: "list[ForgettingEvent]" = []
    n = len(acc_matrix)
    for k in range(1, n):
        for j in range(k):  # tasks learned before k
            cur = acc_matrix[k][j]
            if cur is None or (isinstance(cur, float) and np.isnan(cur)):
                continue
            peak = max((acc_matrix[r][j] for r in range(k)
                        if acc_matrix[r][j] is not None and not (isinstance(acc_matrix[r][j], float) and np.isnan(acc_matrix[r][j]))),
                       default=None)
            if peak is None or peak <= 1e-9:
                continue
            if (peak - cur) / peak >= rel_drop:
                events.append(ForgettingEvent(forgotten_task=j, after_task=k, peak_acc=float(peak),
                                              current_acc=float(cur), rel_drop=float((peak - cur) / peak)))
    return events


def attribute_event(event: ForgettingEvent, *,
                    params_before: "dict[str, torch.Tensor]",
                    params_after: "dict[str, torch.Tensor]",
                    eval_forgotten_with_overrides: "Callable[[dict], float]",
                    acts_forgotten: "dict[str, np.ndarray]",
                    acts_interfering: "dict[str, np.ndarray]",
                    use_cka: bool = False) -> EventAttribution:
    tests = [
        substrate_test(event.peak_acc, event.current_acc, params_before, eval_forgotten_with_overrides),
        timescale_test(event.peak_acc, event.current_acc, params_before, params_after, eval_forgotten_with_overrides),
        basis_test(acts_forgotten, acts_interfering, use_cka=use_cka),
    ]
    attributed = [t.cause for t in tests if not t.excluded]
    return EventAttribution(event=event, tests=tests, attributed_causes=attributed,
                            unattributed=len(attributed) == 0)


def summarize_run(attributions: "list[EventAttribution]") -> RunAttribution:
    run = RunAttribution(events=list(attributions), n_events=len(attributions))
    run.n_attributed = sum(1 for e in attributions if not e.unattributed)
    total_mag = sum(e.event.peak_acc - e.event.current_acc for e in attributions)
    unattr_mag = sum(e.event.peak_acc - e.event.current_acc for e in attributions if e.unattributed)
    run.unattributed_fraction = (unattr_mag / total_mag) if total_mag > 1e-9 else 0.0
    run.incomplete = run.unattributed_fraction > INCOMPLETENESS_FRACTION
    return run
