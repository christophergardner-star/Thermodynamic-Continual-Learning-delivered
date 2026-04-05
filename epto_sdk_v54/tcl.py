"""Thermodynamic Continual Learning (TCL).

During Task T, :class:`ThermalImportance` accumulates a per-parameter gradient
energy estimate EMA(g_i²) — the same signal as Adam's ``v_i`` state, but
tracked across the *whole* task rather than decayed at every step.

At task end, :class:`ThermalMemory`.commit() freezes both the weights and the
importance map into a :class:`TaskSnapshot`.  For Task T+1,
:class:`TCLRegularizer` adds::

    L_total = L_new + λ · Σ_{past t} decay^(T−t) · Σ_i I_i^t · A · (θ_i − θ_i^t)²

where A = ``anneal_rate``^step softens frozen knowledge over time (reannealing),
and ``decay`` down-weights older tasks exponentially.

:class:`TCLEngine` ties all components together into a single convenience API.

Why TCL differs from EWC
------------------------
* **EWC** takes a one-shot Fisher diagonal at task end, frozen forever.
* **TCL** uses a continuous EMA throughout training; with ``anneal_rate < 1``
  the importance fades over the next task's steps so that very long task
  sequences do not become rigidly locked — a critical property for 10+ task
  benchmarks such as Split-CIFAR-100 or Permuted MNIST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "TaskSnapshot",
    "ThermalImportance",
    "ThermalMemory",
    "TCLRegularizer",
    "TCLEngine",
]


# ---------------------------------------------------------------------------
# 1. TaskSnapshot
# ---------------------------------------------------------------------------


@dataclass
class TaskSnapshot:
    """Immutable record of a model's weights and importance for one task.

    Parameters
    ----------
    params:
        Per-parameter weight tensors copied at task end (``{name: tensor}``).
    importance:
        Per-parameter thermal importance map, i.e. EMA(g_i²) at task end.
    task_id:
        Zero-based index of the task this snapshot belongs to.
    """

    params: Dict[str, torch.Tensor]
    importance: Dict[str, torch.Tensor]
    task_id: int = 0


# ---------------------------------------------------------------------------
# 2. ThermalImportance
# ---------------------------------------------------------------------------


class ThermalImportance:
    """Per-parameter gradient-energy tracker using exponential moving average.

    After every ``loss.backward()`` during task T, call :meth:`update` to
    incorporate the current ``.grad`` tensors into the running EMA(g_i²).
    Unlike a one-shot Fisher diagonal (EWC), the EMA accumulates signal across
    the *entire* task, giving a richer measure of which parameters mattered.

    Parameters
    ----------
    model:
        The ``nn.Module`` whose parameters are tracked.
    alpha:
        EMA smoothing coefficient in ``(0, 1]``.  Higher values weight recent
        gradient steps more heavily.
    """

    def __init__(self, model: nn.Module, alpha: float = 0.1) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        self.model = model
        self.alpha = alpha
        self._ema: Dict[str, torch.Tensor] = {}
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the running EMA state.  Call before starting a new task."""
        self._ema.clear()
        self._n_updates = 0

    def update(self) -> None:
        """Incorporate current ``.grad`` tensors into the importance EMA.

        Parameters without gradients are silently skipped.  Must be called
        after every ``loss.backward()`` during training on the current task.
        """
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            g_sq = param.grad.detach().pow(2)
            if name not in self._ema:
                self._ema[name] = g_sq.clone()
            else:
                self._ema[name].mul_(1.0 - self.alpha).add_(
                    g_sq, alpha=self.alpha
                )
        self._n_updates += 1

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of the current importance map.

        Returns
        -------
        dict
            ``{parameter_name: cloned_importance_tensor}``
        """
        return {k: v.clone() for k, v in self._ema.items()}

    @property
    def n_updates(self) -> int:
        """Number of :meth:`update` calls since the last :meth:`reset`."""
        return self._n_updates


# ---------------------------------------------------------------------------
# 3. ThermalMemory
# ---------------------------------------------------------------------------


class ThermalMemory:
    """Stores :class:`TaskSnapshot` objects for every completed task.

    After training on task T, call :meth:`commit` to freeze both the current
    model weights and the accumulated importance into a new snapshot appended
    to the memory bank.  The importance tracker is reset automatically so it
    is ready for task T+1.

    Parameters
    ----------
    model:
        The ``nn.Module`` being trained continually.
    importance_tracker:
        A :class:`ThermalImportance` instance monitoring ``model``.
    """

    def __init__(
        self,
        model: nn.Module,
        importance_tracker: ThermalImportance,
    ) -> None:
        self.model = model
        self.importance_tracker = importance_tracker
        self._snapshots: List[TaskSnapshot] = []

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def commit(self) -> TaskSnapshot:
        """Snapshot current weights + importance, append to memory, reset tracker.

        Returns
        -------
        TaskSnapshot
            The snapshot that was just committed.
        """
        task_id = len(self._snapshots)
        params: Dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        importance = self.importance_tracker.get()
        snap = TaskSnapshot(params=params, importance=importance, task_id=task_id)
        self._snapshots.append(snap)
        self.importance_tracker.reset()
        return snap

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    @property
    def snapshots(self) -> List[TaskSnapshot]:
        """Ordered list of committed snapshots, oldest first."""
        return list(self._snapshots)

    @property
    def num_tasks(self) -> int:
        """Number of tasks committed so far."""
        return len(self._snapshots)


# ---------------------------------------------------------------------------
# 4. TCLRegularizer
# ---------------------------------------------------------------------------


class TCLRegularizer:
    """Thermodynamic continual-learning regularization penalty.

    Computes::

        penalty = λ · Σ_{t=0}^{T-1} decay^(T-t) · Σ_i I_i^t · A · (θ_i − θ_i^t)²

    where:

    * **T** = ``memory.num_tasks`` (number of completed tasks)
    * **I_i^t** = thermal importance of parameter *i* from task *t*
    * **θ_i^t** = frozen weight value at the end of task *t*
    * **θ_i** = current (trainable) weight value
    * **A** = ``anneal_rate ** step`` — the reannealing factor

    With ``anneal_rate < 1.0``, the effective importance of older task
    memories fades as training on the new task progresses.  This prevents the
    rigid over-constraint that afflicts EWC on long task sequences.

    Parameters
    ----------
    model:
        The model being trained.
    memory:
        The :class:`ThermalMemory` holding committed task snapshots.
    lambda_reg:
        Global regularization strength (≥ 0).
    decay:
        Temporal decay in ``(0, 1]``.  With ``decay < 1``, older tasks
        receive exponentially less weight than recent ones.
    anneal_rate:
        Per-step importance multiplier in ``(0, 1]``.  Values below 1 enable
        reannealing so frozen knowledge gradually softens during a new task.
    """

    def __init__(
        self,
        model: nn.Module,
        memory: ThermalMemory,
        *,
        lambda_reg: float = 1.0,
        decay: float = 1.0,
        anneal_rate: float = 1.0,
    ) -> None:
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be >= 0, got {lambda_reg!r}")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay!r}")
        if not 0.0 < anneal_rate <= 1.0:
            raise ValueError(
                f"anneal_rate must be in (0, 1], got {anneal_rate!r}"
            )
        self.model = model
        self.memory = memory
        self.lambda_reg = lambda_reg
        self.decay = decay
        self.anneal_rate = anneal_rate
        self._step: int = 0

    # ------------------------------------------------------------------
    # Annealing control
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the annealing counter.  Call once per optimizer step."""
        self._step += 1

    def reset_step(self) -> None:
        """Reset the step counter; call at the start of each new task."""
        self._step = 0

    @property
    def current_step(self) -> int:
        """Current value of the annealing step counter."""
        return self._step

    # ------------------------------------------------------------------
    # Penalty / loss
    # ------------------------------------------------------------------

    def penalty(self) -> torch.Tensor:
        """Compute the scalar regularization penalty.

        Returns a zero-dimensional tensor attached to the autograd graph so
        that gradients flow back through the model parameters.

        Returns
        -------
        torch.Tensor
            Scalar penalty (0-d tensor).
        """
        snapshots = self.memory.snapshots
        if not snapshots:
            return self._zero()

        T = len(snapshots)
        current_params = dict(self.model.named_parameters())
        anneal: float = self.anneal_rate ** self._step
        total: Optional[torch.Tensor] = None

        for t, snap in enumerate(snapshots):
            task_weight = self.decay ** (T - t)
            task_sum: Optional[torch.Tensor] = None

            for name, imp in snap.importance.items():
                if name not in current_params:
                    continue
                p = current_params[name]
                ref = snap.params[name].to(p.device)
                imp_dev = imp.to(p.device)
                contrib = (imp_dev * (p - ref).pow(2) * anneal).sum()
                task_sum = contrib if task_sum is None else task_sum + contrib

            if task_sum is not None:
                weighted = task_weight * task_sum
                total = weighted if total is None else total + weighted

        return self.lambda_reg * total if total is not None else self._zero()

    def loss(self, base_loss: torch.Tensor) -> torch.Tensor:
        """Return ``base_loss + self.penalty()``.

        Parameters
        ----------
        base_loss:
            The task-specific loss computed on the current mini-batch.

        Returns
        -------
        torch.Tensor
            Total loss (base + regularization).
        """
        return base_loss + self.penalty()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return torch.zeros((), device=device)


# ---------------------------------------------------------------------------
# 5. TCLEngine
# ---------------------------------------------------------------------------


class TCLEngine:
    """High-level orchestrator that wires together all TCL components.

    Typical usage::

        engine = TCLEngine(model, alpha=0.05, lambda_reg=400.0)

        # ── Task 0 ──────────────────────────────────────────────────────
        for x, y in task0_loader:
            loss = criterion(model(x), y)
            engine.backward_and_update(loss, optimizer)
        engine.end_task()

        # ── Task 1 ──────────────────────────────────────────────────────
        for x, y in task1_loader:
            base_loss = criterion(model(x), y)
            loss = engine.regularized_loss(base_loss)
            engine.backward_and_update(loss, optimizer)
        engine.end_task()

    Parameters
    ----------
    model:
        The continually trained ``nn.Module``.
    alpha:
        Importance EMA smoothing coefficient (passed to
        :class:`ThermalImportance`).
    lambda_reg:
        Regularization strength (passed to :class:`TCLRegularizer`).
    decay:
        Temporal decay across tasks (passed to :class:`TCLRegularizer`).
    anneal_rate:
        Per-step importance annealing (passed to :class:`TCLRegularizer`).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        alpha: float = 0.1,
        lambda_reg: float = 1.0,
        decay: float = 1.0,
        anneal_rate: float = 1.0,
    ) -> None:
        self.model = model
        self.importance = ThermalImportance(model, alpha=alpha)
        self.memory = ThermalMemory(model, self.importance)
        self.regularizer = TCLRegularizer(
            model,
            self.memory,
            lambda_reg=lambda_reg,
            decay=decay,
            anneal_rate=anneal_rate,
        )

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def update_importance(self) -> None:
        """Update importance from current gradients.  Call after backward."""
        self.importance.update()

    def end_task(self) -> TaskSnapshot:
        """Commit the current task snapshot and reset the annealing counter.

        Returns
        -------
        TaskSnapshot
            The snapshot that was committed.
        """
        snapshot = self.memory.commit()
        self.regularizer.reset_step()
        return snapshot

    def regularized_loss(self, base_loss: torch.Tensor) -> torch.Tensor:
        """Return *base_loss* augmented with the TCL regularization penalty."""
        return self.regularizer.loss(base_loss)

    def step(self) -> None:
        """Advance the annealing step counter.  Call once per optimizer step."""
        self.regularizer.step()

    def backward_and_update(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Convenience: zero_grad → backward → update importance → optimizer step → anneal.

        Parameters
        ----------
        loss:
            The (possibly regularized) loss tensor to differentiate.
        optimizer:
            The optimizer managing the model parameters.
        """
        optimizer.zero_grad()
        loss.backward()
        self.update_importance()
        optimizer.step()
        self.step()

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    @property
    def num_tasks(self) -> int:
        """Number of tasks committed so far."""
        return self.memory.num_tasks
