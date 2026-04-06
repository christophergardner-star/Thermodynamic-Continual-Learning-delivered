"""
tcl.py — Thermodynamic Continual Learning
==========================================
EPTO SDK v5.4.1

Continual learning via entropy-production importance weighting.

The core insight
----------------
During training, each parameter's gradient energy EMA(g_i²(t)) measures
how actively it participated in learning task T. Parameters with high
accumulated energy are "hot" — they encoded important task-specific
structure. Parameters with low energy are "cold" — they were largely
unused or already at equilibrium.

TCL uses this thermal history to protect hot parameters from forgetting
while allowing cold parameters to freely adapt to new tasks. Unlike EWC,
which requires a one-time Fisher snapshot at task end, TCL accumulates
importance continuously and supports gradual reannealing.

Algorithm
---------
Task T training:
  1. Call importance.accumulate() after every loss.backward().
     Internally: v_i = β·v_i + (1−β)·g_i²  (per-parameter EMA)
  2. At task end: call memory.commit(model, importance, task_id).
     Stores normalized importance I_i^T and a weight checkpoint θ^T.

Task T+1 training (any optimizer):
  1. Compute regularizer.penalty(model).
     L_reg = λ · Σ_{past t} decay^(T+1−t) · Σ_i I_i^t · (θ_i − θ_i^t)²
  2. Add to task loss: L_total = L_new + L_reg
  3. Backpropagate normally.

Reannealing (optional):
  importance.step_anneal()  — call once per optimizer step.
  Decays I_i^T → I_i^T · exp(−step / anneal_steps) over time.
  Hot parameters gradually become free, preventing rigidity across long
  task sequences.

Key differences from EWC
-------------------------
  - Importance from continuous gradient energy, not one-time Fisher snapshot.
  - Reannealing: knowledge softens naturally over time.
  - Naturally integrates with ThermoObserver for visualization.
  - Optimizer-agnostic — works with EPTO, AdamW, SGD, or anything.

Usage
-----
    from tcl import ThermalImportance, ThermalMemory, TCLRegularizer

    memory = ThermalMemory(max_tasks=5)

    # --- Task 0 ---
    importance = ThermalImportance(model, ema_beta=0.99)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for x, y in task0_loader:
        loss = criterion(model(x), y)
        loss.backward()
        importance.accumulate(model)   # <-- after backward, before step
        optimizer.step()
        optimizer.zero_grad()

    memory.commit(model, importance, task_id=0)

    # --- Task 1 ---
    importance = ThermalImportance(model, ema_beta=0.99)
    regularizer = TCLRegularizer(memory, lambda_tcl=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for x, y in task1_loader:
        loss = criterion(model(x), y)
        loss.backward()
        importance.accumulate(model)
        reg_loss = regularizer.penalty(model)   # <-- forgetting protection
        (loss + reg_loss).backward()            # only reg_loss backprop needed
        optimizer.step()
        optimizer.zero_grad()

    memory.commit(model, importance, task_id=1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ── ThermalImportance ──────────────────────────────────────────────────────────

class ThermalImportance:
    """Accumulates per-parameter gradient energy during one training task.

    For each parameter p with name n, maintains:
        v_n  = EMA_{steps}(g_n²)   (per-element, same shape as p)

    After task training, .finalize() returns normalized importance tensors
    that quantify how actively each parameter participated in learning.

    Parameters
    ----------
    model : nn.Module
        Model being trained.
    ema_beta : float
        EMA momentum for gradient energy. Higher = smoother / slower decay.
        0.99 works well for sequences of ~1000 steps.
    min_steps : int
        Minimum accumulation steps before finalize() produces meaningful values.
    """

    def __init__(
        self,
        model: nn.Module,
        ema_beta: float = 0.99,
        min_steps: int = 10,
    ):
        self.model = model
        self.ema_beta = ema_beta
        self.min_steps = min_steps

        # Per-parameter EMA of g²
        self._v: Dict[str, torch.Tensor] = {}
        self._step = 0

        # Register all trainable parameters
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._v[name] = torch.zeros_like(p.data, dtype=torch.float32)

    def accumulate(self, model: Optional[nn.Module] = None) -> None:
        """Call after loss.backward(), before optimizer.step().

        Updates the per-parameter EMA of squared gradients in-place.
        Silently skips parameters whose gradient is None.
        """
        m = model if model is not None else self.model
        beta = self.ema_beta
        for name, p in m.named_parameters():
            if p.requires_grad and p.grad is not None and name in self._v:
                g2 = p.grad.detach().float() ** 2
                self._v[name].mul_(beta).add_(g2, alpha=1.0 - beta)
        self._step += 1

    def finalize(self, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """Return importance map {param_name → importance_tensor}.

        Each tensor has the same shape as the corresponding parameter.
        Values are in [0, 1] if normalize=True (normalized within each
        named parameter group by its own max).

        Parameters
        ----------
        normalize : bool
            If True, normalize each parameter's importance to [0, 1].
            Ensures consistent penalty scale regardless of layer size.
        """
        result: Dict[str, torch.Tensor] = {}
        for name, v in self._v.items():
            imp = v.clone()
            if normalize:
                vmax = imp.max()
                if vmax > 1e-30:
                    imp = imp / vmax
            result[name] = imp
        return result

    @property
    def steps_accumulated(self) -> int:
        return self._step

    def is_ready(self) -> bool:
        return self._step >= self.min_steps


# ── ThermalCheckpoint ─────────────────────────────────────────────────────────

@dataclass
class ThermalCheckpoint:
    """Weight snapshot + importance map for one completed task.

    Attributes
    ----------
    task_id : int
        Caller-supplied task identifier.
    weights : Dict[str, Tensor]
        Detached CPU copies of model weights at task end.
    importance : Dict[str, Tensor]
        Normalized importance tensors (same shapes as weights).
    steps_trained : int
        Number of accumulation steps for this task's importance.
    """
    task_id: int
    weights: Dict[str, torch.Tensor]
    importance: Dict[str, torch.Tensor]
    steps_trained: int = 0
    _annealing_factors: Dict[str, torch.Tensor] = field(default_factory=dict)

    def anneal_step(self, anneal_rate: float) -> None:
        """Decay importance by anneal_rate (0 < anneal_rate < 1).

        Call once per optimizer step during subsequent task training.
        After ~1/anneal_rate steps the importance halves.
        """
        for name in self.importance:
            if name not in self._annealing_factors:
                self._annealing_factors[name] = torch.ones_like(self.importance[name])
            self._annealing_factors[name].mul_(anneal_rate)

    def effective_importance(self, name: str) -> torch.Tensor:
        """Return importance for this parameter, after any annealing applied."""
        base = self.importance.get(name)
        if base is None:
            return torch.zeros(1)
        factor = self._annealing_factors.get(name)
        if factor is None:
            return base
        return base * factor


# ── ThermalMemory ─────────────────────────────────────────────────────────────

class ThermalMemory:
    """Ring buffer of task checkpoints.

    Stores up to max_tasks completed task checkpoints. When full, the oldest
    task is dropped (least-recently-added). This keeps memory bounded while
    protecting recent task knowledge.

    Parameters
    ----------
    max_tasks : int
        Maximum checkpoints to retain. Default 10.
    task_decay : float
        Per-task multiplicative decay for older checkpoints in the penalty.
        E.g., 0.9 means task T-2 contributes 0.9² × its base penalty.
        Set to 1.0 for no decay (pure sum over all past tasks).
    anneal_rate : float
        Per-step decay applied to importance during subsequent training.
        E.g., 0.9999 → ~10% drop over 1000 steps.
        Set to 1.0 to disable reannealing (EWC-like frozen importance).
    """

    def __init__(
        self,
        max_tasks: int = 10,
        task_decay: float = 1.0,
        anneal_rate: float = 1.0,
    ):
        self.max_tasks = max_tasks
        self.task_decay = task_decay
        self.anneal_rate = anneal_rate
        self._tasks: List[ThermalCheckpoint] = []

    def commit(
        self,
        model: nn.Module,
        importance: ThermalImportance,
        task_id: int,
        normalize: bool = True,
    ) -> ThermalCheckpoint:
        """Snapshot current model weights and finalized importance.

        Stores on CPU to avoid occupying GPU memory.

        Parameters
        ----------
        model : nn.Module
            Model at task end (weights to protect).
        importance : ThermalImportance
            Accumulated gradient energy for this task.
        task_id : int
            Identifier for the completed task.
        normalize : bool
            Pass through to ThermalImportance.finalize().

        Returns
        -------
        ThermalCheckpoint
            The newly-stored checkpoint (also appended to memory).
        """
        weights: Dict[str, torch.Tensor] = {
            name: p.data.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        imp_map = importance.finalize(normalize=normalize)
        # Move to CPU for storage
        imp_map_cpu = {k: v.cpu() for k, v in imp_map.items()}

        ckpt = ThermalCheckpoint(
            task_id=task_id,
            weights=weights,
            importance=imp_map_cpu,
            steps_trained=importance.steps_accumulated,
        )

        self._tasks.append(ckpt)
        if len(self._tasks) > self.max_tasks:
            self._tasks.pop(0)

        return ckpt

    def anneal_all(self) -> None:
        """Advance reannealing by one step for all stored checkpoints.

        Call once per optimizer step during task T+1 training to let
        importance decay, gradually freeing frozen parameters.
        Only active when anneal_rate < 1.0.
        """
        if self.anneal_rate >= 1.0:
            return
        for ckpt in self._tasks:
            ckpt.anneal_step(self.anneal_rate)

    def penalty(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute the thermodynamic elastic penalty against all stored tasks.

        L_reg = λ_external · Σ_{t} task_decay^(N-1-idx) · Σ_i I_i^t · (θ_i − θ_i^t)²

        λ_external is the lambda_tcl from TCLRegularizer (not applied here).
        The per-task task_decay weights recent tasks more strongly.

        Parameters
        ----------
        model : nn.Module
            Model with current parameters.
        device : torch.device, optional
            Device for penalty computation. Inferred from model if None.

        Returns
        -------
        Tensor
            Scalar penalty (on same device as model parameters).
        """
        if not self._tasks:
            return torch.zeros(1, requires_grad=False)

        # Infer device
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        total_penalty = torch.zeros(1, device=device)
        n = len(self._tasks)

        param_map: Dict[str, torch.Tensor] = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }

        for idx, ckpt in enumerate(self._tasks):
            # More recent tasks get higher weight
            task_weight = self.task_decay ** (n - 1 - idx)

            for name, p in param_map.items():
                if name not in ckpt.weights:
                    continue

                theta_t = ckpt.weights[name].to(device)
                imp = ckpt.effective_importance(name).to(device)

                # Elastic term: I_i * (θ_i - θ_i^t)²
                diff = p.float() - theta_t.float()
                term = (imp * diff * diff).sum()
                total_penalty = total_penalty + task_weight * term

        return total_penalty

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    def task_ids(self) -> List[int]:
        return [ckpt.task_id for ckpt in self._tasks]

    def thermal_profile(self) -> List[Dict]:
        """Return a summary of importance statistics per stored task."""
        profile = []
        for ckpt in self._tasks:
            stats: Dict = {'task_id': ckpt.task_id, 'steps_trained': ckpt.steps_trained, 'layers': {}}
            for name, imp in ckpt.importance.items():
                stats['layers'][name] = {
                    'mean': float(imp.mean()),
                    'max': float(imp.max()),
                    'hot_fraction': float((imp > 0.5).float().mean()),
                }
            profile.append(stats)
        return profile


# ── TCLRegularizer ────────────────────────────────────────────────────────────

class TCLRegularizer:
    """Wraps ThermalMemory and scales the elastic penalty.

    Parameters
    ----------
    memory : ThermalMemory
        Memory containing past task checkpoints.
    lambda_tcl : float
        Regularization strength. 0 = no protection. 1.0 is a reasonable default.
        Scale similarly to EWC lambda (task-dependent; start with 0.5–5.0).
    """

    def __init__(self, memory: ThermalMemory, lambda_tcl: float = 1.0):
        self.memory = memory
        self.lambda_tcl = lambda_tcl

    def penalty(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return scaled thermodynamic elastic penalty.

        Typical usage in a training loop::

            loss = criterion(model(x), y)
            loss.backward()                     # task-specific gradients
            importance.accumulate(model)

            reg = regularizer.penalty(model)
            reg.backward()                      # protection gradients

            optimizer.step()
            optimizer.zero_grad()
            memory.anneal_all()                 # optional: decay importance

        Parameters
        ----------
        model : nn.Module
            Current model.
        device : optional
            Penalty device (inferred from model if None).

        Returns
        -------
        Tensor
            Scalar penalty = lambda_tcl * raw_penalty.
        """
        raw = self.memory.penalty(model, device=device)
        return self.lambda_tcl * raw


# ── TCLTrainer ────────────────────────────────────────────────────────────────

class TCLTrainer:
    """High-level interface for thermodynamic continual learning.

    Manages the task lifecycle: accumulate → commit → regularize → evaluate.

    Parameters
    ----------
    model : nn.Module
        The continually-trained model.
    optimizer_factory : callable
        Function that takes model.parameters() and returns a fresh optimizer.
        Called at the start of each task.
    memory : ThermalMemory, optional
        Shared memory across tasks. Created with defaults if not provided.
    ema_beta : float
        EMA momentum for ThermalImportance.
    lambda_tcl : float
        Elastic penalty strength.

    Example
    -------
    ::

        trainer = TCLTrainer(
            model=model,
            optimizer_factory=lambda p: torch.optim.AdamW(p, lr=3e-4),
            lambda_tcl=1.0,
        )

        for task_id, (train_loader, val_loader) in enumerate(tasks):
            history = trainer.learn_task(
                task_id=task_id,
                loader=train_loader,
                criterion=nn.CrossEntropyLoss(),
                epochs=5,
            )
            acc = trainer.evaluate(val_loader, criterion)
            print(f"Task {task_id} val acc: {acc:.3f}")

        # Full backward-transfer evaluation
        bt = trainer.evaluate_forgetting(all_val_loaders, nn.CrossEntropyLoss())
        print(f"Average forgetting: {bt['avg_forgetting']:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_factory,
        memory: Optional[ThermalMemory] = None,
        ema_beta: float = 0.99,
        lambda_tcl: float = 1.0,
    ):
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.memory = memory or ThermalMemory()
        self.ema_beta = ema_beta
        self.lambda_tcl = lambda_tcl

        # Per-task best accuracy snapshots (for forgetting measurement)
        self._task_peak_acc: Dict[int, float] = {}
        self._completed_tasks: List[int] = []

    def learn_task(
        self,
        task_id: int,
        loader,
        criterion: nn.Module,
        epochs: int = 1,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """Train the model on one task with thermodynamic forgetting protection.

        Parameters
        ----------
        task_id : int
            Identifier for this task (used in memory and reporting).
        loader : DataLoader
            Training data for this task.
        criterion : nn.Module
            Loss function. Must return a scalar.
        epochs : int
            Number of passes over loader.
        device : torch.device, optional
            Inferred from model if None.
        verbose : bool
            Print per-epoch summary.

        Returns
        -------
        List[Dict]
            Per-step training history with keys: step, epoch, task_loss, reg_loss, total_loss.
        """
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        optimizer = self.optimizer_factory(self.model.parameters())
        importance = ThermalImportance(self.model, ema_beta=self.ema_beta)
        regularizer = TCLRegularizer(self.memory, lambda_tcl=self.lambda_tcl)

        history: List[Dict] = []
        global_step = 0

        for epoch in range(epochs):
            epoch_task_loss = 0.0
            epoch_reg_loss = 0.0
            n_batches = 0

            for batch in loader:
                # Support (x, y) tuple or dict loaders
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")

                optimizer.zero_grad()

                # Forward + task loss
                logits = self.model(x)
                task_loss = criterion(logits, y)
                task_loss.backward()

                # Accumulate importance from task-specific gradients
                importance.accumulate(self.model)

                # Forgetting protection: elastic penalty
                reg_loss = regularizer.penalty(self.model, device=device)
                if self.memory.num_tasks > 0:
                    reg_loss.backward()

                optimizer.step()

                # Advance reannealing
                self.memory.anneal_all()

                tl = float(task_loss.item())
                rl = float(reg_loss.item()) if self.memory.num_tasks > 0 else 0.0
                epoch_task_loss += tl
                epoch_reg_loss += rl
                n_batches += 1
                global_step += 1

                history.append({
                    'step': global_step,
                    'epoch': epoch,
                    'task_loss': round(tl, 6),
                    'reg_loss': round(rl, 6),
                    'total_loss': round(tl + rl, 6),
                })

            if verbose and n_batches > 0:
                avg_tl = epoch_task_loss / n_batches
                avg_rl = epoch_reg_loss / n_batches
                print(
                    f"  [TCL] Task {task_id} | epoch {epoch+1}/{epochs}"
                    f" | task_loss={avg_tl:.4f} | reg_loss={avg_rl:.4f}"
                    f" | protected_tasks={self.memory.num_tasks}"
                )

        # Commit this task to memory
        self.memory.commit(self.model, importance, task_id=task_id)
        self._completed_tasks.append(task_id)

        return history

    def evaluate(
        self,
        loader,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
    ) -> Tuple[float, float]:
        """Evaluate model accuracy and loss on a single task loader.

        Returns
        -------
        Tuple[float, float]
            (accuracy, avg_loss)
        """
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")

                logits = self.model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item())

                preds = logits.argmax(dim=-1)
                correct += int((preds == y).sum())
                total += len(y)

        self.model.train()
        acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(list(loader)), 1)
        return acc, avg_loss

    def evaluate_forgetting(
        self,
        task_loaders: Dict[int, object],
        criterion: nn.Module,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """Compute backward transfer and average forgetting across all tasks.

        Requires that per-task peak accuracy was recorded during training
        (call .record_peak_acc(task_id, acc) after training each task).

        Parameters
        ----------
        task_loaders : dict
            {task_id: DataLoader} for each task to evaluate.
        criterion : nn.Module
            Loss function.

        Returns
        -------
        Dict with keys:
            per_task_acc : dict {task_id: current_acc}
            per_task_forgetting : dict {task_id: peak_acc - current_acc}
            avg_forgetting : float
        """
        per_task_acc = {}
        per_task_forgetting = {}

        for task_id, loader in task_loaders.items():
            acc, _ = self.evaluate(loader, criterion, device=device)
            per_task_acc[task_id] = acc
            peak = self._task_peak_acc.get(task_id, acc)
            per_task_forgetting[task_id] = max(0.0, peak - acc)

        avg_forgetting = (
            sum(per_task_forgetting.values()) / max(len(per_task_forgetting), 1)
        )

        return {
            'per_task_acc': per_task_acc,
            'per_task_forgetting': per_task_forgetting,
            'avg_forgetting': avg_forgetting,
        }

    def record_peak_acc(self, task_id: int, acc: float) -> None:
        """Record the best accuracy seen on task_id (for forgetting measurement)."""
        current = self._task_peak_acc.get(task_id, 0.0)
        self._task_peak_acc[task_id] = max(current, acc)
