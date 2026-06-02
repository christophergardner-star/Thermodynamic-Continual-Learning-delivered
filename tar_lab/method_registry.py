"""
CLMethod plugin registry for the TAR generic continual-learning runner.

Built-in methods:  sgd_generic, ewc_generic, si_generic, der_plus_plus
Generated methods: loaded dynamically from tar_state/synthesized_methods/
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

METHOD_REGISTRY: dict[str, type["CLMethod"]] = {}


def register_method(name: str) -> Callable:
    def decorator(cls: type["CLMethod"]) -> type["CLMethod"]:
        METHOD_REGISTRY[name] = cls
        return cls
    return decorator


def load_generated_methods(generated_dir: Path) -> None:
    """Dynamically load validated synthesized method classes."""
    if not generated_dir.exists():
        return
    import importlib.util
    for py_file in sorted(generated_dir.glob("*.py")):
        method_name = py_file.stem
        if method_name in METHOD_REGISTRY:
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"_synth_{method_name}", py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, CLMethod)
                    and attr is not CLMethod
                ):
                    METHOD_REGISTRY[method_name] = attr
                    print(f"[method_registry] Loaded synthesized method: {method_name}", flush=True)
                    break
        except Exception as exc:
            print(f"[method_registry] Failed to load '{method_name}': {exc}", flush=True)


class CLMethod(ABC):
    """
    Base interface for all TAR continual-learning methods.

    The generic runner calls hooks in this order per task:
      1. pre_task(task_id, model, device)           — before task training
      2. Per batch:
           regularization_loss(model)              — added to cross-entropy each step
           augmented_loss(model, x, y, task_id, device)  — replay / extra loss terms
      3. post_task(task_id, model, train_loader, device) — after task completes
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def pre_task(self, task_id: int, model: nn.Module, device: torch.device) -> None:
        pass

    def post_task(
        self,
        task_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        pass

    @abstractmethod
    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        ...

    def augmented_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor(0.0, device=device)


# ── SGD baseline ───────────────────────────────────────────────────────────────

@register_method("sgd_generic")
class SGDBaseline(CLMethod):
    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        return torch.tensor(0.0)


# ── Elastic Weight Consolidation ───────────────────────────────────────────────

@register_method("ewc_generic")
class EWCMethod(CLMethod):
    """Kirkpatrick et al. 2017 — diagonal Fisher penalty."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Default raised to 1000.0 — empirically best on split_cifar10 (Phase 12 sweep:
        # lambda=1000 forgetting=0.160, p=0.318 vs TCL; lambda=100 forgetting=0.191, p=0.019).
        self.ewc_lambda = float(getattr(config, "ewc_lambda", 1000.0))
        self.fisher: dict[str, torch.Tensor] = {}
        self.optimal: dict[str, torch.Tensor] = {}

    def post_task(
        self, task_id: int, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> None:
        new_f: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.eval()
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new_f[n] += p.grad.data.pow(2)
            n_batches += 1
        if n_batches:
            for n in new_f:
                new_f[n] /= n_batches
                self.fisher[n] = (
                    self.fisher.get(n, torch.zeros_like(new_f[n])) + new_f[n]
                )
        self.optimal = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.train()

    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        if not self.fisher:
            return torch.tensor(0.0)
        dev = next(model.parameters()).device
        pen = torch.tensor(0.0, device=dev)
        for n, p in model.named_parameters():
            if n in self.fisher:
                pen = pen + (
                    self.fisher[n].to(dev) * (p - self.optimal[n].to(dev)).pow(2)
                ).sum()
        return self.ewc_lambda * pen


# ── Synaptic Intelligence ──────────────────────────────────────────────────────

@register_method("si_generic")
class SIMethod(CLMethod):
    """Zenke et al. 2017 — path-integral importance weights."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Default c=0.01 — Phase 13 sweep showed c=0.1 causes universal model collapse
        # on split_cifar10 (all 5 seeds at 0.500 accuracy). c=0.01 is the non-collapsing
        # setting and the value used in the locked HPC validation suite.
        self.si_c   = float(getattr(config, "si_c",   0.01))
        self.si_xi  = float(getattr(config, "si_xi",  0.001))
        self.omega:  dict[str, torch.Tensor] = {}
        self._prev:  dict[str, torch.Tensor] = {}
        self._W:     dict[str, torch.Tensor] = {}

    def pre_task(self, task_id: int, model: nn.Module, device: torch.device) -> None:
        self._prev = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self._W = {
            n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad
        }

    def augmented_loss(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
        task_id: int, device: torch.device,
    ) -> torch.Tensor:
        # Accumulate path integral W during training (called after backward)
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and n in self._prev:
                delta = p.data - self._prev[n].to(p.device)
                self._W[n] = self._W.get(n, torch.zeros_like(p)) - p.grad.data * delta
        return torch.tensor(0.0, device=device)

    def post_task(
        self, task_id: int, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            delta_sq = (
                p.data - self._prev.get(n, p.data).to(p.device)
            ).pow(2) + self.si_xi
            new_omega = self._W.get(n, torch.zeros_like(p)) / delta_sq
            self.omega[n] = (
                self.omega.get(n, torch.zeros_like(p)) + F.relu(new_omega)
            )
        self._prev = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }

    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        if not self.omega:
            return torch.tensor(0.0)
        dev = next(model.parameters()).device
        pen = torch.tensor(0.0, device=dev)
        for n, p in model.named_parameters():
            if n in self.omega:
                opt = self._prev.get(n, p.data)
                pen = pen + (
                    self.omega[n].to(dev) * (p - opt.to(dev)).pow(2)
                ).sum()
        return self.si_c * pen


# ── Dark Experience Replay++ ───────────────────────────────────────────────────

@register_method("der_plus_plus")
class DERPlusPlus(CLMethod):
    """Buzzega et al. 2020 — reservoir memory with logit distillation."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.mem_size = int(getattr(config, "der_mem_size", 200))
        self.alpha    = float(getattr(config, "der_alpha",   0.2))
        self.beta     = float(getattr(config, "der_beta",    0.5))
        self._mem_x:      list[torch.Tensor] = []
        self._mem_y:      list[torch.Tensor] = []
        self._mem_logits: list[torch.Tensor] = []
        self._n_seen = 0

    def _reservoir(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor
    ) -> None:
        for i in range(x.size(0)):
            if len(self._mem_x) < self.mem_size:
                self._mem_x.append(x[i].detach().cpu())
                self._mem_y.append(y[i].detach().cpu())
                self._mem_logits.append(logits[i].detach().cpu())
            else:
                # Reservoir sampling (Algorithm R): j ~ Uniform[0, n_seen]
                # Use self._n_seen *before* incrementing so each item's inclusion
                # probability is mem_size / (n_seen + 1) as required.
                idx = random.randint(0, self._n_seen)
                if idx < self.mem_size:
                    self._mem_x[idx]      = x[i].detach().cpu()
                    self._mem_y[idx]      = y[i].detach().cpu()
                    self._mem_logits[idx] = logits[i].detach().cpu()
            self._n_seen += 1

    def augmented_loss(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
        task_id: int, device: torch.device,
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = model(x)
        self._reservoir(x, y, logits)
        if len(self._mem_x) < 4:
            return torch.tensor(0.0, device=device)
        n = min(len(self._mem_x), x.size(0))
        idx = random.sample(range(len(self._mem_x)), n)
        mx = torch.stack([self._mem_x[i]      for i in idx]).to(device)
        my = torch.stack([self._mem_y[i]       for i in idx]).to(device)
        ml = torch.stack([self._mem_logits[i]  for i in idx]).to(device)
        out = model(mx)
        return self.beta * F.cross_entropy(out, my) + self.alpha * F.mse_loss(out, ml)

    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        return torch.tensor(0.0)


# ── Learning without Forgetting ────────────────────────────────────────────────

@register_method("lwf")
class LwFMethod(CLMethod):
    """
    Learning without Forgetting (Li & Hoiem, 2016).

    Distills knowledge from the model snapshot taken after training task T into
    the model during training on task T+1.  The distillation loss is a
    temperature-scaled KL divergence between the new model's output
    distribution and the frozen old model's output distribution:

        L_distill = alpha * T^2 * KL( softmax(new/T) || softmax(old/T) )

    The T^2 factor compensates for the gradient magnitude reduction caused by
    temperature scaling: when logits are divided by T, the magnitude of the
    softmax-input gradients is reduced by 1/T^2 relative to the T=1 case
    (Hinton, Vinyals & Dean, 2015, "Distilling the Knowledge in a Neural
    Network").  Multiplying by T^2 restores the effective gradient scale so
    that the distillation signal is comparable to the cross-entropy loss
    regardless of the chosen temperature.

    Implementation notes
    --------------------
    * The old model snapshot is stored on CPU after each task to avoid
      occupying GPU VRAM between tasks; it is moved to the active device
      in pre_task() at the start of the next task.
    * regularization_loss() returns 0.0; all knowledge-transfer cost is
      computed in augmented_loss() to keep a clean gradient graph.
    * augmented_loss() performs a *fresh* forward pass through the new model
      (not reusing the CE forward pass) so that the distillation gradient
      graph is independent.

    Hyperparameters (set via config_overrides)
    ------------------------------------------
    lwf_alpha       : distillation weight (default 0.5)
    lwf_temperature : softening temperature (default 2.0)

    References
    ----------
    Li, Z. & Hoiem, D. (2016). Learning without Forgetting. ECCV 2016.
    arXiv:1606.09282.

    Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the Knowledge in a
    Neural Network. NIPS 2015 Deep Learning Workshop. arXiv:1503.02531.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.alpha       = float(getattr(config, "lwf_alpha",       0.5))
        self.temperature = float(getattr(config, "lwf_temperature", 2.0))
        self._old_model: nn.Module | None = None

    def pre_task(self, task_id: int, model: nn.Module, device: torch.device) -> None:
        """Move the previous-task snapshot to the active device before training begins."""
        if task_id > 0 and self._old_model is not None:
            self._old_model = self._old_model.to(device)

    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """No parameter-space penalty; distillation is handled in augmented_loss."""
        return torch.tensor(0.0)

    def augmented_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute the temperature-scaled KL distillation loss on the current batch.

        Returns 0.0 for task 0 (no previous model exists).

        The returned loss is non-zero and will trigger a second backward() in
        the generic runner, accumulating distillation gradients into the same
        .grad buffers as the cross-entropy backward.
        """
        if self._old_model is None or task_id == 0:
            return torch.tensor(0.0, device=device)

        x = x.to(device)

        # Fresh forward pass through the *new* model for the distillation graph.
        # We cannot reuse the CE forward pass because that graph was consumed by
        # loss.backward() already.
        new_logits = model(x)

        with torch.no_grad():
            # Reference logits from the frozen snapshot; no gradient needed.
            old_logits = self._old_model(x)

        # Temperature-scaled distributions
        old_soft     = F.softmax(old_logits / self.temperature, dim=-1)
        new_log_soft = F.log_softmax(new_logits / self.temperature, dim=-1)

        # KL divergence: reduction="batchmean" gives the mean over the batch,
        # which matches the definition used in the original LwF paper.
        distill_loss = F.kl_div(new_log_soft, old_soft, reduction="batchmean")

        # T^2 restores gradient magnitude (see class docstring for derivation).
        return self.alpha * (self.temperature ** 2) * distill_loss

    def post_task(
        self,
        task_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        Snapshot the current model after task T completes.

        Stored on CPU so that GPU VRAM is freed during the next task's training;
        pre_task() moves the snapshot back to the GPU at the start of task T+1.
        """
        import copy
        self._old_model = copy.deepcopy(model).eval().cpu()


# ── Averaged Gradient Episodic Memory ─────────────────────────────────────────

@register_method("agem")
class AGEMMethod(CLMethod):
    """
    Averaged Gradient Episodic Memory (Chaudhry et al., 2019).

    Enforces that every gradient update for the current task does not increase
    the average loss on an episodic reference memory sampled from all
    previously seen tasks.  Formally, the constraint is:

        g_task . g_ref >= 0

    where g_task is the gradient of the current-task cross-entropy (already
    accumulated in parameter .grad buffers when augmented_loss() is called)
    and g_ref is the gradient of the cross-entropy on a random mini-batch
    drawn from episodic memory.

    If the constraint is violated (inner product < 0), g_task is projected
    onto the constraint half-space defined by g_ref:

        g_task' = g_task - ( g_task . g_ref / ||g_ref||^2 ) * g_ref

    This projection is performed *in-place* on the .grad buffers so that the
    corrected gradient is used by the subsequent optimizer.step() call.
    augmented_loss() then returns tensor(0.0) so the generic runner's
    `if aug.item() != 0.0` guard does NOT trigger a second backward() — the
    gradient correction has already been applied.

    The reference gradient is computed via torch.autograd.grad(), which
    returns fresh gradient tensors WITHOUT accumulating into .grad, thereby
    keeping the task gradient and reference gradient computations independent.

    Episodic memory is populated by reservoir sampling (Algorithm R, Vitter
    1985) in post_task(), guaranteeing a uniform inclusion probability of
    mem_size / n_seen for each observed sample.

    Hyperparameters (set via config_overrides)
    ------------------------------------------
    agem_mem_size : total episodic memory capacity across all tasks (default 200)

    References
    ----------
    Chaudhry, A., Ranzato, M., Rohrbach, M. & Elhoseiny, M. (2019).
    Efficient Lifelong Learning with A-GEM. ICLR 2019. arXiv:1812.00420.

    Vitter, J. S. (1985). Random Sampling with a Reservoir. ACM Transactions
    on Mathematical Software, 11(1), 37-57.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.mem_size    = int(getattr(config, "agem_mem_size", 200))
        self._mem_x:     list[torch.Tensor] = []
        self._mem_y:     list[torch.Tensor] = []
        self._total_seen: int = 0

    def pre_task(self, task_id: int, model: nn.Module, device: torch.device) -> None:
        """No-op: A-GEM requires no task-boundary model surgery."""
        pass

    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """No parameter-space penalty; constraint is enforced in augmented_loss."""
        return torch.tensor(0.0)

    def augmented_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Enforce the A-GEM gradient constraint and project if violated.

        Called AFTER loss.backward() has populated .grad on all parameters.
        Returns tensor(0.0) in all cases; the correction is applied in-place.

        Steps
        -----
        1. If no episodic memory exists (task 0), return zero immediately.
        2. Sample up to 256 items uniformly at random from episodic memory.
        3. Compute g_ref via torch.autograd.grad() -- does NOT touch .grad.
        4. Compute dot = g_task . g_ref and ref_norm_sq = ||g_ref||^2.
        5. If dot < 0 (constraint violated) project g_task in-place:
               g_task' = g_task - (dot / ref_norm_sq) * g_ref
        6. Return tensor(0.0) so the runner does not call aug.backward().
        """
        if task_id == 0 or not self._mem_x:
            return torch.tensor(0.0, device=device)

        # --- Sample reference batch ---
        n_ref = min(256, len(self._mem_x))
        idx   = random.sample(range(len(self._mem_x)), n_ref)
        mx = torch.stack([self._mem_x[i] for i in idx]).to(device)
        my = torch.stack([self._mem_y[i] for i in idx]).to(device)

        # --- Parameters with existing task gradients ---
        # Only consider parameters that already have a .grad from the CE backward.
        # Parameters without .grad cannot be projected and are skipped.
        params = [
            p for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if not params:
            return torch.tensor(0.0, device=device)

        # --- Reference gradient (does NOT accumulate into .grad) ---
        mem_out  = model(mx)
        mem_loss = F.cross_entropy(mem_out, my)

        g_ref_list = torch.autograd.grad(
            mem_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # --- Inner product g_task . g_ref and ||g_ref||^2 ---
        dot         = torch.tensor(0.0, device=device)
        ref_norm_sq = torch.tensor(0.0, device=device)
        for p, g_ref in zip(params, g_ref_list):
            if g_ref is not None and p.grad is not None:
                dot         = dot         + (p.grad.data * g_ref.data).sum()
                ref_norm_sq = ref_norm_sq + (g_ref.data ** 2).sum()

        # --- Project in-place if constraint is violated ---
        # Guard ref_norm_sq > epsilon to avoid division by a degenerate reference.
        if dot.item() < 0.0 and ref_norm_sq.item() > 1e-10:
            proj_coeff = dot / (ref_norm_sq + 1e-10)
            for p, g_ref in zip(params, g_ref_list):
                if g_ref is not None and p.grad is not None:
                    p.grad.data.sub_(proj_coeff * g_ref.data)

        # Return zero -- projection was in-place; no second backward() needed.
        return torch.tensor(0.0, device=device)

    def post_task(
        self,
        task_id: int,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        Populate episodic memory via reservoir sampling (Algorithm R).

        Iterates through the full training loader for the completed task and
        applies reservoir sampling to maintain a memory of size self.mem_size
        with a uniform inclusion probability of mem_size / n_total_seen for
        every sample observed across all tasks.

        Samples are stored detached on CPU to minimise memory footprint.
        """
        for x_batch, y_batch in loader:
            for xi, yi in zip(x_batch, y_batch):
                if self._total_seen < self.mem_size:
                    # Memory not yet full: always insert.
                    self._mem_x.append(xi.detach().cpu())
                    self._mem_y.append(yi.detach().cpu())
                else:
                    # Reservoir replacement: draw j ~ Uniform[0, total_seen].
                    # Replace slot j if j < mem_size, preserving the invariant
                    # that each sample has inclusion probability
                    # mem_size / (total_seen + 1).
                    j = random.randint(0, self._total_seen)
                    if j < self.mem_size:
                        self._mem_x[j] = xi.detach().cpu()
                        self._mem_y[j] = yi.detach().cpu()
                self._total_seen += 1


# ── Thermodynamic Continual Learning ──────────────────────────────────────────

@register_method("tcl")
class TCLMethod(CLMethod):
    """
    Thermodynamic Continual Learning via gradient-energy EMA importance.

    Accumulates per-parameter gradient-squared EMA (ThermalImportance) during
    task training. After each task, commits a checkpoint (weights + importance)
    to the elastic ring buffer (ThermalMemory). During subsequent tasks applies
    an importance-weighted L2 penalty proportional to parameter drift from the
    committed checkpoints.

    The ThermalImportance.accumulate() call is made inside augmented_loss()
    (after loss.backward() has populated .grad) so that gradient energy is
    captured from the fully-computed gradient, not just the CE loss component.
    augmented_loss() returns tensor(0.0) so no second backward() is triggered.

    Key config fields (read via getattr with defaults):
      tcl_penalty_lambda: float = 1.0   — elastic penalty weight
      tcl_ema_beta:       float = 0.99  — EMA decay for importance accumulation

    Reference: ThermalImportance / ThermalMemory / TCLRegularizer in tcl.py
               (TAR internal; not yet published).
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Deferred imports to avoid circular import issues at module load time
        from tcl import ThermalMemory, TCLRegularizer  # noqa: PLC0415
        self.tcl_penalty_lambda = float(getattr(config, "tcl_penalty_lambda", 1.0))
        self.tcl_ema_beta       = float(getattr(config, "tcl_ema_beta",       0.99))
        self._memory             = ThermalMemory()
        self._regularizer        = TCLRegularizer(self._memory, lambda_tcl=self.tcl_penalty_lambda)
        self._importance         = None  # created per-task in pre_task()

    def pre_task(self, task_id: int, model: nn.Module, device: torch.device) -> None:
        """Create a fresh ThermalImportance accumulator for this task."""
        from tcl import ThermalImportance  # noqa: PLC0415
        self._importance = ThermalImportance(model, ema_beta=self.tcl_ema_beta)

    def augmented_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Accumulate gradient energy from the already-computed .grad buffers.

        Called AFTER loss.backward() has populated .grad on all parameters.
        Returns tensor(0.0) so the generic runner's
        `if aug.item() != 0.0` guard does NOT trigger a second backward().
        """
        if self._importance is not None:
            self._importance.accumulate(model)
        return torch.tensor(0.0, device=device)

    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Return the importance-weighted elastic penalty from all committed tasks."""
        device = next(model.parameters()).device
        return self._regularizer.penalty(model, device=device)

    def post_task(
        self,
        task_id: int,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        Commit the current task's weight checkpoint and importance map to memory.

        ThermalMemory.commit() accepts the ThermalImportance object directly and
        calls .finalize() internally to produce the normalized importance tensors.
        A fresh importance accumulator is created in the next pre_task() call.
        """
        if self._importance is not None:
            self._memory.commit(model, self._importance, task_id=task_id)
        # Reset so stale gradients cannot bleed into the next task's accumulation
        self._importance = None
