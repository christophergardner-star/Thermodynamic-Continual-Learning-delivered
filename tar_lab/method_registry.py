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
