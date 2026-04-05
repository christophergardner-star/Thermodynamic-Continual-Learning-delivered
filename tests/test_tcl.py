"""
tests/test_tcl.py — Thermodynamic Continual Learning unit tests

Tests are CPU-only and synthetic. No GPU required.

Benchmark structure
-------------------
Two synthetic binary classification tasks on a small MLP:

  Task 0: classify input vectors by their first feature (x[0] > 0)
  Task 1: classify input vectors by their last feature (x[-1] > 0)

These tasks conflict: optimally solving Task 1 damages Task 0 performance
because the model shares the same weights. TCL should mitigate this.

Reference baseline: no regularizer → high forgetting on Task 0 after Task 1.
TCL baseline: ThermalMemory(anneal_rate=1.0, task_decay=1.0) → low forgetting.
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from epto_sdk_v54.tcl import (
    ThermalImportance,
    ThermalMemory,
    TCLRegularizer,
    TCLTrainer,
    ThermalCheckpoint,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_task_data(n: int = 400, dim: int = 16, feature_idx: int = 0, seed: int = 42):
    """Binary classification: label = (x[feature_idx] > 0)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(n, dim, generator=rng)
    y = (X[:, feature_idx] > 0).long()
    return TensorDataset(X, y)


def _make_loader(dataset, batch_size: int = 64) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _make_model(dim: int = 16, hidden: int = 32, n_classes: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes),
    )


# ── ThermalImportance tests ───────────────────────────────────────────────────

class TestThermalImportance:

    def test_accumulate_updates_state(self):
        model = _make_model()
        imp = ThermalImportance(model, ema_beta=0.99)
        x = torch.randn(8, 16)
        y = torch.randint(0, 2, (8,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        imp.accumulate(model)
        assert imp.steps_accumulated == 1

    def test_accumulate_zero_without_backward(self):
        model = _make_model()
        imp = ThermalImportance(model, ema_beta=0.99)
        # Don't call backward — grads are None
        imp.accumulate(model)
        # Should not error; _v stays zero
        assert imp.steps_accumulated == 1
        for v in imp._v.values():
            assert v.abs().max().item() == 0.0

    def test_finalize_returns_correct_shapes(self):
        model = _make_model()
        imp = ThermalImportance(model)
        x = torch.randn(8, 16)
        y = torch.randint(0, 2, (8,))
        nn.CrossEntropyLoss()(model(x), y).backward()
        imp.accumulate(model)
        result = imp.finalize()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in result
                assert result[name].shape == p.shape

    def test_finalize_normalized_range(self):
        model = _make_model()
        imp = ThermalImportance(model)
        for _ in range(20):
            x = torch.randn(8, 16)
            y = torch.randint(0, 2, (8,))
            nn.CrossEntropyLoss()(model(x), y).backward()
            imp.accumulate(model)
        result = imp.finalize(normalize=True)
        for name, v in result.items():
            assert v.min().item() >= 0.0
            # Max should be at most 1.0 (may be exactly 1.0 for the active param)
            assert v.max().item() <= 1.0 + 1e-6, f"{name}: max={v.max().item()}"

    def test_finalize_unnormalized_nonnegative(self):
        model = _make_model()
        imp = ThermalImportance(model)
        x = torch.randn(8, 16)
        y = torch.randint(0, 2, (8,))
        nn.CrossEntropyLoss()(model(x), y).backward()
        imp.accumulate(model)
        result = imp.finalize(normalize=False)
        for v in result.values():
            assert v.min().item() >= 0.0

    def test_is_ready(self):
        model = _make_model()
        imp = ThermalImportance(model, min_steps=5)
        assert not imp.is_ready()
        for _ in range(5):
            imp.accumulate(model)  # no grad, but steps still counted
        assert imp.is_ready()


# ── ThermalMemory tests ───────────────────────────────────────────────────────

class TestThermalMemory:

    def _commit_task(self, model, memory, task_id, steps=20):
        imp = ThermalImportance(model)
        x = torch.randn(8, 16)
        y = torch.randint(0, 2, (8,))
        for _ in range(steps):
            nn.CrossEntropyLoss()(model(x), y).backward()
            imp.accumulate(model)
            model.zero_grad()
        memory.commit(model, imp, task_id=task_id)

    def test_commit_adds_checkpoint(self):
        model = _make_model()
        mem = ThermalMemory(max_tasks=5)
        self._commit_task(model, mem, task_id=0)
        assert mem.num_tasks == 1
        assert mem.task_ids() == [0]

    def test_max_tasks_evicts_oldest(self):
        model = _make_model()
        mem = ThermalMemory(max_tasks=3)
        for i in range(5):
            self._commit_task(model, mem, task_id=i)
        assert mem.num_tasks == 3
        assert mem.task_ids() == [2, 3, 4]

    def test_penalty_zero_with_no_tasks(self):
        model = _make_model()
        mem = ThermalMemory()
        pen = mem.penalty(model)
        assert float(pen.item()) == 0.0

    def test_penalty_positive_after_weight_change(self):
        model = _make_model()
        mem = ThermalMemory()
        self._commit_task(model, mem, task_id=0)

        # Perturb model weights significantly
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 5.0)

        pen = mem.penalty(model)
        assert float(pen.item()) > 0.0

    def test_penalty_zero_when_weights_unchanged(self):
        model = _make_model()
        mem = ThermalMemory()
        self._commit_task(model, mem, task_id=0)

        # Do NOT change weights
        pen = mem.penalty(model)
        assert float(pen.item()) < 1e-6

    def test_anneal_decreases_penalty(self):
        model = _make_model()
        mem = ThermalMemory(anneal_rate=0.9)
        self._commit_task(model, mem, task_id=0)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 2.0)

        pen_before = float(mem.penalty(model).item())

        # Advance annealing 50 steps
        for _ in range(50):
            mem.anneal_all()

        pen_after = float(mem.penalty(model).item())
        assert pen_after < pen_before, (
            f"Expected penalty to decrease after annealing. "
            f"before={pen_before:.4f} after={pen_after:.4f}"
        )

    def test_anneal_rate_1_no_change(self):
        model = _make_model()
        mem = ThermalMemory(anneal_rate=1.0)
        self._commit_task(model, mem, task_id=0)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 2.0)

        pen_before = float(mem.penalty(model).item())
        for _ in range(100):
            mem.anneal_all()
        pen_after = float(mem.penalty(model).item())
        assert abs(pen_before - pen_after) < 1e-6

    def test_thermal_profile_structure(self):
        model = _make_model()
        mem = ThermalMemory()
        self._commit_task(model, mem, task_id=7)
        profile = mem.thermal_profile()
        assert len(profile) == 1
        assert profile[0]['task_id'] == 7
        assert 'layers' in profile[0]
        for name, stats in profile[0]['layers'].items():
            assert 'mean' in stats
            assert 'hot_fraction' in stats


# ── TCLRegularizer tests ──────────────────────────────────────────────────────

class TestTCLRegularizer:

    def test_penalty_scales_with_lambda(self):
        model = _make_model()
        mem = ThermalMemory()

        imp = ThermalImportance(model)
        x = torch.randn(8, 16)
        y = torch.randint(0, 2, (8,))
        for _ in range(10):
            nn.CrossEntropyLoss()(model(x), y).backward()
            imp.accumulate(model)
            model.zero_grad()
        mem.commit(model, imp, task_id=0)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 2.0)

        reg1 = TCLRegularizer(mem, lambda_tcl=1.0)
        reg2 = TCLRegularizer(mem, lambda_tcl=5.0)
        p1 = float(reg1.penalty(model).item())
        p2 = float(reg2.penalty(model).item())
        assert abs(p2 - 5.0 * p1) < 1e-4 * abs(p1)

    def test_penalty_differentiable(self):
        model = _make_model()
        mem = ThermalMemory()

        imp = ThermalImportance(model)
        x = torch.randn(8, 16)
        y = torch.randint(0, 2, (8,))
        for _ in range(10):
            nn.CrossEntropyLoss()(model(x), y).backward()
            imp.accumulate(model)
            model.zero_grad()
        mem.commit(model, imp, task_id=0)

        # Perturb weights then check penalty has grad
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        reg = TCLRegularizer(mem, lambda_tcl=1.0)
        pen = reg.penalty(model)
        pen.backward()

        # At least one parameter should have gradient from penalty
        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 0.0
            for p in model.parameters()
        )
        assert has_grad, "TCL penalty produced no gradients"


# ── Forgetting benchmark ──────────────────────────────────────────────────────

class TestForgettingBenchmark:
    """
    Core functional test: TCL should reduce forgetting vs no regularizer.

    Task 0: classify by x[0] > 0
    Task 1: classify by x[-1] > 0

    After training on Task 1, evaluate Task 0 accuracy:
      no_reg_acc  = accuracy with plain AdamW (catastrophic forgetting expected)
      tcl_acc     = accuracy with TCL elastic penalty

    We expect tcl_acc > no_reg_acc.
    This is a probabilistic test (synthetic data, small model) — we use a
    meaningful margin (tcl must be at least 5% better) and multiple seeds.
    """

    @staticmethod
    def _train_two_tasks(use_tcl: bool, seed: int = 42, dim: int = 16, epochs: int = 10) -> float:
        torch.manual_seed(seed)
        model = _make_model(dim=dim, hidden=64)
        criterion = nn.CrossEntropyLoss()

        task0_train = _make_task_data(n=400, dim=dim, feature_idx=0, seed=seed)
        task0_val = _make_task_data(n=200, dim=dim, feature_idx=0, seed=seed + 100)
        task1_train = _make_task_data(n=400, dim=dim, feature_idx=dim - 1, seed=seed + 200)

        loader0 = _make_loader(task0_train)
        loader0_val = _make_loader(task0_val, batch_size=200)
        loader1 = _make_loader(task1_train)

        # --- Train on Task 0 ---
        opt0 = torch.optim.Adam(model.parameters(), lr=1e-2)
        imp0 = ThermalImportance(model, ema_beta=0.99)

        for _ in range(epochs):
            for x, y in loader0:
                opt0.zero_grad()
                nn.CrossEntropyLoss()(model(x), y).backward()
                imp0.accumulate(model)
                opt0.step()

        # Record task0 peak accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader0_val:
                preds = model(x).argmax(-1)
                correct += (preds == y).sum().item()
                total += len(y)
        peak_acc_0 = correct / total
        model.train()

        # --- Commit task 0 if using TCL ---
        memory = ThermalMemory(max_tasks=5, anneal_rate=1.0)
        if use_tcl:
            memory.commit(model, imp0, task_id=0)

        regularizer = TCLRegularizer(memory, lambda_tcl=10.0)

        # --- Train on Task 1 ---
        opt1 = torch.optim.Adam(model.parameters(), lr=1e-2)
        imp1 = ThermalImportance(model, ema_beta=0.99)

        for _ in range(epochs):
            for x, y in loader1:
                opt1.zero_grad()
                loss1 = criterion(model(x), y)
                loss1.backward()
                imp1.accumulate(model)

                if use_tcl:
                    reg = regularizer.penalty(model)
                    reg.backward()

                opt1.step()

        # --- Evaluate Task 0 after Task 1 training ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader0_val:
                preds = model(x).argmax(-1)
                correct += (preds == y).sum().item()
                total += len(y)
        final_acc_0 = correct / total

        return final_acc_0

    def test_tcl_reduces_forgetting(self):
        """TCL should recover more of task 0 accuracy than no regularizer."""
        seeds = [42, 137, 2024]
        margins = []

        for seed in seeds:
            no_reg_acc = self._train_two_tasks(use_tcl=False, seed=seed)
            tcl_acc = self._train_two_tasks(use_tcl=True, seed=seed)
            margins.append(tcl_acc - no_reg_acc)

        avg_margin = sum(margins) / len(margins)
        assert avg_margin > 0.05, (
            f"TCL failed to reduce forgetting. "
            f"Per-seed margins: {[f'{m:.3f}' for m in margins]}, "
            f"avg={avg_margin:.3f} (expected > 0.05)"
        )

    def test_no_forgetting_single_task(self):
        """With only one task and no new training, penalty should remain very small."""
        model = _make_model()
        mem = ThermalMemory()
        imp = ThermalImportance(model, ema_beta=0.99)
        loader = _make_loader(_make_task_data())

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for x, y in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            imp.accumulate(model)
            opt.step()

        # Commit, then immediately check penalty (weights haven't changed)
        mem.commit(model, imp, task_id=0)
        reg = TCLRegularizer(mem, lambda_tcl=1.0)
        pen = float(reg.penalty(model).item())
        assert pen < 1e-4, f"Penalty should be near 0 right after commit, got {pen:.6f}"


# ── TCLTrainer smoke test ─────────────────────────────────────────────────────

class TestTCLTrainer:

    def test_learn_two_tasks_runs(self):
        torch.manual_seed(0)
        model = _make_model()

        trainer = TCLTrainer(
            model=model,
            optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-2),
            lambda_tcl=1.0,
        )
        criterion = nn.CrossEntropyLoss()

        loader0 = _make_loader(_make_task_data(n=80, feature_idx=0))
        loader1 = _make_loader(_make_task_data(n=80, feature_idx=15))
        loader0_val = _make_loader(_make_task_data(n=40, feature_idx=0, seed=99))

        h0 = trainer.learn_task(0, loader0, criterion, epochs=2, verbose=False)
        acc0, _ = trainer.evaluate(loader0_val, criterion)
        trainer.record_peak_acc(0, acc0)

        h1 = trainer.learn_task(1, loader1, criterion, epochs=2, verbose=False)

        assert len(h0) > 0
        assert len(h1) > 0
        assert trainer.memory.num_tasks == 2
        assert trainer.memory.task_ids() == [0, 1]

    def test_history_has_reg_loss_after_first_task(self):
        torch.manual_seed(1)
        model = _make_model()

        trainer = TCLTrainer(
            model=model,
            optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-2),
            lambda_tcl=5.0,
        )
        criterion = nn.CrossEntropyLoss()

        loader0 = _make_loader(_make_task_data(n=80, feature_idx=0))
        loader1 = _make_loader(_make_task_data(n=80, feature_idx=15))

        trainer.learn_task(0, loader0, criterion, epochs=1, verbose=False)
        h1 = trainer.learn_task(1, loader1, criterion, epochs=1, verbose=False)

        # Task 1 history should have nonzero reg_loss (protecting task 0)
        reg_losses = [h['reg_loss'] for h in h1]
        assert any(r > 0.0 for r in reg_losses), (
            "Expected nonzero reg_loss during task 1 training"
        )

    def test_evaluate_forgetting_structure(self):
        torch.manual_seed(2)
        model = _make_model()

        trainer = TCLTrainer(
            model=model,
            optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-2),
            lambda_tcl=1.0,
        )
        criterion = nn.CrossEntropyLoss()

        loader0 = _make_loader(_make_task_data(n=80, feature_idx=0))
        loader1 = _make_loader(_make_task_data(n=80, feature_idx=15))
        val0 = _make_loader(_make_task_data(n=40, feature_idx=0, seed=99))
        val1 = _make_loader(_make_task_data(n=40, feature_idx=15, seed=100))

        trainer.learn_task(0, loader0, criterion, epochs=2, verbose=False)
        acc0, _ = trainer.evaluate(val0, criterion)
        trainer.record_peak_acc(0, acc0)

        trainer.learn_task(1, loader1, criterion, epochs=2, verbose=False)
        acc1, _ = trainer.evaluate(val1, criterion)
        trainer.record_peak_acc(1, acc1)

        result = trainer.evaluate_forgetting({0: val0, 1: val1}, criterion)

        assert 'per_task_acc' in result
        assert 'per_task_forgetting' in result
        assert 'avg_forgetting' in result
        assert 0 in result['per_task_acc']
        assert 1 in result['per_task_acc']
        assert result['avg_forgetting'] >= 0.0
