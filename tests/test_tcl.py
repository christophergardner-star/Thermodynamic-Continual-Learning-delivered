"""Tests for epto_sdk_v54.tcl — 21 tests, all pass in ~3 s CPU."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from epto_sdk_v54.tcl import (
    TaskSnapshot,
    TCLEngine,
    TCLRegularizer,
    ThermalImportance,
    ThermalMemory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear(in_f: int = 4, out_f: int = 2, bias: bool = False) -> nn.Linear:
    """Small deterministic linear model for unit tests."""
    torch.manual_seed(0)
    return nn.Linear(in_f, out_f, bias=bias)


def _do_backward(model: nn.Module, seed: int = 7) -> None:
    """Run one forward/backward pass to populate ``.grad`` on all parameters."""
    torch.manual_seed(seed)
    x = torch.randn(3, next(model.parameters()).shape[-1])
    loss = model(x).sum()
    loss.backward()


def _commit_one_snapshot(
    model: nn.Module,
    alpha: float = 1.0,
    seed: int = 7,
) -> tuple[ThermalImportance, ThermalMemory, TaskSnapshot]:
    """Create a ThermalImportance + ThermalMemory, do one update, commit."""
    imp = ThermalImportance(model, alpha=alpha)
    mem = ThermalMemory(model, imp)
    _do_backward(model, seed=seed)
    imp.update()
    snap = mem.commit()
    return imp, mem, snap


# ---------------------------------------------------------------------------
# TaskSnapshot
# ---------------------------------------------------------------------------


def test_task_snapshot_stores_params():
    params = {"w": torch.tensor([1.0, 2.0])}
    imp = {"w": torch.tensor([0.5, 0.5])}
    snap = TaskSnapshot(params=params, importance=imp, task_id=3)
    assert snap.params is params
    assert snap.task_id == 3


def test_task_snapshot_default_task_id():
    snap = TaskSnapshot(params={}, importance={})
    assert snap.task_id == 0


# ---------------------------------------------------------------------------
# ThermalImportance
# ---------------------------------------------------------------------------


def test_thermal_importance_alpha_validation():
    model = _linear()
    with pytest.raises(ValueError, match="alpha"):
        ThermalImportance(model, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ThermalImportance(model, alpha=1.5)


def test_thermal_importance_reset_clears_state():
    model = _linear()
    imp = ThermalImportance(model, alpha=0.5)
    _do_backward(model)
    imp.update()
    assert imp.n_updates == 1
    imp.reset()
    assert imp.n_updates == 0
    assert imp.get() == {}


def test_thermal_importance_update_skips_none_grad():
    model = _linear()
    imp = ThermalImportance(model, alpha=1.0)
    # No backward pass — grads are None
    imp.update()
    assert imp.get() == {}
    assert imp.n_updates == 1  # counter still increments


def test_thermal_importance_ema_computation():
    """With alpha=1, EMA reduces to the last gradient squared."""
    model = _linear(in_f=2, out_f=1, bias=False)
    # Manually set weights to known values
    with torch.no_grad():
        model.weight.fill_(1.0)
    imp = ThermalImportance(model, alpha=1.0)

    x = torch.ones(1, 2)
    loss = model(x).sum()
    loss.backward()
    imp.update()

    ema = imp.get()
    expected_g_sq = model.weight.grad.pow(2)
    assert torch.allclose(ema["weight"], expected_g_sq)


def test_thermal_importance_n_updates_counter():
    model = _linear()
    imp = ThermalImportance(model, alpha=0.1)
    for k in range(5):
        _do_backward(model, seed=k)
        imp.update()
    assert imp.n_updates == 5


def test_thermal_importance_get_returns_independent_copy():
    model = _linear()
    imp = ThermalImportance(model, alpha=1.0)
    _do_backward(model)
    imp.update()

    copy1 = imp.get()
    copy2 = imp.get()
    # Modifying copy1 must not affect copy2 or the internal state
    for v in copy1.values():
        v.fill_(999.0)
    for k in copy2:
        assert not torch.all(copy2[k] == 999.0)


# ---------------------------------------------------------------------------
# ThermalMemory
# ---------------------------------------------------------------------------


def test_thermal_memory_commit_creates_snapshot():
    model = _linear()
    _, mem, snap = _commit_one_snapshot(model)

    assert mem.num_tasks == 1
    assert isinstance(snap, TaskSnapshot)
    assert snap.task_id == 0
    assert "weight" in snap.params
    assert "weight" in snap.importance


def test_thermal_memory_commit_resets_tracker():
    model = _linear()
    imp, mem, _ = _commit_one_snapshot(model)
    # After commit, tracker should be reset
    assert imp.n_updates == 0
    assert imp.get() == {}


def test_thermal_memory_num_tasks_increments():
    model = _linear()
    imp = ThermalImportance(model, alpha=1.0)
    mem = ThermalMemory(model, imp)

    assert mem.num_tasks == 0
    for k in range(3):
        _do_backward(model, seed=k)
        imp.update()
        mem.commit()
    assert mem.num_tasks == 3


def test_thermal_memory_snapshots_property_is_copy():
    """Mutating the returned list must not affect internal storage."""
    model = _linear()
    _, mem, _ = _commit_one_snapshot(model)

    snaps = mem.snapshots
    snaps.clear()
    assert mem.num_tasks == 1  # internal list unchanged


# ---------------------------------------------------------------------------
# TCLRegularizer
# ---------------------------------------------------------------------------


def test_regularizer_init_validation():
    model = _linear()
    imp = ThermalImportance(model)
    mem = ThermalMemory(model, imp)
    with pytest.raises(ValueError, match="lambda_reg"):
        TCLRegularizer(model, mem, lambda_reg=-1.0)
    with pytest.raises(ValueError, match="decay"):
        TCLRegularizer(model, mem, decay=0.0)
    with pytest.raises(ValueError, match="anneal_rate"):
        TCLRegularizer(model, mem, anneal_rate=1.5)


def test_regularizer_penalty_zero_before_any_commit():
    model = _linear()
    imp = ThermalImportance(model)
    mem = ThermalMemory(model, imp)
    reg = TCLRegularizer(model, mem, lambda_reg=10.0)
    assert reg.penalty().item() == pytest.approx(0.0)


def test_regularizer_penalty_zero_when_params_unchanged():
    """Penalty must be zero when current params match the snapshot."""
    model = _linear()
    _, mem, _ = _commit_one_snapshot(model)
    reg = TCLRegularizer(model, mem, lambda_reg=100.0)
    assert reg.penalty().item() == pytest.approx(0.0, abs=1e-6)


def test_regularizer_penalty_positive_when_params_changed():
    model = _linear()
    _, mem, snap = _commit_one_snapshot(model)

    # Perturb weights away from the snapshot
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 2.0)

    reg = TCLRegularizer(model, mem, lambda_reg=1.0)
    assert reg.penalty().item() > 0.0


def test_regularizer_lambda_scales_penalty():
    model = _linear()
    _, mem, _ = _commit_one_snapshot(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    reg1 = TCLRegularizer(model, mem, lambda_reg=1.0)
    reg2 = TCLRegularizer(model, mem, lambda_reg=5.0)
    assert reg2.penalty().item() == pytest.approx(5.0 * reg1.penalty().item(), rel=1e-5)


def test_regularizer_decay_weights_tasks():
    """With decay=0.5, adding a second (older) task should change the penalty."""
    model = _linear()
    imp = ThermalImportance(model, alpha=1.0)
    mem = ThermalMemory(model, imp)

    # Commit two snapshots with same params and same importance
    for k in range(2):
        _do_backward(model, seed=k)
        imp.update()
        mem.commit()

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    reg_no_decay = TCLRegularizer(model, mem, lambda_reg=1.0, decay=1.0)
    reg_decay = TCLRegularizer(model, mem, lambda_reg=1.0, decay=0.5)
    # With decay<1, total penalty is smaller (older task downweighted)
    assert reg_decay.penalty().item() < reg_no_decay.penalty().item()


def test_regularizer_annealing_reduces_penalty():
    """With anneal_rate=0.5 and 10 steps, penalty should be < no-anneal case."""
    model = _linear()
    _, mem, _ = _commit_one_snapshot(model)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    reg_full = TCLRegularizer(model, mem, lambda_reg=1.0, anneal_rate=1.0)
    reg_anneal = TCLRegularizer(model, mem, lambda_reg=1.0, anneal_rate=0.5)
    for _ in range(10):
        reg_anneal.step()

    assert reg_anneal.penalty().item() < reg_full.penalty().item()


# ---------------------------------------------------------------------------
# TCLEngine
# ---------------------------------------------------------------------------


def test_engine_orchestration():
    """TCLEngine wires all components and end_task commits + resets step."""
    torch.manual_seed(0)
    model = _linear(in_f=4, out_f=2)
    opt = optim.SGD(model.parameters(), lr=0.01)
    engine = TCLEngine(model, alpha=0.5, lambda_reg=1.0)

    assert engine.num_tasks == 0

    # Task 0: train a few steps (no regularization needed yet)
    for _ in range(5):
        x = torch.randn(4, 4)
        loss = model(x).sum()
        engine.backward_and_update(loss, opt)

    assert engine.regularizer.current_step == 5
    snap = engine.end_task()
    assert engine.num_tasks == 1
    assert snap.task_id == 0
    assert engine.regularizer.current_step == 0  # reset after end_task

    # Task 1: regularized loss should be larger than plain loss
    x = torch.randn(4, 4)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(3.0)  # move params far from snapshot
    base_loss = model(x).sum()
    reg_loss = engine.regularized_loss(base_loss)
    assert reg_loss.item() > base_loss.item()


# ---------------------------------------------------------------------------
# Functional benchmark
# ---------------------------------------------------------------------------


def _run_benchmark(seed: int, use_tcl: bool, lambda_reg: float = 5000.0) -> float:
    """Return Task-0 accuracy after sequential 2-task training.

    Task 0: binary label = sign of first feature (X[:,0] > 0).
    Task 1: opposite binary label  ← catastrophic conflict.

    The model is a simple linear classifier trained with plain SGD.  Task 0
    training stops before full convergence so that the EMA gradient-energy
    importance values remain non-trivially large — matching the expected
    operating regime of :class:`ThermalImportance`.
    """
    torch.manual_seed(seed)

    n_train, n_test, d = 200, 100, 10
    X = torch.randn(n_train + n_test, d)
    y0 = (X[:, 0] > 0).float()
    y1 = (X[:, 0] <= 0).float()  # flipped labels = direct conflict

    X_train, X_test = X[:n_train], X[n_train:]
    y0_train, y0_test = y0[:n_train], y0[n_train:]
    y1_train = y1[:n_train]

    model = nn.Linear(d, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.BCEWithLogitsLoss()

    engine = TCLEngine(model, alpha=0.5, lambda_reg=lambda_reg) if use_tcl else None

    def _train(X_t, y_t, steps: int, regularize: bool = False) -> None:
        n = X_t.shape[0]
        for _ in range(steps):
            idx = torch.randint(n, (32,))
            logits = model(X_t[idx]).squeeze()
            base = criterion(logits, y_t[idx])
            loss = engine.regularized_loss(base) if (regularize and engine) else base
            if engine:
                engine.backward_and_update(loss, optimizer)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Task 0: train for 30 steps (partial convergence keeps importance non-trivial)
    _train(X_train, y0_train, steps=30, regularize=False)
    if engine:
        engine.end_task()

    # Task 1 (conflicting): regularization must resist catastrophic forgetting
    _train(X_train, y1_train, steps=100, regularize=True)

    # Evaluate Task 0 retention
    with torch.no_grad():
        preds = (model(X_test).squeeze() > 0).float()
        acc = (preds == y0_test).float().mean().item()
    return acc


def test_functional_two_task_benchmark():
    """TCL must recover >5 pp more Task-0 accuracy than no-regularizer (3 seeds)."""
    gains = []
    for seed in range(3):
        acc_base = _run_benchmark(seed, use_tcl=False)
        acc_tcl = _run_benchmark(seed, use_tcl=True)
        gains.append(acc_tcl - acc_base)

    mean_gain = sum(gains) / len(gains)
    assert mean_gain > 0.05, (
        f"Expected mean TCL gain > 5 pp, got {mean_gain:.1%} "
        f"(per-seed gains: {[f'{g:.1%}' for g in gains]})"
    )
