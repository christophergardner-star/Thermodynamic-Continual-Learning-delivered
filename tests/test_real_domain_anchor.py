from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from pydantic import ValidationError

from tar_lab.multimodal_payloads import run_split_cifar10_benchmark
from tar_lab.schemas import (
    ContinualLearningBenchmarkConfig,
    ContinualLearningBenchmarkResult,
    ContinualLearningMetrics,
)
from tar_lab.thermoobserver import ActivationThermoObserver


def test_cl_metrics_schema_valid():
    metrics = ContinualLearningMetrics(
        task_id=0,
        task_accuracy=0.8,
        accuracy_right_after_training=0.85,
        backward_transfer=-0.05,
        forgetting_measure=0.05,
        forward_transfer=0.0,
        stability_plasticity_gap=0.0588235294,
    )
    assert isinstance(metrics.backward_transfer, float)
    with pytest.raises(ValidationError):
        ContinualLearningMetrics(
            task_id=0,
            task_accuracy=0.8,
            accuracy_right_after_training=0.85,
            backward_transfer=-0.05,
            forgetting_measure=0.05,
            forward_transfer=0.0,
            stability_plasticity_gap=0.0588235294,
            unknown="x",
        )


def test_cl_benchmark_result_schema_valid():
    result = ContinualLearningBenchmarkResult(
        benchmark_id="bench-1",
        method="sgd_baseline",
        seed=42,
        per_task_metrics=[],
        mean_backward_transfer=0.0,
        mean_forgetting=0.0,
        final_mean_accuracy=0.0,
        last_task_accuracy=0.0,
    )
    assert result.thermodynamic_trace_path == ""
    with pytest.raises(ValidationError):
        ContinualLearningBenchmarkResult(
            benchmark_id="bench-1",
            method="sgd_baseline",
            seed=42,
            per_task_metrics=[],
            mean_backward_transfer=0.0,
            mean_forgetting=0.0,
            final_mean_accuracy=0.0,
            last_task_accuracy=0.0,
            unknown="x",
        )


def test_cl_benchmark_config_schema_valid():
    config = ContinualLearningBenchmarkConfig()
    assert config.class_order == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert config.setting == "task_incremental"
    assert config.tcl_alpha == 0.5
    assert config.tcl_ordered_lr_scale == 0.5
    assert config.tcl_disordered_lr_scale == 1.2
    assert config.tcl_reset_on_task_boundary is True


def test_thermoobserver_current_regime_unknown_before_step():
    observer = ActivationThermoObserver(nn.Linear(4, 2))
    assert observer.current_regime == "unknown"


def test_thermoobserver_current_regime_after_step():
    model = nn.Linear(4, 2)
    observer = ActivationThermoObserver(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    observer.step(optimizer)
    assert observer.current_regime in ("ordered", "critical", "disordered", "unknown")


@pytest.mark.slow
def test_split_cifar10_sgd_runs():
    try:
        import torchvision  # noqa: F401
    except ImportError:
        pytest.skip("torchvision not installed")

    config = ContinualLearningBenchmarkConfig(
        n_tasks=2,
        train_epochs_per_task=1,
        class_order=[[0, 1], [2, 3]],
    )
    result = run_split_cifar10_benchmark(config, method="sgd_baseline", workspace=None)
    assert result.n_tasks == 2
    assert len(result.per_task_metrics) == 2
    assert result.mean_forgetting >= 0.0
    assert 0.0 <= result.final_mean_accuracy <= 1.0


@pytest.mark.slow
def test_split_cifar10_class_incremental_sgd_runs():
    try:
        import torchvision  # noqa: F401
    except ImportError:
        pytest.skip("torchvision not installed")

    config = ContinualLearningBenchmarkConfig(
        setting="class_incremental",
        n_tasks=2,
        train_epochs_per_task=1,
        class_order=[[0, 1], [2, 3]],
    )
    result = run_split_cifar10_benchmark(config, method="sgd_baseline", workspace=None)
    assert result.n_tasks == 2
    assert len(result.per_task_metrics) == 2
    assert result.mean_forgetting >= 0.0
    assert 0.0 <= result.final_mean_accuracy <= 1.0
