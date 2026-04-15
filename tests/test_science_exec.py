from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tar_lab.science_exec as science_exec
from tar_lab.science_exec import execute_study_payload
from tar_lab.science_profiles import ScienceProfileRegistry


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_benchmark_executes_on_laptop_safe_path(tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("computer_vision")
    spec = registry.resolve_benchmark_suite(profile, "corruption_robustness", tier="smoke")
    availability = registry.benchmark_availability(spec)
    payload = {
        "problem_id": "cv-smoke",
        "problem": "Investigate corruption robustness in computer vision",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "smoke",
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "augmentation_robustness",
                "name": "Augmentation Robustness",
                "benchmark": "corruption_robustness",
                "benchmark_tier": "smoke",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["top1_accuracy", "corruption_robustness", "calibration_ece"],
                "parameter_grid": {"augmentation": ["baseline"], "severity": [0]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "cv_smoke.json")
    assert report.status == "completed"
    assert report.experiments[0].benchmark_tier == "smoke"
    assert report.experiments[0].benchmark_truth_status == "smoke_only"
    assert report.experiments[0].benchmark_alignment == "aligned"
    assert report.experiments[0].proxy_benchmark_used
    assert report.benchmark_names == [spec.name]


def test_non_proxy_validation_benchmark_refuses_missing_dataset(tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("generic_ml")
    spec = registry.resolve_benchmark_suite(profile, "cross_validation", tier="validation")
    availability = registry.benchmark_availability(spec).model_copy(
        update={"dataset_ready": False, "reason": "dataset cache missing"}
    )
    payload = {
        "problem_id": "generic-validation",
        "problem": "Investigate calibration in generic ML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "validation",
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "baseline_sweep",
                "name": "Baseline Sweep",
                "benchmark": "cross_validation",
                "benchmark_tier": "validation",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["accuracy", "f1", "seed_variance"],
                "parameter_grid": {"learning_rate": [0.001]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "generic_validation_refusal.json")
    assert report.status == "failed"
    assert report.experiments[0].execution_mode == "benchmark_unavailable"
    assert report.experiments[0].benchmark_truth_status == "validation_only"
    assert not report.experiments[0].provenance_complete


def test_canonical_comparability_stays_false_on_refusal(tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("graph_ml")
    spec = registry.resolve_benchmark_suite(profile, "depth_sweep", tier="canonical")
    assert spec.truth_status == "unsupported"
    availability = registry.benchmark_availability(spec)
    payload = {
        "problem_id": "graph-canonical",
        "problem": "Investigate depth behavior in graph ML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "canonical_only": True,
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "depth_oversmoothing",
                "name": "Depth and Oversmoothing",
                "benchmark": "depth_sweep",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["node_accuracy", "oversmoothing_gap", "representation_dimensionality"],
                "parameter_grid": {"layers": [2, 4]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "graph_canonical.json")
    assert not report.canonical_comparable
    assert report.status == "failed"
    assert report.benchmark_alignment == "refused"
    assert report.experiments[0].execution_mode == "truth_refusal"


def test_qml_canonical_depth_and_init_use_real_backend_when_available(monkeypatch, tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("quantum_ml")
    depth_spec = registry.resolve_benchmark_suite(profile, "depth_trainability_curve", tier="canonical")
    init_spec = registry.resolve_benchmark_suite(profile, "initialization_trainability", tier="canonical")
    depth_availability = registry.benchmark_availability(depth_spec)
    init_availability = registry.benchmark_availability(init_spec)

    monkeypatch.setattr(
        science_exec,
        "_load_quantum_runtime",
        lambda: {
            "qml": object(),
            "pnp": object(),
            "pennylane_ready": True,
            "noise_ready": True,
            "reason": "ready",
            "noise_reason": "ready",
        },
    )
    monkeypatch.setattr(
        science_exec,
        "_run_quantum_with_pennylane",
        lambda qml, pnp, experiment: (
            {"gradient_norm_variance": 0.25, "seed_variance": 0.02, "barren_plateau_slope": -0.1, "trainability_gap": 1.8},
            "backend=pennylane:default.qubit qubits=[4] ansatz_depth=[4] init_scale=[0.05] seeds=[7, 18].",
        ),
    )

    payload = {
        "problem_id": "qml-canonical-real",
        "problem": "Investigate barren plateaus in QML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "environment": {"validation_imports": []},
        "benchmark_availability": [
            depth_availability.model_dump(mode="json"),
            init_availability.model_dump(mode="json"),
        ],
        "experiments": [
            {
                "template_id": "ansatz_depth_sweep",
                "name": "Ansatz Depth Sweep",
                "benchmark": "depth_trainability_curve",
                "benchmark_tier": "canonical",
                "benchmark_spec": depth_spec.model_dump(mode="json"),
                "benchmark_availability": depth_availability.model_dump(mode="json"),
                "metrics": ["gradient_norm_variance", "barren_plateau_slope", "trainability_gap"],
                "parameter_grid": {},
                "success_criteria": [],
            },
            {
                "template_id": "initialization_variance",
                "name": "Initialization Variance Study",
                "benchmark": "initialization_trainability",
                "benchmark_tier": "canonical",
                "benchmark_spec": init_spec.model_dump(mode="json"),
                "benchmark_availability": init_availability.model_dump(mode="json"),
                "metrics": ["gradient_norm_variance", "seed_variance"],
                "parameter_grid": {},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "qml_canonical_real.json")
    assert report.status == "completed"
    assert report.canonical_comparable
    assert all(item.execution_mode == "pennylane_backend" for item in report.experiments)
    assert all(not item.proxy_benchmark_used for item in report.experiments)
    assert all(item.benchmark_truth_status == "canonical_ready" for item in report.experiments)
    assert all(item.execution_mode != "analytic_proxy" for item in report.experiments)


def test_qml_canonical_statistical_summary_can_be_ready_with_five_seed_evidence(monkeypatch, tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("quantum_ml")
    spec = registry.resolve_benchmark_suite(profile, "depth_trainability_curve", tier="canonical")
    availability = registry.benchmark_availability(spec)

    monkeypatch.setattr(
        science_exec,
        "_load_quantum_runtime",
        lambda: {
            "qml": object(),
            "pnp": object(),
            "pennylane_ready": True,
            "noise_ready": True,
            "reason": "ready",
            "noise_reason": "ready",
        },
    )
    monkeypatch.setattr(
        science_exec,
        "_run_quantum_with_pennylane",
        lambda qml, pnp, experiment: (
            {"gradient_norm_variance": 0.22, "barren_plateau_slope": -0.09, "trainability_gap": 1.95},
            "backend=pennylane:default.qubit qubits=[4] ansatz_depth=[2, 4, 8] init_scale=0.05 seeds=[7, 18, 29, 41, 53].",
            {"sample_count": 5, "primary_metric": "trainability_gap", "std_dev": 0.08},
        ),
    )

    payload = {
        "problem_id": "qml-canonical-stats-ready",
        "problem": "Validate canonical QML statistical readiness",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "requested_benchmark": spec.benchmark_id,
        "canonical_only": True,
        "no_proxy_benchmarks": True,
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "ansatz_depth_sweep",
                "name": "Ansatz Depth Sweep",
                "benchmark": "depth_trainability_curve",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["gradient_norm_variance", "barren_plateau_slope", "trainability_gap"],
                "parameter_grid": {"seed": [7, 18, 29, 41, 53]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "qml_canonical_stats_ready.json")
    assert report.status == "completed"
    assert report.canonical_comparable
    assert report.benchmark_statistical_summary is not None
    assert report.benchmark_statistical_summary.statistically_ready
    summary = report.experiments[0].statistical_summary
    assert summary is not None
    assert summary.statistically_ready
    assert summary.sample_count == 5
    assert summary.primary_metric == "trainability_gap"
    assert summary.metrics[0].ci95_low is not None
    assert summary.metrics[0].ci95_high is not None


def test_qml_noise_validation_uses_real_backend_when_available(monkeypatch, tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("quantum_ml")
    spec = registry.resolve_benchmark_suite(profile, "noise_trainability_ablation", tier="validation")
    availability = registry.benchmark_availability(spec)

    monkeypatch.setattr(
        science_exec,
        "_load_quantum_runtime",
        lambda: {
            "qml": object(),
            "pnp": object(),
            "pennylane_ready": True,
            "noise_ready": True,
            "reason": "ready",
            "noise_reason": "ready",
        },
    )
    monkeypatch.setattr(
        science_exec,
        "_run_quantum_noise_with_pennylane",
        lambda qml, experiment: (
            {"shot_noise_robustness": 0.71, "gradient_norm_variance": 0.04, "trainability_gap": 1.9},
            "backend=pennylane:default.mixed qubits=3 ansatz_depth=3 init_scale=0.05 shots=[128, 512] noise_level=[0.0, 0.01, 0.05] seeds=[7, 18].",
        ),
    )

    payload = {
        "problem_id": "qml-validation-real",
        "problem": "Investigate shot noise in QML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "validation",
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "noise_shot_ablation",
                "name": "Noise and Shot Ablation",
                "benchmark": "noise_trainability_ablation",
                "benchmark_tier": "validation",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["shot_noise_robustness", "gradient_norm_variance", "trainability_gap"],
                "parameter_grid": {},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "qml_validation_real.json")
    assert spec.benchmark_id == "qml_noise_validation"
    assert report.status == "completed"
    assert not report.canonical_comparable
    assert report.experiments[0].execution_mode == "pennylane_noise_backend"
    assert report.experiments[0].benchmark_truth_status == "validation_only"
    assert not report.experiments[0].proxy_benchmark_used


def test_qml_canonical_noise_refuses_when_noise_backend_missing(monkeypatch, tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("quantum_ml")
    spec = registry.resolve_benchmark_suite(profile, "noise_trainability_ablation", tier="canonical")
    availability = registry.benchmark_availability(spec)

    monkeypatch.setattr(
        science_exec,
        "_load_quantum_runtime",
        lambda: {
            "qml": object(),
            "pnp": object(),
            "pennylane_ready": True,
            "noise_ready": False,
            "reason": "ready",
            "noise_reason": "PennyLane default.mixed backend unavailable: simulated test failure",
        },
    )

    payload = {
        "problem_id": "qml-canonical-refusal",
        "problem": "Investigate shot noise in QML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "canonical_only": True,
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "noise_shot_ablation",
                "name": "Noise and Shot Ablation",
                "benchmark": "noise_trainability_ablation",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["shot_noise_robustness", "gradient_norm_variance", "trainability_gap"],
                "parameter_grid": {},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "qml_canonical_refusal.json")
    assert spec.benchmark_id == "qml_noise_canonical"
    assert spec.truth_status == "canonical_ready"
    assert report.status == "failed"
    assert not report.canonical_comparable
    assert report.experiments[0].execution_mode == "dependency_failure"
    assert report.experiments[0].benchmark_alignment == "refused"
    assert report.experiments[0].execution_mode != "analytic_proxy"


def test_generic_ml_canonical_adult_calibration_uses_real_openml_path(monkeypatch, tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("generic_ml")
    spec = registry.resolve_benchmark_suite(profile, "holdout_calibration", tier="canonical")
    availability = registry.benchmark_availability(spec).model_copy(
        update={"imports_ready": True, "dataset_ready": True, "canonical_ready": True, "reason": None}
    )

    row_count = 3200
    rng = np.random.default_rng(7)
    feature_frame = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=row_count),
            "hours-per-week": rng.integers(20, 60, size=row_count),
            "education-num": rng.integers(8, 16, size=row_count),
            "workclass": rng.choice(["Private", "Self-emp", "Government"], size=row_count),
            "occupation": rng.choice(["Tech", "Sales", "Admin"], size=row_count),
            "sex": rng.choice(["Male", "Female"], size=row_count),
        }
    )
    target = pd.Series(
        np.where(
            (
                (feature_frame["age"].to_numpy() > 40)
                & (feature_frame["hours-per-week"].to_numpy() > 42)
            ),
            ">50K",
            "<=50K",
        ),
        name="class",
    )
    frame = feature_frame.copy()
    frame["class"] = target

    class _FakeOpenMLResult:
        def __init__(self, frame: pd.DataFrame, target: pd.Series) -> None:
            self.frame = frame
            self.target = target

    monkeypatch.setattr(
        science_exec,
        "fetch_openml",
        lambda *args, **kwargs: _FakeOpenMLResult(frame.copy(), target.copy()),
    )

    payload = {
        "problem_id": "generic-canonical-adult",
        "problem": "Investigate calibration in generic ML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "requested_benchmark": spec.benchmark_id,
        "canonical_only": True,
        "no_proxy_benchmarks": True,
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "calibration_check",
                "name": "Calibration Check",
                "benchmark": "holdout_calibration",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["calibration_ece", "accuracy", "auroc"],
                "parameter_grid": {"temperature_scaling": [False, True]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "generic_canonical_adult.json")
    assert report.status == "completed"
    assert report.canonical_comparable
    assert report.benchmark_alignment == "aligned"
    assert report.benchmark_truth_statuses == ["canonical_ready"]
    experiment = report.experiments[0]
    assert experiment.execution_mode == "tabular_benchmark"
    assert not experiment.proxy_benchmark_used
    assert experiment.benchmark_id == "openml_adult_calibration"
    assert experiment.statistical_summary is not None
    assert experiment.statistical_summary.statistically_ready
    assert experiment.statistical_summary.sample_count == 5
    assert experiment.statistical_summary.primary_metric == "calibration_ece"
    assert experiment.statistical_summary.metrics[0].ci95_low is not None
    assert experiment.statistical_summary.metrics[0].ci95_high is not None


def test_graph_ml_canonical_roman_empire_uses_real_dataset_path(monkeypatch, tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("graph_ml")
    spec = registry.resolve_benchmark_suite(profile, "heterophily_control", tier="canonical")
    availability = registry.benchmark_availability(spec).model_copy(
        update={"imports_ready": True, "dataset_ready": True, "canonical_ready": True, "reason": None}
    )

    node_count = 20
    labels = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
    feature_rows = []
    for index in range(node_count):
        if index < 10:
            feature_rows.append([1.0, 0.0, 0.2 * (index % 3), 0.0])
        else:
            feature_rows.append([0.0, 1.0, 0.0, 0.2 * (index % 3)])
    features = torch.tensor(feature_rows, dtype=torch.float32)
    adjacency = torch.eye(node_count, dtype=torch.float32)
    for index in range(node_count):
        for neighbor in ((index - 1) % node_count, (index + 1) % node_count):
            adjacency[index, neighbor] = 1.0
            adjacency[neighbor, index] = 1.0
    adjacency = torch.from_numpy(science_exec._normalize_adjacency(adjacency.numpy()))

    splits = []
    for offset in range(5):
        train_mask = torch.zeros(node_count, dtype=torch.bool)
        val_mask = torch.zeros(node_count, dtype=torch.bool)
        test_mask = torch.zeros(node_count, dtype=torch.bool)
        train_indices = [offset % 10, (offset + 2) % 10, 10 + (offset % 10), 10 + ((offset + 2) % 10)]
        val_indices = [((offset + 4) % 10), 10 + ((offset + 4) % 10)]
        for index in train_indices:
            train_mask[index] = True
        for index in val_indices:
            val_mask[index] = True
        test_mask = ~(train_mask | val_mask)
        splits.append((train_mask, val_mask, test_mask))

    monkeypatch.setattr(
        science_exec,
        "_load_roman_empire_canonical_dataset",
        lambda: (adjacency.clone(), features.clone(), labels.clone(), list(splits)),
    )

    payload = {
        "problem_id": "graph-canonical-roman",
        "problem": "Investigate heterophily in graph ML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "requested_benchmark": spec.benchmark_id,
        "canonical_only": True,
        "no_proxy_benchmarks": True,
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "heterophily_ablation",
                "name": "Heterophily Ablation",
                "benchmark": "heterophily_control",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["node_accuracy", "seed_variance"],
                "parameter_grid": {"rewiring": ["off", "knn"], "normalization": ["batch", "pairnorm"]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "graph_canonical_roman.json")
    assert report.status == "completed"
    assert report.canonical_comparable
    assert report.benchmark_alignment == "aligned"
    assert report.benchmark_truth_statuses == ["canonical_ready"]
    experiment = report.experiments[0]
    assert experiment.execution_mode == "graph_benchmark"
    assert not experiment.proxy_benchmark_used
    assert experiment.benchmark_id == "roman_empire_heterophily_canonical"
    assert experiment.statistical_summary is not None
    assert experiment.statistical_summary.statistically_ready
    assert experiment.statistical_summary.sample_count == 5
    assert experiment.statistical_summary.primary_metric == "node_accuracy"
    assert experiment.statistical_summary.metrics[0].ci95_low is not None
    assert experiment.statistical_summary.metrics[0].ci95_high is not None


def test_computer_vision_canonical_imagenette_uses_real_dataset_path(monkeypatch, tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("computer_vision")
    spec = registry.resolve_benchmark_suite(profile, "transfer_comparison", tier="canonical")
    availability = registry.benchmark_availability(spec).model_copy(
        update={"imports_ready": True, "dataset_ready": True, "canonical_ready": True, "reason": None}
    )

    train_images = torch.rand(120, 3, 32, 32, dtype=torch.float32)
    val_images = torch.rand(60, 3, 32, 32, dtype=torch.float32)
    train_labels = torch.tensor([index % 10 for index in range(120)], dtype=torch.long)
    val_labels = torch.tensor([index % 10 for index in range(60)], dtype=torch.long)

    monkeypatch.setattr(
        science_exec,
        "_load_imagenette_canonical_dataset",
        lambda **kwargs: (
            train_images.clone(),
            train_labels.clone(),
            val_images.clone(),
            val_labels.clone(),
        ),
    )

    payload = {
        "problem_id": "cv-canonical-imagenette",
        "problem": "Investigate backbone transfer in computer vision",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "requested_benchmark": spec.benchmark_id,
        "canonical_only": True,
        "no_proxy_benchmarks": True,
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "backbone_transfer",
                "name": "Backbone Transfer",
                "benchmark": "transfer_comparison",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["top1_accuracy", "seed_variance"],
                "parameter_grid": {"backbone": ["resnet18", "vit_tiny"]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "cv_canonical_imagenette.json")
    assert report.status == "completed"
    assert report.canonical_comparable
    assert report.benchmark_alignment == "aligned"
    assert report.benchmark_truth_statuses == ["canonical_ready"]
    experiment = report.experiments[0]
    assert experiment.execution_mode == "vision_benchmark"
    assert not experiment.proxy_benchmark_used
    assert experiment.benchmark_id == "imagenette_transfer_canonical"
    assert experiment.statistical_summary is not None
    assert experiment.statistical_summary.statistically_ready
    assert experiment.statistical_summary.sample_count == 5
    assert experiment.statistical_summary.primary_metric == "top1_accuracy"
    assert experiment.statistical_summary.metrics[0].ci95_low is not None
    assert experiment.statistical_summary.metrics[0].ci95_high is not None


def test_deep_learning_canonical_cifar10_optimizer_uses_real_dataset_path(monkeypatch, tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("deep_learning")
    spec = registry.resolve_benchmark_suite(profile, "optimizer_comparison", tier="canonical")
    availability = registry.benchmark_availability(spec).model_copy(
        update={"imports_ready": True, "dataset_ready": True, "canonical_ready": True, "reason": None}
    )

    train_images = torch.rand(96, 3, 32, 32, dtype=torch.float32)
    val_images = torch.rand(48, 3, 32, 32, dtype=torch.float32)
    train_labels = torch.tensor([index % 10 for index in range(96)], dtype=torch.long)
    val_labels = torch.tensor([index % 10 for index in range(48)], dtype=torch.long)

    monkeypatch.setattr(
        science_exec,
        "_load_cifar10_canonical_dataset",
        lambda **kwargs: (
            train_images.clone(),
            train_labels.clone(),
            val_images.clone(),
            val_labels.clone(),
        ),
    )

    payload = {
        "problem_id": "dl-canonical-cifar10-optimizer",
        "problem": "Investigate optimizer sensitivity in deep learning",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "requested_benchmark": spec.benchmark_id,
        "canonical_only": True,
        "no_proxy_benchmarks": True,
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "optimizer_ablation",
                "name": "Optimizer Ablation",
                "benchmark": "optimizer_comparison",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["loss", "accuracy", "gradient_norm", "calibration_ece"],
                "parameter_grid": {"optimizer": ["adamw", "sgd"], "weight_decay": [0.0]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "dl_canonical_cifar10_optimizer.json")
    assert report.status == "completed"
    assert report.canonical_comparable
    assert report.benchmark_alignment == "aligned"
    assert report.benchmark_truth_statuses == ["canonical_ready"]
    experiment = report.experiments[0]
    assert experiment.execution_mode == "torch_benchmark"
    assert not experiment.proxy_benchmark_used
    assert experiment.benchmark_id == "cifar10_optimizer_canonical"
    assert experiment.statistical_summary is not None
    assert experiment.statistical_summary.statistically_ready
    assert experiment.statistical_summary.sample_count == 5
    assert experiment.statistical_summary.primary_metric == "accuracy"
    assert experiment.statistical_summary.metrics[0].ci95_low is not None
    assert experiment.statistical_summary.metrics[0].ci95_high is not None


def test_reinforcement_learning_canonical_offline_online_stays_refused(tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("reinforcement_learning")
    spec = registry.resolve_benchmark_suite(profile, "offline_online_transfer", tier="canonical")
    assert spec.truth_status == "unsupported"
    availability = registry.benchmark_availability(spec).model_copy(
        update={"imports_ready": True, "dataset_ready": True, "canonical_ready": False}
    )

    payload = {
        "problem_id": "rl-canonical-refusal",
        "problem": "Investigate offline-to-online transfer in reinforcement learning",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "requested_benchmark": spec.benchmark_id,
        "canonical_only": True,
        "no_proxy_benchmarks": True,
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "offline_online_gap",
                "name": "Offline-to-Online Gap",
                "benchmark": "offline_online_transfer",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["episodic_return", "sample_efficiency"],
                "parameter_grid": {"dataset_quality": ["medium"], "fine_tune_steps": [0, 10000]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "rl_canonical_refusal.json")
    assert report.status == "failed"
    assert not report.canonical_comparable
    assert report.benchmark_alignment == "refused"
    assert report.benchmark_truth_statuses == ["unsupported"]
    assert report.experiments[0].execution_mode == "truth_refusal"
