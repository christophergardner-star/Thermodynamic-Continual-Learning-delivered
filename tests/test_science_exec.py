from __future__ import annotations

from pathlib import Path

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
    spec = registry.resolve_benchmark_suite(profile, "heterophily_control", tier="canonical")
    assert spec.truth_status == "unsupported"
    availability = registry.benchmark_availability(spec)
    payload = {
        "problem_id": "graph-canonical",
        "problem": "Investigate heterophily in graph ML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "canonical_only": True,
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
                "parameter_grid": {"rewiring": ["off"], "normalization": ["batch"]},
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
