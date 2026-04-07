from __future__ import annotations

from pathlib import Path

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
    assert not report.experiments[0].provenance_complete


def test_canonical_comparability_stays_false_on_refusal(tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("graph_ml")
    spec = registry.resolve_benchmark_suite(profile, "heterophily_control", tier="canonical")
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
