from __future__ import annotations

import tempfile
from pathlib import Path

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.science_exec import execute_study_payload
from tar_lab.science_profiles import ScienceProfileRegistry


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_status_exposes_statistical_policy_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            payload = orchestrator.benchmark_status(profile_id="generic_ml")
            smoke_suite = next(
                item
                for item in payload["benchmarks"]
                if item["spec"]["benchmark_id"] == "generic_cv_smoke"
            )
            validation_suite = next(
                item
                for item in payload["benchmarks"]
                if item["spec"]["benchmark_id"] == "breast_cancer_validation"
            )
            canonical_suite = next(
                item
                for item in payload["benchmarks"]
                if item["spec"]["benchmark_id"] == "openml_cc18_classification"
            )

            assert smoke_suite["spec"]["recommended_seed_runs"] == 1
            assert smoke_suite["spec"]["statistical_validation_required"] is False
            assert validation_suite["spec"]["recommended_seed_runs"] == 3
            assert validation_suite["spec"]["statistical_validation_required"] is True
            assert canonical_suite["spec"]["recommended_seed_runs"] == 5
            assert canonical_suite["spec"]["statistical_validation_required"] is True
        finally:
            orchestrator.shutdown()


def test_validation_seeded_benchmark_reports_statistically_ready(tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("generic_ml")
    spec = registry.resolve_benchmark_suite(profile, "cross_validation", tier="validation")
    availability = registry.benchmark_availability(spec)
    payload = {
        "problem_id": "generic-validation-stat-ready",
        "problem": "Investigate generic ML validation performance",
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
                "parameter_grid": {"learning_rate": [0.001], "batch_size": [32]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "generic_validation_stat_ready.json")
    summary = report.experiments[0].statistical_summary

    assert report.status == "completed"
    assert summary is not None
    assert summary.statistically_ready
    assert summary.primary_metric == "accuracy"
    assert summary.sample_count == 3
    assert summary.recommended_seed_runs == 3
    assert summary.metrics
    assert summary.metrics[0].ci95_low is not None
    assert summary.metrics[0].ci95_high is not None
    assert report.benchmark_statistical_summary is not None
    assert report.benchmark_statistical_summary.statistically_ready
    assert report.benchmark_statistical_summary.statistically_ready_experiment_count == 1


def test_validation_non_seeded_benchmark_reports_statistical_insufficiency(tmp_path: Path):
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("generic_ml")
    spec = registry.resolve_benchmark_suite(profile, "holdout_calibration", tier="validation")
    availability = registry.benchmark_availability(spec)
    payload = {
        "problem_id": "generic-validation-underpowered",
        "problem": "Investigate generic ML calibration",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "validation",
        "environment": {"validation_imports": []},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": "calibration_check",
                "name": "Calibration Check",
                "benchmark": "holdout_calibration",
                "benchmark_tier": "validation",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["calibration_ece", "accuracy", "auroc"],
                "parameter_grid": {"temperature_scaling": [False, True]},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "generic_validation_underpowered.json")
    summary = report.experiments[0].statistical_summary

    assert report.status == "completed"
    assert summary is not None
    assert summary.primary_metric == "calibration_ece"
    assert not summary.statistically_ready
    assert summary.sample_count == 1
    assert summary.recommended_seed_runs == 3
    assert any("requires 3 seed runs" in note for note in summary.notes)
    assert report.benchmark_statistical_summary is not None
    assert not report.benchmark_statistical_summary.statistically_ready
    assert any("under-powered" in note for note in report.benchmark_statistical_summary.notes)
