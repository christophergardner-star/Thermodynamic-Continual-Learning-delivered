from __future__ import annotations

import tempfile
from pathlib import Path

import tar_lab.science_exec as science_exec
import tar_lab.science_profiles as science_profiles
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.science_exec import execute_study_payload
from tar_lab.science_profiles import ScienceProfileRegistry


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_benchmark_refuses_unavailable_dataset(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("TAR_ALLOW_DATA_DOWNLOAD", raising=False)
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("natural_language_processing")
    spec = registry.resolve_benchmark_suite(profile, "retrieval_prompt_ablation", tier="canonical")
    assert spec.truth_status == "unsupported"
    availability = registry.benchmark_availability(spec)
    assert not availability.canonical_ready

    payload = {
        "problem_id": "nlp-canonical",
        "problem": "Investigate retrieval failures in NLP",
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
                "template_id": "prompt_retrieval_ablation",
                "name": "Prompt and Retrieval Ablation",
                "benchmark": "retrieval_prompt_ablation",
                "benchmark_tier": "canonical",
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": ["rouge", "hallucination_rate", "calibration_ece"],
                "parameter_grid": {},
                "success_criteria": [],
            }
        ],
    }

    report = execute_study_payload(payload, tmp_path / "canonical_refusal.json")
    assert report.status == "failed"
    assert report.benchmark_ids == [spec.benchmark_id]
    assert report.benchmark_names == [spec.name]
    assert report.actual_benchmark_tiers == ["canonical"]
    assert report.benchmark_truth_statuses == ["unsupported"]
    assert report.benchmark_alignment == "refused"
    assert report.experiments[0].execution_mode == "truth_refusal"
    assert report.experiments[0].benchmark_truth_status == "unsupported"
    assert report.experiments[0].benchmark_alignment == "refused"
    assert not report.canonical_comparable


def test_study_problem_persists_benchmark_identity():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem(
                "Investigate optimization stability in deep learning",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            assert study.benchmark_ids
            assert study.benchmark_names
            assert study.actual_benchmark_tiers
            assert study.benchmark_truth_statuses
            assert study.benchmark_alignment in {"aligned", "downgraded", "refused", "mixed"}
            assert all(tier in {"smoke", "validation", "canonical"} for tier in study.actual_benchmark_tiers)
            persisted = orchestrator.store.latest_problem_study()
            assert persisted is not None
            assert persisted.benchmark_ids == study.benchmark_ids
            assert persisted.benchmark_names == study.benchmark_names
            assert persisted.benchmark_truth_statuses == study.benchmark_truth_statuses
        finally:
            orchestrator.shutdown()


def test_qml_canonical_report_preserves_truth_status_and_comparability(monkeypatch, tmp_path: Path):
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
            "noise_ready": True,
            "reason": "ready",
            "noise_reason": "ready",
        },
    )
    monkeypatch.setattr(
        science_exec,
        "_run_quantum_noise_with_pennylane",
        lambda qml, experiment: (
            {"shot_noise_robustness": 0.74, "gradient_norm_variance": 0.03, "trainability_gap": 2.1},
            "backend=pennylane:default.mixed qubits=3 ansatz_depth=3 init_scale=0.05 shots=[128, 512] noise_level=[0.0, 0.01, 0.05] seeds=[7, 18].",
        ),
    )

    payload = {
        "problem_id": "qml-canonical-provenance",
        "problem": "Investigate noisy trainability in QML",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": "canonical",
        "requested_benchmark": spec.benchmark_id,
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

    report = execute_study_payload(payload, tmp_path / "qml_canonical_provenance.json")
    assert report.status == "completed"
    assert report.canonical_comparable
    assert report.benchmark_truth_statuses == ["canonical_ready"]
    assert report.benchmark_alignment == "aligned"
    assert report.experiments[0].benchmark_truth_status == "canonical_ready"
    assert report.experiments[0].execution_mode == "pennylane_noise_backend"


def test_generic_ml_canonical_availability_splits_ready_and_refused(monkeypatch):
    monkeypatch.setenv("TAR_ALLOW_DATA_DOWNLOAD", "1")
    monkeypatch.setattr(science_profiles, "_validate_modules", lambda modules: (True, []))
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("generic_ml")

    refused_spec = registry.resolve_benchmark_suite(profile, "cross_validation", tier="canonical")
    ready_spec = registry.resolve_benchmark_suite(profile, "holdout_calibration", tier="canonical")

    refused = registry.benchmark_availability(refused_spec)
    ready = registry.benchmark_availability(ready_spec)

    assert refused_spec.benchmark_id == "openml_cc18_classification"
    assert refused_spec.truth_status == "unsupported"
    assert not refused.canonical_ready
    assert refused.reason == "registered benchmark is not yet executor-aligned and must be refused"

    assert ready_spec.benchmark_id == "openml_adult_calibration"
    assert ready_spec.truth_status == "canonical_ready"
    assert ready.imports_ready
    assert ready.dataset_ready
    assert ready.canonical_ready
    assert ready.reason is None


def test_graph_ml_canonical_availability_splits_ready_and_refused(monkeypatch):
    monkeypatch.setenv("TAR_ALLOW_DATA_DOWNLOAD", "1")
    monkeypatch.setattr(science_profiles, "_validate_modules", lambda modules: (True, []))
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("graph_ml")

    refused_spec = registry.resolve_benchmark_suite(profile, "depth_sweep", tier="canonical")
    ready_spec = registry.resolve_benchmark_suite(profile, "heterophily_control", tier="canonical")

    refused = registry.benchmark_availability(refused_spec)
    ready = registry.benchmark_availability(ready_spec)

    assert refused_spec.benchmark_id == "cora_depth_canonical"
    assert refused_spec.truth_status == "unsupported"
    assert not refused.canonical_ready
    assert refused.reason == "registered benchmark is not yet executor-aligned and must be refused"

    assert ready_spec.benchmark_id == "roman_empire_heterophily_canonical"
    assert ready_spec.truth_status == "canonical_ready"
    assert ready.imports_ready
    assert ready.dataset_ready
    assert ready.canonical_ready
    assert ready.reason is None


def test_computer_vision_canonical_availability_splits_ready_and_refused(monkeypatch):
    monkeypatch.setenv("TAR_ALLOW_DATA_DOWNLOAD", "1")
    monkeypatch.setattr(science_profiles, "_validate_modules", lambda modules: (True, []))
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("computer_vision")

    refused_spec = registry.resolve_benchmark_suite(profile, "corruption_robustness", tier="canonical")
    ready_spec = registry.resolve_benchmark_suite(profile, "transfer_comparison", tier="canonical")

    refused = registry.benchmark_availability(refused_spec)
    ready = registry.benchmark_availability(ready_spec)

    assert refused_spec.benchmark_id == "cifar10_c_corruption"
    assert refused_spec.truth_status == "unsupported"
    assert not refused.canonical_ready
    assert refused.reason == "registered benchmark is not yet executor-aligned and must be refused"

    assert ready_spec.benchmark_id == "imagenette_transfer_canonical"
    assert ready_spec.truth_status == "canonical_ready"
    assert ready.imports_ready
    assert ready.dataset_ready
    assert ready.canonical_ready
    assert ready.reason is None


def test_deep_learning_canonical_availability_splits_ready_and_refused(monkeypatch):
    monkeypatch.setenv("TAR_ALLOW_DATA_DOWNLOAD", "1")
    monkeypatch.setattr(science_profiles, "_validate_modules", lambda modules: (True, []))
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("deep_learning")

    ready_spec = registry.resolve_benchmark_suite(profile, "optimizer_comparison", tier="canonical")
    refused_spec = registry.resolve_benchmark_suite(profile, "scaling_law_probe", tier="canonical")

    ready = registry.benchmark_availability(ready_spec)
    refused = registry.benchmark_availability(refused_spec)

    assert ready_spec.benchmark_id == "cifar10_optimizer_canonical"
    assert ready_spec.truth_status == "canonical_ready"
    assert ready.imports_ready
    assert ready.dataset_ready
    assert ready.canonical_ready
    assert ready.reason is None

    assert refused_spec.benchmark_id == "cifar10_scaling_canonical"
    assert refused_spec.truth_status == "unsupported"
    assert not refused.canonical_ready
    assert refused.reason == "registered benchmark is not yet executor-aligned and must be refused"


def test_reinforcement_learning_canonical_suites_stay_refused(monkeypatch):
    monkeypatch.setenv("TAR_ALLOW_DATA_DOWNLOAD", "1")
    monkeypatch.setattr(science_profiles, "_validate_modules", lambda modules: (True, []))
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("reinforcement_learning")

    exploration_spec = registry.resolve_benchmark_suite(profile, "exploration_sweep", tier="canonical")
    offline_spec = registry.resolve_benchmark_suite(profile, "offline_online_transfer", tier="canonical")

    exploration = registry.benchmark_availability(exploration_spec)
    offline = registry.benchmark_availability(offline_spec)

    assert exploration_spec.benchmark_id == "minari_cartpole_exploration"
    assert exploration_spec.truth_status == "unsupported"
    assert not exploration.canonical_ready
    assert exploration.reason == "registered benchmark is not yet executor-aligned and must be refused"

    assert offline_spec.benchmark_id == "minari_offline_online_transfer"
    assert offline_spec.truth_status == "unsupported"
    assert not offline.canonical_ready
    assert offline.reason == "registered benchmark is not yet executor-aligned and must be refused"
