from __future__ import annotations

import tempfile
from pathlib import Path

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.science_exec import execute_study_payload
from tar_lab.science_profiles import ScienceProfileRegistry


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_benchmark_refuses_unavailable_dataset(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("TAR_ALLOW_DATA_DOWNLOAD", raising=False)
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    profile = registry.get("natural_language_processing")
    spec = registry.resolve_benchmark_suite(profile, "retrieval_prompt_ablation", tier="canonical")
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
    assert report.experiments[0].execution_mode == "canonical_refusal"
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
            assert all(tier in {"smoke", "validation", "canonical"} for tier in study.actual_benchmark_tiers)
            persisted = orchestrator.store.latest_problem_study()
            assert persisted is not None
            assert persisted.benchmark_ids == study.benchmark_ids
            assert persisted.benchmark_names == study.benchmark_names
        finally:
            orchestrator.shutdown()
