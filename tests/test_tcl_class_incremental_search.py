from __future__ import annotations

import json

from tar_lab.multimodal_payloads import _tcl_class_incremental_mechanism_specs
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import TCLMechanismCandidate, TCLMechanismSearchResult


def test_tcl_class_incremental_mechanism_specs_cover_search_space():
    specs = _tcl_class_incremental_mechanism_specs()
    names = {spec["name"] for spec in specs}

    assert "tcl_balanced" in names
    assert "tcl_stability_bias" in names
    assert "tcl_plasticity_bias" in names
    assert "tcl_carryover_anchor" in names
    assert "tcl_governor_only" in names
    assert "tcl_penalty_only" in names
    assert all("method" in spec for spec in specs)
    assert all(isinstance(spec.get("overrides", {}), dict) for spec in specs)


def test_tcl_mechanism_search_result_schema_valid():
    candidate = TCLMechanismCandidate(
        name="tcl_balanced",
        method="tcl",
        config_overrides={"tcl_penalty_lambda": 0.01},
        mean_forgetting=0.2,
        std_forgetting=0.01,
        mean_accuracy=0.6,
        std_accuracy=0.02,
        mean_jaf=0.48,
        delta_vs_sgd=-0.03,
        wins_vs_sgd=2,
    )
    result = TCLMechanismSearchResult(
        search_id="search-1",
        created_at="2026-04-24T00:00:00",
        seeds=[42, 0, 1],
        candidates=[candidate],
        best_candidate_name="tcl_balanced",
        best_delta_vs_strong_baseline=-0.02,
        p_value_vs_strong_baseline=0.04,
        effect_size_vs_strong_baseline=0.8,
        external_breakthrough_candidate=True,
        publishability_status="reviewer_grade_candidate",
        summary="Candidate clears the strong-baseline bar.",
    )

    assert result.setting == "class_incremental"
    assert result.strong_baseline_method == "ewc"
    assert result.external_breakthrough_candidate is True


def test_orchestrator_run_tcl_class_incremental_mechanism_search_persists(tmp_path, monkeypatch):
    orchestrator = TAROrchestrator(workspace=str(tmp_path))
    try:
        fake_result = TCLMechanismSearchResult(
            search_id="search-ci-1",
            created_at="2026-04-24T00:00:00",
            problem_id="problem-ci",
            seeds=[42, 0, 1],
            candidates=[
                TCLMechanismCandidate(
                    name="tcl_balanced",
                    method="tcl",
                    config_overrides={"tcl_penalty_lambda": 0.01},
                    mean_forgetting=0.2,
                    std_forgetting=0.01,
                    mean_accuracy=0.6,
                    std_accuracy=0.02,
                    mean_jaf=0.48,
                    delta_vs_sgd=-0.03,
                    wins_vs_sgd=3,
                )
            ],
            best_candidate_name="tcl_balanced",
            best_delta_vs_strong_baseline=-0.02,
            p_value_vs_strong_baseline=0.04,
            effect_size_vs_strong_baseline=0.8,
            external_breakthrough_candidate=True,
            publishability_status="reviewer_grade_candidate",
            summary="Candidate clears the strong-baseline bar.",
        )

        def _fake_search(*args, **kwargs):
            return fake_result

        monkeypatch.setattr(
            "tar_lab.multimodal_payloads.search_tcl_class_incremental_mechanisms",
            _fake_search,
        )

        result = orchestrator.run_tcl_class_incremental_mechanism_search(
            problem_id="problem-ci",
            seeds=[42, 0, 1],
        )

        assert result.search_id == "search-ci-1"
        search_path = tmp_path / "tar_state" / "comparisons" / "search-ci-1.json"
        assessment_path = (
            tmp_path / "tar_state" / "comparisons" / "external-breakthrough-search-ci-1.json"
        )
        assert search_path.exists()
        assert assessment_path.exists()
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
        assert assessment["problem_id"] == "problem-ci"
        assert assessment["external_breakthrough_candidate"] is True
        assert assessment["publishability_status"] == "reviewer_grade_candidate"
    finally:
        orchestrator.shutdown()
