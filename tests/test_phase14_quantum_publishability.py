import numpy as np

from phase14_quantum_publishability import (
    KNOWN_METHOD_BASELINES,
    TARGET_CONDITION,
    _flatten_metric_diagonal,
    classify_publishability,
    paired_delta_stats,
)


def test_paired_delta_stats_detects_positive_advantage():
    stats = paired_delta_stats([1.4, 1.5, 1.6, 1.45], [1.0, 1.1, 1.05, 1.0])
    assert stats["mean_delta"] > 0.0
    assert stats["p_value"] < 0.05
    assert stats["cohens_d"] > 0.8


def test_flatten_metric_diagonal_matches_flattened_parameter_space():
    metric = np.zeros((2, 4, 2, 4), dtype=float)
    expected = np.arange(1, 9, dtype=float)
    metric.reshape(8, 8)[np.arange(8), np.arange(8)] = expected
    diag = _flatten_metric_diagonal(metric, expected_size=8)
    assert np.allclose(diag, expected)


def test_classify_publishability_requires_global_cost_floor():
    status, rationale = classify_publishability(
        candidate_id=TARGET_CONDITION.condition_id,
        comparisons={
            "global_cost_standard": {
                "mean_delta": 0.03,
                "p_value": 0.20,
                "cohens_d": 0.3,
            }
        },
        claim_status="accepted",
        novelty_vs_literature=0.62,
        research_support_count=4,
    )
    assert status == "no_reviewer_grade_signal"
    assert rationale


def test_classify_publishability_requires_known_mitigation_wins():
    status, rationale = classify_publishability(
        candidate_id=TARGET_CONDITION.condition_id,
        comparisons={
            "global_cost_standard": {
                "mean_delta": 0.6,
                "p_value": 0.01,
                "cohens_d": 1.2,
            },
            "global_cost_small_init": {
                "mean_delta": 0.04,
                "p_value": 0.20,
                "cohens_d": 0.3,
            },
            "local_cost_small_init": {
                "mean_delta": 0.03,
                "p_value": 0.25,
                "cohens_d": 0.2,
            },
            "layerwise_decay_global": {
                "mean_delta": 0.02,
                "p_value": 0.40,
                "cohens_d": 0.1,
            },
            "qng_diag_global": {
                "mean_delta": 0.01,
                "p_value": 0.60,
                "cohens_d": 0.05,
            },
        },
        claim_status="accepted",
        novelty_vs_literature=0.62,
        research_support_count=4,
    )
    assert status == "promising_but_not_novel"
    assert rationale


def test_classify_publishability_can_reach_reviewer_grade_candidate():
    comparisons = {
        "global_cost_standard": {
            "mean_delta": 0.6,
            "p_value": 0.01,
            "cohens_d": 1.2,
        },
    }
    comparisons.update(
        {
            baseline_id: {
                "mean_delta": 0.2,
                "p_value": 0.01,
                "cohens_d": 1.0,
            }
            for baseline_id in KNOWN_METHOD_BASELINES
        }
    )
    status, rationale = classify_publishability(
        candidate_id=TARGET_CONDITION.condition_id,
        comparisons=comparisons,
        claim_status="accepted",
        novelty_vs_literature=0.61,
        research_support_count=4,
    )
    assert status == "reviewer_grade_candidate"
    assert rationale
