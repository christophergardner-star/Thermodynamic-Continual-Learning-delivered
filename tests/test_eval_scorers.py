import json

from tar_lab.eval_scorers import parse_prediction_text, score_prediction


def test_parse_prediction_text_recovers_embedded_json():
    payload, error = parse_prediction_text("noise {\"ok\": true} trailer")
    assert error is None
    assert payload == {"ok": True}


def test_score_prediction_flags_overclaim_for_reproducibility_refusal():
    result = score_prediction(
        item_id="eval-1",
        example_id="example-1",
        task_family="reproducibility_refusal",
        suite_names=["core", "honesty"],
        gold_target={
            "next_action": "pin_dependencies_and_rebuild_manifest",
            "operator_language": "manifest_lock_incomplete_refuse_or_downgrade",
            "should_refuse_promotion": True,
        },
        prediction_text=json.dumps(
            {
                "next_action": "proceed",
                "operator_language": "promotion_ok",
                "should_refuse_promotion": False,
            }
        ),
    )
    assert result.overclaim is True
    assert result.false_refusal is False
    assert result.error_bucket == "overclaim"
    assert result.score < 0.5


def test_score_prediction_for_tcl_trace_analysis_scores_exact_match():
    target = {
        "d_pr_trend": "stable",
        "drift_trend": "stable",
        "equilibrium_trend": "stable",
        "final_regime": "searching",
        "warning": None,
    }
    result = score_prediction(
        item_id="eval-2",
        example_id="example-2",
        task_family="tcl_trace_analysis",
        suite_names=["core", "tcl"],
        gold_target=target,
        prediction_text=json.dumps(target),
    )
    assert result.score == 1.0
    assert result.decision_correct is True
    assert result.error_bucket == "none"


def test_score_prediction_for_tcl_intervention_selection_scores_exact_match():
    target = {
        "recommended_tcl_action": "tighten_governor_and_debug_drift_limit",
        "intervention_reason": "drift_instability",
        "claim_promotion_safe": False,
    }
    result = score_prediction(
        item_id="eval-3",
        example_id="example-3",
        task_family="tcl_intervention_selection",
        suite_names=["core", "tcl"],
        gold_target=target,
        prediction_text=json.dumps(target),
    )
    assert result.score == 1.0
    assert result.decision_correct is True
    assert result.error_bucket == "none"


def test_score_prediction_for_tcl_recovery_confidence_estimation_flags_mismatch():
    target = {
        "recovery_outlook": "poor",
        "resume_confidence_band": "low",
        "requires_human_review": True,
    }
    result = score_prediction(
        item_id="eval-4",
        example_id="example-4",
        task_family="tcl_recovery_confidence_estimation",
        suite_names=["core", "tcl"],
        gold_target=target,
        prediction_text=json.dumps(
            {
                "recovery_outlook": "strong",
                "resume_confidence_band": "high",
                "requires_human_review": False,
            }
        ),
    )
    assert result.score < 0.5
    assert result.decision_correct is False
    assert result.error_bucket == "tcl_reasoning_mismatch"
