from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    BaselineComparisonPlan,
    BaselineComparisonResult,
    StatisticalTestRecord,
)


def test_statistical_test_record_schema_valid():
    record = StatisticalTestRecord(
        test_id="test-1",
        timestamp="2026-04-18T10:00:00",
        test_type="mann_whitney_u",
        metric="mean_forgetting",
        group_a="tcl",
        group_b="ewc",
        group_a_values=[0.2, 0.3, 0.4],
        group_b_values=[0.3, 0.4, 0.5],
        statistic=5.0,
        p_value=0.2,
        effect_size=0.0,
        significant=False,
    )
    assert isinstance(record.significant, bool)
    with pytest.raises(ValidationError):
        StatisticalTestRecord(
            test_id="test-1",
            timestamp="2026-04-18T10:00:00",
            test_type="mann_whitney_u",
            metric="mean_forgetting",
            group_a="tcl",
            group_b="ewc",
            group_a_values=[0.2, 0.3, 0.4],
            group_b_values=[0.3, 0.4, 0.5],
            statistic=5.0,
            p_value=0.2,
            effect_size=0.0,
            significant=False,
            unknown="x",
        )


def test_comparison_plan_schema_valid():
    plan = BaselineComparisonPlan(
        plan_id="plan-1",
        timestamp="2026-04-18T10:00:00",
        project_id="proj-1",
    )
    assert plan.methods == ["tcl", "ewc", "si", "sgd_baseline"]
    assert plan.seeds == [42, 123, 456, 789, 1337]
    assert plan.status == "proposed"


def test_comparison_result_schema_valid():
    result = BaselineComparisonResult(
        result_id="result-1",
        plan_id="plan-1",
        project_id="proj-1",
        completed_at="2026-04-18T10:00:00",
        method_means={"tcl": {"mean_forgetting": 0.2}},
        method_stds={"tcl": {"mean_forgetting": 0.01}},
        pairwise_pvalues={"tcl_vs_ewc": 0.2},
        pairwise_effect_sizes={"tcl_vs_ewc": -0.5},
        tcl_is_significantly_better=False,
        tcl_is_significantly_worse=False,
        honest_assessment="TCL is not significantly different.",
        statistical_test_ids=["test-1"],
    )
    assert isinstance(result.tcl_is_significantly_better, bool)


def test_plan_baseline_comparison_persists(tmp_path):
    orchestrator = TAROrchestrator(str(tmp_path))
    try:
        plan = orchestrator.plan_baseline_comparison("proj_test")
        plan_path = Path(tmp_path) / "tar_state" / "comparisons" / f"{plan.plan_id}.json"
        assert plan_path.exists()
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        assert payload["plan_id"] == plan.plan_id
    finally:
        orchestrator.shutdown()


def test_mann_whitney_not_significant_equal_groups(tmp_path):
    orchestrator = TAROrchestrator(str(tmp_path))
    try:
        record = orchestrator._run_mann_whitney(
            "tcl",
            "ewc",
            "mean_forgetting",
            [0.3, 0.3, 0.3, 0.3, 0.3],
            [0.3, 0.3, 0.3, 0.3, 0.3],
        )
        assert record.significant is False
        assert record.p_value >= 0.05
    finally:
        orchestrator.shutdown()


def test_cohens_d_zero_for_equal_groups(tmp_path):
    orchestrator = TAROrchestrator(str(tmp_path))
    try:
        record = orchestrator._compute_cohens_d(
            "tcl",
            "ewc",
            "mean_forgetting",
            [0.3, 0.3, 0.3, 0.3, 0.3],
            [0.3, 0.3, 0.3, 0.3, 0.3],
        )
        assert abs(record.effect_size) < 0.01
    finally:
        orchestrator.shutdown()


def test_honest_assessment_not_significant():
    result = BaselineComparisonResult(
        result_id="result-1",
        plan_id="plan-1",
        project_id="proj-1",
        completed_at="2026-04-18T10:00:00",
        method_means={
            "tcl": {"mean_forgetting": 0.25},
            "ewc": {"mean_forgetting": 0.24},
            "si": {"mean_forgetting": 0.26},
            "sgd_baseline": {"mean_forgetting": 0.28},
        },
        method_stds={},
        pairwise_pvalues={
            "tcl_vs_ewc": 0.4,
            "tcl_vs_si": 0.6,
            "tcl_vs_sgd_baseline": 0.2,
        },
        pairwise_effect_sizes={
            "tcl_vs_ewc": 0.1,
            "tcl_vs_si": -0.1,
            "tcl_vs_sgd_baseline": -0.2,
        },
        tcl_is_significantly_better=False,
        tcl_is_significantly_worse=False,
        honest_assessment="Overall: TCL does not significantly differ from baselines on primary metric.",
        statistical_test_ids=[],
    )
    assert "not significantly" in result.honest_assessment
