"""Acceptance tests for the Stack-A -> Stack-B science_exec bridge.

Covers the adapter/translation logic without running any real training:
  - _build_study_payload_from_spec (spec -> Stack-B payload)
  - _accuracy_verdict (accuracy-domain, higher-is-better, default NULL)
  - _adapt_science_report (ProblemExecutionReport -> ExperimentResult)
  - _run_science_exec end-to-end with execute_study_payload monkeypatched
    (verifies provenance dump + adapted result, no GPU/training)
"""
import pytest

from tar_experiment_orchestrator import ExperimentOrchestrator, ExperimentSpec, ExperimentResult
from tar_lab.schemas import ProblemExecutionReport, ProblemExperimentResult


def _orch():
    # Bypass heavy __init__ — the bridge methods use no instance state except
    # _exp_dir / _log (set explicitly where needed).
    return ExperimentOrchestrator.__new__(ExperimentOrchestrator)


def _exp(status="completed", acc=0.9, name="exp1"):
    return ProblemExperimentResult(
        template_id="t1", name=name, benchmark="bench1",
        execution_mode="local_python", status=status,
        metrics={"accuracy": acc} if status == "completed" else {},
    )


def _report(exps, domain="generic_ml", status="completed"):
    return ProblemExecutionReport(
        problem_id="p1", problem="does X help", profile_id="prof1",
        domain=domain, execution_mode="local_python",
        experiments=exps, summary="ok",
        status=status, recommended_next_step="none", artifact_path="probe.json",
    )


def _spec(overrides):
    return ExperimentSpec(
        name="sci1", project_id="proj1", hypothesis_name="hyp1",
        dataset="domain_experiment", method="science_exec", seeds=[0],
        config_overrides=overrides, runner_key="science_exec",
    )


# ── _build_study_payload_from_spec ─────────────────────────────────────────────
def test_build_payload_ok():
    spec = _spec({
        "domain": "quantum_ml",
        "experiments": [{"template_id": "t", "name": "e", "benchmark": "b"}],
        "benchmark_tier": "validation",
        "validation_imports": ["numpy", "pennylane"],
    })
    payload = _orch()._build_study_payload_from_spec(spec)
    assert payload["domain"] == "quantum_ml"
    assert payload["experiments"] == spec.config_overrides["experiments"]
    assert payload["environment"]["validation_imports"] == ["numpy", "pennylane"]
    assert payload["problem_id"] == spec.id  # no frontier_problem_id set


def test_build_payload_domain_from_runtime_context():
    spec = _spec({"experiments": [{"template_id": "t", "name": "e", "benchmark": "b"}]})
    spec.runtime_context = {"domain_id": "graph_ml"}
    payload = _orch()._build_study_payload_from_spec(spec)
    assert payload["domain"] == "graph_ml"


def test_build_payload_requires_experiments():
    spec = _spec({"domain": "quantum_ml"})  # no experiments
    with pytest.raises(ValueError):
        _orch()._build_study_payload_from_spec(spec)


# ── _accuracy_verdict ──────────────────────────────────────────────────────────
def test_verdict_error_when_no_values():
    assert ExperimentOrchestrator._accuracy_verdict([], {}, _spec({"experiments": [1]})) == "ERROR"


def test_verdict_null_without_baseline():
    assert ExperimentOrchestrator._accuracy_verdict([0.9], {}, _spec({"experiments": [1]})) == "NULL"


def test_verdict_breakthrough_when_better_and_significant():
    spec = _spec({"experiments": [1], "baseline_metric_value": 0.8, "alpha": 0.05})
    assert ExperimentOrchestrator._accuracy_verdict([0.9, 0.92], {"p_value": 0.01}, spec) == "BREAKTHROUGH"


def test_verdict_directional_when_better_not_significant():
    spec = _spec({"experiments": [1], "baseline_metric_value": 0.8})
    assert ExperimentOrchestrator._accuracy_verdict([0.9], {"p_value": 0.4}, spec) == "DIRECTIONAL"


def test_verdict_adverse_when_worse():
    spec = _spec({"experiments": [1], "baseline_metric_value": 0.95})
    assert ExperimentOrchestrator._accuracy_verdict([0.80], {"p_value": 0.01}, spec) == "ADVERSE"


# ── _adapt_science_report ──────────────────────────────────────────────────────
def test_adapt_completed_report():
    spec = _spec({"experiments": [1], "primary_metric": "accuracy"})
    report = _report([_exp(acc=0.8, name="a"), _exp(acc=0.9, name="b")])
    res = _orch()._adapt_science_report(spec, report)
    assert isinstance(res, ExperimentResult)
    assert res.dataset == "generic_ml"
    assert abs(res.mean_accuracy - 0.85) < 1e-9
    assert res.mean_forgetting == 0.0          # not applicable to accuracy domain
    assert res.verdict == "NULL"               # no baseline provided
    assert len(res.seed_results) == 2


def test_adapt_all_failed_is_error():
    spec = _spec({"experiments": [1]})
    report = _report([_exp(status="failed"), _exp(status="skipped")])
    res = _orch()._adapt_science_report(spec, report)
    assert res.verdict == "ERROR"


# ── _run_science_exec (monkeypatched core; no training) ────────────────────────
def test_run_science_exec_persists_and_adapts(tmp_path, monkeypatch):
    obj = _orch()
    obj._exp_dir = tmp_path
    obj._log = lambda *a, **k: None
    report = _report([_exp(acc=0.77)])

    import tar_lab.science_exec as se
    monkeypatch.setattr(se, "execute_study_payload", lambda payload, artifact: report)

    spec = _spec({
        "domain": "generic_ml",
        "experiments": [{"template_id": "t", "name": "e", "benchmark": "b"}],
    })
    res = obj._run_science_exec(spec)
    assert (tmp_path / spec.id / "science_exec_report.json").exists()
    assert abs(res.mean_accuracy - 0.77) < 1e-9
    assert res.verdict == "NULL"
