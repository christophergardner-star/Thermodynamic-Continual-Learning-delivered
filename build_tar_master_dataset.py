from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SYSTEM_PROMPT = (
    "You are TAR, a disciplined research operator. Separate measured results, "
    "inferences, hypotheses, blockers, and next actions. Stay honest when "
    "benchmark truth, reproducibility, or capability is incomplete."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TAR/TCL master dataset from TAR state artifacts."
    )
    parser.add_argument("--state-dir", action="append", dest="state_dirs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="tar-master-v1")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _maybe_json(path: Path) -> Any:
    if not path.exists():
        return None
    return _load_json(path)


def _maybe_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_jsonl(path)


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _compact_hypotheses(raw_hypotheses: Any) -> list[str]:
    compact: list[str] = []
    for item in _ensure_list(raw_hypotheses):
        if isinstance(item, str):
            compact.append(item)
        elif isinstance(item, dict):
            text = item.get("hypothesis") or item.get("name") or item.get("summary")
            if text:
                compact.append(str(text))
    return compact[:4]


def _compact_experiments(experiments: Any, limit: int = 3) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in _ensure_list(experiments)[:limit]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "template_id": item.get("template_id"),
                "name": item.get("name"),
                "benchmark": item.get("benchmark"),
                "benchmark_tier": item.get("benchmark_tier"),
                "metrics": _ensure_list(item.get("metrics"))[:5],
                "success_criteria": _ensure_list(item.get("success_criteria"))[:3],
            }
        )
    return compact


def _compact_import_failures(imports_failed: Any, limit: int = 5) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for item in _ensure_list(imports_failed)[:limit]:
        if isinstance(item, dict):
            compact.append(
                {
                    "module": str(item.get("module", "")),
                    "error": str(item.get("error", "")),
                }
            )
    return compact


def _compact_thermo_metric(metric: Any) -> dict[str, Any]:
    if not isinstance(metric, dict):
        return {}
    return {
        "step": metric.get("step"),
        "energy_e": metric.get("energy_e"),
        "entropy_sigma": metric.get("entropy_sigma"),
        "drift_rho": metric.get("drift_rho"),
        "grad_norm": metric.get("grad_norm"),
        "effective_dimensionality": metric.get("effective_dimensionality"),
        "dimensionality_ratio": metric.get("dimensionality_ratio"),
        "equilibrium_fraction": metric.get("equilibrium_fraction"),
        "equilibrium_gate": metric.get("equilibrium_gate"),
        "statistically_ready": metric.get("statistically_ready"),
        "training_loss": metric.get("training_loss"),
        "gpu_temperature_c": metric.get("gpu_temperature_c"),
        "gpu_power_w": metric.get("gpu_power_w"),
    }


def _derive_tcl_regime(metrics: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    equilibrium_fraction = float(metrics.get("equilibrium_fraction") or 0.0)
    equilibrium_gate = bool(metrics.get("equilibrium_gate"))
    dimensionality_ratio = float(metrics.get("dimensionality_ratio") or 0.0)
    training_loss = metrics.get("training_loss")
    statistically_ready = bool(metrics.get("statistically_ready"))
    if (
        not statistically_ready
        and metrics.get("effective_dimensionality") is not None
        and metrics.get("equilibrium_fraction") is not None
        and metrics.get("dimensionality_ratio") is not None
    ):
        statistically_ready = True

    max_quenching_loss = float(thresholds.get("max_quenching_loss", 0.8))
    min_dimensionality_ratio = float(thresholds.get("min_dimensionality_ratio", 0.55))

    regime = "searching"
    warning = None
    if (
        statistically_ready
        and dimensionality_ratio > 0.0
        and training_loss is not None
        and float(training_loss) <= max_quenching_loss
        and dimensionality_ratio < min_dimensionality_ratio
    ):
        regime = "thermodynamic_quenching"
        warning = "loss is holding while D_PR has collapsed"
    elif equilibrium_gate and statistically_ready:
        regime = "equilibrium"
    elif equilibrium_fraction >= 0.5 and statistically_ready:
        regime = "stabilizing"
    elif not statistically_ready:
        regime = "warming_up"
        warning = "statistical warmup incomplete"

    return {
        "regime": regime,
        "warning": warning,
        "statistically_ready": statistically_ready,
        "equilibrium_fraction": equilibrium_fraction,
        "equilibrium_gate": equilibrium_gate,
        "dimensionality_ratio": dimensionality_ratio,
    }


def _trend_label(start: Any, end: Any, *, collapse_threshold: float = 0.75) -> str:
    if start is None or end is None:
        return "unknown"
    start_value = float(start)
    end_value = float(end)
    if abs(start_value) <= 1e-9:
        if abs(end_value) <= 1e-9:
            return "stable"
        return "increasing" if end_value > 0.0 else "decreasing"
    ratio = end_value / start_value
    if ratio <= collapse_threshold:
        return "collapsing"
    if ratio >= 1.10:
        return "increasing"
    if ratio <= 0.90:
        return "decreasing"
    return "stable"


def _derive_tcl_recovery_action(recovery: dict[str, Any]) -> dict[str, Any]:
    status = str(recovery.get("status") or "unknown")
    consecutive_fail_fast = int(recovery.get("consecutive_fail_fast") or 0)
    if status == "completed":
        next_action = "resume_from_last_known_stable_hyperparameters"
    elif status == "fail_fast" and consecutive_fail_fast >= 3:
        next_action = "force_strategy_pivot"
    elif status == "fail_fast":
        next_action = "debug_governor_breach_before_resume"
    elif status == "pivoted":
        next_action = "prepare_alternative_strategy_family"
    else:
        next_action = "inspect_recovery_state"
    return {
        "next_action": next_action,
        "anchor_reuse_recommended": bool(recovery.get("last_anchor_path")),
        "stable_hyperparameters_available": bool(recovery.get("last_known_stable_hyperparameters")),
    }


def _hash_split(example_id: str) -> str:
    bucket = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _sanitize_value(value: Any, *, repo_root: Path, state_dir: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_value(inner, repo_root=repo_root, state_dir=state_dir)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_value(item, repo_root=repo_root, state_dir=state_dir)
            for item in value
        ]
    if not isinstance(value, str):
        return value

    text = value
    replacements = {
        str(repo_root): "<REPO_ROOT>",
        str(state_dir): "<STATE_DIR>",
        str(repo_root).replace("\\", "/"): "<REPO_ROOT>",
        str(state_dir).replace("\\", "/"): "<STATE_DIR>",
    }
    for raw, token in replacements.items():
        if raw:
            text = text.replace(raw, token)
    if text.startswith("/workspace/"):
        text = text.replace("/workspace", "<WORKSPACE_ROOT>", 1)
    return text


def _render_user_prompt(task_family: str, task_name: str, input_context: dict[str, Any]) -> str:
    payload = json.dumps(input_context, indent=2, sort_keys=True)
    return (
        f"Task family: {task_family}\n"
        f"Task: {task_name}\n"
        "Use the TAR operator contract. Return a concise JSON answer aligned to the target.\n\n"
        f"Input state:\n{payload}"
    )


def _example_record(
    *,
    version: str,
    task_family: str,
    task_name: str,
    source_kind: str,
    source_id: str,
    source_file: str,
    input_context: dict[str, Any],
    target: dict[str, Any],
    repo_root: Path,
    state_dir: Path,
    tags: Iterable[str] = (),
) -> dict[str, Any]:
    example_id = f"{task_family}:{task_name}:{source_kind}:{source_id}"
    safe_input = _sanitize_value(input_context, repo_root=repo_root, state_dir=state_dir)
    safe_target = _sanitize_value(target, repo_root=repo_root, state_dir=state_dir)
    return {
        "example_id": example_id,
        "dataset_version": version,
        "split": _hash_split(example_id),
        "task_family": task_family,
        "task_name": task_name,
        "source_kind": source_kind,
        "tags": list(tags),
        "input_context": safe_input,
        "target": safe_target,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _render_user_prompt(task_family, task_name, safe_input)},
            {"role": "assistant", "content": json.dumps(safe_target, indent=2, sort_keys=True)},
        ],
        "provenance": {
            "state_file": source_file,
            "source_id": source_id,
            "observed": True,
        },
    }


def _examples_from_problem_study(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    problem_id = str(record.get("problem_id") or record.get("problem") or "unknown-problem")
    hypotheses = _compact_hypotheses(record.get("hypotheses"))
    experiments = _compact_experiments(record.get("experiments"))
    environment = record.get("environment") if isinstance(record.get("environment"), dict) else {}
    benchmark_statuses = _ensure_list(record.get("benchmark_truth_statuses")) or _ensure_list(record.get("benchmark_truth_status"))
    benchmark_alignment = record.get("benchmark_alignment")
    unresolved_packages = _ensure_list(environment.get("unresolved_packages"))[:8]
    examples = [
        _example_record(
            version=version,
            task_family="problem_scoping",
            task_name="study_to_next_action",
            source_kind="problem_study",
            source_id=problem_id,
            source_file="problem_studies.jsonl",
            input_context={
                "problem": record.get("problem"),
                "domain": record.get("domain"),
                "profile_id": record.get("profile_id"),
                "resolution_confidence": record.get("resolution_confidence"),
                "hypotheses": hypotheses,
                "experiments": experiments,
                "benchmark_tier": record.get("benchmark_tier"),
                "benchmark_ids": _ensure_list(record.get("benchmark_ids"))[:5],
                "benchmark_truth_statuses": benchmark_statuses[:5],
                "benchmark_alignment": benchmark_alignment,
                "reproducibility_complete": environment.get("reproducibility_complete"),
                "unresolved_packages": unresolved_packages,
                "status": record.get("status"),
            },
            target={
                "next_action": record.get("next_action"),
                "primary_hypotheses": hypotheses[:3],
                "benchmark_assessment": {
                    "requested_tier": record.get("benchmark_tier"),
                    "alignment": benchmark_alignment,
                    "canonical_comparable": record.get("canonical_comparable"),
                    "truth_statuses": benchmark_statuses[:5],
                },
                "reproducibility_risk": {
                    "complete": environment.get("reproducibility_complete"),
                    "unresolved_packages": unresolved_packages,
                    "lock_incomplete_reason": environment.get("lock_incomplete_reason"),
                },
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("planning", "benchmark_truth", "reproducibility"),
        )
    ]
    if record.get("benchmark_tier") or benchmark_alignment:
        examples.append(
            _example_record(
                version=version,
                task_family="benchmark_honesty",
                task_name="study_truth_assessment",
                source_kind="problem_study",
                source_id=problem_id,
                source_file="problem_studies.jsonl",
                input_context={
                    "problem": record.get("problem"),
                    "benchmark_ids": _ensure_list(record.get("benchmark_ids"))[:5],
                    "benchmark_names": _ensure_list(record.get("benchmark_names"))[:5],
                    "requested_benchmark_tier": record.get("benchmark_tier"),
                    "canonical_comparable": record.get("canonical_comparable"),
                    "benchmark_alignment": benchmark_alignment,
                    "benchmark_truth_statuses": benchmark_statuses[:5],
                    "benchmark_availability": _ensure_list(record.get("benchmark_availability"))[:5],
                },
                target={
                    "benchmark_alignment": benchmark_alignment,
                    "canonical_comparable": record.get("canonical_comparable"),
                    "truthful_statuses": benchmark_statuses[:5],
                    "recommended_operator_language": (
                        "canonical" if record.get("canonical_comparable") else "validation_or_refused"
                    ),
                },
                repo_root=repo_root,
                state_dir=state_dir,
                tags=("benchmark_truth",),
            )
        )
    return examples


def _examples_from_problem_execution(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    problem_id = str(record.get("problem_id") or record.get("problem") or "unknown-problem")
    return [
        _example_record(
            version=version,
            task_family="execution_diagnosis",
            task_name="execution_report_to_recovery_action",
            source_kind="problem_execution",
            source_id=problem_id,
            source_file="problem_executions.jsonl",
            input_context={
                "problem": record.get("problem"),
                "domain": record.get("domain"),
                "status": record.get("status"),
                "execution_mode": record.get("execution_mode"),
                "summary": record.get("summary"),
                "imports_ok": _ensure_list(record.get("imports_ok"))[:8],
                "imports_failed": _compact_import_failures(record.get("imports_failed")),
            },
            target={
                "diagnosis": record.get("status"),
                "summary": record.get("summary"),
                "recommended_next_step": record.get("recommended_next_step"),
                "blockers": [item.get("module") for item in _compact_import_failures(record.get("imports_failed"))],
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("runtime", "dependency_diagnosis"),
        )
    ]


def _examples_from_verification_report(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    trial_id = str(record.get("trial_id") or "unknown-trial")
    seed_variance = record.get("seed_variance") if isinstance(record.get("seed_variance"), dict) else {}
    calibration = record.get("calibration") if isinstance(record.get("calibration"), dict) else {}
    ablations = _ensure_list(record.get("ablations"))[:5]
    return [
        _example_record(
            version=version,
            task_family="verification_judgement",
            task_name="verification_report_to_verdict",
            source_kind="verification_report",
            source_id=trial_id,
            source_file="verification_reports.jsonl",
            input_context={
                "trial_id": trial_id,
                "control_score": record.get("control_score"),
                "seed_variance": {
                    "num_runs": seed_variance.get("num_runs"),
                    "stable": seed_variance.get("stable"),
                    "loss_std": seed_variance.get("loss_std"),
                    "dimensionality_std": seed_variance.get("dimensionality_std"),
                    "calibration_ece_mean": seed_variance.get("calibration_ece_mean"),
                },
                "calibration": {
                    "ece": calibration.get("ece"),
                    "accuracy": calibration.get("accuracy"),
                    "mean_confidence": calibration.get("mean_confidence"),
                },
                "ablations": [
                    {
                        "name": item.get("name"),
                        "delta_vs_control": item.get("delta_vs_control"),
                    }
                    for item in ablations
                    if isinstance(item, dict)
                ],
            },
            target={
                "verdict": record.get("verdict"),
                "recommendations": _ensure_list(record.get("recommendations"))[:5],
                "replication_status": {
                    "num_runs": seed_variance.get("num_runs"),
                    "stable": seed_variance.get("stable"),
                },
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("verification", "falsification_pressure"),
        )
    ]


def _examples_from_research_decision(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    decision_id = str(record.get("decision_id") or "unknown-decision")
    evidence_bundle = record.get("evidence_bundle") if isinstance(record.get("evidence_bundle"), dict) else {}
    traces = _ensure_list(evidence_bundle.get("traces"))[:4]
    hypotheses = _compact_hypotheses(record.get("hypotheses"))
    return [
        _example_record(
            version=version,
            task_family="decision_rationale",
            task_name="evidence_bundle_to_selected_action",
            source_kind="research_decision",
            source_id=decision_id,
            source_file="research_decisions.jsonl",
            input_context={
                "prompt": record.get("prompt"),
                "mode": record.get("mode"),
                "problem_id": record.get("problem_id"),
                "evidence_confidence": evidence_bundle.get("confidence"),
                "supporting_documents": evidence_bundle.get("supporting_document_ids"),
                "top_trace_ids": [item.get("document_id") for item in traces if isinstance(item, dict)],
                "hypotheses": hypotheses,
            },
            target={
                "selected_action": record.get("selected_action"),
                "confidence": record.get("confidence"),
                "top_supporting_documents": evidence_bundle.get("supporting_document_ids"),
                "notes": _ensure_list(record.get("notes"))[:5],
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("decision", "evidence_conditioning"),
        )
    ]


def _examples_from_research_project(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    project_id = str(record.get("project_id") or "unknown-project")
    resume_snapshot = record.get("resume_snapshot") if isinstance(record.get("resume_snapshot"), dict) else {}
    budget = record.get("budget_ledger") if isinstance(record.get("budget_ledger"), dict) else {}
    next_action = None
    for item in _ensure_list(record.get("planned_actions")):
        if isinstance(item, dict) and item.get("action_id") == resume_snapshot.get("next_action_id"):
            next_action = item
            break
    return [
        _example_record(
            version=version,
            task_family="project_resume",
            task_name="project_state_to_resume_snapshot",
            source_kind="research_project",
            source_id=project_id,
            source_file="research_projects.json",
            input_context={
                "title": record.get("title"),
                "goal": record.get("goal"),
                "status": record.get("status"),
                "active_thread_id": record.get("active_thread_id"),
                "latest_decision_summary": record.get("latest_decision_summary"),
                "budget": budget,
                "current_question_id": resume_snapshot.get("current_question_id"),
                "blockers": resume_snapshot.get("blockers"),
                "budget_remaining_summary": resume_snapshot.get("budget_remaining_summary"),
            },
            target={
                "resume_snapshot": resume_snapshot,
                "next_action": next_action,
                "budget_pressure_level": budget.get("budget_pressure_level"),
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("continuity", "resume"),
        )
    ]


def _examples_from_falsification_plan(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    plan_id = str(record.get("plan_id") or "unknown-plan")
    coverage = record.get("coverage") if isinstance(record.get("coverage"), dict) else {}
    tests = _ensure_list(record.get("tests"))[:6]
    return [
        _example_record(
            version=version,
            task_family="falsification_planning",
            task_name="trigger_bundle_to_meta_tests",
            source_kind="falsification_plan",
            source_id=plan_id,
            source_file="falsification_plans.jsonl",
            input_context={
                "project_id": record.get("project_id"),
                "thread_id": record.get("thread_id"),
                "status": record.get("status"),
                "trigger_reason": record.get("trigger_reason"),
                "coverage": coverage,
            },
            target={
                "tests": [
                    {
                        "kind": item.get("kind"),
                        "description": item.get("description"),
                        "expected_falsification_value": item.get("expected_falsification_value"),
                    }
                    for item in tests
                    if isinstance(item, dict)
                ],
                "coverage": coverage,
                "overall_sufficient": coverage.get("overall_sufficient"),
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("falsification", "meta_tests"),
        )
    ]


def _examples_from_project_priority_record(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    record_id = str(record.get("record_id") or record.get("project_id") or "unknown-priority")
    return [
        _example_record(
            version=version,
            task_family="prioritization",
            task_name="project_priority_record_to_recommendation",
            source_kind="project_priority_record",
            source_id=record_id,
            source_file="project_priority_records.jsonl",
            input_context={
                "project_id": record.get("project_id"),
                "action_id": record.get("action_id"),
                "priority_score": record.get("priority_score"),
                "expected_value": record.get("expected_value"),
                "evidence_debt": record.get("evidence_debt"),
                "contradiction_pressure": record.get("contradiction_pressure"),
                "staleness_penalty": record.get("staleness_penalty"),
                "budget_pressure": record.get("budget_pressure"),
                "benchmark_readiness": record.get("benchmark_readiness"),
            },
            target={
                "recommended_state": record.get("recommended_state"),
                "rationale": _ensure_list(record.get("rationale"))[:5],
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("prioritization", "budgeting"),
        )
    ]


def _examples_from_portfolio_decision(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    decision_id = str(record.get("decision_id") or "unknown-portfolio-decision")
    return [
        _example_record(
            version=version,
            task_family="portfolio_governance",
            task_name="portfolio_decision_to_budget_allocation",
            source_kind="portfolio_decision",
            source_id=decision_id,
            source_file="portfolio_decisions.jsonl",
            input_context={
                "selected_project_id": record.get("selected_project_id"),
                "selected_action_id": record.get("selected_action_id"),
                "deferred_project_ids": _ensure_list(record.get("deferred_project_ids"))[:10],
                "parked_project_ids": _ensure_list(record.get("parked_project_ids"))[:10],
                "escalated_project_ids": _ensure_list(record.get("escalated_project_ids"))[:10],
                "retired_project_ids": _ensure_list(record.get("retired_project_ids"))[:10],
            },
            target={
                "selected_project_id": record.get("selected_project_id"),
                "selected_action_id": record.get("selected_action_id"),
                "rationale": _ensure_list(record.get("rationale"))[:6],
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("portfolio", "governance"),
        )
    ]


def _examples_from_tcl_payload_bundle(
    bundle: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    config = bundle.get("config") if isinstance(bundle.get("config"), dict) else {}
    metrics = _compact_thermo_metric(summary.get("last_metrics"))
    thresholds = config.get("governor_thresholds") if isinstance(config.get("governor_thresholds"), dict) else {}
    regime = _derive_tcl_regime(metrics, thresholds)
    trial_id = str(summary.get("trial_id") or bundle.get("source_id") or "unknown-tcl-trial")
    source_file = bundle.get("summary_file") or "tar_runs/payload_summary.json"
    data_provenance = config.get("data_provenance") if isinstance(config.get("data_provenance"), dict) else {}
    backend_provenance = config.get("backend_provenance") if isinstance(config.get("backend_provenance"), dict) else {}
    return [
        _example_record(
            version=version,
            task_family="tcl_regime_diagnosis",
            task_name="payload_summary_to_regime_assessment",
            source_kind="tcl_payload_summary",
            source_id=trial_id,
            source_file=str(source_file),
            input_context={
                "trial_id": trial_id,
                "strategy_family": summary.get("strategy_family") or config.get("strategy_family"),
                "governor_action": summary.get("governor_action"),
                "governor_reasons": _ensure_list(summary.get("governor_reasons"))[:6],
                "last_metrics": metrics,
                "anchor_effective_dimensionality": summary.get("anchor_effective_dimensionality"),
                "calibration": summary.get("calibration"),
                "research_grade": data_provenance.get("research_grade"),
                "has_fallback_data": data_provenance.get("has_fallback"),
                "backend_id": config.get("backend_id"),
                "required_metrics": _ensure_list(backend_provenance.get("required_metrics"))[:6],
            },
            target={
                "regime": regime.get("regime"),
                "warning": regime.get("warning"),
                "governor_action": summary.get("governor_action"),
                "governor_reasons": _ensure_list(summary.get("governor_reasons"))[:6],
                "recommended_tcl_action": (
                    "stabilize_before_claim"
                    if regime.get("regime") == "thermodynamic_quenching"
                    else "replicate_or_promote"
                    if regime.get("regime") == "equilibrium" and not data_provenance.get("has_fallback")
                    else "continue_monitoring"
                ),
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "thermodynamics", "governor"),
        )
    ]


def _examples_from_tcl_trace_bundle(
    bundle: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    metrics = [item for item in _ensure_list(bundle.get("metrics")) if isinstance(item, dict)]
    if not metrics:
        return []
    config = bundle.get("config") if isinstance(bundle.get("config"), dict) else {}
    first_metric = _compact_thermo_metric(metrics[0])
    last_metric = _compact_thermo_metric(metrics[-1])
    thresholds = config.get("governor_thresholds") if isinstance(config.get("governor_thresholds"), dict) else {}
    regime = _derive_tcl_regime(last_metric, thresholds)
    trial_id = str(last_metric.get("trial_id") or bundle.get("source_id") or "unknown-tcl-trace")
    source_file = bundle.get("trace_file") or "tar_runs/thermo_metrics.jsonl"
    return [
        _example_record(
            version=version,
            task_family="tcl_trace_analysis",
            task_name="thermo_trace_to_dynamics_summary",
            source_kind="tcl_thermo_trace",
            source_id=trial_id,
            source_file=str(source_file),
            input_context={
                "trial_id": trial_id,
                "num_steps": len(metrics),
                "first_metric": first_metric,
                "last_metric": last_metric,
                "governor_thresholds": thresholds,
            },
            target={
                "final_regime": regime.get("regime"),
                "warning": regime.get("warning"),
                "d_pr_trend": _trend_label(
                    first_metric.get("effective_dimensionality"),
                    last_metric.get("effective_dimensionality"),
                ),
                "equilibrium_trend": _trend_label(
                    first_metric.get("equilibrium_fraction"),
                    last_metric.get("equilibrium_fraction"),
                ),
                "drift_trend": _trend_label(
                    first_metric.get("drift_rho"),
                    last_metric.get("drift_rho"),
                ),
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "trace", "thermodynamics"),
        )
    ]


def _examples_from_recovery_state(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    trial_id = str(record.get("trial_id") or "unknown-recovery")
    recovery_target = _derive_tcl_recovery_action(record)
    return [
        _example_record(
            version=version,
            task_family="tcl_recovery_planning",
            task_name="recovery_state_to_resume_decision",
            source_kind="tcl_recovery_state",
            source_id=trial_id,
            source_file="recovery.json",
            input_context={
                "trial_id": trial_id,
                "status": record.get("status"),
                "last_known_stable_hyperparameters": record.get("last_known_stable_hyperparameters"),
                "last_fail_reason": record.get("last_fail_reason"),
                "last_fail_metrics": record.get("last_fail_metrics"),
                "consecutive_fail_fast": record.get("consecutive_fail_fast"),
                "last_strategy_family": record.get("last_strategy_family"),
                "last_anchor_path": record.get("last_anchor_path"),
                "max_effective_dimensionality_achieved": record.get("max_effective_dimensionality_achieved"),
            },
            target={
                "next_action": recovery_target.get("next_action"),
                "anchor_reuse_recommended": recovery_target.get("anchor_reuse_recommended"),
                "stable_hyperparameters_available": recovery_target.get("stable_hyperparameters_available"),
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "recovery", "continuation"),
        )
    ]


def _iter_research_projects(path: Path) -> list[dict[str, Any]]:
    payload = _maybe_json(path)
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        projects = payload.get("projects")
        if isinstance(projects, list):
            normalized: list[dict[str, Any]] = []
            for item in projects:
                if isinstance(item, dict):
                    normalized.append(item.get("project", item))
            return normalized
        return [payload]
    return []


def _iter_tcl_payload_bundles(repo_root: Path) -> list[dict[str, Any]]:
    tar_runs_root = repo_root / "tar_runs"
    if not tar_runs_root.exists():
        return []
    bundles: list[dict[str, Any]] = []
    for summary_path in sorted(tar_runs_root.rglob("payload_summary.json")):
        config_path = summary_path.with_name("config.json")
        thermo_path = summary_path.with_name("thermo_metrics.jsonl")
        summary = _maybe_json(summary_path)
        if not isinstance(summary, dict):
            continue
        bundle = {
            "summary": summary,
            "config": _maybe_json(config_path) if config_path.exists() else {},
            "metrics": _maybe_jsonl(thermo_path) if thermo_path.exists() else [],
            "summary_file": str(summary_path.relative_to(repo_root)),
            "trace_file": str(thermo_path.relative_to(repo_root)) if thermo_path.exists() else None,
            "source_id": summary.get("trial_id") or summary_path.parent.name,
        }
        bundles.append(bundle)
    return bundles


def _collect_examples_for_state_dir(state_dir: Path, *, version: str) -> list[dict[str, Any]]:
    repo_root = state_dir.parent.resolve()
    examples: list[dict[str, Any]] = []
    tcl_payload_bundles = _iter_tcl_payload_bundles(repo_root)
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "problem_studies.jsonl")
        for example in _examples_from_problem_study(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "problem_executions.jsonl")
        for example in _examples_from_problem_execution(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "verification_reports.jsonl")
        for example in _examples_from_verification_report(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "research_decisions.jsonl")
        for example in _examples_from_research_decision(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for record in _iter_research_projects(state_dir / "research_projects.json")
        for example in _examples_from_research_project(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "falsification_plans.jsonl")
        for example in _examples_from_falsification_plan(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "project_priority_records.jsonl")
        for example in _examples_from_project_priority_record(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "portfolio_decisions.jsonl")
        for example in _examples_from_portfolio_decision(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for bundle in tcl_payload_bundles
        for example in _examples_from_tcl_payload_bundle(
            bundle, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for bundle in tcl_payload_bundles
        for example in _examples_from_tcl_trace_bundle(
            bundle, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    recovery_state = _maybe_json(state_dir / "recovery.json")
    if isinstance(recovery_state, dict):
        examples.extend(
            _examples_from_recovery_state(
                recovery_state, version=version, repo_root=repo_root, state_dir=state_dir
            )
        )
    return examples


def build_master_dataset(
    state_dirs: Path | Iterable[Path], output_dir: Path, *, version: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_state_dirs = (
        [state_dirs] if isinstance(state_dirs, Path) else [Path(item) for item in state_dirs]
    )

    deduped: dict[str, dict[str, Any]] = {}
    for state_dir in normalized_state_dirs:
        for example in _collect_examples_for_state_dir(state_dir.resolve(), version=version):
            deduped[example["example_id"]] = example
    examples = sorted(deduped.values(), key=lambda item: item["example_id"])

    master_path = output_dir / "tar_master_dataset.jsonl"
    train_path = output_dir / "tar_master_dataset_train.jsonl"
    validation_path = output_dir / "tar_master_dataset_validation.jsonl"
    test_path = output_dir / "tar_master_dataset_test.jsonl"

    split_examples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    task_families = Counter()
    source_kinds = Counter()

    with master_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            task_families[example["task_family"]] += 1
            source_kinds[example["source_kind"]] += 1
            split_examples[example["split"]].append(example)
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    for path, split_name in (
        (train_path, "train"),
        (validation_path, "validation"),
        (test_path, "test"),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for example in split_examples[split_name]:
                handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    manifest = {
        "dataset_version": version,
        "state_dirs": [str(path.resolve()) for path in normalized_state_dirs],
        "output_dir": str(output_dir),
        "records": len(examples),
        "splits": {name: len(items) for name, items in split_examples.items()},
        "task_families": dict(sorted(task_families.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
        "files": {
            "master": str(master_path),
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    state_dirs = [Path(item).resolve() for item in (args.state_dirs or ["tar_state"])]
    output_dir = Path(args.output_dir).resolve()
    manifest = build_master_dataset(state_dirs, output_dir, version=args.version)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
