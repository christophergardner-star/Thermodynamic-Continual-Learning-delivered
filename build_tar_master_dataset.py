from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SYSTEM_PROMPT = (
    "You are TAR, a disciplined research operator. Separate measured results, "
    "inferences, hypotheses, blockers, and next actions. Stay honest when "
    "benchmark truth, reproducibility, or capability is incomplete. Return "
    "only a JSON object and do not include prose or markdown fences."
)

STATE_ARTIFACT_FILES = [
    "problem_studies.jsonl",
    "problem_executions.jsonl",
    "verification_reports.jsonl",
    "research_decisions.jsonl",
    "research_projects.json",
    "claim_verdicts.jsonl",
    "falsification_plans.jsonl",
    "project_priority_records.jsonl",
    "evidence_debt_records.jsonl",
    "project_staleness_records.jsonl",
    "portfolio_decisions.jsonl",
    "inference_endpoints.json",
    "recovery.json",
    "recovery_history.jsonl",
]


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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _derive_tcl_failure_mode(
    *,
    regime: dict[str, Any],
    metrics: dict[str, Any],
    governor_action: Any,
    governor_reasons: Any,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    drift_rho = _safe_float(metrics.get("drift_rho")) or 0.0
    dimensionality_ratio = _safe_float(metrics.get("dimensionality_ratio")) or 0.0
    equilibrium_fraction = _safe_float(metrics.get("equilibrium_fraction")) or 0.0
    training_loss = _safe_float(metrics.get("training_loss"))
    calibration_ece = _safe_float((calibration or {}).get("ece"))
    reasons = {str(item) for item in _ensure_list(governor_reasons) if item is not None}
    regime_name = str(regime.get("regime") or "searching")

    failure_mode = "stable_control"
    severity = "low"
    primary_signal = "equilibrium_gate"
    claim_promotion_safe = bool(metrics.get("equilibrium_gate")) and (calibration_ece or 0.0) <= 0.10

    if (
        dimensionality_ratio > 0.0
        and training_loss is not None
        and training_loss <= 0.80
        and dimensionality_ratio < 0.55
    ):
        failure_mode = "dimensionality_collapse"
        severity = "high"
        primary_signal = "dimensionality_ratio"
        claim_promotion_safe = False
    elif drift_rho >= 0.12 or "weight_drift_limit" in reasons:
        failure_mode = "drift_instability"
        severity = "high"
        primary_signal = "drift_rho"
        claim_promotion_safe = False
    elif calibration_ece is not None and calibration_ece >= 0.16:
        failure_mode = "calibration_degradation"
        severity = "medium"
        primary_signal = "calibration_ece"
        claim_promotion_safe = False
    elif regime_name == "warming_up" or equilibrium_fraction < 0.20:
        failure_mode = "warmup_incomplete"
        severity = "medium"
        primary_signal = "equilibrium_fraction"
        claim_promotion_safe = False
    elif str(governor_action or "") == "terminate":
        failure_mode = "governor_termination"
        severity = "medium"
        primary_signal = "governor_action"
        claim_promotion_safe = False

    return {
        "failure_mode": failure_mode,
        "severity": severity,
        "primary_signal": primary_signal,
        "claim_promotion_safe": claim_promotion_safe,
    }


def _derive_tcl_anchor_policy(
    *,
    regime: dict[str, Any],
    metrics: dict[str, Any],
    anchor_effective_dimensionality: Any,
    governor_reasons: Any,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_dimensionality = _safe_float(metrics.get("effective_dimensionality"))
    anchor_dimensionality = _safe_float(anchor_effective_dimensionality)
    reasons = {str(item) for item in _ensure_list(governor_reasons) if item is not None}
    regime_name = str(regime.get("regime") or "searching")

    rationale_signals: list[str] = []
    anchor_policy = "hold_anchor_constant"
    anchor_reuse_recommended = bool((recovery or {}).get("last_anchor_path"))

    if anchor_dimensionality is None or anchor_dimensionality <= 0.0:
        anchor_policy = "anchor_absent_review"
        anchor_reuse_recommended = False
        rationale_signals.append("anchor_missing")
    elif regime_name == "thermodynamic_quenching" or "weight_drift_limit" in reasons:
        anchor_policy = "reset_anchor_state"
        anchor_reuse_recommended = False
        rationale_signals.append("quenching_or_drift_limit")
    elif regime_name in {"equilibrium", "stabilizing"} and current_dimensionality is not None and anchor_dimensionality >= current_dimensionality:
        anchor_policy = "reuse_last_stable_anchor"
        anchor_reuse_recommended = True
        rationale_signals.extend(["stable_regime", "anchor_capacity_available"])
    elif regime_name == "warming_up":
        anchor_policy = "suppress_anchor_changes_during_warmup"
        anchor_reuse_recommended = False
        rationale_signals.append("warmup_incomplete")
    else:
        rationale_signals.append("monitor_anchor_pressure")

    return {
        "anchor_policy": anchor_policy,
        "anchor_reuse_recommended": anchor_reuse_recommended,
        "rationale_signals": rationale_signals[:3],
    }


def _derive_tcl_trace_anomaly(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    drift_values = [_safe_float(item.get("drift_rho")) for item in metrics]
    dim_ratios = [_safe_float(item.get("dimensionality_ratio")) for item in metrics]
    equilibrium_values = [_safe_float(item.get("equilibrium_fraction")) for item in metrics]
    losses = [_safe_float(item.get("training_loss")) for item in metrics]

    anomaly_flags: list[str] = []
    drift_clean = [value for value in drift_values if value is not None]
    ratio_clean = [value for value in dim_ratios if value is not None]
    equilibrium_clean = [value for value in equilibrium_values if value is not None]
    losses_clean = [value for value in losses if value is not None]

    if drift_clean and max(drift_clean) >= 0.12:
        anomaly_flags.append("drift_spike")
    if ratio_clean and losses_clean and min(ratio_clean) < 0.55 and max(losses_clean) <= 0.85:
        anomaly_flags.append("quenching_signature")
    if equilibrium_clean and max(equilibrium_clean) >= 0.50 and equilibrium_clean[-1] <= max(equilibrium_clean) - 0.25:
        anomaly_flags.append("equilibrium_backslide")
    if len(losses_clean) >= 2 and losses_clean[-1] > losses_clean[0] * 1.15:
        anomaly_flags.append("loss_instability")

    dominant_priority = (
        "quenching_signature",
        "drift_spike",
        "equilibrium_backslide",
        "loss_instability",
    )
    dominant_anomaly = "none"
    for label in dominant_priority:
        if label in anomaly_flags:
            dominant_anomaly = label
            break

    return {
        "anomaly_present": bool(anomaly_flags),
        "dominant_anomaly": dominant_anomaly,
        "anomaly_flags": anomaly_flags,
    }


def _derive_tcl_transition_forecast(
    *,
    first_metric: dict[str, Any],
    last_metric: dict[str, Any],
    regime: dict[str, Any],
    anomaly: dict[str, Any],
) -> dict[str, Any]:
    d_pr_trend = _trend_label(
        first_metric.get("effective_dimensionality"),
        last_metric.get("effective_dimensionality"),
    )
    equilibrium_trend = _trend_label(
        first_metric.get("equilibrium_fraction"),
        last_metric.get("equilibrium_fraction"),
    )
    drift_trend = _trend_label(
        first_metric.get("drift_rho"),
        last_metric.get("drift_rho"),
    )
    regime_name = str(regime.get("regime") or "searching")

    predicted_next_regime = regime_name
    intervention_urgency = "medium"
    if anomaly.get("dominant_anomaly") in {"quenching_signature", "drift_spike"} or d_pr_trend == "collapsing":
        predicted_next_regime = "thermodynamic_quenching"
        intervention_urgency = "high"
    elif regime_name == "warming_up" and equilibrium_trend == "increasing":
        predicted_next_regime = "stabilizing"
        intervention_urgency = "medium"
    elif regime_name in {"searching", "stabilizing"} and equilibrium_trend == "increasing" and drift_trend != "increasing":
        predicted_next_regime = "equilibrium"
        intervention_urgency = "medium"
    elif regime_name == "equilibrium":
        intervention_urgency = "low"

    known_trends = sum(label != "unknown" for label in (d_pr_trend, equilibrium_trend, drift_trend))
    confidence_band = "high" if known_trends == 3 else "medium" if known_trends >= 2 else "low"
    return {
        "predicted_next_regime": predicted_next_regime,
        "intervention_urgency": intervention_urgency,
        "confidence_band": confidence_band,
    }


def _derive_tcl_intervention_policy(
    *,
    regime: dict[str, Any],
    failure_mode: dict[str, Any],
    anomaly: dict[str, Any],
    has_fallback_data: bool,
) -> dict[str, Any]:
    regime_name = str(regime.get("regime") or "searching")
    failure_label = str(failure_mode.get("failure_mode") or "stable_control")
    dominant_anomaly = str(anomaly.get("dominant_anomaly") or "none")

    recommended_tcl_action = "continue_monitoring"
    intervention_reason = "no_high_risk_signal"
    claim_promotion_safe = False

    if failure_label == "dimensionality_collapse" or dominant_anomaly == "quenching_signature":
        recommended_tcl_action = "reduce_anchor_pressure_and_run_short_recovery_probe"
        intervention_reason = "quenching_signature"
    elif failure_label == "drift_instability" or dominant_anomaly == "drift_spike":
        recommended_tcl_action = "tighten_governor_and_debug_drift_limit"
        intervention_reason = "drift_instability"
    elif failure_label == "calibration_degradation":
        recommended_tcl_action = "run_calibration_repair_before_promotion"
        intervention_reason = "calibration_gap"
    elif failure_label == "warmup_incomplete":
        recommended_tcl_action = "continue_statistical_warmup"
        intervention_reason = "warmup_incomplete"
    elif regime_name == "equilibrium" and not has_fallback_data and not anomaly.get("anomaly_present"):
        recommended_tcl_action = "replicate_before_promotion"
        intervention_reason = "equilibrium_without_fallback"
        claim_promotion_safe = True

    return {
        "recommended_tcl_action": recommended_tcl_action,
        "intervention_reason": intervention_reason,
        "claim_promotion_safe": claim_promotion_safe,
    }


def _derive_tcl_recovery_confidence(recovery: dict[str, Any]) -> dict[str, Any]:
    status = str(recovery.get("status") or "unknown")
    consecutive_fail_fast = int(recovery.get("consecutive_fail_fast") or 0)
    stable_hyperparameters_available = bool(recovery.get("last_known_stable_hyperparameters"))
    achieved_dimensionality = _safe_float(recovery.get("max_effective_dimensionality_achieved")) or 0.0

    recovery_outlook = "uncertain"
    resume_confidence_band = "low"
    requires_human_review = True
    if status == "completed" and stable_hyperparameters_available:
        recovery_outlook = "strong"
        resume_confidence_band = "high"
        requires_human_review = False
    elif status == "fail_fast" and consecutive_fail_fast >= 3:
        recovery_outlook = "poor"
        resume_confidence_band = "low"
    elif status == "pivoted":
        recovery_outlook = "guarded"
        resume_confidence_band = "medium"
    elif stable_hyperparameters_available and achieved_dimensionality >= 7.0:
        recovery_outlook = "recoverable"
        resume_confidence_band = "medium"
        requires_human_review = False

    return {
        "recovery_outlook": recovery_outlook,
        "resume_confidence_band": resume_confidence_band,
        "requires_human_review": requires_human_review,
    }


def _derive_tcl_run_triage(
    recovery: dict[str, Any], *, recovery_confidence: dict[str, Any]
) -> dict[str, Any]:
    status = str(recovery.get("status") or "unknown")
    consecutive_fail_fast = int(recovery.get("consecutive_fail_fast") or 0)

    operator_decision = "inspect_recovery_state"
    urgency = "medium"
    if status == "completed" and recovery_confidence.get("resume_confidence_band") == "high":
        operator_decision = "resume_controlled"
        urgency = "medium"
    elif status == "fail_fast" and consecutive_fail_fast >= 3:
        operator_decision = "pivot_or_terminate"
        urgency = "high"
    elif status == "fail_fast":
        operator_decision = "debug_before_resume"
        urgency = "high"
    elif status == "pivoted":
        operator_decision = "prepare_alternative_strategy"
        urgency = "medium"

    return {
        "operator_decision": operator_decision,
        "urgency": urgency,
        "human_review_required": bool(recovery_confidence.get("requires_human_review")),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _lineage_key(
    *,
    project_id: Any = None,
    thread_id: Any = None,
    problem_id: Any = None,
    trial_id: Any = None,
    endpoint_name: Any = None,
    manifest_id: Any = None,
    fallback: Any = None,
) -> str:
    if project_id:
        return f"project:{project_id}"
    if thread_id:
        return f"thread:{thread_id}"
    if problem_id:
        return f"problem:{problem_id}"
    if trial_id:
        return f"trial:{trial_id}"
    if endpoint_name:
        return f"endpoint:{endpoint_name}"
    if manifest_id:
        return f"manifest:{manifest_id}"
    return f"source:{fallback or 'unknown'}"


def _hash_split(lineage_key: str) -> str:
    bucket = int(hashlib.sha256(lineage_key.encode("utf-8")).hexdigest()[:8], 16) % 100
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


def _task_contract_suffix(task_family: str) -> str:
    if task_family == "project_resume":
        return (
            "Return exactly this compact JSON schema and no nested objects beyond the top level: "
            "{"
            '"budget_pressure_level", '
            '"active_thread_id", '
            '"current_question_id", '
            '"next_action_id", '
            '"next_action_kind", '
            '"next_action_status"'
            "}."
        )
    return "Return only a concise JSON object aligned to the target."


def _render_user_prompt(task_family: str, task_name: str, input_context: dict[str, Any]) -> str:
    payload = json.dumps(input_context, indent=2, sort_keys=True)
    return (
        f"Task family: {task_family}\n"
        f"Task: {task_name}\n"
        f"Use the TAR operator contract. {_task_contract_suffix(task_family)} "
        "Do not add explanatory prose or markdown fences.\n\n"
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
    lineage_key: str,
    input_context: dict[str, Any],
    target: dict[str, Any],
    repo_root: Path,
    state_dir: Path,
    tags: Iterable[str] = (),
    observed: bool = True,
) -> dict[str, Any]:
    example_id = f"{task_family}:{task_name}:{source_kind}:{source_id}"
    safe_input = _sanitize_value(input_context, repo_root=repo_root, state_dir=state_dir)
    safe_target = _sanitize_value(target, repo_root=repo_root, state_dir=state_dir)
    content_hash = _sha256_text(
        json.dumps(
            {
                "task_family": task_family,
                "task_name": task_name,
                "source_kind": source_kind,
                "source_id": source_id,
                "lineage_key": lineage_key,
                "input_context": safe_input,
                "target": safe_target,
            },
            sort_keys=True,
        )
    )
    return {
        "example_id": example_id,
        "dedupe_key": example_id,
        "dataset_version": version,
        "lineage_key": lineage_key,
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
            "state_root": _relative_or_name(state_dir, repo_root),
            "observed": observed,
            "content_hash": content_hash,
        },
    }


def _examples_from_problem_study(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    problem_id = str(record.get("problem_id") or record.get("problem") or "unknown-problem")
    lineage = _lineage_key(
        project_id=record.get("project_id"),
        thread_id=record.get("thread_id"),
        problem_id=problem_id,
        fallback=problem_id,
    )
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
            lineage_key=lineage,
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
                lineage_key=lineage,
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
    if (
        environment
        and (
            not bool(environment.get("reproducibility_complete"))
            or unresolved_packages
            or environment.get("lock_incomplete_reason")
        )
    ):
        examples.append(
            _example_record(
                version=version,
                task_family="reproducibility_refusal",
                task_name="study_environment_to_lock_refusal",
                source_kind="problem_study",
                source_id=f"{problem_id}:repro",
                source_file="problem_studies.jsonl",
                lineage_key=lineage,
                input_context={
                    "problem": record.get("problem"),
                    "requested_benchmark_tier": record.get("benchmark_tier"),
                    "benchmark_alignment": benchmark_alignment,
                    "reproducibility_complete": environment.get("reproducibility_complete"),
                    "unresolved_packages": unresolved_packages,
                    "lock_incomplete_reason": environment.get("lock_incomplete_reason"),
                },
                target={
                    "should_refuse_promotion": True,
                    "operator_language": "reproducibility_incomplete_refuse_or_downgrade",
                    "next_action": record.get("next_action"),
                },
                repo_root=repo_root,
                state_dir=state_dir,
                tags=("reproducibility", "refusal"),
            )
        )
    return examples


def _examples_from_problem_execution(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    problem_id = str(record.get("problem_id") or record.get("problem") or "unknown-problem")
    lineage = _lineage_key(
        project_id=record.get("project_id"),
        thread_id=record.get("thread_id"),
        problem_id=problem_id,
        fallback=problem_id,
    )
    examples = [
        _example_record(
            version=version,
            task_family="execution_diagnosis",
            task_name="execution_report_to_recovery_action",
            source_kind="problem_execution",
            source_id=problem_id,
            source_file="problem_executions.jsonl",
            lineage_key=lineage,
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
    sandbox_policy = record.get("sandbox_policy") if isinstance(record.get("sandbox_policy"), dict) else {}
    if sandbox_policy:
        writable_mounts = _ensure_list(sandbox_policy.get("writable_mounts"))
        examples.append(
            _example_record(
                version=version,
                task_family="sandbox_policy_reasoning",
                task_name="execution_report_sandbox_to_write_scope_assessment",
                source_kind="problem_execution",
                source_id=f"{problem_id}:sandbox",
                source_file="problem_executions.jsonl",
                lineage_key=lineage,
                input_context={
                    "problem": record.get("problem"),
                    "status": record.get("status"),
                    "execution_mode": record.get("execution_mode"),
                    "sandbox_mode": sandbox_policy.get("mode"),
                    "profile": sandbox_policy.get("profile"),
                    "network_policy": sandbox_policy.get("network_policy"),
                    "read_only_mounts": _ensure_list(sandbox_policy.get("read_only_mounts"))[:8],
                    "writable_mounts": writable_mounts[:8],
                    "artifact_dir": sandbox_policy.get("artifact_dir"),
                    "workspace_root": sandbox_policy.get("workspace_root"),
                },
                target={
                    "artifact_only_write_scope": bool(sandbox_policy.get("artifact_dir"))
                    and bool(writable_mounts)
                    and all(str(item).startswith(str(sandbox_policy.get("artifact_dir"))) for item in writable_mounts),
                    "network_policy": sandbox_policy.get("network_policy"),
                    "operator_assessment": (
                        "sandbox_ok_for_production"
                        if sandbox_policy.get("profile") == "production"
                        and sandbox_policy.get("network_policy") == "off"
                        else "sandbox_requires_operator_review"
                    ),
                },
                repo_root=repo_root,
                state_dir=state_dir,
                tags=("sandbox", "runtime", "policy"),
            )
        )
    return examples


def _examples_from_verification_report(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    trial_id = str(record.get("trial_id") or "unknown-trial")
    lineage = _lineage_key(trial_id=trial_id, fallback=trial_id)
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
            lineage_key=lineage,
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
    lineage = _lineage_key(
        thread_id=record.get("thread_id"),
        problem_id=record.get("problem_id"),
        trial_id=record.get("trial_id"),
        fallback=decision_id,
    )
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
            lineage_key=lineage,
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
    lineage = _lineage_key(project_id=project_id, fallback=project_id)
    resume_snapshot = record.get("resume_snapshot") if isinstance(record.get("resume_snapshot"), dict) else {}
    budget = record.get("budget_ledger") if isinstance(record.get("budget_ledger"), dict) else {}
    next_action = None
    for item in _ensure_list(record.get("planned_actions")):
        if isinstance(item, dict) and item.get("action_id") == resume_snapshot.get("next_action_id"):
            next_action = item
            break
    compact_resume_target = {
        "budget_pressure_level": budget.get("budget_pressure_level"),
        "active_thread_id": resume_snapshot.get("active_thread_id") or record.get("active_thread_id"),
        "current_question_id": resume_snapshot.get("current_question_id"),
        "next_action_id": resume_snapshot.get("next_action_id"),
        "next_action_kind": (next_action or {}).get("action_kind"),
        "next_action_status": (next_action or {}).get("status"),
    }
    return [
        _example_record(
            version=version,
            task_family="project_resume",
            task_name="project_state_to_resume_snapshot",
            source_kind="research_project",
            source_id=project_id,
            source_file="research_projects.json",
            lineage_key=lineage,
            input_context={
                "title": record.get("title"),
                "goal": record.get("goal"),
                "status": record.get("status"),
                "latest_decision_summary": record.get("latest_decision_summary"),
                "budget_pressure_level": budget.get("budget_pressure_level"),
                "resume_state": {
                    "active_thread_id": compact_resume_target["active_thread_id"],
                    "current_question_id": compact_resume_target["current_question_id"],
                    "next_action_id": compact_resume_target["next_action_id"],
                },
                "next_action_state": {
                    "action_kind": compact_resume_target["next_action_kind"],
                    "status": compact_resume_target["next_action_status"],
                },
                "blockers": resume_snapshot.get("blockers"),
                "budget_remaining_summary": resume_snapshot.get("budget_remaining_summary"),
                "latest_evidence_summary": resume_snapshot.get("latest_evidence_summary"),
            },
            target=compact_resume_target,
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("continuity", "resume"),
        )
    ]


def _examples_from_falsification_plan(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    plan_id = str(record.get("plan_id") or "unknown-plan")
    lineage = _lineage_key(
        project_id=record.get("project_id"),
        thread_id=record.get("thread_id"),
        fallback=plan_id,
    )
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
            lineage_key=lineage,
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
    lineage = _lineage_key(project_id=record.get("project_id"), fallback=record_id)
    return [
        _example_record(
            version=version,
            task_family="prioritization",
            task_name="project_priority_record_to_recommendation",
            source_kind="project_priority_record",
            source_id=record_id,
            source_file="project_priority_records.jsonl",
            lineage_key=lineage,
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
    lineage = _lineage_key(
        project_id=record.get("selected_project_id"),
        fallback=decision_id,
    )
    return [
        _example_record(
            version=version,
            task_family="portfolio_governance",
            task_name="portfolio_decision_to_budget_allocation",
            source_kind="portfolio_decision",
            source_id=decision_id,
            source_file="portfolio_decisions.jsonl",
            lineage_key=lineage,
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


def _examples_from_claim_verdict(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    verdict_id = str(record.get("verdict_id") or record.get("trial_id") or "unknown-verdict")
    lineage = _lineage_key(
        problem_id=record.get("benchmark_problem_id"),
        trial_id=record.get("trial_id"),
        fallback=verdict_id,
    )
    linkage_status = str(record.get("linkage_status") or "none")
    canonical_required = bool(record.get("canonical_benchmark_required"))
    canonical_satisfied = bool(record.get("canonical_benchmark_satisfied"))
    verdict_status = str(record.get("status") or "insufficient_evidence")
    lineage_ok = linkage_status == "exact" and bool(record.get("verdict_inputs_complete"))
    if verdict_status in {"accepted", "provisional"} and canonical_required and not canonical_satisfied:
        operator_language = "promotion_blocked_missing_canonical_support"
    elif not lineage_ok:
        operator_language = "lineage_incomplete_or_ambiguous"
    elif verdict_status == "contradicted":
        operator_language = "claim_under_active_contradiction"
    else:
        operator_language = "lineage_ready_for_review"
    return [
        _example_record(
            version=version,
            task_family="claim_lineage_audit",
            task_name="claim_verdict_to_lineage_audit",
            source_kind="claim_verdict",
            source_id=verdict_id,
            source_file="claim_verdicts.jsonl",
            lineage_key=lineage,
            input_context={
                "trial_id": record.get("trial_id"),
                "status": verdict_status,
                "benchmark_problem_id": record.get("benchmark_problem_id"),
                "benchmark_execution_mode": record.get("benchmark_execution_mode"),
                "supporting_benchmark_ids": _ensure_list(record.get("supporting_benchmark_ids"))[:6],
                "supporting_benchmark_names": _ensure_list(record.get("supporting_benchmark_names"))[:6],
                "linkage_status": linkage_status,
                "linkage_note": record.get("linkage_note"),
                "canonical_comparability_source": record.get("canonical_comparability_source"),
                "verdict_inputs_complete": record.get("verdict_inputs_complete"),
                "canonical_benchmark_required": canonical_required,
                "canonical_benchmark_satisfied": canonical_satisfied,
                "confidence": record.get("confidence"),
            },
            target={
                "lineage_ok": lineage_ok,
                "canonical_support_ok": (not canonical_required) or canonical_satisfied,
                "operator_language": operator_language,
                "recommended_audit_action": (
                    "review_claim_for_promotion"
                    if lineage_ok and ((not canonical_required) or canonical_satisfied)
                    else "hold_claim_and_recheck_lineage"
                ),
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("claims", "lineage", "benchmark_truth"),
        )
    ]


def _examples_from_evidence_debt_record(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    record_id = str(record.get("record_id") or record.get("project_id") or "unknown-evidence-debt")
    lineage = _lineage_key(project_id=record.get("project_id"), fallback=record_id)
    debt_components = {
        "falsification_gap": float(record.get("falsification_gap") or 0.0),
        "replication_gap": float(record.get("replication_gap") or 0.0),
        "benchmark_gap": float(record.get("benchmark_gap") or 0.0),
        "claim_linkage_gap": float(record.get("claim_linkage_gap") or 0.0),
        "calibration_gap": float(record.get("calibration_gap") or 0.0),
    }
    ordered_gaps = [
        key
        for key, value in sorted(debt_components.items(), key=lambda item: (-item[1], item[0]))
        if value > 0.0
    ]
    overall_debt = float(record.get("overall_debt") or 0.0)
    promotion_blocked = bool(record.get("promotion_blocked"))
    return [
        _example_record(
            version=version,
            task_family="evidence_debt_judgement",
            task_name="evidence_debt_record_to_promotion_gate",
            source_kind="evidence_debt_record",
            source_id=record_id,
            source_file="evidence_debt_records.jsonl",
            lineage_key=lineage,
            input_context={
                "project_id": record.get("project_id"),
                **debt_components,
                "overall_debt": overall_debt,
                "promotion_blocked": promotion_blocked,
                "rationale": _ensure_list(record.get("rationale"))[:6],
            },
            target={
                "promotion_gate": "blocked" if promotion_blocked else "open",
                "recommended_state": "defer" if promotion_blocked or overall_debt >= 0.45 else "continue",
                "primary_gaps": ordered_gaps[:3],
                "operator_language": (
                    "under_proved_result_requires_more_evidence"
                    if promotion_blocked
                    else "evidence_debt_present_but_not_blocking"
                ),
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("evidence", "promotion_gate", "portfolio"),
        )
    ]


def _examples_from_project_staleness_record(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    record_id = str(record.get("record_id") or record.get("project_id") or "unknown-staleness")
    lineage = _lineage_key(project_id=record.get("project_id"), fallback=record_id)
    level = str(record.get("staleness_level") or "fresh")
    resume_candidate = bool(record.get("resume_candidate"))
    closure_candidate = bool(record.get("closure_candidate"))
    if closure_candidate:
        next_action = "consider_retire_or_close"
    elif resume_candidate:
        next_action = "surface_for_resume_review"
    elif level in {"stale", "critical"}:
        next_action = "inspect_for_blockers_or_reprioritize"
    else:
        next_action = "continue_monitoring"
    return [
        _example_record(
            version=version,
            task_family="portfolio_staleness_recovery",
            task_name="staleness_record_to_resume_or_retire_action",
            source_kind="project_staleness_record",
            source_id=record_id,
            source_file="project_staleness_records.jsonl",
            lineage_key=lineage,
            input_context={
                "project_id": record.get("project_id"),
                "last_progress_at": record.get("last_progress_at"),
                "hours_since_progress": record.get("hours_since_progress"),
                "staleness_level": level,
                "reason": record.get("reason"),
                "resume_candidate": resume_candidate,
                "closure_candidate": closure_candidate,
            },
            target={
                "recommended_operator_action": next_action,
                "resume_candidate": resume_candidate,
                "closure_candidate": closure_candidate,
                "staleness_level": level,
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("portfolio", "staleness", "resume"),
        )
    ]


def _examples_from_endpoint_record(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    endpoint_name = str(record.get("endpoint_name") or "unknown-endpoint")
    lineage = _lineage_key(endpoint_name=endpoint_name, fallback=endpoint_name)
    health = record.get("health") if isinstance(record.get("health"), dict) else {}
    inspect_paths = [
        value
        for value in (
            record.get("manifest_path"),
            record.get("stdout_log_path"),
            record.get("stderr_log_path"),
        )
        if value
    ]
    diagnosis = (
        "failed_endpoint_requires_log_inspection"
        if record.get("status") == "failed"
        else "unhealthy_endpoint_requires_health_review"
        if health and not health.get("ok", False)
        else "endpoint_observability_ready"
    )
    return [
        _example_record(
            version=version,
            task_family="endpoint_observability_diagnosis",
            task_name="endpoint_record_to_operator_diagnosis",
            source_kind="endpoint_record",
            source_id=endpoint_name,
            source_file="inference_endpoints.json",
            lineage_key=lineage,
            input_context={
                "endpoint_name": endpoint_name,
                "checkpoint_name": record.get("checkpoint_name"),
                "role": record.get("role"),
                "backend": record.get("backend"),
                "status": record.get("status"),
                "last_error": record.get("last_error"),
                "last_health_at": record.get("last_health_at"),
                "trust_remote_code": record.get("trust_remote_code"),
                "health": {
                    "status": health.get("status"),
                    "ok": health.get("ok"),
                    "detail": health.get("detail"),
                    "http_status": health.get("http_status"),
                },
                "inspect_paths": inspect_paths,
            },
            target={
                "diagnosis": diagnosis,
                "restart_recommended": record.get("status") in {"failed", "stopped"} or (health and not health.get("ok", False)),
                "inspect_paths": inspect_paths,
                "trust_policy": "remote_code_enabled" if record.get("trust_remote_code") else "remote_code_disabled",
            },
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("endpoints", "observability", "ops"),
        )
    ]


def _examples_from_run_manifest(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path, source_file: str
) -> list[dict[str, Any]]:
    manifest_id = str(record.get("manifest_id") or "unknown-manifest")
    lineage = _lineage_key(
        problem_id=record.get("problem_id"),
        trial_id=record.get("trial_id"),
        manifest_id=manifest_id,
        fallback=manifest_id,
    )
    sandbox_policy = record.get("sandbox_policy") if isinstance(record.get("sandbox_policy"), dict) else {}
    examples: list[dict[str, Any]] = []
    if (
        not bool(record.get("reproducibility_complete"))
        or _ensure_list(record.get("unresolved_packages"))
        or record.get("lock_incomplete_reason")
    ):
        examples.append(
            _example_record(
                version=version,
                task_family="reproducibility_refusal",
                task_name="run_manifest_to_lock_refusal",
                source_kind="run_manifest",
                source_id=manifest_id,
                source_file=source_file,
                lineage_key=lineage,
                input_context={
                    "kind": record.get("kind"),
                    "trial_id": record.get("trial_id"),
                    "problem_id": record.get("problem_id"),
                    "reproducibility_complete": record.get("reproducibility_complete"),
                    "unresolved_packages": _ensure_list(record.get("unresolved_packages"))[:12],
                    "lock_incomplete_reason": record.get("lock_incomplete_reason"),
                },
                target={
                    "should_refuse_promotion": True,
                    "operator_language": "manifest_lock_incomplete_refuse_or_downgrade",
                    "next_action": "pin_dependencies_and_rebuild_manifest",
                },
                repo_root=repo_root,
                state_dir=state_dir,
                tags=("reproducibility", "lock_refusal"),
            )
        )
    if sandbox_policy:
        writable_mounts = _ensure_list(sandbox_policy.get("writable_mounts"))
        examples.append(
            _example_record(
                version=version,
                task_family="sandbox_policy_reasoning",
                task_name="run_manifest_sandbox_to_write_scope_assessment",
                source_kind="run_manifest",
                source_id=f"{manifest_id}:sandbox",
                source_file=source_file,
                lineage_key=lineage,
                input_context={
                    "kind": record.get("kind"),
                    "sandbox_mode": sandbox_policy.get("mode"),
                    "profile": sandbox_policy.get("profile"),
                    "network_policy": sandbox_policy.get("network_policy"),
                    "read_only_mounts": _ensure_list(sandbox_policy.get("read_only_mounts"))[:8],
                    "writable_mounts": writable_mounts[:8],
                    "artifact_dir": sandbox_policy.get("artifact_dir"),
                    "workspace_root": sandbox_policy.get("workspace_root"),
                },
                target={
                    "artifact_only_write_scope": bool(sandbox_policy.get("artifact_dir"))
                    and bool(writable_mounts)
                    and all(str(item).startswith(str(sandbox_policy.get("artifact_dir"))) for item in writable_mounts),
                    "network_policy": sandbox_policy.get("network_policy"),
                    "operator_assessment": (
                        "production_sandbox_ok"
                        if sandbox_policy.get("profile") == "production"
                        and sandbox_policy.get("network_policy") == "off"
                        else "non_production_or_broad_sandbox_requires_review"
                    ),
                },
                repo_root=repo_root,
                state_dir=state_dir,
                tags=("sandbox", "runtime", "policy"),
            )
        )
    return examples


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
    lineage = _lineage_key(trial_id=trial_id, fallback=trial_id)
    failure_mode = _derive_tcl_failure_mode(
        regime=regime,
        metrics=metrics,
        governor_action=summary.get("governor_action"),
        governor_reasons=summary.get("governor_reasons"),
        calibration=summary.get("calibration") if isinstance(summary.get("calibration"), dict) else {},
    )
    anchor_policy = _derive_tcl_anchor_policy(
        regime=regime,
        metrics=metrics,
        anchor_effective_dimensionality=summary.get("anchor_effective_dimensionality"),
        governor_reasons=summary.get("governor_reasons"),
    )
    intervention_policy = _derive_tcl_intervention_policy(
        regime=regime,
        failure_mode=failure_mode,
        anomaly={"anomaly_present": False, "dominant_anomaly": "none", "anomaly_flags": []},
        has_fallback_data=bool(data_provenance.get("has_fallback")),
    )
    return [
        _example_record(
            version=version,
            task_family="tcl_regime_diagnosis",
            task_name="payload_summary_to_regime_assessment",
            source_kind="tcl_payload_summary",
            source_id=trial_id,
            source_file=str(source_file),
            lineage_key=lineage,
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
        ),
        _example_record(
            version=version,
            task_family="tcl_failure_mode_classification",
            task_name="payload_summary_to_failure_mode",
            source_kind="tcl_payload_summary",
            source_id=f"{trial_id}:failure_mode",
            source_file=str(source_file),
            lineage_key=lineage,
            input_context={
                "trial_id": trial_id,
                "regime": regime.get("regime"),
                "warning": regime.get("warning"),
                "governor_action": summary.get("governor_action"),
                "governor_reasons": _ensure_list(summary.get("governor_reasons"))[:6],
                "last_metrics": metrics,
                "calibration": summary.get("calibration"),
            },
            target=failure_mode,
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "diagnosis", "failure_mode"),
        ),
        _example_record(
            version=version,
            task_family="tcl_anchor_policy_judgement",
            task_name="payload_summary_to_anchor_policy",
            source_kind="tcl_payload_summary",
            source_id=f"{trial_id}:anchor_policy",
            source_file=str(source_file),
            lineage_key=lineage,
            input_context={
                "trial_id": trial_id,
                "regime": regime.get("regime"),
                "warning": regime.get("warning"),
                "governor_reasons": _ensure_list(summary.get("governor_reasons"))[:6],
                "anchor_effective_dimensionality": summary.get("anchor_effective_dimensionality"),
                "last_metrics": metrics,
            },
            target=anchor_policy,
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "anchor", "policy"),
        ),
        _example_record(
            version=version,
            task_family="tcl_intervention_selection",
            task_name="payload_summary_to_intervention_selection",
            source_kind="tcl_payload_summary",
            source_id=f"{trial_id}:intervention",
            source_file=str(source_file),
            lineage_key=lineage,
            input_context={
                "trial_id": trial_id,
                "regime": regime.get("regime"),
                "warning": regime.get("warning"),
                "failure_mode": failure_mode.get("failure_mode"),
                "severity": failure_mode.get("severity"),
                "governor_action": summary.get("governor_action"),
                "governor_reasons": _ensure_list(summary.get("governor_reasons"))[:6],
                "has_fallback_data": data_provenance.get("has_fallback"),
                "calibration": summary.get("calibration"),
            },
            target=intervention_policy,
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "intervention", "operator"),
        ),
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
    lineage = _lineage_key(trial_id=trial_id, fallback=trial_id)
    anomaly = _derive_tcl_trace_anomaly(metrics)
    transition_forecast = _derive_tcl_transition_forecast(
        first_metric=first_metric,
        last_metric=last_metric,
        regime=regime,
        anomaly=anomaly,
    )
    return [
        _example_record(
            version=version,
            task_family="tcl_trace_analysis",
            task_name="thermo_trace_to_dynamics_summary",
            source_kind="tcl_thermo_trace",
            source_id=trial_id,
            source_file=str(source_file),
            lineage_key=lineage,
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
        ),
        _example_record(
            version=version,
            task_family="tcl_trace_anomaly_diagnosis",
            task_name="thermo_trace_to_anomaly_report",
            source_kind="tcl_thermo_trace",
            source_id=f"{trial_id}:anomaly",
            source_file=str(source_file),
            lineage_key=lineage,
            input_context={
                "trial_id": trial_id,
                "num_steps": len(metrics),
                "first_metric": first_metric,
                "last_metric": last_metric,
                "governor_thresholds": thresholds,
            },
            target=anomaly,
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "trace", "anomaly"),
        ),
        _example_record(
            version=version,
            task_family="tcl_regime_transition_forecast",
            task_name="thermo_trace_to_regime_forecast",
            source_kind="tcl_thermo_trace",
            source_id=f"{trial_id}:forecast",
            source_file=str(source_file),
            lineage_key=lineage,
            input_context={
                "trial_id": trial_id,
                "num_steps": len(metrics),
                "first_metric": first_metric,
                "last_metric": last_metric,
                "current_regime": regime.get("regime"),
                "anomaly": anomaly,
            },
            target=transition_forecast,
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "trace", "forecast"),
        ),
    ]


def _examples_from_recovery_state(
    record: dict[str, Any], *, version: str, repo_root: Path, state_dir: Path
) -> list[dict[str, Any]]:
    trial_id = str(record.get("trial_id") or "unknown-recovery")
    lineage = _lineage_key(trial_id=trial_id, fallback=trial_id)
    recovery_target = _derive_tcl_recovery_action(record)
    recovery_confidence = _derive_tcl_recovery_confidence(record)
    run_triage = _derive_tcl_run_triage(record, recovery_confidence=recovery_confidence)
    return [
        _example_record(
            version=version,
            task_family="tcl_recovery_planning",
            task_name="recovery_state_to_resume_decision",
            source_kind="tcl_recovery_state",
            source_id=trial_id,
            source_file="recovery.json",
            lineage_key=lineage,
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
        ),
        _example_record(
            version=version,
            task_family="tcl_recovery_confidence_estimation",
            task_name="recovery_state_to_confidence_band",
            source_kind="tcl_recovery_state",
            source_id=f"{trial_id}:confidence",
            source_file="recovery.json",
            lineage_key=lineage,
            input_context={
                "trial_id": trial_id,
                "status": record.get("status"),
                "consecutive_fail_fast": record.get("consecutive_fail_fast"),
                "last_known_stable_hyperparameters": record.get("last_known_stable_hyperparameters"),
                "last_fail_reason": record.get("last_fail_reason"),
                "max_effective_dimensionality_achieved": record.get("max_effective_dimensionality_achieved"),
            },
            target=recovery_confidence,
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "recovery", "confidence"),
        ),
        _example_record(
            version=version,
            task_family="tcl_run_triage",
            task_name="recovery_state_to_operator_triage",
            source_kind="tcl_recovery_state",
            source_id=f"{trial_id}:triage",
            source_file="recovery.json",
            lineage_key=lineage,
            input_context={
                "trial_id": trial_id,
                "status": record.get("status"),
                "consecutive_fail_fast": record.get("consecutive_fail_fast"),
                "last_fail_reason": record.get("last_fail_reason"),
                "last_strategy_family": record.get("last_strategy_family"),
                "recovery_confidence": recovery_confidence,
            },
            target=run_triage,
            repo_root=repo_root,
            state_dir=state_dir,
            tags=("tcl", "recovery", "triage"),
        ),
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


def _iter_endpoint_records(path: Path) -> list[dict[str, Any]]:
    payload = _maybe_json(path)
    if payload is None:
        return []
    if isinstance(payload, dict):
        entries = payload.get("entries")
        if isinstance(entries, list):
            return [item for item in entries if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _iter_run_manifests(state_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    manifests_dir = state_dir / "manifests"
    if not manifests_dir.exists():
        return []
    manifests: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(manifests_dir.glob("*.json")):
        payload = _maybe_json(path)
        if not isinstance(payload, dict):
            continue
        if payload.get("manifest_version") != "tar.run.v1":
            continue
        manifests.append((_relative_or_name(path, state_dir.parent), payload))
    return manifests


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
        for record in _maybe_jsonl(state_dir / "claim_verdicts.jsonl")
        for example in _examples_from_claim_verdict(
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
        for record in _maybe_jsonl(state_dir / "evidence_debt_records.jsonl")
        for example in _examples_from_evidence_debt_record(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "project_staleness_records.jsonl")
        for example in _examples_from_project_staleness_record(
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
        for record in _iter_endpoint_records(state_dir / "inference_endpoints.json")
        for example in _examples_from_endpoint_record(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    examples.extend(
        example
        for source_file, record in _iter_run_manifests(state_dir)
        for example in _examples_from_run_manifest(
            record,
            version=version,
            repo_root=repo_root,
            state_dir=state_dir,
            source_file=source_file,
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
    examples.extend(
        example
        for record in _maybe_jsonl(state_dir / "recovery_history.jsonl")
        for example in _examples_from_recovery_state(
            record, version=version, repo_root=repo_root, state_dir=state_dir
        )
    )
    return examples


def _iter_source_artifact_paths(state_dir: Path) -> list[Path]:
    repo_root = state_dir.parent.resolve()
    paths: list[Path] = []
    for relative in STATE_ARTIFACT_FILES:
        path = state_dir / relative
        if path.exists():
            paths.append(path)
    manifests_dir = state_dir / "manifests"
    if manifests_dir.exists():
        paths.extend(sorted(manifests_dir.glob("*.json")))
    tar_runs_root = repo_root / "tar_runs"
    if tar_runs_root.exists():
        for summary_path in sorted(tar_runs_root.rglob("payload_summary.json")):
            paths.append(summary_path)
            config_path = summary_path.with_name("config.json")
            thermo_path = summary_path.with_name("thermo_metrics.jsonl")
            if config_path.exists():
                paths.append(config_path)
            if thermo_path.exists():
                paths.append(thermo_path)
    deduped = {str(path.resolve()): path for path in paths}
    return [deduped[key] for key in sorted(deduped)]


def _fingerprint_file(path: Path, *, root: Path) -> dict[str, Any]:
    payload = {
        "path": _relative_or_name(path, root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        payload["records"] = _count_nonempty_lines(path)
    return payload


def build_master_dataset(
    state_dirs: Path | Iterable[Path], output_dir: Path, *, version: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_state_dirs = (
        [state_dirs] if isinstance(state_dirs, Path) else [Path(item) for item in state_dirs]
    )

    deduped: dict[str, dict[str, Any]] = {}
    pre_dedup_count = 0
    for state_dir in normalized_state_dirs:
        for example in _collect_examples_for_state_dir(state_dir.resolve(), version=version):
            pre_dedup_count += 1
            deduped[example["dedupe_key"]] = example
    examples = sorted(deduped.values(), key=lambda item: item["example_id"])

    master_path = output_dir / "tar_master_dataset.jsonl"
    train_path = output_dir / "tar_master_dataset_train.jsonl"
    validation_path = output_dir / "tar_master_dataset_validation.jsonl"
    test_path = output_dir / "tar_master_dataset_test.jsonl"

    split_examples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    task_families = Counter()
    source_kinds = Counter()
    lineage_to_split: dict[str, str] = {}
    lineages_by_split: dict[str, set[str]] = defaultdict(set)
    families_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }

    for example in examples:
        split = _hash_split(str(example.get("lineage_key") or example["example_id"]))
        example["split"] = split
        task_families[example["task_family"]] += 1
        source_kinds[example["source_kind"]] += 1
        split_examples[split].append(example)
        families_by_split[split][example["task_family"]] += 1
        lineage_key = str(example.get("lineage_key") or example["example_id"])
        lineages_by_split[split].add(lineage_key)
        lineage_to_split.setdefault(lineage_key, split)

    with master_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    for path, split_name in (
        (train_path, "train"),
        (validation_path, "validation"),
        (test_path, "test"),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for example in split_examples[split_name]:
                handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    source_artifacts = [
        _fingerprint_file(path, root=state_dir.parent.resolve())
        for state_dir in normalized_state_dirs
        for path in _iter_source_artifact_paths(state_dir.resolve())
    ]
    output_files = {
        "master": _fingerprint_file(master_path, root=output_dir),
        "train": _fingerprint_file(train_path, root=output_dir),
        "validation": _fingerprint_file(validation_path, root=output_dir),
        "test": _fingerprint_file(test_path, root=output_dir),
    }
    manifest = {
        "dataset_version": version,
        "state_dirs": [
            _relative_or_name(path.resolve(), output_dir.parent.resolve())
            for path in normalized_state_dirs
        ],
        "output_dir": str(output_dir.resolve()),
        "records": len(examples),
        "pre_dedup_records": pre_dedup_count,
        "duplicate_examples_removed": pre_dedup_count - len(examples),
        "dedupe_strategy": "stable_dedupe_key",
        "splits": {name: len(items) for name, items in split_examples.items()},
        "split_lineages": {
            name: len(items)
            for name, items in sorted(lineages_by_split.items())
        },
        "split_task_families": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(families_by_split.items())
        },
        "split_integrity": {
            "lineage_safe": True,
            "lineage_count": len(lineage_to_split),
        },
        "task_families": dict(sorted(task_families.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
        "files": output_files,
        "source_artifacts": source_artifacts,
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
