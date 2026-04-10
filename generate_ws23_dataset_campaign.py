from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


CAMPAIGNS = ("continuity", "benchmark", "falsification", "portfolio", "tcl", "runtime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic WS23 TAR state.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--projects-per-campaign", type=int, default=24)
    parser.add_argument("--prefix", default="ws23")
    return parser.parse_args()


def _utc(base: datetime, offset_minutes: int) -> str:
    return (base + timedelta(minutes=offset_minutes)).replace(microsecond=0).isoformat()


def _hash_token(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _sandbox_policy(*, trial_id: str, permissive: bool) -> dict:
    artifact_dir = f"/workspace/tar_runs/{trial_id}/artifacts"
    return {
        "mode": "docker_only",
        "profile": "development" if permissive else "production",
        "network_policy": "limited" if permissive else "off",
        "allowed_mounts": ["/workspace", "/data", artifact_dir],
        "read_only_mounts": ["/workspace", "/data"],
        "writable_mounts": [artifact_dir] if not permissive else [artifact_dir, "/workspace/tmp"],
        "artifact_dir": artifact_dir,
        "workspace_root": "/workspace",
        "cpu_limit": 2,
        "memory_limit_gb": 4,
        "timeout_s": 1800,
    }


def _run_manifest(*, prefix: str, scenario_id: str, problem_id: str, trial_id: str, reproducible: bool, permissive: bool) -> dict:
    unresolved = [] if reproducible else ["evaluate", "trl"]
    return {
        "manifest_version": "tar.run.v1",
        "manifest_id": f"run-{scenario_id}",
        "kind": "science_bundle",
        "trial_id": trial_id,
        "problem_id": problem_id,
        "command": ["python", "run_problem.py"],
        "config_path": f"<WORKSPACE_ROOT>/tar_runs/{trial_id}/config.json",
        "image_manifest": {
            "manifest_version": "tar.repro.v1",
            "image_tag": f"{prefix}-{scenario_id}:locked",
            "base_image": "python:3.11-slim",
            "dockerfile_path": "Dockerfile.locked",
            "build_context_path": "<WORKSPACE_ROOT>",
            "build_command": ["docker", "build", "."],
            "dependency_lock": {
                "manifest_version": "tar.repro.v1",
                "lock_id": f"lock-{scenario_id}",
                "requirements_path": "requirements.txt",
                "packages": ["torch==2.5.1", "transformers==4.57.0"],
                "package_records": [
                    {
                        "requested_spec": "torch==2.5.1",
                        "normalized_name": "torch",
                        "resolved_spec": "torch==2.5.1",
                        "version": "2.5.1",
                        "required": True,
                        "resolution_status": "pinned",
                    }
                ],
                "unresolved_packages": unresolved,
                "fully_pinned": reproducible,
                "lock_incomplete_reason": None if reproducible else "Dependency lock incomplete.",
                "hash_sha256": _hash_token("lock", scenario_id),
            },
            "environment_fingerprint": {
                "manifest_version": "tar.repro.v1",
                "fingerprint_id": f"fingerprint-{scenario_id}",
                "workspace_root": "<WORKSPACE_ROOT>",
                "source_hash_sha256": _hash_token("src", scenario_id),
                "dockerfile_hash_sha256": _hash_token("docker", scenario_id),
                "requirements_hash_sha256": _hash_token("req", scenario_id),
                "python_version": "3.11",
            },
            "hash_sha256": _hash_token("image", scenario_id),
            "locked": reproducible,
            "build_status": "built" if reproducible else "failed",
        },
        "sandbox_policy": _sandbox_policy(trial_id=trial_id, permissive=permissive),
        "created_at": "2026-04-10T00:00:00+00:00",
        "hash_sha256": _hash_token("run", scenario_id),
        "reproducibility_complete": reproducible,
        "unresolved_packages": unresolved,
        "lock_incomplete_reason": None if reproducible else "Dependency lock incomplete for requested runtime.",
    }


def build_campaign_workspace(output_root: Path, *, projects_per_campaign: int, prefix: str) -> dict:
    workspace = output_root.resolve()
    state_dir = workspace / "tar_state"
    manifests_dir = state_dir / "manifests"
    tar_runs_root = workspace / "tar_runs"
    logs_dir = workspace / "logs"
    for path in (state_dir, manifests_dir, tar_runs_root, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    base = datetime(2026, 4, 10, tzinfo=timezone.utc)
    studies, executions, decisions, verdicts = [], [], [], []
    verifications, recovery_history = [], []
    plans, priorities, debts, stale = [], [], [], []
    portfolio, projects, endpoints, alerts = [], [], [], []

    index = 0
    for campaign in CAMPAIGNS:
        for replica in range(projects_per_campaign):
            index += 1
            scenario_id = f"{prefix}-{campaign}-{replica:04d}"
            project_id = f"project-{scenario_id}"
            thread_id = f"thread-{scenario_id}"
            question_id = f"question-{scenario_id}"
            action_id = f"action-{scenario_id}"
            problem_id = f"problem-{scenario_id}"
            trial_id = f"trial-{scenario_id}"
            ts = _utc(base, index)
            old_ts = _utc(base, max(0, index - 60 * 24))
            reproducible = not (campaign in {"benchmark", "runtime"} and replica % 3 == 0)
            permissive = campaign == "runtime" and replica % 2 == 0
            canonical = not (campaign == "benchmark" and replica % 2 == 0)
            truth_status = "canonical_ready" if canonical else "unsupported"
            alignment = "aligned" if canonical else "refused"
            verdict_status = "contradicted" if campaign == "falsification" and replica % 2 == 0 else "provisional"
            project_status = "blocked" if campaign == "portfolio" and replica % 4 == 0 else ("paused" if campaign == "continuity" and replica % 3 == 0 else "active")
            thread_status = "contradicted" if verdict_status == "contradicted" else ("parked" if project_status != "active" else "supported")
            stop_reason = "dependency_missing" if project_status == "paused" else ("benchmark_unavailable" if project_status == "blocked" else None)
            blockers = [] if project_status == "active" else [stop_reason, "operator review required"]

            studies.append({"problem_id": problem_id, "project_id": project_id, "thread_id": thread_id, "action_id": action_id, "created_at": ts, "problem": f"Investigate {campaign} scenario {replica}", "domain": "quantum_ml" if campaign in {"tcl", "benchmark"} else "generic_ml", "profile_id": "quantum_ml" if campaign in {"tcl", "benchmark"} else "generic_ml", "resolution_confidence": 0.6, "hypotheses": [f"{campaign} hypothesis A", f"{campaign} hypothesis B"], "experiments": [{"template_id": f"template-{campaign}", "name": f"{campaign.title()} validation experiment", "benchmark": "qml_canonical" if canonical else "validation_proxy", "benchmark_tier": "canonical" if canonical else "validation", "metrics": ["accuracy", "ece", "d_pr"], "success_criteria": ["accuracy improves", "ece improves"]}], "benchmark_tier": "canonical", "benchmark_ids": [f"benchmark-{campaign}-{replica:04d}"], "benchmark_names": [f"{campaign.title()} Benchmark {replica}"], "benchmark_truth_statuses": [truth_status], "benchmark_alignment": alignment, "canonical_comparable": canonical, "cited_research_ids": [f"research:{campaign}:{replica:04d}"], "retrieved_memory_ids": [f"memory:{campaign}:{replica:04d}"], "environment": {"reproducibility_complete": reproducible, "unresolved_packages": [] if reproducible else ["evaluate", "trl"], "lock_incomplete_reason": None if reproducible else "Dependency lock incomplete for requested science environment."}, "next_action": "Run the canonical study." if canonical and reproducible else "Downgrade scope and rebuild the reproducible environment.", "status": "planned"})
            executions.append({"problem_id": problem_id, "project_id": project_id, "thread_id": thread_id, "action_id": action_id, "executed_at": ts, "problem": f"Investigate {campaign} scenario {replica}", "profile_id": "quantum_ml" if campaign in {"tcl", "benchmark"} else "generic_ml", "domain": "quantum_ml" if campaign in {"tcl", "benchmark"} else "generic_ml", "benchmark_tier": "canonical", "requested_benchmark": f"benchmark-{campaign}-{replica:04d}", "canonical_comparable": canonical, "proxy_benchmarks_used": not canonical, "benchmark_ids": [f"benchmark-{campaign}-{replica:04d}"], "benchmark_names": [f"{campaign.title()} Benchmark {replica}"], "actual_benchmark_tiers": ["canonical" if canonical else "validation"], "benchmark_truth_statuses": [truth_status], "benchmark_alignment": alignment, "execution_mode": "docker_bundle", "imports_ok": ["numpy", "torch"], "imports_failed": [] if reproducible else [{"module": "evaluate", "error": "No module named 'evaluate'"}], "image_tag": f"{prefix}-{scenario_id}:locked", "manifest_path": f"<STATE_DIR>/manifests/run-{scenario_id}.json", "manifest_hash": _hash_token("run", scenario_id), "dependency_hash": _hash_token("dep", scenario_id), "reproducibility_complete": reproducible, "sandbox_policy": _sandbox_policy(trial_id=trial_id, permissive=permissive), "summary": "Execution completed with aligned evidence." if reproducible and canonical else "Execution is not yet safe for strong promotion.", "recommended_next_step": "Replicate and run falsification pressure." if canonical else "Resolve reproducibility or benchmark scope gaps first.", "artifact_path": f"<WORKSPACE_ROOT>/tar_runs/{trial_id}/verification/control/report.json", "status": "completed" if reproducible else "dependency_failure"})
            verifications.append({"trial_id": trial_id, "verified_at": ts, "control_score": 1.2 if reproducible else 0.6, "seed_variance": {"num_runs": 3, "loss_mean": 0.52, "loss_std": 0.03 if reproducible else 0.11, "dimensionality_mean": 6.4, "dimensionality_std": 0.18 if reproducible else 0.92, "calibration_ece_mean": 0.08 if reproducible else 0.19, "stable": reproducible, "runs": []}, "calibration": {"ece": 0.07 if reproducible else 0.19, "accuracy": 0.58 if reproducible else 0.41, "mean_confidence": 0.61 if reproducible else 0.69, "bins": []}, "ablations": [{"name": "no_anchor_penalty", "training_loss": 0.58, "effective_dimensionality": 5.9, "equilibrium_fraction": 0.42, "calibration_ece": 0.12, "score": 0.45, "delta_vs_control": -0.08}], "verdict": "verified" if reproducible and canonical else ("unstable" if campaign in {"tcl", "falsification"} else "inconclusive"), "recommendations": ["Increase falsification pressure before promotion." if not reproducible else "Proceed to replication and contradiction review."]})
            decisions.append({"decision_id": f"decision-{scenario_id}", "created_at": ts, "prompt": f"Investigate {campaign} scenario {replica}", "mode": "problem_study", "problem_id": problem_id, "thread_id": thread_id, "action_id": action_id, "evidence_bundle": {"bundle_id": f"bundle-{scenario_id}", "query": f"Investigate {campaign} scenario {replica}", "confidence": 0.6, "supporting_document_ids": [f"research:{campaign}:{replica:04d}"], "traces": [{"document_id": f"research:{campaign}:{replica:04d}"}]}, "hypotheses": [f"{campaign} hypothesis A"], "selected_action": "Create a contradiction-focused follow-up." if campaign == "falsification" else "Continue with the next benchmark-aligned study.", "confidence": 0.6, "notes": [f"campaign={campaign}"]})
            verdicts.append({"verdict_id": f"verdict-{scenario_id}", "trial_id": trial_id, "created_at": ts, "decision_scope": "trial_local", "status": verdict_status, "rationale": [f"{campaign} verdict rationale"], "policy": {"policy_id": "frontier_claim_policy_v1", "min_seed_runs": 3, "max_seed_loss_std": 0.08, "max_seed_dimensionality_std": 0.75, "max_calibration_ece": 0.15, "min_ablation_gap": 0.05, "min_supporting_sources": 2, "max_allowed_contradictions": 0, "require_canonical_benchmark": True}, "supporting_research_ids": [f"research:{campaign}:{replica:04d}"], "supporting_evidence_ids": [f"evidence:{scenario_id}:support"], "verification_report_trial_id": trial_id, "benchmark_problem_id": problem_id, "benchmark_execution_created_at": ts, "benchmark_execution_mode": "docker_bundle", "supporting_benchmark_ids": [f"benchmark-{campaign}-{replica:04d}"], "supporting_benchmark_names": [f"{campaign.title()} Benchmark {replica}"], "evidence_bundle_id": f"bundle-{scenario_id}", "canonical_comparability_source": "problem_execution" if reproducible else "problem_study", "verdict_inputs_complete": reproducible, "linkage_status": "exact" if reproducible else "ambiguous", "linkage_note": "Linked to the current study/execution family.", "canonical_benchmark_required": True, "canonical_benchmark_satisfied": canonical and reproducible, "confidence": 0.58})
            plans.append({"plan_id": f"plan-{scenario_id}", "project_id": project_id, "thread_id": thread_id, "created_at": ts, "status": "active", "trigger_reason": "contradiction pressure" if verdict_status == "contradicted" else "confidence rising", "coverage": {"ablation_coverage": 1.0 if campaign == "falsification" else 0.0, "replication_coverage": 0.0 if replica % 2 == 0 else 1.0, "contradiction_coverage": 1.0 if verdict_status == "contradicted" else 0.0, "benchmark_pressure_coverage": 1.0 if canonical else 0.0, "calibration_coverage": 0.5, "overall_sufficient": False}, "tests": [{"kind": "contradiction_resolution" if verdict_status == "contradicted" else "mechanism_ablation", "description": "Run the minimum adversarial follow-up.", "expected_falsification_value": 0.75}]})
            debt_value = 0.6 if (not reproducible or verdict_status == "contradicted") else 0.32
            debts.append({"record_id": f"debt-{scenario_id}", "project_id": project_id, "created_at": ts, "falsification_gap": 0.65 if verdict_status != "contradicted" else 0.25, "replication_gap": 0.55 if replica % 2 == 0 else 0.2, "benchmark_gap": 0.7 if not canonical else 0.15, "claim_linkage_gap": 0.4 if not reproducible else 0.1, "calibration_gap": 0.35 if campaign == "tcl" else 0.15, "overall_debt": debt_value, "promotion_blocked": debt_value >= 0.45, "rationale": ["Need stronger adversarial and benchmark pressure before promotion."]})
            stale_level = "fresh" if campaign != "portfolio" else ("critical" if replica % 2 else "stale")
            stale.append({"record_id": f"stale-{scenario_id}", "project_id": project_id, "created_at": ts, "last_progress_at": ts if stale_level == "fresh" else old_ts, "hours_since_progress": 4.0 if stale_level == "fresh" else 72.0 + replica, "staleness_level": stale_level, "reason": f"{campaign} backlog review", "resume_candidate": campaign == "portfolio" and replica % 3 != 0, "closure_candidate": campaign == "portfolio" and replica % 5 == 0})
            priorities.append({"record_id": f"priority-{scenario_id}", "project_id": project_id, "created_at": ts, "action_id": action_id, "priority_score": 0.85 if reproducible and canonical else 0.42, "strategic_priority": 0.6, "expected_value": 0.7 if canonical else 0.45, "evidence_debt": debt_value, "contradiction_pressure": 0.8 if verdict_status == "contradicted" else 0.25, "staleness_penalty": 0.3 if stale_level != "fresh" else 0.0, "budget_pressure": 0.2 if reproducible else 0.55, "benchmark_readiness": 0.9 if canonical else 0.35, "recommended_state": "continue" if reproducible else "defer", "rationale": ["Choose the cheapest decisive action under current evidence pressure."]})
            portfolio.append({"decision_id": f"portfolio-{scenario_id}", "created_at": ts, "selected_project_id": project_id if reproducible else None, "selected_action_id": action_id if reproducible else None, "deferred_project_ids": [] if reproducible else [project_id], "parked_project_ids": [project_id] if stale_level == "critical" else [], "resumed_project_ids": [project_id] if campaign == "portfolio" and replica % 3 != 0 else [], "escalated_project_ids": [project_id] if verdict_status == "contradicted" else [], "retired_project_ids": [project_id] if campaign == "portfolio" and replica % 5 == 0 else [], "rationale": ["Portfolio review selected the best project under current budget and evidence debt."]})
            projects.append({"project_id": project_id, "title": f"{campaign}_project_{replica:04d}", "goal": f"Investigate {campaign} scenario {replica}", "domain_profile": "quantum_ml" if campaign in {"tcl", "benchmark"} else "generic_ml", "status": project_status, "priority": 1 + (replica % 5), "created_at": ts, "updated_at": ts, "active_thread_id": thread_id, "budget_ledger": {"wall_clock_minutes_budget": 240.0, "wall_clock_minutes_spent": 25.0 + (replica % 6), "gpu_hours_budget": 6.0, "gpu_hours_spent": 0.8 + (replica % 4) * 0.2, "experiment_budget": 6, "experiments_spent": 1 + (replica % 3), "replication_budget": 3, "replications_spent": replica % 2, "budget_exhausted": False, "budget_pressure_level": "medium" if not reproducible else "low"}, "resume_snapshot": {"project_id": project_id, "active_thread_id": thread_id, "current_question_id": question_id, "next_action_id": action_id, "latest_evidence_summary": "Aligned evidence but still requires falsification pressure." if reproducible else "Evidence is blocked by reproducibility or benchmark gaps.", "blockers": blockers, "budget_remaining_summary": {"experiments_remaining": 4.0, "replications_remaining": 2.0, "gpu_hours_remaining": 5.2, "wall_clock_minutes_remaining": 210.0}, "captured_at": ts}, "latest_decision_summary": "Continue with the next aligned study." if reproducible else "Resolve runtime or benchmark blockers before promotion.", "hypothesis_threads": [{"thread_id": thread_id, "project_id": project_id, "hypothesis": f"{campaign} hypothesis A", "status": thread_status, "confidence_state": "provisional", "supporting_evidence_ids": [f"evidence:{scenario_id}:support"], "contradicting_evidence_ids": [] if verdict_status != "contradicted" else [f"evidence:{scenario_id}:contradiction"], "open_question_ids": [question_id], "next_action_id": action_id, "stop_reason": stop_reason, "resume_reason": None, "created_at": ts, "updated_at": ts}], "open_questions": [{"question_id": question_id, "project_id": project_id, "thread_id": thread_id, "question": f"What is the next decisive action for {campaign} scenario {replica}?", "importance": 0.7, "uncertainty_type": "next_action", "blocking": bool(blockers), "status": "open", "created_at": ts, "resolved_at": None}], "planned_actions": [{"action_id": action_id, "project_id": project_id, "thread_id": thread_id, "action_kind": "run_problem_study", "description": "Run the next benchmark-aligned study.", "estimated_cost": 0.35, "expected_evidence_gain": 0.72 if reproducible else 0.4, "depends_on": [question_id], "status": "planned", "scheduled_job_id": None, "result_refs": []}]})
            _write_json(manifests_dir / f"run-{scenario_id}.json", _run_manifest(prefix=prefix, scenario_id=scenario_id, problem_id=problem_id, trial_id=trial_id, reproducible=reproducible, permissive=permissive))

            if campaign in {"runtime", "tcl"}:
                endpoints.append({"endpoint_name": f"endpoint-{scenario_id}", "checkpoint_name": f"checkpoint-{scenario_id}", "role": "assistant", "host": "127.0.0.1", "port": 8000 + (index % 1000), "backend": "transformers", "base_url": f"http://127.0.0.1:{8000 + (index % 1000)}", "command": ["python", "serve_local.py"], "env": {"CUDA_VISIBLE_DEVICES": "0"}, "status": "failed" if campaign == "runtime" and replica % 2 == 0 else "running", "process_pid": None if campaign == "runtime" and replica % 2 == 0 else 40000 + index, "started_at": ts, "stopped_at": ts if campaign == "runtime" and replica % 2 == 0 else None, "last_error": "health check timeout after startup" if campaign == "runtime" and replica % 2 == 0 else None, "last_health_at": ts, "manifest_path": f"<STATE_DIR>/endpoints/endpoint-{scenario_id}/endpoint_manifest.json", "stdout_log_path": f"<STATE_DIR>/endpoints/endpoint-{scenario_id}/stdout.log", "stderr_log_path": f"<STATE_DIR>/endpoints/endpoint-{scenario_id}/stderr.log", "trust_remote_code": campaign == "runtime" and replica % 3 == 0, "health": {"endpoint_name": f"endpoint-{scenario_id}", "checked_at": ts, "status": "failed" if campaign == "runtime" and replica % 2 == 0 else "healthy", "ok": not (campaign == "runtime" and replica % 2 == 0), "http_status": 500 if campaign == "runtime" and replica % 2 == 0 else 200, "latency_ms": 82.0 + (replica % 7), "detail": "startup timeout; inspect stderr tail" if campaign == "runtime" and replica % 2 == 0 else "healthy", "model_id": f"checkpoint-{scenario_id}", "backend": "transformers", "role": "assistant", "trust_remote_code": campaign == "runtime" and replica % 3 == 0}})
                alerts.append({"alert_id": f"alert-{scenario_id}", "created_at": ts, "severity": "warning" if campaign == "runtime" else "info", "source": "inference_endpoint", "message": "Endpoint requires operator review." if campaign == "runtime" else "Endpoint healthy.", "metadata": {"endpoint_name": f"endpoint-{scenario_id}"}})

            if campaign == "tcl":
                trial_dir = tar_runs_root / trial_id / "verification" / "control"
                trial_dir.mkdir(parents=True, exist_ok=True)
                last_metrics = {"step": 2, "entropy_sigma": 0.002 + (replica % 3) * 0.001, "drift_rho": 0.05 + (replica % 4) * 0.03, "grad_norm": 0.5 + (replica % 4) * 0.05, "effective_dimensionality": 2.8 + (replica % 5) * 0.4, "dimensionality_ratio": 0.48 + (replica % 6) * 0.11, "equilibrium_fraction": 0.15 + (replica % 5) * 0.15, "equilibrium_gate": replica % 2 == 1, "training_loss": 0.54 + (replica % 4) * 0.08}
                _write_json(trial_dir / "config.json", {"backend_id": "asc_text", "strategy_family": "elastic_anchor", "governor_thresholds": {"max_quenching_loss": 0.8, "min_dimensionality_ratio": 0.55}, "data_provenance": {"research_grade": False, "has_fallback": replica % 2 == 0}, "backend_provenance": {"required_metrics": ["training_loss", "effective_dimensionality", "entropy_sigma", "drift_rho"]}})
                _write_json(trial_dir / "payload_summary.json", {"trial_id": trial_id, "strategy_family": "elastic_anchor", "governor_action": "continue" if replica % 2 else "terminate", "governor_reasons": ["equilibrium_gate"] if replica % 2 else ["weight_drift_limit"], "anchor_effective_dimensionality": 3.4 + (replica % 4) * 0.3, "last_metrics": last_metrics, "calibration": {"ece": 0.07 + (replica % 5) * 0.02, "accuracy": 0.44 + (replica % 4) * 0.06}})
                _write_jsonl(trial_dir / "thermo_metrics.jsonl", [{"trial_id": trial_id, "step": 1, "drift_rho": 0.04 + (replica % 4) * 0.02, "effective_dimensionality": 2.4 + (replica % 5) * 0.3, "dimensionality_ratio": 0.42 + (replica % 6) * 0.09, "equilibrium_fraction": 0.1 + (replica % 4) * 0.12, "equilibrium_gate": False, "training_loss": 0.51 + (replica % 4) * 0.09}, {"trial_id": trial_id, **last_metrics}])
                recovery_history.append({"trial_id": trial_id, "status": "completed" if replica % 2 else "pivoted", "last_known_stable_hyperparameters": {"alpha": 0.04, "eta": 0.01}, "last_fail_reason": None if replica % 2 else "weight_drift_limit", "last_fail_metrics": None, "consecutive_fail_fast": 0 if replica % 2 else 2, "last_strategy_family": "elastic_anchor", "last_anchor_path": "anchors/thermodynamic_anchor.safetensors", "max_effective_dimensionality_achieved": 7.0 + (replica % 5) * 0.2})

    _write_jsonl(state_dir / "problem_studies.jsonl", studies)
    _write_jsonl(state_dir / "problem_executions.jsonl", executions)
    _write_jsonl(state_dir / "verification_reports.jsonl", verifications)
    _write_jsonl(state_dir / "research_decisions.jsonl", decisions)
    _write_jsonl(state_dir / "claim_verdicts.jsonl", verdicts)
    _write_jsonl(state_dir / "falsification_plans.jsonl", plans)
    _write_jsonl(state_dir / "project_priority_records.jsonl", priorities)
    _write_jsonl(state_dir / "evidence_debt_records.jsonl", debts)
    _write_jsonl(state_dir / "project_staleness_records.jsonl", stale)
    _write_jsonl(state_dir / "portfolio_decisions.jsonl", portfolio)
    _write_json(state_dir / "research_projects.json", projects)
    _write_json(state_dir / "inference_endpoints.json", {"entries": endpoints})
    _write_jsonl(state_dir / "alerts.jsonl", alerts)
    _write_json(state_dir / "recovery.json", {"trial_id": f"trial-{prefix}-tcl-0000", "status": "fail_fast", "last_known_stable_hyperparameters": {"alpha": 0.04, "eta": 0.01}, "last_fail_reason": "weight_drift_limit", "last_fail_metrics": None, "consecutive_fail_fast": 3, "last_strategy_family": "elastic_anchor", "last_anchor_path": "anchors/thermodynamic_anchor.safetensors", "max_effective_dimensionality_achieved": 7.9})
    _write_jsonl(state_dir / "recovery_history.jsonl", recovery_history)
    _write_json(state_dir / "runtime_heartbeat.json", {"started_at": "2026-04-10T00:00:00+00:00", "finished_at": "2026-04-10T00:10:00+00:00", "status": "completed", "executed_jobs": len(executions), "stale_cleanups": len([x for x in stale if x["staleness_level"] != "fresh"]), "failed_jobs": len([x for x in executions if x["status"] != "completed"]), "active_leases": 0, "retry_waiting": len([x for x in executions if x["status"] == "dependency_failure"]), "alert_count": len(alerts), "notes": ["WS23 campaign state generated."]})
    _write_json(workspace / "ws23_campaign_manifest.json", {"prefix": prefix, "campaigns": list(CAMPAIGNS), "projects_per_campaign": projects_per_campaign, "projects": len(projects), "endpoint_records": len(endpoints)})
    return {"workspace": str(workspace), "tar_state": str(state_dir), "projects": len(projects), "problem_studies": len(studies), "claim_verdicts": len(verdicts)}


def main() -> int:
    args = parse_args()
    result = build_campaign_workspace(Path(args.output_root), projects_per_campaign=args.projects_per_campaign, prefix=args.prefix)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
