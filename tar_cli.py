from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tar_lab.control import DEFAULT_HOST, DEFAULT_PORT, TARControlServer, send_command
from tar_lab.orchestrator import TAROrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAR command and control CLI")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--last-fail", action="store_true", dest="last_fail")
    parser.add_argument("--force-fail-fast", action="store_true", dest="force_fail_fast")
    parser.add_argument("--topic", default="frontier ai")
    parser.add_argument("--max-results", type=int, default=6, dest="max_results")
    parser.add_argument("--trial-id", dest="trial_id")
    parser.add_argument("--problem-id", dest="problem_id")
    parser.add_argument("--project-id", dest="project_id")
    parser.add_argument("--schedule-id", dest="schedule_id")
    parser.add_argument("--endpoint-name", dest="endpoint_name")
    parser.add_argument("--profile-id", dest="profile_id")
    parser.add_argument("--manifest-id", dest="manifest_id")
    parser.add_argument("--manifest-path", dest="manifest_path")
    parser.add_argument("--problem")
    parser.add_argument("--benchmark", dest="benchmark")
    parser.add_argument("--benchmark-tier", default="validation", dest="benchmark_tier", choices=["smoke", "validation", "canonical"])
    parser.add_argument("--canonical-only", action="store_true", dest="canonical_only")
    parser.add_argument("--no-proxy-benchmarks", action="store_true", dest="no_proxy_benchmarks")
    parser.add_argument("--include-blocked", action="store_true", dest="include_blocked")
    parser.add_argument("--schedule-selected", action="store_true", dest="schedule_selected")
    parser.add_argument(
        "--prioritization-mode",
        default="balanced",
        dest="prioritization_mode",
        choices=["balanced", "falsification_first"],
    )
    parser.add_argument("--paper-path", action="append", dest="paper_paths")
    parser.add_argument("--checkpoint-name", dest="checkpoint_name")
    parser.add_argument("--model-path", dest="model_path")
    parser.add_argument("--base-model-id", dest="base_model_id")
    parser.add_argument("--adapter-path", dest="adapter_path")
    parser.add_argument("--backend-name", default="transformers", dest="backend_name")
    parser.add_argument("--role-name", default="assistant", dest="role_name")
    parser.add_argument(
        "--operator-mode",
        default="tuned_local",
        dest="operator_mode",
        choices=["prompt_only", "tuned_local"],
    )
    parser.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.set_defaults(trust_remote_code=None)
    parser.add_argument("--wait-for-health", action="store_true", dest="wait_for_health")
    parser.add_argument("--build-env", action="store_true", dest="build_env")
    parser.add_argument("--use-docker", action="store_true", dest="use_docker")
    parser.add_argument("--run-at", dest="run_at")
    parser.add_argument("--delay-s", type=int, default=0, dest="delay_s")
    parser.add_argument("--repeat-interval-s", type=int, dest="repeat_interval_s")
    parser.add_argument("--max-runs", type=int, default=1, dest="max_runs")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument(
        "--pause-reason",
        default="operator_paused",
        dest="pause_reason",
        choices=[
            "budget_exhausted",
            "evidence_saturated",
            "contradiction_detected",
            "dependency_missing",
            "benchmark_unavailable",
            "awaiting_human_review",
            "superseded_by_better_thread",
            "runtime_failure",
            "goal_completed",
            "operator_paused",
        ],
    )
    parser.add_argument(
        "--resume-reason",
        default="human_requested_resume",
        dest="resume_reason",
        choices=[
            "new_budget_allocated",
            "dependency_restored",
            "new_evidence_arrived",
            "scheduled_followup_due",
            "contradiction_requires_resolution",
            "human_requested_resume",
        ],
    )
    parser.add_argument("--max-jobs", type=int, default=1, dest="max_jobs")
    parser.add_argument("--stale-after-s", type=int, default=900, dest="stale_after_s")
    parser.add_argument("--alert-count", type=int, default=20, dest="alert_count")
    parser.add_argument("--message")
    parser.add_argument("--voice-file", dest="voice_file")
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--wake-word", default="lab")
    parser.add_argument("--listen-duration", type=float, default=3.0, dest="listen_duration")
    parser.add_argument("--listen-timeout", type=float, default=15.0, dest="listen_timeout")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--serve", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--frontier-status", action="store_true", dest="frontier_status")
    group.add_argument("--runtime-status", action="store_true", dest="runtime_status")
    group.add_argument("--dry-run", action="store_true", dest="dry_run")
    group.add_argument("--live-docker-test", action="store_true", dest="live_docker_test")
    group.add_argument("--check-regime", action="store_true", dest="check_regime")
    group.add_argument("--ingest-research", action="store_true", dest="ingest_research")
    group.add_argument("--verify-last-trial", action="store_true", dest="verify_last_trial")
    group.add_argument("--breakthrough-report", action="store_true", dest="breakthrough_report")
    group.add_argument("--resolve-problem", action="store_true", dest="resolve_problem")
    group.add_argument("--prepare-science-env", action="store_true", dest="prepare_science_env")
    group.add_argument("--study-problem", action="store_true", dest="study_problem")
    group.add_argument("--run-problem-study", action="store_true", dest="run_problem_study")
    group.add_argument("--schedule-problem-study", action="store_true", dest="schedule_problem_study")
    group.add_argument("--scheduler-status", action="store_true", dest="scheduler_status")
    group.add_argument("--run-scheduler-once", action="store_true", dest="run_scheduler_once")
    group.add_argument("--list-benchmarks", action="store_true", dest="list_benchmarks")
    group.add_argument("--benchmark-status", action="store_true", dest="benchmark_status")
    group.add_argument("--prepare-payload-env", action="store_true", dest="prepare_payload_env")
    group.add_argument("--rebuild-locked-image", action="store_true", dest="rebuild_locked_image")
    group.add_argument("--show-manifest", action="store_true", dest="show_manifest")
    group.add_argument("--ingest-papers", action="store_true", dest="ingest_papers")
    group.add_argument("--list-experiment-backends", action="store_true", dest="list_experiment_backends")
    group.add_argument("--run-runtime-cycle", action="store_true", dest="run_runtime_cycle")
    group.add_argument("--list-alerts", action="store_true", dest="list_alerts")
    group.add_argument("--retry-failed-job", action="store_true", dest="retry_failed_job")
    group.add_argument("--cancel-job", action="store_true", dest="cancel_job")
    group.add_argument("--sandbox-policy", action="store_true", dest="sandbox_policy")
    group.add_argument("--register-checkpoint", action="store_true", dest="register_checkpoint")
    group.add_argument("--list-checkpoints", action="store_true", dest="list_checkpoints")
    group.add_argument("--build-inference-endpoint", action="store_true", dest="build_inference_endpoint")
    group.add_argument("--list-endpoints", action="store_true", dest="list_endpoints")
    group.add_argument("--start-endpoint", action="store_true", dest="start_endpoint")
    group.add_argument("--stop-endpoint", action="store_true", dest="stop_endpoint")
    group.add_argument("--restart-endpoint", action="store_true", dest="restart_endpoint")
    group.add_argument("--endpoint-health", action="store_true", dest="endpoint_health")
    group.add_argument("--assign-role", action="store_true", dest="assign_role")
    group.add_argument("--select-operator-checkpoint", action="store_true", dest="select_operator_checkpoint")
    group.add_argument("--operator-serving-status", action="store_true", dest="operator_serving_status")
    group.add_argument("--claim-policy", action="store_true", dest="claim_policy")
    group.add_argument("--claim-verdict", action="store_true", dest="claim_verdict")
    group.add_argument("--research-decision-log", action="store_true", dest="research_decision_log")
    group.add_argument("--create-project", action="store_true", dest="create_project")
    group.add_argument("--list-projects", action="store_true", dest="list_projects")
    group.add_argument("--project-status", action="store_true", dest="project_status")
    group.add_argument("--pause-project", action="store_true", dest="pause_project")
    group.add_argument("--resume-project", action="store_true", dest="resume_project")
    group.add_argument("--next-action", action="store_true", dest="next_action")
    group.add_argument("--operator-view", action="store_true", dest="operator_view")
    group.add_argument("--project-timeline", action="store_true", dest="project_timeline")
    group.add_argument("--evidence-map", action="store_true", dest="evidence_map")
    group.add_argument("--claim-lineage", action="store_true", dest="claim_lineage")
    group.add_argument("--resume-dashboard", action="store_true", dest="resume_dashboard")
    group.add_argument("--publication-handoff", action="store_true", dest="publication_handoff")
    group.add_argument("--publication-log", action="store_true", dest="publication_log")
    group.add_argument("--portfolio-status", action="store_true", dest="portfolio_status")
    group.add_argument("--portfolio-review", action="store_true", dest="portfolio_review")
    group.add_argument("--portfolio-decide", action="store_true", dest="portfolio_decide")
    group.add_argument("--rank-actions", action="store_true", dest="rank_actions")
    group.add_argument("--allocate-budget", action="store_true", dest="allocate_budget")
    group.add_argument("--prioritization-log", action="store_true", dest="prioritization_log")
    group.add_argument("--generate-falsification-plan", action="store_true", dest="generate_falsification_plan")
    group.add_argument("--falsification-status", action="store_true", dest="falsification_status")
    group.add_argument("--falsification-log", action="store_true", dest="falsification_log")
    group.add_argument("--stale-projects", action="store_true", dest="stale_projects")
    group.add_argument("--evidence-debt", action="store_true", dest="evidence_debt")
    group.add_argument("--resume-candidates", action="store_true", dest="resume_candidates")
    group.add_argument("--chat", action="store_true")
    group.add_argument("--pivot", action="store_true")
    group.add_argument("--explain", action="store_true")
    group.add_argument("--panic", action="store_true")
    return parser.parse_args()


def _direct_dispatch(orchestrator: TAROrchestrator, args: argparse.Namespace) -> Dict[str, Any]:
    if args.status:
        return orchestrator.status()
    if args.frontier_status:
        return orchestrator.frontier_status().model_dump(mode="json")
    if args.runtime_status:
        return orchestrator.runtime_status()
    if args.dry_run:
        return orchestrator.run_dry_run(force_fail_fast=args.force_fail_fast).model_dump(mode="json")
    if args.live_docker_test:
        return orchestrator.live_docker_test().model_dump(mode="json")
    if args.check_regime:
        return orchestrator.check_regime()
    if args.ingest_research:
        return orchestrator.ingest_research(topic=args.topic, max_results=args.max_results).model_dump(mode="json")
    if args.verify_last_trial:
        return orchestrator.verify_last_trial(trial_id=args.trial_id).model_dump(mode="json")
    if args.breakthrough_report:
        return orchestrator.breakthrough_report(
            trial_id=args.trial_id,
            problem_id=args.problem_id,
        ).model_dump(mode="json")
    if args.resolve_problem:
        return orchestrator.resolve_problem(
            args.problem or args.message or "",
            benchmark_tier=args.benchmark_tier,  # type: ignore[arg-type]
            requested_benchmark=args.benchmark,
        ).model_dump(mode="json")
    if args.prepare_science_env:
        return orchestrator.prepare_science_environment(
            problem=args.problem or args.message or "",
            build=args.build_env,
            benchmark_tier=args.benchmark_tier,  # type: ignore[arg-type]
            requested_benchmark=args.benchmark,
            canonical_only=args.canonical_only,
            no_proxy_benchmarks=args.no_proxy_benchmarks,
        ).model_dump(mode="json")
    if args.study_problem:
        return orchestrator.study_problem(
            problem=args.problem or args.message or "",
            project_id=args.project_id,
            build_env=args.build_env,
            max_results=args.max_results,
            benchmark_tier=args.benchmark_tier,  # type: ignore[arg-type]
            requested_benchmark=args.benchmark,
            canonical_only=args.canonical_only,
            no_proxy_benchmarks=args.no_proxy_benchmarks,
        ).model_dump(mode="json")
    if args.create_project:
        project = orchestrator.create_project(
            problem=args.problem or args.message or "",
            benchmark_tier=args.benchmark_tier,  # type: ignore[arg-type]
            requested_benchmark=args.benchmark,
        )
        return orchestrator.project_status(project.project_id)
    if args.list_projects:
        return orchestrator.list_projects()
    if args.project_status:
        return orchestrator.project_status(args.project_id or "")
    if args.pause_project:
        project = orchestrator.pause_project(
            args.project_id or "",
            reason=args.pause_reason,
            note=args.message,
        )
        return orchestrator.project_status(project.project_id)
    if args.resume_project:
        project = orchestrator.resume_project(
            args.project_id or "",
            reason=args.resume_reason,
            note=args.message,
        )
        return orchestrator.project_status(project.project_id)
    if args.next_action:
        return orchestrator.next_action(args.project_id or "")
    if args.operator_view:
        return orchestrator.operator_view(
            include_blocked=args.include_blocked,
            limit=args.max_results,
            mode=args.prioritization_mode,
        )
    if args.project_timeline:
        return orchestrator.project_timeline(args.project_id or "", limit=args.max_results)
    if args.evidence_map:
        return orchestrator.project_evidence_map(args.project_id or "")
    if args.claim_lineage:
        return orchestrator.claim_lineage(args.project_id or "")
    if args.resume_dashboard:
        return orchestrator.resume_dashboard(args.project_id or "")
    if args.publication_handoff:
        return orchestrator.publication_handoff(args.project_id or "")
    if args.publication_log:
        return orchestrator.publication_log(count=args.max_results)
    if args.portfolio_status:
        return orchestrator.portfolio_status(
            include_blocked=args.include_blocked,
            limit=args.max_results,
            mode=args.prioritization_mode,
        )
    if args.portfolio_review:
        return orchestrator.portfolio_review(
            include_blocked=args.include_blocked,
            limit=args.max_results,
            mode=args.prioritization_mode,
        )
    if args.portfolio_decide:
        return orchestrator.portfolio_decide(
            include_blocked=args.include_blocked,
            limit=args.max_results,
            mode=args.prioritization_mode,
        )
    if args.rank_actions:
        return orchestrator.rank_actions(
            project_id=args.project_id,
            include_blocked=args.include_blocked,
            limit=args.max_results,
            mode=args.prioritization_mode,
        )
    if args.allocate_budget:
        return orchestrator.allocate_budget(
            project_id=args.project_id,
            include_blocked=args.include_blocked,
            limit=args.max_results,
            mode=args.prioritization_mode,
            schedule_selected=args.schedule_selected,
        )
    if args.prioritization_log:
        return orchestrator.prioritization_log(count=args.max_results)
    if args.generate_falsification_plan:
        return orchestrator.generate_falsification_plan(args.project_id or "", force=args.force)
    if args.falsification_status:
        return orchestrator.falsification_status(args.project_id)
    if args.falsification_log:
        return orchestrator.falsification_log(count=args.max_results)
    if args.stale_projects:
        return orchestrator.stale_projects(limit=args.max_results, mode=args.prioritization_mode)
    if args.evidence_debt:
        return orchestrator.evidence_debt(
            project_id=args.project_id,
            limit=args.max_results,
            mode=args.prioritization_mode,
        )
    if args.resume_candidates:
        return orchestrator.resume_candidates(limit=args.max_results, mode=args.prioritization_mode)
    if args.list_benchmarks:
        return orchestrator.list_benchmarks(profile_id=args.profile_id, tier=args.benchmark_tier if args.benchmark_tier else None)  # type: ignore[arg-type]
    if args.benchmark_status:
        return orchestrator.benchmark_status(profile_id=args.profile_id, tier=args.benchmark_tier if args.benchmark_tier else None)  # type: ignore[arg-type]
    if args.run_problem_study:
        return orchestrator.run_problem_study(
            problem_id=args.problem_id,
            use_docker=args.use_docker,
            build_env=args.build_env,
        ).model_dump(mode="json")
    if args.schedule_problem_study:
        return orchestrator.schedule_problem_study(
            problem_id=args.problem_id,
            use_docker=args.use_docker,
            build_env=args.build_env,
            run_at=args.run_at,
            delay_s=args.delay_s,
            repeat_interval_s=args.repeat_interval_s,
            max_runs=args.max_runs,
            priority=args.priority,
        ).model_dump(mode="json")
    if args.scheduler_status:
        return orchestrator.scheduler_status()
    if args.run_scheduler_once:
        return orchestrator.run_scheduler_once(max_jobs=args.max_jobs).model_dump(mode="json")
    if args.prepare_payload_env:
        return orchestrator.prepare_payload_environment().model_dump(mode="json")
    if args.rebuild_locked_image:
        return orchestrator.rebuild_locked_image().model_dump(mode="json")
    if args.show_manifest:
        return orchestrator.show_manifest(manifest_id=args.manifest_id, manifest_path=args.manifest_path)
    if args.ingest_papers:
        return orchestrator.ingest_papers(args.paper_paths or []).model_dump(mode="json")
    if args.list_experiment_backends:
        return {"backends": orchestrator.list_experiment_backends()}
    if args.run_runtime_cycle:
        return orchestrator.run_runtime_cycle(max_jobs=args.max_jobs, stale_after_s=args.stale_after_s).model_dump(mode="json")
    if args.list_alerts:
        return orchestrator.list_alerts(count=args.alert_count)
    if args.retry_failed_job:
        return orchestrator.retry_failed_job(args.schedule_id or "").model_dump(mode="json")
    if args.cancel_job:
        return orchestrator.cancel_job(args.schedule_id or "").model_dump(mode="json")
    if args.sandbox_policy:
        return orchestrator.sandbox_policy()
    if args.register_checkpoint:
        return orchestrator.register_checkpoint(
            name=args.checkpoint_name or "",
            model_path=args.model_path or "",
            backend=args.backend_name,
            role=args.role_name,
            base_model_id=args.base_model_id,
            adapter_path=args.adapter_path,
            trust_remote_code=args.trust_remote_code,
        ).model_dump(mode="json")
    if args.list_checkpoints:
        return {"checkpoints": [item.model_dump(mode="json") for item in orchestrator.list_checkpoints()]}
    if args.build_inference_endpoint:
        return orchestrator.build_inference_endpoint(
            name=args.checkpoint_name or "",
            host=args.host,
            port=args.port,
            role=args.role_name,
            trust_remote_code=args.trust_remote_code,
        ).model_dump(mode="json")
    if args.list_endpoints:
        return {"endpoints": [item.model_dump(mode="json") for item in orchestrator.list_endpoints()]}
    if args.start_endpoint:
        return orchestrator.start_endpoint(
            name=args.checkpoint_name or "",
            host=args.host,
            port=args.port,
            role=args.role_name,
            trust_remote_code=args.trust_remote_code,
            wait_for_health=args.wait_for_health,
        ).model_dump(mode="json")
    if args.stop_endpoint:
        return orchestrator.stop_endpoint(args.endpoint_name or "").model_dump(mode="json")
    if args.restart_endpoint:
        return orchestrator.restart_endpoint(
            args.endpoint_name or "",
            trust_remote_code=args.trust_remote_code,
            wait_for_health=args.wait_for_health,
        ).model_dump(mode="json")
    if args.endpoint_health:
        return orchestrator.endpoint_health(args.endpoint_name or "")
    if args.assign_role:
        return orchestrator.assign_role(
            role=args.role_name,
            checkpoint_name=args.checkpoint_name or "",
            endpoint_name=args.endpoint_name,
        ).model_dump(mode="json")
    if args.select_operator_checkpoint:
        return orchestrator.select_operator_checkpoint(
            checkpoint_name=args.checkpoint_name or "",
            mode=args.operator_mode,
            role=args.role_name,
            endpoint_name=args.endpoint_name,
        ).model_dump(mode="json")
    if args.operator_serving_status:
        return orchestrator.operator_serving_status().model_dump(mode="json")
    if args.claim_policy:
        return orchestrator.claim_policy()
    if args.claim_verdict:
        return orchestrator.claim_verdict(
            trial_id=args.trial_id,
            problem_id=args.problem_id,
        ).model_dump(mode="json")
    if args.research_decision_log:
        return orchestrator.research_decision_log(count=args.max_results)
    if args.chat:
        return orchestrator.chat(_resolve_chat_prompt(orchestrator, args)).model_dump(mode="json")
    if args.pivot:
        return orchestrator.pivot_force(force=args.force)
    if args.explain:
        return orchestrator.explain_last_fail().model_dump(mode="json")
    if args.panic:
        return orchestrator.panic()
    raise ValueError("No command selected")


def _render_status(payload: Dict[str, Any]) -> str:
    recovery = payload["recovery"]
    last = payload["last_three_metrics"][-1] if payload["last_three_metrics"] else {}
    gpu = payload.get("gpu", {})
    memory = payload.get("memory", {})
    lines = [
        f"Trial ID: {recovery.get('trial_id') or 'none'}",
        f"Status: {recovery.get('status')}",
        f"Fail-Fast Streak: {recovery.get('consecutive_fail_fast', 0)}",
        f"Run Intent: {payload.get('run_intent', 'unknown')}",
        f"Backend: {payload.get('backend_id') or 'unknown'}",
        f"Backend Readiness: {payload.get('backend_readiness', 'unknown')}",
        f"Data Purity: {payload.get('data_purity', 'unknown')}",
        f"Tokenizer Integrity: {payload.get('tokenizer_integrity', False)}",
        f"Research Grade: {payload.get('research_grade', False)}",
        f"Image Tag: {payload.get('image_tag', 'n/a')}",
        f"Reproducibility Complete: {payload.get('reproducibility_complete', False)}",
        f"Manifest Hash: {payload.get('manifest_hash', 'n/a')}",
        f"Unresolved Dependencies: {payload.get('unresolved_dependency_count', 0)}",
        f"Sandbox Mode: {payload.get('safe_execution_mode', 'n/a')}",
        f"Sandbox Profile: {payload.get('sandbox_profile', 'n/a')}",
        f"Sandbox Dev Override: {payload.get('sandbox_dev_override_active', False)}",
        f"Read-Only Mounts: {', '.join(payload.get('sandbox_read_only_mounts', [])) or 'n/a'}",
        f"Writable Mounts: {', '.join(payload.get('sandbox_writable_mounts', [])) or 'n/a'}",
        f"Alert Count: {payload.get('alerts', 0)}",
        f"Endpoints: {len(payload.get('endpoints', []))}",
        f"Role Assignments: {len(payload.get('role_assignments', []))}",
        f"Research Projects: {payload.get('research_projects', 0)}",
        f"Active Research Projects: {payload.get('active_research_projects', 0)}",
        f"Prioritization Snapshots: {payload.get('prioritization_snapshots', 0)}",
        f"Budget Allocation Decisions: {payload.get('budget_allocation_decisions', 0)}",
        f"Falsification Plans: {payload.get('falsification_plans', 0)}",
        f"Portfolio Decisions: {payload.get('portfolio_decisions', 0)}",
        f"Evidence Debt Records: {payload.get('evidence_debt_records', 0)}",
        f"Staleness Records: {payload.get('project_staleness_records', 0)}",
        f"Publication Handoffs: {payload.get('publication_handoffs', 0)}",
        f"Latest Claim Verdict: {(payload.get('latest_claim_verdict') or {}).get('status', 'n/a')}",
        f"Benchmark: {payload.get('benchmark_name') or ', '.join(payload.get('benchmark_ids', [])) or 'n/a'}",
        f"Benchmark Tier: {payload.get('benchmark_tier', 'n/a')}",
        f"Benchmark Actual Tier(s): {', '.join(payload.get('actual_benchmark_tiers', [])) or 'n/a'}",
        f"Benchmark Truth: {', '.join(payload.get('benchmark_truth_statuses', [])) or 'n/a'}",
        f"Benchmark Alignment: {payload.get('benchmark_alignment', 'n/a')}",
        f"Canonical Comparable: {payload.get('canonical_comparable', 'n/a')}",
        f"Memory State: {memory.get('state', 'unknown')}",
        f"Memory Collection: {memory.get('collection_name', 'n/a')}",
        f"Memory Embedder: {memory.get('embedder', 'n/a')}",
        f"Memory Dim: {memory.get('embedding_dim', 'n/a')}",
        f"E: {last.get('energy_e', 'n/a')}",
        f"sigma: {last.get('entropy_sigma', 'n/a')}",
        f"rho: {last.get('drift_rho', 'n/a')}",
        f"grad: {last.get('grad_norm', 'n/a')}",
        f"D_PR: {last.get('effective_dimensionality', 'n/a')}",
        f"D ratio: {last.get('dimensionality_ratio', 'n/a')}",
        f"Eq frac: {last.get('equilibrium_fraction', 'n/a')}",
        f"GPU Temp C: {gpu.get('temperature_c', 'n/a')}",
        f"GPU Power W: {gpu.get('power_w', 'n/a')}",
    ]
    if payload.get("memory_warning"):
        lines.append(f"Memory Warning: {payload['memory_warning']}")
    if payload.get("lock_incomplete_reason"):
        lines.append(f"Lock Warning: {payload['lock_incomplete_reason']}")
    latest_project = payload.get("latest_research_project") or {}
    if latest_project:
        lines.append(f"Latest Project: {latest_project.get('project_id', 'n/a')} ({latest_project.get('status', 'n/a')})")
    latest_priority = payload.get("latest_priority_snapshot") or {}
    if latest_priority:
        lines.append(
            f"Latest Prioritized Action: {latest_priority.get('selected_action_id', 'n/a')} "
            f"for project {latest_priority.get('selected_project_id', 'n/a')}"
        )
    latest_falsification = payload.get("latest_falsification_plan") or {}
    if latest_falsification:
        lines.append(
            f"Latest Falsification Plan: {latest_falsification.get('plan_id', 'n/a')} "
            f"for project {latest_falsification.get('project_id', 'n/a')}"
        )
    latest_portfolio = payload.get("latest_portfolio") or {}
    if latest_portfolio:
        lines.append(
            f"Latest Portfolio Selected Project: {latest_portfolio.get('latest_selected_project_id', 'n/a')}"
        )
    latest_portfolio_decision = payload.get("latest_portfolio_decision") or {}
    if latest_portfolio_decision:
        lines.append(
            f"Latest Portfolio Decision: {latest_portfolio_decision.get('decision_id', 'n/a')} "
            f"selected {latest_portfolio_decision.get('selected_project_id', 'n/a')}"
        )
    latest_publication = payload.get("latest_publication_handoff") or {}
    if latest_publication:
        lines.append(
            f"Latest Publication Handoff: {latest_publication.get('package_id', 'n/a')} "
            f"for project {latest_publication.get('project_id', 'n/a')}"
        )
    operator_serving = payload.get("operator_serving") or {}
    operator_state = operator_serving.get("state") or {}
    operator_checkpoint = operator_serving.get("checkpoint") or {}
    if operator_state.get("active_checkpoint_name"):
        lines.append(
            f"Operator Serving: {operator_state.get('mode', 'n/a')} via "
            f"{operator_state.get('active_checkpoint_name', 'n/a')}"
        )
        lines.append(f"Operator Endpoint: {operator_state.get('endpoint_name', 'n/a')}")
        lines.append(f"Operator Checkpoint Kind: {operator_checkpoint.get('checkpoint_kind', 'n/a')}")
    return "\n".join(lines)


def _render_sandbox_policy(payload: Dict[str, Any]) -> str:
    lines = [
        f"Sandbox Mode: {payload.get('mode', 'n/a')}",
        f"Sandbox Profile: {payload.get('profile', 'n/a')}",
        f"Dev Override Active: {payload.get('dev_override_active', False)}",
        f"Network Policy: {payload.get('network_policy', 'n/a')}",
        f"Workspace Root: {payload.get('workspace_root', 'n/a')}",
        f"Artifact Dir: {payload.get('artifact_dir', 'n/a')}",
        f"Read-Only Mounts: {', '.join(payload.get('read_only_mounts', [])) or 'n/a'}",
        f"Writable Mounts: {', '.join(payload.get('writable_mounts', [])) or 'n/a'}",
        f"Allowed Mounts: {', '.join(payload.get('allowed_mounts', [])) or 'n/a'}",
    ]
    return "\n".join(lines)


def _render_runtime_status(payload: Dict[str, Any]) -> str:
    sandbox = payload.get("sandbox_policy", {})
    lines = [
        f"Sandbox Mode: {payload.get('safe_execution_mode', 'n/a')}",
        f"Sandbox Profile: {sandbox.get('profile', 'n/a')}",
        f"Dev Override Active: {sandbox.get('dev_override_active', False)}",
        f"Payload Image: {payload.get('payload_image', 'n/a')}",
        f"Manifest Hash: {payload.get('manifest_hash', 'n/a')}",
        f"Reproducibility Complete: {payload.get('reproducibility_complete', False)}",
        f"Read-Only Mounts: {', '.join(sandbox.get('read_only_mounts', [])) or 'n/a'}",
        f"Writable Mounts: {', '.join(sandbox.get('writable_mounts', [])) or 'n/a'}",
        f"Active Leases: {len(payload.get('active_leases', []))}",
        f"Retry Waiting: {len(payload.get('retry_waiting', []))}",
        f"Terminal Failures: {len(payload.get('terminal_failures', []))}",
        f"Alert Count: {len(payload.get('alerts', []))}",
    ]
    if payload.get("lock_incomplete_reason"):
        lines.append(f"Lock Warning: {payload['lock_incomplete_reason']}")
    return "\n".join(lines)


def _render_endpoint_record(endpoint: Dict[str, Any]) -> str:
    health = endpoint.get("health") or {}
    lines = [
        f"Endpoint: {endpoint.get('endpoint_name', 'n/a')}",
        f"Status: {endpoint.get('status', 'n/a')}",
        f"Role: {endpoint.get('role', 'n/a')}",
        f"Checkpoint: {endpoint.get('checkpoint_name', 'n/a')}",
        f"Backend: {endpoint.get('backend', 'n/a')}",
        f"Base URL: {endpoint.get('base_url', 'n/a')}",
        f"Trust Remote Code: {endpoint.get('trust_remote_code', False)}",
        f"Process PID: {endpoint.get('process_pid', 'n/a')}",
        f"Last Health: {health.get('status', 'n/a')} at {endpoint.get('last_health_at', 'n/a')}",
        f"Last Error: {endpoint.get('last_error', 'n/a')}",
        f"Stdout Log: {endpoint.get('stdout_log_path', 'n/a')}",
        f"Stderr Log: {endpoint.get('stderr_log_path', 'n/a')}",
        f"Manifest: {endpoint.get('manifest_path', 'n/a')}",
    ]
    return "\n".join(lines)


def _render_endpoint_list(payload: Dict[str, Any]) -> str:
    endpoints = payload.get("endpoints", [])
    if not endpoints:
        return "No managed inference endpoints registered."
    return "\n\n".join(_render_endpoint_record(endpoint) for endpoint in endpoints)


def _render_endpoint_health(payload: Dict[str, Any]) -> str:
    endpoint = payload.get("endpoint", {})
    health = payload.get("health", {})
    lines = [
        f"Endpoint: {endpoint.get('endpoint_name', 'n/a')}",
        f"Status: {endpoint.get('status', 'n/a')}",
        f"Health: {health.get('status', 'n/a')}",
        f"Healthy: {health.get('ok', False)}",
        f"Backend: {health.get('backend', endpoint.get('backend', 'n/a'))}",
        f"Model: {health.get('model_id', 'n/a')}",
        f"Role: {health.get('role', endpoint.get('role', 'n/a'))}",
        f"Trust Remote Code: {health.get('trust_remote_code', endpoint.get('trust_remote_code', False))}",
        f"Detail: {health.get('detail', endpoint.get('last_error', 'n/a'))}",
        f"Checked At: {health.get('checked_at', endpoint.get('last_health_at', 'n/a'))}",
        f"Stdout Log: {endpoint.get('stdout_log_path', 'n/a')}",
        f"Stderr Log: {endpoint.get('stderr_log_path', 'n/a')}",
    ]
    return "\n".join(lines)


def _render_operator_serving_status(payload: Dict[str, Any]) -> str:
    state = payload.get("state", {})
    checkpoint = payload.get("checkpoint", {})
    endpoint = payload.get("endpoint", {})
    role_assignment = payload.get("role_assignment", {})
    lines = [
        f"Operator Mode: {state.get('mode', 'n/a')}",
        f"Active Checkpoint: {state.get('active_checkpoint_name', 'n/a')}",
        f"Role: {state.get('role', 'n/a')}",
        f"Selected Endpoint: {state.get('endpoint_name', 'n/a')}",
        f"Checkpoint Kind: {checkpoint.get('checkpoint_kind', 'n/a')}",
        f"Base Model: {checkpoint.get('base_model_id', checkpoint.get('model_path', 'n/a'))}",
        f"Adapter Path: {checkpoint.get('adapter_path', 'n/a')}",
        f"Endpoint Status: {endpoint.get('status', 'n/a')}",
        f"Role Assignment: {role_assignment.get('checkpoint_name', 'n/a')}",
    ]
    return "\n".join(lines)


def _render_regime(payload: Dict[str, Any]) -> str:
    lines = [
        f"Trial ID: {payload.get('trial_id', 'none')}",
        f"Step: {payload.get('step', 'n/a')}",
        f"Regime: {payload.get('regime', 'unknown')}",
        f"D_PR: {payload.get('effective_dimensionality', 'n/a')}",
        f"D ratio: {payload.get('dimensionality_ratio', 'n/a')}",
        f"Eq frac: {payload.get('equilibrium_fraction', 'n/a')}",
        f"Eq gate: {payload.get('equilibrium_gate', 'n/a')}",
        f"Loss: {payload.get('training_loss', 'n/a')}",
    ]
    warning = payload.get("warning")
    if warning:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def _render_project_status(payload: Dict[str, Any]) -> str:
    project = payload.get("project", {})
    thread = payload.get("active_thread") or {}
    question = payload.get("current_question") or {}
    action = payload.get("next_action") or {}
    budget = payload.get("budget_remaining") or {}
    lines = [
        f"Project: {project.get('title', project.get('project_id', 'n/a'))}",
        f"Project ID: {project.get('project_id', 'n/a')}",
        f"Status: {project.get('status', 'n/a')}",
        f"Domain Profile: {project.get('domain_profile', 'n/a')}",
        f"Priority: {project.get('priority', 'n/a')}",
        f"Latest Decision: {project.get('latest_decision_summary', 'n/a')}",
        f"Active Thread: {thread.get('thread_id', 'n/a')}",
        f"Thread Status: {thread.get('status', 'n/a')}",
        f"Confidence State: {thread.get('confidence_state', 'n/a')}",
        f"Current Question: {question.get('question', 'n/a')}",
        f"Question Status: {question.get('status', 'n/a')}",
        f"Next Action: {action.get('description', 'n/a')}",
        f"Next Action ID: {action.get('action_id', 'n/a')}",
        f"Next Action Status: {action.get('status', 'n/a')}",
        f"Budget Remaining: {json.dumps(budget, sort_keys=True)}",
    ]
    resume_snapshot = project.get("resume_snapshot") or {}
    if resume_snapshot:
        lines.append(f"Resume Snapshot At: {resume_snapshot.get('captured_at', 'n/a')}")
        blockers = resume_snapshot.get("blockers") or []
        lines.append(f"Resume Blockers: {', '.join(blockers) or 'n/a'}")
    if thread.get("stop_reason"):
        lines.append(f"Stop Reason: {thread.get('stop_reason')}")
    if thread.get("resume_reason"):
        lines.append(f"Resume Reason: {thread.get('resume_reason')}")
    return "\n".join(lines)


def _render_project_list(payload: Dict[str, Any]) -> str:
    projects = payload.get("projects", [])
    if not projects:
        return "No research projects recorded."
    return "\n\n".join(_render_project_status(project) for project in projects)


def _render_next_action(payload: Dict[str, Any]) -> str:
    action = payload.get("next_action") or {}
    question = payload.get("current_question") or {}
    return "\n".join(
        [
            f"Project ID: {payload.get('project_id', 'n/a')}",
            f"Project Status: {payload.get('project_status', 'n/a')}",
            f"Current Question: {question.get('question', 'n/a')}",
            f"Next Action: {action.get('description', 'n/a')}",
            f"Action Kind: {action.get('action_kind', 'n/a')}",
            f"Action Status: {action.get('status', 'n/a')}",
            f"Estimated Cost: {action.get('estimated_cost', 'n/a')}",
            f"Expected Evidence Gain: {action.get('expected_evidence_gain', 'n/a')}",
            f"Budget Remaining: {json.dumps(payload.get('budget_remaining', {}), sort_keys=True)}",
        ]
    )


def _render_operator_view(payload: Dict[str, Any]) -> str:
    counts = payload.get("project_counts") or {}
    health = payload.get("portfolio_health") or {}
    lines = [
        f"Generated At: {payload.get('generated_at', 'n/a')}",
        f"Projects: total={counts.get('total', 0)} active={counts.get('active', 0)} paused={counts.get('paused', 0)} blocked={counts.get('blocked', 0)} stale={counts.get('stale', 0)}",
        f"Portfolio Health: selected={health.get('selected_project_id', 'n/a')} resume_candidates={health.get('resume_candidates', 0)} promotion_blocked={health.get('promotion_blocked_projects', 0)}",
        "",
        "Active Projects:",
    ]
    active_projects = payload.get("active_projects") or []
    if active_projects:
        for item in active_projects:
            lines.append(
                f"- {item.get('title', item.get('project_id', 'n/a'))} :: status={item.get('status', 'n/a')} next={item.get('next_action', 'n/a')}"
            )
    else:
        lines.append("No active projects.")
    lines.extend(
        [
            "",
            "Top Action Candidates:",
            _render_ranked_candidates(payload.get("top_candidates", [])),
        ]
    )
    promotion_blocked = payload.get("promotion_blocked_projects") or []
    if promotion_blocked:
        lines.append("")
        lines.append(
            "Promotion Blocked Projects: "
            + ", ".join(item.get("project_id", "n/a") for item in promotion_blocked)
        )
    return "\n".join(lines)


def _render_project_timeline(payload: Dict[str, Any]) -> str:
    events = payload.get("events", [])
    if not events:
        return "No project timeline events recorded."
    lines = [
        f"Project: {payload.get('project_title', payload.get('project_id', 'n/a'))}",
        f"Project ID: {payload.get('project_id', 'n/a')}",
        f"Event Count: {payload.get('event_count', len(events))}",
        "",
        "Timeline:",
    ]
    for item in events:
        lines.append(
            f"- {item.get('timestamp', 'n/a')} :: {item.get('event_type', 'event')} :: {item.get('summary', 'n/a')}"
        )
    return "\n".join(lines)


def _render_evidence_map(payload: Dict[str, Any]) -> str:
    counts = payload.get("evidence_counts") or {}
    benchmark_context = payload.get("benchmark_context") or {}
    latest_debt = payload.get("latest_evidence_debt") or {}
    lines = [
        f"Project: {payload.get('project_title', payload.get('project_id', 'n/a'))}",
        f"Project ID: {payload.get('project_id', 'n/a')}",
        f"Evidence Summary: {payload.get('latest_evidence_summary', 'n/a')}",
        f"Supporting Evidence: {counts.get('supporting', 0)}",
        f"Contradicting Evidence: {counts.get('contradicting', 0)}",
        f"Cited Research IDs: {', '.join(payload.get('cited_research_ids', [])) or 'n/a'}",
        f"Retrieved Memory IDs: {', '.join(payload.get('retrieved_memory_ids', [])) or 'n/a'}",
        f"Benchmark Context: ids={', '.join(benchmark_context.get('benchmark_ids', [])) or 'n/a'} alignment={benchmark_context.get('benchmark_alignment', 'n/a')} comparable={benchmark_context.get('canonical_comparable', False)}",
        f"Latest Evidence Debt: overall={latest_debt.get('overall_debt', 'n/a')} blocked={latest_debt.get('promotion_blocked', False)}",
    ]
    contradiction_review = payload.get("contradiction_review")
    if contradiction_review:
        lines.append("Contradiction Review Present: yes")
    claim_verdicts = payload.get("claim_verdicts") or []
    if claim_verdicts:
        lines.append("Claim Verdicts:")
        for item in claim_verdicts:
            lines.append(
                f"- {item.get('verdict_id', 'n/a')} :: status={item.get('status', 'n/a')} linkage={item.get('linkage_status', 'n/a')} confidence={item.get('confidence', 'n/a')}"
            )
    return "\n".join(lines)


def _render_claim_lineage(payload: Dict[str, Any]) -> str:
    lines = [
        f"Project: {payload.get('project_title', payload.get('project_id', 'n/a'))}",
        f"Project ID: {payload.get('project_id', 'n/a')}",
        f"Problem IDs: {', '.join(payload.get('problem_ids', [])) or 'n/a'}",
        f"Studies: {len(payload.get('studies', []))}",
        f"Executions: {len(payload.get('executions', []))}",
        f"Research Decisions: {len(payload.get('research_decisions', []))}",
        f"Claim Verdicts: {len(payload.get('verdicts', []))}",
    ]
    verdicts = payload.get("verdicts") or []
    if verdicts:
        lines.append("Verdicts:")
        for item in verdicts:
            lines.append(
                f"- {item.get('created_at', 'n/a')} :: {item.get('status', 'n/a')} benchmark_problem={item.get('benchmark_problem_id', 'n/a')} linkage={item.get('linkage_status', 'n/a')} source={item.get('canonical_comparability_source', 'n/a')}"
            )
    return "\n".join(lines)


def _render_resume_dashboard(payload: Dict[str, Any]) -> str:
    project_payload = payload.get("project") or {}
    project = project_payload.get("project", project_payload) if isinstance(project_payload, dict) else {}
    next_action = payload.get("next_action") or {}
    latest_debt = payload.get("latest_evidence_debt") or {}
    latest_staleness = payload.get("latest_staleness") or {}
    lines = [
        f"Project: {project.get('title', project.get('project_id', 'n/a'))}",
        f"Project ID: {project.get('project_id', 'n/a')}",
        f"Resume State: {payload.get('resume_state', 'n/a')}",
        f"Blockers: {', '.join(payload.get('blockers', [])) or 'none'}",
        f"Next Action: {next_action.get('description', 'n/a')}",
        f"Budget Remaining: {json.dumps(payload.get('budget_remaining', {}), sort_keys=True)}",
        f"Latest Evidence Debt: overall={latest_debt.get('overall_debt', 'n/a')} blocked={latest_debt.get('promotion_blocked', False)}",
        f"Latest Staleness: level={latest_staleness.get('staleness_level', 'n/a')} resume_candidate={latest_staleness.get('resume_candidate', False)}",
    ]
    priority = payload.get("latest_priority_record") or {}
    if priority:
        lines.append(
            f"Latest Priority Record: score={priority.get('priority_score', 'n/a')} state={priority.get('recommended_state', 'n/a')}"
        )
    plan = payload.get("latest_falsification_plan") or {}
    if plan:
        lines.append(
            f"Latest Falsification Plan: {plan.get('plan_id', 'n/a')} status={plan.get('status', 'n/a')} tests={len(plan.get('tests', []))}"
        )
    return "\n".join(lines)


def _render_publication_handoff(payload: Dict[str, Any]) -> str:
    package = payload.get("package", payload)
    accepted = package.get("accepted_claims", [])
    provisional = package.get("provisional_claims", [])
    rejected = package.get("rejected_alternatives", [])
    lines = [
        f"Publication Package: {package.get('package_id', 'n/a')}",
        f"Project: {package.get('project_title', package.get('project_id', 'n/a'))}",
        f"Project ID: {package.get('project_id', 'n/a')}",
        f"Status: {package.get('package_status', 'n/a')}",
        f"Claim Readiness: {package.get('claim_readiness_summary', 'n/a')}",
        f"Accepted Claims: {len(accepted)}",
        f"Provisional Claims: {len(provisional)}",
        f"Rejected Alternatives: {len(rejected)}",
        f"Benchmark Attachments: {len(package.get('benchmark_truth_attachments', []))}",
        f"Lineage Events: {len(package.get('experiment_lineage', []))}",
        f"Artifact Path: {package.get('artifact_path', 'n/a')}",
    ]
    if accepted:
        lines.append("Accepted Claim Bundles:")
        for item in accepted[:5]:
            lines.append(
                f"- {item.get('verdict_id', 'n/a')} :: confidence={item.get('confidence', 'n/a')} :: {item.get('summary', 'n/a')}"
            )
    if provisional:
        lines.append("Provisional Claim Bundles:")
        for item in provisional[:5]:
            lines.append(
                f"- {item.get('verdict_id', 'n/a')} :: confidence={item.get('confidence', 'n/a')} :: {item.get('summary', 'n/a')}"
            )
    if package.get("limitations"):
        lines.append("Limitations:")
        for item in package.get("limitations", [])[:5]:
            lines.append(f"- {item}")
    if package.get("open_questions"):
        lines.append("Open Questions:")
        for item in package.get("open_questions", [])[:5]:
            lines.append(f"- {item}")
    if package.get("writer_cautions"):
        lines.append("Writer Cautions:")
        for item in package.get("writer_cautions", [])[:5]:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _render_publication_log(payload: Dict[str, Any]) -> str:
    packages = payload.get("packages", [])
    if not packages:
        return "No publication handoff packages recorded."
    lines = ["Publication Handoff Packages:"]
    for item in packages:
        lines.append(
            f"- {item.get('package_id', 'n/a')} :: project={item.get('project_id', 'n/a')} :: status={item.get('package_status', 'n/a')} :: accepted={len(item.get('accepted_claims', []))} provisional={len(item.get('provisional_claims', []))}"
        )
    return "\n".join(lines)


def _render_ranked_candidates(candidates: list[Dict[str, Any]]) -> str:
    if not candidates:
        return "No prioritized action candidates were available."
    blocks: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        breakdown = candidate.get("score_breakdown") or {}
        blocks.append(
            "\n".join(
                [
                    f"{index}. {candidate.get('project_title', 'n/a')} :: {candidate.get('action_description', 'n/a')}",
                    f"   Score: {candidate.get('score', 'n/a')}",
                    f"   Recommended: {candidate.get('recommended', False)}",
                    f"   Project Status: {candidate.get('project_status', 'n/a')}",
                    f"   Action Kind: {candidate.get('action_kind', 'n/a')}",
                    f"   Blocked: {candidate.get('blocked', False)}",
                    f"   Question: {candidate.get('current_question', 'n/a')}",
                    f"   Breakdown: evidence={breakdown.get('expected_evidence_gain', 'n/a')}, "
                    f"falsification={breakdown.get('falsification_value', 'n/a')}, "
                    f"uncertainty={breakdown.get('uncertainty_reduction', 'n/a')}, "
                    f"cost_penalty={breakdown.get('cost_penalty', 'n/a')}, "
                    f"budget_penalty={breakdown.get('budget_pressure_penalty', 'n/a')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_portfolio_status(payload: Dict[str, Any]) -> str:
    counts = payload.get("project_counts") or {}
    lines = [
        f"Projects: {counts.get('total', 0)}",
        f"Active: {counts.get('active', 0)}",
        f"Blocked: {counts.get('blocked', 0)}",
        f"Paused: {counts.get('paused', 0)}",
        "",
        "Top Candidates:",
        _render_ranked_candidates(payload.get("top_candidates", [])),
    ]
    latest = payload.get("latest_budget_allocation") or {}
    if latest:
        lines.extend(
            [
                "",
                f"Latest Allocation Decision: {latest.get('decision_id', 'n/a')}",
                f"Schedule Created: {latest.get('schedule_created', False)}",
            ]
        )
    return "\n".join(lines)


def _render_budget_allocation(payload: Dict[str, Any]) -> str:
    candidate = payload.get("selected_candidate") or {}
    lines = [
        f"Allocation Decision: {payload.get('decision_id', 'n/a')}",
        f"Schedule Created: {payload.get('schedule_created', False)}",
        f"Scheduled Job ID: {payload.get('scheduled_job_id', 'n/a')}",
        f"Selected Project: {candidate.get('project_title', candidate.get('project_id', 'n/a'))}",
        f"Selected Action: {candidate.get('action_description', 'n/a')}",
        f"Selected Score: {candidate.get('score', 'n/a')}",
        f"Selected Blocked: {candidate.get('blocked', False)}",
    ]
    rationale = payload.get("rationale") or []
    if rationale:
        lines.append(f"Rationale: {'; '.join(rationale)}")
    return "\n".join(lines)


def _render_prioritization_log(payload: Dict[str, Any]) -> str:
    snapshots = payload.get("snapshots", [])
    if not snapshots:
        return "No prioritization snapshots recorded."
    lines: list[str] = []
    for snapshot in snapshots:
        lines.append(
            f"{snapshot.get('created_at', 'n/a')} :: selected {snapshot.get('selected_action_id', 'n/a')} "
            f"for {snapshot.get('selected_project_id', 'n/a')} ({snapshot.get('candidate_count', 0)} candidates)"
        )
    return "\n".join(lines)


def _render_portfolio_review(payload: Dict[str, Any]) -> str:
    portfolio = payload.get("portfolio") or {}
    health = portfolio.get("health_snapshot") or {}
    lines = [
        f"Portfolio: {portfolio.get('portfolio_id', 'n/a')}",
        f"Selected Project: {portfolio.get('latest_selected_project_id', 'n/a')}",
        f"Active Projects: {len(portfolio.get('active_project_ids', []))}",
        f"Paused Projects: {len(portfolio.get('paused_project_ids', []))}",
        f"Blocked Projects: {len(portfolio.get('blocked_project_ids', []))}",
        f"Stale Projects: {len(portfolio.get('stale_project_ids', []))}",
        f"Parked Projects: {len(portfolio.get('parked_project_ids', []))}",
        f"Resume Candidates: {health.get('resume_candidates', 0)}",
        f"Promotion Blocked: {health.get('promotion_blocked_projects', 0)}",
        "",
        "Top Projects:",
    ]
    top_projects = payload.get("top_projects", [])
    if top_projects:
        for item in top_projects:
            lines.append(
                f"- {item.get('project_id', 'n/a')}: score={item.get('priority_score', 'n/a')} "
                f"state={item.get('recommended_state', 'n/a')} debt={item.get('evidence_debt', 'n/a')}"
            )
    else:
        lines.append("No portfolio priority records available.")
    return "\n".join(lines)


def _render_portfolio_decision(payload: Dict[str, Any]) -> str:
    decision = payload.get("decision") or {}
    lines = [
        f"Portfolio Decision: {decision.get('decision_id', 'n/a')}",
        f"Selected Project: {decision.get('selected_project_id', 'n/a')}",
        f"Selected Action: {decision.get('selected_action_id', 'n/a')}",
        f"Deferred: {', '.join(decision.get('deferred_project_ids', [])) or 'n/a'}",
        f"Parked: {', '.join(decision.get('parked_project_ids', [])) or 'n/a'}",
        f"Resumed: {', '.join(decision.get('resumed_project_ids', [])) or 'n/a'}",
        f"Escalated: {', '.join(decision.get('escalated_project_ids', [])) or 'n/a'}",
        f"Retired: {', '.join(decision.get('retired_project_ids', [])) or 'n/a'}",
    ]
    rationale = decision.get("rationale") or []
    if rationale:
        lines.append(f"Rationale: {'; '.join(rationale)}")
    return "\n".join(lines)


def _render_stale_projects(payload: Dict[str, Any]) -> str:
    rows = payload.get("stale_projects", [])
    if not rows:
        return "No stale projects detected."
    lines = []
    for item in rows:
        lines.append(
            f"{item.get('project_id', 'n/a')} :: level={item.get('staleness_level', 'n/a')} "
            f"hours={item.get('hours_since_progress', 'n/a')} resume={item.get('resume_candidate', False)} "
            f"closure={item.get('closure_candidate', False)}"
        )
    return "\n".join(lines)


def _render_evidence_debt(payload: Dict[str, Any]) -> str:
    rows = payload.get("records", [])
    if not rows:
        return "No evidence debt records available."
    lines = []
    for item in rows:
        lines.append(
            f"{item.get('project_id', 'n/a')} :: overall={item.get('overall_debt', 'n/a')} "
            f"blocked={item.get('promotion_blocked', False)} "
            f"falsification={item.get('falsification_gap', 'n/a')} "
            f"replication={item.get('replication_gap', 'n/a')} "
            f"benchmark={item.get('benchmark_gap', 'n/a')}"
        )
    return "\n".join(lines)


def _render_resume_candidates(payload: Dict[str, Any]) -> str:
    rows = payload.get("resume_candidates", [])
    if not rows:
        return "No resume candidates detected."
    lines = []
    for item in rows:
        staleness = item.get("staleness") or {}
        priority = item.get("priority") or {}
        lines.append(
            f"{staleness.get('project_id', 'n/a')} :: level={staleness.get('staleness_level', 'n/a')} "
            f"score={priority.get('priority_score', 'n/a')} state={priority.get('recommended_state', 'n/a')}"
        )
    return "\n".join(lines)


def _render_falsification_plan(payload: Dict[str, Any]) -> str:
    plan = payload.get("plan") or {}
    project_payload = payload.get("project") or {}
    project = project_payload.get("project", project_payload) if isinstance(project_payload, dict) else {}
    coverage = payload.get("coverage") or plan.get("coverage") or {}
    pending_tests = payload.get("pending_tests") or plan.get("tests") or []
    lines = [
        f"Project: {project.get('title', project.get('project_id', payload.get('project_id', 'n/a')))}",
        f"Project ID: {project.get('project_id', payload.get('project_id', 'n/a'))}",
        f"Generated: {payload.get('generated', plan != {})}",
        f"Plan ID: {plan.get('plan_id', 'n/a')}",
        f"Plan Status: {plan.get('status', 'n/a')}",
        f"Trigger Reason: {plan.get('trigger_reason', payload.get('reason', 'n/a'))}",
        f"Pending Tests: {len(pending_tests)}",
        f"Coverage: ablation={coverage.get('ablation_coverage', 'n/a')}, "
        f"replication={coverage.get('replication_coverage', 'n/a')}, "
        f"contradiction={coverage.get('contradiction_coverage', 'n/a')}, "
        f"benchmark={coverage.get('benchmark_pressure_coverage', 'n/a')}, "
        f"calibration={coverage.get('calibration_coverage', 'n/a')}, "
        f"overall={coverage.get('overall_sufficient', 'n/a')}",
    ]
    tests = pending_tests or []
    if tests:
        lines.append("Tests:")
        for test in tests:
            lines.append(
                f"- {test.get('kind', 'n/a')}: {test.get('description', 'n/a')} "
                f"(status={test.get('status', 'n/a')}, falsification_value={test.get('expected_falsification_value', 'n/a')})"
            )
    attached_actions = payload.get("attached_actions") or []
    if attached_actions:
        lines.append("Attached Actions:")
        for action in attached_actions:
            lines.append(
                f"- {action.get('action_id', 'n/a')}: {action.get('action_kind', 'n/a')} "
                f"(cost={action.get('estimated_cost', 'n/a')}, evidence_gain={action.get('expected_evidence_gain', 'n/a')})"
            )
    return "\n".join(lines)


def _render_falsification_log(payload: Dict[str, Any]) -> str:
    plans = payload.get("plans", [])
    if not plans:
        return "No falsification plans recorded."
    lines: list[str] = []
    for plan in plans:
        lines.append(
            f"{plan.get('created_at', 'n/a')} :: {plan.get('plan_id', 'n/a')} "
            f"for {plan.get('project_id', 'n/a')} status={plan.get('status', 'n/a')} "
            f"tests={len(plan.get('tests', []))}"
        )
    return "\n".join(lines)


def _resolve_chat_prompt(orchestrator: TAROrchestrator, args: argparse.Namespace) -> str:
    if args.message:
        return args.message
    if args.voice_file:
        from tar_lab.voice import SpeechProcessor

        processor = SpeechProcessor(
            workspace=orchestrator.workspace,
            wake_word=args.wake_word,
        )
        processor.start()
        try:
            processor.submit_audio(args.voice_file)
            prompt = processor.poll_command(timeout=args.listen_timeout)
        finally:
            processor.stop()
        if not prompt:
            raise RuntimeError("No wake-word command was detected in the supplied audio file")
        return prompt
    if args.listen:
        from tar_lab.voice import SpeechProcessor

        processor = SpeechProcessor(
            workspace=orchestrator.workspace,
            wake_word=args.wake_word,
        )
        processor.start()
        try:
            prompt = processor.listen_once(
                duration_s=args.listen_duration,
                timeout=args.listen_timeout,
            )
        finally:
            processor.stop()
        if not prompt:
            raise RuntimeError("No wake-word command was detected from the microphone")
        return prompt
    return input("lab> ").strip()


def main() -> int:
    args = parse_args()
    orchestrator = TAROrchestrator(workspace=args.workspace, start_memory_indexer=args.serve)
    try:
        orchestrator.store.append_audit_event("cli", "command", {"argv": vars(args)})

        if args.serve:
            server = TARControlServer(orchestrator, host=args.host, port=args.port)
            try:
                server.serve_forever()
            finally:
                server.shutdown()
            return 0

        if args.direct:
            payload = _direct_dispatch(orchestrator, args)
        else:
            if args.status:
                response = send_command("status", host=args.host, port=args.port)
            elif args.frontier_status:
                response = send_command("frontier_status", host=args.host, port=args.port)
            elif args.runtime_status:
                response = send_command("runtime_status", host=args.host, port=args.port)
            elif args.dry_run:
                response = send_command("dry_run", payload={"force_fail_fast": args.force_fail_fast}, host=args.host, port=args.port)
            elif args.live_docker_test:
                response = send_command("live_docker_test", host=args.host, port=args.port)
            elif args.check_regime:
                response = send_command("check_regime", host=args.host, port=args.port)
            elif args.ingest_research:
                response = send_command(
                    "ingest_research",
                    payload={"topic": args.topic, "max_results": args.max_results},
                    host=args.host,
                    port=args.port,
                )
            elif args.verify_last_trial:
                response = send_command(
                    "verify_last_trial",
                    payload={"trial_id": args.trial_id},
                    host=args.host,
                    port=args.port,
                )
            elif args.breakthrough_report:
                response = send_command(
                    "breakthrough_report",
                    payload={"trial_id": args.trial_id, "problem_id": args.problem_id},
                    host=args.host,
                    port=args.port,
                )
            elif args.resolve_problem:
                response = send_command(
                    "resolve_problem",
                    payload={
                        "problem": args.problem or args.message or "",
                        "benchmark_tier": args.benchmark_tier,
                        "benchmark": args.benchmark,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.prepare_science_env:
                response = send_command(
                    "prepare_science_env",
                    payload={
                        "problem": args.problem or args.message or "",
                        "build": args.build_env,
                        "benchmark_tier": args.benchmark_tier,
                        "benchmark": args.benchmark,
                        "canonical_only": args.canonical_only,
                        "no_proxy_benchmarks": args.no_proxy_benchmarks,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.study_problem:
                response = send_command(
                    "study_problem",
                    payload={
                        "problem": args.problem or args.message or "",
                        "project_id": args.project_id,
                        "build_env": args.build_env,
                        "max_results": args.max_results,
                        "benchmark_tier": args.benchmark_tier,
                        "benchmark": args.benchmark,
                        "canonical_only": args.canonical_only,
                        "no_proxy_benchmarks": args.no_proxy_benchmarks,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.create_project:
                response = send_command(
                    "create_project",
                    payload={
                        "problem": args.problem or args.message or "",
                        "benchmark_tier": args.benchmark_tier,
                        "benchmark": args.benchmark,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.list_projects:
                response = send_command("list_projects", host=args.host, port=args.port)
            elif args.project_status:
                response = send_command(
                    "project_status",
                    payload={"project_id": args.project_id or ""},
                    host=args.host,
                    port=args.port,
                )
            elif args.pause_project:
                response = send_command(
                    "pause_project",
                    payload={
                        "project_id": args.project_id or "",
                        "reason": args.pause_reason,
                        "note": args.message,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.resume_project:
                response = send_command(
                    "resume_project",
                    payload={
                        "project_id": args.project_id or "",
                        "reason": args.resume_reason,
                        "note": args.message,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.next_action:
                response = send_command(
                    "next_action",
                    payload={"project_id": args.project_id or ""},
                    host=args.host,
                    port=args.port,
                )
            elif args.operator_view:
                response = send_command(
                    "operator_view",
                    payload={
                        "include_blocked": args.include_blocked,
                        "limit": args.max_results,
                        "mode": args.prioritization_mode,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.project_timeline:
                response = send_command(
                    "project_timeline",
                    payload={"project_id": args.project_id or "", "limit": args.max_results},
                    host=args.host,
                    port=args.port,
                )
            elif args.evidence_map:
                response = send_command(
                    "evidence_map",
                    payload={"project_id": args.project_id or ""},
                    host=args.host,
                    port=args.port,
                )
            elif args.claim_lineage:
                response = send_command(
                    "claim_lineage",
                    payload={"project_id": args.project_id or ""},
                    host=args.host,
                    port=args.port,
                )
            elif args.resume_dashboard:
                response = send_command(
                    "resume_dashboard",
                    payload={"project_id": args.project_id or ""},
                    host=args.host,
                    port=args.port,
                )
            elif args.publication_handoff:
                response = send_command(
                    "publication_handoff",
                    payload={"project_id": args.project_id or ""},
                    host=args.host,
                    port=args.port,
                )
            elif args.publication_log:
                response = send_command(
                    "publication_log",
                    payload={"count": args.max_results},
                    host=args.host,
                    port=args.port,
                )
            elif args.portfolio_status:
                response = send_command(
                    "portfolio_status",
                    payload={
                        "include_blocked": args.include_blocked,
                        "limit": args.max_results,
                        "mode": args.prioritization_mode,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.portfolio_review:
                response = send_command(
                    "portfolio_review",
                    payload={
                        "include_blocked": args.include_blocked,
                        "limit": args.max_results,
                        "mode": args.prioritization_mode,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.portfolio_decide:
                response = send_command(
                    "portfolio_decide",
                    payload={
                        "include_blocked": args.include_blocked,
                        "limit": args.max_results,
                        "mode": args.prioritization_mode,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.rank_actions:
                response = send_command(
                    "rank_actions",
                    payload={
                        "project_id": args.project_id,
                        "include_blocked": args.include_blocked,
                        "limit": args.max_results,
                        "mode": args.prioritization_mode,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.allocate_budget:
                response = send_command(
                    "allocate_budget",
                    payload={
                        "project_id": args.project_id,
                        "include_blocked": args.include_blocked,
                        "limit": args.max_results,
                        "mode": args.prioritization_mode,
                        "schedule_selected": args.schedule_selected,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.prioritization_log:
                response = send_command(
                    "prioritization_log",
                    payload={"count": args.max_results},
                    host=args.host,
                    port=args.port,
                )
            elif args.generate_falsification_plan:
                response = send_command(
                    "generate_falsification_plan",
                    payload={"project_id": args.project_id or "", "force": args.force},
                    host=args.host,
                    port=args.port,
                )
            elif args.falsification_status:
                response = send_command(
                    "falsification_status",
                    payload={"project_id": args.project_id},
                    host=args.host,
                    port=args.port,
                )
            elif args.falsification_log:
                response = send_command(
                    "falsification_log",
                    payload={"count": args.max_results},
                    host=args.host,
                    port=args.port,
                )
            elif args.stale_projects:
                response = send_command(
                    "stale_projects",
                    payload={"limit": args.max_results, "mode": args.prioritization_mode},
                    host=args.host,
                    port=args.port,
                )
            elif args.evidence_debt:
                response = send_command(
                    "evidence_debt",
                    payload={
                        "project_id": args.project_id,
                        "limit": args.max_results,
                        "mode": args.prioritization_mode,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.resume_candidates:
                response = send_command(
                    "resume_candidates",
                    payload={"limit": args.max_results, "mode": args.prioritization_mode},
                    host=args.host,
                    port=args.port,
                )
            elif args.list_benchmarks:
                response = send_command(
                    "list_benchmarks",
                    payload={"profile_id": args.profile_id, "benchmark_tier": args.benchmark_tier},
                    host=args.host,
                    port=args.port,
                )
            elif args.benchmark_status:
                response = send_command(
                    "benchmark_status",
                    payload={"profile_id": args.profile_id, "benchmark_tier": args.benchmark_tier},
                    host=args.host,
                    port=args.port,
                )
            elif args.run_problem_study:
                response = send_command(
                    "run_problem_study",
                    payload={
                        "problem_id": args.problem_id,
                        "use_docker": args.use_docker,
                        "build_env": args.build_env,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.schedule_problem_study:
                response = send_command(
                    "schedule_problem_study",
                    payload={
                        "problem_id": args.problem_id,
                        "use_docker": args.use_docker,
                        "build_env": args.build_env,
                        "run_at": args.run_at,
                        "delay_s": args.delay_s,
                        "repeat_interval_s": args.repeat_interval_s,
                        "max_runs": args.max_runs,
                        "priority": args.priority,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.scheduler_status:
                response = send_command("scheduler_status", host=args.host, port=args.port)
            elif args.run_scheduler_once:
                response = send_command(
                    "run_scheduler_once",
                    payload={"max_jobs": args.max_jobs},
                    host=args.host,
                    port=args.port,
                )
            elif args.prepare_payload_env:
                response = send_command("prepare_payload_env", host=args.host, port=args.port)
            elif args.rebuild_locked_image:
                response = send_command("rebuild_locked_image", host=args.host, port=args.port)
            elif args.show_manifest:
                response = send_command(
                    "show_manifest",
                    payload={"manifest_id": args.manifest_id, "manifest_path": args.manifest_path},
                    host=args.host,
                    port=args.port,
                )
            elif args.ingest_papers:
                response = send_command(
                    "ingest_papers",
                    payload={"paths": args.paper_paths or []},
                    host=args.host,
                    port=args.port,
                )
            elif args.list_experiment_backends:
                response = send_command("list_experiment_backends", host=args.host, port=args.port)
            elif args.run_runtime_cycle:
                response = send_command(
                    "run_runtime_cycle",
                    payload={"max_jobs": args.max_jobs, "stale_after_s": args.stale_after_s},
                    host=args.host,
                    port=args.port,
                )
            elif args.list_alerts:
                response = send_command(
                    "list_alerts",
                    payload={"count": args.alert_count},
                    host=args.host,
                    port=args.port,
                )
            elif args.retry_failed_job:
                response = send_command(
                    "retry_failed_job",
                    payload={"schedule_id": args.schedule_id},
                    host=args.host,
                    port=args.port,
                )
            elif args.cancel_job:
                response = send_command(
                    "cancel_job",
                    payload={"schedule_id": args.schedule_id},
                    host=args.host,
                    port=args.port,
                )
            elif args.sandbox_policy:
                response = send_command("sandbox_policy", host=args.host, port=args.port)
            elif args.register_checkpoint:
                response = send_command(
                    "register_checkpoint",
                    payload={
                        "name": args.checkpoint_name or "",
                        "model_path": args.model_path or "",
                        "base_model_id": args.base_model_id,
                        "adapter_path": args.adapter_path,
                        "backend": args.backend_name,
                        "role": args.role_name,
                        "trust_remote_code": args.trust_remote_code,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.list_checkpoints:
                response = send_command("list_checkpoints", host=args.host, port=args.port)
            elif args.build_inference_endpoint:
                response = send_command(
                    "build_inference_endpoint",
                    payload={
                        "name": args.checkpoint_name or "",
                        "host": args.host,
                        "port": args.port,
                        "role": args.role_name,
                        "trust_remote_code": args.trust_remote_code,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.list_endpoints:
                response = send_command("list_endpoints", host=args.host, port=args.port)
            elif args.start_endpoint:
                response = send_command(
                    "start_endpoint",
                    payload={
                        "name": args.checkpoint_name or "",
                        "host": args.host,
                        "port": args.port,
                        "role": args.role_name,
                        "trust_remote_code": args.trust_remote_code,
                        "wait_for_health": args.wait_for_health,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.stop_endpoint:
                response = send_command(
                    "stop_endpoint",
                    payload={"endpoint_name": args.endpoint_name or ""},
                    host=args.host,
                    port=args.port,
                )
            elif args.restart_endpoint:
                response = send_command(
                    "restart_endpoint",
                    payload={
                        "endpoint_name": args.endpoint_name or "",
                        "trust_remote_code": args.trust_remote_code,
                        "wait_for_health": args.wait_for_health,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.endpoint_health:
                response = send_command(
                    "endpoint_health",
                    payload={"endpoint_name": args.endpoint_name or ""},
                    host=args.host,
                    port=args.port,
                )
            elif args.assign_role:
                response = send_command(
                    "assign_role",
                    payload={
                        "role": args.role_name,
                        "checkpoint_name": args.checkpoint_name or "",
                        "endpoint_name": args.endpoint_name,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.select_operator_checkpoint:
                response = send_command(
                    "select_operator_checkpoint",
                    payload={
                        "checkpoint_name": args.checkpoint_name or "",
                        "mode": args.operator_mode,
                        "role": args.role_name,
                        "endpoint_name": args.endpoint_name,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.operator_serving_status:
                response = send_command("operator_serving_status", host=args.host, port=args.port)
            elif args.claim_policy:
                response = send_command("claim_policy", host=args.host, port=args.port)
            elif args.claim_verdict:
                response = send_command(
                    "claim_verdict",
                    payload={"trial_id": args.trial_id, "problem_id": args.problem_id},
                    host=args.host,
                    port=args.port,
                )
            elif args.research_decision_log:
                response = send_command(
                    "research_decision_log",
                    payload={"count": args.max_results},
                    host=args.host,
                    port=args.port,
                )
            elif args.chat:
                response = send_command(
                    "chat",
                    payload={"prompt": _resolve_chat_prompt(orchestrator, args)},
                    host=args.host,
                    port=args.port,
                )
            elif args.pivot:
                response = send_command("pivot", payload={"force": args.force}, host=args.host, port=args.port)
            elif args.explain:
                response = send_command("explain_last_fail", host=args.host, port=args.port)
            else:
                response = send_command("panic", host=args.host, port=args.port)
            if not response.ok:
                raise SystemExit(response.error or "Unknown TAR control error")
            payload = response.payload

        if args.json:
            print(json.dumps(payload, indent=2))
        elif args.status:
            print(_render_status(payload))
        elif args.runtime_status:
            print(_render_runtime_status(payload))
        elif args.sandbox_policy:
            print(_render_sandbox_policy(payload))
        elif args.list_endpoints:
            print(_render_endpoint_list(payload))
        elif args.endpoint_health:
            print(_render_endpoint_health(payload))
        elif args.operator_serving_status or args.select_operator_checkpoint:
            print(_render_operator_serving_status(payload))
        elif args.create_project or args.project_status or args.pause_project or args.resume_project:
            print(_render_project_status(payload))
        elif args.list_projects:
            print(_render_project_list(payload))
        elif args.next_action:
            print(_render_next_action(payload))
        elif args.operator_view:
            print(_render_operator_view(payload))
        elif args.project_timeline:
            print(_render_project_timeline(payload))
        elif args.evidence_map:
            print(_render_evidence_map(payload))
        elif args.claim_lineage:
            print(_render_claim_lineage(payload))
        elif args.resume_dashboard:
            print(_render_resume_dashboard(payload))
        elif args.publication_handoff:
            print(_render_publication_handoff(payload))
        elif args.publication_log:
            print(_render_publication_log(payload))
        elif args.portfolio_status or args.rank_actions:
            snapshot_payload = payload.get("snapshot", payload) if args.rank_actions else payload
            candidates = snapshot_payload.get("candidates", payload.get("top_candidates", []))
            if args.portfolio_status:
                print(_render_portfolio_status(payload))
            else:
                print(_render_ranked_candidates(candidates))
        elif args.portfolio_review:
            print(_render_portfolio_review(payload))
        elif args.portfolio_decide:
            print(_render_portfolio_decision(payload))
        elif args.allocate_budget:
            print(_render_budget_allocation(payload))
        elif args.prioritization_log:
            print(_render_prioritization_log(payload))
        elif args.generate_falsification_plan or args.falsification_status:
            print(_render_falsification_plan(payload))
        elif args.falsification_log:
            print(_render_falsification_log(payload))
        elif args.stale_projects:
            print(_render_stale_projects(payload))
        elif args.evidence_debt:
            print(_render_evidence_debt(payload))
        elif args.resume_candidates:
            print(_render_resume_candidates(payload))
        elif args.check_regime:
            print(_render_regime(payload))
        elif args.ingest_research:
            print(f"Ingested {payload.get('indexed', 0)} research documents for topic: {payload.get('topic', '')}")
        elif args.verify_last_trial or args.breakthrough_report or args.resolve_problem or args.prepare_science_env or args.study_problem or args.run_problem_study or args.schedule_problem_study or args.scheduler_status or args.run_scheduler_once or args.frontier_status or args.runtime_status or args.list_benchmarks or args.benchmark_status or args.prepare_payload_env or args.rebuild_locked_image or args.show_manifest or args.ingest_papers or args.list_experiment_backends or args.run_runtime_cycle or args.list_alerts or args.retry_failed_job or args.cancel_job or args.sandbox_policy or args.register_checkpoint or args.list_checkpoints or args.build_inference_endpoint or args.list_endpoints or args.start_endpoint or args.stop_endpoint or args.restart_endpoint or args.endpoint_health or args.assign_role or args.select_operator_checkpoint or args.operator_serving_status or args.claim_policy or args.claim_verdict or args.research_decision_log:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return 0
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
