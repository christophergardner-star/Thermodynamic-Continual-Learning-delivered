from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tar_lab.control import DEFAULT_HOST, DEFAULT_PORT, TARControlServer, send_command
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.voice import SpeechProcessor


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
    parser.add_argument("--paper-path", action="append", dest="paper_paths")
    parser.add_argument("--checkpoint-name", dest="checkpoint_name")
    parser.add_argument("--model-path", dest="model_path")
    parser.add_argument("--backend-name", default="transformers", dest="backend_name")
    parser.add_argument("--role-name", default="assistant", dest="role_name")
    parser.add_argument("--wait-for-health", action="store_true", dest="wait_for_health")
    parser.add_argument("--build-env", action="store_true", dest="build_env")
    parser.add_argument("--use-docker", action="store_true", dest="use_docker")
    parser.add_argument("--run-at", dest="run_at")
    parser.add_argument("--delay-s", type=int, default=0, dest="delay_s")
    parser.add_argument("--repeat-interval-s", type=int, dest="repeat_interval_s")
    parser.add_argument("--max-runs", type=int, default=1, dest="max_runs")
    parser.add_argument("--priority", type=int, default=0)
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
    group.add_argument("--claim-policy", action="store_true", dest="claim_policy")
    group.add_argument("--claim-verdict", action="store_true", dest="claim_verdict")
    group.add_argument("--research-decision-log", action="store_true", dest="research_decision_log")
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
        return orchestrator.breakthrough_report(trial_id=args.trial_id).model_dump(mode="json")
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
            build_env=args.build_env,
            max_results=args.max_results,
            benchmark_tier=args.benchmark_tier,  # type: ignore[arg-type]
            requested_benchmark=args.benchmark,
            canonical_only=args.canonical_only,
            no_proxy_benchmarks=args.no_proxy_benchmarks,
        ).model_dump(mode="json")
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
        ).model_dump(mode="json")
    if args.list_checkpoints:
        return {"checkpoints": [item.model_dump(mode="json") for item in orchestrator.list_checkpoints()]}
    if args.build_inference_endpoint:
        return orchestrator.build_inference_endpoint(
            name=args.checkpoint_name or "",
            host=args.host,
            port=args.port,
            role=args.role_name,
        ).model_dump(mode="json")
    if args.list_endpoints:
        return {"endpoints": [item.model_dump(mode="json") for item in orchestrator.list_endpoints()]}
    if args.start_endpoint:
        return orchestrator.start_endpoint(
            name=args.checkpoint_name or "",
            host=args.host,
            port=args.port,
            role=args.role_name,
            wait_for_health=args.wait_for_health,
        ).model_dump(mode="json")
    if args.stop_endpoint:
        return orchestrator.stop_endpoint(args.endpoint_name or "").model_dump(mode="json")
    if args.restart_endpoint:
        return orchestrator.restart_endpoint(args.endpoint_name or "", wait_for_health=args.wait_for_health).model_dump(mode="json")
    if args.endpoint_health:
        return orchestrator.endpoint_health(args.endpoint_name or "")
    if args.assign_role:
        return orchestrator.assign_role(
            role=args.role_name,
            checkpoint_name=args.checkpoint_name or "",
            endpoint_name=args.endpoint_name,
        ).model_dump(mode="json")
    if args.claim_policy:
        return orchestrator.claim_policy()
    if args.claim_verdict:
        return orchestrator.claim_verdict(trial_id=args.trial_id).model_dump(mode="json")
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
        f"Sandbox Mode: {payload.get('safe_execution_mode', 'n/a')}",
        f"Alert Count: {payload.get('alerts', 0)}",
        f"Endpoints: {len(payload.get('endpoints', []))}",
        f"Role Assignments: {len(payload.get('role_assignments', []))}",
        f"Latest Claim Verdict: {(payload.get('latest_claim_verdict') or {}).get('status', 'n/a')}",
        f"Benchmark: {payload.get('benchmark_name') or ', '.join(payload.get('benchmark_ids', [])) or 'n/a'}",
        f"Benchmark Tier: {payload.get('benchmark_tier', 'n/a')}",
        f"Benchmark Actual Tier(s): {', '.join(payload.get('actual_benchmark_tiers', [])) or 'n/a'}",
        f"Canonical Comparable: {payload.get('canonical_comparable', 'n/a')}",
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


def _resolve_chat_prompt(orchestrator: TAROrchestrator, args: argparse.Namespace) -> str:
    if args.message:
        return args.message
    if args.voice_file:
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
                    payload={"trial_id": args.trial_id},
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
                        "backend": args.backend_name,
                        "role": args.role_name,
                    },
                    host=args.host,
                    port=args.port,
                )
            elif args.list_checkpoints:
                response = send_command("list_checkpoints", host=args.host, port=args.port)
            elif args.build_inference_endpoint:
                response = send_command(
                    "build_inference_endpoint",
                    payload={"name": args.checkpoint_name or "", "host": args.host, "port": args.port, "role": args.role_name},
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
                    payload={"endpoint_name": args.endpoint_name or "", "wait_for_health": args.wait_for_health},
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
            elif args.claim_policy:
                response = send_command("claim_policy", host=args.host, port=args.port)
            elif args.claim_verdict:
                response = send_command(
                    "claim_verdict",
                    payload={"trial_id": args.trial_id},
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
        elif args.check_regime:
            print(_render_regime(payload))
        elif args.ingest_research:
            print(f"Ingested {payload.get('indexed', 0)} research documents for topic: {payload.get('topic', '')}")
        elif args.verify_last_trial or args.breakthrough_report or args.resolve_problem or args.prepare_science_env or args.study_problem or args.run_problem_study or args.schedule_problem_study or args.scheduler_status or args.run_scheduler_once or args.frontier_status or args.runtime_status or args.list_benchmarks or args.benchmark_status or args.prepare_payload_env or args.rebuild_locked_image or args.show_manifest or args.ingest_papers or args.list_experiment_backends or args.run_runtime_cycle or args.list_alerts or args.retry_failed_job or args.cancel_job or args.sandbox_policy or args.register_checkpoint or args.list_checkpoints or args.build_inference_endpoint or args.list_endpoints or args.start_endpoint or args.stop_endpoint or args.restart_endpoint or args.endpoint_health or args.assign_role or args.claim_policy or args.claim_verdict or args.research_decision_log:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return 0
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
