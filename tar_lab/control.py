from __future__ import annotations

import socket
import socketserver
from typing import Any, Dict, Optional

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ControlRequest, ControlResponse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def handle_request(orchestrator: TAROrchestrator, request: ControlRequest) -> ControlResponse:
    try:
        if request.command == "status":
            payload = orchestrator.status()
        elif request.command == "frontier_status":
            payload = orchestrator.frontier_status().model_dump(mode="json")
        elif request.command == "runtime_status":
            payload = orchestrator.runtime_status()
        elif request.command == "dry_run":
            payload = orchestrator.run_dry_run(
                force_fail_fast=bool(request.payload.get("force_fail_fast", False))
            ).model_dump(mode="json")
        elif request.command == "pivot":
            payload = orchestrator.pivot_force(force=bool(request.payload.get("force", False)))
        elif request.command == "explain_last_fail":
            payload = orchestrator.explain_last_fail().model_dump(mode="json")
        elif request.command == "panic":
            payload = orchestrator.panic()
        elif request.command == "live_docker_test":
            payload = orchestrator.live_docker_test().model_dump(mode="json")
        elif request.command == "chat":
            payload = orchestrator.chat(str(request.payload.get("prompt", ""))).model_dump(mode="json")
        elif request.command == "check_regime":
            payload = orchestrator.check_regime()
        elif request.command == "ingest_research":
            payload = orchestrator.ingest_research(
                topic=str(request.payload.get("topic", "frontier ai")),
                max_results=int(request.payload.get("max_results", 6)),
            ).model_dump(mode="json")
        elif request.command == "verify_last_trial":
            payload = orchestrator.verify_last_trial(
                trial_id=request.payload.get("trial_id"),
            ).model_dump(mode="json")
        elif request.command == "breakthrough_report":
            payload = orchestrator.breakthrough_report(
                trial_id=request.payload.get("trial_id"),
                problem_id=request.payload.get("problem_id"),
            ).model_dump(mode="json")
        elif request.command == "resolve_problem":
            payload = orchestrator.resolve_problem(
                str(request.payload.get("problem", "")),
                benchmark_tier=str(request.payload.get("benchmark_tier", "validation")),  # type: ignore[arg-type]
                requested_benchmark=request.payload.get("benchmark"),
            ).model_dump(mode="json")
        elif request.command == "prepare_science_env":
            payload = orchestrator.prepare_science_environment(
                problem=str(request.payload.get("problem", "")),
                build=bool(request.payload.get("build", False)),
                benchmark_tier=str(request.payload.get("benchmark_tier", "validation")),  # type: ignore[arg-type]
                requested_benchmark=request.payload.get("benchmark"),
                canonical_only=bool(request.payload.get("canonical_only", False)),
                no_proxy_benchmarks=bool(request.payload.get("no_proxy_benchmarks", False)),
            ).model_dump(mode="json")
        elif request.command == "study_problem":
            payload = orchestrator.study_problem(
                problem=str(request.payload.get("problem", "")),
                project_id=request.payload.get("project_id"),
                build_env=bool(request.payload.get("build_env", False)),
                max_results=int(request.payload.get("max_results", 6)),
                benchmark_tier=str(request.payload.get("benchmark_tier", "validation")),  # type: ignore[arg-type]
                requested_benchmark=request.payload.get("benchmark"),
                canonical_only=bool(request.payload.get("canonical_only", False)),
                no_proxy_benchmarks=bool(request.payload.get("no_proxy_benchmarks", False)),
            ).model_dump(mode="json")
        elif request.command == "create_project":
            project = orchestrator.create_project(
                problem=str(request.payload.get("problem", "")),
                benchmark_tier=str(request.payload.get("benchmark_tier", "validation")),  # type: ignore[arg-type]
                requested_benchmark=request.payload.get("benchmark"),
            )
            payload = orchestrator.project_status(project.project_id)
        elif request.command == "list_projects":
            payload = orchestrator.list_projects()
        elif request.command == "project_status":
            payload = orchestrator.project_status(str(request.payload.get("project_id", "")))
        elif request.command == "pause_project":
            project = orchestrator.pause_project(
                str(request.payload.get("project_id", "")),
                reason=str(request.payload.get("reason", "operator_paused")),  # type: ignore[arg-type]
                note=request.payload.get("note"),
            )
            payload = orchestrator.project_status(project.project_id)
        elif request.command == "resume_project":
            project = orchestrator.resume_project(
                str(request.payload.get("project_id", "")),
                reason=str(request.payload.get("reason", "human_requested_resume")),  # type: ignore[arg-type]
                note=request.payload.get("note"),
            )
            payload = orchestrator.project_status(project.project_id)
        elif request.command == "next_action":
            payload = orchestrator.next_action(str(request.payload.get("project_id", "")))
        elif request.command == "operator_view":
            payload = orchestrator.operator_view(
                include_blocked=bool(request.payload.get("include_blocked", True)),
                limit=int(request.payload.get("limit", 5)),
                mode=str(request.payload.get("mode", "balanced")),
            )
        elif request.command == "project_timeline":
            payload = orchestrator.project_timeline(
                str(request.payload.get("project_id", "")),
                limit=int(request.payload.get("limit", 25)),
            )
        elif request.command == "evidence_map":
            payload = orchestrator.project_evidence_map(str(request.payload.get("project_id", "")))
        elif request.command == "claim_lineage":
            payload = orchestrator.claim_lineage(str(request.payload.get("project_id", "")))
        elif request.command == "resume_dashboard":
            payload = orchestrator.resume_dashboard(str(request.payload.get("project_id", "")))
        elif request.command == "publication_handoff":
            payload = orchestrator.publication_handoff(str(request.payload.get("project_id", "")))
        elif request.command == "publication_log":
            payload = orchestrator.publication_log(count=int(request.payload.get("count", 20)))
        elif request.command == "portfolio_status":
            payload = orchestrator.portfolio_status(
                include_blocked=bool(request.payload.get("include_blocked", True)),
                limit=int(request.payload.get("limit", 5)),
                mode=str(request.payload.get("mode", "balanced")),
            )
        elif request.command == "portfolio_review":
            payload = orchestrator.portfolio_review(
                include_blocked=bool(request.payload.get("include_blocked", True)),
                limit=int(request.payload.get("limit", 10)),
                mode=str(request.payload.get("mode", "balanced")),
            )
        elif request.command == "portfolio_decide":
            payload = orchestrator.portfolio_decide(
                include_blocked=bool(request.payload.get("include_blocked", True)),
                limit=int(request.payload.get("limit", 10)),
                mode=str(request.payload.get("mode", "balanced")),
            )
        elif request.command == "rank_actions":
            payload = orchestrator.rank_actions(
                project_id=request.payload.get("project_id"),
                include_blocked=bool(request.payload.get("include_blocked", False)),
                limit=int(request.payload.get("limit", 10)),
                mode=str(request.payload.get("mode", "balanced")),
            )
        elif request.command == "allocate_budget":
            payload = orchestrator.allocate_budget(
                project_id=request.payload.get("project_id"),
                include_blocked=bool(request.payload.get("include_blocked", False)),
                limit=int(request.payload.get("limit", 10)),
                mode=str(request.payload.get("mode", "balanced")),
                schedule_selected=bool(request.payload.get("schedule_selected", False)),
            )
        elif request.command == "prioritization_log":
            payload = orchestrator.prioritization_log(count=int(request.payload.get("count", 20)))
        elif request.command == "generate_falsification_plan":
            payload = orchestrator.generate_falsification_plan(
                str(request.payload.get("project_id", "")),
                force=bool(request.payload.get("force", False)),
            )
        elif request.command == "falsification_status":
            payload = orchestrator.falsification_status(request.payload.get("project_id"))
        elif request.command == "falsification_log":
            payload = orchestrator.falsification_log(count=int(request.payload.get("count", 20)))
        elif request.command == "stale_projects":
            payload = orchestrator.stale_projects(
                limit=int(request.payload.get("limit", 10)),
                mode=str(request.payload.get("mode", "balanced")),
            )
        elif request.command == "evidence_debt":
            payload = orchestrator.evidence_debt(
                project_id=request.payload.get("project_id"),
                limit=int(request.payload.get("limit", 10)),
                mode=str(request.payload.get("mode", "balanced")),
            )
        elif request.command == "resume_candidates":
            payload = orchestrator.resume_candidates(
                limit=int(request.payload.get("limit", 10)),
                mode=str(request.payload.get("mode", "balanced")),
            )
        elif request.command == "list_benchmarks":
            payload = orchestrator.list_benchmarks(
                profile_id=request.payload.get("profile_id"),
                tier=request.payload.get("benchmark_tier"),
            )
        elif request.command == "benchmark_status":
            payload = orchestrator.benchmark_status(
                profile_id=request.payload.get("profile_id"),
                tier=request.payload.get("benchmark_tier"),
            )
        elif request.command == "run_problem_study":
            payload = orchestrator.run_problem_study(
                problem_id=request.payload.get("problem_id"),
                use_docker=bool(request.payload.get("use_docker", False)),
                build_env=bool(request.payload.get("build_env", False)),
            ).model_dump(mode="json")
        elif request.command == "schedule_problem_study":
            payload = orchestrator.schedule_problem_study(
                problem_id=request.payload.get("problem_id"),
                use_docker=bool(request.payload.get("use_docker", False)),
                build_env=bool(request.payload.get("build_env", False)),
                run_at=request.payload.get("run_at"),
                delay_s=int(request.payload.get("delay_s", 0)),
                repeat_interval_s=request.payload.get("repeat_interval_s"),
                max_runs=int(request.payload.get("max_runs", 1)),
                priority=int(request.payload.get("priority", 0)),
            ).model_dump(mode="json")
        elif request.command == "scheduler_status":
            payload = orchestrator.scheduler_status()
        elif request.command == "run_scheduler_once":
            payload = orchestrator.run_scheduler_once(
                max_jobs=int(request.payload.get("max_jobs", 1)),
            ).model_dump(mode="json")
        elif request.command == "prepare_payload_env":
            payload = orchestrator.prepare_payload_environment().model_dump(mode="json")
        elif request.command == "rebuild_locked_image":
            payload = orchestrator.rebuild_locked_image().model_dump(mode="json")
        elif request.command == "show_manifest":
            payload = orchestrator.show_manifest(
                manifest_id=request.payload.get("manifest_id"),
                manifest_path=request.payload.get("manifest_path"),
            )
        elif request.command == "ingest_papers":
            payload = orchestrator.ingest_papers(
                [str(item) for item in request.payload.get("paths", [])]
            ).model_dump(mode="json")
        elif request.command == "list_experiment_backends":
            payload = {"backends": orchestrator.list_experiment_backends()}
        elif request.command == "run_runtime_cycle":
            payload = orchestrator.run_runtime_cycle(
                max_jobs=int(request.payload.get("max_jobs", 1)),
                stale_after_s=int(request.payload.get("stale_after_s", 900)),
            ).model_dump(mode="json")
        elif request.command == "list_alerts":
            payload = orchestrator.list_alerts(count=int(request.payload.get("count", 20)))
        elif request.command == "retry_failed_job":
            payload = orchestrator.retry_failed_job(str(request.payload.get("schedule_id", ""))).model_dump(mode="json")
        elif request.command == "cancel_job":
            payload = orchestrator.cancel_job(str(request.payload.get("schedule_id", ""))).model_dump(mode="json")
        elif request.command == "sandbox_policy":
            payload = orchestrator.sandbox_policy()
        elif request.command == "register_checkpoint":
            payload = orchestrator.register_checkpoint(
                name=str(request.payload.get("name", "")),
                model_path=str(request.payload.get("model_path", "")),
                backend=str(request.payload.get("backend", "transformers")),
                role=str(request.payload.get("role", "assistant")),
                trust_remote_code=request.payload.get("trust_remote_code"),
            ).model_dump(mode="json")
        elif request.command == "list_checkpoints":
            payload = {"checkpoints": [item.model_dump(mode="json") for item in orchestrator.list_checkpoints()]}
        elif request.command == "build_inference_endpoint":
            payload = orchestrator.build_inference_endpoint(
                name=str(request.payload.get("name", "")),
                host=str(request.payload.get("host", DEFAULT_HOST)),
                port=int(request.payload.get("port", 8000)),
                role=request.payload.get("role"),
                trust_remote_code=request.payload.get("trust_remote_code"),
            ).model_dump(mode="json")
        elif request.command == "list_endpoints":
            payload = {"endpoints": [item.model_dump(mode="json") for item in orchestrator.list_endpoints()]}
        elif request.command == "start_endpoint":
            payload = orchestrator.start_endpoint(
                name=str(request.payload.get("name", "")),
                host=str(request.payload.get("host", DEFAULT_HOST)),
                port=int(request.payload.get("port", 8000)),
                role=request.payload.get("role"),
                trust_remote_code=request.payload.get("trust_remote_code"),
                wait_for_health=bool(request.payload.get("wait_for_health", False)),
            ).model_dump(mode="json")
        elif request.command == "stop_endpoint":
            payload = orchestrator.stop_endpoint(str(request.payload.get("endpoint_name", ""))).model_dump(mode="json")
        elif request.command == "restart_endpoint":
            payload = orchestrator.restart_endpoint(
                str(request.payload.get("endpoint_name", "")),
                trust_remote_code=request.payload.get("trust_remote_code"),
                wait_for_health=bool(request.payload.get("wait_for_health", False)),
            ).model_dump(mode="json")
        elif request.command == "endpoint_health":
            payload = orchestrator.endpoint_health(str(request.payload.get("endpoint_name", "")))
        elif request.command == "assign_role":
            payload = orchestrator.assign_role(
                role=str(request.payload.get("role", "assistant")),
                checkpoint_name=str(request.payload.get("checkpoint_name", "")),
                endpoint_name=request.payload.get("endpoint_name"),
            ).model_dump(mode="json")
        elif request.command == "claim_policy":
            payload = orchestrator.claim_policy()
        elif request.command == "claim_verdict":
            payload = orchestrator.claim_verdict(
                trial_id=request.payload.get("trial_id"),
                problem_id=request.payload.get("problem_id"),
            ).model_dump(mode="json")
        elif request.command == "research_decision_log":
            payload = orchestrator.research_decision_log(count=int(request.payload.get("count", 20)))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported command: {request.command}")
        return ControlResponse(ok=True, command=request.command, payload=payload)
    except Exception as exc:  # pragma: no cover - defensive path
        return ControlResponse(ok=False, command=request.command, error=str(exc))


class _ControlHandler(socketserver.StreamRequestHandler):
    orchestrator: TAROrchestrator

    def handle(self) -> None:
        raw = self.rfile.readline().decode("utf-8").strip()
        request = ControlRequest.model_validate_json(raw)
        response = handle_request(self.orchestrator, request)
        self.wfile.write((response.model_dump_json() + "\n").encode("utf-8"))


class TARControlServer:
    def __init__(self, orchestrator: TAROrchestrator, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.orchestrator = orchestrator
        handler = type("BoundControlHandler", (_ControlHandler,), {"orchestrator": orchestrator})
        self.server = socketserver.ThreadingTCPServer((host, port), handler)

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def send_command(
    command: str,
    payload: Optional[Dict[str, Any]] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ControlResponse:
    request = ControlRequest(command=command, payload=payload or {})
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall((request.model_dump_json() + "\n").encode("utf-8"))
        data = sock.recv(65536).decode("utf-8").strip()
    return ControlResponse.model_validate_json(data)
