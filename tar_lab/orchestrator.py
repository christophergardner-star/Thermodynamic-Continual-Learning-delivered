from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional

import torch

from tar_lab.data_manager import DataManager
from tar_lab.docker_runner import DockerRunner
from tar_lab.errors import ScientificValidityError
from tar_lab.experiment_backends import ExperimentBackendRegistry
from tar_lab.governor import ThermodynamicGovernor
from tar_lab.hardware import NvidiaSMI
from tar_lab.hierarchy import TriModelHierarchy
from tar_lab.hierarchy import build_evidence_bundle, build_hypotheses
from tar_lab.inference_bridge import InferenceBridge
from tar_lab.literature_engine import LiteratureEngine
from tar_lab.memory import MemoryIndexer, VectorVault
from tar_lab.research_ingest import ResearchIngestor
from tar_lab.reproducibility import PayloadEnvironmentBuilder
from tar_lab.runtime_daemon import LabRuntimeDaemon
from tar_lab.scheduler import ProblemStudyScheduler
from tar_lab.science_profiles import ProblemResearchEngine, ScienceProfileRegistry
from tar_lab.schemas import (
    AlertRecord,
    BenchmarkTier,
    BreakthroughReport,
    ClaimAcceptancePolicy,
    ClaimVerdict,
    CheckpointRecord,
    ContradictionReview,
    DryRunReport,
    EndpointRecord,
    FrontierStatus,
    FailureAutopsy,
    GovernorDecision,
    GovernorMetrics,
    InferenceEndpointPlan,
    KnowledgeGraphEntry,
    LabChatResponse,
    LiveDockerTestReport,
    PaperIngestReport,
    PayloadEnvironmentReport,
    ProblemExecutionReport,
    ProblemScheduleEntry,
    ProblemResolutionReport,
    ProblemStudyReport,
    PreparedDataBundle,
    ResearchIngestReport,
    ResearchDecisionRecord,
    RecoveryState,
    RoleAssignment,
    RunIntent,
    RuntimeHeartbeat,
    SandboxPolicy,
    SchedulerCycleReport,
    SelfCorrectionNote,
    ScienceEnvironmentBundle,
    TrainingPayloadConfig,
    VerificationReport,
)
from tar_lab.state import TARStateStore
from tar_lab.verification import VerificationRunner


class TAROrchestrator:
    def __init__(
        self,
        workspace: str = ".",
        hierarchy: Optional[TriModelHierarchy] = None,
        governor: Optional[ThermodynamicGovernor] = None,
        docker_runner: Optional[DockerRunner] = None,
        hardware: Optional[NvidiaSMI] = None,
        start_memory_indexer: bool = False,
    ):
        self.store = TARStateStore(workspace)
        self.workspace = str(self.store.workspace)
        self.data_manager = DataManager(workspace)
        self.hierarchy = hierarchy or TriModelHierarchy()
        self.governor = governor or ThermodynamicGovernor()
        self.docker_runner = docker_runner or DockerRunner()
        self.hardware = hardware or NvidiaSMI()
        self.research_ingestor = ResearchIngestor(workspace)
        self.verification_runner = VerificationRunner(workspace)
        self.profile_registry = ScienceProfileRegistry(workspace)
        self.problem_engine = ProblemResearchEngine(workspace, registry=self.profile_registry)
        self.scheduler = ProblemStudyScheduler(self.store, execute_callback=self._execute_scheduled_problem)
        self.runtime_daemon = LabRuntimeDaemon(self.store, self.scheduler)
        self.experiment_backends = ExperimentBackendRegistry(workspace)
        self.literature_engine = LiteratureEngine(workspace)
        self.payload_environment = PayloadEnvironmentBuilder(workspace)
        self.inference_bridge = InferenceBridge(workspace)
        self.safe_execution_mode = "docker_container_only"
        self.vault: Optional[VectorVault] = None
        self.memory_indexer: Optional[MemoryIndexer] = None
        self.memory_error: Optional[str] = None
        try:
            self.vault = VectorVault(workspace)
            self.memory_indexer = MemoryIndexer(self.store, self.vault)
            if start_memory_indexer:
                self.memory_indexer.start()
        except Exception as exc:
            self.memory_error = str(exc)

    def seed_mock_metrics(self) -> list[GovernorMetrics]:
        trial_id = "seed"
        metrics = [
            GovernorMetrics(trial_id=trial_id, step=1, energy_e=0.010, entropy_sigma=0.020, drift_l2=0.030, drift_rho=0.010, grad_norm=0.40, regime_rho=1.02, effective_dimensionality=6.8, effective_dimensionality_std_err=0.12, dimensionality_ratio=0.92, entropy_sigma_std_err=0.004, regime_rho_std_err=0.03, stat_window_size=5, stat_sample_count=5, statistically_ready=True, equilibrium_fraction=0.66, equilibrium_gate=False, training_loss=0.62),
            GovernorMetrics(trial_id=trial_id, step=2, energy_e=0.014, entropy_sigma=0.028, drift_l2=0.036, drift_rho=0.014, grad_norm=0.48, regime_rho=0.99, effective_dimensionality=7.1, effective_dimensionality_std_err=0.10, dimensionality_ratio=0.96, entropy_sigma_std_err=0.004, regime_rho_std_err=0.02, stat_window_size=5, stat_sample_count=5, statistically_ready=True, equilibrium_fraction=0.82, equilibrium_gate=True, training_loss=0.54),
            GovernorMetrics(trial_id=trial_id, step=3, energy_e=0.019, entropy_sigma=0.034, drift_l2=0.041, drift_rho=0.018, grad_norm=0.55, regime_rho=1.01, effective_dimensionality=7.4, effective_dimensionality_std_err=0.09, dimensionality_ratio=0.99, entropy_sigma_std_err=0.003, regime_rho_std_err=0.02, stat_window_size=5, stat_sample_count=5, statistically_ready=True, equilibrium_fraction=0.84, equilibrium_gate=True, training_loss=0.48),
        ]
        for item in metrics:
            self.store.append_metric(item)
        return metrics

    def _ensure_recent_metrics(self) -> None:
        if len(self.store.tail_metrics(3)) < 3:
            self.seed_mock_metrics()

    def _resolve_run_intent(self, *, dry_run: bool) -> RunIntent:
        raw = str(os.environ.get("TAR_RUN_INTENT", "control" if dry_run else "research")).strip().lower()
        if raw not in {"control", "plumbing", "research"}:
            return "control" if dry_run else "research"
        return raw  # type: ignore[return-value]

    def _ensure_data_prepared(
        self,
        force: bool = False,
        *,
        dry_run: bool = False,
        run_intent: Optional[RunIntent] = None,
    ) -> PreparedDataBundle:
        run_intent = run_intent or self._resolve_run_intent(dry_run=dry_run)
        data_mode = os.environ.get("TAR_DATA_MODE")
        if not data_mode:
            data_mode = "OFFLINE_FALLBACK" if dry_run else "CACHED_REAL"
        tokenizer_id = os.environ.get("TAR_TOKENIZER_ID")
        if not tokenizer_id and data_mode in {"CACHED_REAL", "DOWNLOAD_REAL"}:
            tokenizer_id = os.environ.get("TAR_PAYLOAD_MODEL", "deepseek-ai/deepseek-coder-1.3b-base")
        bundle = self.data_manager.prepare_dual_stream(
            force=force,
            data_mode=data_mode,
            tokenizer_id=tokenizer_id,
            run_intent=run_intent,
        )
        self.store.append_audit_event(
            "data_manager",
            "prepare_dual_stream",
            {
                "anchor_records": bundle.anchor_manifest.records,
                "research_records": bundle.research_manifest.records,
                "data_purity": bundle.data_provenance.data_purity if bundle.data_provenance else "unknown",
                "run_intent": run_intent,
                "research_grade": bundle.research_grade,
            },
        )
        return bundle

    def _sync_memory(self) -> None:
        if self.memory_indexer is not None:
            self.memory_indexer.sync_once()

    def _retrieve_strategy_hits(self, metrics: Optional[list[GovernorMetrics]] = None) -> list[Any]:
        self._sync_memory()
        if self.vault is None:
            return []
        return self.vault.search_similar_trials(metrics or self.store.tail_metrics(3))

    def plan_trial(self, dry_run: bool = False) -> tuple[Any, Any, Any]:
        self._ensure_recent_metrics()
        run_intent = self._resolve_run_intent(dry_run=dry_run)
        data_bundle = self._ensure_data_prepared(dry_run=dry_run, run_intent=run_intent)
        trial_id = self.store.next_trial_id()
        recovery = self.store.load_recovery()
        self.store.save_recovery(recovery.model_copy(update={"trial_id": trial_id, "status": "planning"}))
        memory_hits = self._retrieve_strategy_hits()
        policy, plan, task = self.hierarchy.produce_bundle(
            self.store,
            trial_id=trial_id,
            workspace=self.workspace,
            dry_run=dry_run,
            memory_hits=memory_hits,
        )
        payload = self._build_payload_config(plan, data_bundle, run_intent=run_intent)
        payload_path = self.store.write_payload_config(payload)
        payload_env = self.prepare_payload_environment()
        if payload_env.image_manifest is None or payload_env.run_manifest is None:
            raise RuntimeError("Locked payload environment was not prepared correctly.")
        if task.payload_config_path != str(payload_path):
            task = task.model_copy(update={"payload_config_path": str(payload_path)})
        task = task.model_copy(
            update={
                "run_manifest_path": str(self.store.manifests_dir / f"{payload_env.run_manifest.manifest_id}.json"),
                "runtime": task.runtime.model_copy(
                    update={
                        "image": payload_env.image_tag,
                        "image_locked": True,
                        "image_manifest_path": str(self.store.manifests_dir / f"image-{payload_env.image_manifest.hash_sha256[:16]}.json"),
                        "run_manifest_path": str(self.store.manifests_dir / f"{payload_env.run_manifest.manifest_id}.json"),
                    }
                ),
            }
        )
        self.store.write_policy_bundle(policy, plan, task)
        self.store.append_knowledge_entry(
            KnowledgeGraphEntry(
                trial_id=trial_id,
                strategy_family=plan.strategy_family,
                outcome="running",
                hyperparameters=plan.hyperparameters,
            )
        )
        self.store.append_audit_event("cli", "plan_trial", {"trial_id": trial_id, "dry_run": dry_run})
        self.store.save_recovery(
            RecoveryState(
                trial_id=trial_id,
                status="running",
                last_known_stable_hyperparameters=plan.hyperparameters,
                consecutive_fail_fast=recovery.consecutive_fail_fast,
                last_strategy_family=plan.strategy_family,
                last_anchor_path=plan.anchor_path,
            )
        )
        self._sync_memory()
        return policy, plan, task

    def ingest_research(self, topic: str = "frontier ai", max_results: int = 6) -> ResearchIngestReport:
        report = self.research_ingestor.ingest(topic=topic, max_results=max_results)
        for document in report.documents:
            self.store.append_research_document(document)
            if self.vault is not None:
                self.vault.index_research_document(document)
        self.store.append_audit_event(
            "research",
            "ingest",
            {
                "topic": topic,
                "fetched": report.fetched,
                "indexed": report.indexed,
                "sources": report.sources,
            },
        )
        self._sync_memory()
        return report

    def ingest_papers(self, paths: list[str]) -> PaperIngestReport:
        report = self.literature_engine.ingest_paths(paths)
        if self.vault is not None:
            self.vault.ensure_research_ready()
            for artifact in report.artifacts:
                self.vault.index_paper_artifact(artifact)
        self.store.append_audit_event(
            "literature",
            "ingest_papers",
            {
                "requested_paths": paths,
                "ingested": report.ingested,
                "failed": report.failed,
                "conflicts": len(report.conflicts),
            },
        )
        self._sync_memory()
        return report

    def prepare_payload_environment(self) -> PayloadEnvironmentReport:
        report = self.payload_environment.prepare()
        self.store.append_audit_event(
            "reproducibility",
            "prepare_payload_environment",
            report.model_dump(mode="json"),
        )
        return report

    def rebuild_locked_image(self) -> PayloadEnvironmentReport:
        report = self.payload_environment.prepare()
        build = self.docker_runner.build_payload_environment(report, dry_run=False)
        report = report.model_copy(
            update={
                "build_status": "built" if build.returncode == 0 else "failed",
                "build_command": build.command,
            }
        )
        manifest_path = Path(report.manifest_path)
        manifest_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        self.store.append_audit_event(
            "reproducibility",
            "rebuild_locked_image",
            {
                "image_tag": report.image_tag,
                "build_status": report.build_status,
                "manifest_hash": report.run_manifest.hash_sha256 if report.run_manifest is not None else None,
            },
        )
        return report

    def show_manifest(self, manifest_id: Optional[str] = None, manifest_path: Optional[str] = None) -> dict[str, Any]:
        if manifest_path:
            manifest = self.store.load_run_manifest_path(manifest_path)
        elif manifest_id:
            manifest = self.store.load_run_manifest(manifest_id)
        else:
            payload_env = self.payload_environment.load()
            manifest = payload_env.run_manifest if payload_env is not None else None
        if manifest is None:
            raise RuntimeError("Manifest not found.")
        return manifest.model_dump(mode="json")

    def list_alerts(self, count: int = 20) -> dict[str, Any]:
        return {"alerts": [item.model_dump(mode="json") for item in self.store.latest_alerts(count)]}

    def runtime_status(self) -> dict[str, Any]:
        schedules = list(self.store.iter_problem_schedules())
        heartbeat = self.runtime_daemon.load_heartbeat()
        payload_env = self.payload_environment.load()
        return {
            "heartbeat": heartbeat.model_dump(mode="json") if heartbeat is not None else None,
            "active_leases": [item.model_dump(mode="json") for item in schedules if item.status in {"leased", "running"}],
            "retry_waiting": [item.model_dump(mode="json") for item in schedules if item.status == "retry_wait"],
            "terminal_failures": [item.model_dump(mode="json") for item in schedules if item.status == "failed_terminal"],
            "alerts": [item.model_dump(mode="json") for item in self.store.latest_alerts(20)],
            "safe_execution_mode": self.safe_execution_mode,
            "sandbox_policy": self.sandbox_policy(),
            "payload_image": payload_env.image_tag if payload_env is not None else None,
            "manifest_hash": payload_env.run_manifest.hash_sha256 if payload_env and payload_env.run_manifest else None,
            "reproducibility_complete": bool(payload_env is not None and payload_env.reproducibility_complete),
        }

    def list_experiment_backends(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.experiment_backends.list_backends()]

    def run_runtime_cycle(self, *, max_jobs: int = 1, stale_after_s: int = 900) -> RuntimeHeartbeat:
        heartbeat = self.runtime_daemon.run_cycle(max_jobs=max_jobs, stale_after_s=stale_after_s)
        self.store.append_audit_event(
            "runtime",
            "run_cycle",
            heartbeat.model_dump(mode="json"),
        )
        self._sync_memory()
        return heartbeat

    def retry_failed_job(self, schedule_id: str) -> ProblemScheduleEntry:
        entry = self.scheduler.retry_failed_job(schedule_id)
        self.store.append_audit_event("runtime", "retry_failed_job", entry.model_dump(mode="json"))
        return entry

    def cancel_job(self, schedule_id: str) -> ProblemScheduleEntry:
        entry = self.scheduler.cancel_job(schedule_id)
        self.store.append_audit_event("runtime", "cancel_job", entry.model_dump(mode="json"))
        return entry

    def sandbox_policy(self) -> dict[str, Any]:
        policy = self.payload_environment.default_sandbox_policy(artifact_dir="/workspace/tar_runs")
        return policy.model_dump(mode="json")

    def _claim_policy(self) -> ClaimAcceptancePolicy:
        return self.inference_bridge.default_claim_policy()

    def _build_research_decision(
        self,
        *,
        prompt: str,
        evidence_bundle: Any,
        hypotheses: list[Any],
        selected_action: str,
        claim_verdict: Optional[ClaimVerdict] = None,
        mode: str = "research_chat",
    ) -> ResearchDecisionRecord:
        import hashlib

        notes = list(evidence_bundle.notes)
        if claim_verdict is not None:
            notes.append(f"claim_verdict={claim_verdict.status}")
        digest = hashlib.sha256(f"{prompt}|{selected_action}|{evidence_bundle.bundle_id}".encode("utf-8")).hexdigest()[:16]
        return ResearchDecisionRecord(
            decision_id=f"decision-{digest}",
            prompt=prompt,
            mode=mode,  # type: ignore[arg-type]
            evidence_bundle=evidence_bundle,
            hypotheses=hypotheses,
            selected_action=selected_action,
            confidence=evidence_bundle.confidence,
            contradiction_review=evidence_bundle.contradiction_review,
            claim_verdict_id=claim_verdict.verdict_id if claim_verdict is not None else None,
            notes=notes,
        )

    def register_checkpoint(
        self,
        *,
        name: str,
        model_path: str,
        backend: str = "transformers",
        role: str = "assistant",
    ) -> CheckpointRecord:
        record = self.inference_bridge.register_checkpoint(
            name=name,
            model_path=model_path,
            backend=backend,
            role=role,
        )
        self.store.append_audit_event(
            "inference",
            "register_checkpoint",
            record.model_dump(mode="json"),
        )
        return record

    def list_checkpoints(self) -> list[CheckpointRecord]:
        return self.inference_bridge.list_checkpoints()

    def build_inference_endpoint(
        self,
        *,
        name: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        role: Optional[str] = None,
    ) -> InferenceEndpointPlan:
        plan = self.inference_bridge.build_endpoint(name=name, host=host, port=port, role=role)
        self.store.append_audit_event(
            "inference",
            "build_endpoint",
            plan.model_dump(mode="json"),
        )
        return plan

    def list_endpoints(self) -> list[EndpointRecord]:
        return self.inference_bridge.list_endpoints()

    def start_endpoint(
        self,
        *,
        name: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        role: Optional[str] = None,
        wait_for_health: bool = False,
    ) -> EndpointRecord:
        record = self.inference_bridge.start_endpoint(
            name=name,
            host=host,
            port=port,
            role=role,
            wait_for_health=wait_for_health,
        )
        self.store.append_audit_event("inference", "start_endpoint", record.model_dump(mode="json"))
        return record

    def stop_endpoint(self, endpoint_name: str) -> EndpointRecord:
        record = self.inference_bridge.stop_endpoint(endpoint_name)
        self.store.append_audit_event("inference", "stop_endpoint", record.model_dump(mode="json"))
        return record

    def restart_endpoint(self, endpoint_name: str, *, wait_for_health: bool = False) -> EndpointRecord:
        record = self.inference_bridge.restart_endpoint(endpoint_name, wait_for_health=wait_for_health)
        self.store.append_audit_event("inference", "restart_endpoint", record.model_dump(mode="json"))
        return record

    def endpoint_health(self, endpoint_name: str) -> dict[str, Any]:
        health = self.inference_bridge.endpoint_health(endpoint_name)
        record = self.inference_bridge.refresh_endpoint_health(endpoint_name)
        self.store.append_audit_event("inference", "endpoint_health", health.model_dump(mode="json"))
        return {
            "endpoint": record.model_dump(mode="json"),
            "health": health.model_dump(mode="json"),
        }

    def assign_role(
        self,
        *,
        role: str,
        checkpoint_name: str,
        endpoint_name: Optional[str] = None,
    ) -> RoleAssignment:
        assignment = self.inference_bridge.assign_role(
            role=role,
            checkpoint_name=checkpoint_name,
            endpoint_name=endpoint_name,
        )
        self.store.append_audit_event("inference", "assign_role", assignment.model_dump(mode="json"))
        return assignment

    def claim_policy(self) -> dict[str, Any]:
        return self.inference_bridge.default_claim_policy().model_dump(mode="json")

    def research_decision_log(self, count: int = 20) -> dict[str, Any]:
        rows = list(self.store.iter_research_decisions())[-count:]
        return {"decisions": [item.model_dump(mode="json") for item in rows]}

    def frontier_status(self) -> FrontierStatus:
        payload_env = self.payload_environment.load()
        literature_status = self.literature_engine.status()
        vault_status = self.vault.stats() if self.vault is not None else {}
        literature_capability = self.literature_engine.capability_report(
            notes=list(literature_status.get("capability_report", {}).get("notes", []))  # type: ignore[union-attr]
        )
        if self.vault is not None:
            vault_capability = self.vault.capability_report()
            literature_capability = literature_capability.model_copy(
                update={
                    "semantic_model": vault_capability.semantic_model,
                    "semantic_ready": vault_capability.semantic_ready,
                    "reranker": vault_capability.reranker,
                    "reranker_ready": vault_capability.reranker_ready,
                    "notes": sorted(set(literature_capability.notes + vault_capability.notes)),
                }
            )
        return FrontierStatus(
            experiment_backends=self.experiment_backends.list_backends(),
            payload_environment=payload_env,
            literature_artifacts=literature_status["artifacts"],  # type: ignore[index]
            literature_conflicts=literature_status["conflicts"],  # type: ignore[index]
            embedder=vault_status.get("embedder", "unavailable"),
            semantic_research_ready=bool(vault_status.get("semantic_research_ready", False)),
            reranker=str(vault_status.get("reranker", "scientific-hybrid-reranker")),
            reranker_ready=bool(vault_status.get("reranker_ready", False)),
            literature_capabilities=literature_capability,
            runtime_heartbeat=self.runtime_daemon.load_heartbeat(),
            registered_checkpoints=self.inference_bridge.list_checkpoints(),
            managed_endpoints=self.inference_bridge.list_endpoints(),
            role_assignments=self.inference_bridge.list_role_assignments(),
            claim_policy=self._claim_policy(),
            recent_claim_verdicts=list(self.store.iter_claim_verdicts())[-5:],
            benchmark_profiles=self.profile_registry.benchmark_profile_counts(),
            safe_execution_mode=self.safe_execution_mode,
            active_leases=len([item for item in self.store.iter_problem_schedules() if item.status in {"leased", "running"}]),
            alert_count=len(list(self.store.iter_alerts())),
            reproducibility_ready=bool(payload_env is not None and payload_env.reproducibility_complete),
        )

    def list_benchmarks(
        self,
        *,
        profile_id: Optional[str] = None,
        tier: Optional[BenchmarkTier] = None,
    ) -> dict[str, Any]:
        suites = self.profile_registry.list_benchmarks(profile_id=profile_id, tier=tier)
        return {"benchmarks": [item.model_dump(mode="json") for item in suites]}

    def benchmark_status(
        self,
        *,
        profile_id: Optional[str] = None,
        tier: Optional[BenchmarkTier] = None,
    ) -> dict[str, Any]:
        suites = self.profile_registry.list_benchmarks(profile_id=profile_id, tier=tier)
        return {
            "benchmarks": [
                {
                    "spec": suite.model_dump(mode="json"),
                    "availability": self.profile_registry.benchmark_availability(suite).model_dump(mode="json"),
                }
                for suite in suites
            ]
        }

    def resolve_problem(
        self,
        problem: str,
        *,
        benchmark_tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
    ) -> ProblemResolutionReport:
        report = self.problem_engine.registry.resolve_problem(
            problem,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
        )
        self.store.append_audit_event(
            "science",
            "resolve_problem",
            report.model_dump(mode="json"),
        )
        return report

    def prepare_science_environment(
        self,
        problem: str,
        build: bool = False,
        *,
        benchmark_tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
        canonical_only: bool = False,
        no_proxy_benchmarks: bool = False,
    ) -> ScienceEnvironmentBundle:
        resolution = self.resolve_problem(
            problem,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
        )
        bundle = self.problem_engine.prepare_environment(
            problem,
            resolution=resolution,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
            canonical_only=canonical_only,
            no_proxy_benchmarks=no_proxy_benchmarks,
        )
        bundle = self.payload_environment.lock_science_bundle(bundle)
        if build:
            build_result = self.docker_runner.build_science_environment(bundle, dry_run=False)
            status = "built" if build_result.returncode == 0 else "failed"
            bundle = bundle.model_copy(
                update={
                    "build_status": status,
                    "build_stdout": build_result.stdout,
                    "build_stderr": build_result.stderr,
                }
            )
        self.store.append_audit_event(
            "science",
            "prepare_science_environment",
            {
                "problem": problem,
                "profile_id": bundle.profile_id,
                "domain": bundle.domain,
                "docker_image_tag": bundle.docker_image_tag,
                "benchmark_tier": bundle.benchmark_tier,
                "requested_benchmark": bundle.requested_benchmark,
                "build_status": bundle.build_status,
            },
        )
        return bundle

    def study_problem(
        self,
        problem: str,
        build_env: bool = False,
        max_results: int = 6,
        *,
        benchmark_tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
        canonical_only: bool = False,
        no_proxy_benchmarks: bool = False,
    ) -> ProblemStudyReport:
        resolution = self.resolve_problem(
            problem,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
        )
        research = self.ingest_research(topic=problem, max_results=max_results)
        self._sync_memory()
        try:
            memory_hits = self.vault.search(problem, n_results=5, require_research_grade=True) if self.vault is not None else []
        except Exception:
            memory_hits = self.vault.search(problem, n_results=5) if self.vault is not None else []
        evidence_bundle = build_evidence_bundle(problem, memory_hits)
        bundle = self.problem_engine.prepare_environment(
            problem,
            resolution=resolution,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
            canonical_only=canonical_only,
            no_proxy_benchmarks=no_proxy_benchmarks,
        )
        bundle = self.payload_environment.lock_science_bundle(bundle)
        if build_env:
            build_result = self.docker_runner.build_science_environment(bundle, dry_run=False)
            status = "built" if build_result.returncode == 0 else "failed"
            bundle = bundle.model_copy(
                update={
                    "build_status": status,
                    "build_stdout": build_result.stdout,
                    "build_stderr": build_result.stderr,
                }
            )
        report = self.problem_engine.build_study_report(
            problem=problem,
            resolution=resolution,
            environment=bundle,
            research_ids=[f"research:{doc.document_id}" for doc in research.documents],
            memory_ids=[hit.document_id for hit in memory_hits[:5]],
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
            canonical_only=canonical_only,
            no_proxy_benchmarks=no_proxy_benchmarks,
        )
        report = report.model_copy(
            update={
                "evidence_bundle": evidence_bundle,
                "hypotheses": build_hypotheses(problem, evidence_bundle, benchmark_ids=report.benchmark_ids),
                "contradiction_review": evidence_bundle.contradiction_review,
            }
        )
        if build_env and bundle.build_status == "failed":
            report = report.model_copy(update={"status": "build_failed"})
        self.store.append_problem_study(report)
        decision = self._build_research_decision(
            prompt=problem,
            evidence_bundle=evidence_bundle,
            hypotheses=report.hypotheses,
            selected_action=report.next_action,
            mode="problem_study",
        )
        self.store.append_research_decision(decision)
        if self.vault is not None:
            self.vault.index_problem_study(report)
        self.store.append_audit_event(
            "science",
            "study_problem",
            report.model_dump(mode="json"),
        )
        self._sync_memory()
        return report

    def _execute_scheduled_problem(
        self,
        problem_id: str,
        use_docker: bool,
        build_env: bool,
    ) -> ProblemExecutionReport:
        return self.run_problem_study(
            problem_id=problem_id,
            use_docker=use_docker,
            build_env=build_env,
        )

    def schedule_problem_study(
        self,
        problem_id: Optional[str] = None,
        *,
        use_docker: bool = False,
        build_env: bool = False,
        run_at: Optional[str] = None,
        delay_s: int = 0,
        repeat_interval_s: Optional[int] = None,
        max_runs: int = 1,
        priority: int = 0,
    ) -> ProblemScheduleEntry:
        study = self.store.latest_problem_study(problem_id)
        if study is None:
            raise RuntimeError("No problem study is available to schedule.")
        entry = self.scheduler.schedule(
            study,
            use_docker=use_docker,
            build_env=build_env,
            run_at=run_at,
            delay_s=delay_s,
            repeat_interval_s=repeat_interval_s,
            max_runs=max_runs,
            priority=priority,
        )
        self.store.append_audit_event(
            "science",
            "schedule_problem_study",
            entry.model_dump(mode="json"),
        )
        return entry

    def scheduler_status(self) -> dict[str, Any]:
        return self.scheduler.status()

    def run_scheduler_once(self, *, max_jobs: int = 1) -> SchedulerCycleReport:
        report = self.scheduler.run_once(max_jobs=max_jobs)
        self.store.append_audit_event(
            "science",
            "run_scheduler_once",
            report.model_dump(mode="json"),
        )
        self._sync_memory()
        return report

    def run_problem_study(
        self,
        problem_id: Optional[str] = None,
        use_docker: bool = False,
        build_env: bool = False,
    ) -> ProblemExecutionReport:
        study = self.store.latest_problem_study(problem_id)
        if study is None:
            raise RuntimeError("No problem study is available to execute.")
        bundle = study.environment
        execution_report_path = Path(bundle.execution_report_path)

        if use_docker:
            if build_env or bundle.build_status not in {"built"}:
                build_result = self.docker_runner.build_science_environment(bundle, dry_run=False)
                build_status = "built" if build_result.returncode == 0 else "failed"
                bundle = bundle.model_copy(
                    update={
                        "build_status": build_status,
                        "build_stdout": build_result.stdout,
                        "build_stderr": build_result.stderr,
                    }
                )
                if build_result.returncode != 0:
                    report = ProblemExecutionReport(
                        problem_id=study.problem_id,
                        problem=study.problem,
                        profile_id=study.profile_id,
                        domain=study.domain,
                        benchmark_tier=study.benchmark_tier,
                        requested_benchmark=study.requested_benchmark,
                        canonical_comparable=False,
                        proxy_benchmarks_used=False,
                        benchmark_ids=study.benchmark_ids,
                        benchmark_names=study.benchmark_names,
                        actual_benchmark_tiers=study.actual_benchmark_tiers,
                        execution_mode="docker_bundle",
                        imports_ok=[],
                        imports_failed=[],
                        benchmark_availability=study.benchmark_availability,
                        experiments=[],
                        image_tag=bundle.docker_image_tag,
                        manifest_path=bundle.run_manifest_path,
                        manifest_hash=bundle.run_manifest.hash_sha256 if bundle.run_manifest else None,
                        dependency_hash=bundle.image_manifest.dependency_lock.hash_sha256 if bundle.image_manifest else None,
                        reproducibility_complete=bundle.reproducibility_complete,
                        sandbox_policy=bundle.sandbox_policy,
                        summary="Docker build failed before study execution.",
                        recommended_next_step="Inspect build_stderr and repair the locked science profile or base image.",
                        artifact_path=str(execution_report_path),
                        status="failed",
                    )
                    self.store.append_problem_execution(report)
                    if self.vault is not None:
                        self.vault.index_problem_execution(report)
                    return report
            run_result = self.docker_runner.run_science_environment(bundle, dry_run=False)
            if run_result.returncode != 0 and not execution_report_path.exists():
                report = ProblemExecutionReport(
                    problem_id=study.problem_id,
                    problem=study.problem,
                    profile_id=study.profile_id,
                    domain=study.domain,
                    benchmark_tier=study.benchmark_tier,
                    requested_benchmark=study.requested_benchmark,
                    canonical_comparable=False,
                    proxy_benchmarks_used=False,
                    benchmark_ids=study.benchmark_ids,
                    benchmark_names=study.benchmark_names,
                    actual_benchmark_tiers=study.actual_benchmark_tiers,
                    execution_mode="docker_bundle",
                    imports_ok=[],
                    imports_failed=[],
                    benchmark_availability=study.benchmark_availability,
                    experiments=[],
                    image_tag=bundle.docker_image_tag,
                    manifest_path=bundle.run_manifest_path,
                    manifest_hash=bundle.run_manifest.hash_sha256 if bundle.run_manifest else None,
                    dependency_hash=bundle.image_manifest.dependency_lock.hash_sha256 if bundle.image_manifest else None,
                    reproducibility_complete=bundle.reproducibility_complete,
                    sandbox_policy=bundle.sandbox_policy,
                    summary="Study execution failed before a report was produced.",
                    recommended_next_step="Inspect container stderr and rerun after fixing the study environment.",
                    artifact_path=str(execution_report_path),
                    status="failed",
                )
                self.store.append_problem_execution(report)
                if self.vault is not None:
                    self.vault.index_problem_execution(report)
                return report
        else:
            proc = subprocess.run(
                (
                    [
                        sys.executable,
                        "-m",
                        "tar_lab.problem_runner",
                        "--study-plan",
                        bundle.study_plan_path,
                        "--output",
                        bundle.execution_report_path,
                        "--benchmark-tier",
                        study.benchmark_tier,
                    ]
                    + (["--benchmark", study.requested_benchmark] if study.requested_benchmark else [])
                    + (["--canonical-only"] if study.canonical_only else [])
                    + (["--no-proxy-benchmarks"] if study.no_proxy_benchmarks else [])
                ),
                capture_output=True,
                text=True,
                check=False,
                cwd=self.workspace,
            )
            if proc.returncode != 0 and not execution_report_path.exists():
                report = ProblemExecutionReport(
                    problem_id=study.problem_id,
                    problem=study.problem,
                    profile_id=study.profile_id,
                    domain=study.domain,
                    benchmark_tier=study.benchmark_tier,
                    requested_benchmark=study.requested_benchmark,
                    canonical_comparable=False,
                    proxy_benchmarks_used=False,
                    benchmark_ids=study.benchmark_ids,
                    benchmark_names=study.benchmark_names,
                    actual_benchmark_tiers=study.actual_benchmark_tiers,
                    execution_mode="local_python",
                    imports_ok=[],
                    imports_failed=[],
                    benchmark_availability=study.benchmark_availability,
                    experiments=[],
                    image_tag=bundle.docker_image_tag,
                    manifest_path=bundle.run_manifest_path,
                    manifest_hash=bundle.run_manifest.hash_sha256 if bundle.run_manifest else None,
                    dependency_hash=bundle.image_manifest.dependency_lock.hash_sha256 if bundle.image_manifest else None,
                    reproducibility_complete=bundle.reproducibility_complete,
                    sandbox_policy=bundle.sandbox_policy,
                    summary="Local study execution failed before a report was produced.",
                    recommended_next_step="Use the Docker bundle for locked dependencies or inspect local stderr.",
                    artifact_path=str(execution_report_path),
                    status="failed",
                )
                self.store.append_problem_execution(report)
                if self.vault is not None:
                    self.vault.index_problem_execution(report)
                return report

        if not execution_report_path.exists():
            raise RuntimeError(f"Problem execution report was not produced: {execution_report_path}")
        report = ProblemExecutionReport.model_validate_json(execution_report_path.read_text(encoding="utf-8"))
        report = report.model_copy(
            update={
                "execution_mode": "docker_bundle" if use_docker else "local_python",
                "image_tag": report.image_tag or bundle.docker_image_tag,
                "manifest_path": report.manifest_path or bundle.run_manifest_path,
                "manifest_hash": report.manifest_hash or (bundle.run_manifest.hash_sha256 if bundle.run_manifest else None),
                "dependency_hash": report.dependency_hash or (bundle.image_manifest.dependency_lock.hash_sha256 if bundle.image_manifest else None),
                "reproducibility_complete": report.reproducibility_complete or bundle.reproducibility_complete,
                "sandbox_policy": report.sandbox_policy or bundle.sandbox_policy,
            }
        )
        self.store.append_problem_execution(report)
        if self.vault is not None:
            self.vault.index_problem_execution(report)
        self.store.append_audit_event(
            "science",
            "run_problem_study",
            {
                "problem_id": report.problem_id,
                "domain": report.domain,
                "status": report.status,
                "execution_mode": report.execution_mode,
            },
        )
        self._sync_memory()
        return report

    def _build_payload_config(
        self,
        plan: Any,
        data_bundle: PreparedDataBundle,
        *,
        run_intent: RunIntent,
    ) -> TrainingPayloadConfig:
        hyper = dict(plan.hyperparameters)
        backend_id = str(hyper.pop("backend_id", os.environ.get("TAR_PAYLOAD_BACKEND", "asc_text"))).strip().lower()
        backend_id = self.experiment_backends.canonical_backend_id(backend_id)
        if backend_id not in {"asc_text", "toy_anchor", "asc_cv", "asc_rl", "asc_qml", "coding_asc"}:
            backend_id = "asc_text"
        backend_spec = self.experiment_backends.get(backend_id)
        backend_provenance = self.experiment_backends.as_provenance(backend_id)
        if run_intent == "research":
            if backend_spec.status != "executable":
                raise ScientificValidityError(
                    f"Backend '{backend_id}' is scaffold-only and cannot be launched as a research run."
                )
            if backend_spec.control_only:
                raise ScientificValidityError(
                    f"Backend '{backend_id}' is control-only and cannot be launched as a research run."
                )
            if not backend_spec.research_grade_capable:
                raise ScientificValidityError(
                    f"Backend '{backend_id}' is not yet validated for research-grade TAR runs."
                )
            if backend_spec.expected_data_type != "token_sequence":
                raise ScientificValidityError(
                    f"Backend '{backend_id}' requires backend-specific {backend_spec.expected_data_type} provenance. "
                    "Launch it through the canonical backend registry instead of the text payload planner."
                )
            if not self.data_manager.is_research_safe(data_bundle.data_provenance):
                raise ScientificValidityError(
                    "Research payload planning refused because dataset/tokenizer provenance is incomplete or fallback-tainted."
                )
        alpha = float(hyper.pop("alpha", hyper.pop("anchor_alpha", 0.05)))
        eta = float(hyper.pop("eta", hyper.pop("lr", 0.01)))
        steps = int(hyper.pop("steps", 20))
        batch_size = int(hyper.pop("batch_size", 8))
        anchor_path = self._resolve_container_path(plan.anchor_path)
        hyper.setdefault("base_model_name", os.environ.get("TAR_PAYLOAD_MODEL", "deepseek-ai/deepseek-coder-1.3b-base"))
        hyper.setdefault("max_seq_len", int(os.environ.get("TAR_PAYLOAD_SEQ_LEN", "64")))
        hyper.setdefault("consistency_lambda", float(os.environ.get("TAR_PAYLOAD_CONSISTENCY_LAMBDA", "0.3")))
        hyper.setdefault("warp_lr_multiplier", float(os.environ.get("TAR_PAYLOAD_WARP_LR_MULTIPLIER", "3.0")))
        hyper.setdefault("weight_decay", float(os.environ.get("TAR_PAYLOAD_WEIGHT_DECAY", "0.01")))
        return TrainingPayloadConfig(
            trial_id=plan.trial_id,
            backend_id=backend_id,  # type: ignore[arg-type]
            run_intent=run_intent,
            strategy_family=plan.strategy_family,
            anchor_path=str(anchor_path),
            alpha=alpha,
            eta=eta,
            fim_lambda=plan.fim_lambda,
            bregman_budget=plan.bregman_budget,
            drift_budget=plan.drift_budget,
            batch_size=batch_size,
            steps=steps,
            seed=int(hyper.pop("seed", os.environ.get("TAR_PAYLOAD_SEED", "7"))),
            train_split=float(os.environ.get("TAR_PAYLOAD_TRAIN_SPLIT", "0.70")),
            val_split=float(os.environ.get("TAR_PAYLOAD_VAL_SPLIT", "0.15")),
            test_split=float(os.environ.get("TAR_PAYLOAD_TEST_SPLIT", "0.15")),
            stat_window_size=int(os.environ.get("TAR_PAYLOAD_STAT_WINDOW", "5")),
            min_stat_steps=int(os.environ.get("TAR_PAYLOAD_MIN_STAT_STEPS", "5")),
            anchor_batches=int(os.environ.get("TAR_PAYLOAD_ANCHOR_BATCHES", "5")),
            calibration_batches=int(os.environ.get("TAR_PAYLOAD_CALIBRATION_BATCHES", "5")),
            resume_from_checkpoint=None,
            adapter_mode=os.environ.get("TAR_PAYLOAD_ADAPTER_MODE", "lora"),  # type: ignore[arg-type]
            lora_r=int(os.environ.get("TAR_PAYLOAD_LORA_R", "8")),
            lora_alpha=int(os.environ.get("TAR_PAYLOAD_LORA_ALPHA", "16")),
            lora_dropout=float(os.environ.get("TAR_PAYLOAD_LORA_DROPOUT", "0.05")),
            dry_run_backbone=os.environ.get("TAR_PAYLOAD_DRY_RUN_BACKBONE", "__tiny_gpt2__"),
            log_path="/workspace/logs/thermo_metrics.jsonl",
            output_dir=f"/workspace/tar_runs/{plan.trial_id}/output",
            anchor_manifest_path=data_bundle.anchor_manifest_path,
            research_manifest_path=data_bundle.research_manifest_path,
            backend_provenance=backend_provenance,
            data_provenance=data_bundle.data_provenance,
            tokenizer_provenance=(data_bundle.data_provenance.tokenizer_provenance if data_bundle.data_provenance else {}),
            provenance_complete=data_bundle.provenance_complete,
            research_grade=data_bundle.research_grade and backend_spec.research_grade_capable and run_intent == "research",
            governor_thresholds=plan.governor_thresholds,
            protected_layers=plan.protected_layers,
            mutable_layers=plan.mutable_layers,
            notes=hyper,
        )

    def _resolve_container_path(self, anchor_path: str) -> str:
        raw = Path(anchor_path)
        if raw.is_absolute():
            try:
                rel = raw.resolve().relative_to(self.store.workspace)
                return f"/workspace/{rel.as_posix()}"
            except ValueError:
                return f"/workspace/anchors/{raw.name}"
        return f"/workspace/{raw.as_posix()}"

    def _mock_anchor_state(self) -> dict[str, torch.Tensor]:
        return {
            "layer.weight": torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32),
            "layer.bias": torch.tensor([0.0], dtype=torch.float32),
        }

    def _mock_current_state(self, force_fail_fast: bool) -> dict[str, torch.Tensor]:
        magnitude = 0.6 if force_fail_fast else 0.004
        return {
            "layer.weight": torch.tensor([magnitude, 0.1 + magnitude, 0.2 + magnitude], dtype=torch.float32),
            "layer.bias": torch.tensor([magnitude], dtype=torch.float32),
        }

    def _current_payload_config(self) -> Optional[TrainingPayloadConfig]:
        recovery = self.store.load_recovery()
        if not recovery.trial_id:
            return None
        return self.store.load_payload_config(recovery.trial_id)

    def _mock_gradients(self, force_fail_fast: bool) -> dict[str, torch.Tensor]:
        magnitude = 3.2 if force_fail_fast else 0.1
        return {
            "layer.weight": torch.full((3,), magnitude, dtype=torch.float32),
            "layer.bias": torch.tensor([magnitude], dtype=torch.float32),
        }

    def _mock_regime(self, force_fail_fast: bool) -> dict[str, Any]:
        if force_fail_fast:
            return {
                "regime_rho": 1.28,
                "effective_dimensionality": 1.2,
                "effective_dimensionality_std_err": 0.08,
                "dimensionality_ratio": 0.18,
                "entropy_sigma_std_err": 0.01,
                "regime_rho_std_err": 0.03,
                "stat_window_size": 5,
                "stat_sample_count": 5,
                "statistically_ready": True,
                "equilibrium_fraction": 0.0,
                "equilibrium_gate": False,
                "training_loss": 0.58,
            }
        return {
            "regime_rho": 1.01,
            "effective_dimensionality": 7.8,
            "effective_dimensionality_std_err": 0.07,
            "dimensionality_ratio": 0.97,
            "entropy_sigma_std_err": 0.004,
            "regime_rho_std_err": 0.02,
            "stat_window_size": 5,
            "stat_sample_count": 5,
            "statistically_ready": True,
            "equilibrium_fraction": 0.85,
            "equilibrium_gate": True,
            "training_loss": 0.41,
        }

    def _record_fail_fast(self, decision: GovernorDecision, strategy_family: str) -> None:
        recovery = self.store.load_recovery()
        fail_reason = decision.reasons[0] if decision.reasons else "unknown"
        self.store.save_recovery(
            recovery.model_copy(
                update={
                    "status": "fail_fast",
                    "last_fail_reason": fail_reason,
                    "last_fail_metrics": decision.metrics,
                    "consecutive_fail_fast": recovery.consecutive_fail_fast + 1,
                    "last_strategy_family": strategy_family,
                    "max_effective_dimensionality_achieved": max(
                        recovery.max_effective_dimensionality_achieved,
                        decision.metrics.effective_dimensionality,
                    ),
                }
            )
        )
        self.store.update_knowledge_entry(
            decision.metrics.trial_id,
            outcome="fail_fast",
            fail_reason=fail_reason,
            ended_at=_utc_now(),
        )
        self.store.append_audit_event(
            "governor",
            "sigterm",
            {
                "trial_id": decision.metrics.trial_id,
                "reason_of_death": fail_reason,
                "reasons": decision.reasons,
            },
        )

    def _record_success(
        self,
        trial_id: str,
        strategy_family: str,
        metrics: Optional[GovernorMetrics] = None,
    ) -> None:
        recovery = self.store.load_recovery()
        self.store.save_recovery(
            recovery.model_copy(
                update={
                    "status": "completed",
                    "last_fail_reason": None,
                    "last_fail_metrics": None,
                    "consecutive_fail_fast": 0,
                    "last_strategy_family": strategy_family,
                    "max_effective_dimensionality_achieved": max(
                        recovery.max_effective_dimensionality_achieved,
                        metrics.effective_dimensionality if metrics is not None else 0.0,
                    ),
                }
            )
        )
        self.store.update_knowledge_entry(trial_id, outcome="completed", ended_at=_utc_now())
        self.store.append_audit_event("orchestrator", "trial_completed", {"trial_id": trial_id})
        self._sync_memory()

    def _latest_metric(self, trial_id: Optional[str] = None) -> Optional[GovernorMetrics]:
        rows = list(self.store.iter_metrics())
        if trial_id is None:
            return rows[-1] if rows else None
        for metric in reversed(rows):
            if metric.trial_id == trial_id:
                return metric
        return None

    def recursive_analysis(
        self,
        trial_id: str,
        outcome: str,
        metrics: Optional[GovernorMetrics] = None,
    ) -> Optional[SelfCorrectionNote]:
        current = metrics or self._latest_metric(trial_id)
        if current is None:
            return None
        memory_hits = self._retrieve_strategy_hits([current])
        note = self.hierarchy.build_self_correction(
            trial_id=trial_id,
            outcome=outcome,
            metrics=current,
            memory_hits=memory_hits,
        )
        if self.vault is not None:
            self.vault.index_self_correction(note)
        self.store.append_audit_event(
            "director",
            "self_correction",
            note.model_dump(mode="json"),
        )
        self._sync_memory()
        return note

    def verify_last_trial(self, trial_id: Optional[str] = None) -> VerificationReport:
        resolved_trial_id = trial_id or self.store.load_recovery().trial_id
        if not resolved_trial_id:
            raise RuntimeError("No trial is available to verify.")
        config_path = self.store.workspace / "tar_runs" / resolved_trial_id / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"Verification config not found for trial {resolved_trial_id}")
        config = TrainingPayloadConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        report = self.verification_runner.run(config)
        self.store.append_verification_report(report)
        if self.vault is not None:
            self.vault.index_verification_report(report)
        self.store.append_audit_event(
            "verification",
            "verify_last_trial",
            report.model_dump(mode="json"),
        )
        self._sync_memory()
        return report

    def claim_verdict(self, trial_id: Optional[str] = None) -> ClaimVerdict:
        resolved_trial_id = trial_id or self.store.load_recovery().trial_id
        if not resolved_trial_id:
            raise RuntimeError("No trial is available for claim review.")
        verification = self.store.latest_verification_report(resolved_trial_id)
        if verification is None:
            verification = self.verify_last_trial(resolved_trial_id)
        latest_problem_execution = self.store.latest_problem_execution()
        latest_problem_study = self.store.latest_problem_study()
        canonical_comparable = False
        if latest_problem_execution is not None:
            canonical_comparable = latest_problem_execution.canonical_comparable
        elif latest_problem_study is not None:
            canonical_comparable = latest_problem_study.canonical_comparable
        query = f"claim review {resolved_trial_id} thermodynamic calibration dimensionality"
        try:
            research_hits = self.vault.search(query, n_results=5, require_research_grade=True) if self.vault is not None else []
        except Exception:
            research_hits = []
        evidence_bundle = build_evidence_bundle(query, research_hits)
        verdict = self.verification_runner.assess_claim(
            verification,
            supporting_research_ids=[hit.document_id for hit in research_hits],
            contradiction_review=evidence_bundle.contradiction_review,
            canonical_comparable=canonical_comparable,
            policy=self._claim_policy(),
        )
        self.store.append_claim_verdict(verdict)
        self.store.append_research_decision(
            self._build_research_decision(
                prompt=query,
                evidence_bundle=evidence_bundle,
                hypotheses=[],
                selected_action="claim_review",
                claim_verdict=verdict,
                mode="claim_review",
            )
        )
        self.store.append_audit_event("verification", "claim_verdict", verdict.model_dump(mode="json"))
        return verdict

    def breakthrough_report(self, trial_id: Optional[str] = None) -> BreakthroughReport:
        resolved_trial_id = trial_id or self.store.load_recovery().trial_id
        if not resolved_trial_id:
            raise RuntimeError("No trial is available for breakthrough analysis.")
        verification = self.store.latest_verification_report(resolved_trial_id)
        if verification is None:
            verification = self.verify_last_trial(resolved_trial_id)
        latest_metric = self._latest_metric(resolved_trial_id)
        query = (
            "current ai problems thermodynamic learning calibration robustness "
            f"trial {resolved_trial_id}"
        )
        if latest_metric is not None:
            query += (
                f" D_PR={latest_metric.effective_dimensionality:.4f}"
                f" sigma={latest_metric.entropy_sigma:.4f}"
            )
        research_hits = self.vault.search(query, n_results=5, kind="research") if self.vault is not None else []
        claim_verdict = self.claim_verdict(resolved_trial_id)
        report = self.verification_runner.build_breakthrough_report(
            verification,
            supporting_research_ids=[hit.document_id for hit in research_hits],
            claim_verdict=claim_verdict,
        )
        if claim_verdict.status == "accepted":
            report = report.model_copy(update={"status": "breakthrough"})
        elif claim_verdict.status in {"provisional", "insufficient_evidence"} and report.status == "breakthrough":
            report = report.model_copy(update={"status": "candidate"})
        elif claim_verdict.status in {"rejected", "contradicted"}:
            report = report.model_copy(update={"status": "rejected"})
        self.store.append_breakthrough_report(report)
        if self.vault is not None:
            self.vault.index_breakthrough_report(report)
        self.store.append_audit_event(
            "director",
            "breakthrough_report",
            report.model_dump(mode="json"),
        )
        self._sync_memory()
        return report

    def run_dry_run(self, force_fail_fast: bool = False) -> DryRunReport:
        _, plan, task = self.plan_trial(dry_run=True)
        self.store.append_audit_event("cli", "dry_run", {"trial_id": task.trial_id})
        self.hardware.prepare_run(task.runtime.gpu_index, task.runtime.power_limit_w, task.runtime.gpu_target_temp_c, apply=False)
        launch = self.docker_runner.launch(task, dry_run=True)
        telemetry = self.hardware.query_telemetry(task.runtime.gpu_index)
        regime = self._mock_regime(force_fail_fast=force_fail_fast)
        metrics = self.governor.compute_metrics(
            trial_id=task.trial_id,
            step=1,
            anchor=self._mock_anchor_state(),
            current=self._mock_current_state(force_fail_fast=force_fail_fast),
            gradients=self._mock_gradients(force_fail_fast=force_fail_fast),
            gpu_temperature_c=telemetry.temperature_c,
            gpu_memory_temperature_c=telemetry.memory_temperature_c,
            gpu_power_w=telemetry.power_w,
            regime_rho=regime["regime_rho"],
            effective_dimensionality=regime["effective_dimensionality"],
            effective_dimensionality_std_err=regime["effective_dimensionality_std_err"],
            dimensionality_ratio=regime["dimensionality_ratio"],
            entropy_sigma_std_err=regime["entropy_sigma_std_err"],
            regime_rho_std_err=regime["regime_rho_std_err"],
            stat_window_size=regime["stat_window_size"],
            stat_sample_count=regime["stat_sample_count"],
            statistically_ready=regime["statistically_ready"],
            equilibrium_fraction=regime["equilibrium_fraction"],
            equilibrium_gate=regime["equilibrium_gate"],
            training_loss=regime["training_loss"],
        )
        self.store.append_metric(metrics)
        decision = self.governor.evaluate(metrics, plan.governor_thresholds)

        if decision.action == "terminate":
            self._record_fail_fast(decision, strategy_family=plan.strategy_family)
            self.recursive_analysis(task.trial_id, outcome="fail_fast", metrics=metrics)
        else:
            self._record_success(task.trial_id, strategy_family=plan.strategy_family, metrics=metrics)
            self.recursive_analysis(task.trial_id, outcome="dry_run", metrics=metrics)

        recovery = self.store.load_recovery()
        return DryRunReport(
            trial_id=task.trial_id,
            json_schema_ok=True,
            docker_command_ok=bool(launch.command),
            memory_ok=self.vault is not None and len(self.store.tail_metrics(3)) == 3,
            pivot_force_ready=recovery.consecutive_fail_fast >= 3 or recovery.status == "completed",
            governor_action=decision.action,
            recovery_status=recovery.status,
            composed_command=launch.command,
        )

    def live_docker_test(self) -> LiveDockerTestReport:
        _, plan, task = self.plan_trial(dry_run=True)
        try:
            visible_gpus = self.hardware.list_gpus()
            if visible_gpus and task.runtime.gpu_index >= len(visible_gpus):
                raise RuntimeError(
                    f"Requested GPU index {task.runtime.gpu_index} is unavailable. "
                    f"Visible devices: {visible_gpus}"
                )
            self.hardware.prepare_run(
                task.runtime.gpu_index,
                task.runtime.power_limit_w,
                task.runtime.gpu_target_temp_c,
                apply=True,
            )
            launch = self.docker_runner.live_test(task)
            launched = launch.returncode in (None, 0)
            if launched:
                self._record_success(
                    task.trial_id,
                    strategy_family=plan.strategy_family,
                    metrics=self._latest_metric(task.trial_id),
                )
                self.recursive_analysis(task.trial_id, outcome="completed")
            else:
                self.recursive_analysis(task.trial_id, outcome="fail_fast")
            return LiveDockerTestReport(
                launched=launched,
                image=task.runtime.image,
                mode=launch.mode,
                command=launch.command,
                returncode=launch.returncode,
                payload_config_path=task.payload_config_path,
                gpu_visible=launch.gpu_visible,
                gpu_probe_output=launch.probe_output,
                error=(
                    None
                    if launched
                    else launch.probe_output or f"Container exited with return code {launch.returncode}"
                ),
            )
        except Exception as exc:
            return LiveDockerTestReport(
                launched=False,
                image=task.runtime.image,
                mode="error",
                command=self.docker_runner.compose_command(task),
                payload_config_path=task.payload_config_path,
                gpu_visible=False,
                error=str(exc),
            )

    def status(self) -> Dict[str, Any]:
        self._sync_memory()
        payload = self.store.status_payload()
        telemetry = self.hardware.query_telemetry()
        anchor_manifest = self.store.load_dataset_manifest("anchor")
        research_manifest = self.store.load_dataset_manifest("research")
        payload_config = self._current_payload_config()
        run_intent = payload_config.run_intent if payload_config is not None else self._resolve_run_intent(dry_run=False)
        bundle_provenance = self.data_manager.compose_bundle_provenance(anchor_manifest, research_manifest, run_intent=run_intent)
        payload["gpu"] = {
            "temperature_c": telemetry.temperature_c,
            "memory_temperature_c": telemetry.memory_temperature_c,
            "power_w": telemetry.power_w,
            "power_limit_w": telemetry.power_limit_w,
            "target_temperature_c": telemetry.target_temperature_c,
            "visible_devices": self.hardware.list_gpus(),
        }
        payload["datasets"] = {
            "anchor": anchor_manifest.model_dump(mode="json") if anchor_manifest else None,
            "research": research_manifest.model_dump(mode="json") if research_manifest else None,
        }
        payload["data_purity"] = bundle_provenance.data_purity if bundle_provenance is not None else "unknown"
        payload["data_provenance"] = bundle_provenance.model_dump(mode="json") if bundle_provenance is not None else None
        payload["run_intent"] = run_intent
        payload["research_grade"] = bool(payload_config.research_grade) if payload_config is not None else self.data_manager.is_research_safe(bundle_provenance)
        payload["provenance_complete"] = bool(payload_config.provenance_complete) if payload_config is not None else self.data_manager.is_provenance_complete(bundle_provenance)
        payload["tokenizer_integrity"] = bool(bundle_provenance.integrity_check) if bundle_provenance is not None else False
        payload["backend_id"] = payload_config.backend_id if payload_config is not None else None
        payload["backend_provenance"] = payload_config.backend_provenance.model_dump(mode="json") if payload_config and payload_config.backend_provenance else None
        payload["backend_readiness"] = (
            payload_config.backend_provenance.status
            if payload_config is not None and payload_config.backend_provenance is not None
            else "unknown"
        )
        payload_env = self.payload_environment.load()
        payload["image_tag"] = payload_env.image_tag if payload_env is not None else None
        payload["reproducibility_complete"] = bool(payload_env is not None and payload_env.reproducibility_complete)
        payload["manifest_hash"] = payload_env.run_manifest.hash_sha256 if payload_env and payload_env.run_manifest else None
        payload["safe_execution_mode"] = self.safe_execution_mode
        payload["alerts"] = len(list(self.store.iter_alerts()))
        payload["endpoints"] = [item.model_dump(mode="json") for item in self.inference_bridge.list_endpoints()]
        payload["role_assignments"] = [item.model_dump(mode="json") for item in self.inference_bridge.list_role_assignments()]
        payload["claim_policy"] = self._claim_policy().model_dump(mode="json")
        latest_claim_verdict = self.store.latest_claim_verdict()
        payload["latest_claim_verdict"] = latest_claim_verdict.model_dump(mode="json") if latest_claim_verdict else None
        latest_problem_execution = self.store.latest_problem_execution()
        latest_problem_study = self.store.latest_problem_study()
        benchmark_ids = (
            latest_problem_execution.benchmark_ids
            if latest_problem_execution is not None
            else (latest_problem_study.benchmark_ids if latest_problem_study is not None else [])
        )
        benchmark_names = (
            latest_problem_execution.benchmark_names
            if latest_problem_execution is not None
            else (latest_problem_study.benchmark_names if latest_problem_study is not None else [])
        )
        actual_benchmark_tiers = (
            latest_problem_execution.actual_benchmark_tiers
            if latest_problem_execution is not None
            else (latest_problem_study.actual_benchmark_tiers if latest_problem_study is not None else [])
        )
        payload["benchmark_tier"] = (
            latest_problem_execution.benchmark_tier
            if latest_problem_execution is not None
            else (latest_problem_study.benchmark_tier if latest_problem_study is not None else None)
        )
        payload["benchmark_ids"] = benchmark_ids
        payload["benchmark_names"] = benchmark_names
        payload["actual_benchmark_tiers"] = actual_benchmark_tiers
        payload["benchmark_name"] = benchmark_names[0] if benchmark_names else None
        payload["canonical_comparable"] = (
            latest_problem_execution.canonical_comparable
            if latest_problem_execution is not None
            else (latest_problem_study.canonical_comparable if latest_problem_study is not None else False)
        )
        payload["memory"] = self.vault.stats() if self.vault is not None else {"error": self.memory_error}
        payload["regime"] = self.check_regime()
        payload["frontier"] = self.frontier_status().model_dump(mode="json")
        payload["runtime"] = self.runtime_status()
        return payload

    def check_regime(self) -> Dict[str, Any]:
        latest = self._latest_metric()
        if latest is None:
            self.seed_mock_metrics()
            latest = self._latest_metric()
        if latest is None:
            return {"status": "unavailable"}

        warning = None
        regime = "searching"
        if (
            latest.statistically_ready
            and latest.dimensionality_ratio > 0.0
            and latest.training_loss is not None
            and latest.training_loss <= self.governor.thresholds.max_quenching_loss
            and latest.dimensionality_ratio < self.governor.thresholds.min_dimensionality_ratio
        ):
            regime = "thermodynamic_quenching"
            warning = "Degeneracy Warning: loss is holding while D_PR has collapsed."
        elif latest.equilibrium_gate and latest.statistically_ready:
            regime = "equilibrium"
        elif latest.equilibrium_fraction >= 0.5 and latest.statistically_ready:
            regime = "stabilizing"
        elif not latest.statistically_ready:
            regime = "warming_up"
            warning = "Statistical Warmup: waiting for the rolling regime window to fill before firing gates."

        return {
            "trial_id": latest.trial_id,
            "step": latest.step,
            "regime": regime,
            "effective_dimensionality": latest.effective_dimensionality,
            "effective_dimensionality_std_err": latest.effective_dimensionality_std_err,
            "dimensionality_ratio": latest.dimensionality_ratio,
            "stat_sample_count": latest.stat_sample_count,
            "stat_window_size": latest.stat_window_size,
            "statistically_ready": latest.statistically_ready,
            "equilibrium_fraction": latest.equilibrium_fraction,
            "equilibrium_gate": latest.equilibrium_gate,
            "training_loss": latest.training_loss,
            "warning": warning,
        }

    def chat(self, prompt: str) -> LabChatResponse:
        self._ensure_recent_metrics()
        self._ensure_data_prepared(dry_run=True)
        self._sync_memory()
        status = self.status()
        prompt_lower = prompt.lower()
        research_prompt = any(term in prompt_lower for term in ("investigate", "find the solution", "solve", "research plan", "barren plateau", "barren landscapes", "quantum ai", "current ai", "ai problems", "research landscape", "frontier"))
        if any(term in prompt_lower for term in ("investigate", "find the solution", "solve", "research plan", "barren plateau", "barren landscapes", "quantum ai")):
            study = self.study_problem(prompt, build_env=False, max_results=6, benchmark_tier="validation")
            memory_hits = self.vault.search(prompt, n_results=3, require_research_grade=True) if self.vault is not None else []
            evidence_bundle = study.evidence_bundle or build_evidence_bundle(prompt, memory_hits)
            hypotheses = study.hypotheses or build_hypotheses(prompt, evidence_bundle, benchmark_ids=study.benchmark_ids)
            state_summary = (
                f"Problem routed to domain={study.domain}, profile={study.profile_id}, "
                f"confidence={study.resolution_confidence:.2f}, benchmark_tier={study.benchmark_tier}, "
                f"benchmarks={', '.join(study.benchmark_ids[:2]) if study.benchmark_ids else 'none'}, status={study.status}."
            )
            response = LabChatResponse(
                response_text=(
                    f"Problem study created for '{prompt}'. "
                    f"Environment bundle is at {study.environment.build_context_path}. "
                    f"Next action: {study.next_action}"
                ),
                state_summary=state_summary,
                confidence=evidence_bundle.confidence,
                cited_trial_ids=study.cited_research_ids[:3],
                retrieved_memories=memory_hits,
                evidence_traces=evidence_bundle.traces,
                evidence_bundle=evidence_bundle,
                hypotheses=hypotheses,
                contradiction_review=evidence_bundle.contradiction_review,
            )
            self.store.append_research_decision(
                self._build_research_decision(
                    prompt=prompt,
                    evidence_bundle=evidence_bundle,
                    hypotheses=hypotheses,
                    selected_action=study.next_action,
                )
            )
            self.store.append_audit_event(
                "director",
                "chat_problem_study",
                {
                    "prompt": prompt,
                    "problem_id": study.problem_id,
                    "domain": study.domain,
                },
            )
            return response
        if any(term in prompt_lower for term in ("current ai", "ai problems", "research landscape", "frontier")):
            existing_research = list(self.store.iter_research_documents())
            if not existing_research:
                try:
                    self.ingest_research(topic=prompt, max_results=6)
                except Exception:
                    pass
                self._sync_memory()
        memory_hits = (
            self.vault.search(prompt, n_results=3, require_research_grade=research_prompt)
            if self.vault is not None
            else []
        )
        response = self.hierarchy.director_chat(
            self.store,
            prompt=prompt,
            memory_hits=memory_hits,
        )
        if status.get("run_intent") in {"control", "plumbing"} or not status.get("research_grade", False):
            response = response.model_copy(
                update={
                    "state_summary": (
                        f"{response.state_summary} Current lab run is {status.get('run_intent', 'unknown')} "
                        f"with research_grade={status.get('research_grade', False)}."
                    )
                }
            )
        self.store.append_audit_event(
            "director",
            "chat",
            {
                "prompt": prompt,
                "response": response.model_dump(mode="json"),
            },
        )
        if response.evidence_bundle is not None:
            self.store.append_research_decision(
                self._build_research_decision(
                    prompt=prompt,
                    evidence_bundle=response.evidence_bundle,
                    hypotheses=response.hypotheses,
                    selected_action="director_chat_response",
                    claim_verdict=response.claim_verdict,
                )
            )
        return response

    def pivot_force(self, force: bool = False) -> Dict[str, Any]:
        recovery = self.store.load_recovery()
        next_streak = max(recovery.consecutive_fail_fast, 3 if force else recovery.consecutive_fail_fast)
        updated = recovery.model_copy(update={"status": "pivoted", "consecutive_fail_fast": next_streak})
        self.store.save_recovery(updated)
        if updated.trial_id:
            self.store.update_knowledge_entry(updated.trial_id, outcome="pivoted", ended_at=_utc_now())
        self.store.append_audit_event("cli", "pivot_force", {"force": force})
        return updated.model_dump(mode="json")

    def explain_last_fail(self) -> FailureAutopsy:
        recovery = self.store.load_recovery()
        if recovery.status != "fail_fast" or recovery.last_fail_reason is None:
            return FailureAutopsy(status="no_failure")
        return FailureAutopsy(
            trial_id=recovery.trial_id,
            status="fail_fast",
            reason_of_death=recovery.last_fail_reason,
            strategy_family=recovery.last_strategy_family,
            metrics=recovery.last_fail_metrics,
            consecutive_fail_fast=recovery.consecutive_fail_fast,
        )

    def panic(self) -> Dict[str, Any]:
        commands = self.docker_runner.panic_kill(dry_run=True)
        self.store.append_audit_event("cli", "panic", {"commands": commands})
        return {"commands": commands}

    def shutdown(self) -> None:
        if self.memory_indexer is not None:
            self.memory_indexer.stop()
        if self.vault is not None:
            self.vault.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
