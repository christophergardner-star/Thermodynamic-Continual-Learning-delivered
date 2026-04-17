from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Optional

import torch

from tar_lab.data_manager import DataManager
from tar_lab.docker_runner import DockerRunner
from tar_lab.errors import (
    ExecutionPolicyViolation,
    MemoryIntegrityError,
    ReproducibilityLockError,
    ScientificValidityError,
)
from tar_lab.experiment_backends import ExperimentBackendRegistry
from tar_lab.generative_director import GenerativeDirector
from tar_lab.governor import ThermodynamicGovernor
from tar_lab.hardware import NvidiaSMI
from tar_lab.hierarchy import DirectorDraft, LocalOpenAIRole, TriModelHierarchy
from tar_lab.hierarchy import build_evidence_bundle, build_hypotheses
from tar_lab.inference_bridge import InferenceBridge
from tar_lab.literature_engine import LiteratureEngine
from tar_lab.memory import MemoryIndexer, VectorVault
from tar_lab.research_ingest import ResearchIngestor
from tar_lab.reproducibility import PayloadEnvironmentBuilder
from tar_lab.runtime_daemon import LabRuntimeDaemon
from tar_lab.scheduler import ProblemStudyScheduler
from tar_lab.science_profiles import ProblemResearchEngine, ScienceProfileRegistry
from tar_lab.self_improvement import SelfImprovementEngine
from tar_lab.schemas import (
    ActionScoreBreakdown,
    AgendaDecisionRecord,
    AgendaReviewConfig,
    AgendaReviewRecord,
    AgendaSnapshot,
    AlertRecord,
    BenchmarkTier,
    BudgetAllocationDecision,
    BreakthroughReport,
    ClaimAcceptancePolicy,
    ClaimVerdict,
    CheckpointRecord,
    ContradictionReview,
    CuratedDeltaRecord,
    DirectorPolicy,
    DryRunReport,
    EvidenceDebtRecord,
    EndpointRecord,
    FalsificationCoverage,
    FalsificationPlan,
    FalsificationTest,
    FalsificationTrigger,
    FrozenAnchorPackManifest,
    FrontierGapRecord,
    FrontierGapScanReport,
    FrontierStatus,
    FailureAutopsy,
    GenerativeDirectorProposal,
    GovernorDecision,
    GovernorMetrics,
    InferenceEndpointPlan,
    KnowledgeGraphEntry,
    LabChatResponse,
    LiteraturePolicySignal,
    LiveDockerTestReport,
    MemorySearchHit,
    OperatorServingStatus,
    PaperIngestReport,
    PayloadEnvironmentReport,
    PortfolioDecision,
    PortfolioHealthSnapshot,
    PortfolioPrioritySnapshot,
    PrioritizationPolicy,
    PrioritizedActionCandidate,
    ProblemExecutionReport,
    ProblemScheduleEntry,
    ProblemResolutionReport,
    ProblemStudyReport,
    ProposedExperimentFamily,
    PublicationAlternativeBundle,
    PublicationBenchmarkAttachment,
    PublicationClaimBundle,
    PublicationHandoffPackage,
    PublicationLineageEntry,
    PreparedDataBundle,
    QuantitativeJustification,
    ResearchActionStatus,
    ResearchBudgetLedger,
    ResearchDocument,
    ResearchPortfolio,
    ResearchOpenQuestion,
    ResearchPlannedAction,
    ResearchProject,
    ResearchProjectStatus,
    ResearchResumeSnapshot,
    ResearchResumeReason,
    ResearchStopReason,
    ResearchThreadStatus,
    ResearchHypothesisThread,
    ProjectPriorityRecord,
    ProjectStalenessRecord,
    ResearchIngestReport,
    ResearchDecisionRecord,
    RecoveryState,
    RoleAssignment,
    RunIntent,
    RetrainRecord,
    RuntimeHeartbeat,
    SandboxPolicy,
    SchedulerCycleReport,
    SelfImprovementCycleRecord,
    SelfImprovementPolicy,
    SelfCorrectionNote,
    ScienceEnvironmentBundle,
    TARExecutionPolicy,
    TrainingSignalRecord,
    TrainingPayloadConfig,
    VerificationReport,
)
from tar_lab.state import TARStateStore
from tar_lab.verification import VerificationRunner


def _compute_gap_content_hash(source_paper_ids: list[str], title: str) -> str:
    normalized_title = re.sub(r"\s+", " ", title.strip().lower())
    key = "|".join(sorted(source_paper_ids)) + "|" + normalized_title
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


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
        self.hierarchy = hierarchy or TriModelHierarchy(workspace=self.workspace)
        self.governor = governor or ThermodynamicGovernor()
        self.docker_runner = docker_runner or DockerRunner()
        self.hardware = hardware or NvidiaSMI()
        self.research_ingestor = ResearchIngestor(workspace)
        self.verification_runner = VerificationRunner(workspace)
        self.profile_registry = ScienceProfileRegistry(workspace)
        self.problem_engine = ProblemResearchEngine(workspace, registry=self.profile_registry)
        self.scheduler = ProblemStudyScheduler(self.store, execute_callback=self._execute_scheduled_problem)
        self.runtime_daemon = LabRuntimeDaemon(self.store, self.scheduler)
        self._continuity_counter = 0
        self.experiment_backends = ExperimentBackendRegistry(workspace)
        self.literature_engine = LiteratureEngine(workspace)
        self.payload_environment = PayloadEnvironmentBuilder(workspace)
        self.inference_bridge = InferenceBridge(workspace)
        self.safe_execution_mode = "docker_container_only"
        self.execution_policy: TARExecutionPolicy = self.store.load_execution_policy()
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

    def _proposal_policy_stub(self, objective_slug: str) -> DirectorPolicy:
        self._ensure_recent_metrics()
        recent = self.store.tail_metrics(3)
        recovery = self.store.load_recovery()
        latest = recent[-1]
        return DirectorPolicy(
            trial_id=self._continuity_id("family-proposal"),
            objective_slug=objective_slug,
            anchor_path=recovery.last_anchor_path or "anchors/thermodynamic_anchor.safetensors",
            experiment_family=(
                recovery.last_strategy_family
                if recovery.last_strategy_family in {"elastic_anchor", "ou_drift_jitter", "layer_freeze"}
                else "elastic_anchor"
            ),
            pivot_required=True,
            failure_streak=max(GenerativeDirector.PROPOSAL_TRIGGER_STREAK, recovery.consecutive_fail_fast),
            quantitative_justification=QuantitativeJustification(
                energy_e=latest.energy_e,
                entropy_sigma=latest.entropy_sigma,
                drift_rho=latest.drift_rho,
                grad_norm=latest.grad_norm,
                regime_rho=latest.regime_rho,
                effective_dimensionality=latest.effective_dimensionality,
                effective_dimensionality_std_err=latest.effective_dimensionality_std_err,
                equilibrium_fraction=latest.equilibrium_fraction,
                energy_slope=recent[-1].energy_e - recent[0].energy_e,
                entropy_slope=recent[-1].entropy_sigma - recent[0].entropy_sigma,
                drift_slope=recent[-1].drift_rho - recent[0].drift_rho,
                dimensionality_slope=recent[-1].effective_dimensionality - recent[0].effective_dimensionality,
            ),
            data_anchor=recent,
        )

    def _family_operator_role(self) -> Optional[LocalOpenAIRole]:
        if self.hierarchy.director_config is None:
            return None
        return LocalOpenAIRole(
            "director",
            self.hierarchy.director_config,
            DirectorDraft,
            client_factory=self.hierarchy.client_factory,
        )

    def _get_family_proposal(self, proposal_id: str) -> GenerativeDirectorProposal:
        for proposal in self.store.load_family_proposals():
            if proposal.proposal_id == proposal_id:
                return proposal
        raise RuntimeError(f"Unknown family proposal: {proposal_id}")

    @staticmethod
    def _apply_family_config_delta(plan: Any, config_delta: Dict[str, Any]) -> Any:
        hyperparameters = dict(getattr(plan, "hyperparameters", {}))
        updates: Dict[str, Any] = {}
        for key, value in config_delta.items():
            if key == "hyperparameters" and isinstance(value, dict):
                hyperparameters.update(value)
            elif key == "fim_lambda_multiplier" and isinstance(value, (int, float)):
                updates["fim_lambda"] = float(plan.fim_lambda) * float(value)
            elif key == "drift_budget_multiplier" and isinstance(value, (int, float)):
                updates["drift_budget"] = float(plan.drift_budget) * float(value)
            elif key == "bregman_budget_multiplier" and isinstance(value, (int, float)):
                updates["bregman_budget"] = float(plan.bregman_budget) * float(value)
            elif key in {"fim_lambda", "bregman_budget", "drift_budget", "protected_layers", "mutable_layers"}:
                updates[key] = value
            else:
                hyperparameters[key] = value
        updates["hyperparameters"] = hyperparameters
        return plan.model_copy(update=updates)

    def _execution_path_allowed_unsandboxed(self, source_path: str) -> bool:
        return source_path in set(self.execution_policy.allowed_unsandboxed_paths)

    def _assert_execution_policy(
        self,
        *,
        execution_kind: str,
        source_path: str,
        sandboxed: bool,
        deliberate_exception_reason: Optional[str] = None,
    ) -> None:
        if sandboxed or self._execution_path_allowed_unsandboxed(source_path):
            return
        if execution_kind == "trusted_internal":
            if deliberate_exception_reason:
                return
            raise ExecutionPolicyViolation(
                f"Unsandboxed trusted internal execution requires an explicit documented exception: {source_path}"
            )
        if execution_kind == "generated_code" and self.execution_policy.require_sandbox_for_generated_code:
            raise ExecutionPolicyViolation(
                f"Execution policy {self.execution_policy.policy_version} forbids unsandboxed generated code: {source_path}"
            )
        if execution_kind == "external_code" and self.execution_policy.require_sandbox_for_external_code:
            raise ExecutionPolicyViolation(
                f"Execution policy {self.execution_policy.policy_version} forbids unsandboxed external code: {source_path}"
            )

    def _run_subprocess(
        self,
        command: list[str],
        *,
        source_path: str,
        execution_kind: str,
        sandboxed: bool = False,
        deliberate_exception_reason: Optional[str] = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        self._assert_execution_policy(
            execution_kind=execution_kind,
            source_path=source_path,
            sandboxed=sandboxed,
            deliberate_exception_reason=deliberate_exception_reason,
        )
        return subprocess.run(command, **kwargs)

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

    def _sync_memory(self) -> bool:
        if self.memory_indexer is None:
            return False
        try:
            self.memory_indexer.sync_once()
            self.memory_error = None
            return True
        except MemoryIntegrityError as exc:
            if self.vault is not None:
                self.vault.mark_degraded(str(exc))
            self.memory_error = str(exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive boundary
            if self.vault is not None:
                self.vault.mark_degraded(str(exc))
            self.memory_error = str(exc)
            return False

    def _retrieve_strategy_hits(self, metrics: Optional[list[GovernorMetrics]] = None) -> list[Any]:
        synced = self._sync_memory()
        if self.vault is None or not synced:
            return []
        return self.vault.search_similar_trials(metrics or self.store.tail_metrics(3))

    def _retrieve_literature_policy_hits(self, objective_slug: str, limit: int = 6) -> list[MemorySearchHit]:
        synced = self._sync_memory()
        if self.vault is None or not synced:
            return []
        query = objective_slug.replace("-", " ")
        hits: list[MemorySearchHit] = []
        for kind, count in (("paper_claim", 4), ("research", 2)):
            try:
                rows = self.vault.search(query, n_results=count, kind=kind, require_research_grade=True)
            except Exception:
                rows = self.vault.search(query, n_results=count, kind=kind)
            hits.extend(rows)
        deduped: dict[str, MemorySearchHit] = {}
        for hit in hits:
            deduped.setdefault(hit.document_id, hit)
        return list(deduped.values())[: max(1, limit)]

    def _distil_literature_policy_signal(self, objective_slug: str) -> Optional[LiteraturePolicySignal]:
        hits = self._retrieve_literature_policy_hits(objective_slug)
        if not hits:
            return None
        topic_counts: dict[str, int] = {}
        cited_document_ids: list[str] = []
        positive = 0
        negative = 0
        anchor_tokens = {"anchor", "continual", "forget", "memory", "retention", "stability"}
        drift_tokens = {"drift", "entropy", "instability", "collapse", "recovery", "quench", "noise"}
        layer_tokens = {"layer", "freeze", "depth", "capacity", "backbone", "adapter"}
        for hit in hits:
            metadata = hit.metadata or {}
            cited_document_ids.append(
                str(
                    metadata.get("paper_id")
                    or metadata.get("document_id")
                    or hit.document_id
                )
            )
            polarity = str(metadata.get("polarity") or "").lower()
            if polarity == "positive":
                positive += 1
            elif polarity == "negative":
                negative += 1
            for token in re.findall(r"[a-z0-9]+", hit.document.lower()):
                if len(token) <= 4 or token in {"paper", "result", "study", "model", "method"}:
                    continue
                topic_counts[token] = topic_counts.get(token, 0) + 1
        topic_terms = [token for token, _ in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))[:8]]
        token_set = set(topic_terms)
        contradiction_pressure = 0.0
        if positive and negative:
            contradiction_pressure = min(1.0, 0.45 + (0.1 * min(positive, negative)))
        elif negative:
            contradiction_pressure = min(1.0, 0.2 + (0.1 * negative))
        recommended_family = None
        rationale: list[str] = []
        if token_set & drift_tokens or contradiction_pressure >= 0.45:
            recommended_family = "ou_drift_jitter"
            rationale.append("literature signal emphasizes instability, drift, or contradiction pressure")
        elif token_set & anchor_tokens:
            recommended_family = "elastic_anchor"
            rationale.append("literature signal emphasizes anchoring and retention pressure")
        elif token_set & layer_tokens:
            recommended_family = "layer_freeze"
            rationale.append("literature signal emphasizes layer-localized or capacity effects")
        dominant_polarity = "mixed"
        if positive and not negative:
            dominant_polarity = "positive"
        elif negative and not positive:
            dominant_polarity = "negative"
        elif not positive and not negative:
            dominant_polarity = "neutral"
        confidence = min(
            1.0,
            0.2
            + (0.08 * len(hits))
            + (0.15 if recommended_family is not None else 0.0)
            + (0.1 if contradiction_pressure >= 0.45 else 0.0),
        )
        return LiteraturePolicySignal(
            objective_slug=objective_slug,
            evidence_count=len(hits),
            recommended_family=recommended_family,
            dominant_polarity=dominant_polarity,  # type: ignore[arg-type]
            contradiction_pressure=round(contradiction_pressure, 6),
            confidence=round(confidence, 6),
            topic_terms=topic_terms,
            cited_document_ids=sorted(dict.fromkeys(cited_document_ids)),
            rationale=rationale or ["literature signal was present but not decisive"],
        )

    def plan_trial(self, dry_run: bool = False) -> tuple[Any, Any, Any]:
        self._ensure_recent_metrics()
        run_intent = self._resolve_run_intent(dry_run=dry_run)
        data_bundle = self._ensure_data_prepared(dry_run=dry_run, run_intent=run_intent)
        trial_id = self.store.next_trial_id()
        recovery = self.store.load_recovery()
        self.store.save_recovery(recovery.model_copy(update={"trial_id": trial_id, "status": "planning"}))
        memory_hits = self._retrieve_strategy_hits()
        literature_signal = self._distil_literature_policy_signal("thermodynamic-anchor")
        policy, plan, task = self.hierarchy.produce_bundle(
            self.store,
            trial_id=trial_id,
            workspace=self.workspace,
            dry_run=dry_run,
            memory_hits=memory_hits,
            literature_signal=literature_signal,
        )
        payload = self._build_payload_config(plan, data_bundle, run_intent=run_intent)
        payload_path = self.store.write_payload_config(payload)
        payload_env = self.prepare_payload_environment()
        if payload_env.image_manifest is None or payload_env.run_manifest is None:
            raise ReproducibilityLockError(
                payload_env.lock_incomplete_reason
                or "Locked payload environment was not prepared correctly."
            )
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

    def propose_experiment_family(
        self,
        objective_slug: str,
        trigger_reason: str,
    ) -> GenerativeDirectorProposal:
        policy = self._proposal_policy_stub(objective_slug)
        proposal = GenerativeDirector(
            workspace_root=self.workspace,
            operator_role=self._family_operator_role(),
        ).propose_family(policy, trigger_reason)
        self.store.save_family_proposal(proposal)
        self.store.append_audit_event(
            "family_proposal",
            "propose_experiment_family",
            {
                "proposal_id": proposal.proposal_id,
                "objective_slug": objective_slug,
                "family_id": proposal.proposed_family.family_id,
                "family_name": proposal.proposed_family.name,
                "operator_available": proposal.operator_available,
            },
        )
        return proposal

    def run_family_feasibility(self, proposal_id: str) -> ProposedExperimentFamily:
        proposal = self._get_family_proposal(proposal_id)
        running_family = proposal.proposed_family.model_copy(
            update={
                "status": "feasibility_running",
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        )
        running_proposal = proposal.model_copy(update={"proposed_family": running_family})
        self.store.save_family_proposal(running_proposal)
        try:
            policy = self._proposal_policy_stub(proposal.objective_slug)
            plan = self.hierarchy.rule_strategist.propose(policy, memory_hits=[])
            feasible_plan = self._apply_family_config_delta(plan, running_family.config_delta)
            task = self.hierarchy.rule_scout.propose(feasible_plan, workspace=self.workspace, dry_run=True)
            updated_family = running_family.model_copy(
                update={
                    "status": "approved",
                    "feasibility_note": "dry_run_feasibility_passed",
                    "feasibility_trial_id": task.trial_id,
                    "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                }
            )
        except Exception as exc:
            updated_family = running_family.model_copy(
                update={
                    "status": "feasibility_failed",
                    "feasibility_note": str(exc),
                    "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                }
            )
        self.store.save_family_proposal(
            running_proposal.model_copy(update={"proposed_family": updated_family})
        )
        self.store.append_audit_event(
            "family_proposal",
            "run_family_feasibility",
            {
                "proposal_id": proposal_id,
                "family_id": updated_family.family_id,
                "status": updated_family.status,
                "feasibility_trial_id": updated_family.feasibility_trial_id,
            },
        )
        return updated_family

    def approve_family_proposal(self, proposal_id: str) -> ProposedExperimentFamily:
        proposal = self._get_family_proposal(proposal_id)
        approved_family = proposal.proposed_family.model_copy(
            update={
                "status": "approved",
                "approved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        )
        self.store.save_family_proposal(
            proposal.model_copy(update={"proposed_family": approved_family})
        )
        registered = self.store.load_registered_families()
        entries = [item for item in registered.entries if item.family_id != approved_family.family_id]
        entries.append(approved_family)
        self.store.save_registered_families(registered.model_copy(update={"entries": entries}))
        self.store.append_audit_event(
            "family_proposal",
            "approve_family_proposal",
            {"proposal_id": proposal_id, "family_id": approved_family.family_id},
        )
        return approved_family

    def reject_family_proposal(self, proposal_id: str, reason: str) -> None:
        proposal = self._get_family_proposal(proposal_id)
        rejected_family = proposal.proposed_family.model_copy(
            update={
                "status": "rejected",
                "rejection_reason": reason,
                "rejected_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        )
        self.store.save_family_proposal(
            proposal.model_copy(update={"proposed_family": rejected_family})
        )
        self.store.append_audit_event(
            "family_proposal",
            "reject_family_proposal",
            {"proposal_id": proposal_id, "family_id": rejected_family.family_id, "reason": reason},
        )

    def list_family_proposals(self) -> list[GenerativeDirectorProposal]:
        proposals = self.store.load_family_proposals()
        return sorted(proposals, key=lambda item: (item.created_at, item.proposal_id), reverse=True)

    def list_registered_families(self) -> list[ProposedExperimentFamily]:
        families = self.store.get_approved_families()
        return sorted(
            families,
            key=lambda item: (item.approved_at or "", item.updated_at, item.created_at, item.family_id),
            reverse=True,
        )

    def _self_improvement_engine(self) -> SelfImprovementEngine:
        return SelfImprovementEngine(
            workspace_root=self.workspace,
            policy=SelfImprovementPolicy(),
        )

    def initialize_anchor_pack(
        self,
        pack_path: str,
        run_manifest_path: str,
        baseline_mean_score: float,
        baseline_overclaim_rate: float,
    ) -> FrozenAnchorPackManifest:
        return self._self_improvement_engine().initialize_anchor_pack(
            pack_path=pack_path,
            run_manifest_path=run_manifest_path,
            baseline_mean_score=baseline_mean_score,
            baseline_overclaim_rate=baseline_overclaim_rate,
        )

    def curate_training_signal(self, signal: TrainingSignalRecord) -> bool:
        return self._self_improvement_engine().curate_signal(signal)

    def list_training_signals(self) -> list[TrainingSignalRecord]:
        return self._self_improvement_engine().list_signals()

    def assemble_curated_delta(self, cycle_id: str) -> CuratedDeltaRecord:
        engine = self._self_improvement_engine()
        cycle = engine.load_cycle(cycle_id) if cycle_id else None
        if cycle is None:
            cycle = engine.start_cycle()
        delta = engine.assemble_delta(cycle.cycle_id)
        updated_cycle = cycle.model_copy(
            update={
                "delta_id": delta.delta_id,
                "status": "curating",
                "updated_at": _utc_now(),
            }
        )
        engine.save_cycle(updated_cycle)
        return delta

    def run_self_improvement_probe(self, cycle_id: str) -> RetrainRecord:
        engine = self._self_improvement_engine()
        cycle = engine.load_cycle(cycle_id) if cycle_id else None
        if cycle is None:
            cycle = engine.start_cycle()
        anchor = engine.load_anchor_manifest()
        probing_cycle = cycle.model_copy(update={"status": "probing", "updated_at": _utc_now()})
        engine.save_cycle(probing_cycle)
        anchor_hash_verified = engine.verify_anchor_integrity()
        passed, reason = engine.evaluate_gate(
            probe_mean_score=anchor.baseline_mean_score,
            probe_overclaim_rate=0.0,
            anchor_hash_verified=anchor_hash_verified,
        )
        retrain = RetrainRecord(
            retrain_id=f"retrain-{hashlib.sha256(f'{probing_cycle.cycle_id}:probe'.encode('utf-8')).hexdigest()[:8]}",
            cycle_id=probing_cycle.cycle_id,
            delta_id=probing_cycle.delta_id or "",
            run_kind="probe",
            probe_mean_score=anchor.baseline_mean_score,
            probe_overclaim_rate=0.0,
            anchor_hash_verified=anchor_hash_verified,
            gate_passed=passed,
            gate_failure_reason=None if passed else reason,
            completed_at=_utc_now(),
            notes=["heuristic_probe_stub"],
        )
        if passed:
            completed_cycle = probing_cycle.model_copy(
                update={
                    "probe_retrain_id": retrain.retrain_id,
                    "status": "completed",
                    "total_cycles_completed": probing_cycle.total_cycles_completed + 1,
                    "consecutive_gate_failures": 0,
                    "paused_reason": None,
                    "human_resume_required": False,
                    "updated_at": _utc_now(),
                }
            )
            engine.save_cycle(completed_cycle)
        else:
            failed_cycle = engine.record_gate_failure(probing_cycle, reason)
            engine.save_cycle(
                failed_cycle.model_copy(
                    update={
                        "probe_retrain_id": retrain.retrain_id,
                        "updated_at": _utc_now(),
                    }
                )
            )
        return retrain

    def run_self_improvement_run1(self, cycle_id: str, delta_id: str) -> RetrainRecord:
        raise NotImplementedError("run1 requires pod with CUDA; use pod session")

    def deploy_improved_adapter(self, cycle_id: str, retrain_id: str) -> str:
        raise NotImplementedError("deployment requires pod with CUDA; use pod session")

    def self_improvement_status(self) -> SelfImprovementCycleRecord:
        return self._self_improvement_engine().current_status()

    def resume_self_improvement(self, cycle_id: str) -> SelfImprovementCycleRecord:
        return self._self_improvement_engine().resume_self_improvement(cycle_id)

    def run_agenda_review(self) -> AgendaReviewRecord:
        from tar_lab.agenda import AgendaEngine

        return AgendaEngine(str(self.workspace), self).run_agenda_review()

    def agenda_status(self) -> AgendaSnapshot:
        from tar_lab.agenda import AgendaEngine

        return AgendaEngine(str(self.workspace), self).get_snapshot()

    def list_agenda_decisions(self, status: Optional[str] = None) -> list[AgendaDecisionRecord]:
        from tar_lab.agenda import AgendaEngine

        return AgendaEngine(str(self.workspace), self)._list_decisions(status=status)

    def veto_agenda_decision(self, decision_id: str, reason: str) -> AgendaDecisionRecord:
        from tar_lab.agenda import AgendaEngine

        return AgendaEngine(str(self.workspace), self).veto_agenda_decision(decision_id, reason)

    def commit_agenda_decisions(self) -> list[AgendaDecisionRecord]:
        from tar_lab.agenda import AgendaEngine

        return AgendaEngine(str(self.workspace), self).commit_pending_decisions()

    def update_agenda_config(self, config: AgendaReviewConfig) -> None:
        from tar_lab.agenda import AgendaEngine

        AgendaEngine(str(self.workspace), self).update_config(config)

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

    def literature_status(self) -> dict[str, Any]:
        return self.literature_engine.status()

    def list_paper_artifacts(self, limit: int = 20) -> dict[str, Any]:
        return {
            "artifacts": self.literature_engine.list_artifacts(limit=limit),
            "count": len(list(self.literature_engine.iter_artifacts())),
            "latest_manifest": (
                self.literature_engine.latest_manifest().model_dump(mode="json")
                if self.literature_engine.latest_manifest() is not None
                else None
            ),
        }

    def paper_artifact(self, paper_id: str) -> dict[str, Any]:
        artifact = self.literature_engine.get_artifact(paper_id)
        if artifact is None:
            raise RuntimeError(f"Paper artifact not found: {paper_id}")
        conflicts = self.literature_engine.conflict_report(paper_id=paper_id, limit=100)
        return {
            "artifact": artifact.model_dump(mode="json"),
            "conflicts": conflicts["conflicts"],
            "conflict_count": conflicts["count"],
        }

    def literature_conflicts(self, *, paper_id: Optional[str] = None, limit: int = 20) -> dict[str, Any]:
        return self.literature_engine.conflict_report(paper_id=paper_id, limit=limit)

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
        self._assert_execution_policy(
            execution_kind="external_code",
            source_path="tar_lab.docker_runner.build_payload_environment",
            sandboxed=True,
        )
        build = self.docker_runner.build_payload_environment(report, dry_run=False)
        report = report.model_copy(
            update={
                "build_status": "built" if build.returncode == 0 else "failed",
                "build_command": build.command,
            }
        )
        report = self.payload_environment.attach_payload_build_attestation(
            report,
            build_result=build,
        )
        self.store.append_audit_event(
            "reproducibility",
            "rebuild_locked_image",
            {
                "image_tag": report.image_tag,
                "build_status": report.build_status,
                "manifest_hash": report.run_manifest.hash_sha256 if report.run_manifest is not None else None,
                "build_attestation_id": report.build_attestation.attestation_id if report.build_attestation else None,
                "image_digest": report.build_attestation.image_digest if report.build_attestation else None,
                "unresolved_dependencies": report.unresolved_packages,
                "lock_incomplete_reason": report.lock_incomplete_reason,
            },
        )
        return report

    @staticmethod
    def _frontier_signature(text: str) -> set[str]:
        tokens: set[str] = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) <= 3:
                continue
            if token.endswith("ing") and len(token) > 5:
                token = token[:-3]
            elif token.endswith("ed") and len(token) > 4:
                token = token[:-2]
            elif token.endswith("s") and len(token) > 4:
                token = token[:-1]
            if token in {"using", "study", "paper", "result", "method", "approach"}:
                continue
            tokens.add(token)
        return tokens

    def _frontier_similarity(self, left: str, right: str) -> float:
        if not left.strip() or not right.strip():
            return 0.0
        if self.vault is not None and getattr(self.vault, "semantic_ready", False):
            try:
                left_embedding = self.vault.embedder.embed(left)
                right_embedding = self.vault.embedder.embed(right)
                return max(0.0, min(1.0, VectorVault._cosine(left_embedding, right_embedding)))
            except Exception:
                pass
        left_tokens = self._frontier_signature(left)
        right_tokens = self._frontier_signature(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return max(0.0, min(1.0, overlap / max(1, union)))

    def _novelty_score(self, description: str, existing_projects: list[ResearchProject]) -> float:
        if not existing_projects:
            return 1.0
        similarity = max(
            self._frontier_similarity(description, f"{project.title} {project.goal}")
            for project in existing_projects
        )
        return round(max(0.0, min(1.0, 1.0 - similarity)), 6)

    def _resolve_frontier_domain_profile(self, description: str) -> Optional[str]:
        resolution = self.resolve_problem(description, benchmark_tier="validation")
        if not resolution.matched_keywords:
            return None
        if resolution.confidence < 0.32:
            return None
        return resolution.profile_id

    def _extract_frontier_gaps(self, topic: str, limit: int = 20) -> list[FrontierGapRecord]:
        documents = list(self.store.iter_research_documents())
        statements: list[tuple[str, str]] = []
        for document in documents:
            for statement in document.problem_statements:
                cleaned = statement.strip()
                if cleaned:
                    statements.append((document.document_id, cleaned))
        if not statements:
            return []

        parent = list(range(len(statements)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        signatures = [self._frontier_signature(statement) for _, statement in statements]
        for left in range(len(statements)):
            for right in range(left + 1, len(statements)):
                if statements[left][0] == statements[right][0]:
                    continue
                overlap = len(signatures[left] & signatures[right])
                if overlap >= 3:
                    union(left, right)

        clusters: dict[int, list[tuple[str, str]]] = {}
        for index, item in enumerate(statements):
            clusters.setdefault(find(index), []).append(item)

        projects = self.store.list_research_projects()
        gaps: list[FrontierGapRecord] = []
        for items in clusters.values():
            document_ids = sorted({document_id for document_id, _ in items})
            if len(document_ids) < 2:
                continue
            descriptions = [statement for _, statement in items]
            description = sorted(descriptions, key=lambda value: (-len(self._frontier_signature(value)), -len(value), value))[0]
            similarity_to_existing = 0.0
            if projects:
                similarity_to_existing = max(
                    self._frontier_similarity(description, f"{project.title} {project.goal}")
                    for project in projects
                )
            novelty_score = self._novelty_score(description, projects)
            domain_profile = self._resolve_frontier_domain_profile(description)
            confidence = min(0.95, 0.2 + (0.14 * len(document_ids)) + (0.25 * novelty_score))
            gaps.append(
                FrontierGapRecord(
                    gap_id=self._continuity_id("gap"),
                    content_hash=_compute_gap_content_hash(document_ids, description),
                    description=description,
                    domain_profile=domain_profile,
                    evidence_count=len(document_ids),
                    source_document_ids=document_ids,
                    novelty_score=round(novelty_score, 6),
                    similarity_to_existing=round(max(0.0, min(1.0, similarity_to_existing)), 6),
                    confidence=round(confidence, 6),
                )
            )
        gaps.sort(key=lambda item: (item.confidence, item.novelty_score, item.evidence_count, item.created_at), reverse=True)
        return gaps[: max(1, limit)]

    def scan_frontier_gaps(
        self,
        topic: str = "thermodynamic continual learning",
        max_gaps: int = 10,
    ) -> FrontierGapScanReport:
        if not list(self.store.iter_research_documents()):
            try:
                self.ingest_research(topic=topic, max_results=max(4, max_gaps))
            except Exception:
                pass
        scan_id = self._continuity_id("gap-scan")
        gaps = self._extract_frontier_gaps(topic, limit=max_gaps)
        persisted: list[FrontierGapRecord] = []
        existing_gap_hashes = {
            gap.content_hash
            for gap in self.store.iter_frontier_gaps()
            if gap.content_hash
        }
        rejected = 0
        skipped_cross_scan = 0
        for gap in gaps:
            updated_gap = gap.model_copy(update={"scan_id": scan_id})
            if updated_gap.content_hash and updated_gap.content_hash in existing_gap_hashes:
                skipped_cross_scan += 1
                continue
            if gap.similarity_to_existing > 0.75:
                updated_gap = updated_gap.model_copy(
                    update={
                        "status": "rejected",
                        "rejection_reason": "too_similar_to_existing_project",
                    }
                )
            elif gap.domain_profile is None:
                updated_gap = updated_gap.model_copy(
                    update={
                        "status": "rejected",
                        "rejection_reason": "domain_profile_unresolved",
                    }
                )
            if updated_gap.status == "rejected":
                rejected += 1
            self.store.append_frontier_gap(updated_gap)
            persisted.append(updated_gap)
        report = FrontierGapScanReport(
            scan_id=scan_id,
            topic=topic,
            gaps_identified=len([item for item in persisted if item.status == "identified"]),
            gaps_proposed=0,
            gaps_rejected=rejected,
            gaps_skipped_cross_scan=skipped_cross_scan,
            gaps=persisted,
            existing_project_count=len(self.store.list_research_projects()),
            retrieval_mode="semantic" if self.vault is not None and getattr(self.vault, "semantic_ready", False) else "lexical_fallback",
        )
        self.store.append_gap_scan_report(report)
        self.store.append_audit_event(
            "frontier",
            "scan_frontier_gaps",
            {
                "scan_id": report.scan_id,
                "topic": topic,
                "gaps_identified": report.gaps_identified,
                "gaps_rejected": report.gaps_rejected,
                "gaps_skipped_cross_scan": report.gaps_skipped_cross_scan,
            },
        )
        return report

    def frontier_gap_status(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        gaps = list(self.store.iter_frontier_gaps())
        scans = list(self.store.iter_gap_scan_reports())
        counts = {
            "total": len(gaps),
            "identified": len([item for item in gaps if item.status == "identified"]),
            "proposed": len([item for item in gaps if item.status == "proposed"]),
            "rejected": len([item for item in gaps if item.status == "rejected"]),
            "promoted": len([item for item in gaps if item.status == "promoted"]),
        }
        filtered = [
            item
            for item in gaps
            if (status is None or item.status == status) and item.confidence >= min_confidence
        ]
        filtered = sorted(filtered, key=lambda item: (item.created_at, item.gap_id), reverse=True)
        latest_scan = scans[-1] if scans else None
        recent_scans = sorted(scans, key=lambda item: (item.created_at, item.scan_id), reverse=True)[: max(1, limit)]
        return {
            "counts": counts,
            "scan_count": len(scans),
            "latest_scan": latest_scan.model_dump(mode="json") if latest_scan else None,
            "recent_scans": [item.model_dump(mode="json") for item in recent_scans],
            "gaps": [item.model_dump(mode="json") for item in filtered[: max(1, limit)]],
            "status_filter": status,
            "min_confidence": round(max(0.0, min(1.0, min_confidence)), 6),
        }

    def frontier_gap_scan_history(
        self,
        *,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        scans = list(self.store.iter_gap_scan_reports())
        if topic:
            topic_lower = topic.strip().lower()
            scans = [item for item in scans if topic_lower in item.topic.lower()]
        scans = sorted(scans, key=lambda item: (item.created_at, item.scan_id), reverse=True)
        recent = scans[: max(1, limit)]
        return {
            "scan_count": len(scans),
            "topic_filter": topic,
            "latest_scan": recent[0].model_dump(mode="json") if recent else None,
            "scans": [item.model_dump(mode="json") for item in recent],
        }

    def show_manifest(self, manifest_id: Optional[str] = None, manifest_path: Optional[str] = None) -> dict[str, Any]:
        if manifest_path:
            manifest = self.store.load_run_manifest_path(manifest_path)
        elif manifest_id:
            manifest = self.store.load_run_manifest(manifest_id)
        else:
            payload_env = self.payload_environment.load()
            manifest = payload_env.run_manifest if payload_env is not None else None
            if manifest is None and payload_env is not None:
                return {
                    "manifest_found": False,
                    "image_tag": payload_env.image_tag,
                    "reproducibility_complete": payload_env.reproducibility_complete,
                    "unresolved_packages": payload_env.unresolved_packages,
                    "lock_incomplete_reason": payload_env.lock_incomplete_reason,
                    "packages": payload_env.packages,
                }
        if manifest is None:
            raise RuntimeError("Manifest not found.")
        return manifest.model_dump(mode="json")

    def list_alerts(self, count: int = 20) -> dict[str, Any]:
        return {"alerts": [item.model_dump(mode="json") for item in self.store.latest_alerts(count)]}

    def runtime_status(self) -> dict[str, Any]:
        schedules = list(self.store.iter_problem_schedules())
        heartbeat = self.runtime_daemon.load_heartbeat()
        payload_env = self.payload_environment.load()
        sandbox_policy = self.sandbox_policy()
        runtime_policy = self.store.load_runtime_policy()
        verdict_summary = self._claim_verdict_lifecycle_summary(limit=20)
        queue_health = self.queue_health()
        build_attestation = (
            payload_env.build_attestation
            if payload_env is not None and payload_env.build_attestation is not None
            else self.store.latest_build_attestation(scope_kind="payload_environment")
        )
        return {
            "heartbeat": heartbeat.model_dump(mode="json") if heartbeat is not None else None,
            "active_leases": [item.model_dump(mode="json") for item in schedules if item.status in {"leased", "running"}],
            "retry_waiting": [item.model_dump(mode="json") for item in schedules if item.status == "retry_wait"],
            "recoverable_crashes": [item.model_dump(mode="json") for item in schedules if item.status == "recoverable_crash"],
            "terminal_failures": [item.model_dump(mode="json") for item in schedules if item.status == "failed_terminal"],
            "alerts": [item.model_dump(mode="json") for item in self.store.latest_alerts(20)],
            "safe_execution_mode": self.safe_execution_mode,
            "execution_policy": self.execution_policy.model_dump(mode="json"),
            "sandbox_policy": sandbox_policy,
            "runtime_policy": runtime_policy.model_dump(mode="json"),
            "claim_verdict_lifecycle": verdict_summary["counts"],
            "recent_verdict_window": verdict_summary["window"],
            "escalated_verdict_ids": verdict_summary["escalated_verdict_ids"],
            "queue_health": queue_health,
            "payload_image": payload_env.image_tag if payload_env is not None else None,
            "manifest_hash": payload_env.run_manifest.hash_sha256 if payload_env and payload_env.run_manifest else None,
            "reproducibility_complete": bool(payload_env is not None and payload_env.reproducibility_complete),
            "payload_build_status": payload_env.build_status if payload_env is not None else "not_requested",
            "build_attestation_id": build_attestation.attestation_id if build_attestation is not None else None,
            "build_attestation": build_attestation.model_dump(mode="json") if build_attestation is not None else None,
            "image_digest": build_attestation.image_digest if build_attestation is not None else None,
            "unresolved_dependency_count": len(payload_env.unresolved_packages) if payload_env is not None else 0,
            "unresolved_dependencies": list(payload_env.unresolved_packages) if payload_env is not None else [],
            "lock_incomplete_reason": payload_env.lock_incomplete_reason if payload_env is not None else None,
        }

    def list_experiment_backends(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.experiment_backends.list_backends()]

    def experiment_backend_runtime_status(
        self,
        *,
        backend_id: str | None = None,
        trial_name: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        rows = self.store.list_experiment_backend_runtimes()
        if backend_id:
            rows = [item for item in rows if item.backend_id == backend_id]
        if trial_name:
            rows = [item for item in rows if item.trial_name == trial_name]
        rows = sorted(rows, key=lambda item: item.updated_at, reverse=True)
        limited = rows[: max(1, limit)]
        counts: dict[str, int] = {}
        for item in rows:
            counts[item.status] = counts.get(item.status, 0) + 1
        resumable = len([item for item in rows if item.supports_resume])
        return {
            "filters": {
                "backend_id": backend_id,
                "trial_name": trial_name,
                "limit": max(1, limit),
            },
            "counts": {
                "total": len(rows),
                "resumable": resumable,
                **counts,
            },
            "latest": limited[0].model_dump(mode="json") if limited else None,
            "records": [item.model_dump(mode="json") for item in limited],
        }

    def run_runtime_cycle(self, *, max_jobs: int = 1, stale_after_s: int = 900) -> RuntimeHeartbeat:
        reprioritized = self._reprioritize_scheduled_jobs()
        if reprioritized:
            self.store.append_audit_event(
                "runtime",
                "reprioritize_scheduled_jobs",
                {"updated_schedule_ids": [item.schedule_id for item in reprioritized]},
            )
        orphaned = self.recover_orphaned_runs()
        heartbeat = self.runtime_daemon.run_cycle(max_jobs=max_jobs, stale_after_s=stale_after_s)
        escalated_verdicts = self._age_claim_verdicts()
        if orphaned or escalated_verdicts:
            notes = list(heartbeat.notes)
            if orphaned:
                notes.append(f"orphan_recoveries={len(orphaned)}")
            if escalated_verdicts:
                notes.append(f"escalated_verdicts={len(escalated_verdicts)}")
            heartbeat = heartbeat.model_copy(
                update={
                    "stale_cleanups": heartbeat.stale_cleanups + len(orphaned),
                    "escalated_verdicts": escalated_verdicts,
                    "notes": notes,
                }
            )
            self.runtime_daemon.heartbeat_path.write_text(heartbeat.model_dump_json(indent=2), encoding="utf-8")
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

    def confirm_recovery(self, schedule_id: str) -> ProblemScheduleEntry:
        entry = self.scheduler.confirm_recovery(schedule_id)
        self.store.append_audit_event("runtime", "confirm_recovery", entry.model_dump(mode="json"))
        return entry

    def cancel_job(self, schedule_id: str) -> ProblemScheduleEntry:
        entry = self.scheduler.cancel_job(schedule_id)
        self.store.append_audit_event("runtime", "cancel_job", entry.model_dump(mode="json"))
        return entry

    def _active_process_commands(self) -> list[str]:
        try:
            if os.name == "nt":
                # execution_policy: deliberate_exception - reason: trusted internal host process inspection.
                result = self._run_subprocess(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "Get-CimInstance Win32_Process | Select-Object -ExpandProperty CommandLine",
                    ],
                    source_path="tar_lab.orchestrator._active_process_commands",
                    execution_kind="trusted_internal",
                    deliberate_exception_reason="trusted internal host process inspection",
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            else:
                # execution_policy: deliberate_exception - reason: trusted internal host process inspection.
                result = self._run_subprocess(
                    ["ps", "-ax", "-o", "command="],
                    source_path="tar_lab.orchestrator._active_process_commands",
                    execution_kind="trusted_internal",
                    deliberate_exception_reason="trusted internal host process inspection",
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
        except Exception:
            return []
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _schedule_has_active_owner(
        self,
        schedule_id: str,
        *,
        process_commands: Optional[list[str]] = None,
    ) -> bool:
        if schedule_id in self.scheduler.active_schedule_ids():
            return True
        commands = process_commands if process_commands is not None else self._active_process_commands()
        return any(schedule_id in command for command in commands)

    def recover_orphaned_runs(self) -> list[ProblemScheduleEntry]:
        now_dt = datetime.now(timezone.utc)
        entries = list(self.store.iter_problem_schedules())
        active_rows = [item for item in entries if item.status in {"running", "leased"} and item.lease is not None]
        process_commands = self._active_process_commands() if active_rows else []
        recovered: list[ProblemScheduleEntry] = []
        for entry in active_rows:
            if entry.status not in {"running", "leased"} or entry.lease is None:
                continue
            heartbeat_at = self._parse_timestamp(entry.lease.heartbeat_at)
            if heartbeat_at is None:
                continue
            stale_after = timedelta(seconds=max(1, entry.lease.heartbeat_interval_s * 3))
            if heartbeat_at + stale_after > now_dt:
                continue
            if self._schedule_has_active_owner(entry.schedule_id, process_commands=process_commands):
                continue
            updated = self.store.update_problem_schedule(
                entry.schedule_id,
                status="recoverable_crash",
                lease=None,
                last_error="orphan_detected",
                crash_provenance="orphan_detected",
                crash_at=now_dt.replace(microsecond=0).isoformat(),
                recovery_required=True,
            )
            if updated is None:
                continue
            recovered.append(updated)
            alert = AlertRecord(
                alert_id=self._continuity_id("alert"),
                severity="critical",
                source="runtime",
                message=f"Orphaned schedule detected for {entry.schedule_id}",
                related_schedule_id=entry.schedule_id,
                metadata={
                    "status": "recoverable_crash",
                    "problem_id": entry.problem_id,
                    "reason": "orphan_detected",
                },
            )
            self.store.append_alert(alert)
            refreshed = self.store.get_problem_schedule(entry.schedule_id)
            if refreshed is not None:
                self.store.update_problem_schedule(
                    entry.schedule_id,
                    alert_ids=[*refreshed.alert_ids, alert.alert_id],
                )
            self.store.append_audit_event(
                "runtime",
                "recover_orphaned_run",
                {
                    "schedule_id": entry.schedule_id,
                    "problem_id": entry.problem_id,
                    "heartbeat_at": entry.lease.heartbeat_at,
                },
            )
        return recovered

    def queue_health(self) -> dict[str, Any]:
        entries = list(self.store.iter_problem_schedules())
        now_dt = datetime.now(timezone.utc)
        active_rows = [item for item in entries if item.status in {"leased", "running"} and item.lease is not None]
        process_commands = self._active_process_commands() if active_rows else []
        stale_lease_count = 0
        orphan_count = 0
        oldest_pending_age_minutes = 0.0
        pending_statuses = {"scheduled", "leased", "running", "retry_wait", "recoverable_crash"}
        pending_ages: list[float] = []
        completed_at: list[datetime] = []
        failed_at: list[datetime] = []

        for entry in entries:
            if entry.status in pending_statuses:
                created_at = self._parse_timestamp(entry.created_at)
                if created_at is not None:
                    pending_ages.append(max(0.0, (now_dt - created_at).total_seconds() / 60.0))
            if entry.status == "completed":
                last = self._parse_timestamp(entry.last_execution_at or entry.created_at)
                if last is not None:
                    completed_at.append(last)
            if entry.status in {"failed_terminal", "recoverable_crash"}:
                last_failed = self._parse_timestamp(entry.crash_at or entry.last_execution_at or entry.created_at)
                if last_failed is not None:
                    failed_at.append(last_failed)
            if entry.status in {"leased", "running"} and entry.lease is not None:
                heartbeat_at = self._parse_timestamp(entry.lease.heartbeat_at)
                expires_at = self._parse_timestamp(entry.lease.expires_at)
                if heartbeat_at is not None and heartbeat_at + timedelta(seconds=max(1, entry.lease.heartbeat_interval_s * 3)) <= now_dt:
                    stale_lease_count += 1
                    if not self._schedule_has_active_owner(entry.schedule_id, process_commands=process_commands):
                        orphan_count += 1
                elif expires_at is not None and expires_at <= now_dt:
                    stale_lease_count += 1

        if pending_ages:
            oldest_pending_age_minutes = round(max(pending_ages), 3)

        return {
            "scheduled": len([item for item in entries if item.status == "scheduled"]),
            "leased": len([item for item in entries if item.status == "leased"]),
            "running": len([item for item in entries if item.status == "running"]),
            "recoverable_crash": len([item for item in entries if item.status == "recoverable_crash"]),
            "retry_wait": len([item for item in entries if item.status == "retry_wait"]),
            "failed_terminal": len([item for item in entries if item.status == "failed_terminal"]),
            "stale_lease_count": stale_lease_count,
            "orphan_count": orphan_count,
            "oldest_pending_age_minutes": oldest_pending_age_minutes,
            "last_completed_at": max(completed_at).replace(microsecond=0).isoformat() if completed_at else None,
            "last_failed_at": max(failed_at).replace(microsecond=0).isoformat() if failed_at else None,
        }

    def sandbox_policy(self) -> dict[str, Any]:
        policy = self.payload_environment.default_sandbox_policy(artifact_dir="/workspace/tar_runs")
        payload = policy.model_dump(mode="json")
        payload["dev_override_active"] = policy.profile == "dev_override"
        payload["read_only_mount_count"] = len(policy.read_only_mounts)
        payload["writable_mount_count"] = len(policy.writable_mounts)
        return payload

    def runtime_policy(self) -> dict[str, Any]:
        return self.store.load_runtime_policy().model_dump(mode="json")

    def _project_now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _continuity_id(self, prefix: str) -> str:
        self._continuity_counter += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{prefix}-{stamp}-{self._continuity_counter:06d}"

    def _default_project_budget(self) -> ResearchBudgetLedger:
        return ResearchBudgetLedger()

    def _budget_remaining_summary(self, ledger: ResearchBudgetLedger) -> dict[str, float]:
        return {
            "wall_clock_minutes_remaining": max(0.0, ledger.wall_clock_minutes_budget - ledger.wall_clock_minutes_spent),
            "gpu_hours_remaining": max(0.0, ledger.gpu_hours_budget - ledger.gpu_hours_spent),
            "experiments_remaining": float(max(0, ledger.experiment_budget - ledger.experiments_spent)),
            "replications_remaining": float(max(0, ledger.replication_budget - ledger.replications_spent)),
        }

    def _classify_budget_pressure(self, ledger: ResearchBudgetLedger) -> tuple[bool, str]:
        ratios: list[float] = []
        if ledger.wall_clock_minutes_budget > 0:
            ratios.append(ledger.wall_clock_minutes_spent / ledger.wall_clock_minutes_budget)
        if ledger.gpu_hours_budget > 0:
            ratios.append(ledger.gpu_hours_spent / ledger.gpu_hours_budget)
        if ledger.experiment_budget > 0:
            ratios.append(ledger.experiments_spent / ledger.experiment_budget)
        if ledger.replication_budget > 0:
            ratios.append(ledger.replications_spent / ledger.replication_budget)
        max_ratio = max(ratios or [0.0])
        exhausted = max_ratio >= 1.0
        if exhausted:
            return True, "exhausted"
        if max_ratio >= 0.8:
            return False, "high"
        if max_ratio >= 0.5:
            return False, "medium"
        return False, "low"

    def _spend_project_budget(
        self,
        ledger: ResearchBudgetLedger,
        *,
        wall_clock_minutes: float = 0.0,
        gpu_hours: float = 0.0,
        experiments: int = 0,
        replications: int = 0,
    ) -> ResearchBudgetLedger:
        updated = ledger.model_copy(
            update={
                "wall_clock_minutes_spent": max(0.0, ledger.wall_clock_minutes_spent + max(0.0, wall_clock_minutes)),
                "gpu_hours_spent": max(0.0, ledger.gpu_hours_spent + max(0.0, gpu_hours)),
                "experiments_spent": max(0, ledger.experiments_spent + max(0, experiments)),
                "replications_spent": max(0, ledger.replications_spent + max(0, replications)),
            }
        )
        exhausted, pressure = self._classify_budget_pressure(updated)
        return updated.model_copy(update={"budget_exhausted": exhausted, "budget_pressure_level": pressure})

    def _project_thread(
        self,
        project: ResearchProject,
        thread_id: Optional[str] = None,
    ) -> Optional[ResearchHypothesisThread]:
        target = thread_id or project.active_thread_id
        if target is None and project.hypothesis_threads:
            return project.hypothesis_threads[0]
        for thread in project.hypothesis_threads:
            if thread.thread_id == target:
                return thread
        return None

    def _project_question(
        self,
        project: ResearchProject,
        question_id: Optional[str] = None,
    ) -> Optional[ResearchOpenQuestion]:
        target = question_id
        if target is None:
            thread = self._project_thread(project)
            if thread and thread.open_question_ids:
                target = thread.open_question_ids[-1]
        if target is None:
            return None
        for question in project.open_questions:
            if question.question_id == target:
                return question
        return None

    def _project_action(
        self,
        project: ResearchProject,
        action_id: Optional[str] = None,
    ) -> Optional[ResearchPlannedAction]:
        target = action_id
        if target is None:
            thread = self._project_thread(project)
            if thread is not None:
                target = thread.next_action_id
        if target is None:
            return None
        for action in project.planned_actions:
            if action.action_id == target:
                return action
        return None

    def _replace_project_thread(self, project: ResearchProject, thread: ResearchHypothesisThread) -> ResearchProject:
        threads = [thread if item.thread_id == thread.thread_id else item for item in project.hypothesis_threads]
        return project.model_copy(update={"hypothesis_threads": threads})

    def _replace_project_question(self, project: ResearchProject, question: ResearchOpenQuestion) -> ResearchProject:
        questions = [question if item.question_id == question.question_id else item for item in project.open_questions]
        return project.model_copy(update={"open_questions": questions})

    def _replace_project_action(self, project: ResearchProject, action: ResearchPlannedAction) -> ResearchProject:
        actions = [action if item.action_id == action.action_id else item for item in project.planned_actions]
        return project.model_copy(update={"planned_actions": actions})

    def _invalidate_active_action(self, project: ResearchProject, thread: Optional[ResearchHypothesisThread]) -> ResearchProject:
        if thread is None or not thread.next_action_id:
            return project
        action = self._project_action(project, thread.next_action_id)
        if action is None or action.status not in {"planned", "queued", "running"}:
            return project
        updated_action = action.model_copy(update={"status": "invalidated", "updated_at": self._project_now()})
        return self._replace_project_action(project, updated_action)

    def _build_resume_snapshot(
        self,
        project: ResearchProject,
        *,
        latest_evidence_summary: Optional[str] = None,
        blockers: Optional[list[str]] = None,
    ) -> ResearchResumeSnapshot:
        thread = self._project_thread(project)
        question = self._project_question(project, question_id=thread.open_question_ids[-1] if thread and thread.open_question_ids else None)
        action = self._project_action(project, action_id=thread.next_action_id if thread else None)
        return ResearchResumeSnapshot(
            project_id=project.project_id,
            active_thread_id=thread.thread_id if thread else None,
            current_question_id=question.question_id if question else None,
            next_action_id=action.action_id if action else None,
            latest_evidence_summary=latest_evidence_summary or project.latest_decision_summary or project.goal,
            blockers=list(blockers or []),
            budget_remaining_summary=self._budget_remaining_summary(project.budget_ledger),
            captured_at=self._project_now(),
        )

    def _persist_project(
        self,
        project: ResearchProject,
        *,
        latest_evidence_summary: Optional[str] = None,
        blockers: Optional[list[str]] = None,
    ) -> ResearchProject:
        updated = project.model_copy(
            update={
                "updated_at": self._project_now(),
                "resume_snapshot": self._build_resume_snapshot(
                    project,
                    latest_evidence_summary=latest_evidence_summary,
                    blockers=blockers,
                ),
            }
        )
        self.store.upsert_research_project(updated)
        return updated

    def create_project(
        self,
        problem: str,
        *,
        benchmark_tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
        status: ResearchProjectStatus = "active",
    ) -> ResearchProject:
        resolution = self.resolve_problem(
            problem,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
        )
        project_id = self._continuity_id("project")
        thread_id = self._continuity_id("thread")
        question_id = self._continuity_id("question")
        action_id = self._continuity_id("action")
        thread = ResearchHypothesisThread(
            thread_id=thread_id,
            project_id=project_id,
            hypothesis=f"Investigate: {problem}",
            status="open",
            confidence_state="exploratory",
            open_question_ids=[question_id],
            next_action_id=action_id,
        )
        question = ResearchOpenQuestion(
            question_id=question_id,
            project_id=project_id,
            thread_id=thread_id,
            question=f"What is the strongest next study for '{problem}'?",
            importance=max(0.25, resolution.confidence),
            uncertainty_type="problem_formulation",
            blocking=True,
            status="open",
        )
        action = ResearchPlannedAction(
            action_id=action_id,
            project_id=project_id,
            thread_id=thread_id,
            action_kind="create_problem_study",
            description="Create a problem study, evidence bundle, and initial benchmark plan.",
            estimated_cost=0.2,
            expected_evidence_gain=max(0.25, resolution.confidence),
        )
        project = ResearchProject(
            project_id=project_id,
            title=problem,
            goal=problem,
            domain_profile=resolution.profile_id,
            status=status,
            active_thread_id=thread_id,
            budget_ledger=self._default_project_budget(),
            latest_decision_summary=resolution.summary,
            hypothesis_threads=[thread],
            open_questions=[question],
            planned_actions=[action],
        )
        project = self._persist_project(project, latest_evidence_summary=resolution.summary)
        self.store.append_audit_event(
            "research_projects",
            "create_project",
            project.model_dump(mode="json"),
        )
        return project

    def propose_projects_from_gaps(
        self,
        max_proposals: int = 3,
        confidence_threshold: float = 0.45,
    ) -> list[ResearchProject]:
        created: list[ResearchProject] = []
        candidates = [
            gap
            for gap in self.store.iter_frontier_gaps()
            if gap.status == "identified" and gap.confidence >= confidence_threshold
        ]
        candidates = sorted(
            candidates,
            key=lambda item: (item.confidence * item.novelty_score, item.evidence_count, item.created_at),
            reverse=True,
        )
        for gap in candidates[: max(1, max_proposals)]:
            project = self.create_project(gap.description, status="proposed")
            project = self._persist_project(
                project.model_copy(update={"latest_decision_summary": "Project proposed by WS36 frontier gap scanner."}),
                latest_evidence_summary=f"Frontier gap identified from {gap.evidence_count} supporting documents.",
            )
            self.store.update_frontier_gap(
                gap.gap_id,
                status="proposed",
                proposed_project_id=project.project_id,
            )
            created.append(project)
        self.store.append_audit_event(
            "frontier",
            "propose_projects_from_gaps",
            {
                "created_project_ids": [item.project_id for item in created],
                "confidence_threshold": confidence_threshold,
            },
        )
        return created

    def promote_gap_project(self, gap_id: str, note: Optional[str] = None) -> ResearchProject:
        gap = self.store.get_frontier_gap(gap_id)
        if gap is None:
            raise RuntimeError(f"Unknown frontier gap: {gap_id}")
        if not gap.proposed_project_id:
            raise RuntimeError(f"Frontier gap has no proposed project: {gap_id}")
        project = self.store.get_research_project(gap.proposed_project_id)
        if project is None:
            raise RuntimeError(f"Unknown proposed project for frontier gap: {gap.proposed_project_id}")
        updated_project = self._persist_project(
            project.model_copy(
                update={
                    "status": "active",
                    "latest_decision_summary": note
                    or "Promoted from proposed frontier gap project to active research.",
                }
            ),
            latest_evidence_summary=f"Promoted from frontier gap {gap_id}.",
        )
        review_note = note or "operator review approved this frontier gap for active research"
        self.store.update_frontier_gap(
            gap_id,
            status="promoted",
            review_note=review_note,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.store.append_audit_event(
            "frontier",
            "promote_gap_project",
            {"gap_id": gap_id, "project_id": updated_project.project_id, "note": review_note},
        )
        return updated_project

    def reject_gap_project(self, gap_id: str, reason: str, note: Optional[str] = None) -> FrontierGapRecord:
        gap = self.store.get_frontier_gap(gap_id)
        if gap is None:
            raise RuntimeError(f"Unknown frontier gap: {gap_id}")
        review_note = note or reason
        if gap.proposed_project_id:
            project = self.store.get_research_project(gap.proposed_project_id)
            if project is not None:
                self._persist_project(
                    project.model_copy(
                        update={
                            "status": "parked",
                            "latest_decision_summary": f"Frontier gap proposal rejected: {review_note}",
                        }
                    ),
                    blockers=[reason],
                )
        updated_gap = self.store.update_frontier_gap(
            gap_id,
            status="rejected",
            rejection_reason=reason,
            review_note=review_note,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
        if updated_gap is None:
            raise RuntimeError(f"Failed to update frontier gap: {gap_id}")
        self.store.append_audit_event(
            "frontier",
            "reject_gap_project",
            {"gap_id": gap_id, "reason": reason, "note": review_note},
        )
        return updated_gap

    def list_projects(self) -> dict[str, Any]:
        projects = sorted(
            self.store.list_research_projects(),
            key=lambda item: (item.updated_at, item.created_at, item.project_id),
            reverse=True,
        )
        return {"projects": [self.project_status(item.project_id) for item in projects]}

    def _project_summary(self, project: ResearchProject) -> dict[str, Any]:
        thread = self._project_thread(project)
        action = self._project_action(project)
        return {
            "project_id": project.project_id,
            "title": project.title,
            "status": project.status,
            "priority": project.priority,
            "domain_profile": project.domain_profile,
            "active_thread_id": project.active_thread_id,
            "thread_status": thread.status if thread else None,
            "confidence_state": thread.confidence_state if thread else None,
            "next_action_id": action.action_id if action else None,
            "next_action": action.description if action else None,
            "latest_decision_summary": project.latest_decision_summary,
            "updated_at": project.updated_at,
        }

    def project_status(self, project_id: str) -> dict[str, Any]:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        thread = self._project_thread(project)
        question = self._project_question(project)
        action = self._project_action(project)
        return {
            "project": project.model_dump(mode="json"),
            "active_thread": thread.model_dump(mode="json") if thread else None,
            "current_question": question.model_dump(mode="json") if question else None,
            "next_action": action.model_dump(mode="json") if action else None,
            "budget_remaining": self._budget_remaining_summary(project.budget_ledger),
        }

    def _project_studies(self, project_id: str) -> list[ProblemStudyReport]:
        return [item for item in self.store.iter_problem_studies() if item.project_id == project_id]

    def _project_executions(self, project_id: str) -> list[ProblemExecutionReport]:
        return [item for item in self.store.iter_problem_executions() if item.project_id == project_id]

    def _project_research_decisions(self, project_id: str) -> list[ResearchDecisionRecord]:
        return [item for item in self.store.iter_research_decisions() if item.problem_id and any(study.problem_id == item.problem_id for study in self._project_studies(project_id))]

    def _project_claim_verdicts(self, project_id: str) -> list[ClaimVerdict]:
        problem_ids = {item.problem_id for item in self._project_studies(project_id)}
        return [
            item
            for item in self.store.iter_claim_verdicts()
            if item.benchmark_problem_id in problem_ids
        ]

    def _project_falsification_plans(self, project_id: str) -> list[FalsificationPlan]:
        return [item for item in self.store.iter_falsification_plans() if item.project_id == project_id]

    def _project_priority_records(self, project_id: str) -> list[ProjectPriorityRecord]:
        return [item for item in self.store.iter_project_priority_records() if item.project_id == project_id]

    def _project_related_portfolio_decisions(self, project_id: str) -> list[PortfolioDecision]:
        related: list[PortfolioDecision] = []
        for item in self.store.iter_portfolio_decisions():
            if project_id == item.selected_project_id or project_id in {
                *item.deferred_project_ids,
                *item.parked_project_ids,
                *item.resumed_project_ids,
                *item.escalated_project_ids,
                *item.retired_project_ids,
            }:
                related.append(item)
        return related

    def _project_verification_reports(self, project_id: str) -> list[VerificationReport]:
        trial_ids = {item.trial_id for item in self._project_claim_verdicts(project_id)}
        return [item for item in self.store.iter_verification_reports() if item.trial_id in trial_ids]

    def _publication_claim_bundle(self, verdict: ClaimVerdict) -> PublicationClaimBundle:
        summary = verdict.rationale[0] if verdict.rationale else f"Claim {verdict.verdict_id} is {verdict.status}."
        return PublicationClaimBundle(
            verdict_id=verdict.verdict_id,
            disposition=verdict.status,
            summary=summary,
            rationale=list(verdict.rationale),
            trial_id=verdict.trial_id,
            confidence=verdict.confidence,
            supporting_research_ids=list(verdict.supporting_research_ids),
            supporting_evidence_ids=list(verdict.supporting_evidence_ids),
            benchmark_names=list(verdict.supporting_benchmark_names),
            canonical_benchmark_required=verdict.canonical_benchmark_required,
            canonical_benchmark_satisfied=verdict.canonical_benchmark_satisfied,
            linkage_status=verdict.linkage_status,
        )

    def _publication_alternatives(
        self,
        project: ResearchProject,
        verdicts: list[ClaimVerdict],
        decisions: list[ResearchDecisionRecord],
    ) -> list[PublicationAlternativeBundle]:
        alternatives: list[PublicationAlternativeBundle] = []
        for verdict in verdicts:
            if verdict.status in {"rejected", "contradicted", "insufficient_evidence"}:
                alternatives.append(
                    PublicationAlternativeBundle(
                        alternative_id=verdict.verdict_id,
                        source="claim_verdict",
                        status=verdict.status,
                        summary=verdict.rationale[0] if verdict.rationale else f"Claim {verdict.verdict_id} was not accepted.",
                        why_rejected=list(verdict.rationale),
                        related_verdict_id=verdict.verdict_id,
                    )
                )
        for decision in decisions:
            if decision.contradiction_review is not None and decision.contradiction_review.contradiction_count > 0:
                alternatives.append(
                    PublicationAlternativeBundle(
                        alternative_id=decision.contradiction_review.review_id,
                        source="contradiction_review",
                        status="contradicted",
                        summary=decision.contradiction_review.summary,
                        why_rejected=[
                            decision.contradiction_review.recommended_resolution,
                            "Contradictory evidence remains unresolved.",
                        ],
                    )
                )
            for hypothesis in decision.hypotheses:
                if hypothesis.unresolved_assumptions:
                    alternatives.append(
                        PublicationAlternativeBundle(
                            alternative_id=hypothesis.hypothesis_id,
                            source="hypothesis",
                            status="insufficient_evidence",
                            summary=hypothesis.hypothesis,
                            why_rejected=list(hypothesis.unresolved_assumptions),
                        )
                    )
        deduped: dict[str, PublicationAlternativeBundle] = {}
        for item in alternatives:
            deduped[item.alternative_id] = item
        return list(deduped.values())

    def _publication_benchmark_attachments(
        self,
        studies: list[ProblemStudyReport],
        executions: list[ProblemExecutionReport],
        verdicts: list[ClaimVerdict],
    ) -> list[PublicationBenchmarkAttachment]:
        attachments: list[PublicationBenchmarkAttachment] = []
        for item in studies:
            attachments.append(
                PublicationBenchmarkAttachment(
                    source_id=item.problem_id,
                    source_kind="problem_study",
                    benchmark_ids=list(item.benchmark_ids),
                    benchmark_names=list(item.benchmark_names),
                    benchmark_truth_statuses=list(item.benchmark_truth_statuses),
                    benchmark_alignment=item.benchmark_alignment,
                    canonical_comparable=item.canonical_comparable,
                    requested_tier=item.benchmark_tier,
                    actual_tiers=list(item.actual_benchmark_tiers),
                )
            )
        for item in executions:
            attachments.append(
                PublicationBenchmarkAttachment(
                    source_id=item.problem_id,
                    source_kind="problem_execution",
                    benchmark_ids=list(item.benchmark_ids),
                    benchmark_names=list(item.benchmark_names),
                    benchmark_truth_statuses=list(item.benchmark_truth_statuses),
                    benchmark_alignment=item.benchmark_alignment,
                    canonical_comparable=item.canonical_comparable,
                    requested_tier=item.benchmark_tier,
                    actual_tiers=list(item.actual_benchmark_tiers),
                )
            )
        for item in verdicts:
            attachments.append(
                PublicationBenchmarkAttachment(
                    source_id=item.verdict_id,
                    source_kind="claim_verdict",
                    benchmark_ids=list(item.supporting_benchmark_ids),
                    benchmark_names=list(item.supporting_benchmark_names),
                    benchmark_truth_statuses=[],
                    benchmark_alignment="aligned" if item.canonical_benchmark_satisfied else "downgraded",
                    canonical_comparable=item.canonical_benchmark_satisfied,
                    requested_tier="canonical" if item.canonical_benchmark_required else "validation",
                    actual_tiers=[],
                )
            )
        deduped: dict[tuple[str, str], PublicationBenchmarkAttachment] = {}
        for item in attachments:
            deduped[(item.source_kind, item.source_id)] = item
        return list(deduped.values())

    def _publication_lineage(
        self,
        studies: list[ProblemStudyReport],
        executions: list[ProblemExecutionReport],
        decisions: list[ResearchDecisionRecord],
        verdicts: list[ClaimVerdict],
        verifications: list[VerificationReport],
        plans: list[FalsificationPlan],
        portfolio_decisions: list[PortfolioDecision],
    ) -> list[PublicationLineageEntry]:
        lineage: list[PublicationLineageEntry] = []
        for item in studies:
            lineage.append(
                PublicationLineageEntry(
                    event_id=f"study:{item.problem_id}",
                    timestamp=item.created_at,
                    event_type="problem_study",
                    summary=f"Problem study created for '{item.problem}' with benchmark alignment {item.benchmark_alignment}.",
                    source_id=item.problem_id,
                    metadata={
                        "benchmark_ids": list(item.benchmark_ids),
                        "benchmark_truth_statuses": list(item.benchmark_truth_statuses),
                        "canonical_comparable": item.canonical_comparable,
                    },
                )
            )
        for item in executions:
            lineage.append(
                PublicationLineageEntry(
                    event_id=f"execution:{item.problem_id}:{item.executed_at}",
                    timestamp=item.executed_at,
                    event_type="problem_execution",
                    summary=f"Problem execution finished with status {item.status} and benchmark alignment {item.benchmark_alignment}.",
                    source_id=item.problem_id,
                    metadata={
                        "execution_mode": item.execution_mode,
                        "reproducibility_complete": item.reproducibility_complete,
                        "benchmark_truth_statuses": list(item.benchmark_truth_statuses),
                    },
                )
            )
        for item in decisions:
            lineage.append(
                PublicationLineageEntry(
                    event_id=item.decision_id,
                    timestamp=item.created_at,
                    event_type="research_decision",
                    summary=f"Research decision selected '{item.selected_action}' at confidence {item.confidence}.",
                    source_id=item.problem_id,
                    metadata={"notes": list(item.notes)},
                )
            )
        for item in verifications:
            lineage.append(
                PublicationLineageEntry(
                    event_id=f"verification:{item.trial_id}",
                    timestamp=item.verified_at,
                    event_type="verification",
                    summary=f"Verification report marked trial {item.trial_id} as {item.verdict}.",
                    source_id=item.trial_id,
                    metadata={"recommendations": list(item.recommendations)},
                )
            )
        for item in verdicts:
            lineage.append(
                PublicationLineageEntry(
                    event_id=item.verdict_id,
                    timestamp=item.created_at,
                    event_type="claim_verdict",
                    summary=f"Claim verdict {item.verdict_id} is {item.status} at confidence {item.confidence}.",
                    source_id=item.trial_id,
                    metadata={
                        "canonical_benchmark_satisfied": item.canonical_benchmark_satisfied,
                        "linkage_status": item.linkage_status,
                    },
                )
            )
        for item in plans:
            pending = len([test for test in item.tests if test.status in {"planned", "attached", "running"}])
            lineage.append(
                PublicationLineageEntry(
                    event_id=item.plan_id,
                    timestamp=item.created_at,
                    event_type="falsification_plan",
                    summary=f"Falsification plan {item.plan_id} recorded with {pending} pending tests.",
                    source_id=item.plan_id,
                    metadata={"status": item.status, "pending_tests": pending},
                )
            )
        for item in portfolio_decisions:
            lineage.append(
                PublicationLineageEntry(
                    event_id=item.decision_id,
                    timestamp=item.created_at,
                    event_type="portfolio_decision",
                    summary=f"Portfolio decision selected project {item.selected_project_id or 'none'}.",
                    source_id=item.decision_id,
                    metadata={
                        "deferred_project_ids": list(item.deferred_project_ids),
                        "parked_project_ids": list(item.parked_project_ids),
                    },
                )
            )
        lineage.sort(key=lambda item: (item.timestamp, item.event_id))
        return lineage

    def _publication_evidence_gaps(
        self,
        *,
        project: ResearchProject,
        evidence_debt: Optional[EvidenceDebtRecord],
        plans: list[FalsificationPlan],
        verdicts: list[ClaimVerdict],
    ) -> list[str]:
        gaps: list[str] = []
        if evidence_debt is not None:
            gap_fields = {
                "falsification_gap": "falsification coverage remains incomplete",
                "replication_gap": "replication coverage remains incomplete",
                "benchmark_gap": "benchmark coverage remains incomplete",
                "claim_linkage_gap": "claim linkage/provenance remains incomplete",
                "calibration_gap": "calibration evidence remains incomplete",
            }
            for field_name, message in gap_fields.items():
                if getattr(evidence_debt, field_name) > 0.0 and message not in gaps:
                    gaps.append(message)
        if any(plan.status == "active" for plan in plans):
            gaps.append("an active falsification plan remains open")
        if any(item.linkage_status != "exact" for item in verdicts):
            gaps.append("at least one claim verdict has non-exact linkage")
        thread = self._project_thread(project)
        if thread and thread.contradicting_evidence_ids:
            gaps.append("contradicting evidence remains attached to the active thread")
        return gaps

    def _publication_limitations(
        self,
        *,
        project: ResearchProject,
        executions: list[ProblemExecutionReport],
        evidence_debt: Optional[EvidenceDebtRecord],
        plans: list[FalsificationPlan],
        verdicts: list[ClaimVerdict],
        open_questions: list[str],
    ) -> list[str]:
        limitations = list(open_questions)
        if evidence_debt is not None and evidence_debt.promotion_blocked:
            limitations.append(
                f"Evidence debt remains at {evidence_debt.overall_debt:.2f}; promotion is currently blocked."
            )
        if any(plan.status == "active" for plan in plans):
            limitations.append("A falsification plan is still active, so publication claims should remain bounded.")
        if any(not verdict.canonical_benchmark_satisfied and verdict.canonical_benchmark_required for verdict in verdicts):
            limitations.append("Canonical benchmark comparability is required for at least one claim and is not yet satisfied.")
        if any(not item.reproducibility_complete for item in executions):
            limitations.append("At least one linked execution is not fully reproducible yet.")
        return list(dict.fromkeys(limitations))

    def _publication_writer_cautions(
        self,
        *,
        package_status: str,
        accepted_claims: list[PublicationClaimBundle],
        provisional_claims: list[PublicationClaimBundle],
        limitations: list[str],
        evidence_gaps: list[str],
    ) -> list[str]:
        cautions: list[str] = []
        if package_status != "ready":
            cautions.append("Treat this package as evidence-bounded research support, not publication-ready proof.")
        if provisional_claims:
            cautions.append("Do not elevate provisional claims to accepted findings in downstream writing.")
        if not accepted_claims:
            cautions.append("No accepted claim bundle exists yet; any narrative must stay exploratory.")
        if evidence_gaps:
            cautions.append("Explicitly disclose unresolved evidence gaps and active falsification pressure.")
        if limitations:
            cautions.append("Carry project limitations and open questions into any downstream manuscript draft.")
        return cautions

    def _timeline_event(
        self,
        *,
        timestamp: Optional[str],
        event_type: str,
        summary: str,
        details: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return {
            "timestamp": timestamp,
            "event_type": event_type,
            "summary": summary,
            "details": details or {},
        }

    def _retrieval_mode_summary(self, *, limit: int = 20) -> dict[str, Any]:
        studies = list(self.store.iter_problem_studies())[-max(1, limit):]
        breakdown = {"semantic": 0, "lexical_fallback": 0}
        degraded = 0
        for study in studies:
            mode = study.retrieval_mode
            breakdown[mode] = breakdown.get(mode, 0) + 1
            if study.status == "retrieval_degraded":
                degraded += 1
        return {
            "window": len(studies),
            "retrieval_mode_breakdown": breakdown,
            "degraded_retrieval_studies": degraded,
        }

    def _claim_verdict_lifecycle_summary(self, *, limit: int = 20) -> dict[str, Any]:
        verdicts = list(self.store.iter_claim_verdicts())[-max(1, limit):]
        counts = {"active": 0, "aging": 0, "escalated": 0, "resolved": 0}
        escalated_verdict_ids: list[str] = []
        for verdict in verdicts:
            lifecycle = verdict.lifecycle_status
            counts[lifecycle] = counts.get(lifecycle, 0) + 1
            if lifecycle == "escalated":
                escalated_verdict_ids.append(verdict.verdict_id)
        return {
            "window": len(verdicts),
            "counts": counts,
            "escalated_verdict_ids": escalated_verdict_ids,
        }

    def operator_view(self, *, include_blocked: bool = True, limit: int = 5, mode: str = "balanced") -> dict[str, Any]:
        portfolio, priorities, evidence_debts, stale_records = self._evaluate_portfolio(
            include_blocked=include_blocked,
            mode=mode,
        )
        projects = self.store.list_research_projects()
        top_candidates = self._rank_action_candidates(
            include_blocked=include_blocked,
            limit=limit,
            mode=mode,
            persist=False,
        )
        promotion_blocked = [item.model_dump(mode="json") for item in evidence_debts if item.promotion_blocked][: max(1, limit)]
        retrieval_summary = self._retrieval_mode_summary(limit=20)
        verdict_summary = self._claim_verdict_lifecycle_summary(limit=20)
        frontier_summary = self.frontier_gap_status(limit=limit)
        return {
            "generated_at": self._project_now(),
            "project_counts": {
                "total": len(projects),
                "active": len([item for item in projects if item.status == "active"]),
                "paused": len([item for item in projects if item.status == "paused"]),
                "blocked": len([item for item in projects if item.status == "blocked"]),
                "stale": len([item for item in stale_records if item.staleness_level in {"stale", "critical"}]),
            },
            "frontier_gap_counts": frontier_summary["counts"],
            "frontier_gap_scan_count": frontier_summary["scan_count"],
            "frontier_gaps": frontier_summary["gaps"],
            "latest_frontier_gap_scan": frontier_summary["latest_scan"],
            "portfolio_health": portfolio.health_snapshot.model_dump(mode="json"),
            "active_projects": [self._project_summary(item) for item in projects if item.status == "active"][: max(1, limit)],
            "blocked_projects": [self._project_summary(item) for item in projects if item.status == "blocked"][: max(1, limit)],
            "top_candidates": [item.model_dump(mode="json") for item in top_candidates.candidates],
            "top_projects": [item.model_dump(mode="json") for item in priorities[: max(1, limit)]],
            "stale_projects": [
                item.model_dump(mode="json")
                for item in stale_records
                if item.staleness_level in {"stale", "critical"}
            ][: max(1, limit)],
            "resume_candidates": [
                {
                    "staleness": item.model_dump(mode="json"),
                    "priority": next(
                        (record.model_dump(mode="json") for record in priorities if record.project_id == item.project_id),
                        None,
                    ),
                }
                for item in stale_records
                if item.resume_candidate
            ][: max(1, limit)],
            "promotion_blocked_projects": promotion_blocked,
            "retrieval_mode_breakdown": retrieval_summary["retrieval_mode_breakdown"],
            "degraded_retrieval_studies": retrieval_summary["degraded_retrieval_studies"],
            "recent_study_window": retrieval_summary["window"],
            "claim_verdict_lifecycle": verdict_summary["counts"],
            "recent_verdict_window": verdict_summary["window"],
            "escalated_verdict_ids": verdict_summary["escalated_verdict_ids"],
            "latest_portfolio_decision": self.store.latest_portfolio_decision().model_dump(mode="json")
            if self.store.latest_portfolio_decision()
            else None,
        }

    def project_timeline(self, project_id: str, *, limit: int = 25) -> dict[str, Any]:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        studies = self._project_studies(project_id)
        executions = self._project_executions(project_id)
        decisions = self._project_research_decisions(project_id)
        verdicts = self._project_claim_verdicts(project_id)
        plans = self._project_falsification_plans(project_id)
        portfolio_decisions = self._project_related_portfolio_decisions(project_id)
        priority_records = self._project_priority_records(project_id)
        evidence_debt = [item for item in self.store.iter_evidence_debt_records() if item.project_id == project_id]
        staleness_records = [item for item in self.store.iter_project_staleness_records() if item.project_id == project_id]

        events: list[dict[str, Any]] = [
            self._timeline_event(
                timestamp=project.created_at,
                event_type="project_created",
                summary=f"Project '{project.title}' created.",
                details={"project_id": project.project_id, "status": project.status},
            )
        ]
        for event in self.store.iter_audit_events():
            payload = event.get("payload") or {}
            if not isinstance(payload, dict) or payload.get("project_id") != project_id:
                continue
            if event.get("source") != "research_projects":
                continue
            action = str(event.get("action", "audit"))
            if action == "create_project":
                continue
            events.append(
                self._timeline_event(
                    timestamp=event.get("timestamp"),
                    event_type=f"audit_{action}",
                    summary=f"Project {action.replace('_', ' ')}.",
                    details=payload,
                )
            )
        for study in studies:
            events.append(
                self._timeline_event(
                    timestamp=study.created_at,
                    event_type="problem_study",
                    summary=f"Study planned for {study.problem_id} with {study.benchmark_alignment} benchmark alignment.",
                    details={
                        "problem_id": study.problem_id,
                        "benchmark_ids": study.benchmark_ids,
                        "benchmark_truth_statuses": study.benchmark_truth_statuses,
                        "benchmark_alignment": study.benchmark_alignment,
                        "next_action": study.next_action,
                    },
                )
            )
        for execution in executions:
            events.append(
                self._timeline_event(
                    timestamp=execution.executed_at,
                    event_type="problem_execution",
                    summary=f"Execution {execution.status} via {execution.execution_mode}.",
                    details={
                        "problem_id": execution.problem_id,
                        "execution_mode": execution.execution_mode,
                        "status": execution.status,
                        "benchmark_alignment": execution.benchmark_alignment,
                        "canonical_comparable": execution.canonical_comparable,
                        "summary": execution.summary,
                    },
                )
            )
        for decision in decisions:
            events.append(
                self._timeline_event(
                    timestamp=decision.created_at,
                    event_type="research_decision",
                    summary=f"Decision selected action: {decision.selected_action}",
                    details={
                        "decision_id": decision.decision_id,
                        "problem_id": decision.problem_id,
                        "action_id": decision.action_id,
                        "confidence": decision.confidence,
                    },
                )
            )
        for item in priority_records:
            events.append(
                self._timeline_event(
                    timestamp=item.created_at,
                    event_type="project_priority",
                    summary=f"Project priority record: {item.recommended_state} at score {item.priority_score}.",
                    details=item.model_dump(mode="json"),
                )
            )
        for allocation in self.store.iter_budget_allocations():
            candidate = allocation.selected_candidate
            if candidate is None or candidate.project_id != project_id:
                continue
            events.append(
                self._timeline_event(
                    timestamp=allocation.created_at,
                    event_type="budget_allocation",
                    summary=f"Budget allocation selected {candidate.action_kind}.",
                    details={
                        "decision_id": allocation.decision_id,
                        "action_id": candidate.action_id,
                        "score": candidate.score,
                        "schedule_created": allocation.schedule_created,
                    },
                )
            )
        for plan in plans:
            events.append(
                self._timeline_event(
                    timestamp=plan.created_at,
                    event_type="falsification_plan",
                    summary=f"Falsification plan created with {len(plan.tests)} tests.",
                    details={
                        "plan_id": plan.plan_id,
                        "trigger_reason": plan.trigger_reason,
                        "status": plan.status,
                    },
                )
            )
        for verdict in verdicts:
            events.append(
                self._timeline_event(
                    timestamp=verdict.created_at,
                    event_type="claim_verdict",
                    summary=f"Claim verdict {verdict.status} with linkage {verdict.linkage_status}.",
                    details={
                        "verdict_id": verdict.verdict_id,
                        "benchmark_problem_id": verdict.benchmark_problem_id,
                        "confidence": verdict.confidence,
                        "canonical_comparability_source": verdict.canonical_comparability_source,
                    },
                )
            )
        for record in evidence_debt:
            events.append(
                self._timeline_event(
                    timestamp=record.created_at,
                    event_type="evidence_debt",
                    summary=f"Evidence debt recorded at {record.overall_debt}.",
                    details=record.model_dump(mode="json"),
                )
            )
        for record in staleness_records:
            events.append(
                self._timeline_event(
                    timestamp=record.created_at,
                    event_type="project_staleness",
                    summary=f"Project staleness level is {record.staleness_level}.",
                    details=record.model_dump(mode="json"),
                )
            )
        for decision in portfolio_decisions:
            events.append(
                self._timeline_event(
                    timestamp=decision.created_at,
                    event_type="portfolio_decision",
                    summary="Portfolio decision affected this project.",
                    details=decision.model_dump(mode="json"),
                )
            )
        events.sort(key=lambda item: (item.get("timestamp") or "", item.get("event_type") or ""), reverse=True)
        return {
            "project_id": project.project_id,
            "project_title": project.title,
            "event_count": len(events),
            "events": events[: max(1, limit)],
        }

    def project_evidence_map(self, project_id: str) -> dict[str, Any]:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        thread = self._project_thread(project)
        studies = self._project_studies(project_id)
        executions = self._project_executions(project_id)
        latest_study = studies[-1] if studies else None
        latest_execution = executions[-1] if executions else None
        latest_debt = self.store.latest_evidence_debt_record(project_id=project_id)
        latest_plan = self.store.latest_falsification_plan(project_id=project_id, thread_id=thread.thread_id if thread else None)
        verdicts = self._project_claim_verdicts(project_id)
        contradiction_review = None
        if latest_execution and latest_execution.claim_verdict and latest_execution.claim_verdict.contradiction_review:
            contradiction_review = latest_execution.claim_verdict.contradiction_review.model_dump(mode="json")
        elif latest_study and latest_study.contradiction_review:
            contradiction_review = latest_study.contradiction_review.model_dump(mode="json")
        benchmark_context = latest_execution or latest_study
        cited_research_ids = list(latest_study.cited_research_ids) if latest_study else []
        retrieved_memory_ids = list(latest_study.retrieved_memory_ids) if latest_study else []
        return {
            "project_id": project.project_id,
            "project_title": project.title,
            "latest_evidence_summary": project.resume_snapshot.latest_evidence_summary,
            "supporting_evidence_ids": list(thread.supporting_evidence_ids) if thread else [],
            "contradicting_evidence_ids": list(thread.contradicting_evidence_ids) if thread else [],
            "cited_research_ids": cited_research_ids,
            "retrieved_memory_ids": retrieved_memory_ids,
            "contradiction_review": contradiction_review,
            "benchmark_context": {
                "benchmark_ids": list(benchmark_context.benchmark_ids) if benchmark_context else [],
                "benchmark_names": list(benchmark_context.benchmark_names) if benchmark_context else [],
                "benchmark_truth_statuses": list(benchmark_context.benchmark_truth_statuses) if benchmark_context else [],
                "benchmark_alignment": benchmark_context.benchmark_alignment if benchmark_context else None,
                "canonical_comparable": benchmark_context.canonical_comparable if benchmark_context else False,
            },
            "latest_evidence_debt": latest_debt.model_dump(mode="json") if latest_debt else None,
            "latest_falsification_plan": latest_plan.model_dump(mode="json") if latest_plan else None,
            "claim_verdicts": [
                {
                    "verdict_id": item.verdict_id,
                    "status": item.status,
                    "benchmark_problem_id": item.benchmark_problem_id,
                    "linkage_status": item.linkage_status,
                    "canonical_comparability_source": item.canonical_comparability_source,
                    "confidence": item.confidence,
                }
                for item in verdicts
            ],
            "evidence_counts": {
                "supporting": len(thread.supporting_evidence_ids) if thread else 0,
                "contradicting": len(thread.contradicting_evidence_ids) if thread else 0,
                "cited_research": len(cited_research_ids),
                "retrieved_memory": len(retrieved_memory_ids),
                "claim_verdicts": len(verdicts),
            },
        }

    def claim_lineage(self, project_id: str) -> dict[str, Any]:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        studies = self._project_studies(project_id)
        executions = self._project_executions(project_id)
        verdicts = self._project_claim_verdicts(project_id)
        decisions = self._project_research_decisions(project_id)
        problem_ids = sorted({item.problem_id for item in studies})
        return {
            "project_id": project.project_id,
            "project_title": project.title,
            "problem_ids": problem_ids,
            "studies": [
                {
                    "problem_id": item.problem_id,
                    "created_at": item.created_at,
                    "benchmark_ids": item.benchmark_ids,
                    "benchmark_truth_statuses": item.benchmark_truth_statuses,
                    "benchmark_alignment": item.benchmark_alignment,
                    "canonical_comparable": item.canonical_comparable,
                }
                for item in studies
            ],
            "executions": [
                {
                    "problem_id": item.problem_id,
                    "executed_at": item.executed_at,
                    "status": item.status,
                    "execution_mode": item.execution_mode,
                    "benchmark_alignment": item.benchmark_alignment,
                    "canonical_comparable": item.canonical_comparable,
                    "benchmark_names": item.benchmark_names,
                }
                for item in executions
            ],
            "research_decisions": [
                {
                    "decision_id": item.decision_id,
                    "created_at": item.created_at,
                    "problem_id": item.problem_id,
                    "selected_action": item.selected_action,
                    "confidence": item.confidence,
                }
                for item in decisions
            ],
            "verdicts": [
                {
                    "verdict_id": item.verdict_id,
                    "created_at": item.created_at,
                    "status": item.status,
                    "trial_id": item.trial_id,
                    "benchmark_problem_id": item.benchmark_problem_id,
                    "benchmark_execution_mode": item.benchmark_execution_mode,
                    "supporting_benchmark_names": item.supporting_benchmark_names,
                    "linkage_status": item.linkage_status,
                    "canonical_comparability_source": item.canonical_comparability_source,
                    "confidence": item.confidence,
                }
                for item in verdicts
            ],
        }

    def resume_dashboard(self, project_id: str) -> dict[str, Any]:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        thread = self._project_thread(project)
        action = self._project_action(project)
        latest_staleness = self.store.latest_project_staleness_record(project_id=project_id)
        latest_debt = self.store.latest_evidence_debt_record(project_id=project_id)
        latest_plan = self.store.latest_falsification_plan(project_id=project_id, thread_id=thread.thread_id if thread else None)
        priority = next(
            (
                item
                for item in reversed(list(self.store.iter_project_priority_records()))
                if item.project_id == project_id
            ),
            None,
        )
        blockers = list(project.resume_snapshot.blockers)
        if thread and thread.stop_reason and thread.stop_reason not in blockers:
            blockers.append(thread.stop_reason)
        if project.budget_ledger.budget_exhausted and "budget_exhausted" not in blockers:
            blockers.append("budget_exhausted")
        resume_state = "active"
        if project.status == "blocked":
            resume_state = "blocked"
        elif project.status == "paused":
            if blockers:
                resume_state = "review"
            elif latest_debt and latest_debt.promotion_blocked:
                resume_state = "defer"
            else:
                resume_state = "ready"
        return {
            "project": self.project_status(project_id),
            "resume_state": resume_state,
            "resume_snapshot": project.resume_snapshot.model_dump(mode="json"),
            "blockers": blockers,
            "next_action": action.model_dump(mode="json") if action else None,
            "budget_remaining": self._budget_remaining_summary(project.budget_ledger),
            "latest_staleness": latest_staleness.model_dump(mode="json") if latest_staleness else None,
            "latest_evidence_debt": latest_debt.model_dump(mode="json") if latest_debt else None,
            "latest_priority_record": priority.model_dump(mode="json") if priority else None,
            "latest_falsification_plan": latest_plan.model_dump(mode="json") if latest_plan else None,
        }

    def publication_handoff(self, project_id: str) -> dict[str, Any]:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        studies = self._project_studies(project_id)
        executions = self._project_executions(project_id)
        decisions = self._project_research_decisions(project_id)
        verdicts = self._project_claim_verdicts(project_id)
        verifications = self._project_verification_reports(project_id)
        plans = self._project_falsification_plans(project_id)
        portfolio_decisions = self._project_related_portfolio_decisions(project_id)
        evidence_debt = self.store.latest_evidence_debt_record(project_id=project_id)
        resume = self.resume_dashboard(project_id)

        accepted_claims = [self._publication_claim_bundle(item) for item in verdicts if item.status == "accepted"]
        provisional_claims = [self._publication_claim_bundle(item) for item in verdicts if item.status == "provisional"]
        rejected_alternatives = self._publication_alternatives(project, verdicts, decisions)
        benchmark_truth_attachments = self._publication_benchmark_attachments(studies, executions, verdicts)
        experiment_lineage = self._publication_lineage(
            studies,
            executions,
            decisions,
            verdicts,
            verifications,
            plans,
            portfolio_decisions,
        )
        open_questions = [
            item.question
            for item in project.open_questions
            if item.status != "resolved"
        ]
        evidence_gaps = self._publication_evidence_gaps(
            project=project,
            evidence_debt=evidence_debt,
            plans=plans,
            verdicts=verdicts,
        )
        limitations = self._publication_limitations(
            project=project,
            executions=executions,
            evidence_debt=evidence_debt,
            plans=plans,
            verdicts=verdicts,
            open_questions=open_questions,
        )
        package_status = "not_ready"
        if accepted_claims and not evidence_gaps and not limitations:
            package_status = "ready"
        elif accepted_claims or provisional_claims:
            package_status = "provisional"
        writer_cautions = self._publication_writer_cautions(
            package_status=package_status,
            accepted_claims=accepted_claims,
            provisional_claims=provisional_claims,
            limitations=limitations,
            evidence_gaps=evidence_gaps,
        )
        if accepted_claims and not evidence_gaps and not limitations:
            claim_readiness_summary = "Accepted claim bundles are publication-ready under the current evidence package."
        elif accepted_claims or provisional_claims:
            claim_readiness_summary = "Structured claim bundles exist, but publication claims should remain provisional."
        else:
            claim_readiness_summary = "No accepted claim bundle exists yet; the package is suitable only for bounded research handoff."

        package = PublicationHandoffPackage(
            package_id=self._continuity_id("publication"),
            project_id=project.project_id,
            project_title=project.title,
            domain_profile=project.domain_profile,
            package_status=package_status,
            project_status=project.status,
            latest_evidence_summary=(
                project.resume_snapshot.latest_evidence_summary
                if project.resume_snapshot is not None
                else (project.latest_decision_summary or project.goal)
            ),
            claim_readiness_summary=claim_readiness_summary,
            accepted_claims=accepted_claims,
            provisional_claims=provisional_claims,
            rejected_alternatives=rejected_alternatives,
            experiment_lineage=experiment_lineage,
            benchmark_truth_attachments=benchmark_truth_attachments,
            limitations=limitations,
            open_questions=open_questions,
            evidence_gaps=evidence_gaps,
            writer_cautions=writer_cautions,
        )
        artifact_path = self.store.save_publication_handoff(package)
        package = package.model_copy(update={"artifact_path": str(artifact_path)})
        self.store.append_publication_handoff(package)
        self.store.append_audit_event(
            "publication",
            "publication_handoff",
            package.model_dump(mode="json"),
        )
        return {
            "generated": True,
            "package": package.model_dump(mode="json"),
            "resume_state": resume.get("resume_state"),
        }

    def publication_log(self, count: int = 20) -> dict[str, Any]:
        packages = list(self.store.iter_publication_handoffs())
        rows = sorted(packages, key=lambda item: (item.created_at, item.package_id), reverse=True)
        return {
            "packages": [item.model_dump(mode="json") for item in rows[: max(1, count)]],
        }

    def pause_project(
        self,
        project_id: str,
        *,
        reason: ResearchStopReason = "operator_paused",
        note: Optional[str] = None,
    ) -> ResearchProject:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        thread = self._project_thread(project)
        blockers = [reason]
        if note:
            blockers.append(note)
        updated_project = project.model_copy(
            update={
                "status": "blocked" if reason in {"dependency_missing", "benchmark_unavailable", "runtime_failure"} else "paused",
                "latest_decision_summary": note or f"Project paused because {reason}.",
            }
        )
        if thread is not None:
            updated_thread = thread.model_copy(
                update={
                    "status": "parked",
                    "stop_reason": reason,
                    "updated_at": self._project_now(),
                }
            )
            updated_project = self._replace_project_thread(updated_project, updated_thread)
        updated_project = self._persist_project(updated_project, blockers=blockers)
        self.store.append_audit_event(
            "research_projects",
            "pause_project",
            {"project_id": project_id, "reason": reason, "note": note},
        )
        return updated_project

    def resume_project(
        self,
        project_id: str,
        *,
        reason: ResearchResumeReason = "human_requested_resume",
        note: Optional[str] = None,
    ) -> ResearchProject:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        thread = self._project_thread(project)
        updated_project = project.model_copy(
            update={
                "status": "active",
                "latest_decision_summary": note or f"Project resumed because {reason}.",
            }
        )
        if thread is not None:
            updated_thread = thread.model_copy(
                update={
                    "status": "open",
                    "resume_reason": reason,
                    "updated_at": self._project_now(),
                }
            )
            updated_project = self._replace_project_thread(updated_project, updated_thread)
        if self._project_action(updated_project) is None and thread is not None:
            action = ResearchPlannedAction(
                action_id=self._continuity_id("action"),
                project_id=updated_project.project_id,
                thread_id=thread.thread_id,
                action_kind="custom",
                description="Reassess the project state and choose the next experiment.",
                estimated_cost=0.1,
                expected_evidence_gain=0.25,
            )
            updated_thread = (self._project_thread(updated_project, thread.thread_id) or thread).model_copy(
                update={"next_action_id": action.action_id, "updated_at": self._project_now()}
            )
            updated_project = updated_project.model_copy(update={"planned_actions": [*updated_project.planned_actions, action]})
            updated_project = self._replace_project_thread(updated_project, updated_thread)
        updated_project = self._persist_project(updated_project)
        self.store.append_audit_event(
            "research_projects",
            "resume_project",
            {"project_id": project_id, "reason": reason, "note": note},
        )
        return updated_project

    def next_action(self, project_id: str) -> dict[str, Any]:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        action = self._project_action(project)
        question = self._project_question(project)
        return {
            "project_id": project.project_id,
            "project_status": project.status,
            "current_question": question.model_dump(mode="json") if question else None,
            "next_action": action.model_dump(mode="json") if action else None,
            "budget_remaining": self._budget_remaining_summary(project.budget_ledger),
        }

    def _prioritization_policy(self, *, mode: str = "balanced") -> PrioritizationPolicy:
        if mode == "falsification_first":
            return PrioritizationPolicy(
                mode="falsification_first",
                evidence_gain_weight=0.3,
                falsification_weight=0.3,
                uncertainty_reduction_weight=0.15,
                benchmark_value_weight=0.08,
                replication_value_weight=0.05,
                contradiction_urgency_weight=0.15,
                strategic_priority_weight=0.08,
                dependency_readiness_weight=0.15,
                cost_penalty_weight=0.22,
                budget_pressure_penalty_weight=0.14,
            )
        return PrioritizationPolicy()

    def _prioritization_id(self, prefix: str) -> str:
        return self._continuity_id(prefix)

    def _budget_pressure_penalty_value(self, ledger: ResearchBudgetLedger) -> float:
        return {
            "low": 0.05,
            "medium": 0.35,
            "high": 0.7,
            "exhausted": 1.0,
        }.get(ledger.budget_pressure_level, 0.1)

    def _cost_penalty_value(self, action: ResearchPlannedAction, ledger: ResearchBudgetLedger) -> float:
        remaining_experiments = max(1.0, float(ledger.experiment_budget - ledger.experiments_spent))
        scale = max(0.5, remaining_experiments / 3.0)
        return min(1.0, action.estimated_cost / scale)

    def _dependency_readiness_value(
        self,
        project: ResearchProject,
        thread: Optional[ResearchHypothesisThread],
    ) -> float:
        if project.status == "blocked":
            return 0.0
        if thread is None:
            return 0.25
        if thread.stop_reason in {"dependency_missing", "benchmark_unavailable"}:
            return 0.0
        if thread.stop_reason == "runtime_failure":
            return 0.25
        return 1.0

    def _benchmark_value(
        self,
        project: ResearchProject,
        action: ResearchPlannedAction,
        question: Optional[ResearchOpenQuestion],
    ) -> float:
        score = 0.3
        description = action.description.lower()
        if action.action_kind == "run_problem_study":
            score += 0.2
        if "benchmark" in description or "canonical" in description:
            score += 0.3
        if question is not None and question.uncertainty_type in {"execution_gap", "followup_decision"}:
            score += 0.1
        if project.status == "blocked":
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _falsification_value(
        self,
        thread: Optional[ResearchHypothesisThread],
        action: ResearchPlannedAction,
        question: Optional[ResearchOpenQuestion],
    ) -> float:
        score = 0.1
        description = action.description.lower()
        if action.action_kind == "verify_claim":
            score += 0.7
        if action.action_kind == "review_execution_result":
            score += 0.4
        if any(token in description for token in ("ablation", "falsif", "contradict", "stress", "probe")):
            score += 0.35
        if question is not None and question.blocking:
            score += 0.15
        if thread is not None and thread.confidence_state in {"provisional", "supported"}:
            score += 0.1
        return max(0.0, min(1.0, score))

    def _replication_value(self, action: ResearchPlannedAction) -> float:
        description = action.description.lower()
        if any(token in description for token in ("seed", "replicat", "variance", "stability")):
            return 0.8
        if action.action_kind in {"verify_claim", "review_execution_result"}:
            return 0.4
        return 0.15

    def _contradiction_urgency(self, thread: Optional[ResearchHypothesisThread]) -> float:
        if thread is None:
            return 0.0
        score = min(1.0, len(thread.contradicting_evidence_ids) * 0.4)
        if thread.status == "contradicted" or thread.confidence_state == "contradicted":
            score = max(score, 0.9)
        return score

    @staticmethod
    def _evidence_debt_remediation_actions() -> set[str]:
        return {
            "verify_claim",
            "review_execution_result",
            "mechanism_ablation",
            "replication_check",
            "seed_variance_check",
            "contradiction_resolution",
            "benchmark_stress_probe",
            "calibration_check",
            "environment_reproduction_check",
            "claim_linkage_sanity_check",
        }

    def _evidence_debt_gate(
        self,
        project: ResearchProject,
        action: ResearchPlannedAction,
    ) -> tuple[EvidenceDebtRecord, Optional[str]]:
        evidence_debt = self._compute_evidence_debt(project)
        if not evidence_debt.promotion_blocked:
            return evidence_debt, None
        if action.action_kind in self._evidence_debt_remediation_actions():
            return evidence_debt, None
        return (
            evidence_debt,
            "evidence debt blocks non-remediation scheduling; only falsification, replication, and repair work may proceed",
        )

    def _strategic_priority_value(self, project: ResearchProject) -> float:
        return max(0.0, min(1.0, 0.25 + (project.priority * 0.15)))

    def _build_action_candidate(
        self,
        project: ResearchProject,
        *,
        policy: PrioritizationPolicy,
        action_id: Optional[str] = None,
    ) -> Optional[PrioritizedActionCandidate]:
        thread = self._project_thread(project)
        action = self._project_action(project, action_id=action_id)
        if thread is None or action is None or action.status not in {"planned", "queued"}:
            return None
        evidence_debt, evidence_gate_reason = self._evidence_debt_gate(project, action)
        question = self._project_question(project, question_id=action.depends_on[0] if action.depends_on else None) or self._project_question(project)
        dependency_readiness = self._dependency_readiness_value(project, thread)
        benchmark_value = self._benchmark_value(project, action, question)
        falsification_value = self._falsification_value(thread, action, question)
        replication_value = self._replication_value(action)
        contradiction_urgency = self._contradiction_urgency(thread)
        strategic_priority = self._strategic_priority_value(project)
        uncertainty_reduction = min(
            1.0,
            (question.importance if question is not None else 0.35) + (0.15 if question is not None and question.blocking else 0.0),
        )
        cost_penalty = self._cost_penalty_value(action, project.budget_ledger)
        budget_pressure_penalty = self._budget_pressure_penalty_value(project.budget_ledger)
        total_score = round(
            (
                (action.expected_evidence_gain * policy.evidence_gain_weight)
                + (falsification_value * policy.falsification_weight)
                + (uncertainty_reduction * policy.uncertainty_reduction_weight)
                + (benchmark_value * policy.benchmark_value_weight)
                + (replication_value * policy.replication_value_weight)
                + (contradiction_urgency * policy.contradiction_urgency_weight)
                + (strategic_priority * policy.strategic_priority_weight)
                + (dependency_readiness * policy.dependency_readiness_weight)
                - (cost_penalty * policy.cost_penalty_weight)
                - (budget_pressure_penalty * policy.budget_pressure_penalty_weight)
                - (0.4 if project.status == "blocked" else 0.0)
            ),
            6,
        )
        blocked = (
            dependency_readiness <= 0.0
            or project.status == "blocked"
            or project.budget_ledger.budget_exhausted
            or evidence_gate_reason is not None
        )
        rationale: list[str] = []
        if question is not None and question.blocking:
            rationale.append("blocking question raises uncertainty-reduction value")
        if action.action_kind in {"verify_claim", "review_execution_result"}:
            rationale.append("verification-style action increases falsification pressure")
        if action.estimated_cost <= 0.5:
            rationale.append("low estimated cost improves evidence-per-budget")
        if project.budget_ledger.budget_pressure_level in {"high", "exhausted"}:
            rationale.append("budget pressure penalizes further expensive work")
        if project.status == "blocked":
            rationale.append("project is currently blocked and must be deprioritized")
        if evidence_gate_reason is not None:
            rationale.append(evidence_gate_reason)
        elif evidence_debt.promotion_blocked:
            rationale.append("promotion remains blocked but this action is evidence-debt remediation work")
        return PrioritizedActionCandidate(
            candidate_id=f"{project.project_id}:{action.action_id}",
            project_id=project.project_id,
            thread_id=thread.thread_id,
            action_id=action.action_id,
            project_title=project.title,
            domain_profile=project.domain_profile,
            project_status=project.status,
            action_kind=action.action_kind,
            action_status=action.status,
            action_description=action.description,
            current_question=question.question if question is not None else None,
            budget_pressure_level=project.budget_ledger.budget_pressure_level,
            blocked=blocked,
            score=total_score,
            score_breakdown=ActionScoreBreakdown(
                expected_evidence_gain=action.expected_evidence_gain,
                falsification_value=falsification_value,
                uncertainty_reduction=uncertainty_reduction,
                benchmark_value=benchmark_value,
                replication_value=replication_value,
                contradiction_urgency=contradiction_urgency,
                strategic_priority=strategic_priority,
                dependency_readiness=dependency_readiness,
                cost_penalty=cost_penalty,
                budget_pressure_penalty=budget_pressure_penalty,
                total_score=total_score,
            ),
            rationale=rationale,
        )

    def _rank_action_candidates(
        self,
        *,
        project_id: Optional[str] = None,
        include_blocked: bool = False,
        limit: int = 10,
        mode: str = "balanced",
        persist: bool = False,
    ) -> PortfolioPrioritySnapshot:
        policy = self._prioritization_policy(mode=mode)
        projects = (
            [self.store.get_research_project(project_id)] if project_id else self.store.list_research_projects()
        )
        filtered_projects = [
            item
            for item in projects
            if item is not None and item.status not in {"completed", "abandoned"}
        ]
        candidates: list[PrioritizedActionCandidate] = []
        blocked_project_count = len([item for item in filtered_projects if item.status == "blocked"])
        active_project_count = len([item for item in filtered_projects if item.status == "active"])
        for project in filtered_projects:
            action_ids = [
                item.action_id
                for item in project.planned_actions
                if item.status in {"planned", "queued"}
            ]
            if not action_ids:
                current_action = self._project_action(project)
                if current_action is not None:
                    action_ids = [current_action.action_id]
            for action_id in action_ids:
                candidate = self._build_action_candidate(project, policy=policy, action_id=action_id)
                if candidate is None:
                    continue
                if candidate.blocked and not include_blocked:
                    continue
                candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                -item.score,
                -self.store.get_research_project(item.project_id).priority if self.store.get_research_project(item.project_id) else 0,
                item.project_title,
                item.action_id,
            )
        )
        candidates = candidates[: max(1, limit)]
        if candidates:
            candidates[0] = candidates[0].model_copy(update={"recommended": True})
        snapshot = PortfolioPrioritySnapshot(
            snapshot_id=self._prioritization_id("priority"),
            project_id=project_id,
            policy=policy,
            candidate_count=len(candidates),
            active_project_count=active_project_count,
            blocked_project_count=blocked_project_count,
            selected_project_id=candidates[0].project_id if candidates else None,
            selected_action_id=candidates[0].action_id if candidates else None,
            candidates=candidates,
            notes=[
                "ranked by evidence gain, falsification value, readiness, and cost penalties",
                "blocked or budget-exhausted work is deprioritized unless explicitly requested",
            ],
        )
        if persist:
            self.store.append_priority_snapshot(snapshot)
            self.store.append_audit_event("research_projects", "rank_actions", snapshot.model_dump(mode="json"))
        return snapshot

    def _latest_project_study(self, project_id: str) -> Optional[ProblemStudyReport]:
        rows = list(self.store.iter_problem_studies())
        for report in reversed(rows):
            if report.project_id == project_id:
                return report
        return None

    def _has_active_schedule_for_action(self, action_id: str) -> bool:
        for entry in self.store.iter_problem_schedules():
            if entry.action_id == action_id and entry.status in {"scheduled", "leased", "running", "retry_wait"}:
                return True
        return False

    def _priority_to_queue_value(self, score: float) -> int:
        return max(0, min(1000, int(round(max(0.0, score) * 1000))))

    def _reprioritize_scheduled_jobs(self) -> list[ProblemScheduleEntry]:
        snapshot = self._rank_action_candidates(include_blocked=True, limit=100, persist=False)
        candidate_map = {(item.project_id, item.action_id): item for item in snapshot.candidates}
        updated_entries: list[ProblemScheduleEntry] = []
        for entry in self.store.iter_problem_schedules():
            if entry.status not in {"scheduled", "retry_wait"} or not entry.project_id or not entry.action_id:
                continue
            candidate = candidate_map.get((entry.project_id, entry.action_id))
            if candidate is None:
                continue
            priority = self._priority_to_queue_value(candidate.score)
            if entry.priority == priority and entry.priority_score == candidate.score and entry.priority_source == "ws18_ranked_action":
                continue
            updated = self.store.update_problem_schedule(
                entry.schedule_id,
                priority=priority,
                priority_score=candidate.score,
                priority_source="ws18_ranked_action",
            )
            if updated is not None:
                updated_entries.append(updated)
        return updated_entries

    def portfolio_status(self, *, include_blocked: bool = True, limit: int = 5, mode: str = "balanced") -> dict[str, Any]:
        snapshot = self._rank_action_candidates(
            include_blocked=include_blocked,
            limit=limit,
            mode=mode,
            persist=False,
        )
        return {
            "project_counts": {
                "total": len(self.store.list_research_projects()),
                "active": len([item for item in self.store.list_research_projects() if item.status == "active"]),
                "blocked": len([item for item in self.store.list_research_projects() if item.status == "blocked"]),
                "paused": len([item for item in self.store.list_research_projects() if item.status == "paused"]),
            },
            "top_candidates": [item.model_dump(mode="json") for item in snapshot.candidates],
            "policy": snapshot.policy.model_dump(mode="json"),
            "latest_priority_snapshot": self.store.latest_priority_snapshot().model_dump(mode="json") if self.store.latest_priority_snapshot() else None,
            "latest_budget_allocation": self.store.latest_budget_allocation().model_dump(mode="json") if self.store.latest_budget_allocation() else None,
            "latest_portfolio": self.store.load_research_portfolio().model_dump(mode="json"),
            "latest_portfolio_decision": self.store.latest_portfolio_decision().model_dump(mode="json") if self.store.latest_portfolio_decision() else None,
        }

    def rank_actions(
        self,
        *,
        project_id: Optional[str] = None,
        include_blocked: bool = False,
        limit: int = 10,
        mode: str = "balanced",
    ) -> dict[str, Any]:
        snapshot = self._rank_action_candidates(
            project_id=project_id,
            include_blocked=include_blocked,
            limit=limit,
            mode=mode,
            persist=True,
        )
        return snapshot.model_dump(mode="json")

    def prioritization_log(self, count: int = 20) -> dict[str, Any]:
        rows = list(self.store.iter_priority_snapshots())[-count:]
        return {"snapshots": [item.model_dump(mode="json") for item in rows]}

    def allocate_budget(
        self,
        *,
        project_id: Optional[str] = None,
        include_blocked: bool = False,
        limit: int = 10,
        mode: str = "balanced",
        schedule_selected: bool = False,
    ) -> dict[str, Any]:
        snapshot = self._rank_action_candidates(
            project_id=project_id,
            include_blocked=include_blocked,
            limit=limit,
            mode=mode,
            persist=True,
        )
        selected = snapshot.candidates[0] if snapshot.candidates else None
        scheduled_job_id: Optional[str] = None
        schedule_created = False
        rationale = ["selected highest-ranked candidate under the current prioritization policy"]
        if selected is not None and selected.blocked:
            rationale.append("selected candidate remains blocked and was not scheduled automatically")
        if selected is not None and schedule_selected and not selected.blocked and selected.action_kind == "run_problem_study":
            study = self._latest_project_study(selected.project_id)
            if study is not None and not self._has_active_schedule_for_action(selected.action_id):
                entry = self.schedule_problem_study(
                    problem_id=study.problem_id,
                    priority=self._priority_to_queue_value(selected.score),
                )
                scheduled_job_id = entry.schedule_id
                schedule_created = True
                rationale.append("created a scheduled study for the selected run_problem_study action")
            else:
                rationale.append("selected action was already queued or had no resolvable study to schedule")
        decision = BudgetAllocationDecision(
            decision_id=self._prioritization_id("budget"),
            policy=snapshot.policy,
            selected_candidate=selected,
            scheduled_job_id=scheduled_job_id,
            schedule_created=schedule_created,
            rationale=rationale,
            considered_candidates=snapshot.candidates,
        )
        self.store.append_budget_allocation(decision)
        self.store.append_audit_event("research_projects", "allocate_budget", decision.model_dump(mode="json"))
        return decision.model_dump(mode="json")

    def _latest_project_execution(self, project_id: str) -> Optional[ProblemExecutionReport]:
        rows = list(self.store.iter_problem_executions())
        for report in reversed(rows):
            if report.project_id == project_id:
                return report
        return None

    def _related_project_claim_verdict(
        self,
        *,
        project_id: str,
        problem_id: Optional[str],
    ) -> Optional[ClaimVerdict]:
        rows = list(self.store.iter_claim_verdicts())
        for verdict in reversed(rows):
            if problem_id is not None and verdict.benchmark_problem_id == problem_id:
                return verdict
        return None

    def _dedupe_falsification_triggers(self, triggers: list[FalsificationTrigger]) -> list[FalsificationTrigger]:
        deduped: list[FalsificationTrigger] = []
        seen: set[tuple[str, str]] = set()
        for trigger in triggers:
            key = (trigger.trigger_type, trigger.reason)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(trigger)
        return deduped

    def _falsification_plan_id(self) -> str:
        return self._continuity_id("falsification")

    def _falsification_test_id(self) -> str:
        return self._continuity_id("falsify-test")

    def _falsification_coverage(self, tests: list[FalsificationTest]) -> FalsificationCoverage:
        kinds = {test.kind for test in tests}
        ablation = 1.0 if "mechanism_ablation" in kinds else 0.0
        replication = 1.0 if {"replication_check", "seed_variance_check"} & kinds else 0.0
        contradiction = 1.0 if "contradiction_resolution" in kinds else 0.0
        benchmark = 1.0 if "benchmark_stress_probe" in kinds else 0.0
        calibration = 1.0 if "calibration_check" in kinds else 0.0
        overall = ((ablation + replication + contradiction + benchmark + calibration) / 5.0) >= 0.45
        return FalsificationCoverage(
            ablation_coverage=ablation,
            replication_coverage=replication,
            contradiction_coverage=contradiction,
            benchmark_pressure_coverage=benchmark,
            calibration_coverage=calibration,
            overall_sufficient=overall,
        )

    def _falsification_tests_from_triggers(
        self,
        *,
        plan_id: str,
        project: ResearchProject,
        thread: ResearchHypothesisThread,
        triggers: list[FalsificationTrigger],
    ) -> list[FalsificationTest]:
        tests_by_kind: dict[str, FalsificationTest] = {}
        for trigger in triggers:
            kind = {
                "confidence_rising": "mechanism_ablation",
                "contradiction_pressure": "contradiction_resolution",
                "low_replication": "seed_variance_check" if "seed variance" in trigger.reason.lower() else "replication_check",
                "benchmark_pressure": "benchmark_stress_probe",
                "calibration_weakness": "calibration_check",
                "claim_linkage_gap": "claim_linkage_sanity_check",
                "environment_reproduction_risk": "environment_reproduction_check",
            }.get(trigger.trigger_type, "mechanism_ablation")
            description = {
                "mechanism_ablation": "Ablate the claimed mechanism and confirm the effect degrades materially.",
                "contradiction_resolution": "Run a targeted contradiction-resolution check against conflicting evidence.",
                "replication_check": "Replicate the strongest result under an additional controlled rerun.",
                "seed_variance_check": "Run a seed-variance sweep to test whether the signal is stable.",
                "benchmark_stress_probe": "Push the result through a stronger benchmark or alignment stress probe.",
                "calibration_check": "Re-run calibration-focused evaluation before any promotion step.",
                "claim_linkage_sanity_check": "Audit claim linkage, provenance completeness, and evidence binding.",
                "environment_reproduction_check": "Repeat the environment reproduction path to rule out runtime-specific artifacts.",
            }[kind]
            estimated_cost = {
                "mechanism_ablation": 0.35,
                "contradiction_resolution": 0.25,
                "replication_check": 0.45,
                "seed_variance_check": 0.5,
                "benchmark_stress_probe": 0.6,
                "calibration_check": 0.25,
                "claim_linkage_sanity_check": 0.15,
                "environment_reproduction_check": 0.4,
            }[kind]
            falsification_value = {
                "mechanism_ablation": 0.8,
                "contradiction_resolution": 0.9,
                "replication_check": 0.75,
                "seed_variance_check": 0.8,
                "benchmark_stress_probe": 0.7,
                "calibration_check": 0.65,
                "claim_linkage_sanity_check": 0.85,
                "environment_reproduction_check": 0.55,
            }[kind]
            if kind in tests_by_kind:
                continue
            tests_by_kind[kind] = FalsificationTest(
                test_id=self._falsification_test_id(),
                plan_id=plan_id,
                project_id=project.project_id,
                thread_id=thread.thread_id,
                kind=kind,  # type: ignore[arg-type]
                description=description,
                estimated_cost=estimated_cost,
                expected_falsification_value=falsification_value,
                depends_on=[],
                status="planned",
            )
        return list(tests_by_kind.values())

    def generate_falsification_plan(
        self,
        project_id: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        project = self.store.get_research_project(project_id)
        if project is None:
            raise RuntimeError(f"Unknown project: {project_id}")
        if project.status in {"completed", "abandoned"} and not force:
            raise RuntimeError(f"Project {project_id} is {project.status} and does not accept new falsification plans.")
        thread = self._project_thread(project)
        if thread is None:
            raise RuntimeError(f"Project {project_id} has no active hypothesis thread.")

        existing = self.store.latest_falsification_plan(project_id=project_id, thread_id=thread.thread_id)
        if (
            existing is not None
            and existing.status == "active"
            and any(test.status in {"planned", "attached", "running"} for test in existing.tests)
            and not force
        ):
            return self.falsification_status(project_id)

        latest_study = self._latest_project_study(project_id)
        latest_execution = self._latest_project_execution(project_id)
        problem_id = (
            latest_execution.problem_id
            if latest_execution is not None
            else (latest_study.problem_id if latest_study is not None else None)
        )
        claim_verdict = self._related_project_claim_verdict(project_id=project_id, problem_id=problem_id)
        verification = (
            self.store.latest_verification_report(claim_verdict.trial_id)
            if claim_verdict is not None
            else None
        )
        contradiction_review = (
            claim_verdict.contradiction_review
            if claim_verdict is not None and claim_verdict.contradiction_review is not None
            else (latest_study.contradiction_review if latest_study is not None else None)
        )
        benchmark_alignment = (
            latest_execution.benchmark_alignment
            if latest_execution is not None
            else (latest_study.benchmark_alignment if latest_study is not None else "aligned")
        )
        canonical_comparable = (
            latest_execution.canonical_comparable
            if latest_execution is not None
            else (latest_study.canonical_comparable if latest_study is not None else True)
        )

        triggers: list[FalsificationTrigger] = []
        if thread.confidence_state in {"provisional", "supported"} or thread.status in {"supported", "contradicted"}:
            triggers.append(
                FalsificationTrigger(
                    trigger_type="confidence_rising",
                    reason="thread confidence is rising and now requires adversarial pressure",
                    severity="high" if thread.confidence_state == "supported" else "medium",
                    evidence_refs=[thread.thread_id],
                )
            )
        if thread.contradicting_evidence_ids:
            triggers.append(
                FalsificationTrigger(
                    trigger_type="contradiction_pressure",
                    reason="project thread already carries contradictory evidence that needs direct resolution",
                    severity="high",
                    evidence_refs=thread.contradicting_evidence_ids,
                )
            )
        if project.budget_ledger.replications_spent < 1 and thread.confidence_state in {"provisional", "supported"}:
            triggers.append(
                FalsificationTrigger(
                    trigger_type="low_replication",
                    reason="confidence is rising before sufficient replication pressure has been spent",
                    severity="medium",
                    evidence_refs=[project.project_id],
                )
            )
        if benchmark_alignment != "aligned" or not canonical_comparable:
            triggers.append(
                FalsificationTrigger(
                    trigger_type="benchmark_pressure",
                    reason="benchmark alignment or comparability is still weak",
                    severity="medium",
                    evidence_refs=[problem_id] if problem_id else [project.project_id],
                )
            )
        if project.status == "blocked" and thread.stop_reason == "runtime_failure":
            triggers.append(
                FalsificationTrigger(
                    trigger_type="environment_reproduction_risk",
                    reason="runtime failure suggests environment reproduction should be checked before trusting the result",
                    severity="medium",
                    evidence_refs=[project.project_id],
                )
            )
        if verification is not None:
            triggers.extend(
                self.verification_runner.suggest_falsification_triggers(
                    verification,
                    contradiction_review=contradiction_review,
                    claim_verdict=claim_verdict,
                    canonical_comparable=canonical_comparable,
                )
            )
        triggers = self._dedupe_falsification_triggers(triggers)
        if not triggers and not force:
            return {
                "generated": False,
                "project_id": project_id,
                "reason": "no_active_falsification_pressure",
                "plan": None,
            }
        if not triggers and force:
            triggers = [
                FalsificationTrigger(
                    trigger_type="confidence_rising",
                    reason="operator requested an explicit falsification plan",
                    severity="low",
                    evidence_refs=[project.project_id],
                )
            ]

        plan_id = self._falsification_plan_id()
        tests = self._falsification_tests_from_triggers(
            plan_id=plan_id,
            project=project,
            thread=thread,
            triggers=triggers,
        )
        question = ResearchOpenQuestion(
            question_id=self._continuity_id("question"),
            project_id=project.project_id,
            thread_id=thread.thread_id,
            question="Which falsification test is most likely to break the current explanation?",
            importance=0.85,
            uncertainty_type="falsification_pressure",
            blocking=True,
            status="open",
        )
        attached_actions: list[ResearchPlannedAction] = []
        linked_tests: list[FalsificationTest] = []
        for test in tests:
            action = ResearchPlannedAction(
                action_id=self._continuity_id("action"),
                project_id=project.project_id,
                thread_id=thread.thread_id,
                action_kind=test.kind,  # type: ignore[arg-type]
                description=f"Falsification: {test.description}",
                estimated_cost=test.estimated_cost,
                expected_evidence_gain=test.expected_falsification_value,
                depends_on=[question.question_id],
                status="planned",
                falsification_plan_id=plan_id,
                falsification_test_id=test.test_id,
            )
            attached_actions.append(action)
            linked_tests.append(
                test.model_copy(
                    update={
                        "depends_on": [question.question_id],
                        "status": "attached",
                        "linked_action_id": action.action_id,
                        "updated_at": self._project_now(),
                    }
                )
            )

        coverage = self._falsification_coverage(linked_tests)
        plan = FalsificationPlan(
            plan_id=plan_id,
            project_id=project.project_id,
            thread_id=thread.thread_id,
            status="active",
            trigger_reason=triggers[0].reason,
            triggers=triggers,
            tests=linked_tests,
            coverage=coverage,
            notes=[
                "generated from project continuity, contradiction pressure, and verification signals",
                "tests are attached back into the project as planned actions for WS18 ranking",
            ],
        )
        updated_thread = thread.model_copy(
            update={
                "status": "falsifying",
                "open_question_ids": [*thread.open_question_ids, question.question_id],
                "next_action_id": attached_actions[0].action_id if attached_actions else thread.next_action_id,
                "updated_at": self._project_now(),
            }
        )
        updated_project = project.model_copy(
            update={
                "latest_decision_summary": f"Active falsification plan {plan_id} generated with {len(linked_tests)} tests.",
                "open_questions": [*project.open_questions, question],
                "planned_actions": [*project.planned_actions, *attached_actions],
            }
        )
        updated_project = self._replace_project_thread(updated_project, updated_thread)
        updated_project = self._persist_project(
            updated_project,
            latest_evidence_summary=updated_project.latest_decision_summary,
        )
        self.store.append_falsification_plan(plan)
        self.store.append_audit_event("research_projects", "generate_falsification_plan", plan.model_dump(mode="json"))
        return {
            "generated": True,
            "project": updated_project.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "attached_actions": [item.model_dump(mode="json") for item in attached_actions],
        }

    def falsification_status(self, project_id: Optional[str] = None) -> dict[str, Any]:
        project = (
            self.store.get_research_project(project_id)
            if project_id is not None
            else self.store.latest_research_project()
        )
        if project is None:
            return {"project": None, "plan": None, "pending_tests": [], "coverage": None}
        thread = self._project_thread(project)
        plan = self.store.latest_falsification_plan(
            project_id=project.project_id,
            thread_id=thread.thread_id if thread is not None else None,
        )
        pending = []
        if plan is not None:
            pending = [
                test.model_dump(mode="json")
                for test in plan.tests
                if test.status in {"planned", "attached", "running"}
            ]
        return {
            "project": self.project_status(project.project_id),
            "plan": plan.model_dump(mode="json") if plan is not None else None,
            "pending_tests": pending,
            "coverage": plan.coverage.model_dump(mode="json") if plan is not None else None,
        }

    def falsification_log(self, count: int = 20) -> dict[str, Any]:
        rows = list(self.store.iter_falsification_plans())[-count:]
        return {"plans": [item.model_dump(mode="json") for item in rows]}

    def _parse_timestamp(self, raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        text = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _project_last_progress_at(self, project: ResearchProject) -> Optional[str]:
        return project.updated_at or project.created_at

    def _project_for_problem(self, problem_id: Optional[str]) -> Optional[ResearchProject]:
        if not problem_id:
            return None
        for study in reversed(list(self.store.iter_problem_studies())):
            if study.problem_id == problem_id and study.project_id:
                return self.store.get_research_project(study.project_id)
        return None

    def _ensure_verdict_followup_question(
        self,
        project: ResearchProject,
        verdict: ClaimVerdict,
    ) -> ResearchProject:
        thread = self._project_thread(project)
        if thread is None:
            return project
        existing = next(
            (
                item
                for item in project.open_questions
                if item.uncertainty_type == "unresolved_verdict" and verdict.verdict_id in item.question
            ),
            None,
        )
        if existing is not None:
            return project
        question = ResearchOpenQuestion(
            question_id=self._continuity_id("question"),
            project_id=project.project_id,
            thread_id=thread.thread_id,
            question=(
                f"Resolve aged claim verdict {verdict.verdict_id} for trial {verdict.trial_id} "
                "before further promotion or autonomous scheduling."
            ),
            importance=max(0.7, verdict.confidence or 0.0),
            uncertainty_type="unresolved_verdict",
            blocking=True,
            status="open",
        )
        updated_thread = thread.model_copy(
            update={
                "open_question_ids": [*thread.open_question_ids, question.question_id],
                "updated_at": self._project_now(),
            }
        )
        updated_project = project.model_copy(update={"open_questions": [*project.open_questions, question]})
        updated_project = self._replace_project_thread(updated_project, updated_thread)
        return self._persist_project(
            updated_project,
            latest_evidence_summary=project.latest_decision_summary,
            blockers=["aged_claim_verdict"],
        )

    def _age_claim_verdicts(self) -> list[str]:
        policy = self.store.load_runtime_policy()
        threshold_days = max(1, policy.verdict_aging_days)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        escalated: list[str] = []
        for verdict in list(self.store.iter_claim_verdicts()):
            updates: dict[str, Any] = {}
            created_at = self._parse_timestamp(verdict.created_at)
            if created_at is None:
                continue
            if verdict.status in {"accepted", "rejected", "contradicted"}:
                if verdict.lifecycle_status != "resolved":
                    updates["lifecycle_status"] = "resolved"
                if updates:
                    self.store.upsert_claim_verdict(verdict.model_copy(update=updates))
                continue
            deadline_dt = self._parse_timestamp(verdict.review_required_before) if verdict.review_required_before else None
            if deadline_dt is None:
                deadline_dt = created_at.astimezone(timezone.utc) + timedelta(days=threshold_days)
                updates["review_required_before"] = deadline_dt.replace(microsecond=0).isoformat()
            if now >= deadline_dt.astimezone(timezone.utc):
                if verdict.lifecycle_status != "escalated":
                    updates.update(
                        {
                            "lifecycle_status": "escalated",
                            "escalated_at": now.isoformat(),
                            "escalation_reason": "verdict_timeout",
                        }
                    )
                    updated_verdict = verdict.model_copy(update=updates)
                    self.store.upsert_claim_verdict(updated_verdict)
                    project = self._project_for_problem(updated_verdict.benchmark_problem_id)
                    if project is not None:
                        self._ensure_verdict_followup_question(project, updated_verdict)
                    self.store.append_audit_event(
                        "claim_verdict",
                        "escalate",
                        {
                            "verdict_id": updated_verdict.verdict_id,
                            "trial_id": updated_verdict.trial_id,
                            "benchmark_problem_id": updated_verdict.benchmark_problem_id,
                            "review_required_before": updated_verdict.review_required_before,
                        },
                    )
                    escalated.append(updated_verdict.verdict_id)
                continue
            if verdict.lifecycle_status != "aging":
                updates["lifecycle_status"] = "aging"
            if updates:
                self.store.upsert_claim_verdict(verdict.model_copy(update=updates))
        return escalated

    def _project_benchmark_readiness(self, project: ResearchProject) -> float:
        latest_execution = self._latest_project_execution(project.project_id)
        if latest_execution is not None:
            if latest_execution.benchmark_alignment == "aligned" and latest_execution.canonical_comparable:
                return 1.0
            if latest_execution.benchmark_alignment == "aligned":
                return 0.7
            if latest_execution.benchmark_alignment == "downgraded":
                return 0.35
            return 0.0
        latest_study = self._latest_project_study(project.project_id)
        if latest_study is not None:
            if latest_study.benchmark_alignment == "aligned" and latest_study.canonical_comparable:
                return 0.9
            if latest_study.benchmark_alignment == "aligned":
                return 0.65
            if latest_study.benchmark_alignment == "downgraded":
                return 0.3
            return 0.0
        thread = self._project_thread(project)
        if thread is not None and thread.stop_reason == "benchmark_unavailable":
            return 0.0
        return 0.5

    def _retrieval_conflict_hits(self, memory_hits: list[MemorySearchHit]) -> list[MemorySearchHit]:
        conflict_hits: list[MemorySearchHit] = []
        for hit in memory_hits:
            metadata = hit.metadata or {}
            kind = str(metadata.get("kind", ""))
            contradiction_count = 0
            for key in ("contradiction_count", "contradiction_pair_count", "conflict_count"):
                value = metadata.get(key)
                try:
                    contradiction_count = max(contradiction_count, int(value))
                except (TypeError, ValueError):
                    continue
            if kind == "claim_conflict":
                conflict_hits.append(hit)
                continue
            if kind == "claim_cluster" and contradiction_count > 0:
                conflict_hits.append(hit)
                continue
            contradictory_claims = metadata.get("contradictory_claims")
            if isinstance(contradictory_claims, list) and contradictory_claims:
                conflict_hits.append(hit)
        return conflict_hits

    def _update_evidence_debt_from_retrieval(
        self,
        project_id: str,
        conflict_hits: list[MemorySearchHit],
    ) -> Optional[EvidenceDebtRecord]:
        if not conflict_hits:
            return None
        project = self.store.get_research_project(project_id)
        if project is None:
            return None
        record = self._compute_evidence_debt(project)
        self.store.append_evidence_debt_record(record)
        self.store.append_audit_event(
            "research_evidence",
            "retrieval_conflict_pressure",
            {
                "project_id": project_id,
                "conflict_hits": len(conflict_hits),
                "falsification_gap": record.falsification_gap,
                "overall_debt": record.overall_debt,
                "promotion_blocked": record.promotion_blocked,
            },
        )
        return record

    def _compute_evidence_debt(self, project: ResearchProject) -> EvidenceDebtRecord:
        thread = self._project_thread(project)
        latest_study = self._latest_project_study(project.project_id)
        latest_execution = self._latest_project_execution(project.project_id)
        problem_id = (
            latest_execution.problem_id
            if latest_execution is not None
            else (latest_study.problem_id if latest_study is not None else None)
        )
        claim_verdict = self._related_project_claim_verdict(project_id=project.project_id, problem_id=problem_id)
        verification = (
            self.store.latest_verification_report(claim_verdict.trial_id)
            if claim_verdict is not None
            else None
        )
        latest_plan = self.store.latest_falsification_plan(
            project_id=project.project_id,
            thread_id=thread.thread_id if thread is not None else None,
        )

        rationale: list[str] = []
        falsification_gap = 0.0
        if thread is not None and thread.confidence_state in {"provisional", "supported"}:
            if latest_plan is None:
                falsification_gap = 1.0
                rationale.append("confidence is rising without an attached falsification plan")
            elif not latest_plan.coverage.overall_sufficient:
                falsification_gap = 0.6
                rationale.append("falsification coverage remains incomplete")
            else:
                falsification_gap = 0.1
        retrieval_conflict_count = latest_study.retrieval_conflict_count if latest_study is not None else 0
        if retrieval_conflict_count > 0:
            falsification_gap = min(1.0, falsification_gap + min(0.5, 0.1 * retrieval_conflict_count))
            rationale.append(
                f"retrieval surfaced {retrieval_conflict_count} contradiction-bearing evidence hit(s); falsification pressure increased"
            )

        replication_gap = 0.0
        if thread is not None and thread.confidence_state in {"provisional", "supported"}:
            if project.budget_ledger.replications_spent < 1:
                replication_gap = 1.0
                rationale.append("replication pressure has not yet been spent")
            elif project.budget_ledger.replications_spent < 2:
                replication_gap = 0.45

        benchmark_gap = 0.0
        if latest_execution is not None:
            if latest_execution.benchmark_alignment != "aligned" or not latest_execution.canonical_comparable:
                benchmark_gap = 0.85
                rationale.append("execution benchmark alignment is still weak")
        elif latest_study is not None:
            if latest_study.benchmark_alignment != "aligned" or not latest_study.canonical_comparable:
                benchmark_gap = 0.75
                rationale.append("study benchmark alignment is still weak")
        elif thread is not None and thread.confidence_state in {"provisional", "supported"}:
            benchmark_gap = 0.35

        claim_linkage_gap = 0.0
        if claim_verdict is None:
            if thread is not None and thread.confidence_state in {"provisional", "supported"}:
                claim_linkage_gap = 0.3
                rationale.append("no claim verdict has yet bound the current signal")
        elif claim_verdict.linkage_status == "exact":
            claim_linkage_gap = 0.0
        elif claim_verdict.linkage_status == "ambiguous":
            claim_linkage_gap = 1.0
            rationale.append("claim linkage is ambiguous")
        else:
            claim_linkage_gap = 0.75
            rationale.append("claim linkage is incomplete")

        calibration_gap = 0.0
        if verification is not None:
            if verification.calibration.ece > 0.15:
                calibration_gap = 0.8
                rationale.append("calibration remains outside the acceptable envelope")
        elif thread is not None and thread.confidence_state in {"provisional", "supported"}:
            calibration_gap = 0.3

        overall_debt = round(
            min(
                1.0,
                (
                    falsification_gap
                    + replication_gap
                    + benchmark_gap
                    + claim_linkage_gap
                    + calibration_gap
                )
                / 5.0,
            ),
            6,
        )
        promotion_blocked = overall_debt >= 0.45 or any(
            gap >= 0.8
            for gap in (
                falsification_gap,
                replication_gap,
                benchmark_gap,
                claim_linkage_gap,
                calibration_gap,
            )
        )
        if promotion_blocked:
            rationale.append("promotion should remain blocked until evidence debt is reduced")
        return EvidenceDebtRecord(
            record_id=self._continuity_id("debt"),
            project_id=project.project_id,
            falsification_gap=falsification_gap,
            replication_gap=replication_gap,
            benchmark_gap=benchmark_gap,
            claim_linkage_gap=claim_linkage_gap,
            calibration_gap=calibration_gap,
            overall_debt=overall_debt,
            promotion_blocked=promotion_blocked,
            rationale=rationale,
        )

    def _compute_project_staleness(self, project: ResearchProject) -> ProjectStalenessRecord:
        thread = self._project_thread(project)
        last_progress_at = self._project_last_progress_at(project)
        last_progress = self._parse_timestamp(last_progress_at)
        hours_since_progress = 0.0
        if last_progress is not None:
            hours_since_progress = max(
                0.0,
                (datetime.now(timezone.utc) - last_progress.astimezone(timezone.utc)).total_seconds() / 3600.0,
            )
        if project.status in {"completed", "abandoned"}:
            level = "fresh"
        elif hours_since_progress >= 72.0:
            level = "critical"
        elif hours_since_progress >= 24.0:
            level = "stale"
        elif hours_since_progress >= 8.0:
            level = "watch"
        else:
            level = "fresh"

        verdicts = self._project_claim_verdicts(project.project_id)
        if any(item.lifecycle_status == "escalated" for item in verdicts):
            level = "critical"
        elif level == "fresh" and any(item.lifecycle_status == "aging" for item in verdicts):
            level = "watch"

        resume_candidate = (
            project.status in {"paused", "blocked"}
            and level in {"stale", "critical"}
            and not project.budget_ledger.budget_exhausted
            and (thread is None or thread.stop_reason not in {"dependency_missing", "benchmark_unavailable"})
        )
        closure_candidate = (
            project.status in {"paused", "blocked"}
            and level == "critical"
            and (
                project.budget_ledger.budget_exhausted
                or thread is None
                or thread.stop_reason in {"goal_completed", "superseded_by_better_thread"}
            )
        )

        reason = "project remains fresh"
        if level == "watch":
            reason = "project has gone quiet and should be monitored"
        elif level == "stale":
            reason = "project has gone stale and should be reviewed for resume or deferral"
        elif level == "critical":
            reason = "project has been inactive for too long and needs explicit action"
        if any(item.lifecycle_status == "escalated" for item in verdicts):
            reason = "project has an escalated unresolved claim verdict that requires review"
        elif any(item.lifecycle_status == "aging" for item in verdicts) and level == "watch":
            reason = "project has aging unresolved claim verdicts that should be reviewed soon"
        if resume_candidate:
            reason = "project is stale but resume-worthy under the current budget and dependency state"
        if closure_candidate:
            reason = "project has gone critically stale and is a closure candidate"

        return ProjectStalenessRecord(
            record_id=self._continuity_id("staleness"),
            project_id=project.project_id,
            last_progress_at=last_progress_at,
            hours_since_progress=round(hours_since_progress, 3),
            staleness_level=level,  # type: ignore[arg-type]
            reason=reason,
            resume_candidate=resume_candidate,
            closure_candidate=closure_candidate,
        )

    def _recommend_portfolio_state(
        self,
        *,
        project: ResearchProject,
        thread: Optional[ResearchHypothesisThread],
        candidate: Optional[PrioritizedActionCandidate],
        evidence_debt: EvidenceDebtRecord,
        staleness: ProjectStalenessRecord,
        priority_score: float,
    ) -> tuple[str, list[str]]:
        rationale: list[str] = []
        falsification_actions = self._evidence_debt_remediation_actions()
        if project.status == "completed":
            return "complete", ["project goal is already complete"]
        if project.status == "abandoned":
            return "retire", ["project has already been abandoned"]
        if thread is not None and thread.stop_reason in {"dependency_missing", "benchmark_unavailable"}:
            return "block", [f"project is blocked by {thread.stop_reason}"]
        if evidence_debt.promotion_blocked:
            if candidate is None:
                return "defer", ["promotion is blocked by evidence debt and no remediation action is currently ready"]
            if candidate.action_kind not in self._evidence_debt_remediation_actions():
                if candidate.score_breakdown.contradiction_urgency >= 0.7:
                    rationale.append("contradiction pressure is high and promotion is blocked by evidence debt")
                    return "escalate", rationale
                return "defer", ["promotion is blocked by evidence debt until remediation work is selected"]
        if staleness.resume_candidate:
            rationale.append("project is stale but resume-worthy")
            return "resume", rationale
        if staleness.closure_candidate:
            rationale.append("project is critically stale and should be parked")
            return "park", rationale
        if project.budget_ledger.budget_exhausted:
            rationale.append("budget is exhausted")
            return "park", rationale
        if (
            evidence_debt.promotion_blocked
            and candidate is not None
            and candidate.action_kind not in falsification_actions
            and candidate.score_breakdown.contradiction_urgency >= 0.7
        ):
            rationale.append("contradiction pressure is high and promotion is blocked by evidence debt")
            return "escalate", rationale
        if project.status in {"paused", "blocked"}:
            rationale.append("project should remain deferred until explicitly resumed")
            return "defer", rationale
        if priority_score >= 0.45:
            rationale.append("project is currently the strongest continuation candidate")
            return "continue", rationale
        rationale.append("project remains below the current portfolio continuation threshold")
        return "defer", rationale

    def _project_priority_record(
        self,
        project: ResearchProject,
        candidate: Optional[PrioritizedActionCandidate],
        evidence_debt: EvidenceDebtRecord,
        staleness: ProjectStalenessRecord,
    ) -> ProjectPriorityRecord:
        thread = self._project_thread(project)
        strategic_priority = min(1.0, max(0.0, float(project.priority) / 5.0))
        expected_value = 0.0
        contradiction_pressure = 0.0
        benchmark_readiness = self._project_benchmark_readiness(project)
        action_id = None
        action_kind = None
        if candidate is not None:
            expected_value = max(0.0, min(1.0, candidate.score))
            contradiction_pressure = candidate.score_breakdown.contradiction_urgency
            benchmark_readiness = max(benchmark_readiness, candidate.score_breakdown.benchmark_value)
            action_id = candidate.action_id
            action_kind = candidate.action_kind
        elif thread is not None:
            contradiction_pressure = min(1.0, float(len(thread.contradicting_evidence_ids)) * 0.5)

        staleness_penalty = {
            "fresh": 0.0,
            "watch": 0.15,
            "stale": 0.35,
            "critical": 0.55,
        }.get(staleness.staleness_level, 0.0)
        budget_pressure = {
            "low": 0.05,
            "medium": 0.35,
            "high": 0.7,
            "exhausted": 1.0,
        }.get(project.budget_ledger.budget_pressure_level, 0.1)
        debt_penalty_scale = 0.25 if action_kind in {
            "mechanism_ablation",
            "replication_check",
            "seed_variance_check",
            "contradiction_resolution",
            "benchmark_stress_probe",
            "calibration_check",
            "environment_reproduction_check",
            "claim_linkage_sanity_check",
        } else 0.5
        priority_score = round(
            max(
                0.0,
                expected_value
                + (0.25 * strategic_priority)
                + (0.15 * contradiction_pressure)
                + (0.15 * benchmark_readiness)
                - (0.2 * staleness_penalty)
                - (0.2 * budget_pressure)
                - (evidence_debt.overall_debt * debt_penalty_scale),
            ),
            6,
        )
        recommended_state, rationale = self._recommend_portfolio_state(
            project=project,
            thread=thread,
            candidate=candidate,
            evidence_debt=evidence_debt,
            staleness=staleness,
            priority_score=priority_score,
        )
        if evidence_debt.promotion_blocked:
            rationale.append("promotion remains blocked by evidence debt")
        if staleness.staleness_level in {"stale", "critical"}:
            rationale.append(staleness.reason)
        return ProjectPriorityRecord(
            record_id=self._continuity_id("project-priority"),
            project_id=project.project_id,
            action_id=action_id,
            priority_score=priority_score,
            strategic_priority=strategic_priority,
            expected_value=expected_value,
            evidence_debt=evidence_debt.overall_debt,
            contradiction_pressure=contradiction_pressure,
            staleness_penalty=staleness_penalty,
            budget_pressure=budget_pressure,
            benchmark_readiness=benchmark_readiness,
            recommended_state=recommended_state,  # type: ignore[arg-type]
            rationale=rationale,
        )

    def _evaluate_portfolio(
        self,
        *,
        include_blocked: bool = True,
        mode: str = "balanced",
    ) -> tuple[ResearchPortfolio, list[ProjectPriorityRecord], list[EvidenceDebtRecord], list[ProjectStalenessRecord]]:
        projects = self.store.list_research_projects()
        ranking = self._rank_action_candidates(
            include_blocked=include_blocked,
            limit=max(1, max(10, len(projects) * 6)),
            mode=mode,
            persist=False,
        )
        best_candidates: dict[str, PrioritizedActionCandidate] = {}
        for candidate in ranking.candidates:
            best_candidates.setdefault(candidate.project_id, candidate)

        evidence_debts = [self._compute_evidence_debt(project) for project in projects]
        evidence_map = {item.project_id: item for item in evidence_debts}
        stale_records = [self._compute_project_staleness(project) for project in projects]
        stale_map = {item.project_id: item for item in stale_records}
        priorities = [
            self._project_priority_record(
                project,
                best_candidates.get(project.project_id),
                evidence_map[project.project_id],
                stale_map[project.project_id],
            )
            for project in projects
        ]
        recommendation_rank = {
            "continue": 0,
            "resume": 1,
            "escalate": 2,
            "defer": 3,
            "park": 4,
            "block": 5,
            "complete": 6,
            "retire": 7,
        }
        priorities.sort(
            key=lambda item: (
                recommendation_rank.get(item.recommended_state, 99),
                -item.priority_score,
                item.project_id,
            )
        )
        selected = next(
            (
                item.project_id
                for item in priorities
                if item.recommended_state in {"continue", "resume"}
            ),
            None,
        )
        existing_portfolio = self.store.load_research_portfolio()
        health = PortfolioHealthSnapshot(
            total_projects=len(projects),
            active_projects=len([item for item in projects if item.status == "active"]),
            paused_projects=len([item for item in projects if item.status == "paused"]),
            blocked_projects=len([item for item in projects if item.status == "blocked"]),
            stale_projects=len([item for item in stale_records if item.staleness_level in {"stale", "critical"}]),
            parked_projects=len([item for item in priorities if item.recommended_state == "park"]),
            completed_projects=len([item for item in projects if item.status == "completed"]),
            abandoned_projects=len([item for item in projects if item.status == "abandoned"]),
            resume_candidates=len([item for item in stale_records if item.resume_candidate]),
            promotion_blocked_projects=len([item for item in evidence_debts if item.promotion_blocked]),
            selected_project_id=selected,
        )
        portfolio = existing_portfolio.model_copy(
            update={
                "active_project_ids": [item.project_id for item in projects if item.status == "active"],
                "paused_project_ids": [item.project_id for item in projects if item.status == "paused"],
                "blocked_project_ids": [item.project_id for item in projects if item.status == "blocked"],
                "stale_project_ids": [item.project_id for item in stale_records if item.staleness_level in {"stale", "critical"}],
                "parked_project_ids": [item.project_id for item in priorities if item.recommended_state == "park"],
                "completed_project_ids": [item.project_id for item in projects if item.status == "completed"],
                "abandoned_project_ids": [item.project_id for item in projects if item.status == "abandoned"],
                "latest_selected_project_id": selected,
                "health_snapshot": health,
            }
        )
        return portfolio, priorities, evidence_debts, stale_records

    def portfolio_review(
        self,
        *,
        include_blocked: bool = True,
        limit: int = 10,
        mode: str = "balanced",
    ) -> dict[str, Any]:
        portfolio, priorities, evidence_debts, stale_records = self._evaluate_portfolio(
            include_blocked=include_blocked,
            mode=mode,
        )
        self.store.save_research_portfolio(portfolio)
        for record in priorities:
            self.store.append_project_priority_record(record)
        for record in evidence_debts:
            self.store.append_evidence_debt_record(record)
        for record in stale_records:
            self.store.append_project_staleness_record(record)
        self.store.append_audit_event(
            "research_portfolio",
            "portfolio_review",
            {
                "portfolio_id": portfolio.portfolio_id,
                "selected_project_id": portfolio.latest_selected_project_id,
                "priority_records": len(priorities),
            },
        )
        return {
            "portfolio": portfolio.model_dump(mode="json"),
            "top_projects": [item.model_dump(mode="json") for item in priorities[: max(1, limit)]],
            "evidence_debts": [item.model_dump(mode="json") for item in evidence_debts[: max(1, limit)]],
            "stale_projects": [
                item.model_dump(mode="json")
                for item in stale_records
                if item.staleness_level in {"stale", "critical"}
            ][: max(1, limit)],
            "resume_candidates": [
                item.model_dump(mode="json")
                for item in stale_records
                if item.resume_candidate
            ][: max(1, limit)],
            "latest_portfolio_decision": self.store.latest_portfolio_decision().model_dump(mode="json")
            if self.store.latest_portfolio_decision()
            else None,
        }

    def stale_projects(self, *, limit: int = 10, mode: str = "balanced") -> dict[str, Any]:
        _, _, _, stale_records = self._evaluate_portfolio(include_blocked=True, mode=mode)
        rows = [
            item.model_dump(mode="json")
            for item in stale_records
            if item.staleness_level in {"stale", "critical"}
        ]
        rows.sort(key=lambda item: (-item.get("hours_since_progress", 0.0), item.get("project_id", "")))
        return {"stale_projects": rows[: max(1, limit)]}

    def evidence_debt(
        self,
        *,
        project_id: Optional[str] = None,
        limit: int = 10,
        mode: str = "balanced",
    ) -> dict[str, Any]:
        _, _, evidence_debts, _ = self._evaluate_portfolio(include_blocked=True, mode=mode)
        rows = [item for item in evidence_debts if project_id is None or item.project_id == project_id]
        rows.sort(key=lambda item: (-item.overall_debt, item.project_id))
        return {"records": [item.model_dump(mode="json") for item in rows[: max(1, limit)]]}

    def resume_candidates(self, *, limit: int = 10, mode: str = "balanced") -> dict[str, Any]:
        _, priorities, _, stale_records = self._evaluate_portfolio(include_blocked=True, mode=mode)
        priority_map = {item.project_id: item for item in priorities}
        rows = []
        for item in stale_records:
            if not item.resume_candidate:
                continue
            rows.append(
                {
                    "staleness": item.model_dump(mode="json"),
                    "priority": priority_map[item.project_id].model_dump(mode="json")
                    if item.project_id in priority_map
                    else None,
                }
            )
        rows.sort(
            key=lambda item: (
                -(item["priority"]["priority_score"] if item["priority"] else 0.0),
                item["staleness"].get("project_id", ""),
            )
        )
        return {"resume_candidates": rows[: max(1, limit)]}

    def portfolio_decide(
        self,
        *,
        include_blocked: bool = True,
        limit: int = 10,
        mode: str = "balanced",
    ) -> dict[str, Any]:
        portfolio, priorities, evidence_debts, stale_records = self._evaluate_portfolio(
            include_blocked=include_blocked,
            mode=mode,
        )
        selected_record = next(
            (item for item in priorities if item.recommended_state in {"continue", "resume"}),
            None,
        )
        decision = PortfolioDecision(
            decision_id=self._continuity_id("portfolio-decision"),
            selected_project_id=selected_record.project_id if selected_record is not None else None,
            selected_action_id=selected_record.action_id if selected_record is not None else None,
            deferred_project_ids=[item.project_id for item in priorities if item.recommended_state == "defer"],
            parked_project_ids=[item.project_id for item in priorities if item.recommended_state == "park"],
            resumed_project_ids=[item.project_id for item in priorities if item.recommended_state == "resume"],
            escalated_project_ids=[item.project_id for item in priorities if item.recommended_state == "escalate"],
            retired_project_ids=[item.project_id for item in priorities if item.recommended_state in {"retire", "complete"}],
            rationale=[
                "selected the highest-ranked project whose recommended state was continue or resume",
                "deferred or parked lower-value, blocked, or budget-exhausted projects",
            ],
            policy_snapshot=self._prioritization_policy(mode=mode),
            project_priority_records=priorities,
        )
        updated_portfolio = portfolio.model_copy(
            update={
                "latest_decision_id": decision.decision_id,
                "latest_selected_project_id": decision.selected_project_id,
                "health_snapshot": portfolio.health_snapshot.model_copy(
                    update={"selected_project_id": decision.selected_project_id}
                ),
            }
        )
        self.store.save_research_portfolio(updated_portfolio)
        self.store.append_portfolio_decision(decision)
        for record in priorities:
            self.store.append_project_priority_record(record)
        for record in evidence_debts:
            self.store.append_evidence_debt_record(record)
        for record in stale_records:
            self.store.append_project_staleness_record(record)
        self.store.append_audit_event(
            "research_portfolio",
            "portfolio_decide",
            {
                "decision_id": decision.decision_id,
                "selected_project_id": decision.selected_project_id,
                "selected_action_id": decision.selected_action_id,
            },
        )
        return {
            "portfolio": updated_portfolio.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "top_projects": [item.model_dump(mode="json") for item in priorities[: max(1, limit)]],
            "evidence_debts": [item.model_dump(mode="json") for item in evidence_debts[: max(1, limit)]],
            "stale_projects": [
                item.model_dump(mode="json")
                for item in stale_records
                if item.staleness_level in {"stale", "critical"}
            ][: max(1, limit)],
        }

    def _attach_project_to_study(
        self,
        report: ProblemStudyReport,
        *,
        problem: str,
        resolution: ProblemResolutionReport,
        evidence_summary: str,
        project_id: Optional[str] = None,
        wall_clock_minutes: float = 0.0,
    ) -> ProblemStudyReport:
        project = self.store.get_research_project(project_id) if project_id else None
        if project is None:
            project = self.create_project(
                problem,
                benchmark_tier=report.benchmark_tier,
                requested_benchmark=report.requested_benchmark,
            )
        thread = self._project_thread(project)
        if thread is None:
            raise RuntimeError(f"Project {project.project_id} has no active hypothesis thread.")
        project = self._invalidate_active_action(project, thread)
        question_id = self._continuity_id("question")
        action_id = self._continuity_id("action")
        hypothesis_text = (
            report.hypotheses[0].hypothesis
            if report.hypotheses
            else f"Investigate benchmark-aligned evidence for '{problem}'."
        )
        open_question = ResearchOpenQuestion(
            question_id=question_id,
            project_id=project.project_id,
            thread_id=thread.thread_id,
            question=(
                report.hypotheses[0].unresolved_assumptions[0]
                if report.hypotheses and report.hypotheses[0].unresolved_assumptions
                else f"What execution should TAR run next for '{problem}'?"
            ),
            importance=max(0.25, resolution.confidence),
            uncertainty_type="execution_gap",
            blocking=False,
            status="open",
        )
        action = ResearchPlannedAction(
            action_id=action_id,
            project_id=project.project_id,
            thread_id=thread.thread_id,
            action_kind="run_problem_study",
            description=report.next_action,
            estimated_cost=1.0,
            expected_evidence_gain=max(0.25, resolution.confidence),
            depends_on=[question_id],
            status="planned",
        )
        updated_thread = thread.model_copy(
            update={
                "hypothesis": hypothesis_text,
                "status": "testing" if report.status != "build_failed" else "parked",
                "confidence_state": "exploratory",
                "supporting_evidence_ids": sorted(
                    set(
                        [
                            *thread.supporting_evidence_ids,
                            *(report.cited_research_ids or []),
                            *(report.retrieved_memory_ids or []),
                        ]
                    )
                ),
                "contradicting_evidence_ids": sorted(
                    set(
                        [
                            *thread.contradicting_evidence_ids,
                            *(
                                (report.contradiction_review.conflicting_document_ids if report.contradiction_review else [])
                            ),
                            *(
                                (report.contradiction_review.conflicting_claim_ids if report.contradiction_review else [])
                            ),
                        ]
                    )
                ),
                "open_question_ids": [*thread.open_question_ids, question_id],
                "next_action_id": action_id,
                "stop_reason": "runtime_failure" if report.status == "build_failed" else None,
                "updated_at": self._project_now(),
            }
        )
        updated_project = project.model_copy(
            update={
                "status": "blocked" if report.status == "build_failed" else "active",
                "domain_profile": report.profile_id,
                "latest_decision_summary": report.next_action,
                "budget_ledger": self._spend_project_budget(project.budget_ledger, wall_clock_minutes=wall_clock_minutes),
                "open_questions": [*project.open_questions, open_question],
                "planned_actions": [*project.planned_actions, action],
            }
        )
        updated_project = self._replace_project_thread(updated_project, updated_thread)
        updated_project = self._persist_project(
            updated_project,
            latest_evidence_summary=evidence_summary,
            blockers=["build_failed"] if report.status == "build_failed" else [],
        )
        return report.model_copy(
            update={
                "project_id": updated_project.project_id,
                "thread_id": updated_thread.thread_id,
                "open_question_id": open_question.question_id,
                "next_action_id": action.action_id,
            }
        )

    def _update_project_after_schedule(self, entry: ProblemScheduleEntry) -> None:
        if not entry.project_id or not entry.action_id:
            return
        project = self.store.get_research_project(entry.project_id)
        if project is None:
            return
        action = self._project_action(project, entry.action_id)
        if action is None:
            return
        updated_action = action.model_copy(
            update={
                "status": "queued",
                "scheduled_job_id": entry.schedule_id,
                "updated_at": self._project_now(),
            }
        )
        updated_project = self._replace_project_action(project, updated_action)
        self._persist_project(updated_project)

    def _update_project_after_execution(
        self,
        study: ProblemStudyReport,
        report: ProblemExecutionReport,
        *,
        wall_clock_minutes: float,
    ) -> ProblemExecutionReport:
        if not study.project_id:
            return report
        project = self.store.get_research_project(study.project_id)
        if project is None:
            return report
        thread = self._project_thread(project, study.thread_id)
        if thread is None:
            return report
        question = self._project_question(project, study.open_question_id)
        action = self._project_action(project, study.next_action_id)
        updated_project = project
        if question is not None and question.status == "open":
            updated_question = question.model_copy(update={"status": "resolved", "resolved_at": self._project_now()})
            updated_project = self._replace_project_question(updated_project, updated_question)
        if action is not None:
            updated_action = action.model_copy(
                update={
                    "status": "completed" if report.status == "completed" else "failed",
                    "result_refs": [*action.result_refs, report.artifact_path],
                    "updated_at": self._project_now(),
                }
            )
            updated_project = self._replace_project_action(updated_project, updated_action)
        next_question = ResearchOpenQuestion(
            question_id=self._continuity_id("question"),
            project_id=project.project_id,
            thread_id=thread.thread_id,
            question=f"What should TAR do next after execution '{report.problem_id}'?",
            importance=0.6,
            uncertainty_type="followup_decision",
            blocking=report.status != "completed",
            status="open",
        )
        next_action = ResearchPlannedAction(
            action_id=self._continuity_id("action"),
            project_id=project.project_id,
            thread_id=thread.thread_id,
            action_kind="review_execution_result",
            description=report.recommended_next_step,
            estimated_cost=0.5,
            expected_evidence_gain=0.5,
            depends_on=[next_question.question_id],
            status="planned",
        )
        updated_thread = thread.model_copy(
            update={
                "status": "supported" if report.status == "completed" else ("parked" if report.status in {"dependency_failure", "failed"} else "testing"),
                "confidence_state": "provisional" if report.status == "completed" else "exploratory",
                "supporting_evidence_ids": sorted(set([*thread.supporting_evidence_ids, report.artifact_path])),
                "open_question_ids": [*thread.open_question_ids, next_question.question_id],
                "next_action_id": next_action.action_id,
                "stop_reason": (
                    "dependency_missing"
                    if report.status == "dependency_failure"
                    else ("runtime_failure" if report.status == "failed" else None)
                ),
                "updated_at": self._project_now(),
            }
        )
        updated_ledger = self._spend_project_budget(
            updated_project.budget_ledger,
            wall_clock_minutes=wall_clock_minutes,
            experiments=1,
        )
        project_status = (
            "blocked"
            if report.status in {"dependency_failure", "failed"}
            else ("paused" if updated_ledger.budget_exhausted else "active")
        )
        updated_project = updated_project.model_copy(
            update={
                "status": project_status,
                "budget_ledger": updated_ledger,
                "latest_decision_summary": report.recommended_next_step,
                "open_questions": [*updated_project.open_questions, next_question],
                "planned_actions": [*updated_project.planned_actions, next_action],
            }
        )
        updated_project = self._replace_project_thread(updated_project, updated_thread)
        blockers = []
        if report.status == "dependency_failure":
            blockers.append("dependency_missing")
        elif report.status == "failed":
            blockers.append("runtime_failure")
        elif updated_ledger.budget_exhausted:
            blockers.append("budget_exhausted")
        updated_project = self._persist_project(
            updated_project,
            latest_evidence_summary=report.summary,
            blockers=blockers,
        )
        return report.model_copy(
            update={
                "project_id": updated_project.project_id,
                "thread_id": updated_thread.thread_id,
                "action_id": report.action_id or study.next_action_id,
            }
        )

    def _claim_policy(self) -> ClaimAcceptancePolicy:
        return self.inference_bridge.default_claim_policy()

    def _claim_review_query(self, trial_id: str, verification: VerificationReport) -> str:
        return (
            f"claim review {trial_id} thermodynamic calibration dimensionality "
            f"control_score={verification.control_score:.4f} "
            f"verification={verification.verdict}"
        )

    def _resolve_claim_benchmark_context(self, problem_id: Optional[str]) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "benchmark_problem_id": problem_id,
            "benchmark_execution_created_at": None,
            "benchmark_execution_mode": None,
            "supporting_benchmark_ids": [],
            "supporting_benchmark_names": [],
            "canonical_comparable": False,
            "canonical_comparability_source": "none",
            "verdict_inputs_complete": True,
            "linkage_status": "none",
            "linkage_note": "No benchmark problem was linked to this claim review.",
        }
        if not problem_id:
            return context

        execution = self.store.latest_problem_execution(problem_id)
        if execution is not None:
            context.update(
                {
                    "benchmark_execution_created_at": execution.executed_at,
                    "benchmark_execution_mode": execution.execution_mode,
                    "supporting_benchmark_ids": list(execution.benchmark_ids),
                    "supporting_benchmark_names": list(execution.benchmark_names),
                    "canonical_comparable": execution.canonical_comparable,
                    "canonical_comparability_source": "problem_execution",
                    "verdict_inputs_complete": True,
                    "linkage_status": "exact",
                    "linkage_note": f"Benchmark evidence is bound to problem execution {execution.problem_id}.",
                }
            )
            return context

        study = self.store.latest_problem_study(problem_id)
        if study is not None:
            context.update(
                {
                    "supporting_benchmark_ids": list(study.benchmark_ids),
                    "supporting_benchmark_names": list(study.benchmark_names),
                    "canonical_comparable": study.canonical_comparable,
                    "canonical_comparability_source": "problem_study",
                    "verdict_inputs_complete": True,
                    "linkage_status": "exact",
                    "linkage_note": f"Benchmark evidence is bound to problem study {study.problem_id}.",
                }
            )
            return context

        context.update(
            {
                "verdict_inputs_complete": False,
                "linkage_note": f"No problem study or execution exists for problem_id={problem_id}.",
            }
        )
        return context

    def _build_research_decision(
        self,
        *,
        prompt: str,
        evidence_bundle: Any,
        hypotheses: list[Any],
        selected_action: str,
        claim_verdict: Optional[ClaimVerdict] = None,
        mode: str = "research_chat",
        trial_id: Optional[str] = None,
        problem_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        action_id: Optional[str] = None,
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
            trial_id=trial_id,
            problem_id=problem_id,
            thread_id=thread_id,
            action_id=action_id,
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
        base_model_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
    ) -> CheckpointRecord:
        metadata = {}
        if trust_remote_code is not None:
            metadata["trust_remote_code"] = bool(trust_remote_code)
        record = self.inference_bridge.register_checkpoint(
            name=name,
            model_path=model_path,
            backend=backend,
            role=role,
            base_model_id=base_model_id,
            adapter_path=adapter_path,
            metadata=metadata,
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
        trust_remote_code: Optional[bool] = None,
    ) -> InferenceEndpointPlan:
        plan = self.inference_bridge.build_endpoint(
            name=name,
            host=host,
            port=port,
            role=role,
            trust_remote_code=trust_remote_code,
        )
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
        trust_remote_code: Optional[bool] = None,
        wait_for_health: bool = False,
    ) -> EndpointRecord:
        record = self.inference_bridge.start_endpoint(
            name=name,
            host=host,
            port=port,
            role=role,
            trust_remote_code=trust_remote_code,
            wait_for_health=wait_for_health,
        )
        self.store.append_audit_event("inference", "start_endpoint", record.model_dump(mode="json"))
        return record

    def stop_endpoint(self, endpoint_name: str) -> EndpointRecord:
        record = self.inference_bridge.stop_endpoint(endpoint_name)
        self.store.append_audit_event("inference", "stop_endpoint", record.model_dump(mode="json"))
        return record

    def restart_endpoint(
        self,
        endpoint_name: str,
        *,
        trust_remote_code: Optional[bool] = None,
        wait_for_health: bool = False,
    ) -> EndpointRecord:
        record = self.inference_bridge.restart_endpoint(
            endpoint_name,
            trust_remote_code=trust_remote_code,
            wait_for_health=wait_for_health,
        )
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

    def select_operator_checkpoint(
        self,
        *,
        checkpoint_name: str,
        mode: str = "tuned_local",
        role: str = "assistant",
        endpoint_name: Optional[str] = None,
    ) -> OperatorServingStatus:
        status = self.inference_bridge.select_operator_checkpoint(
            checkpoint_name=checkpoint_name,
            mode=mode,
            role=role,
            endpoint_name=endpoint_name,
        )
        self.store.append_audit_event("inference", "select_operator_checkpoint", status.model_dump(mode="json"))
        return status

    def operator_serving_status(self) -> OperatorServingStatus:
        return self.inference_bridge.operator_serving_status()

    def claim_policy(self) -> dict[str, Any]:
        return self.inference_bridge.default_claim_policy().model_dump(mode="json")

    def research_decision_log(self, count: int = 20) -> dict[str, Any]:
        rows = list(self.store.iter_research_decisions())[-count:]
        return {"decisions": [item.model_dump(mode="json") for item in rows]}

    def frontier_status(self) -> FrontierStatus:
        payload_env = self.payload_environment.load()
        literature_status = self.literature_engine.status()
        vault_status = self.vault.stats() if self.vault is not None else {}
        frontier_summary = self.frontier_gap_status(limit=5)
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
            experiment_backend_runtime_records=self.store.list_experiment_backend_runtimes(),
            payload_environment=payload_env,
            literature_artifacts=literature_status["artifacts"],  # type: ignore[index]
            literature_conflicts=literature_status["conflicts"],  # type: ignore[index]
            literature_manifests=literature_status.get("manifests", 0),  # type: ignore[arg-type]
            latest_literature_manifest=self.literature_engine.latest_manifest(),
            embedder=vault_status.get("embedder", "unavailable"),
            semantic_research_ready=bool(vault_status.get("semantic_research_ready", False)),
            reranker=str(vault_status.get("reranker", "scientific-hybrid-reranker")),
            reranker_ready=bool(vault_status.get("reranker_ready", False)),
            literature_capabilities=literature_capability,
            runtime_heartbeat=self.runtime_daemon.load_heartbeat(),
            registered_checkpoints=self.inference_bridge.list_checkpoints(),
            managed_endpoints=self.inference_bridge.list_endpoints(),
            role_assignments=self.inference_bridge.list_role_assignments(),
            operator_serving=self.inference_bridge.operator_serving_status(),
            claim_policy=self._claim_policy(),
            recent_claim_verdicts=list(self.store.iter_claim_verdicts())[-5:],
            benchmark_profiles=self.profile_registry.benchmark_profile_counts(),
            frontier_gap_counts=frontier_summary["counts"],
            frontier_gap_scans=frontier_summary["scan_count"],
            latest_frontier_gap_scan=FrontierGapScanReport.model_validate(frontier_summary["latest_scan"]) if frontier_summary["latest_scan"] else None,
            recent_frontier_gaps=[FrontierGapRecord.model_validate(item) for item in frontier_summary["gaps"]],
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
            self._assert_execution_policy(
                execution_kind="external_code",
                source_path="tar_lab.docker_runner.build_science_environment",
                sandboxed=True,
            )
            build_result = self.docker_runner.build_science_environment(bundle, dry_run=False)
            bundle = bundle.model_copy(
                update={
                    "build_status": "built" if build_result.returncode == 0 else "failed",
                    "build_stdout": build_result.stdout,
                    "build_stderr": build_result.stderr,
                }
            )
            bundle = self.payload_environment.attach_science_build_attestation(
                bundle,
                build_result=build_result,
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
                "build_attestation_id": bundle.build_attestation.attestation_id if bundle.build_attestation else None,
            },
        )
        return bundle

    def study_problem(
        self,
        problem: str,
        build_env: bool = False,
        max_results: int = 6,
        *,
        project_id: Optional[str] = None,
        benchmark_tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
        canonical_only: bool = False,
        no_proxy_benchmarks: bool = False,
    ) -> ProblemStudyReport:
        started_at = datetime.now(timezone.utc)
        resolution = self.resolve_problem(
            problem,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
        )
        research = self.ingest_research(topic=problem, max_results=max_results)
        self._sync_memory()
        retrieval_mode: str = "lexical_fallback"
        retrieval_degraded = False
        retrieval_note: Optional[str] = None
        try:
            if self.vault is not None:
                memory_hits = self.vault.search(problem, n_results=5, require_research_grade=True)
                retrieval_mode = "semantic"
            else:
                retrieval_degraded = True
                retrieval_note = "Retrieval degraded to lexical fallback. Semantic search unavailable."
                memory_hits = []
        except Exception:
            retrieval_degraded = True
            retrieval_note = "Retrieval degraded to lexical fallback. Semantic search unavailable."
            memory_hits = self.vault.search(problem, n_results=5) if self.vault is not None else []
        conflict_hits = self._retrieval_conflict_hits(memory_hits)
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
            self._assert_execution_policy(
                execution_kind="external_code",
                source_path="tar_lab.docker_runner.build_science_environment",
                sandboxed=True,
            )
            build_result = self.docker_runner.build_science_environment(bundle, dry_run=False)
            bundle = bundle.model_copy(
                update={
                    "build_status": "built" if build_result.returncode == 0 else "failed",
                    "build_stdout": build_result.stdout,
                    "build_stderr": build_result.stderr,
                }
            )
            bundle = self.payload_environment.attach_science_build_attestation(
                bundle,
                build_result=build_result,
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
                "retrieval_mode": retrieval_mode,
                "retrieval_conflict_count": len(conflict_hits),
                "notes": [retrieval_note] if retrieval_note else [],
            }
        )
        if build_env and bundle.build_status == "failed":
            report = report.model_copy(update={"status": "build_failed"})
        elif retrieval_degraded:
            report = report.model_copy(update={"status": "retrieval_degraded"})
        planning_minutes = max(
            0.0,
            (datetime.now(timezone.utc) - started_at).total_seconds() / 60.0,
        )
        report = self._attach_project_to_study(
            report,
            problem=problem,
            resolution=resolution,
            evidence_summary=evidence_bundle.notes[0] if evidence_bundle.notes else resolution.summary,
            project_id=project_id,
            wall_clock_minutes=planning_minutes,
        )
        self.store.append_problem_study(report)
        if report.project_id is not None:
            self._update_evidence_debt_from_retrieval(report.project_id, conflict_hits)
        Path(report.environment.study_plan_path).write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        decision = self._build_research_decision(
            prompt=problem,
            evidence_bundle=evidence_bundle,
            hypotheses=report.hypotheses,
            selected_action=report.next_action,
            mode="problem_study",
            problem_id=report.problem_id,
            thread_id=report.thread_id,
            action_id=report.next_action_id,
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
        derived_priority_score: Optional[float] = None
        priority_source: Optional[str] = None
        if priority == 0 and study.project_id and study.next_action_id:
            project = self.store.get_research_project(study.project_id)
            if project is not None:
                candidate = self._build_action_candidate(
                    project,
                    policy=self._prioritization_policy(),
                    action_id=study.next_action_id,
                )
                if candidate is not None:
                    derived_priority_score = candidate.score
                    priority = self._priority_to_queue_value(candidate.score)
                    priority_source = "ws18_ranked_action"
        entry = self.scheduler.schedule(
            study,
            use_docker=use_docker,
            build_env=build_env,
            run_at=run_at,
            delay_s=delay_s,
            repeat_interval_s=repeat_interval_s,
            max_runs=max_runs,
            priority=priority,
            priority_score=derived_priority_score,
            priority_source=priority_source,
        )
        self.store.append_audit_event(
            "science",
            "schedule_problem_study",
            entry.model_dump(mode="json"),
        )
        self._update_project_after_schedule(entry)
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
        started_at = datetime.now(timezone.utc)
        study = self.store.latest_problem_study(problem_id)
        if study is None:
            raise RuntimeError("No problem study is available to execute.")
        bundle = study.environment
        execution_report_path = Path(bundle.execution_report_path)

        if use_docker:
            if build_env or bundle.build_status not in {"built"}:
                self._assert_execution_policy(
                    execution_kind="external_code",
                    source_path="tar_lab.docker_runner.build_science_environment",
                    sandboxed=True,
                )
                build_result = self.docker_runner.build_science_environment(bundle, dry_run=False)
                bundle = bundle.model_copy(
                    update={
                        "build_status": "built" if build_result.returncode == 0 else "failed",
                        "build_stdout": build_result.stdout,
                        "build_stderr": build_result.stderr,
                    }
                )
                bundle = self.payload_environment.attach_science_build_attestation(
                    bundle,
                    build_result=build_result,
                )
                if build_result.returncode != 0:
                    report = ProblemExecutionReport(
                        problem_id=study.problem_id,
                        project_id=study.project_id,
                        thread_id=study.thread_id,
                        action_id=study.next_action_id,
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
                        build_attestation_path=bundle.build_attestation_path,
                        build_attestation_id=bundle.build_attestation.attestation_id if bundle.build_attestation else None,
                        image_digest=bundle.build_attestation.image_digest if bundle.build_attestation else None,
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
            self._assert_execution_policy(
                execution_kind="external_code",
                source_path="tar_lab.docker_runner.run_science_environment",
                sandboxed=True,
            )
            run_result = self.docker_runner.run_science_environment(bundle, dry_run=False)
            if run_result.returncode != 0 and not execution_report_path.exists():
                report = ProblemExecutionReport(
                    problem_id=study.problem_id,
                    project_id=study.project_id,
                    thread_id=study.thread_id,
                    action_id=study.next_action_id,
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
                    build_attestation_path=bundle.build_attestation_path,
                    build_attestation_id=bundle.build_attestation.attestation_id if bundle.build_attestation else None,
                    image_digest=bundle.build_attestation.image_digest if bundle.build_attestation else None,
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
            # execution_policy: deliberate_exception - reason: trusted internal TAR study runner.
            proc = self._run_subprocess(
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
                source_path="tar_lab.problem_runner",
                execution_kind="trusted_internal",
                deliberate_exception_reason="trusted internal TAR study runner",
                capture_output=True,
                text=True,
                check=False,
                cwd=self.workspace,
            )
            if proc.returncode != 0 and not execution_report_path.exists():
                report = ProblemExecutionReport(
                    problem_id=study.problem_id,
                    project_id=study.project_id,
                    thread_id=study.thread_id,
                    action_id=study.next_action_id,
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
                    build_attestation_path=bundle.build_attestation_path,
                    build_attestation_id=bundle.build_attestation.attestation_id if bundle.build_attestation else None,
                    image_digest=bundle.build_attestation.image_digest if bundle.build_attestation else None,
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
                "project_id": report.project_id or study.project_id,
                "thread_id": report.thread_id or study.thread_id,
                "action_id": report.action_id or study.next_action_id,
                "execution_mode": "docker_bundle" if use_docker else "local_python",
                "image_tag": report.image_tag or bundle.docker_image_tag,
                "manifest_path": report.manifest_path or bundle.run_manifest_path,
                "manifest_hash": report.manifest_hash or (bundle.run_manifest.hash_sha256 if bundle.run_manifest else None),
                "dependency_hash": report.dependency_hash or (bundle.image_manifest.dependency_lock.hash_sha256 if bundle.image_manifest else None),
                "build_attestation_path": report.build_attestation_path or bundle.build_attestation_path,
                "build_attestation_id": report.build_attestation_id or (bundle.build_attestation.attestation_id if bundle.build_attestation else None),
                "image_digest": report.image_digest or (bundle.build_attestation.image_digest if bundle.build_attestation else None),
                "reproducibility_complete": report.reproducibility_complete or bundle.reproducibility_complete,
                "sandbox_policy": report.sandbox_policy or bundle.sandbox_policy,
            }
        )
        execution_minutes = max(
            0.0,
            (datetime.now(timezone.utc) - started_at).total_seconds() / 60.0,
        )
        report = self._update_project_after_execution(
            study,
            report,
            wall_clock_minutes=execution_minutes,
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

    def _claim_review_query(
        self,
        *,
        trial_id: str,
        verification: VerificationReport,
        problem_id: Optional[str] = None,
    ) -> str:
        query = (
            f"claim review trial {trial_id} "
            f"verdict {verification.verdict} "
            "thermodynamic calibration dimensionality"
        )
        if problem_id:
            query += f" benchmark problem {problem_id}"
        return query

    def claim_verdict(
        self,
        trial_id: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> ClaimVerdict:
        resolved_trial_id = trial_id or self.store.load_recovery().trial_id
        if not resolved_trial_id:
            raise RuntimeError("No trial is available for claim review.")
        verification = self.store.latest_verification_report(resolved_trial_id)
        if verification is None:
            verification = self.verify_last_trial(resolved_trial_id)
        benchmark_context = self._resolve_claim_benchmark_context(problem_id)
        query = self._claim_review_query(
            trial_id=resolved_trial_id,
            verification=verification,
            problem_id=problem_id,
        )
        try:
            research_hits = (
                self.vault.search(query, n_results=5, require_research_grade=True)
                if self.vault is not None
                else []
            )
        except Exception:
            research_hits = []
        evidence_bundle = build_evidence_bundle(query, research_hits)
        supporting_research_ids = [
            hit.document_id for hit in research_hits if hit.document_id.startswith("research:")
        ]
        supporting_evidence_ids = [hit.document_id for hit in research_hits]
        verdict = self.verification_runner.assess_claim(
            verification,
            supporting_research_ids=supporting_research_ids,
            supporting_evidence_ids=supporting_evidence_ids,
            contradiction_review=evidence_bundle.contradiction_review,
            canonical_comparable=benchmark_context["canonical_comparable"],
            verification_report_trial_id=verification.trial_id,
            benchmark_problem_id=benchmark_context["benchmark_problem_id"],
            benchmark_execution_created_at=benchmark_context["benchmark_execution_created_at"],
            benchmark_execution_mode=benchmark_context["benchmark_execution_mode"],
            supporting_benchmark_ids=benchmark_context["supporting_benchmark_ids"],
            supporting_benchmark_names=benchmark_context["supporting_benchmark_names"],
            evidence_bundle_id=evidence_bundle.bundle_id,
            canonical_comparability_source=benchmark_context["canonical_comparability_source"],
            verdict_inputs_complete=benchmark_context["verdict_inputs_complete"],
            linkage_status=benchmark_context["linkage_status"],
            linkage_note=benchmark_context["linkage_note"],
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
                trial_id=resolved_trial_id,
                problem_id=problem_id,
            )
        )
        self.store.append_audit_event("verification", "claim_verdict", verdict.model_dump(mode="json"))
        return verdict

    def breakthrough_report(
        self,
        trial_id: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> BreakthroughReport:
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
        claim_verdict = self.claim_verdict(resolved_trial_id, problem_id=problem_id)
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
            self._assert_execution_policy(
                execution_kind="external_code",
                source_path="tar_lab.docker_runner.live_test",
                sandboxed=True,
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
        sandbox_policy = self.sandbox_policy()
        payload["image_tag"] = payload_env.image_tag if payload_env is not None else None
        payload["reproducibility_complete"] = bool(payload_env is not None and payload_env.reproducibility_complete)
        payload["manifest_hash"] = payload_env.run_manifest.hash_sha256 if payload_env and payload_env.run_manifest else None
        payload["build_status"] = payload_env.build_status if payload_env is not None else "not_requested"
        payload["build_attestation_path"] = payload_env.build_attestation_path if payload_env is not None else None
        payload["build_attestation"] = payload_env.build_attestation.model_dump(mode="json") if payload_env and payload_env.build_attestation else None
        payload["build_attestation_id"] = payload_env.build_attestation.attestation_id if payload_env and payload_env.build_attestation else None
        payload["image_digest"] = payload_env.build_attestation.image_digest if payload_env and payload_env.build_attestation else None
        payload["unresolved_dependency_count"] = len(payload_env.unresolved_packages) if payload_env is not None else 0
        payload["unresolved_dependencies"] = list(payload_env.unresolved_packages) if payload_env is not None else []
        payload["lock_incomplete_reason"] = payload_env.lock_incomplete_reason if payload_env is not None else None
        payload["safe_execution_mode"] = self.safe_execution_mode
        payload["execution_policy"] = self.execution_policy.model_dump(mode="json")
        payload["sandbox_profile"] = sandbox_policy.get("profile", "production")
        payload["sandbox_read_only_mounts"] = list(sandbox_policy.get("read_only_mounts", []))
        payload["sandbox_writable_mounts"] = list(sandbox_policy.get("writable_mounts", []))
        payload["sandbox_dev_override_active"] = bool(sandbox_policy.get("dev_override_active", False))
        payload["alerts"] = len(list(self.store.iter_alerts()))
        payload["endpoints"] = [item.model_dump(mode="json") for item in self.inference_bridge.list_endpoints()]
        payload["role_assignments"] = [item.model_dump(mode="json") for item in self.inference_bridge.list_role_assignments()]
        payload["operator_serving"] = self.inference_bridge.operator_serving_status().model_dump(mode="json")
        payload["experiment_backend_runtime_records"] = [
            item.model_dump(mode="json") for item in self.store.list_experiment_backend_runtimes()
        ]
        payload["claim_policy"] = self._claim_policy().model_dump(mode="json")
        latest_claim_verdict = self.store.latest_claim_verdict()
        payload["latest_claim_verdict"] = latest_claim_verdict.model_dump(mode="json") if latest_claim_verdict else None
        latest_problem_execution = self.store.latest_problem_execution()
        latest_problem_study = self.store.latest_problem_study()
        retrieval_summary = self._retrieval_mode_summary(limit=20)
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
        benchmark_truth_statuses = (
            latest_problem_execution.benchmark_truth_statuses
            if latest_problem_execution is not None
            else (latest_problem_study.benchmark_truth_statuses if latest_problem_study is not None else [])
        )
        payload["benchmark_tier"] = (
            latest_problem_execution.benchmark_tier
            if latest_problem_execution is not None
            else (latest_problem_study.benchmark_tier if latest_problem_study is not None else None)
        )
        payload["benchmark_ids"] = benchmark_ids
        payload["benchmark_names"] = benchmark_names
        payload["actual_benchmark_tiers"] = actual_benchmark_tiers
        payload["benchmark_truth_statuses"] = benchmark_truth_statuses
        payload["benchmark_alignment"] = (
            latest_problem_execution.benchmark_alignment
            if latest_problem_execution is not None
            else (latest_problem_study.benchmark_alignment if latest_problem_study is not None else "aligned")
        )
        payload["benchmark_name"] = benchmark_names[0] if benchmark_names else None
        payload["canonical_comparable"] = (
            latest_problem_execution.canonical_comparable
            if latest_problem_execution is not None
            else (latest_problem_study.canonical_comparable if latest_problem_study is not None else False)
        )
        runtime_payload = self.runtime_status()
        frontier_gap_summary = self.frontier_gap_status(limit=5)
        payload["retrieval_mode_breakdown"] = retrieval_summary["retrieval_mode_breakdown"]
        payload["degraded_retrieval_studies"] = retrieval_summary["degraded_retrieval_studies"]
        payload["recent_study_window"] = retrieval_summary["window"]
        payload["frontier_gap_counts"] = frontier_gap_summary["counts"]
        payload["frontier_gap_scans"] = frontier_gap_summary["scan_count"]
        payload["recent_frontier_gaps"] = frontier_gap_summary["gaps"]
        payload["latest_frontier_gap_scan"] = frontier_gap_summary["latest_scan"]
        payload["claim_verdict_lifecycle"] = runtime_payload.get("claim_verdict_lifecycle", {})
        payload["recent_verdict_window"] = runtime_payload.get("recent_verdict_window", 0)
        payload["escalated_verdict_ids"] = runtime_payload.get("escalated_verdict_ids", [])
        payload["runtime_policy"] = runtime_payload.get("runtime_policy", {})
        payload["queue_health"] = runtime_payload.get("queue_health", {})
        payload["literature"] = self.literature_status()
        payload["memory"] = self.vault.stats() if self.vault is not None else {"error": self.memory_error}
        if self.memory_error:
            payload["memory"]["error"] = self.memory_error
        payload["memory_warning"] = self.memory_error
        payload["regime"] = self.check_regime()
        payload["frontier"] = self.frontier_status().model_dump(mode="json")
        payload["runtime"] = runtime_payload
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
                    problem_id=study.problem_id,
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
                    trial_id=response.claim_verdict.trial_id if response.claim_verdict is not None else None,
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
