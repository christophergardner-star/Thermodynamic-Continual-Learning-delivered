from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tar_lab.schemas import (
    AlertRecord,
    BudgetAllocationDecision,
    BreakthroughReport,
    ClaimVerdict,
    CheckpointRecord,
    CheckpointRegistryState,
    DatasetManifest,
    DirectorPolicy,
    EvidenceDebtRecord,
    ExperimentBackendRuntimeRecord,
    EndpointRecord,
    EndpointRegistryState,
    FalsificationPlan,
    ImageManifest,
    MemoryStoreManifest,
    PortfolioDecision,
    ProjectPriorityRecord,
    ProjectStalenessRecord,
    PublicationHandoffPackage,
    GovernorMetrics,
    OperatorServingState,
    KnowledgeGraphEntry,
    KnowledgeGraphState,
    ProblemExecutionReport,
    PortfolioPrioritySnapshot,
    ProblemScheduleEntry,
    ProblemScheduleState,
    ResearchDocument,
    ResearchDecisionRecord,
    RecoveryState,
    RoleAssignment,
    RoleAssignmentState,
    RunManifest,
    ScoutTask,
    ProblemStudyReport,
    ResearchPortfolio,
    ResearchProject,
    ResearchProjectState,
    StrategistPlan,
    TrainingPayloadConfig,
    VerificationReport,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TARStateStore:
    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace).resolve()
        self.state_dir = self.workspace / "tar_state"
        self.logs_dir = self.workspace / "logs"
        self.data_dir = self.state_dir / "data"
        self.policies_dir = self.state_dir / "policies"
        self.recovery_path = self.state_dir / "recovery.json"
        self.knowledge_graph_path = self.state_dir / "knowledge_graph.json"
        self.research_intel_path = self.state_dir / "research_intel.jsonl"
        self.verification_reports_path = self.state_dir / "verification_reports.jsonl"
        self.breakthrough_reports_path = self.state_dir / "breakthrough_reports.jsonl"
        self.claim_verdicts_path = self.state_dir / "claim_verdicts.jsonl"
        self.research_decisions_path = self.state_dir / "research_decisions.jsonl"
        self.problem_studies_path = self.state_dir / "problem_studies.jsonl"
        self.problem_executions_path = self.state_dir / "problem_executions.jsonl"
        self.problem_schedule_path = self.state_dir / "problem_schedule.json"
        self.research_projects_path = self.state_dir / "research_projects.json"
        self.research_portfolio_path = self.state_dir / "research_portfolio.json"
        self.priority_snapshots_path = self.state_dir / "priority_snapshots.jsonl"
        self.budget_allocations_path = self.state_dir / "budget_allocations.jsonl"
        self.falsification_plans_path = self.state_dir / "falsification_plans.jsonl"
        self.project_priority_records_path = self.state_dir / "project_priority_records.jsonl"
        self.evidence_debt_records_path = self.state_dir / "evidence_debt_records.jsonl"
        self.project_staleness_records_path = self.state_dir / "project_staleness_records.jsonl"
        self.portfolio_decisions_path = self.state_dir / "portfolio_decisions.jsonl"
        self.publication_handoffs_path = self.state_dir / "publication_handoffs.jsonl"
        self.publication_handoffs_dir = self.state_dir / "publication_handoffs"
        self.checkpoint_registry_path = self.state_dir / "checkpoint_registry.json"
        self.endpoint_registry_path = self.state_dir / "inference_endpoints.json"
        self.endpoints_dir = self.state_dir / "endpoints"
        self.role_assignments_path = self.state_dir / "role_assignments.json"
        self.operator_serving_path = self.state_dir / "operator_serving.json"
        self.experiment_backends_dir = self.state_dir / "experiment_backends"
        self.memory_manifest_path = self.state_dir / "memory_manifest.json"
        self.alerts_path = self.state_dir / "alerts.jsonl"
        self.runtime_heartbeat_path = self.state_dir / "runtime_heartbeat.json"
        self.manifests_dir = self.state_dir / "manifests"
        self.metrics_log_path = self.logs_dir / "thermo_metrics.jsonl"
        self.audit_log_path = self.logs_dir / "activity_audit.log"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.endpoints_dir.mkdir(parents=True, exist_ok=True)
        self.publication_handoffs_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_backends_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        self._atomic_write_text(path, json.dumps(payload, indent=2))

    def load_recovery(self) -> RecoveryState:
        if not self.recovery_path.exists():
            recovery = RecoveryState()
            self.save_recovery(recovery)
            return recovery
        return RecoveryState.model_validate_json(self.recovery_path.read_text(encoding="utf-8"))

    def save_recovery(self, recovery: RecoveryState) -> None:
        updated = recovery.model_copy(update={"updated_at": _utc_now()})
        self._atomic_write_json(self.recovery_path, updated.model_dump(mode="json"))

    def load_knowledge_graph(self) -> KnowledgeGraphState:
        if not self.knowledge_graph_path.exists():
            graph = KnowledgeGraphState()
            self.save_knowledge_graph(graph)
            return graph
        return KnowledgeGraphState.model_validate_json(
            self.knowledge_graph_path.read_text(encoding="utf-8")
        )

    def save_knowledge_graph(self, graph: KnowledgeGraphState) -> None:
        self._atomic_write_json(self.knowledge_graph_path, graph.model_dump(mode="json"))

    def append_knowledge_entry(self, entry: KnowledgeGraphEntry) -> None:
        graph = self.load_knowledge_graph()
        graph.entries.append(entry)
        self.save_knowledge_graph(graph)

    def update_knowledge_entry(self, trial_id: str, **updates: Any) -> None:
        graph = self.load_knowledge_graph()
        new_entries: List[KnowledgeGraphEntry] = []
        for entry in graph.entries:
            if entry.trial_id == trial_id:
                new_entries.append(entry.model_copy(update=updates))
            else:
                new_entries.append(entry)
        self.save_knowledge_graph(KnowledgeGraphState(entries=new_entries))

    def append_metric(self, metrics: GovernorMetrics) -> None:
        with self.metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics.model_dump(mode="json")) + "\n")

    def iter_metrics(self) -> Iterable[GovernorMetrics]:
        if not self.metrics_log_path.exists():
            return []
        rows: List[GovernorMetrics] = []
        for line in self.metrics_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(GovernorMetrics.model_validate_json(line))
        return rows

    def tail_metrics(self, count: int = 3) -> List[GovernorMetrics]:
        rows = list(self.iter_metrics())
        return rows[-count:]

    def append_research_document(self, document: ResearchDocument) -> None:
        with self.research_intel_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(document.model_dump(mode="json")) + "\n")

    def iter_research_documents(self) -> Iterable[ResearchDocument]:
        if not self.research_intel_path.exists():
            return []
        rows: List[ResearchDocument] = []
        for line in self.research_intel_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(ResearchDocument.model_validate_json(line))
        return rows

    def append_verification_report(self, report: VerificationReport) -> None:
        with self.verification_reports_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report.model_dump(mode="json")) + "\n")

    def iter_verification_reports(self) -> Iterable[VerificationReport]:
        if not self.verification_reports_path.exists():
            return []
        rows: List[VerificationReport] = []
        for line in self.verification_reports_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(VerificationReport.model_validate_json(line))
        return rows

    def latest_verification_report(self, trial_id: Optional[str] = None) -> Optional[VerificationReport]:
        rows = list(self.iter_verification_reports())
        for report in reversed(rows):
            if trial_id is None or report.trial_id == trial_id:
                return report
        return None

    def append_breakthrough_report(self, report: BreakthroughReport) -> None:
        with self.breakthrough_reports_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report.model_dump(mode="json")) + "\n")

    def iter_breakthrough_reports(self) -> Iterable[BreakthroughReport]:
        if not self.breakthrough_reports_path.exists():
            return []
        rows: List[BreakthroughReport] = []
        for line in self.breakthrough_reports_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(BreakthroughReport.model_validate_json(line))
        return rows

    def latest_breakthrough_report(self, trial_id: Optional[str] = None) -> Optional[BreakthroughReport]:
        rows = list(self.iter_breakthrough_reports())
        for report in reversed(rows):
            if trial_id is None or report.trial_id == trial_id:
                return report
        return None

    def append_claim_verdict(self, verdict: ClaimVerdict) -> None:
        with self.claim_verdicts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(verdict.model_dump(mode="json")) + "\n")

    def iter_claim_verdicts(self) -> Iterable[ClaimVerdict]:
        if not self.claim_verdicts_path.exists():
            return []
        rows: List[ClaimVerdict] = []
        for line in self.claim_verdicts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(ClaimVerdict.model_validate_json(line))
        return rows

    def latest_claim_verdict(self, trial_id: Optional[str] = None) -> Optional[ClaimVerdict]:
        rows = list(self.iter_claim_verdicts())
        for verdict in reversed(rows):
            if trial_id is None or verdict.trial_id == trial_id:
                return verdict
        return None

    def append_research_decision(self, record: ResearchDecisionRecord) -> None:
        with self.research_decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")

    def iter_research_decisions(self) -> Iterable[ResearchDecisionRecord]:
        if not self.research_decisions_path.exists():
            return []
        rows: List[ResearchDecisionRecord] = []
        for line in self.research_decisions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(ResearchDecisionRecord.model_validate_json(line))
        return rows

    def latest_research_decision(
        self,
        *,
        mode: Optional[str] = None,
        trial_id: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> Optional[ResearchDecisionRecord]:
        rows = list(self.iter_research_decisions())
        for record in reversed(rows):
            if mode is not None and record.mode != mode:
                continue
            if trial_id is not None and record.trial_id != trial_id:
                continue
            if problem_id is not None and record.problem_id != problem_id:
                continue
            return record
        return None

    def append_problem_study(self, report: ProblemStudyReport) -> None:
        with self.problem_studies_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report.model_dump(mode="json")) + "\n")

    def iter_problem_studies(self) -> Iterable[ProblemStudyReport]:
        if not self.problem_studies_path.exists():
            return []
        rows: List[ProblemStudyReport] = []
        for line in self.problem_studies_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                environment = payload.get("environment")
                if isinstance(environment, dict) and "execution_report_path" not in environment:
                    study_plan_path = environment.get("study_plan_path")
                    if study_plan_path:
                        environment["execution_report_path"] = str(
                            Path(study_plan_path).with_name("execution_report.json")
                        )
                hypotheses = payload.get("hypotheses")
                if isinstance(hypotheses, list) and hypotheses and all(isinstance(item, str) for item in hypotheses):
                    evidence_bundle = payload.get("evidence_bundle") or {}
                    evidence_bundle_id = (
                        evidence_bundle.get("bundle_id")
                        if isinstance(evidence_bundle, dict) and evidence_bundle.get("bundle_id")
                        else f"legacy-evidence-{payload.get('problem_id', 'unknown')}"
                    )
                    payload["hypotheses"] = [
                        {
                            "hypothesis_id": f"legacy-hypothesis-{idx}",
                            "problem": payload.get("problem", ""),
                            "hypothesis": text,
                            "rationale": "Migrated from legacy problem-study record.",
                            "confidence": float(payload.get("resolution_confidence", 0.0) or 0.0),
                            "evidence_bundle_id": evidence_bundle_id,
                            "supporting_document_ids": list(payload.get("cited_research_ids", [])),
                            "supporting_claim_ids": [],
                            "contradiction_review_id": None,
                            "proposed_benchmark_ids": list(payload.get("benchmark_ids", [])),
                            "unresolved_assumptions": [],
                        }
                        for idx, text in enumerate(hypotheses, start=1)
                    ]
                rows.append(ProblemStudyReport.model_validate(payload))
        return rows

    def latest_problem_study(self, problem_id: Optional[str] = None) -> Optional[ProblemStudyReport]:
        rows = list(self.iter_problem_studies())
        for report in reversed(rows):
            if problem_id is None or report.problem_id == problem_id:
                return report
        return None

    def append_problem_execution(self, report: ProblemExecutionReport) -> None:
        with self.problem_executions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report.model_dump(mode="json")) + "\n")

    def write_run_manifest(self, manifest: RunManifest) -> Path:
        path = self.manifests_dir / f"{manifest.manifest_id}.json"
        self._atomic_write_json(path, manifest.model_dump(mode="json"))
        return path

    def load_run_manifest(self, manifest_id: str) -> Optional[RunManifest]:
        path = self.manifests_dir / f"{manifest_id}.json"
        if not path.exists():
            return None
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def load_run_manifest_path(self, path: str) -> Optional[RunManifest]:
        manifest_path = Path(path)
        if not manifest_path.exists():
            return None
        return RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    def write_image_manifest(self, manifest: ImageManifest) -> Path:
        path = self.manifests_dir / f"image-{manifest.hash_sha256[:16]}.json"
        self._atomic_write_json(path, manifest.model_dump(mode="json"))
        return path

    def load_memory_manifest(self) -> Optional[MemoryStoreManifest]:
        if not self.memory_manifest_path.exists():
            return None
        return MemoryStoreManifest.model_validate_json(self.memory_manifest_path.read_text(encoding="utf-8"))

    def save_memory_manifest(self, manifest: MemoryStoreManifest) -> Path:
        updated = manifest.model_copy(update={"updated_at": _utc_now()})
        self._atomic_write_json(self.memory_manifest_path, updated.model_dump(mode="json"))
        return self.memory_manifest_path

    def iter_problem_executions(self) -> Iterable[ProblemExecutionReport]:
        if not self.problem_executions_path.exists():
            return []
        rows: List[ProblemExecutionReport] = []
        for line in self.problem_executions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(ProblemExecutionReport.model_validate_json(line))
        return rows

    def latest_problem_execution(self, problem_id: Optional[str] = None) -> Optional[ProblemExecutionReport]:
        rows = list(self.iter_problem_executions())
        for report in reversed(rows):
            if problem_id is None or report.problem_id == problem_id:
                return report
        return None

    def load_problem_schedule(self) -> ProblemScheduleState:
        if not self.problem_schedule_path.exists():
            state = ProblemScheduleState()
            self.save_problem_schedule(state)
            return state
        return ProblemScheduleState.model_validate_json(self.problem_schedule_path.read_text(encoding="utf-8"))

    def save_problem_schedule(self, state: ProblemScheduleState) -> None:
        self._atomic_write_json(self.problem_schedule_path, state.model_dump(mode="json"))

    def append_problem_schedule(self, entry: ProblemScheduleEntry) -> None:
        state = self.load_problem_schedule()
        state.entries.append(entry)
        self.save_problem_schedule(state)

    def update_problem_schedule(self, schedule_id: str, **updates: Any) -> Optional[ProblemScheduleEntry]:
        state = self.load_problem_schedule()
        updated_entry: Optional[ProblemScheduleEntry] = None
        new_entries: List[ProblemScheduleEntry] = []
        for entry in state.entries:
            if entry.schedule_id == schedule_id:
                updated_entry = entry.model_copy(update=updates)
                new_entries.append(updated_entry)
            else:
                new_entries.append(entry)
        self.save_problem_schedule(ProblemScheduleState(entries=new_entries))
        return updated_entry

    def iter_problem_schedules(self) -> Iterable[ProblemScheduleEntry]:
        return self.load_problem_schedule().entries

    def get_problem_schedule(self, schedule_id: str) -> Optional[ProblemScheduleEntry]:
        for entry in self.iter_problem_schedules():
            if entry.schedule_id == schedule_id:
                return entry
        return None

    def load_research_projects(self) -> ResearchProjectState:
        if not self.research_projects_path.exists():
            state = ResearchProjectState()
            self.save_research_projects(state)
            return state
        return ResearchProjectState.model_validate_json(self.research_projects_path.read_text(encoding="utf-8"))

    def save_research_projects(self, state: ResearchProjectState) -> None:
        self._atomic_write_json(self.research_projects_path, state.model_dump(mode="json"))

    def upsert_research_project(self, project: ResearchProject) -> None:
        state = self.load_research_projects()
        entries = [item for item in state.entries if item.project_id != project.project_id]
        entries.append(project)
        entries.sort(key=lambda item: (item.updated_at, item.created_at, item.project_id))
        self.save_research_projects(ResearchProjectState(entries=entries))

    def list_research_projects(self) -> List[ResearchProject]:
        return self.load_research_projects().entries

    def get_research_project(self, project_id: str) -> Optional[ResearchProject]:
        for entry in self.list_research_projects():
            if entry.project_id == project_id:
                return entry
        return None

    def latest_research_project(self) -> Optional[ResearchProject]:
        entries = self.list_research_projects()
        if not entries:
            return None
        return sorted(entries, key=lambda item: (item.updated_at, item.created_at, item.project_id))[-1]

    def update_research_project(self, project_id: str, **updates: Any) -> Optional[ResearchProject]:
        state = self.load_research_projects()
        updated_project: Optional[ResearchProject] = None
        new_entries: List[ResearchProject] = []
        for entry in state.entries:
            if entry.project_id == project_id:
                updated_project = entry.model_copy(update=updates)
                new_entries.append(updated_project)
            else:
                new_entries.append(entry)
        self.save_research_projects(ResearchProjectState(entries=new_entries))
        return updated_project

    def load_research_portfolio(self) -> ResearchPortfolio:
        if not self.research_portfolio_path.exists():
            portfolio = ResearchPortfolio()
            self.save_research_portfolio(portfolio)
            return portfolio
        return ResearchPortfolio.model_validate_json(self.research_portfolio_path.read_text(encoding="utf-8"))

    def save_research_portfolio(self, portfolio: ResearchPortfolio) -> None:
        updated = portfolio.model_copy(update={"updated_at": _utc_now()})
        self._atomic_write_json(self.research_portfolio_path, updated.model_dump(mode="json"))

    def append_project_priority_record(self, record: ProjectPriorityRecord) -> None:
        with self.project_priority_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")

    def iter_project_priority_records(self) -> Iterable[ProjectPriorityRecord]:
        if not self.project_priority_records_path.exists():
            return []
        rows: List[ProjectPriorityRecord] = []
        for line in self.project_priority_records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(ProjectPriorityRecord.model_validate_json(line))
        return rows

    def latest_project_priority_record(self, project_id: Optional[str] = None) -> Optional[ProjectPriorityRecord]:
        rows = list(self.iter_project_priority_records())
        for record in reversed(rows):
            if project_id is not None and record.project_id != project_id:
                continue
            return record
        return None

    def append_evidence_debt_record(self, record: EvidenceDebtRecord) -> None:
        with self.evidence_debt_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")

    def iter_evidence_debt_records(self) -> Iterable[EvidenceDebtRecord]:
        if not self.evidence_debt_records_path.exists():
            return []
        rows: List[EvidenceDebtRecord] = []
        for line in self.evidence_debt_records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(EvidenceDebtRecord.model_validate_json(line))
        return rows

    def latest_evidence_debt_record(self, project_id: Optional[str] = None) -> Optional[EvidenceDebtRecord]:
        rows = list(self.iter_evidence_debt_records())
        for record in reversed(rows):
            if project_id is not None and record.project_id != project_id:
                continue
            return record
        return None

    def append_project_staleness_record(self, record: ProjectStalenessRecord) -> None:
        with self.project_staleness_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")

    def iter_project_staleness_records(self) -> Iterable[ProjectStalenessRecord]:
        if not self.project_staleness_records_path.exists():
            return []
        rows: List[ProjectStalenessRecord] = []
        for line in self.project_staleness_records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(ProjectStalenessRecord.model_validate_json(line))
        return rows

    def latest_project_staleness_record(self, project_id: Optional[str] = None) -> Optional[ProjectStalenessRecord]:
        rows = list(self.iter_project_staleness_records())
        for record in reversed(rows):
            if project_id is not None and record.project_id != project_id:
                continue
            return record
        return None

    def append_portfolio_decision(self, decision: PortfolioDecision) -> None:
        with self.portfolio_decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision.model_dump(mode="json")) + "\n")

    def iter_portfolio_decisions(self) -> Iterable[PortfolioDecision]:
        if not self.portfolio_decisions_path.exists():
            return []
        rows: List[PortfolioDecision] = []
        for line in self.portfolio_decisions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(PortfolioDecision.model_validate_json(line))
        return rows

    def latest_portfolio_decision(self) -> Optional[PortfolioDecision]:
        rows = list(self.iter_portfolio_decisions())
        if not rows:
            return None
        return rows[-1]

    def publication_handoff_path(self, package_id: str) -> Path:
        return self.publication_handoffs_dir / f"{package_id}.json"

    def save_publication_handoff(self, package: PublicationHandoffPackage) -> Path:
        path = self.publication_handoff_path(package.package_id)
        payload = package.model_copy(update={"artifact_path": str(path)})
        self._atomic_write_json(path, payload.model_dump(mode="json"))
        return path

    def append_publication_handoff(self, package: PublicationHandoffPackage) -> None:
        with self.publication_handoffs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(package.model_dump(mode="json")) + "\n")

    def iter_publication_handoffs(self) -> Iterable[PublicationHandoffPackage]:
        if not self.publication_handoffs_path.exists():
            return []
        rows: List[PublicationHandoffPackage] = []
        for line in self.publication_handoffs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(PublicationHandoffPackage.model_validate_json(line))
        return rows

    def latest_publication_handoff(self, project_id: Optional[str] = None) -> Optional[PublicationHandoffPackage]:
        rows = list(self.iter_publication_handoffs())
        for package in reversed(rows):
            if project_id is not None and package.project_id != project_id:
                continue
            return package
        return None

    def append_priority_snapshot(self, snapshot: PortfolioPrioritySnapshot) -> None:
        with self.priority_snapshots_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot.model_dump(mode="json")) + "\n")

    def iter_priority_snapshots(self) -> Iterable[PortfolioPrioritySnapshot]:
        if not self.priority_snapshots_path.exists():
            return []
        rows: List[PortfolioPrioritySnapshot] = []
        for line in self.priority_snapshots_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(PortfolioPrioritySnapshot.model_validate_json(line))
        return rows

    def latest_priority_snapshot(self, project_id: Optional[str] = None) -> Optional[PortfolioPrioritySnapshot]:
        rows = list(self.iter_priority_snapshots())
        for snapshot in reversed(rows):
            if project_id is not None and snapshot.project_id != project_id:
                continue
            return snapshot
        return None

    def append_budget_allocation(self, decision: BudgetAllocationDecision) -> None:
        with self.budget_allocations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision.model_dump(mode="json")) + "\n")

    def iter_budget_allocations(self) -> Iterable[BudgetAllocationDecision]:
        if not self.budget_allocations_path.exists():
            return []
        rows: List[BudgetAllocationDecision] = []
        for line in self.budget_allocations_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(BudgetAllocationDecision.model_validate_json(line))
        return rows

    def latest_budget_allocation(self) -> Optional[BudgetAllocationDecision]:
        rows = list(self.iter_budget_allocations())
        if not rows:
            return None
        return rows[-1]

    def append_falsification_plan(self, plan: FalsificationPlan) -> None:
        with self.falsification_plans_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(plan.model_dump(mode="json")) + "\n")

    def iter_falsification_plans(self) -> Iterable[FalsificationPlan]:
        if not self.falsification_plans_path.exists():
            return []
        rows: List[FalsificationPlan] = []
        for line in self.falsification_plans_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(FalsificationPlan.model_validate_json(line))
        return rows

    def latest_falsification_plan(
        self,
        *,
        project_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[FalsificationPlan]:
        rows = list(self.iter_falsification_plans())
        for plan in reversed(rows):
            if project_id is not None and plan.project_id != project_id:
                continue
            if thread_id is not None and plan.thread_id != thread_id:
                continue
            return plan
        return None

    def load_endpoint_registry(self) -> EndpointRegistryState:
        if not self.endpoint_registry_path.exists():
            state = EndpointRegistryState()
            self.save_endpoint_registry(state)
            return state
        return EndpointRegistryState.model_validate_json(self.endpoint_registry_path.read_text(encoding="utf-8"))

    def load_checkpoint_registry(self) -> CheckpointRegistryState:
        if not self.checkpoint_registry_path.exists():
            state = CheckpointRegistryState()
            self.save_checkpoint_registry(state)
            return state
        return CheckpointRegistryState.model_validate_json(self.checkpoint_registry_path.read_text(encoding="utf-8"))

    def experiment_backend_state_path(self, trial_name: str, backend_id: str) -> Path:
        return self.experiment_backends_dir / f"{trial_name}__{backend_id}.json"

    def save_experiment_backend_runtime(self, record: ExperimentBackendRuntimeRecord) -> Path:
        updated = record.model_copy(update={"updated_at": _utc_now()})
        path = self.experiment_backend_state_path(updated.trial_name, updated.backend_id)
        self._atomic_write_json(path, updated.model_dump(mode="json"))
        return path

    def load_experiment_backend_runtime(
        self,
        trial_name: str,
        backend_id: str,
    ) -> Optional[ExperimentBackendRuntimeRecord]:
        path = self.experiment_backend_state_path(trial_name, backend_id)
        if not path.exists():
            return None
        return ExperimentBackendRuntimeRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_experiment_backend_runtimes(self) -> List[ExperimentBackendRuntimeRecord]:
        rows: List[ExperimentBackendRuntimeRecord] = []
        for path in sorted(self.experiment_backends_dir.glob("*.json")):
            rows.append(ExperimentBackendRuntimeRecord.model_validate_json(path.read_text(encoding="utf-8")))
        return rows

    def save_checkpoint_registry(self, state: CheckpointRegistryState) -> None:
        self._atomic_write_json(self.checkpoint_registry_path, state.model_dump(mode="json"))

    def upsert_checkpoint(self, record: CheckpointRecord) -> None:
        state = self.load_checkpoint_registry()
        entries = [item for item in state.entries if item.name != record.name]
        entries.append(record)
        self.save_checkpoint_registry(CheckpointRegistryState(entries=entries))

    def list_checkpoints(self) -> List[CheckpointRecord]:
        return self.load_checkpoint_registry().entries

    def get_checkpoint(self, checkpoint_name: str) -> Optional[CheckpointRecord]:
        for entry in self.list_checkpoints():
            if entry.name == checkpoint_name:
                return entry
        return None

    def save_endpoint_registry(self, state: EndpointRegistryState) -> None:
        self._atomic_write_json(self.endpoint_registry_path, state.model_dump(mode="json"))

    def upsert_endpoint(self, record: EndpointRecord) -> None:
        state = self.load_endpoint_registry()
        entries = [item for item in state.entries if item.endpoint_name != record.endpoint_name]
        entries.append(record)
        self.save_endpoint_registry(EndpointRegistryState(entries=entries))

    def list_endpoints(self) -> List[EndpointRecord]:
        return self.load_endpoint_registry().entries

    def get_endpoint(self, endpoint_name: str) -> Optional[EndpointRecord]:
        for entry in self.list_endpoints():
            if entry.endpoint_name == endpoint_name:
                return entry
        return None

    def endpoint_runtime_dir(self, endpoint_name: str) -> Path:
        path = self.endpoints_dir / endpoint_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def endpoint_manifest_path(self, endpoint_name: str) -> Path:
        return self.endpoint_runtime_dir(endpoint_name) / "endpoint_manifest.json"

    def endpoint_stdout_log_path(self, endpoint_name: str) -> Path:
        return self.endpoint_runtime_dir(endpoint_name) / "stdout.log"

    def endpoint_stderr_log_path(self, endpoint_name: str) -> Path:
        return self.endpoint_runtime_dir(endpoint_name) / "stderr.log"

    def load_role_assignments(self) -> RoleAssignmentState:
        if not self.role_assignments_path.exists():
            state = RoleAssignmentState()
            self.save_role_assignments(state)
            return state
        return RoleAssignmentState.model_validate_json(self.role_assignments_path.read_text(encoding="utf-8"))

    def save_role_assignments(self, state: RoleAssignmentState) -> None:
        self._atomic_write_json(self.role_assignments_path, state.model_dump(mode="json"))

    def upsert_role_assignment(self, record: RoleAssignment) -> None:
        state = self.load_role_assignments()
        entries = [item for item in state.entries if item.role != record.role]
        entries.append(record)
        self.save_role_assignments(RoleAssignmentState(entries=entries))

    def list_role_assignments(self) -> List[RoleAssignment]:
        return self.load_role_assignments().entries

    def load_operator_serving_state(self) -> OperatorServingState:
        if not self.operator_serving_path.exists():
            state = OperatorServingState()
            self.save_operator_serving_state(state)
            return state
        return OperatorServingState.model_validate_json(self.operator_serving_path.read_text(encoding="utf-8"))

    def save_operator_serving_state(self, state: OperatorServingState) -> None:
        self._atomic_write_json(self.operator_serving_path, state.model_dump(mode="json"))

    def append_alert(self, alert: AlertRecord) -> None:
        with self.alerts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert.model_dump(mode="json")) + "\n")

    def iter_alerts(self) -> Iterable[AlertRecord]:
        if not self.alerts_path.exists():
            return []
        rows: List[AlertRecord] = []
        for line in self.alerts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(AlertRecord.model_validate_json(line))
        return rows

    def latest_alerts(self, count: int = 20) -> List[AlertRecord]:
        rows = list(self.iter_alerts())
        return rows[-count:]

    def append_audit_event(self, source: str, action: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": _utc_now(),
            "source": source,
            "action": action,
            "payload": payload,
        }
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def iter_audit_events(self) -> Iterable[Dict[str, Any]]:
        if not self.audit_log_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in self.audit_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def latest_audit_events(self, count: int = 20) -> List[Dict[str, Any]]:
        rows = list(self.iter_audit_events())
        return rows[-count:]

    def write_policy_bundle(
        self,
        policy: DirectorPolicy,
        plan: StrategistPlan,
        task: ScoutTask,
    ) -> None:
        self._atomic_write_json(
            self.policies_dir / f"{policy.trial_id}_director_policy.json",
            policy.model_dump(mode="json"),
        )
        self._atomic_write_json(
            self.policies_dir / f"{plan.trial_id}_strategist_plan.json",
            plan.model_dump(mode="json"),
        )
        self._atomic_write_json(
            self.policies_dir / f"{task.trial_id}_scout_task.json",
            task.model_dump(mode="json"),
        )

    def write_payload_config(self, payload: TrainingPayloadConfig) -> Path:
        path = self.workspace / "tar_runs" / payload.trial_id / "config.json"
        self._atomic_write_json(path, payload.model_dump(mode="json"))
        return path

    def load_payload_config(self, trial_id: str) -> Optional[TrainingPayloadConfig]:
        path = self.workspace / "tar_runs" / trial_id / "config.json"
        if not path.exists():
            return None
        return TrainingPayloadConfig.model_validate_json(path.read_text(encoding="utf-8"))

    def dataset_stream_dir(self, stream_name: str) -> Path:
        path = self.data_dir / stream_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dataset_manifest_path(self, stream_name: str) -> Path:
        return self.dataset_stream_dir(stream_name) / "manifest.json"

    def save_dataset_manifest(self, manifest: DatasetManifest) -> Path:
        path = self.dataset_manifest_path(manifest.stream_name)
        self._atomic_write_json(path, manifest.model_dump(mode="json"))
        return path

    def load_dataset_manifest(self, stream_name: str) -> Optional[DatasetManifest]:
        path = self.dataset_manifest_path(stream_name)
        if not path.exists():
            return None
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def next_trial_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"trial-{stamp}"

    def status_payload(self) -> Dict[str, Any]:
        recovery = self.load_recovery()
        graph = self.load_knowledge_graph()
        recent = self.tail_metrics(3)
        latest_breakthrough = self.latest_breakthrough_report()
        latest_claim_verdict = self.latest_claim_verdict()
        latest_research_decision = self.latest_research_decision()
        latest_problem_study = self.latest_problem_study()
        latest_problem_execution = self.latest_problem_execution()
        latest_research_project = self.latest_research_project()
        latest_priority_snapshot = self.latest_priority_snapshot()
        latest_budget_allocation = self.latest_budget_allocation()
        latest_falsification_plan = self.latest_falsification_plan()
        latest_portfolio = self.load_research_portfolio()
        latest_portfolio_decision = self.latest_portfolio_decision()
        latest_evidence_debt = self.latest_evidence_debt_record()
        latest_project_staleness = self.latest_project_staleness_record()
        latest_publication_handoff = self.latest_publication_handoff()
        memory_manifest = self.load_memory_manifest()
        schedules = list(self.iter_problem_schedules())
        research_projects = self.list_research_projects()
        alerts = list(self.iter_alerts())
        experiment_backends = self.list_experiment_backend_runtimes()
        return {
            "recovery": recovery.model_dump(mode="json"),
            "knowledge_graph_entries": len(graph.entries),
            "last_three_metrics": [item.model_dump(mode="json") for item in recent],
            "research_documents": len(list(self.iter_research_documents())),
            "verification_reports": len(list(self.iter_verification_reports())),
            "breakthrough_reports": len(list(self.iter_breakthrough_reports())),
            "latest_breakthrough_report": latest_breakthrough.model_dump(mode="json") if latest_breakthrough else None,
            "claim_verdicts": len(list(self.iter_claim_verdicts())),
            "latest_claim_verdict": latest_claim_verdict.model_dump(mode="json") if latest_claim_verdict else None,
            "research_decisions": len(list(self.iter_research_decisions())),
            "latest_research_decision": latest_research_decision.model_dump(mode="json") if latest_research_decision else None,
            "problem_studies": len(list(self.iter_problem_studies())),
            "latest_problem_study": latest_problem_study.model_dump(mode="json") if latest_problem_study else None,
            "problem_executions": len(list(self.iter_problem_executions())),
            "latest_problem_execution": latest_problem_execution.model_dump(mode="json") if latest_problem_execution else None,
            "research_projects": len(research_projects),
            "active_research_projects": len([item for item in research_projects if item.status == "active"]),
            "latest_research_project": latest_research_project.model_dump(mode="json") if latest_research_project else None,
            "prioritization_snapshots": len(list(self.iter_priority_snapshots())),
            "latest_priority_snapshot": latest_priority_snapshot.model_dump(mode="json") if latest_priority_snapshot else None,
            "budget_allocation_decisions": len(list(self.iter_budget_allocations())),
            "latest_budget_allocation": latest_budget_allocation.model_dump(mode="json") if latest_budget_allocation else None,
            "falsification_plans": len(list(self.iter_falsification_plans())),
            "latest_falsification_plan": latest_falsification_plan.model_dump(mode="json") if latest_falsification_plan else None,
            "latest_portfolio": latest_portfolio.model_dump(mode="json"),
            "portfolio_decisions": len(list(self.iter_portfolio_decisions())),
            "latest_portfolio_decision": latest_portfolio_decision.model_dump(mode="json") if latest_portfolio_decision else None,
            "evidence_debt_records": len(list(self.iter_evidence_debt_records())),
            "latest_evidence_debt_record": latest_evidence_debt.model_dump(mode="json") if latest_evidence_debt else None,
            "project_staleness_records": len(list(self.iter_project_staleness_records())),
            "latest_project_staleness_record": latest_project_staleness.model_dump(mode="json") if latest_project_staleness else None,
            "publication_handoffs": len(list(self.iter_publication_handoffs())),
            "latest_publication_handoff": latest_publication_handoff.model_dump(mode="json") if latest_publication_handoff else None,
            "memory_manifest": memory_manifest.model_dump(mode="json") if memory_manifest else None,
            "problem_schedules": len(schedules),
            "active_problem_schedules": len([item for item in schedules if item.status in {"scheduled", "leased", "running", "retry_wait"}]),
            "latest_problem_schedule": schedules[-1].model_dump(mode="json") if schedules else None,
            "experiment_backend_runs": len(experiment_backends),
            "latest_experiment_backend_run": experiment_backends[-1].model_dump(mode="json") if experiment_backends else None,
            "endpoints": len(self.list_endpoints()),
            "role_assignments": [item.model_dump(mode="json") for item in self.list_role_assignments()],
            "alerts": len(alerts),
            "latest_alerts": [item.model_dump(mode="json") for item in alerts[-5:]],
        }
