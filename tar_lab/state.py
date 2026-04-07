from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tar_lab.schemas import (
    AlertRecord,
    BreakthroughReport,
    ClaimVerdict,
    DatasetManifest,
    DirectorPolicy,
    EndpointRecord,
    EndpointRegistryState,
    ImageManifest,
    MemoryStoreManifest,
    GovernorMetrics,
    KnowledgeGraphEntry,
    KnowledgeGraphState,
    ProblemExecutionReport,
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
        self.endpoint_registry_path = self.state_dir / "inference_endpoints.json"
        self.endpoints_dir = self.state_dir / "endpoints"
        self.role_assignments_path = self.state_dir / "role_assignments.json"
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

    def load_endpoint_registry(self) -> EndpointRegistryState:
        if not self.endpoint_registry_path.exists():
            state = EndpointRegistryState()
            self.save_endpoint_registry(state)
            return state
        return EndpointRegistryState.model_validate_json(self.endpoint_registry_path.read_text(encoding="utf-8"))

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
        memory_manifest = self.load_memory_manifest()
        schedules = list(self.iter_problem_schedules())
        alerts = list(self.iter_alerts())
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
            "memory_manifest": memory_manifest.model_dump(mode="json") if memory_manifest else None,
            "problem_schedules": len(schedules),
            "active_problem_schedules": len([item for item in schedules if item.status in {"scheduled", "leased", "running", "retry_wait"}]),
            "latest_problem_schedule": schedules[-1].model_dump(mode="json") if schedules else None,
            "endpoints": len(self.list_endpoints()),
            "role_assignments": [item.model_dump(mode="json") for item in self.list_role_assignments()],
            "alerts": len(alerts),
            "latest_alerts": [item.model_dump(mode="json") for item in alerts[-5:]],
        }
