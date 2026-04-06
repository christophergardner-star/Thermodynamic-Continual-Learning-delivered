from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tar_lab.schemas import (
    BreakthroughReport,
    DatasetManifest,
    DirectorPolicy,
    GovernorMetrics,
    KnowledgeGraphEntry,
    KnowledgeGraphState,
    ResearchDocument,
    RecoveryState,
    ScoutTask,
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
        self.metrics_log_path = self.logs_dir / "thermo_metrics.jsonl"
        self.audit_log_path = self.logs_dir / "activity_audit.log"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.policies_dir.mkdir(parents=True, exist_ok=True)

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
        return {
            "recovery": recovery.model_dump(mode="json"),
            "knowledge_graph_entries": len(graph.entries),
            "last_three_metrics": [item.model_dump(mode="json") for item in recent],
            "research_documents": len(list(self.iter_research_documents())),
            "verification_reports": len(list(self.iter_verification_reports())),
            "breakthrough_reports": len(list(self.iter_breakthrough_reports())),
            "latest_breakthrough_report": latest_breakthrough.model_dump(mode="json") if latest_breakthrough else None,
        }
