from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from tar_lab.schemas import (
    AgendaDecisionKind,
    AgendaDecisionRecord,
    AgendaReviewConfig,
    AgendaReviewRecord,
    AgendaSnapshot,
    TrainingSignalRecord,
)

if TYPE_CHECKING:
    from tar_lab.orchestrator import TAROrchestrator
    from tar_lab.schemas import FrontierGapRecord, ResearchProject


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_deadline(hours: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).replace(microsecond=0).isoformat()


class AgendaEngine:
    """
    Autonomous agenda engine built on top of the existing WS36/WS38 interfaces.
    """

    REVIEWS_PATH = "tar_state/agenda/reviews"
    DECISIONS_PATH = "tar_state/agenda/decisions"
    CONFIG_PATH = "tar_state/agenda/config.json"

    def __init__(
        self,
        workspace_root: str,
        orchestrator: "TAROrchestrator",
        config: Optional[AgendaReviewConfig] = None,
    ) -> None:
        self._workspace = Path(workspace_root).resolve()
        self._orchestrator = orchestrator
        self._reviews_dir = self._workspace / self.REVIEWS_PATH
        self._decisions_dir = self._workspace / self.DECISIONS_PATH
        for directory in (self._reviews_dir, self._decisions_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._config = config or self._load_or_default_config()

    def run_agenda_review(self) -> AgendaReviewRecord:
        review_id = f"review-{uuid.uuid4().hex[:8]}"
        decisions: List[AgendaDecisionRecord] = []
        active_projects = self._get_active_projects()
        active_count = len(active_projects)

        stale_projects = self._identify_stale_projects(active_projects)
        for project in stale_projects:
            decisions.append(
                self._make_decision(
                    review_id=review_id,
                    kind="park_stale_project",
                    subject_id=project.project_id,
                    subject_title=project.title,
                    rationale=(
                        f"No progress for >{self._config.stale_project_hours}h. "
                        "Auto-parked by agenda engine."
                    ),
                )
            )
            active_count = max(0, active_count - 1)

        gap_candidates = self._get_pending_gaps()
        gaps_reviewed = 0
        promotions = 0
        cap_enforced = active_count >= self._config.max_active_projects

        for gap in gap_candidates:
            gaps_reviewed += 1
            if promotions >= self._config.max_promotions_per_review:
                break
            if active_count >= self._config.max_active_projects:
                decisions.append(
                    self._make_decision(
                        review_id=review_id,
                        kind="cap_enforced",
                        subject_id=gap.gap_id,
                        subject_title=gap.description,
                        rationale=(
                            f"Active project cap ({self._config.max_active_projects}) "
                            "reached. Gap deferred."
                        ),
                    )
                )
                cap_enforced = True
                continue
            if gap.novelty_score < self._config.min_gap_novelty_to_promote:
                decisions.append(
                    self._make_decision(
                        review_id=review_id,
                        kind="defer_gap",
                        subject_id=gap.gap_id,
                        subject_title=gap.description,
                        rationale=(
                            f"novelty_score={gap.novelty_score:.3f} below threshold "
                            f"{self._config.min_gap_novelty_to_promote}"
                        ),
                    )
                )
                continue
            decisions.append(
                self._make_decision(
                    review_id=review_id,
                    kind="promote_gap_project",
                    subject_id=gap.gap_id,
                    subject_title=gap.description,
                    rationale=(
                        f"novelty_score={gap.novelty_score:.3f} meets threshold. "
                        "Promoting to active project pending veto window."
                    ),
                )
            )
            active_count += 1
            promotions += 1

        if not decisions:
            decisions.append(
                self._make_decision(
                    review_id=review_id,
                    kind="no_action",
                    subject_id="none",
                    subject_title="none",
                    rationale="No actionable gaps or stale projects found.",
                )
            )

        review = AgendaReviewRecord(
            review_id=review_id,
            active_project_count=len(active_projects),
            gap_candidates_reviewed=gaps_reviewed,
            decisions=decisions,
            cap_enforced=cap_enforced,
            completed_at=utc_now_iso(),
        )
        self._persist_review(review)
        for decision in decisions:
            self._persist_decision(decision)
        return review

    def veto_agenda_decision(self, decision_id: str, reason: str) -> AgendaDecisionRecord:
        decision = self._load_decision(decision_id)
        if decision.status != "pending_veto":
            raise ValueError(
                f"Decision {decision_id} is not pending veto (status={decision.status})"
            )
        updated = decision.model_copy(
            update={
                "status": "vetoed",
                "vetoed_at": utc_now_iso(),
                "veto_reason": reason,
            }
        )
        self._persist_decision(updated)
        return updated

    def commit_pending_decisions(self) -> List[AgendaDecisionRecord]:
        committed: List[AgendaDecisionRecord] = []
        now = datetime.now(timezone.utc)
        for decision in self._list_decisions(status="pending_veto"):
            deadline = datetime.fromisoformat(decision.veto_deadline)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if now < deadline:
                continue
            self._execute_decision(decision)
            updated = decision.model_copy(
                update={
                    "status": "committed",
                    "committed_at": utc_now_iso(),
                }
            )
            if self._config.recycle_decisions_to_training_signal:
                updated = updated.model_copy(
                    update={"recycled_to_signal_id": self._recycle_to_training_signal(updated)}
                )
            self._persist_decision(updated)
            committed.append(updated)
        return committed

    def get_snapshot(self) -> AgendaSnapshot:
        decisions = self._list_decisions()
        pending = [decision for decision in decisions if decision.status == "pending_veto"]
        committed = [decision for decision in decisions if decision.status == "committed"]
        vetoed = [decision for decision in decisions if decision.status == "vetoed"]
        latest_review_id = None
        reviews = sorted(self._reviews_dir.glob("*.json"))
        if reviews:
            latest_review = AgendaReviewRecord.model_validate_json(
                reviews[-1].read_text(encoding="utf-8")
            )
            latest_review_id = latest_review.review_id
        return AgendaSnapshot(
            active_project_count=len(self._get_active_projects()),
            pending_veto_count=len(pending),
            committed_this_session=len(committed),
            vetoed_this_session=len(vetoed),
            latest_review_id=latest_review_id,
            config=self._config,
        )

    def update_config(self, config: AgendaReviewConfig) -> None:
        self._config = config
        config_path = self._workspace / self.CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    def _make_decision(
        self,
        review_id: str,
        kind: AgendaDecisionKind,
        subject_id: str,
        subject_title: str,
        rationale: str,
    ) -> AgendaDecisionRecord:
        return AgendaDecisionRecord(
            decision_id=f"adec-{uuid.uuid4().hex[:8]}",
            review_id=review_id,
            kind=kind,
            subject_id=subject_id,
            subject_title=subject_title,
            rationale=rationale,
            veto_deadline=utc_deadline(self._config.veto_window_hours),
        )

    def _execute_decision(self, decision: AgendaDecisionRecord) -> None:
        if decision.kind == "promote_gap_project":
            gap = self._orchestrator.store.get_frontier_gap(decision.subject_id)
            if gap is None:
                return
            if not gap.proposed_project_id:
                project = self._orchestrator.create_project(gap.description, status="proposed")
                self._orchestrator.store.update_frontier_gap(
                    gap.gap_id,
                    status="proposed",
                    proposed_project_id=project.project_id,
                    review_note="agenda engine proposed this project automatically",
                    reviewed_at=utc_now_iso(),
                )
            self._orchestrator.promote_gap_project(
                gap.gap_id,
                note="Autonomous agenda promotion after veto window.",
            )
        elif decision.kind == "park_stale_project":
            self._orchestrator.pause_project(
                decision.subject_id,
                reason="superseded_by_better_thread",
                note=decision.rationale,
            )

    def _recycle_to_training_signal(self, decision: AgendaDecisionRecord) -> Optional[str]:
        try:
            from tar_lab.self_improvement import SelfImprovementEngine

            engine = SelfImprovementEngine(str(self._workspace))
            signal = TrainingSignalRecord(
                signal_id=f"sig-{uuid.uuid4().hex[:8]}",
                kind="research_decision",
                source_id=decision.decision_id,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Agenda decision: {decision.kind} on '{decision.subject_title}'. "
                            f"Rationale: {decision.rationale}"
                        ),
                    }
                ],
                gold_response=decision.rationale,
                quality_score=0.7,
                overclaim_present=False,
            )
            if engine.curate_signal(signal):
                return signal.signal_id
            return None
        except Exception:
            return None

    def _get_active_projects(self) -> List["ResearchProject"]:
        return [
            project
            for project in self._orchestrator.store.list_research_projects()
            if project.status == "active"
        ]

    def _get_pending_gaps(self) -> List["FrontierGapRecord"]:
        gaps = [
            gap
            for gap in self._orchestrator.store.iter_frontier_gaps()
            if gap.status in {"identified", "proposed"}
        ]
        return sorted(
            gaps,
            key=lambda item: (item.novelty_score, item.confidence, item.created_at, item.gap_id),
            reverse=True,
        )

    def _identify_stale_projects(self, active_projects: List["ResearchProject"]) -> List["ResearchProject"]:
        stale: List["ResearchProject"] = []
        now = datetime.now(timezone.utc)
        threshold = timedelta(hours=self._config.stale_project_hours)
        for project in active_projects:
            updated = datetime.fromisoformat(project.updated_at)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            if (now - updated) > threshold:
                stale.append(project)
        return stale

    def _persist_review(self, review: AgendaReviewRecord) -> None:
        path = self._reviews_dir / f"{review.review_id}.json"
        path.write_text(review.model_dump_json(indent=2), encoding="utf-8")

    def _persist_decision(self, decision: AgendaDecisionRecord) -> None:
        path = self._decisions_dir / f"{decision.decision_id}.json"
        path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")

    def _load_decision(self, decision_id: str) -> AgendaDecisionRecord:
        path = self._decisions_dir / f"{decision_id}.json"
        if not path.exists():
            raise ValueError(f"Decision not found: {decision_id}")
        return AgendaDecisionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _list_decisions(self, status: Optional[str] = None) -> List[AgendaDecisionRecord]:
        decisions: List[AgendaDecisionRecord] = []
        for path in sorted(self._decisions_dir.glob("*.json")):
            decision = AgendaDecisionRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if status is None or decision.status == status:
                decisions.append(decision)
        return decisions

    def _load_or_default_config(self) -> AgendaReviewConfig:
        config_path = self._workspace / self.CONFIG_PATH
        if config_path.exists():
            return AgendaReviewConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        return AgendaReviewConfig()
