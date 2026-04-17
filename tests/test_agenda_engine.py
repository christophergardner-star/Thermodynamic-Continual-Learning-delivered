import json
import tempfile
from pathlib import Path

from tar_lab.agenda import AgendaEngine
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import AgendaReviewConfig, FrontierGapRecord, ResearchProject


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_science_profiles(tmp: str) -> None:
    source_dir = REPO_ROOT / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_anchor_pack(workspace: Path) -> tuple[str, str]:
    pack_dir = workspace / "eval_artifacts" / "external_validation"
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "run_manifest.json").write_text(
        json.dumps({"pack_manifest_sha256": "sealed-pack-hash", "item_count": 2}, indent=2),
        encoding="utf-8",
    )
    (pack_dir / "predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"item_id": "anchor-item-001"}),
                json.dumps({"item_id": "anchor-item-002"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return "eval_artifacts/external_validation", str(pack_dir / "run_manifest.json")


def _seed_gap(
    orchestrator: TAROrchestrator,
    *,
    gap_id: str,
    description: str,
    novelty_score: float,
    status: str = "identified",
) -> None:
    orchestrator.store.append_frontier_gap(
        FrontierGapRecord(
            gap_id=gap_id,
            description=description,
            domain_profile="natural_language_processing",
            evidence_count=2,
            source_document_ids=[f"{gap_id}-doc-a", f"{gap_id}-doc-b"],
            novelty_score=novelty_score,
            similarity_to_existing=max(0.0, 1.0 - novelty_score),
            confidence=0.8,
            status=status,
        )
    )


def test_agenda_review_produces_record():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            review = AgendaEngine(tmp, orchestrator).run_agenda_review()
            assert review.review_id
            assert len(review.decisions) >= 1
            assert review.decisions[0].kind == "no_action"
        finally:
            orchestrator.shutdown()


def test_gap_below_novelty_threshold_deferred():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_gap(
                orchestrator,
                gap_id="gap-low",
                description="Investigate retrieval augmented generation drift in language model memory systems.",
                novelty_score=0.3,
            )
            review = AgendaEngine(tmp, orchestrator).run_agenda_review()
            assert any(
                decision.kind == "defer_gap" and decision.subject_id == "gap-low"
                for decision in review.decisions
            )
        finally:
            orchestrator.shutdown()


def test_gap_above_threshold_promoted():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_gap(
                orchestrator,
                gap_id="gap-high",
                description="Investigate retrieval augmented generation drift in language model memory systems.",
                novelty_score=0.7,
            )
            review = AgendaEngine(tmp, orchestrator).run_agenda_review()
            assert any(
                decision.kind == "promote_gap_project" and decision.subject_id == "gap-high"
                for decision in review.decisions
            )
        finally:
            orchestrator.shutdown()


def test_cap_enforced_when_at_max():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            for index in range(5):
                orchestrator.store.upsert_research_project(
                    ResearchProject(
                        project_id=f"project-{index}",
                        title=f"Project {index}",
                        goal=f"Goal {index}",
                        domain_profile="natural_language_processing",
                        status="active",
                    )
                )
            _seed_gap(
                orchestrator,
                gap_id="gap-cap",
                description="Investigate retrieval augmented generation drift in language model memory systems.",
                novelty_score=0.9,
            )
            review = AgendaEngine(
                tmp,
                orchestrator,
                config=AgendaReviewConfig(max_active_projects=5),
            ).run_agenda_review()
            assert any(
                decision.kind == "cap_enforced" and decision.subject_id == "gap-cap"
                for decision in review.decisions
            )
        finally:
            orchestrator.shutdown()


def test_veto_window_prevents_immediate_commit():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_gap(
                orchestrator,
                gap_id="gap-window",
                description="Investigate retrieval augmented generation drift in language model memory systems.",
                novelty_score=0.8,
            )
            engine = AgendaEngine(
                tmp,
                orchestrator,
                config=AgendaReviewConfig(veto_window_hours=24.0),
            )
            engine.run_agenda_review()
            committed = engine.commit_pending_decisions()
            assert committed == []
        finally:
            orchestrator.shutdown()


def test_veto_cancels_decision():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_gap(
                orchestrator,
                gap_id="gap-veto",
                description="Investigate retrieval augmented generation drift in language model memory systems.",
                novelty_score=0.8,
            )
            engine = AgendaEngine(tmp, orchestrator)
            review = engine.run_agenda_review()
            vetoed = engine.veto_agenda_decision(review.decisions[0].decision_id, "manual veto")
            assert vetoed.status == "vetoed"
        finally:
            orchestrator.shutdown()


def test_decision_recycled_to_training_signal():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            pack_path, run_manifest_path = _write_anchor_pack(Path(tmp))
            orchestrator.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)
            _seed_gap(
                orchestrator,
                gap_id="gap-recycle",
                description="Investigate retrieval augmented generation drift in language model memory systems.",
                novelty_score=0.8,
            )
            engine = AgendaEngine(
                tmp,
                orchestrator,
                config=AgendaReviewConfig(veto_window_hours=0.0),
            )
            engine.run_agenda_review()
            committed = engine.commit_pending_decisions()
            assert any(decision.recycled_to_signal_id for decision in committed)
        finally:
            orchestrator.shutdown()
