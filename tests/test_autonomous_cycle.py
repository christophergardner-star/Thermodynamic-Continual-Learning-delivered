import json
import tempfile
from pathlib import Path

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import AgendaReviewConfig, ResearchDocument, TrainingSignalRecord


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


def _seed_document(orchestrator: TAROrchestrator, document_id: str, title: str, statement: str) -> None:
    orchestrator.store.append_research_document(
        ResearchDocument(
            document_id=document_id,
            source_kind="manual",
            source_name="seeded",
            title=title,
            summary=title,
            url=f"https://example.com/{document_id}",
            problem_statements=[statement],
        )
    )


def test_full_autonomous_cycle_orchestration():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            pack_path, run_manifest_path = _write_anchor_pack(Path(tmp))
            orchestrator.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)

            _seed_document(
                orchestrator,
                "doc-rag-1",
                "RAG Drift Study A",
                "Investigate retrieval augmented generation drift in language model memory systems.",
            )
            _seed_document(
                orchestrator,
                "doc-rag-2",
                "RAG Drift Study B",
                "Analyze retrieval augmented generation drift in language model memory systems.",
            )

            scan = orchestrator.scan_frontier_gaps(topic="rag drift", max_gaps=5)
            assert scan.gaps_identified >= 1
            gap_id = scan.gaps[0].gap_id

            orchestrator.update_agenda_config(
                AgendaReviewConfig(veto_window_hours=0.0)
            )
            review = orchestrator.run_agenda_review()
            assert any(
                decision.kind == "promote_gap_project" and decision.subject_id == gap_id
                for decision in review.decisions
            )

            committed = orchestrator.commit_agenda_decisions()
            assert any(
                decision.kind == "promote_gap_project" and decision.subject_id == gap_id
                for decision in committed
            )

            stored_gap = orchestrator.store.get_frontier_gap(gap_id)
            assert stored_gap is not None
            assert stored_gap.status == "promoted"
            assert stored_gap.proposed_project_id is not None
            promoted_project = orchestrator.store.get_research_project(stored_gap.proposed_project_id)
            assert promoted_project is not None
            assert promoted_project.status in {"active", "proposed"}

            accepted = orchestrator.curate_training_signal(
                TrainingSignalRecord(
                    signal_id="signal-autonomous-cycle",
                    kind="research_decision",
                    source_id="autonomous-cycle-signal",
                    project_id=promoted_project.project_id,
                    messages=[{"role": "user", "content": "Agenda promoted the RAG drift gap."}],
                    gold_response="Proceed with the promoted RAG drift project.",
                    quality_score=0.8,
                    overclaim_present=False,
                )
            )
            assert accepted is True

            delta = orchestrator.assemble_curated_delta("")
            assert delta.signal_count >= 1

            review_again = orchestrator.run_agenda_review()
            assert all(
                not (
                    decision.kind == "promote_gap_project"
                    and decision.subject_id == gap_id
                )
                for decision in review_again.decisions
            )
        finally:
            orchestrator.shutdown()
