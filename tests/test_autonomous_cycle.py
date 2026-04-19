import json
import tempfile
from pathlib import Path

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import AgendaReviewConfig, ResearchDocument, TrainingSignalRecord, ResearchProject


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


def test_run_full_research_cycle_no_human_input():
    """run_full_research_cycle() completes without raising and returns a valid summary.

    This test verifies that the autonomous loop can tick through a full cycle
    (ingest→scan→agenda→schedule→execute→family-check→self-improve) purely from
    code with no human intervention.  We run two cycles via serve_forever_full
    to confirm the cycle counter increments and throttle logic fires correctly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        _write_anchor_pack(Path(tmp))
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            # Seed two research documents so the frontier scanner has material
            _seed_document(
                orchestrator,
                "doc-auto-1",
                "Thermodynamic Continual Learning",
                "Investigate entropy-based regularisation for preventing catastrophic forgetting.",
            )
            _seed_document(
                orchestrator,
                "doc-auto-2",
                "Activation Space Dynamics",
                "Analyse activation covariance as a proxy for representation stability.",
            )

            # Run two cycles: cycle 0 triggers ingest (0 % 5 == 0), cycle 1 does not
            cycle_ref: list = [0]
            summary0 = orchestrator.run_full_research_cycle(
                ingest_every_n=5,
                self_improve_every_n=10,
                min_signals_for_improvement=99,  # won't fire; not enough signals
                _cycle_count_ref=cycle_ref,
            )
            summary1 = orchestrator.run_full_research_cycle(
                ingest_every_n=5,
                self_improve_every_n=10,
                min_signals_for_improvement=99,
                _cycle_count_ref=cycle_ref,
            )

            # Basic shape checks
            assert summary0["cycle"] == 0
            assert summary1["cycle"] == 1
            assert "started_at" in summary0
            assert "finished_at" in summary0

            # Cycle 0 attempted ingest (key present, even if 0 fetched due to no network)
            assert "ingest_fetched" in summary0 or "ingest_error" in summary0
            # Cycle 1 skipped ingest (throttled)
            assert "ingest_fetched" not in summary1 and "ingest_error" not in summary1

            # Frontier scan ran (gaps_identified key present regardless of count)
            assert "gaps_identified" in summary0

            # Agenda ran (keys present)
            assert "agenda_decisions_reviewed" in summary0
            assert "agenda_decisions_committed" in summary0

            # No unhandled exceptions — runtime/study errors captured, not raised
            assert "studies_scheduled" in summary0

            # serve_forever_full with 1 iteration returns a 1-element list
            summaries = orchestrator.serve_forever_full(
                poll_interval_s=0.0,
                iterations=1,
                ingest_every_n=5,
            )
            assert len(summaries) == 1
            assert summaries[0]["cycle"] == 0  # serve_forever_full creates its own counter

        finally:
            orchestrator.shutdown()
