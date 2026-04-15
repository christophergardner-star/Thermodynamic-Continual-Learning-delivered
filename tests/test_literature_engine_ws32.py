import tempfile
from pathlib import Path

import dashboard
import tar_cli
from tar_lab.control import handle_request
from tar_lab.literature_engine import LiteratureEngine
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ControlRequest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_science_profiles(tmp: str) -> None:
    source_dir = REPO_ROOT / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_literature_paper(
    path: Path,
    *,
    title: str,
    claim_line: str,
    accuracy: str,
    figure_caption: str = "Figure 1 Drift remains stable under bounded drift [1].",
) -> Path:
    path.write_text(
        "\n".join(
            [
                title,
                "",
                "Abstract",
                "This paper studies bounded drift in thermodynamic learning systems.",
                "",
                "1 Results",
                claim_line,
                "",
                "Table 1 Benchmark Metrics",
                "Metric | Value",
                f"Accuracy | {accuracy}",
                "ECE | 0.05",
                "",
                figure_caption,
                "",
                "References",
                '[1] Gardner et al. 2026. "Stable Thermodynamics Under Bounded Drift".',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_ws32_manifest_history_and_content_hash_dedupe():
    with tempfile.TemporaryDirectory() as tmp:
        engine = LiteratureEngine(tmp)
        primary = _write_literature_paper(
            Path(tmp) / "paper-a.txt",
            title="Stable Thermodynamics",
            claim_line="We measure benchmark accuracy improves under bounded drift [1].",
            accuracy="0.84",
        )
        duplicate = Path(tmp) / "paper-a-copy.txt"
        duplicate.write_text(primary.read_text(encoding="utf-8"), encoding="utf-8")

        first = engine.ingest_paths([str(primary), str(duplicate)])

        assert first.ingested == 1
        assert first.deduplicated_existing == 1
        assert first.stored_total == 1
        assert first.manifest_id is not None
        assert first.latest_manifest is not None
        assert first.latest_manifest.manifest_id == first.manifest_id
        assert first.latest_manifest.artifact_count == 1
        assert first.latest_manifest.deduplicated_existing == 1
        assert len(first.latest_manifest.artifact_ids) == 1
        assert first.manifest_path is not None
        assert Path(first.manifest_path).exists()
        assert len(engine.store.load_paper_artifacts()) == 1
        assert first.artifacts[0].source_fingerprint is not None
        assert first.artifacts[0].source_fingerprint.source_hash_sha256

        second = engine.ingest_paths([str(primary)])

        assert second.ingested == 1
        assert second.deduplicated_existing == 1
        assert second.stored_total == 1
        assert second.manifest_id is not None
        assert second.manifest_id != first.manifest_id
        assert engine.latest_manifest() is not None
        assert engine.latest_manifest().manifest_id == second.manifest_id
        assert len(engine.store.list_literature_manifests()) == 2


def test_ws32_structured_metadata_and_conflicts_are_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        engine = LiteratureEngine(tmp)
        positive = _write_literature_paper(
            Path(tmp) / "positive.txt",
            title="Stable Thermodynamics",
            claim_line="We measure benchmark accuracy improves under bounded drift [1].",
            accuracy="0.84",
        )
        negative = _write_literature_paper(
            Path(tmp) / "negative.txt",
            title="Unstable Thermodynamics",
            claim_line="We measure benchmark accuracy does not improve under bounded drift [1].",
            accuracy="0.41",
            figure_caption="Figure 1 Drift becomes unstable under bounded drift [1].",
        )

        report = engine.ingest_paths([str(positive), str(negative)])

        assert report.ingested == 2
        assert report.stored_total == 2
        assert report.latest_manifest is not None
        assert report.latest_manifest.conflict_count >= 1
        assert report.conflicts

        artifact = report.artifacts[0]
        measured_claim = next(item for item in artifact.claims if item.label == "measured_result")
        assert any(section.text_hash for section in artifact.sections)
        assert all(section.word_count >= 0 for section in artifact.sections)
        assert measured_claim.citation_count >= 1
        assert "citation_missing" not in measured_claim.quality_flags

        assert artifact.tables
        table = artifact.tables[0]
        assert table.header == ["Metric", "Value"]
        assert table.row_count >= 3
        assert table.column_count == 2
        assert table.numeric_cell_count >= 2
        assert table.metric_hints
        assert table.related_claim_ids

        assert artifact.figures
        figure = artifact.figures[0]
        assert figure.figure_label == "Figure 1"
        assert figure.caption_hash
        assert figure.related_claim_ids

        conflicts = engine.load_conflicts()
        assert any(item.conflict_kind == "cross_paper_topic_polarity" for item in conflicts)
        assert all(item.shared_token_count > 0 for item in conflicts)

        status = engine.status()
        assert status["artifacts"] == 2
        assert status["conflicts"] >= 1
        assert status["tables"] >= 2
        assert status["figures"] >= 2
        assert status["manifests"] == 1
        assert status["latest_manifest"] is not None


def test_ws32_conflict_scope_guard_avoids_same_topic_false_positive():
    with tempfile.TemporaryDirectory() as tmp:
        engine = LiteratureEngine(tmp)
        left = _write_literature_paper(
            Path(tmp) / "scope-left.txt",
            title="Adapter Drift Study",
            claim_line=(
                "We measure calibration accuracy improves under bounded drift "
                "with adaptive anchors in transformer adapters [1]."
            ),
            accuracy="0.84",
        )
        right = _write_literature_paper(
            Path(tmp) / "scope-right.txt",
            title="Offline Decoder Study",
            claim_line=(
                "We measure calibration accuracy does not improve under offline calibration "
                "with diffusion decoders in vision pipelines [1]."
            ),
            accuracy="0.52",
        )

        report = engine.ingest_paths([str(left), str(right)])

        assert report.ingested == 2
        assert not any(item.conflict_kind == "cross_paper_topic_polarity" for item in report.conflicts)
        assert not engine.load_conflicts()


def test_ws32_operator_surfaces_expose_literature_state():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            positive = _write_literature_paper(
                Path(tmp) / "positive.txt",
                title="Stable Thermodynamics",
                claim_line="We measure benchmark accuracy improves under bounded drift [1].",
                accuracy="0.84",
            )
            negative = _write_literature_paper(
                Path(tmp) / "negative.txt",
                title="Unstable Thermodynamics",
                claim_line="We measure benchmark accuracy does not improve under bounded drift [1].",
                accuracy="0.41",
                figure_caption="Figure 1 Drift becomes unstable under bounded drift [1].",
            )

            report = orchestrator.ingest_papers([str(positive), str(negative)])
            paper_id = report.artifacts[0].paper_id

            literature_status = handle_request(orchestrator, ControlRequest(command="literature_status"))
            artifact_list = handle_request(
                orchestrator,
                ControlRequest(command="list_paper_artifacts", payload={"limit": 10}),
            )
            artifact_detail = handle_request(
                orchestrator,
                ControlRequest(command="paper_artifact", payload={"paper_id": paper_id}),
            )
            conflict_report = handle_request(
                orchestrator,
                ControlRequest(command="literature_conflicts", payload={"limit": 10}),
            )

            assert literature_status.ok is True
            assert artifact_list.ok is True
            assert artifact_detail.ok is True
            assert conflict_report.ok is True

            assert literature_status.payload["artifacts"] == 2
            assert literature_status.payload["manifests"] == 1
            assert artifact_list.payload["count"] == 2
            assert artifact_list.payload["latest_manifest"]["manifest_id"] == report.manifest_id
            assert artifact_detail.payload["artifact"]["paper_id"] == paper_id
            assert artifact_detail.payload["artifact"]["ingest_manifest_id"] == report.manifest_id
            assert conflict_report.payload["count"] >= 1

            frontier = orchestrator.frontier_status()
            assert frontier.literature_manifests == 1
            assert frontier.latest_literature_manifest is not None
            assert frontier.latest_literature_manifest.manifest_id == report.manifest_id

            context = dashboard.build_dashboard_context(orchestrator, max_results=10)
            assert context["literature_status"]["artifacts"] == 2
            assert context["literature_artifacts"]["count"] == 2
            assert context["literature_conflicts"]["count"] >= 1

            assert "Literature Artifacts: 2" in tar_cli._render_literature_status(literature_status.payload)
            assert "Paper Artifacts: 2" in tar_cli._render_paper_artifact_list(artifact_list.payload)
            assert f"Paper ID: {paper_id}" in tar_cli._render_paper_artifact(artifact_detail.payload)
            assert "Conflict Count:" in tar_cli._render_literature_conflicts(conflict_report.payload)
        finally:
            orchestrator.shutdown()
