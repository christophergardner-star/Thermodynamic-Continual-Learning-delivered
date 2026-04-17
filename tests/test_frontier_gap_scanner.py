import tempfile
from pathlib import Path

import dashboard
import tar_cli
from tar_lab.control import handle_request
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ControlRequest, ResearchDocument


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_science_profiles(tmp: str) -> None:
    source_dir = REPO_ROOT / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _seed_document(
    orchestrator: TAROrchestrator,
    document_id: str,
    *,
    title: str,
    problem_statements: list[str],
) -> None:
    orchestrator.store.append_research_document(
        ResearchDocument(
            document_id=document_id,
            source_kind="manual",
            source_name="seeded",
            title=title,
            summary=title,
            url=f"https://example.com/{document_id}",
            problem_statements=problem_statements,
        )
    )


def test_scan_frontier_gaps_proposes_and_promotes_project():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_document(
                orchestrator,
                "doc-rag-1",
                title="RAG Drift Study A",
                problem_statements=[
                    "Investigate retrieval augmented generation drift in language model memory systems.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-rag-2",
                title="RAG Drift Study B",
                problem_statements=[
                    "Analyze retrieval augmented generation drift in language model memory systems.",
                ],
            )

            report = orchestrator.scan_frontier_gaps(topic="rag drift", max_gaps=5)

            assert report.gaps_identified == 1
            assert report.gaps_rejected == 0
            assert len(report.gaps) == 1
            gap = report.gaps[0]
            assert gap.status == "identified"
            assert gap.domain_profile == "natural_language_processing"
            assert gap.evidence_count == 2

            created = orchestrator.propose_projects_from_gaps(max_proposals=1, confidence_threshold=0.45)

            assert len(created) == 1
            project = created[0]
            assert project.status == "proposed"
            stored_gap = orchestrator.store.get_frontier_gap(gap.gap_id)
            assert stored_gap is not None
            assert stored_gap.status == "proposed"
            assert stored_gap.proposed_project_id == project.project_id

            promoted = orchestrator.promote_gap_project(gap.gap_id)

            assert promoted.status == "active"
            promoted_gap = orchestrator.store.get_frontier_gap(gap.gap_id)
            assert promoted_gap is not None
            assert promoted_gap.status == "promoted"
        finally:
            orchestrator.shutdown()


def test_scan_frontier_gaps_rejects_duplicate_against_existing_project():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.create_project("Investigate retrieval augmented generation drift in language model memory systems.")
            _seed_document(
                orchestrator,
                "doc-dup-1",
                title="Duplicate Gap A",
                problem_statements=[
                    "Investigate retrieval augmented generation drift in language model memory systems.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-dup-2",
                title="Duplicate Gap B",
                problem_statements=[
                    "Analyze retrieval augmented generation drift in language model memory systems.",
                ],
            )

            report = orchestrator.scan_frontier_gaps(topic="rag drift duplicate", max_gaps=5)

            assert report.gaps_identified == 0
            assert report.gaps_rejected == 1
            assert len(report.gaps) == 1
            assert report.gaps[0].status == "rejected"
            assert report.gaps[0].rejection_reason == "too_similar_to_existing_project"
            assert report.gaps[0].similarity_to_existing > 0.75
        finally:
            orchestrator.shutdown()


def test_scan_frontier_gaps_rejects_domain_unaligned_cluster():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_document(
                orchestrator,
                "doc-amb-1",
                title="Ambiguous Frontier A",
                problem_statements=[
                    "Investigate emergent frontier signal collapse in adaptive agent loops.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-amb-2",
                title="Ambiguous Frontier B",
                problem_statements=[
                    "Analyze emergent frontier signal collapse in adaptive agent loops.",
                ],
            )

            report = orchestrator.scan_frontier_gaps(topic="ambiguous frontier", max_gaps=5)

            assert report.gaps_identified == 0
            assert report.gaps_rejected == 1
            assert len(report.gaps) == 1
            assert report.gaps[0].status == "rejected"
            assert report.gaps[0].rejection_reason == "domain_profile_unresolved"
            assert report.gaps[0].domain_profile is None
        finally:
            orchestrator.shutdown()


def test_reject_gap_project_parks_proposed_project():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_document(
                orchestrator,
                "doc-reject-1",
                title="Prompt Robustness A",
                problem_statements=[
                    "Investigate prompt robustness failures in language model retrieval pipelines.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-reject-2",
                title="Prompt Robustness B",
                problem_statements=[
                    "Analyze prompt robustness failures in language model retrieval pipelines.",
                ],
            )

            report = orchestrator.scan_frontier_gaps(topic="prompt robustness", max_gaps=5)
            gap = report.gaps[0]
            created = orchestrator.propose_projects_from_gaps(max_proposals=1, confidence_threshold=0.45)
            assert len(created) == 1

            updated_gap = orchestrator.reject_gap_project(gap.gap_id, "duplicate after operator review")

            assert updated_gap.status == "rejected"
            assert updated_gap.rejection_reason == "duplicate after operator review"
            project = orchestrator.store.get_research_project(created[0].project_id)
            assert project is not None
            assert project.status == "parked"
        finally:
            orchestrator.shutdown()


def test_ws36_control_commands_and_cli_renderers():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_document(
                orchestrator,
                "doc-control-1",
                title="Control Gap A",
                problem_statements=[
                    "Investigate retrieval robustness regressions in language model prompt pipelines.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-control-2",
                title="Control Gap B",
                problem_statements=[
                    "Analyze retrieval robustness regressions in language model prompt pipelines.",
                ],
            )

            scan = handle_request(
                orchestrator,
                ControlRequest(command="scan_frontier_gaps", payload={"topic": "retrieval robustness", "max_gaps": 5}),
            )
            assert scan.ok is True
            assert scan.payload["gaps_identified"] == 1
            gap_id = scan.payload["gaps"][0]["gap_id"]

            listed = handle_request(
                orchestrator,
                ControlRequest(command="list_frontier_gaps", payload={"limit": 10}),
            )
            assert listed.ok is True
            assert listed.payload["counts"]["identified"] == 1

            proposed = handle_request(
                orchestrator,
                ControlRequest(
                    command="propose_projects_from_gaps",
                    payload={"max_proposals": 1, "confidence_threshold": 0.45},
                ),
            )
            assert proposed.ok is True
            assert len(proposed.payload["projects"]) == 1
            project_id = proposed.payload["projects"][0]["project_id"]

            promoted = handle_request(
                orchestrator,
                ControlRequest(command="promote_gap_project", payload={"gap_id": gap_id}),
            )
            assert promoted.ok is True
            assert promoted.payload["status"] == "active"

            rendered_scan = tar_cli._render_frontier_gap_scan(scan.payload)
            rendered_list = tar_cli._render_frontier_gap_status(listed.payload)
            rendered_projects = tar_cli._render_gap_project_list(proposed.payload)
            assert "Scan ID:" in rendered_scan
            assert "Frontier Gaps: total=" in rendered_list
            assert f"- {project_id} ::" in rendered_projects
        finally:
            orchestrator.shutdown()


def test_ws36_operator_status_dashboard_and_frontier_status_surfaces():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_document(
                orchestrator,
                "doc-surface-1",
                title="Surface Gap A",
                problem_statements=[
                    "Investigate retrieval alignment drift in language model prompt systems.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-surface-2",
                title="Surface Gap B",
                problem_statements=[
                    "Analyze retrieval alignment drift in language model prompt systems.",
                ],
            )
            orchestrator.scan_frontier_gaps(topic="retrieval alignment drift", max_gaps=5)

            operator_view = orchestrator.operator_view(include_blocked=True, limit=5)
            status_payload = orchestrator.status()
            frontier = orchestrator.frontier_status()
            context = dashboard.build_dashboard_context(orchestrator, include_blocked=True, max_results=5)

            assert operator_view["frontier_gap_counts"]["identified"] >= 1
            assert operator_view["frontier_gap_scan_count"] >= 1
            assert status_payload["frontier_gap_counts"]["identified"] >= 1
            assert status_payload["frontier_gap_scans"] >= 1
            assert frontier.frontier_gap_counts["identified"] >= 1
            assert frontier.frontier_gap_scans >= 1
            assert context["frontier_gap_status"]["counts"]["identified"] >= 1
            assert "Frontier Gaps:" in tar_cli._render_operator_view(operator_view)
            assert "Frontier Gaps: total=" in tar_cli._render_status(status_payload)
            assert "Frontier Gap Scans:" in tar_cli._render_frontier_status(frontier.model_dump(mode="json"))
        finally:
            orchestrator.shutdown()


def test_ws36_filtered_gap_review_surfaces_selected_gap_in_dashboard_context():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _seed_document(
                orchestrator,
                "doc-filter-1",
                title="Filter Gap Proposed A",
                problem_statements=[
                    "Investigate retrieval grounded prompt drift in language model systems.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-filter-2",
                title="Filter Gap Proposed B",
                problem_statements=[
                    "Analyze retrieval grounded prompt drift in language model systems.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-filter-3",
                title="Filter Gap Identified A",
                problem_statements=[
                    "Investigate summarization faithfulness drift in language model reports.",
                ],
            )
            _seed_document(
                orchestrator,
                "doc-filter-4",
                title="Filter Gap Identified B",
                problem_statements=[
                    "Analyze summarization faithfulness drift in language model reports.",
                ],
            )

            report = orchestrator.scan_frontier_gaps(topic="frontier filtering", max_gaps=10)
            proposed_gap_id = next(
                gap.gap_id
                for gap in report.gaps
                if "retrieval grounded prompt drift" in gap.description.lower()
            )
            proposed_projects = orchestrator.propose_projects_from_gaps(max_proposals=1, confidence_threshold=0.45)
            assert len(proposed_projects) == 1

            filtered = handle_request(
                orchestrator,
                ControlRequest(
                    command="list_frontier_gaps",
                    payload={"status": "proposed", "min_confidence": 0.45, "limit": 10},
                ),
            )

            assert filtered.ok is True
            assert filtered.payload["status_filter"] == "proposed"
            assert filtered.payload["min_confidence"] == 0.45
            assert len(filtered.payload["gaps"]) == 1
            assert filtered.payload["gaps"][0]["gap_id"] == proposed_gap_id
            assert filtered.payload["gaps"][0]["status"] == "proposed"
            assert filtered.payload["recent_scans"]

            context = dashboard.build_dashboard_context(
                orchestrator,
                include_blocked=True,
                max_results=10,
                selected_project_id="",
                frontier_gap_status_filter="proposed",
                frontier_gap_min_confidence=0.45,
                selected_frontier_gap_id=proposed_gap_id,
            )

            assert context["frontier_gap_status"]["status_filter"] == "proposed"
            assert context["frontier_gap_status"]["min_confidence"] == 0.45
            assert len(context["frontier_gap_views"]) == 1
            assert context["selected_frontier_gap"]["gap_id"] == proposed_gap_id
            assert context["selected_frontier_gap"]["status"] == "proposed"
        finally:
            orchestrator.shutdown()
