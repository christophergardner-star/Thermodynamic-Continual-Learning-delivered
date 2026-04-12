import tempfile
from pathlib import Path

import dashboard
import tar_cli
from tar_lab.control import handle_request
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ClaimAcceptancePolicy, ClaimVerdict, ControlRequest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_science_profiles(tmp: str) -> None:
    source_dir = REPO_ROOT / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _update_initial_project(
    orchestrator: TAROrchestrator,
    project_id: str,
    *,
    project_updates=None,
    thread_updates=None,
    question_updates=None,
    action_updates=None,
):
    project = orchestrator.store.get_research_project(project_id)
    assert project is not None
    thread = project.hypothesis_threads[0].model_copy(update=thread_updates or {})
    question = project.open_questions[0].model_copy(update=question_updates or {})
    action = project.planned_actions[0].model_copy(update=action_updates or {})
    updated = project.model_copy(
        update={
            **(project_updates or {}),
            "hypothesis_threads": [thread],
            "open_questions": [question],
            "planned_actions": [action],
            "active_thread_id": thread.thread_id,
        }
    )
    orchestrator.store.upsert_research_project(updated)
    return updated


def _seed_ws21_project(orchestrator: TAROrchestrator):
    project = orchestrator.create_project("Investigate operator evidence flow in TAR")
    _update_initial_project(
        orchestrator,
        project.project_id,
        thread_updates={
            "status": "supported",
            "confidence_state": "provisional",
            "supporting_evidence_ids": ["evidence-support-1"],
            "contradicting_evidence_ids": ["evidence-contradict-1"],
        },
        action_updates={
            "action_kind": "run_problem_study",
            "description": "Run the next benchmark-aligned study.",
            "estimated_cost": 0.25,
            "expected_evidence_gain": 0.7,
        },
    )
    study = orchestrator.study_problem(
        "Investigate operator evidence flow in TAR",
        project_id=project.project_id,
        build_env=False,
        max_results=0,
    )
    policy = ClaimAcceptancePolicy.model_validate(orchestrator.claim_policy())
    verdict = ClaimVerdict(
        verdict_id="verdict-ws21-1",
        trial_id="trial-ws21-1",
        status="provisional",
        rationale=["Initial benchmark-aligned evidence supports a provisional claim."],
        policy=policy,
        supporting_research_ids=list(study.cited_research_ids),
        supporting_evidence_ids=["evidence-support-1"],
        verification_report_trial_id="trial-ws21-1",
        benchmark_problem_id=study.problem_id,
        benchmark_execution_mode="problem_study",
        supporting_benchmark_ids=list(study.benchmark_ids),
        supporting_benchmark_names=list(study.benchmark_names),
        canonical_comparability_source="problem_study",
        verdict_inputs_complete=True,
        linkage_status="exact",
        canonical_benchmark_required=study.benchmark_tier == "canonical",
        canonical_benchmark_satisfied=study.canonical_comparable,
        confidence=0.64,
    )
    orchestrator.store.append_claim_verdict(verdict)
    orchestrator.generate_falsification_plan(project.project_id)
    orchestrator.portfolio_review(limit=5)
    orchestrator.portfolio_decide(limit=5)
    return project, study, verdict


def test_ws21_operator_surfaces_and_renderers():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project, study, verdict = _seed_ws21_project(orchestrator)

            operator_view = handle_request(
                orchestrator,
                ControlRequest(command="operator_view", payload={"limit": 5, "include_blocked": True}),
            )
            timeline = handle_request(
                orchestrator,
                ControlRequest(command="project_timeline", payload={"project_id": project.project_id, "limit": 20}),
            )
            evidence_map = handle_request(
                orchestrator,
                ControlRequest(command="evidence_map", payload={"project_id": project.project_id}),
            )
            claim_lineage = handle_request(
                orchestrator,
                ControlRequest(command="claim_lineage", payload={"project_id": project.project_id}),
            )
            resume_dashboard = handle_request(
                orchestrator,
                ControlRequest(command="resume_dashboard", payload={"project_id": project.project_id}),
            )

            assert operator_view.ok is True
            assert timeline.ok is True
            assert evidence_map.ok is True
            assert claim_lineage.ok is True
            assert resume_dashboard.ok is True

            event_types = {item["event_type"] for item in timeline.payload["events"]}
            assert "project_created" in event_types
            assert "problem_study" in event_types
            assert "project_priority" in event_types
            assert "falsification_plan" in event_types
            assert "claim_verdict" in event_types
            assert "portfolio_decision" in event_types

            assert evidence_map.payload["project_id"] == project.project_id
            assert evidence_map.payload["evidence_counts"]["supporting"] == 1
            assert evidence_map.payload["claim_verdicts"][0]["verdict_id"] == verdict.verdict_id
            assert claim_lineage.payload["problem_ids"] == [study.problem_id]
            assert claim_lineage.payload["verdicts"][0]["benchmark_problem_id"] == study.problem_id
            assert resume_dashboard.payload["project"]["project"]["project_id"] == project.project_id

            assert "Top Action Candidates:" in tar_cli._render_operator_view(operator_view.payload)
            assert "Timeline:" in tar_cli._render_project_timeline(timeline.payload)
            assert "Evidence Summary:" in tar_cli._render_evidence_map(evidence_map.payload)
            assert "Claim Verdicts:" in tar_cli._render_claim_lineage(claim_lineage.payload)
            assert "Resume State:" in tar_cli._render_resume_dashboard(resume_dashboard.payload)
        finally:
            orchestrator.shutdown()


def test_ws21_resume_dashboard_surfaces_blockers():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate blocked resume path")
            orchestrator.pause_project(
                project.project_id,
                reason="dependency_missing",
                note="Awaiting benchmark environment restore.",
            )

            payload = orchestrator.resume_dashboard(project.project_id)
            rendered = tar_cli._render_resume_dashboard(payload)

            assert payload["resume_state"] == "blocked"
            assert "dependency_missing" in payload["blockers"]
            assert "Awaiting benchmark environment restore." in payload["resume_snapshot"]["blockers"]
            assert "Resume State: blocked" in rendered
            assert "dependency_missing" in rendered
        finally:
            orchestrator.shutdown()


def test_ws21_dashboard_context_exposes_selected_project_views():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project, _, _ = _seed_ws21_project(orchestrator)
            context = dashboard.build_dashboard_context(
                orchestrator,
                include_blocked=True,
                max_results=5,
                selected_project_id=project.project_id,
            )

            assert context["workspace"] == orchestrator.workspace
            assert project.project_id in context["project_ids"]
            assert context["selected_project_status"]["project"]["project_id"] == project.project_id
            assert context["selected_resume_dashboard"]["project"]["project"]["project_id"] == project.project_id
            assert context["selected_evidence_map"]["project_id"] == project.project_id
            assert context["selected_claim_lineage"]["project_id"] == project.project_id
            assert context["selected_timeline"]["project_id"] == project.project_id
            assert context["selected_publication_handoff"]["package"]["project_id"] == project.project_id
        finally:
            orchestrator.shutdown()
