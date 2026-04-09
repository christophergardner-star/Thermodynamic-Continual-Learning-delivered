import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import tar_cli
from tar_lab.control import handle_request
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ControlRequest, ProblemExecutionReport


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_science_profiles(tmp: str) -> None:
    source_dir = REPO_ROOT / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_create_project_persists_continuity_state():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate barren plateaus in quantum AI")
            payload = orchestrator.project_status(project.project_id)

            assert payload["project"]["project_id"] == project.project_id
            assert payload["project"]["status"] == "active"
            assert payload["active_thread"]["thread_id"] == project.active_thread_id
            assert payload["current_question"]["status"] == "open"
            assert payload["next_action"]["status"] == "planned"
            assert payload["budget_remaining"]["experiments_remaining"] == float(project.budget_ledger.experiment_budget)

            rendered = tar_cli._render_project_status(payload)
            rendered_list = tar_cli._render_project_list(orchestrator.list_projects())
            assert f"Project ID: {project.project_id}" in rendered
            assert "Next Action Status: planned" in rendered
            assert project.goal in rendered_list
        finally:
            orchestrator.shutdown()


def test_schedule_problem_study_preserves_project_linkage():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate calibration in deep learning")
            study = orchestrator.study_problem(
                "Investigate calibration in deep learning",
                project_id=project.project_id,
                build_env=False,
                max_results=0,
            )
            entry = orchestrator.schedule_problem_study(problem_id=study.problem_id, delay_s=0, max_runs=1)
            refreshed = orchestrator.store.get_research_project(project.project_id)

            assert study.project_id == project.project_id
            assert study.thread_id is not None
            assert study.next_action_id is not None
            assert len({item.action_id for item in refreshed.planned_actions}) == len(refreshed.planned_actions)
            assert entry.project_id == project.project_id
            assert entry.thread_id == study.thread_id
            assert entry.action_id == study.next_action_id
            assert refreshed is not None
            active_thread = next(item for item in refreshed.hypothesis_threads if item.thread_id == refreshed.active_thread_id)
            queued_action = next(item for item in refreshed.planned_actions if item.action_id == active_thread.next_action_id)
            assert queued_action.status == "queued"
            assert queued_action.scheduled_job_id == entry.schedule_id
        finally:
            orchestrator.shutdown()


def test_run_problem_study_updates_budget_and_follow_up_action(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem("Investigate barren plateaus in quantum AI", build_env=False, max_results=0)
            project = orchestrator.store.get_research_project(study.project_id or "")
            assert project is not None
            constrained_budget = project.budget_ledger.model_copy(update={"experiment_budget": 1})
            orchestrator.store.upsert_research_project(project.model_copy(update={"budget_ledger": constrained_budget}))

            report_path = Path(study.environment.execution_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            fake = ProblemExecutionReport(
                problem_id=study.problem_id,
                problem=study.problem,
                profile_id=study.profile_id,
                domain=study.domain,
                execution_mode="local_python",
                imports_ok=["numpy"],
                experiments=[],
                summary="Synthetic execution complete.",
                recommended_next_step="Review the first execution and design the next benchmark pass.",
                artifact_path=str(report_path),
                status="completed",
            )
            report_path.write_text(fake.model_dump_json(indent=2), encoding="utf-8")
            monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))

            report = orchestrator.run_problem_study(problem_id=study.problem_id, use_docker=False)
            refreshed = orchestrator.store.get_research_project(study.project_id or "")

            assert report.project_id == study.project_id
            assert report.thread_id == study.thread_id
            assert report.action_id == study.next_action_id
            assert refreshed is not None
            assert refreshed.status == "paused"
            assert refreshed.budget_ledger.budget_exhausted is True
            assert refreshed.budget_ledger.experiments_spent == 1

            active_thread = next(item for item in refreshed.hypothesis_threads if item.thread_id == refreshed.active_thread_id)
            completed_action = next(item for item in refreshed.planned_actions if item.action_id == study.next_action_id)
            follow_up_action = next(item for item in refreshed.planned_actions if item.action_id == active_thread.next_action_id)
            assert completed_action.status == "completed"
            assert follow_up_action.action_kind == "review_execution_result"
            assert follow_up_action.status == "planned"
        finally:
            orchestrator.shutdown()


def test_pause_and_resume_project_record_reasons():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate optimization stability in deep learning")
            paused = orchestrator.pause_project(
                project.project_id,
                reason="dependency_missing",
                note="Awaiting the required benchmark environment.",
            )
            resumed = orchestrator.resume_project(
                project.project_id,
                reason="dependency_restored",
                note="Environment restored and ready to continue.",
            )
            payload = orchestrator.project_status(project.project_id)

            assert paused.status == "blocked"
            assert resumed.status == "active"
            assert payload["active_thread"]["stop_reason"] == "dependency_missing"
            assert payload["active_thread"]["resume_reason"] == "dependency_restored"
            assert "Environment restored" in payload["project"]["latest_decision_summary"]

            next_payload = orchestrator.next_action(project.project_id)
            rendered = tar_cli._render_next_action(next_payload)
            assert f"Project ID: {project.project_id}" in rendered
            assert "Current Question:" in rendered
        finally:
            orchestrator.shutdown()


def test_control_surface_exposes_project_commands():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            created = handle_request(
                orchestrator,
                ControlRequest(
                    command="create_project",
                    payload={"problem": "Investigate retrieval failures in NLP"},
                ),
            )
            assert created.ok is True
            project_id = created.payload["project"]["project_id"]

            listed = handle_request(orchestrator, ControlRequest(command="list_projects", payload={}))
            status = handle_request(
                orchestrator,
                ControlRequest(command="project_status", payload={"project_id": project_id}),
            )
            next_action = handle_request(
                orchestrator,
                ControlRequest(command="next_action", payload={"project_id": project_id}),
            )

            assert listed.ok is True
            assert any(item["project"]["project_id"] == project_id for item in listed.payload["projects"])
            assert status.ok is True
            assert status.payload["project"]["project_id"] == project_id
            assert next_action.ok is True
            assert next_action.payload["project_id"] == project_id
        finally:
            orchestrator.shutdown()
