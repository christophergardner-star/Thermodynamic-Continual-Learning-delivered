import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tar_cli
from tar_lab.control import handle_request
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ControlRequest


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


def test_portfolio_review_surfaces_stale_and_resume_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            stale = orchestrator.create_project("Investigate stalled runtime recovery")
            stale_time = (datetime.now(timezone.utc) - timedelta(hours=80)).replace(microsecond=0).isoformat()
            _update_initial_project(
                orchestrator,
                stale.project_id,
                project_updates={"status": "paused", "updated_at": stale_time},
                thread_updates={"status": "parked", "stop_reason": "runtime_failure", "updated_at": stale_time},
            )

            fresh = orchestrator.create_project("Investigate active benchmark follow-up")
            _update_initial_project(
                orchestrator,
                fresh.project_id,
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run the next benchmark-aligned execution.",
                    "estimated_cost": 0.4,
                    "expected_evidence_gain": 0.55,
                },
            )

            payload = orchestrator.portfolio_review(limit=5)
            stale_ids = {item["project_id"] for item in payload["stale_projects"]}
            resume_ids = {item["project_id"] for item in payload["resume_candidates"]}

            assert stale.project_id in stale_ids
            assert stale.project_id in resume_ids
            assert orchestrator.store.load_research_portfolio().stale_project_ids
            rendered = tar_cli._render_portfolio_review(payload)
            assert "Top Projects:" in rendered
            assert stale.project_id in rendered
        finally:
            orchestrator.shutdown()


def test_evidence_debt_flags_under_proved_project():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate fragile provisional signal")
            _update_initial_project(
                orchestrator,
                project.project_id,
                thread_updates={"confidence_state": "provisional", "status": "supported"},
            )

            payload = orchestrator.evidence_debt(project_id=project.project_id)
            record = payload["records"][0]

            assert record["project_id"] == project.project_id
            assert record["promotion_blocked"] is True
            assert record["overall_debt"] >= 0.45
            rendered = tar_cli._render_evidence_debt(payload)
            assert project.project_id in rendered
        finally:
            orchestrator.shutdown()


def test_portfolio_decide_prefers_ready_project_over_blocked_one():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            ready = orchestrator.create_project("Investigate ready continuation")
            _update_initial_project(
                orchestrator,
                ready.project_id,
                project_updates={"priority": 3},
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run a ready, cheap continuation study.",
                    "estimated_cost": 0.25,
                    "expected_evidence_gain": 0.6,
                },
            )

            blocked = orchestrator.create_project("Investigate missing dependency")
            _update_initial_project(
                orchestrator,
                blocked.project_id,
                project_updates={"status": "blocked", "priority": 5},
                thread_updates={"status": "parked", "stop_reason": "dependency_missing"},
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run a tempting but blocked study.",
                    "estimated_cost": 0.1,
                    "expected_evidence_gain": 0.95,
                },
            )

            payload = orchestrator.portfolio_decide(limit=5)

            assert payload["decision"]["selected_project_id"] == ready.project_id
            assert blocked.project_id in payload["decision"]["deferred_project_ids"] or blocked.project_id not in {payload["decision"]["selected_project_id"]}
            assert orchestrator.store.latest_portfolio_decision() is not None
        finally:
            orchestrator.shutdown()


def test_portfolio_decision_tracks_parked_projects():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            active = orchestrator.create_project("Investigate active path")
            _update_initial_project(
                orchestrator,
                active.project_id,
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run the active project next.",
                    "estimated_cost": 0.3,
                    "expected_evidence_gain": 0.6,
                },
            )

            exhausted = orchestrator.create_project("Investigate exhausted budget")
            project = orchestrator.store.get_research_project(exhausted.project_id)
            assert project is not None
            exhausted_budget = project.budget_ledger.model_copy(
                update={"experiments_spent": project.budget_ledger.experiment_budget, "budget_exhausted": True, "budget_pressure_level": "exhausted"}
            )
            _update_initial_project(
                orchestrator,
                exhausted.project_id,
                project_updates={"status": "paused", "budget_ledger": exhausted_budget},
                thread_updates={"status": "parked", "stop_reason": "budget_exhausted"},
            )

            payload = orchestrator.portfolio_decide(limit=5)

            assert exhausted.project_id in payload["decision"]["parked_project_ids"]
            rendered = tar_cli._render_portfolio_decision(payload)
            assert "Parked:" in rendered
        finally:
            orchestrator.shutdown()


def test_control_surface_exposes_ws20_commands():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate portfolio surface")
            _update_initial_project(
                orchestrator,
                project.project_id,
                thread_updates={"confidence_state": "provisional", "status": "supported"},
            )

            review = handle_request(orchestrator, ControlRequest(command="portfolio_review", payload={"limit": 5}))
            decide = handle_request(orchestrator, ControlRequest(command="portfolio_decide", payload={"limit": 5}))
            stale = handle_request(orchestrator, ControlRequest(command="stale_projects", payload={"limit": 5}))
            debt = handle_request(orchestrator, ControlRequest(command="evidence_debt", payload={"project_id": project.project_id, "limit": 5}))
            resume = handle_request(orchestrator, ControlRequest(command="resume_candidates", payload={"limit": 5}))

            assert review.ok is True
            assert decide.ok is True
            assert stale.ok is True
            assert debt.ok is True
            assert resume.ok is True
            assert "Portfolio:" in tar_cli._render_portfolio_review(review.payload)
            assert "Portfolio Decision:" in tar_cli._render_portfolio_decision(decide.payload)
            assert isinstance(stale.payload["stale_projects"], list)
            assert project.project_id in tar_cli._render_evidence_debt(debt.payload)
            assert isinstance(resume.payload["resume_candidates"], list)
        finally:
            orchestrator.shutdown()
