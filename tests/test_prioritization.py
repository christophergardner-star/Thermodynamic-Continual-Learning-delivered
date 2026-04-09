import tempfile
from pathlib import Path

import tar_cli
from tar_lab.control import handle_request
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ControlRequest, RuntimeHeartbeat


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


def test_rank_actions_prefers_cheap_falsification_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            expensive = orchestrator.create_project("expensive exploratory graph sweep")
            _update_initial_project(
                orchestrator,
                expensive.project_id,
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run broad exploratory benchmark sweep across multiple graph settings.",
                    "estimated_cost": 2.0,
                    "expected_evidence_gain": 0.55,
                },
                question_updates={"importance": 0.4, "blocking": False},
            )

            cheap = orchestrator.create_project("cheap falsification check")
            _update_initial_project(
                orchestrator,
                cheap.project_id,
                thread_updates={"confidence_state": "provisional"},
                action_updates={
                    "action_kind": "verify_claim",
                    "description": "Run a cheap falsification ablation before promotion.",
                    "estimated_cost": 0.2,
                    "expected_evidence_gain": 0.45,
                },
                question_updates={"importance": 0.8, "blocking": True},
            )

            payload = orchestrator.rank_actions(include_blocked=True, limit=5)
            candidates = payload["candidates"]

            assert candidates[0]["project_id"] == cheap.project_id
            assert candidates[0]["action_kind"] == "verify_claim"
            assert candidates[0]["recommended"] is True
            assert orchestrator.store.latest_priority_snapshot() is not None

            rendered = tar_cli._render_ranked_candidates(candidates)
            assert "cheap falsification check" in rendered
            assert "Score:" in rendered
        finally:
            orchestrator.shutdown()


def test_rank_actions_deprioritizes_blocked_projects():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            active = orchestrator.create_project("active benchmark followup")
            _update_initial_project(
                orchestrator,
                active.project_id,
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run the next benchmark-aligned execution.",
                    "estimated_cost": 0.7,
                    "expected_evidence_gain": 0.5,
                },
            )

            blocked = orchestrator.create_project("blocked dependency check")
            _update_initial_project(
                orchestrator,
                blocked.project_id,
                project_updates={"status": "blocked"},
                thread_updates={"stop_reason": "dependency_missing", "status": "parked"},
                action_updates={
                    "action_kind": "verify_claim",
                    "description": "Try a blocked dependency-limited verification pass.",
                    "estimated_cost": 0.1,
                    "expected_evidence_gain": 0.9,
                },
            )

            payload = orchestrator.rank_actions(include_blocked=True, limit=5)
            candidates = payload["candidates"]

            assert candidates[0]["project_id"] == active.project_id
            blocked_candidate = next(item for item in candidates if item["project_id"] == blocked.project_id)
            assert blocked_candidate["blocked"] is True
            assert blocked_candidate["score"] < candidates[0]["score"]
        finally:
            orchestrator.shutdown()


def test_schedule_problem_study_uses_ws18_priority():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem("Investigate barren plateaus in quantum AI", build_env=False, max_results=0)
            entry = orchestrator.schedule_problem_study(problem_id=study.problem_id, delay_s=0, max_runs=1)

            assert entry.priority > 0
            assert entry.priority_score is not None
            assert entry.priority_source == "ws18_ranked_action"
        finally:
            orchestrator.shutdown()


def test_allocate_budget_can_schedule_selected_action():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem("Investigate barren plateaus in quantum AI", build_env=False, max_results=0)
            payload = orchestrator.allocate_budget(project_id=study.project_id, schedule_selected=True)

            assert payload["schedule_created"] is True
            assert payload["scheduled_job_id"]
            latest = orchestrator.store.latest_budget_allocation()
            assert latest is not None
            assert latest.scheduled_job_id == payload["scheduled_job_id"]

            portfolio = orchestrator.portfolio_status()
            assert portfolio["project_counts"]["total"] >= 1
        finally:
            orchestrator.shutdown()


def test_run_runtime_cycle_reprioritizes_scheduled_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem("Investigate calibration in deep learning", build_env=False, max_results=0)
            entry = orchestrator.schedule_problem_study(problem_id=study.problem_id, delay_s=3600, max_runs=1, priority=1)
            orchestrator.store.update_problem_schedule(
                entry.schedule_id,
                priority=0,
                priority_score=None,
                priority_source=None,
            )
            orchestrator.runtime_daemon.run_cycle = lambda **kwargs: RuntimeHeartbeat(  # type: ignore[assignment]
                started_at="2026-04-09T00:00:00+00:00",
                finished_at="2026-04-09T00:00:01+00:00",
                status="completed",
                executed_jobs=0,
                stale_cleanups=0,
                failed_jobs=0,
                active_leases=0,
                retry_waiting=0,
                alert_count=0,
                notes=[],
            )
            orchestrator.run_runtime_cycle(max_jobs=1)
            updated = orchestrator.store.get_problem_schedule(entry.schedule_id)

            assert updated is not None
            assert updated.priority > 0
            assert updated.priority_source == "ws18_ranked_action"
        finally:
            orchestrator.shutdown()


def test_control_surface_exposes_prioritization_commands():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            created = orchestrator.create_project("portfolio triage example")

            ranked = handle_request(
                orchestrator,
                ControlRequest(command="rank_actions", payload={"project_id": created.project_id}),
            )
            allocated = handle_request(
                orchestrator,
                ControlRequest(command="allocate_budget", payload={"project_id": created.project_id}),
            )
            log = handle_request(orchestrator, ControlRequest(command="prioritization_log", payload={"count": 10}))

            assert ranked.ok is True
            assert ranked.payload["selected_project_id"] == created.project_id
            assert allocated.ok is True
            assert allocated.payload["selected_candidate"]["project_id"] == created.project_id
            assert log.ok is True
            assert len(log.payload["snapshots"]) >= 1
            assert "Top Candidates:" in tar_cli._render_portfolio_status(orchestrator.portfolio_status())
            assert "Allocation Decision:" in tar_cli._render_budget_allocation(allocated.payload)
            assert "selected" in tar_cli._render_prioritization_log(log.payload)
        finally:
            orchestrator.shutdown()
