import tempfile
from pathlib import Path

import pytest

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


def test_generate_falsification_plan_for_provisional_thread_attaches_actions():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate thermodynamic drift")
            _update_initial_project(
                orchestrator,
                project.project_id,
                thread_updates={"confidence_state": "provisional", "status": "supported"},
                question_updates={"importance": 0.85, "blocking": True},
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run the next supporting execution.",
                    "estimated_cost": 0.8,
                    "expected_evidence_gain": 0.45,
                },
            )

            payload = orchestrator.generate_falsification_plan(project.project_id)
            kinds = {item["kind"] for item in payload["plan"]["tests"]}

            assert payload["generated"] is True
            assert "mechanism_ablation" in kinds
            assert {"replication_check", "seed_variance_check"} & kinds
            assert payload["attached_actions"]
            assert all(
                action["falsification_plan_id"] == payload["plan"]["plan_id"]
                for action in payload["attached_actions"]
            )
            status_payload = orchestrator.falsification_status(project.project_id)
            assert status_payload["plan"]["plan_id"] == payload["plan"]["plan_id"]

            rendered = tar_cli._render_falsification_plan(status_payload)
            status_render = tar_cli._render_status(orchestrator.status())
            assert "Plan ID:" in rendered
            assert "Pending Tests:" in rendered
            assert "Falsification Plans:" in status_render
        finally:
            orchestrator.shutdown()


def test_contradicting_thread_triggers_contradiction_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate inconsistent benchmark signals")
            _update_initial_project(
                orchestrator,
                project.project_id,
                thread_updates={
                    "confidence_state": "provisional",
                    "status": "contradicted",
                    "contradicting_evidence_ids": ["paper-1", "claim-2"],
                },
            )

            payload = orchestrator.generate_falsification_plan(project.project_id)
            kinds = {item["kind"] for item in payload["plan"]["tests"]}

            assert "contradiction_resolution" in kinds
        finally:
            orchestrator.shutdown()


def test_weak_benchmark_alignment_triggers_benchmark_stress_probe():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project(
                "Investigate retrieval failures in NLP",
                benchmark_tier="canonical",
            )
            study = orchestrator.study_problem(
                "Investigate retrieval failures in NLP",
                project_id=project.project_id,
                benchmark_tier="canonical",
                max_results=0,
                build_env=False,
            )
            assert study.project_id == project.project_id

            payload = orchestrator.generate_falsification_plan(project.project_id)
            kinds = {item["kind"] for item in payload["plan"]["tests"]}

            assert "benchmark_stress_probe" in kinds
        finally:
            orchestrator.shutdown()


def test_falsification_first_ranking_prefers_generated_meta_test():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate fast but fragile result")
            _update_initial_project(
                orchestrator,
                project.project_id,
                thread_updates={"confidence_state": "provisional", "status": "supported"},
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run a broad expensive follow-up benchmark sweep.",
                    "estimated_cost": 1.6,
                    "expected_evidence_gain": 0.35,
                },
            )
            payload = orchestrator.generate_falsification_plan(project.project_id)
            ranked = orchestrator.rank_actions(
                project_id=project.project_id,
                include_blocked=True,
                limit=5,
                mode="falsification_first",
            )

            top = ranked["candidates"][0]
            assert payload["plan"]["plan_id"]
            assert top["action_kind"] in {
                "mechanism_ablation",
                "replication_check",
                "seed_variance_check",
                "contradiction_resolution",
                "benchmark_stress_probe",
                "calibration_check",
                "environment_reproduction_check",
                "claim_linkage_sanity_check",
            }
        finally:
            orchestrator.shutdown()


def test_abandoned_project_requires_force_for_new_falsification_plan():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate abandoned thread recovery")
            _update_initial_project(
                orchestrator,
                project.project_id,
                project_updates={"status": "abandoned"},
                thread_updates={"confidence_state": "provisional", "status": "parked"},
            )

            with pytest.raises(RuntimeError, match="abandoned"):
                orchestrator.generate_falsification_plan(project.project_id)

            forced = orchestrator.generate_falsification_plan(project.project_id, force=True)
            assert forced["generated"] is True
        finally:
            orchestrator.shutdown()


def test_control_surface_exposes_falsification_commands():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate falsification surface")
            _update_initial_project(
                orchestrator,
                project.project_id,
                thread_updates={"confidence_state": "provisional", "status": "supported"},
            )

            generated = handle_request(
                orchestrator,
                ControlRequest(
                    command="generate_falsification_plan",
                    payload={"project_id": project.project_id},
                ),
            )
            status = handle_request(
                orchestrator,
                ControlRequest(
                    command="falsification_status",
                    payload={"project_id": project.project_id},
                ),
            )
            log = handle_request(
                orchestrator,
                ControlRequest(command="falsification_log", payload={"count": 10}),
            )

            assert generated.ok is True
            assert generated.payload["generated"] is True
            assert status.ok is True
            assert status.payload["plan"]["project_id"] == project.project_id
            assert log.ok is True
            assert len(log.payload["plans"]) >= 1
            assert "Plan ID:" in tar_cli._render_falsification_plan(status.payload)
            assert "status=" in tar_cli._render_falsification_log(log.payload)
        finally:
            orchestrator.shutdown()
