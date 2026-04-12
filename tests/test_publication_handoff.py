import tempfile
from pathlib import Path

import tar_cli
from tar_lab.control import handle_request
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    CalibrationReport,
    ClaimAcceptancePolicy,
    ClaimVerdict,
    ControlRequest,
    SeedVarianceReport,
    VerificationReport,
)


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


def _seed_ws22_project(orchestrator: TAROrchestrator):
    project = orchestrator.create_project("Investigate TCL publication handoff readiness")
    _update_initial_project(
        orchestrator,
        project.project_id,
        thread_updates={
            "status": "supported",
            "confidence_state": "provisional",
            "supporting_evidence_ids": ["evidence-support-1", "evidence-support-2"],
            "contradicting_evidence_ids": ["evidence-contradict-1"],
        },
        question_updates={
            "question": "What still blocks a publication-grade TCL claim package?",
            "importance": 0.92,
            "blocking": True,
        },
        action_updates={
            "action_kind": "verify_claim",
            "description": "Verify the strongest TCL signal before promotion.",
            "estimated_cost": 0.35,
            "expected_evidence_gain": 0.72,
        },
    )
    study = orchestrator.study_problem(
        "Investigate TCL publication handoff readiness",
        project_id=project.project_id,
        build_env=False,
        max_results=0,
    )
    verification = VerificationReport(
        trial_id="trial-ws22-accepted",
        control_score=1.91,
        seed_variance=SeedVarianceReport(
            num_runs=3,
            loss_mean=0.31,
            loss_std=0.02,
            dimensionality_mean=8.2,
            dimensionality_std=0.11,
            calibration_ece_mean=0.05,
            stable=True,
        ),
        calibration=CalibrationReport(
            ece=0.05,
            accuracy=0.91,
            mean_confidence=0.87,
        ),
        verdict="verified",
        recommendations=["Keep the claim bounded until falsification coverage is complete."],
    )
    orchestrator.store.append_verification_report(verification)
    policy = ClaimAcceptancePolicy.model_validate(orchestrator.claim_policy())
    accepted = ClaimVerdict(
        verdict_id="verdict-ws22-accepted",
        trial_id="trial-ws22-accepted",
        status="accepted",
        rationale=["Benchmark-linked verification supports an accepted TCL claim bundle."],
        policy=policy,
        supporting_research_ids=list(study.cited_research_ids),
        supporting_evidence_ids=["evidence-support-1", "evidence-support-2"],
        verification_report_trial_id="trial-ws22-accepted",
        benchmark_problem_id=study.problem_id,
        benchmark_execution_mode="problem_study",
        supporting_benchmark_ids=list(study.benchmark_ids),
        supporting_benchmark_names=list(study.benchmark_names),
        canonical_comparability_source="problem_study",
        verdict_inputs_complete=True,
        linkage_status="exact",
        canonical_benchmark_required=study.benchmark_tier == "canonical",
        canonical_benchmark_satisfied=study.canonical_comparable,
        confidence=0.91,
    )
    contradicted = ClaimVerdict(
        verdict_id="verdict-ws22-contradicted",
        trial_id="trial-ws22-contradicted",
        status="contradicted",
        rationale=["A competing TCL interpretation remains contradicted by the current evidence bundle."],
        policy=policy,
        supporting_research_ids=[],
        supporting_evidence_ids=["evidence-contradict-1"],
        verification_report_trial_id="trial-ws22-accepted",
        benchmark_problem_id=study.problem_id,
        benchmark_execution_mode="problem_study",
        supporting_benchmark_ids=list(study.benchmark_ids),
        supporting_benchmark_names=list(study.benchmark_names),
        canonical_comparability_source="problem_study",
        verdict_inputs_complete=True,
        linkage_status="exact",
        canonical_benchmark_required=False,
        canonical_benchmark_satisfied=False,
        confidence=0.22,
    )
    orchestrator.store.append_claim_verdict(accepted)
    orchestrator.store.append_claim_verdict(contradicted)
    orchestrator.generate_falsification_plan(project.project_id)
    orchestrator.evidence_debt(project_id=project.project_id)
    orchestrator.portfolio_review(limit=5)
    orchestrator.portfolio_decide(limit=5)
    return project, study, accepted, contradicted


def test_publication_handoff_packages_claims_lineage_and_limitations():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project, study, accepted, contradicted = _seed_ws22_project(orchestrator)

            payload = orchestrator.publication_handoff(project.project_id)
            package = payload["package"]

            assert payload["generated"] is True
            assert package["project_id"] == project.project_id
            assert package["accepted_claims"][0]["verdict_id"] == accepted.verdict_id
            assert any(item["related_verdict_id"] == contradicted.verdict_id for item in package["rejected_alternatives"])
            assert any(item["source_kind"] == "problem_study" for item in package["benchmark_truth_attachments"])
            assert any(item["event_type"] == "claim_verdict" for item in package["experiment_lineage"])
            assert package["limitations"]
            assert package["writer_cautions"]
            assert Path(package["artifact_path"]).exists()

            rendered = tar_cli._render_publication_handoff(payload)
            status_render = tar_cli._render_status(orchestrator.status())

            assert "Publication Package:" in rendered
            assert "Accepted Claim Bundles:" in rendered
            assert "Open Questions:" in rendered
            assert any(item["source_id"] == study.problem_id for item in package["experiment_lineage"] if item["source_id"])
            assert "Publication Handoffs:" in status_render
        finally:
            orchestrator.shutdown()


def test_control_surface_exposes_publication_handoff_commands():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project, _, _, _ = _seed_ws22_project(orchestrator)

            generated = handle_request(
                orchestrator,
                ControlRequest(command="publication_handoff", payload={"project_id": project.project_id}),
            )
            log = handle_request(
                orchestrator,
                ControlRequest(command="publication_log", payload={"count": 10}),
            )

            assert generated.ok is True
            assert generated.payload["package"]["project_id"] == project.project_id
            assert log.ok is True
            assert len(log.payload["packages"]) >= 1
            assert "Publication Package:" in tar_cli._render_publication_handoff(generated.payload)
            assert "Publication Handoff Packages:" in tar_cli._render_publication_log(log.payload)
        finally:
            orchestrator.shutdown()
