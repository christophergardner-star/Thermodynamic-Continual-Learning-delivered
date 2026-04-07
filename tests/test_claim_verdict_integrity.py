from pathlib import Path
import tempfile

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    AblationResult,
    CalibrationBin,
    CalibrationReport,
    ClaimAcceptancePolicy,
    MemorySearchHit,
    ProblemExecutionReport,
    ProblemStudyReport,
    ScienceEnvironmentBundle,
    SeedRunResult,
    SeedVarianceReport,
    VerificationReport,
)


class _FakeVault:
    def __init__(self, hits: list[MemorySearchHit] | None = None):
        self._hits = hits or []

    def search(self, *args, **kwargs):
        return list(self._hits)

    def close(self) -> None:
        return None


def _verification_report(trial_id: str = "trial-1") -> VerificationReport:
    return VerificationReport(
        trial_id=trial_id,
        control_score=2.1,
        seed_variance=SeedVarianceReport(
            num_runs=3,
            loss_mean=0.5,
            loss_std=0.02,
            dimensionality_mean=3.2,
            dimensionality_std=0.1,
            calibration_ece_mean=0.04,
            stable=True,
            runs=[
                SeedRunResult(
                    seed=7,
                    training_loss=0.5,
                    effective_dimensionality=3.1,
                    equilibrium_fraction=0.7,
                    calibration_ece=0.03,
                ),
                SeedRunResult(
                    seed=18,
                    training_loss=0.52,
                    effective_dimensionality=3.3,
                    equilibrium_fraction=0.69,
                    calibration_ece=0.04,
                ),
                SeedRunResult(
                    seed=29,
                    training_loss=0.48,
                    effective_dimensionality=3.2,
                    equilibrium_fraction=0.71,
                    calibration_ece=0.05,
                ),
            ],
        ),
        calibration=CalibrationReport(
            ece=0.05,
            accuracy=0.8,
            mean_confidence=0.78,
            bins=[
                CalibrationBin(
                    lower=0.0,
                    upper=1.0,
                    count=10,
                    mean_confidence=0.78,
                    accuracy=0.8,
                )
            ],
        ),
        ablations=[
            AblationResult(
                name="no_anchor_penalty",
                training_loss=0.7,
                effective_dimensionality=2.8,
                equilibrium_fraction=0.5,
                calibration_ece=0.09,
                score=1.9,
                delta_vs_control=-0.2,
            )
        ],
        verdict="verified",
        recommendations=[],
    )


def _problem_execution_report(
    problem_id: str,
    *,
    canonical_comparable: bool,
    benchmark_id: str,
    benchmark_name: str,
) -> ProblemExecutionReport:
    return ProblemExecutionReport(
        problem_id=problem_id,
        problem=f"Problem {problem_id}",
        profile_id="deep_learning",
        domain="deep_learning",
        canonical_comparable=canonical_comparable,
        benchmark_ids=[benchmark_id],
        benchmark_names=[benchmark_name],
        actual_benchmark_tiers=["canonical" if canonical_comparable else "validation"],
        execution_mode="local_python",
        imports_ok=["numpy"],
        experiments=[],
        summary=f"Execution for {problem_id}",
        recommended_next_step="Review claim evidence.",
        artifact_path=f"artifacts/{problem_id}.json",
        status="completed",
    )


def _problem_study_report(workspace: str, problem_id: str, *, canonical_comparable: bool) -> ProblemStudyReport:
    workspace_path = Path(workspace)
    return ProblemStudyReport(
        problem_id=problem_id,
        problem=f"Problem {problem_id}",
        profile_id="deep_learning",
        domain="deep_learning",
        resolution_confidence=0.8,
        canonical_comparable=canonical_comparable,
        benchmark_ids=[f"{problem_id}-benchmark"],
        benchmark_names=[f"Benchmark {problem_id}"],
        actual_benchmark_tiers=["canonical" if canonical_comparable else "validation"],
        environment=ScienceEnvironmentBundle(
            problem_id=problem_id,
            problem=f"Problem {problem_id}",
            profile_id="deep_learning",
            domain="deep_learning",
            profile_hash="hash",
            docker_image_tag="tar-science:locked",
            build_context_path=str(workspace_path),
            dockerfile_path=str(workspace_path / "Dockerfile"),
            requirements_path=str(workspace_path / "requirements.txt"),
            study_plan_path=str(workspace_path / f"{problem_id}-study_plan.json"),
            execution_report_path=str(workspace_path / f"{problem_id}-execution_report.json"),
        ),
        next_action="Run validation benchmarks.",
    )


def _configure_orchestrator(orchestrator: TAROrchestrator, hits: list[MemorySearchHit] | None = None) -> None:
    if orchestrator.vault is not None:
        orchestrator.vault.close()
    orchestrator.vault = _FakeVault(hits)
    orchestrator.memory_indexer = None
    orchestrator._claim_policy = lambda: ClaimAcceptancePolicy(min_supporting_sources=0)  # type: ignore[method-assign]


def test_claim_verdict_is_not_affected_by_unrelated_later_problem_study():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _configure_orchestrator(orchestrator)
            orchestrator.store.append_verification_report(_verification_report())

            first = orchestrator.claim_verdict(trial_id="trial-1")
            orchestrator.store.append_problem_study(
                _problem_study_report(tmp, "unrelated-problem", canonical_comparable=True)
            )
            second = orchestrator.claim_verdict(trial_id="trial-1")

            assert first.status == second.status
            assert second.canonical_comparability_source == "none"
            assert second.canonical_benchmark_satisfied is False
            assert second.benchmark_problem_id is None
        finally:
            orchestrator.shutdown()


def test_claim_verdict_is_not_affected_by_unrelated_later_problem_execution():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _configure_orchestrator(orchestrator)
            orchestrator.store.append_verification_report(_verification_report())

            first = orchestrator.claim_verdict(trial_id="trial-1")
            orchestrator.store.append_problem_execution(
                _problem_execution_report(
                    "other-problem",
                    canonical_comparable=True,
                    benchmark_id="other-canonical",
                    benchmark_name="Other Canonical",
                )
            )
            second = orchestrator.claim_verdict(trial_id="trial-1")

            assert first.status == second.status
            assert second.canonical_comparability_source == "none"
            assert second.canonical_benchmark_satisfied is False
            assert second.benchmark_problem_id is None
        finally:
            orchestrator.shutdown()


def test_claim_verdict_uses_explicit_problem_execution_not_latest_unrelated_execution():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _configure_orchestrator(orchestrator)
            orchestrator.store.append_verification_report(_verification_report())
            orchestrator.store.append_problem_execution(
                _problem_execution_report(
                    "problem-a",
                    canonical_comparable=False,
                    benchmark_id="bench-a",
                    benchmark_name="Benchmark A",
                )
            )
            orchestrator.store.append_problem_execution(
                _problem_execution_report(
                    "problem-b",
                    canonical_comparable=True,
                    benchmark_id="bench-b",
                    benchmark_name="Benchmark B",
                )
            )

            verdict = orchestrator.claim_verdict(trial_id="trial-1", problem_id="problem-a")

            assert verdict.benchmark_problem_id == "problem-a"
            assert verdict.canonical_comparability_source == "problem_execution"
            assert verdict.canonical_benchmark_satisfied is False
            assert verdict.supporting_benchmark_ids == ["bench-a"]
            assert verdict.supporting_benchmark_names == ["Benchmark A"]
        finally:
            orchestrator.shutdown()


def test_claim_verdict_returns_insufficient_evidence_for_unknown_problem_id():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _configure_orchestrator(orchestrator)
            orchestrator.store.append_verification_report(_verification_report())

            verdict = orchestrator.claim_verdict(trial_id="trial-1", problem_id="missing-problem")

            assert verdict.status == "insufficient_evidence"
            assert verdict.verdict_inputs_complete is False
            assert verdict.canonical_benchmark_satisfied is False
            assert verdict.linkage_note is not None
            assert "missing-problem" in verdict.linkage_note
        finally:
            orchestrator.shutdown()


def test_claim_verdict_provenance_fields_are_populated():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            hit = MemorySearchHit(
                document_id="research:paper-1",
                score=0.91,
                document="Thermodynamic evidence supports the mechanism.",
                metadata={
                    "kind": "research",
                    "paper_title": "Thermodynamic Evidence",
                    "claim_id": "claim-1",
                    "page_number": 3,
                    "source_excerpt": "Thermodynamic evidence supports the mechanism.",
                },
            )
            _configure_orchestrator(orchestrator, hits=[hit])
            orchestrator.store.append_verification_report(_verification_report())
            orchestrator.store.append_problem_execution(
                _problem_execution_report(
                    "problem-a",
                    canonical_comparable=True,
                    benchmark_id="bench-a",
                    benchmark_name="Benchmark A",
                )
            )

            verdict = orchestrator.claim_verdict(trial_id="trial-1", problem_id="problem-a")
            decision = orchestrator.store.latest_research_decision(
                mode="claim_review",
                trial_id="trial-1",
                problem_id="problem-a",
            )

            assert verdict.decision_scope == "trial_local"
            assert verdict.verification_report_trial_id == "trial-1"
            assert verdict.evidence_bundle_id is not None
            assert verdict.linkage_status == "exact"
            assert verdict.verdict_inputs_complete is True
            assert verdict.supporting_research_ids == ["research:paper-1"]
            assert verdict.supporting_evidence_ids == ["research:paper-1"]
            assert verdict.canonical_comparability_source == "problem_execution"
            assert decision is not None
            assert decision.trial_id == "trial-1"
            assert decision.problem_id == "problem-a"
            assert decision.claim_verdict_id == verdict.verdict_id
        finally:
            orchestrator.shutdown()
