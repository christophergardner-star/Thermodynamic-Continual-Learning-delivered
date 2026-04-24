import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    AblationResult,
    BenchmarkExecutionStatisticalSummary,
    BenchmarkMetricStatistic,
    BenchmarkStatisticalSummary,
    BreakthroughReport,
    CalibrationBin,
    CalibrationReport,
    ClaimAcceptancePolicy,
    ClaimVerdict,
    MemorySearchHit,
    ProblemExecutionReport,
    ProblemExperimentResult,
    ProblemStudyReport,
    RuntimeHeartbeat,
    ScienceEnvironmentBundle,
    SeedRunResult,
    SeedVarianceReport,
    TARRuntimePolicy,
    VerificationReport,
)


class _FakeVault:
    def __init__(self, hits: list[MemorySearchHit] | None = None):
        self._hits = hits or []

    def search(self, *args, **kwargs):
        return list(self._hits)

    def index_verification_report(self, *args, **kwargs) -> None:
        return None

    def index_breakthrough_report(self, *args, **kwargs) -> None:
        return None

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


def _rich_problem_execution_report(problem_id: str = "problem-qml") -> ProblemExecutionReport:
    experiment = ProblemExperimentResult(
        template_id="ansatz_depth_sweep",
        name="Ansatz Depth Sweep",
        benchmark="depth_trainability_curve",
        benchmark_id="pennylane_barren_plateau_canonical",
        benchmark_name="PennyLane Barren Plateau Benchmark",
        benchmark_tier="canonical",
        requested_benchmark_tier="canonical",
        executed_benchmark_tier="canonical",
        benchmark_truth_status="canonical_ready",
        benchmark_alignment="aligned",
        dataset_or_env="pennylane:qml_barren_plateau_suite",
        canonical_comparable=True,
        provenance_complete=True,
        proxy_benchmark_used=False,
        execution_mode="pennylane_backend",
        status="completed",
        metrics={
            "gradient_norm_variance": 0.0019,
            "barren_plateau_slope": 0.105,
            "trainability_gap": 0.98,
        },
        statistical_summary=BenchmarkStatisticalSummary(
            statistically_ready=True,
            sample_count=5,
            recommended_seed_runs=5,
            primary_metric="trainability_gap",
            metrics=[
                BenchmarkMetricStatistic(
                    metric_name="trainability_gap",
                    mean=0.98,
                    std_dev=0.02,
                    ci95_low=0.96,
                    ci95_high=1.0,
                    sample_count=5,
                )
            ],
        ),
    )
    return ProblemExecutionReport(
        problem_id=problem_id,
        problem="Investigate barren plateaus in quantum AI",
        profile_id="quantum_ml",
        domain="quantum_ml",
        benchmark_tier="canonical",
        requested_benchmark="pennylane_barren_plateau_canonical",
        canonical_comparable=True,
        proxy_benchmarks_used=False,
        benchmark_ids=["pennylane_barren_plateau_canonical"],
        benchmark_names=["PennyLane Barren Plateau Benchmark"],
        actual_benchmark_tiers=["canonical"],
        benchmark_truth_statuses=["canonical_ready"],
        benchmark_alignment="aligned",
        execution_mode="local_python",
        imports_ok=["numpy", "pennylane"],
        experiments=[experiment],
        benchmark_statistical_summary=BenchmarkExecutionStatisticalSummary(
            experiment_count=1,
            completed_experiment_count=1,
            statistically_ready_experiment_count=1,
            canonical_ready_completed_count=1,
            statistically_ready=True,
            notes=[],
        ),
        summary="Canonical PennyLane benchmark completed cleanly with bounded seed variance.",
        recommended_next_step="Promote the benchmark-backed signal for claim review.",
        artifact_path=f"artifacts/{problem_id}.json",
        status="completed",
    )


def _problem_study_report(
    workspace: str,
    problem_id: str,
    *,
    canonical_comparable: bool,
    project_id: str | None = None,
    thread_id: str | None = None,
) -> ProblemStudyReport:
    workspace_path = Path(workspace)
    return ProblemStudyReport(
        problem_id=problem_id,
        project_id=project_id,
        thread_id=thread_id,
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


def test_problem_execution_claim_verdict_without_payload_trial_uses_benchmark_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            hit = MemorySearchHit(
                document_id="research:paper-1",
                score=0.91,
                document="Quantum literature supports the trainability claim.",
                metadata={"kind": "research"},
            )
            _configure_orchestrator(orchestrator, hits=[hit])
            orchestrator.store.append_problem_execution(_rich_problem_execution_report("problem-qml"))

            verdict = orchestrator.claim_verdict(problem_id="problem-qml")

            assert verdict.trial_id == "problem_execution:problem-qml"
            assert verdict.verification_report_trial_id == "problem_execution:problem-qml"
            assert verdict.benchmark_problem_id == "problem-qml"
            assert verdict.canonical_comparability_source == "problem_execution"
            assert verdict.linkage_status == "exact"
            assert verdict.status in {"accepted", "provisional"}
        finally:
            orchestrator.shutdown()


def test_problem_execution_breakthrough_report_uses_problem_execution_path():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            hit = MemorySearchHit(
                document_id="research:paper-1",
                score=0.91,
                document="Quantum literature supports the trainability claim.",
                metadata={"kind": "research"},
            )
            _configure_orchestrator(orchestrator, hits=[hit])
            orchestrator.store.append_problem_execution(_rich_problem_execution_report("problem-qml"))

            report = orchestrator.breakthrough_report(problem_id="problem-qml")

            assert report.trial_id == "problem_execution:problem-qml"
            assert report.verification.trial_id == "problem_execution:problem-qml"
            assert report.claim_verdict is not None
            assert report.claim_verdict.benchmark_problem_id == "problem-qml"
            assert report.status in {"breakthrough", "candidate"}
        finally:
            orchestrator.shutdown()


def test_quantum_problem_execution_breakthrough_requires_phase14_gate():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            hit = MemorySearchHit(
                document_id="research:paper-1",
                score=0.91,
                document="Quantum literature supports the trainability claim.",
                metadata={"kind": "research"},
            )
            _configure_orchestrator(orchestrator, hits=[hit])
            orchestrator.store.append_problem_execution(_rich_problem_execution_report("problem-qml"))
            comparisons_dir = Path(tmp) / "tar_state" / "comparisons"
            comparisons_dir.mkdir(parents=True, exist_ok=True)
            (comparisons_dir / "phase14_quantum_publishability.json").write_text(
                json.dumps(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "problem_id": "problem-qml",
                        "publishability_status": "no_reviewer_grade_signal",
                    }
                ),
                encoding="utf-8",
            )

            report = orchestrator.breakthrough_report(problem_id="problem-qml")

            assert report.status == "candidate"
            assert "publishability_status=no_reviewer_grade_signal" in report.rationale
            assert "external_breakthrough_candidate=False" in report.rationale
            assert "reviewer-grade promotion remains open" in report.summary
        finally:
            orchestrator.shutdown()


def test_problem_execution_breakthrough_report_requires_external_assessment_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            hit = MemorySearchHit(
                document_id="research:paper-1",
                score=0.91,
                document="Thermodynamic literature supports the trainability claim.",
                metadata={"kind": "research"},
            )
            _configure_orchestrator(orchestrator, hits=[hit])
            execution = _rich_problem_execution_report("problem-deep")
            execution = execution.model_copy(update={"domain": "deep_learning"})
            orchestrator.store.append_problem_execution(execution)

            report = orchestrator.breakthrough_report(problem_id="problem-deep")

            assert report.status == "candidate"
            assert "external_assessment=missing" in report.rationale
            assert "external_breakthrough_candidate=False" in report.rationale
        finally:
            orchestrator.shutdown()


def test_problem_execution_breakthrough_report_promotes_when_external_assessment_passes():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            hit = MemorySearchHit(
                document_id="research:paper-1",
                score=0.91,
                document="Thermodynamic literature supports the trainability claim.",
                metadata={"kind": "research"},
            )
            _configure_orchestrator(orchestrator, hits=[hit])
            execution = _rich_problem_execution_report("problem-deep")
            execution = execution.model_copy(update={"domain": "deep_learning"})
            orchestrator.store.append_problem_execution(execution)
            verification = _verification_report("problem_execution:problem-deep")
            orchestrator.verification_runner.build_problem_execution_breakthrough_report = lambda *args, **kwargs: BreakthroughReport(
                trial_id=verification.trial_id,
                status="breakthrough",
                summary="Synthetic breakthrough for external gate test.",
                novelty_score=0.8,
                stability_score=0.9,
                calibration_score=0.9,
                supporting_research_ids=["research:paper-1"],
                rationale=["synthetic_base_report"],
                verification=verification,
                claim_verdict=kwargs.get("claim_verdict"),
            )
            comparisons_dir = Path(tmp) / "tar_state" / "comparisons"
            comparisons_dir.mkdir(parents=True, exist_ok=True)
            (comparisons_dir / "external_breakthrough_assessment.json").write_text(
                json.dumps(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "problem_id": "problem-deep",
                        "publishability_status": "reviewer_grade_candidate",
                        "external_breakthrough_candidate": True,
                    }
                ),
                encoding="utf-8",
            )

            report = orchestrator.breakthrough_report(problem_id="problem-deep")

            assert report.status == "breakthrough"
            assert "publishability_status=reviewer_grade_candidate" in report.rationale
            assert "external_breakthrough_candidate=True" in report.rationale
        finally:
            orchestrator.shutdown()


def test_age_claim_verdicts_escalates_old_provisional_and_creates_open_question():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _configure_orchestrator(orchestrator)
            project = orchestrator.create_project("Investigate aged verdict handling")
            thread = project.hypothesis_threads[0]
            study = _problem_study_report(
                tmp,
                "problem-aged",
                canonical_comparable=False,
                project_id=project.project_id,
                thread_id=thread.thread_id,
            )
            orchestrator.store.append_problem_study(study)
            orchestrator.store.save_runtime_policy(TARRuntimePolicy(verdict_aging_days=14))
            verdict = ClaimVerdict(
                verdict_id="verdict-aged-1",
                trial_id="trial-aged-1",
                created_at=(datetime.now(timezone.utc) - timedelta(days=15)).replace(microsecond=0).isoformat(),
                status="provisional",
                rationale=["Awaiting further evidence review."],
                policy=ClaimAcceptancePolicy(min_supporting_sources=0),
                verification_report_trial_id="trial-aged-1",
                benchmark_problem_id=study.problem_id,
                verdict_inputs_complete=True,
                linkage_status="exact",
            )
            orchestrator.store.append_claim_verdict(verdict)

            escalated = orchestrator._age_claim_verdicts()
            updated = next(
                item for item in orchestrator.store.iter_claim_verdicts() if item.verdict_id == verdict.verdict_id
            )
            refreshed_project = orchestrator.store.get_research_project(project.project_id)

            assert verdict.verdict_id in escalated
            assert updated.lifecycle_status == "escalated"
            assert updated.escalation_reason == "verdict_timeout"
            assert updated.escalated_at is not None
            assert updated.review_required_before is not None
            assert refreshed_project is not None
            assert any(item.uncertainty_type == "unresolved_verdict" for item in refreshed_project.open_questions)
        finally:
            orchestrator.shutdown()


def test_run_runtime_cycle_reports_escalated_verdict_ids():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _configure_orchestrator(orchestrator)
            project = orchestrator.create_project("Investigate runtime verdict escalation")
            thread = project.hypothesis_threads[0]
            study = _problem_study_report(
                tmp,
                "problem-runtime-aged",
                canonical_comparable=False,
                project_id=project.project_id,
                thread_id=thread.thread_id,
            )
            orchestrator.store.append_problem_study(study)
            orchestrator.store.save_runtime_policy(TARRuntimePolicy(verdict_aging_days=3))
            verdict = ClaimVerdict(
                verdict_id="verdict-runtime-aged-1",
                trial_id="trial-runtime-aged-1",
                created_at=(datetime.now(timezone.utc) - timedelta(days=4)).replace(microsecond=0).isoformat(),
                status="insufficient_evidence",
                rationale=["Evidence remains incomplete."],
                policy=ClaimAcceptancePolicy(min_supporting_sources=0),
                verification_report_trial_id="trial-runtime-aged-1",
                benchmark_problem_id=study.problem_id,
            )
            orchestrator.store.append_claim_verdict(verdict)
            orchestrator.runtime_daemon.run_cycle = lambda **kwargs: RuntimeHeartbeat(  # type: ignore[assignment]
                started_at="2026-04-15T10:00:00+00:00",
                finished_at="2026-04-15T10:00:01+00:00",
                status="completed",
                executed_jobs=0,
                stale_cleanups=0,
                failed_jobs=0,
                active_leases=0,
                retry_waiting=0,
                alert_count=0,
                notes=["runtime cycle started"],
            )

            heartbeat = orchestrator.run_runtime_cycle(max_jobs=1)
            persisted = orchestrator.runtime_daemon.load_heartbeat()

            assert verdict.verdict_id in heartbeat.escalated_verdicts
            assert persisted is not None
            assert verdict.verdict_id in persisted.escalated_verdicts
        finally:
            orchestrator.shutdown()
