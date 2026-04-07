import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.scheduler import ProblemStudyScheduler
from tar_lab.schemas import ProblemExecutionReport, RetryPolicy
from tar_lab.state import TARStateStore


def _copy_science_profiles(tmp: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_scheduler_retries_then_fails_terminal_with_alerts():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem(
                "Investigate calibration in deep learning",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            scheduler = ProblemStudyScheduler(
                TARStateStore(tmp),
                execute_callback=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            entry = scheduler.schedule(
                study,
                retry_policy=RetryPolicy(max_attempts=2, base_delay_s=1, max_delay_s=1),
            )
            first = scheduler.run_once(now=datetime.now(timezone.utc), max_jobs=1)
            assert first.executed_count == 1
            updated = scheduler.store.get_problem_schedule(entry.schedule_id)
            assert updated is not None
            assert updated.status == "retry_wait"
            assert updated.alert_ids

            scheduler.store.update_problem_schedule(
                entry.schedule_id,
                retry_after=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            )
            second = scheduler.run_once(now=datetime.now(timezone.utc), max_jobs=1)
            assert second.executed_count == 1
            terminal = scheduler.store.get_problem_schedule(entry.schedule_id)
            assert terminal is not None
            assert terminal.status == "failed_terminal"
            assert terminal.terminal_failure_reason == "boom"
            assert len(list(scheduler.store.iter_alerts())) >= 2
        finally:
            orchestrator.shutdown()


def test_runtime_daemon_status_reports_retry_and_alert_counts():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.prepare_payload_environment()
            study = orchestrator.study_problem(
                "Investigate optimization stability in deep learning",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            report_path = Path(study.environment.execution_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report = ProblemExecutionReport(
                problem_id=study.problem_id,
                problem=study.problem,
                profile_id=study.profile_id,
                domain=study.domain,
                benchmark_tier=study.benchmark_tier,
                benchmark_ids=study.benchmark_ids,
                benchmark_names=study.benchmark_names,
                actual_benchmark_tiers=study.actual_benchmark_tiers,
                execution_mode="local_python",
                imports_ok=["numpy"],
                experiments=[],
                summary="Scheduled execution completed.",
                recommended_next_step="Proceed.",
                artifact_path=str(report_path),
                status="completed",
            )
            orchestrator.run_problem_study = lambda *args, **kwargs: report  # type: ignore[assignment]
            orchestrator.schedule_problem_study(problem_id=study.problem_id, delay_s=0, max_runs=1)
            heartbeat = orchestrator.run_runtime_cycle(max_jobs=1)
            runtime = orchestrator.runtime_status()
            assert Path(orchestrator.runtime_daemon.heartbeat_path).exists()
            assert heartbeat.executed_jobs == 1
            assert runtime["reproducibility_complete"] is True
            assert "network_policy" in runtime["sandbox_policy"]
        finally:
            orchestrator.shutdown()
