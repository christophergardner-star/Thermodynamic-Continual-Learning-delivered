import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.scheduler import ProblemStudyScheduler
from tar_lab.schemas import ProblemExecutionReport, RetryPolicy, RuntimeLease
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


def test_scheduler_marks_recoverable_crash_and_requires_confirmation():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem(
                "Investigate recoverable runtime crashes",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            scheduler = ProblemStudyScheduler(
                TARStateStore(tmp),
                execute_callback=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
            )
            entry = scheduler.schedule(
                study,
                retry_policy=RetryPolicy(max_attempts=3, base_delay_s=1, max_delay_s=1),
            )

            cycle = scheduler.run_once(now=datetime.now(timezone.utc), max_jobs=1)
            assert cycle.executed_count == 1
            updated = scheduler.store.get_problem_schedule(entry.schedule_id)
            assert updated is not None
            assert updated.status == "recoverable_crash"
            assert updated.recovery_required is True
            assert updated.crash_provenance == "CUDA out of memory"
            assert updated.crash_at is not None

            with pytest.raises(RuntimeError, match="awaiting explicit recovery confirmation"):
                scheduler.retry_failed_job(entry.schedule_id)

            confirmed = scheduler.confirm_recovery(entry.schedule_id)
            assert confirmed.status == "scheduled"
            assert confirmed.recovery_required is False
            assert confirmed.recovery_confirmed_at is not None
        finally:
            orchestrator.shutdown()


def test_scheduler_renews_lease_heartbeat_for_long_running_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        started = threading.Event()
        release = threading.Event()
        try:
            study = orchestrator.study_problem(
                "Investigate lease heartbeat renewal",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            report_path = Path(study.environment.execution_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            def _callback(*_args, **_kwargs):
                started.set()
                assert release.wait(timeout=5.0)
                return ProblemExecutionReport(
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
                    summary="Heartbeat test completed.",
                    recommended_next_step="Proceed.",
                    artifact_path=str(report_path),
                    status="completed",
                )

            scheduler = ProblemStudyScheduler(TARStateStore(tmp), execute_callback=_callback)
            entry = scheduler.schedule(study, retry_policy=RetryPolicy(max_attempts=2, base_delay_s=1, max_delay_s=1))

            worker = threading.Thread(
                target=lambda: scheduler.run_once(
                    now=datetime.now(timezone.utc),
                    max_jobs=1,
                    lease_timeout_s=60,
                    lease_heartbeat_interval_s=1,
                ),
                daemon=True,
            )
            worker.start()
            assert started.wait(timeout=2.0)
            initial = scheduler.store.get_problem_schedule(entry.schedule_id)
            assert initial is not None
            assert initial.lease is not None
            first_heartbeat = initial.lease.heartbeat_at
            time.sleep(1.4)
            renewed = scheduler.store.get_problem_schedule(entry.schedule_id)
            assert renewed is not None
            assert renewed.lease is not None
            assert renewed.lease.heartbeat_at != first_heartbeat
            release.set()
            worker.join(timeout=5.0)
            completed = scheduler.store.get_problem_schedule(entry.schedule_id)
            assert completed is not None
            assert completed.status == "completed"
        finally:
            release.set()
            orchestrator.shutdown()


def test_recover_orphaned_runs_marks_recoverable_crash():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem(
                "Investigate orphaned runtime recovery",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            entry = orchestrator.schedule_problem_study(problem_id=study.problem_id, delay_s=0, max_runs=1)
            now_dt = datetime.now(timezone.utc)
            orchestrator.store.update_problem_schedule(
                entry.schedule_id,
                status="running",
                lease=RuntimeLease(
                    owner_id="lost-worker",
                    heartbeat_at=(now_dt - timedelta(seconds=120)).isoformat(),
                    expires_at=(now_dt + timedelta(minutes=5)).isoformat(),
                    attempt=1,
                    heartbeat_interval_s=10,
                ),
            )
            orchestrator._active_process_commands = lambda: []  # type: ignore[method-assign]

            recovered = orchestrator.recover_orphaned_runs()
            updated = orchestrator.store.get_problem_schedule(entry.schedule_id)

            assert len(recovered) == 1
            assert updated is not None
            assert updated.status == "recoverable_crash"
            assert updated.recovery_required is True
            assert updated.crash_provenance == "orphan_detected"
            assert updated.crash_at is not None
        finally:
            orchestrator.shutdown()
