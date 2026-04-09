from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from tar_lab.schemas import (
    AlertRecord,
    ProblemExecutionReport,
    ProblemScheduleEntry,
    ProblemStudyReport,
    RetryPolicy,
    RuntimeLease,
    SchedulerCycleReport,
    ScheduleStatus,
    utc_now_iso,
)
from tar_lab.state import TARStateStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso8601(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _schedule_id(problem_id: str) -> str:
    stamp = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"schedule-{problem_id}-{stamp}"


def _alert_id(prefix: str) -> str:
    return f"alert-{prefix}-{_utc_now().strftime('%Y%m%dT%H%M%S%fZ')}"


class ProblemStudyScheduler:
    def __init__(
        self,
        store: TARStateStore,
        execute_callback: Callable[[str, bool, bool], ProblemExecutionReport],
        worker_id: str = "runtime-daemon",
    ):
        self.store = store
        self.execute_callback = execute_callback
        self.worker_id = worker_id

    def schedule(
        self,
        study: ProblemStudyReport,
        *,
        use_docker: bool = False,
        build_env: bool = False,
        run_at: Optional[str] = None,
        delay_s: int = 0,
        repeat_interval_s: Optional[int] = None,
        max_runs: int = 1,
        priority: int = 0,
        priority_score: Optional[float] = None,
        priority_source: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> ProblemScheduleEntry:
        if run_at:
            next_run = _parse_iso8601(run_at)
        else:
            next_run = _utc_now() + timedelta(seconds=max(0, delay_s))
        entry = ProblemScheduleEntry(
            schedule_id=_schedule_id(study.problem_id),
            problem_id=study.problem_id,
            project_id=study.project_id,
            thread_id=study.thread_id,
            action_id=study.next_action_id,
            problem=study.problem,
            profile_id=study.profile_id,
            domain=study.domain,
            benchmark_tier=study.benchmark_tier,
            requested_benchmark=study.requested_benchmark,
            use_docker=use_docker,
            build_env=build_env,
            next_run_at=(next_run or _utc_now()).replace(microsecond=0).isoformat(),
            repeat_interval_s=repeat_interval_s,
            max_runs=max_runs,
            priority=priority,
            priority_score=priority_score,
            priority_source=priority_source,
            retry_policy=retry_policy or RetryPolicy(),
        )
        self.store.append_problem_schedule(entry)
        return entry

    def status(self) -> dict[str, object]:
        entries = list(self.store.iter_problem_schedules())
        return {
            "total": len(entries),
            "active": len([item for item in entries if item.status in {"scheduled", "leased", "running", "retry_wait"}]),
            "leased": len([item for item in entries if item.status == "leased"]),
            "retry_wait": len([item for item in entries if item.status == "retry_wait"]),
            "terminal_failures": len([item for item in entries if item.status == "failed_terminal"]),
            "entries": [item.model_dump(mode="json") for item in entries],
        }

    def run_once(
        self,
        *,
        max_jobs: int = 1,
        now: Optional[datetime] = None,
        lease_timeout_s: int = 300,
    ) -> SchedulerCycleReport:
        now_dt = (now or _utc_now()).astimezone(timezone.utc)
        due_entries = sorted(
            [
                entry
                for entry in self.store.iter_problem_schedules()
                if self._is_due(entry, now_dt)
            ],
            key=lambda item: (-item.priority, item.next_run_at, item.created_at),
        )
        completed_ids: list[str] = []
        rescheduled_ids: list[str] = []
        failed_ids: list[str] = []
        alert_ids: list[str] = []
        updated_entries: list[ProblemScheduleEntry] = []
        leased_count = 0
        retry_wait_count = 0

        for entry in due_entries[: max(1, max_jobs)]:
            leased = self._acquire_lease(entry, now_dt, lease_timeout_s=lease_timeout_s)
            if leased is None:
                continue
            leased_count += 1
            running = self.store.update_problem_schedule(
                leased.schedule_id,
                status="running",
                last_error=None,
                last_execution_at=utc_now_iso(),
                lease=leased.lease.model_copy(update={"heartbeat_at": utc_now_iso()}) if leased.lease else None,
            )
            if running is None:
                continue

            try:
                report = self.execute_callback(running.problem_id, running.use_docker, running.build_env)
                updated, entry_alerts = self._finalize_after_report(running, report, now_dt)
                alert_ids.extend(entry_alerts)
                if updated.status == "completed":
                    completed_ids.append(updated.schedule_id)
                elif updated.status == "retry_wait":
                    rescheduled_ids.append(updated.schedule_id)
                    retry_wait_count += 1
                elif updated.status == "scheduled":
                    rescheduled_ids.append(updated.schedule_id)
                else:
                    failed_ids.append(updated.schedule_id)
            except Exception as exc:
                updated, entry_alerts = self._finalize_after_exception(running, str(exc), now_dt)
                alert_ids.extend(entry_alerts)
                if updated.status == "retry_wait":
                    rescheduled_ids.append(updated.schedule_id)
                    retry_wait_count += 1
                else:
                    failed_ids.append(updated.schedule_id)

            updated_entries.append(updated)

        return SchedulerCycleReport(
            started_at=now_dt.replace(microsecond=0).isoformat(),
            finished_at=_utc_now().replace(microsecond=0).isoformat(),
            due_count=len(due_entries),
            executed_count=len(updated_entries),
            leased_count=leased_count,
            retry_wait_count=retry_wait_count,
            completed_schedule_ids=completed_ids,
            rescheduled_schedule_ids=rescheduled_ids,
            failed_schedule_ids=failed_ids,
            alert_ids=alert_ids,
            updated_entries=updated_entries,
        )

    def retry_failed_job(self, schedule_id: str) -> ProblemScheduleEntry:
        entry = self.store.get_problem_schedule(schedule_id)
        if entry is None:
            raise RuntimeError(f"Unknown schedule: {schedule_id}")
        updated = self.store.update_problem_schedule(
            schedule_id,
            status="scheduled",
            retry_after=None,
            lease=None,
            last_error=None,
            terminal_failure_reason=None,
        )
        if updated is None:
            raise RuntimeError(f"Unable to update schedule: {schedule_id}")
        return updated

    def cancel_job(self, schedule_id: str) -> ProblemScheduleEntry:
        entry = self.store.get_problem_schedule(schedule_id)
        if entry is None:
            raise RuntimeError(f"Unknown schedule: {schedule_id}")
        updated = self.store.update_problem_schedule(
            schedule_id,
            status="cancelled",
            lease=None,
        )
        if updated is None:
            raise RuntimeError(f"Unable to update schedule: {schedule_id}")
        return updated

    def recover_stale_leases(self, *, now: Optional[datetime] = None) -> list[ProblemScheduleEntry]:
        now_dt = (now or _utc_now()).astimezone(timezone.utc)
        recovered: list[ProblemScheduleEntry] = []
        for entry in list(self.store.iter_problem_schedules()):
            if entry.status not in {"leased", "running"} or entry.lease is None:
                continue
            expires = _parse_iso8601(entry.lease.expires_at)
            if expires is None or expires > now_dt:
                continue
            updated, _ = self._promote_retry_or_terminal(
                entry,
                error="stale_runtime_cleanup",
                now_dt=now_dt,
                severity="warning",
            )
            recovered.append(updated)
        return recovered

    def _is_due(self, entry: ProblemScheduleEntry, now_dt: datetime) -> bool:
        if entry.status == "scheduled":
            due_at = _parse_iso8601(entry.next_run_at)
            return due_at is not None and due_at <= now_dt
        if entry.status == "retry_wait":
            retry_at = _parse_iso8601(entry.retry_after or entry.next_run_at)
            return retry_at is not None and retry_at <= now_dt
        return False

    def _acquire_lease(self, entry: ProblemScheduleEntry, now_dt: datetime, *, lease_timeout_s: int) -> Optional[ProblemScheduleEntry]:
        if entry.status not in {"scheduled", "retry_wait"}:
            return None
        lease = RuntimeLease(
            owner_id=self.worker_id,
            expires_at=(now_dt + timedelta(seconds=max(30, lease_timeout_s))).replace(microsecond=0).isoformat(),
            attempt=entry.attempt_count + 1,
        )
        return self.store.update_problem_schedule(
            entry.schedule_id,
            status="leased",
            lease=lease,
            attempt_count=entry.attempt_count + 1,
            retry_after=None,
        )

    def _finalize_after_report(
        self,
        entry: ProblemScheduleEntry,
        report: ProblemExecutionReport,
        now_dt: datetime,
    ) -> tuple[ProblemScheduleEntry, list[str]]:
        run_count = entry.run_count + 1
        update_payload = {
            "run_count": run_count,
            "last_execution_at": utc_now_iso(),
            "last_report_path": report.artifact_path,
            "last_report_status": report.status,
            "last_summary": report.summary,
            "last_error": None if report.status == "completed" else report.summary,
            "last_manifest_path": report.manifest_path,
            "lease": None,
        }
        if report.status == "completed":
            should_repeat = entry.repeat_interval_s is not None and run_count < entry.max_runs
            if should_repeat:
                update_payload.update(
                    {
                        "status": "scheduled",
                        "next_run_at": (now_dt + timedelta(seconds=entry.repeat_interval_s)).replace(microsecond=0).isoformat(),
                    }
                )
                updated = self.store.update_problem_schedule(entry.schedule_id, **update_payload)
                if updated is None:
                    raise RuntimeError(f"Unable to update schedule: {entry.schedule_id}")
                return updated, []
            update_payload["status"] = "completed"
            updated = self.store.update_problem_schedule(entry.schedule_id, **update_payload)
            if updated is None:
                raise RuntimeError(f"Unable to update schedule: {entry.schedule_id}")
            return updated, []
        return self._promote_retry_or_terminal(entry, error=report.summary, now_dt=now_dt, report=report)

    def _finalize_after_exception(
        self,
        entry: ProblemScheduleEntry,
        error: str,
        now_dt: datetime,
    ) -> tuple[ProblemScheduleEntry, list[str]]:
        return self._promote_retry_or_terminal(entry, error=error, now_dt=now_dt, severity="error")

    def _promote_retry_or_terminal(
        self,
        entry: ProblemScheduleEntry,
        *,
        error: str,
        now_dt: datetime,
        report: Optional[ProblemExecutionReport] = None,
        severity: str = "error",
    ) -> tuple[ProblemScheduleEntry, list[str]]:
        retry_policy = entry.retry_policy
        can_retry = entry.attempt_count < retry_policy.max_attempts
        alert = AlertRecord(
            alert_id=_alert_id(entry.schedule_id),
            severity="warning" if can_retry else "critical",  # type: ignore[arg-type]
            source="scheduler",
            message=error,
            related_schedule_id=entry.schedule_id,
            related_manifest_id=(
                Path(report.manifest_path).stem
                if report is not None and report.manifest_path
                else (Path(entry.last_manifest_path).stem if entry.last_manifest_path else None)
            ),
            metadata={
                "status": "retry_wait" if can_retry else "failed_terminal",
                "attempt_count": entry.attempt_count,
                "problem_id": entry.problem_id,
                "severity_hint": severity,
            },
        )
        self.store.append_alert(alert)
        update_payload = {
            "last_execution_at": utc_now_iso(),
            "last_error": error,
            "last_report_path": report.artifact_path if report is not None else entry.last_report_path,
            "last_report_status": report.status if report is not None else entry.last_report_status,
            "last_summary": report.summary if report is not None else entry.last_summary,
            "last_manifest_path": report.manifest_path if report is not None else entry.last_manifest_path,
            "lease": None,
            "alert_ids": [*entry.alert_ids, alert.alert_id],
        }
        if can_retry:
            delay = min(
                retry_policy.max_delay_s,
                int(retry_policy.base_delay_s * (retry_policy.backoff_multiplier ** max(0, entry.attempt_count - 1))),
            )
            update_payload.update(
                {
                    "status": "retry_wait",
                    "retry_after": (now_dt + timedelta(seconds=delay)).replace(microsecond=0).isoformat(),
                }
            )
        else:
            update_payload.update(
                {
                    "status": "failed_terminal",
                    "terminal_failure_reason": error,
                }
            )
        updated = self.store.update_problem_schedule(entry.schedule_id, **update_payload)
        if updated is None:
            raise RuntimeError(f"Unable to update schedule: {entry.schedule_id}")
        return updated, [alert.alert_id]
