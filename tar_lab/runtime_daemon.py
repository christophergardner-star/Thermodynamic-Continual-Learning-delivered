from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Optional

from tar_lab.scheduler import ProblemStudyScheduler
from tar_lab.schemas import RuntimeHeartbeat
from tar_lab.state import TARStateStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LabRuntimeDaemon:
    def __init__(self, store: TARStateStore, scheduler: ProblemStudyScheduler):
        self.store = store
        self.scheduler = scheduler
        self.heartbeat_path = self.store.runtime_heartbeat_path

    def load_heartbeat(self) -> Optional[RuntimeHeartbeat]:
        if not self.heartbeat_path.exists():
            return None
        return RuntimeHeartbeat.model_validate_json(self.heartbeat_path.read_text(encoding="utf-8"))

    def run_cycle(
        self,
        *,
        max_jobs: int = 1,
        stale_after_s: int = 900,
    ) -> RuntimeHeartbeat:
        started = RuntimeHeartbeat(status="running", notes=["runtime cycle started"])
        self.heartbeat_path.write_text(started.model_dump_json(indent=2), encoding="utf-8")
        recovered = self.scheduler.recover_stale_leases(now=_utc_now())
        cycle = self.scheduler.run_once(max_jobs=max_jobs, now=_utc_now(), lease_timeout_s=max(30, stale_after_s))
        schedules = list(self.store.iter_problem_schedules())
        alerts = list(self.store.iter_alerts())
        final = RuntimeHeartbeat(
            started_at=started.started_at,
            finished_at=cycle.finished_at,
            status="failed" if cycle.failed_schedule_ids else "completed",
            executed_jobs=cycle.executed_count,
            stale_cleanups=len(recovered),
            failed_jobs=len(cycle.failed_schedule_ids),
            active_leases=len([item for item in schedules if item.status in {"leased", "running"}]),
            retry_waiting=len([item for item in schedules if item.status == "retry_wait"]),
            alert_count=len(alerts),
            notes=[
                f"due={cycle.due_count}",
                f"leased={cycle.leased_count}",
                f"retries={cycle.retry_wait_count}",
                f"alerts={len(cycle.alert_ids)}",
            ],
        )
        self.heartbeat_path.write_text(final.model_dump_json(indent=2), encoding="utf-8")
        return final

    def serve_forever(
        self,
        *,
        max_jobs: int = 1,
        stale_after_s: int = 900,
        poll_interval_s: float = 30.0,
        iterations: Optional[int] = None,
    ) -> RuntimeHeartbeat:
        last = self.load_heartbeat() or RuntimeHeartbeat()
        count = 0
        while iterations is None or count < iterations:
            last = self.run_cycle(max_jobs=max_jobs, stale_after_s=stale_after_s)
            count += 1
            if iterations is None or count < iterations:
                sleep(poll_interval_s)
        return last
