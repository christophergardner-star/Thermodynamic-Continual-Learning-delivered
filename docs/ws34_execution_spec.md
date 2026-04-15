# WS34 Execution Spec

## Title

`WS34: Durable Lab Runtime`

## Objective

Make TAR's runtime durable enough to recover truthfully from mid-execution
failures before any pod-backed crash validation begins.

This workstream follows `WS34-pre` and assumes:

- evidence debt is already a hard scheduling gate
- unresolved verdicts already age and escalate

## First Slice

The first local slice covers four bounded deliverables:

1. `recoverable_crash` schedule state with explicit `confirm_recovery()`
2. lease heartbeats during long-running execution
3. orphaned-run detection in the runtime cycle
4. queue-health surfacing through runtime status, CLI, and dashboard

## Required Changes

Scheduler state:

- add `recoverable_crash` to `ProblemScheduleEntry.status`
- persist crash provenance, crash timestamp, recovery-required flag, and
  recovery confirmation timestamp
- refuse generic retry for recoverable crashes until explicit confirmation

Lease discipline:

- extend `RuntimeLease` with `heartbeat_interval_s`
- renew lease heartbeat and expiry during long-running execution

Orphan recovery:

- detect `running` / `leased` entries whose heartbeat is stale
- if the schedule is no longer actively owned, move it to
  `recoverable_crash`
- emit alert and audit event

Operator surface:

- expose `queue_health()` from the orchestrator
- include queue health inside runtime status
- wire queue health and recovery confirmation through control and CLI
- surface recoverable crash / orphan / stale lease counts on the dashboard

## Acceptance Criteria

- crash-like exceptions produce `recoverable_crash`, not `retry_wait`
- recoverable crashes require `confirm_recovery()` before requeue
- long-running jobs renew lease heartbeat while executing
- orphaned runs transition to `recoverable_crash`
- queue health reports truthful counts for scheduled, leased, running,
  recoverable-crash, retry-wait, and failed-terminal entries
- focused runtime/operator regressions pass locally

## Pod Posture

No pod for this slice.

Pod use is only justified after the local runtime machinery is stable and the
remaining work is specifically real crash/recovery validation.
