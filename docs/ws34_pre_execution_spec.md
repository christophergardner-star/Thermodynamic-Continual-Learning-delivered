# WS34-pre Execution Spec

## Title

`WS34-pre: Claim Verdict Lifecycle`

## Purpose

Add lifecycle control to unresolved claim verdicts before `WS34` makes the
runtime more autonomous.

The specific problem is straightforward: unresolved `provisional` and
`insufficient_evidence` verdicts currently persist indefinitely unless a human
or later workflow rewrites them. That is not acceptable once the runtime is
making more scheduling decisions on its own.

## First Local Slice

The first local slice is deliberately narrow:

- add lifecycle fields to `ClaimVerdict`
- add a small persisted runtime policy for verdict-aging thresholds
- age and escalate overdue unresolved verdicts
- create project-visible follow-up questions when escalation happens
- surface escalated verdict IDs in the runtime heartbeat

This slice does not yet attempt the full `WS34` runtime durability work.

## Scope

### In Scope

- `ClaimVerdict` lifecycle fields
- `RuntimeHeartbeat.escalated_verdicts`
- persisted `TARRuntimePolicy.verdict_aging_days`
- orchestrator-side `_age_claim_verdicts()`
- runtime-cycle integration
- regression coverage for escalation behavior

### Out Of Scope

- recoverable crash states
- lease heartbeats
- orphan-run recovery
- queue health reporting
- pod-backed runtime validation

## Acceptance Criteria

The first `WS34-pre` slice is acceptable only if all are true:

- aged `provisional` and `insufficient_evidence` verdicts escalate
  automatically once overdue
- escalated verdicts gain `review_required_before`, `escalated_at`, and
  `escalation_reason`
- escalation creates a project-visible unresolved-verdict question when the
  verdict can be mapped back to a project
- `run_runtime_cycle()` reports escalated verdict IDs in the returned heartbeat
- the lifecycle logic is covered by targeted regression tests

## Pod Policy

Do not use a pod.

This is lifecycle/state work and should stay local.
