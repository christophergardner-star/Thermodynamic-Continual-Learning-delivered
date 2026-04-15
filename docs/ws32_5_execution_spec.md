# WS32.5 Execution Spec

## Title

`WS32.5: Loop Closure`

## Purpose

Insert the missing loop-closure workstream between `WS32` and `WS33`.

The Phase 4 audit found that TAR already had strong literature ingestion,
operator serving, and evidence-debt computation, but those components were not
yet wired into one closed-loop planning system:

- literature evidence did not influence `RuleDirector.propose()`
- the served operator line did not feed role resolution in `hierarchy.py`
- evidence debt penalized scheduling but did not hard-block non-remediation
  work

WS32.5 exists to close those gaps before `WS33` deepens retrieval and before
`WS34` automates more runtime decisions.

## Core Hypothesis

If TAR routes literature signal into Director policy, routes active operator
serving state into hierarchy role selection, and turns evidence debt into a
hard scheduling invariant, then the system becomes a genuine research loop
instead of a collection of disciplined but disconnected subsystems.

## Scope

### In Scope

- literature-policy signal distillation in the orchestrator
- typed literature signal on `DirectorPolicy`
- Director family selection biased by literature signal when pivot logic is not
  dominant
- hierarchy live-role fallback from serving state and role assignments
- hard evidence-debt gating in prioritization, budget allocation, and
  portfolio selection
- dedicated WS32.5 regression tests

### Out Of Scope

- contradiction-aware retrieval scoring itself
- claim-graph indexing
- verdict-aging state machines
- runtime crash recovery redesign
- pod use

## Deliverables

### 1. Literature To Director Policy

The orchestrator must:

- retrieve literature-policy hits at plan time
- distill those hits into a typed `LiteraturePolicySignal`
- pass the signal into hierarchy bundle construction

The Director must:

- persist the signal on `DirectorPolicy`
- use the signal to bias family selection when confidence is sufficient and
  fail-fast pivot is not already forced

### 2. Served Operator To Hierarchy Routing

Hierarchy live-role resolution must:

- continue to honor explicit env configuration first
- fall back to checkpoint / endpoint / role-assignment state when env config is
  absent
- resolve base URL and model from the active served operator path truthfully

This makes the validated operator line usable by Director / Strategist / Scout
without manual env-only wiring.

### 3. Evidence Debt Hard Gate

If `promotion_blocked == True`, TAR must only allow:

- falsification work
- replication work
- repair / review work

All non-remediation scheduling paths must be blocked at the scheduling layer,
not merely penalized in ranking.

## Acceptance Criteria

WS32.5 is acceptable only if all are true:

- literature signal is persisted on `DirectorPolicy`
- literature signal can change rule-based family selection in regression tests
- hierarchy can resolve live role configs from serving state without env vars
- non-remediation actions are blocked when evidence debt promotion remains
  blocked
- budget allocation will not auto-schedule blocked non-remediation work
- portfolio selection will not continue/resume blocked non-remediation work
- targeted WS32.5 regression tests pass

## Pod Policy

Do not use a pod for WS32.5.

This is a wiring and policy-invariant workstream. Local validation is the
correct tool.

## Execution Order

1. patch `WS32` close fixes needed by the audit
2. add typed literature signal support to the schema layer
3. wire literature-signal distillation into `plan_trial()`
4. wire serving-state fallback into `TriModelHierarchy`
5. enforce evidence-debt hard gates in ranking and selection paths
6. add dedicated regression coverage
7. only then open `WS33`
