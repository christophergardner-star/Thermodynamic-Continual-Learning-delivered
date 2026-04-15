# WS34-pre Closeout

## Title

`WS34-pre: Claim Verdict Lifecycle`

## Outcome

`WS34-pre` is complete.

The workstream added a real lifecycle to unresolved claim verdicts before the
runtime becomes more autonomous in `WS34`.

## Delivered

Lifecycle state:

- `ClaimVerdict` now carries:
  - `review_required_before`
  - `escalated_at`
  - `escalation_reason`
  - `lifecycle_status`

Runtime policy:

- persisted `TARRuntimePolicy` now governs verdict aging through
  `verdict_aging_days`
- runtime policy is surfaced in runtime/operator status

Automation:

- `_age_claim_verdicts()` escalates overdue `provisional` and
  `insufficient_evidence` verdicts
- escalated verdicts create project-visible `unresolved_verdict` follow-up
  questions
- `run_runtime_cycle()` now reports escalated verdict IDs in the runtime
  heartbeat
- project staleness is raised when aging/escalated verdicts exist

Operator/runtime surfacing:

- operator view now shows verdict lifecycle counts and escalated verdict IDs
- runtime status now shows runtime policy, verdict lifecycle counts, and
  escalated verdict IDs
- dashboard/operator surfaces reflect those counts directly

## Validation

Passed before closeout:

- `python -m pytest tests\test_claim_verdict_integrity.py tests\test_prioritization.py tests\test_runtime_daemon.py tests\test_operator_interface.py -q`
- `python -m pytest tests\test_tar_foundation.py -k "frontier_status_reports_new_foundations" -q`

Result summary:

- claim-verdict/runtime/operator slice: `21 passed`
- frontier/foundation regression: `1 passed`

Compile checks passed on:

- `tar_lab/schemas.py`
- `tar_lab/state.py`
- `tar_lab/orchestrator.py`
- `dashboard.py`
- `tar_cli.py`

## Pod Posture

No pod was needed.

`WS34-pre` remained correctly laptop-first. Pod-backed validation belongs to
`WS34`, not to this lifecycle pre-pass.
