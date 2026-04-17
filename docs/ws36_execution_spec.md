# WS36 Execution Spec

`WS36` opens Phase 5 by turning ingested literature into explicit frontier-gap
records and bounded project proposals.

## Scope Of This First Slice

This slice implements the scanner, proposal state machine, and first
operator-facing surfaces:

- `FrontierGapRecord`
- `FrontierGapScanReport`
- persisted frontier-gap and scan-report state under `tar_state/`
- orchestrator gap extraction from persisted `ResearchDocument.problem_statements`
- novelty scoring against existing projects
- domain-alignment rejection using science-profile resolution confidence and
  matched-keyword checks
- proposal flow into `create_project(..., status="proposed")`
- promotion flow from `proposed -> active`
- rejection flow from `proposed -> parked`
- control commands for scan/list/propose/promote/reject
- CLI renderers for frontier scans, frontier-gap status, and proposal actions
- dashboard frontier-gap panel
- aggregated frontier-gap status in `status()`, `operator_view()`, and
  `frontier_status()`
- filtered frontier-gap review by status and minimum confidence
- direct dashboard actions for scan / propose / promote / reject

## Implemented Files

- `tar_lab/schemas.py`
- `tar_lab/state.py`
- `tar_lab/orchestrator.py`
- `tests/test_frontier_gap_scanner.py`
- `tar_lab/control.py`
- `tar_cli.py`
- `dashboard.py`

## Core Design

### Gap Extraction

- source of truth: persisted `ResearchDocument.problem_statements`
- clustering: token-signature overlap across statements from distinct documents
- minimum evidence floor: at least `2` supporting documents per cluster

### Novelty Scoring

- compare cluster description against existing project title + goal text
- use semantic cosine when the vault embedder is semantic-ready
- otherwise fall back to lexical token overlap
- retain low-novelty candidates so they can be explicitly rejected as
  duplicates instead of silently disappearing

### Domain Alignment

`ScienceProfileRegistry.resolve_problem()` always returns a profile, including a
generic fallback. `WS36` therefore adds a stricter gate:

- at least one matched keyword required
- confidence must be at least `0.32`

If that gate fails, the gap is persisted as rejected with
`domain_profile_unresolved`.

### Proposal Flow

- only `identified` gaps above confidence threshold are eligible
- created projects are `status="proposed"` rather than active
- promotion marks the project `active`
- rejection marks the linked project `parked`

## Validation

Focused regression coverage:

- scan identifies a real gap from seeded literature
- duplicate gap is rejected against an existing project
- domain-misaligned cluster is rejected
- proposal, promotion, and rejection flows persist correctly

Continuity check:

- existing project continuity tests continue to pass with the extended project
  status set

## Next Slice

The next `WS36` slice should add richer interaction and policy surfaces:

- optional operator review notes on gap promotion/rejection
- frontier-gap history/log views beyond the latest scan snapshot
- richer CLI/control filtering and scan-history inspection
