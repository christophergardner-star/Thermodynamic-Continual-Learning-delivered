# Phase 5 Roadmap

This document is the authoritative post-audit roadmap for:

- Phase 4 completion
- Phase 5 full autonomy

It supersedes the older linear `WS32 -> WS33 -> WS34 -> WS35` reading by
making the loop-closure and autonomy dependencies explicit.

## Core Reframe

The audit changed the dependency graph.

Building stronger retrieval before loop closure would only produce better
retrieval that still does not shape planning. Building durable runtime before
evidence debt becomes a hard scheduling gate would automate a system that can
still schedule around unresolved evidence gaps.

The correct order is:

1. `WS32-close`
2. `WS32.5`
3. `WS33`
4. `WS34-pre`
5. `WS34`
6. `WS35`
7. independent eval validation pass
8. `WS36`
9. `WS37`
10. `WS38`
11. `WS39`

## Phase 4 Completion

### `WS32-close` — Literature Engine Deepening

Status:

- local closeout work already exists
- no pod
- single-commit scope

What is already present in the local diff:

- stable paper identity via SHA256 content fingerprinting
- `LiteratureIngestManifest` persistence under `tar_state/literature/manifests`
- deduplication in `ingest_paths()` across different source paths with
  identical content
- richer artifact structure for sections, tables, figures, and claims
- operator/control/CLI/dashboard literature inspection surfaces

Required close fixes:

1. Conflict detection threshold and scope guard

- file: `tar_lab/literature_engine.py`
- minimum shared-token floor: `>= 4`
- add `_is_genuine_contradiction(left_claim, right_claim)` style logic
- same-topic but same-negation framing must not be labeled contradictory unless
  polarity and scope actually oppose
- required regression: same-topic/non-contradictory papers must not produce a
  conflict

2. Retrieval telemetry surface

- `ProblemStudyReport.retrieval_mode` already exists and is recorded
- remaining requirement: operator aggregates must surface a retrieval-mode
  breakdown across recent studies
- add `retrieval_mode_breakdown` to operator view / project-status surfaces

Acceptance criteria:

- duplicate content from two paths deduplicates with
  `deduplicated_existing = 1`
- non-contradictory same-topic papers do not appear in the conflict report
- literature status exposes manifest / artifact / conflict counts
- operator view surfaces `retrieval_mode_breakdown`
- regression suite passes

Commit title:

- `Close WS32 literature engine deepening`

### `WS32.5` — Loop Closure

Status:

- inserted workstream
- laptop-only

Context:

- `LiteraturePolicySignal` exists
- `_distil_literature_policy_signal()` exists
- `RuleDirector.propose()` already accepts and uses `literature_signal`

Three remaining gaps:

1. Literature signal not passed from `plan_trial()`

- file: `tar_lab/orchestrator.py`
- after `memory_hits = self._retrieve_strategy_hits()`, call
  `_distil_literature_policy_signal(...)`
- pass `literature_signal=` into `produce_bundle(...)`

Acceptance:

- seeded drift-heavy literature causes `DirectorPolicy.literature_signal` to be
  non-null
- recommended family becomes `ou_drift_jitter` when confidence is high enough

2. WS27R2 operator not routed into hierarchy

- file: `tar_lab/hierarchy.py`
- add `_role_config_from_serving_state(role, workspace)`
- resolve role configs from:
  - operator serving state
  - role assignment state
  - healthy endpoint registry
- priority order:
  - explicit env vars
  - serving state
  - `None`

Acceptance:

- mock healthy serving state resolves to `LocalLLMConfig`
- hierarchy can activate live roles without env-only wiring

3. Evidence debt as hard scheduling gate

- file: `tar_lab/orchestrator.py`
- `allocate_budget()` must refuse non-repair work when
  `promotion_blocked = True`
- only repair/falsification/replication actions may proceed

Acceptance:

- non-repair actions are not scheduled for promotion-blocked projects
- repair actions still schedule correctly

Commit title:

- `Implement WS32.5 loop closure`

### `WS33` — Scientific Retrieval And Claim Graph

Status:

- laptop-first
- no pod for core work

Deliverable 1: contradiction-aware retrieval scoring

- file: `tar_lab/memory/vault.py`
- extend `ScientificReranker.rerank()`
- add `source_confidence` to literature metadata
- contradictory but high-confidence evidence must be surfaced intentionally,
  not buried
- add `contradiction_surfaced` to returned hit metadata

Target scoring shift:

- existing: dense + lexical + coverage + kind/evidence bonuses
- revised: add explicit `source_confidence` weighting and contradiction
  surfacing behavior

Deliverable 2: claim-graph indexing

- index `ClaimCluster` as `claim_cluster`
- index `ClaimConflict` as `claim_conflict`
- orchestrator paper ingest must index these graph objects into the vault

Deliverable 3: evidence-budgeted retrieval

- `study_problem()` must increase falsification pressure when contradictory
  retrieval is surfaced
- add `retrieval_conflict_count` to `ProblemStudyReport`

Deliverable 4: hard-refuse degraded studies

- lexical fallback must not stay silent
- degraded studies should be marked `retrieval_degraded`
- operator surfaces should count degraded studies explicitly

Acceptance criteria:

- source confidence computed and stored
- contradicting claim pairs can both surface in top-k
- claim clusters/conflicts are searchable directly
- contradiction-bearing retrieval increases falsification pressure
- degraded studies surface explicit degraded status and warning

### `WS34-pre` — Claim Verdict Lifecycle

Status:

- small focused pass
- laptop-only

Deliverable:

- age unresolved verdicts and escalate them automatically

Required additions:

- `review_required_before`
- `escalated_at`
- `escalation_reason`
- `lifecycle_status`

Runtime integration:

- `_age_claim_verdicts()` runs during runtime heartbeat
- escalated verdicts create follow-up open questions
- runtime policy holds configurable aging threshold

Acceptance criteria:

- aged provisional / insufficient-evidence verdicts escalate automatically
- escalations create project-visible follow-up questions
- runtime heartbeat reports escalated verdict IDs

### `WS34` — Durable Lab Runtime

Status:

- laptop-first
- pod required for realistic crash/recovery validation

Deliverable 1: `recoverable_crash` state

- mid-execution crash states must not collapse into generic retry behavior
- explicit confirmation required before retry

Deliverable 2: lease heartbeat

- long-running jobs must renew lease state during execution

Deliverable 3: orphan run recovery

- detect dead workers before generic lease expiry
- transition to `recoverable_crash` with provenance

Deliverable 4: queue health surface

- expose counts by queue/crash/retry state
- wire into CLI/dashboard

Acceptance criteria:

- OOM/crash enters `recoverable_crash`
- no auto-retry without explicit recovery confirmation
- lease heartbeat updates during long runs
- orphan detection works
- queue health reports truthful counts

### `WS35` — Safe Execution Hardening

Status:

- laptop-first
- pod required for Docker hardening validation

Deliverable 1: universal sandbox policy

- all generated/external code execution paths must go through the sandbox or be
  explicitly documented exceptions
- add `TARExecutionPolicy`
- raise `ExecutionPolicyViolation` on unsandboxed prohibited paths

Deliverable 2: capability drops and seccomp

- `--cap-drop=ALL`
- `no-new-privileges`
- configurable seccomp profile
- default restrictive profile committed in-repo

Deliverable 3: read-only workspace policy

- workspace root mounted read-only
- only sandbox tmpfs remains writable
- audit log records actual applied policy

Acceptance criteria:

- all execution paths enforce or declare sandbox policy
- privileged syscalls are blocked by default seccomp policy
- workspace writes fail inside sandbox when policy is read-only
- sandbox audit surface records applied protections

### Independent Eval Validation Pass

Status:

- methodological obligation
- laptop-only

Deliverables:

1. sealed external eval slice
2. independent scorer
3. comparison doc against internal WS27R2 metrics
4. adjustment rule if external performance diverges materially

Acceptance criteria:

- sealed external pack exists with content-hash manifest
- WS27R2 is evaluated on it
- comparison is written honestly
- closeout docs are corrected if internal score is inflated

## Phase 5 Full Autonomy

### `WS36` — Frontier Gap Scanner

Purpose:

- identify open research gaps from ingested literature
- propose projects from those gaps instead of waiting for human-supplied
  problems

Core elements:

- `FrontierGapRecord`
- `FrontierGapScanReport`
- novelty scoring against existing projects
- domain-alignment check against known science profiles
- proposal pipeline into `create_project(..., status=\"proposed\")`

Acceptance criteria:

- seeded literature produces at least one real gap
- duplicate/similar gaps are rejected
- domain-misaligned gaps are rejected
- proposal / promotion / rejection flows work end-to-end

Pod:

- no

### `WS37` — Generative Director

Purpose:

- replace the fixed 3-family ceiling with operator-backed, approval-gated
  family generation

Core elements:

- `ProposedExperimentFamily`
- `GenerativeDirectorProposal`
- `GenerativeDirector`
- approval workflow with feasibility probe
- approved families enter the known family pool

Important dependency:

- relies on WS32.5 serving-state hierarchy routing

Acceptance criteria:

- novel family proposal creates a pending record
- feasibility probe governs approval
- approved families become reusable in future planning
- fallback to rule-based Director remains intact

Pod:

- only for real feasibility probes of new families

### `WS38` — Self-Improvement Loop

Purpose:

- let TAR curate training signal from its own research runs and improve the
  operator line conservatively

Core elements:

- `TrainingSignalRecord`
- `CuratedDeltaRecord`
- `RetrainRecord`
- `FrozenAnchorPackManifest`
- `SelfImprovementEngine`
- pause/failure/cycle limits enforced in runtime policy

Non-negotiable safeguards:

- frozen anchor pack is hash-verified and immutable
- no anchor-pack overlap in curated deltas
- zero overclaim is a hard gate
- diversity enforcement on curated deltas
- three consecutive failures pause the loop
- maximum automatic deployment cycles require human review

Acceptance criteria:

- full cycle completes:
  curate -> probe -> gate -> run1 -> gate -> deploy
- anchor pack is verified at every eval
- zero-overclaim invariant is enforced in code
- WS26 regression gate still passes

Pod:

- required for probe / run1 training and eval

### `WS39` — Autonomous Research Agenda

Purpose:

- let TAR set and execute its own agenda rather than only reacting to manually
  created projects

Dependencies:

- `WS36`
- `WS37`
- `WS38`

Core elements:

- `AgendaDecisionRecord`
- `AgendaReviewConfig`
- `AgendaEngine`
- scheduled agenda review jobs
- veto window
- operator-backed tie-breaking for competing proposals
- agenda decisions recycled into future training signal

Acceptance criteria:

- full autonomous cycle completes after initial configuration:
  ingest -> gap scan -> propose -> promote -> study -> evaluate ->
  update evidence -> feed training signal -> update agenda
- veto window is respected
- max active project cap is enforced
- stale projects park automatically

Pod:

- no for agenda engine itself

## Execution Dependency Map

`WS32-close`
-> `WS32.5`
-> `WS33`
-> `WS34-pre`
-> `WS34`
-> `WS35`
-> independent eval validation pass
-> `WS36`
-> `WS37`
-> `WS38`
-> `WS39`

Additional dependency notes:

- `WS36` requires `WS33`
- `WS37` requires `WS32.5`
- `WS38` requires `WS37`
- `WS39` requires `WS36 + WS37 + WS38`

## New Files Summary

| File | WS | Purpose |
| --- | --- | --- |
| `tar_lab/self_improvement.py` | `WS38` | `SelfImprovementEngine` |
| `tar_lab/agenda.py` | `WS39` | `AgendaEngine` |
| `tar_lab/sandbox_profiles/default_seccomp.json` | `WS35` | default seccomp policy |
| `freeze_anchor_pack.py` | `WS38` | one-time frozen anchor pack initialiser |
| `eval_tar_operator_external.py` | eval pass | independent scorer |
| `tests/test_frontier_gap_scanner.py` | `WS36` | gap scanner tests |
| `tests/test_generative_director.py` | `WS37` | generative director tests |
| `tests/test_self_improvement.py` | `WS38` | self-improvement loop tests |
| `tests/test_agenda_engine.py` | `WS39` | agenda engine tests |
| `tests/test_autonomous_cycle.py` | `WS39` | full-loop integration test |

## Completion Picture

When this roadmap is complete, TAR will be able to:

- read new literature and identify open problems
- create and promote research projects with bounded human oversight
- design experiment strategies through the operator rather than a fixed
  three-family ceiling
- execute, evaluate, falsify, and reprioritize based on evidence
- curate its own runs into training signal
- retrain and redeploy itself under a frozen external standard
- update its own agenda based on scientific outcomes rather than manual
  prompting
