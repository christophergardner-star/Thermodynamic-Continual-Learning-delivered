# Post-Audit Remediation Roadmap

This is the authoritative roadmap after the full-system audit. Workstreams 1-7
are considered closed at the roadmap level. This document governs the
remediation phase required to bring the current stack from "implemented" to
"scientifically and operationally trustworthy."

## Program Basis

The estimates and sequencing below assume:

- one primary developer
- laptop-first implementation and test closure
- workstation / RunPod used only for final scale validation in `WS16`
- "done" includes code, tests, docs, and operator-surface truthfulness

Every remediation workstream closes only when:

- the code path is fixed
- the failure is covered by regression tests
- CLI / dashboard / reports reflect the new truth
- laptop-safe validation passes first
- workstation / RunPod is used for scale confirmation, not correctness excuses

## Current State

- `Workstreams 1-7`: closed at the roadmap level
- post-audit remediation required: `WS8-WS16`

## Milestone Table

| WS | Title | Priority | Depends On | Est. Effort | Why It Matters | Hard Acceptance Gate |
| --- | --- | --- | --- | --- | --- | --- |
| `WS8` | Vault Migration And Memory Integrity | `P0` | none | `2-4 days` | Fixes broken operator commands and restores TAR usability | `--status`, `--dry-run`, and `--study-problem` no longer crash on embedder mismatch; migration/rebuild path is explicit and tested |
| `WS11` | Claim Verdict Integrity | `P0` | none | `1-2 days` | Stops incorrect research verdicts from borrowing unrelated comparability state | verdicts are trial-local, benchmark-local, deterministic, and regression-tested |
| `WS12` | Reproducibility Hardening | `P0` | none | `1-2 days` | Prevents "locked" environments from drifting silently | no bare package names in locked manifests; manifest generation fails closed if versions are missing |
| `WS9` | Benchmark Truthfulness And Canonical Alignment | `P1` | `WS8`, `WS12` recommended | `4-7 days` | Aligns benchmark labels with actual executors | every canonical benchmark either has a real executor or is downgraded/refused; reports are externally honest |
| `WS10` | QML Canonical Closure | `P1` | `WS9` | `2-4 days` | Closes the strongest canonical/proxy contradiction | no canonical QML path uses `analytic_proxy`; missing quantum stack causes refusal |
| `WS13` | Runtime Sandbox Tightening | `P1` | `WS12` | `2-3 days` | Brings runtime writes in line with the stated safety model | production runs use read-only source mounts plus explicit artifact dirs; tests prove write-scope enforcement |
| `WS14` | Inference Safety And Observability | `P2` | `WS12` | `2-3 days` | Makes endpoint failures diagnosable and safer | endpoint logs retained, health failures surfaced, `trust_remote_code` explicit or removed |
| `WS15` | Test Suite Gap Closure | `P0` continuous, final gate | `WS8-WS14` | `2-4 days` | Prevents audit-class failures from recurring | every audited defect has a regression test that fails pre-fix and passes post-fix |
| `WS16` | Workstation / RunPod Validation | `P0` final sign-off | `WS8-WS15` | `3-6 days` | Confirms fixes survive real scale and real hardware | canonical runs, endpoint lifecycle, scheduler endurance, and claim integrity all validated on real infra |

## Critical Path

1. `WS8`
2. `WS11`
3. `WS12`
4. `WS9`
5. `WS10`
6. `WS13`
7. `WS14`
8. `WS15`
9. `WS16`

## Detailed Workstream Breakdown

### `WS8`: Vault Migration And Memory Integrity

**Purpose**

Repair the broken operator surface caused by vector-store and embedder
incompatibility.

**Deliverables**

- vector-store versioning
- embedder identity and dimension persistence
- mismatch detection before Chroma upsert/query
- safe migrate / rebuild / quarantine flow
- operator-safe repair messaging

**Main files**

- `tar_lab/memory/vault.py`
- `tar_lab/orchestrator.py`
- `tar_lab/state.py`
- `tar_cli.py`

**Tests**

- embedder mismatch recovery
- stale collection migration
- idempotent sync after migration

**Exit**

- live commands are usable again on this laptop

### `WS11`: Claim Verdict Integrity

**Purpose**

Make claim acceptance strictly trial-local and evidence-local.

**Deliverables**

- verdict inputs bound to exact trial/problem execution
- canonical comparability sourced only from that execution
- verdict provenance links

**Main files**

- `tar_lab/orchestrator.py`
- `tar_lab/verification.py`
- `tar_lab/schemas.py`
- `tar_lab/state.py`

**Tests**

- unrelated later studies do not alter an earlier verdict
- verdict provenance must include trial and benchmark references

**Exit**

- claim acceptance is machine-local, not global-state-local

### `WS12`: Reproducibility Hardening

**Purpose**

Make locked environments actually locked.

**Deliverables**

- strict dependency pinning
- hard failure when version metadata is missing
- reproducibility completeness becomes trustworthy

**Main files**

- `tar_lab/reproducibility.py`
- `tar_lab/docker_runner.py`
- `tar_lab/schemas.py`

**Tests**

- unpinned package rejection
- stable manifest hashing

**Exit**

- "locked" means fully pinned or refused

### `WS9`: Benchmark Truthfulness And Canonical Alignment

**Purpose**

Stop canonical benchmark names from overstating what the executor actually runs.

**Deliverables**

- benchmark truth table across all domains
- benchmark relabeling or executor upgrades
- canonical comparability tied to true executors only

**Main files**

- `tar_lab/science_exec.py`
- `tar_lab/science_profiles.py`
- `science_profiles/`
- `tar_lab/problem_runner.py`

**Tests**

- benchmark ID/executor consistency
- canonical refusal when a real suite is unavailable

**Exit**

- no domain overclaims canonical status

### `WS10`: QML Canonical Closure

**Purpose**

Finish the quantum path so no benchmark labeled canonical falls back to analytic
proxy behavior.

**Deliverables**

- real PennyLane/Qiskit-backed canonical noise benchmark
- no analytic proxy in canonical QML

**Main files**

- `tar_lab/science_exec.py`
- `science_profiles/quantum_ml.json`

**Tests**

- canonical QML selects the real backend
- missing quantum deps cause refusal

**Exit**

- QML canonical path is scientifically honest

### `WS13`: Runtime Sandbox Tightening

**Purpose**

Bring actual runtime mounts and write surfaces into line with the documented
safety model.

**Deliverables**

- read-only source mounts by default
- explicit writable artifact mounts
- visible mount policy in runtime status

**Main files**

- `tar_lab/hierarchy.py`
- `tar_lab/docker_runner.py`
- `tar_lab/safe_exec.py`

**Tests**

- writes outside artifact dirs fail
- dev mode vs production mode is explicit

**Exit**

- runtime safety claims match actual mount behavior

### `WS14`: Inference Safety And Observability

**Purpose**

Harden the managed endpoint layer so failures are diagnosable and remote-code
risk is explicit.

**Deliverables**

- endpoint log capture
- structured health/failure records
- reduced or explicitly controlled `trust_remote_code`

**Main files**

- `serve_local.py`
- `tar_lab/inference_bridge.py`
- `tar_cli.py`
- `dashboard.py`

**Tests**

- failing endpoint retains logs
- health transitions are persisted and visible

**Exit**

- inference failures are diagnosable, not silent

### `WS15`: Test Suite Gap Closure

**Purpose**

Bring the tests up to the actual failure modes surfaced by the audit.

**Deliverables**

- audit-regression suite
- explicit red-team style cases

**Main files**

- `tests/`

**Tests to add**

- vector-store migration
- benchmark truthfulness
- verdict locality
- manifest pinning
- sandbox write restrictions
- endpoint observability

**Exit**

- the current audit findings become permanent regression coverage

### `WS16`: Workstation / RunPod Validation

**Purpose**

Confirm that the corrected stack holds up under real hardware, real models, and
real workload scale.

**Deliverables**

- real-hardware validation report
- canonical benchmark execution evidence
- endpoint lifecycle endurance
- scheduler endurance

**Environment**

- workstation / RunPod

**Exit**

- final deployment decision backed by real-scale evidence

## Suggested Sprinting

### Sprint A

- `WS8`
- `WS11`
- `WS12`

**Goal**

Restore correctness of the operator surface, verdict integrity, and locked
manifests.

### Sprint B

- `WS9`
- `WS10`

**Goal**

Make benchmark claims scientifically honest.

### Sprint C

- `WS13`
- `WS14`
- `WS15`

**Goal**

Harden runtime safety and close the regression gap.

### Sprint D

- `WS16`

**Goal**

Real-world validation and final sign-off.

## Estimated Total Effort

- laptop remediation: `15-29 focused engineering days`
- workstation / RunPod validation: `3-6 days`
- total program estimate: `18-35 days`

## Go / No-Go Gates

- after `WS8-WS12`: operator correctness gate
- after `WS9-WS10`: scientific honesty gate
- after `WS13-WS15`: infrastructure confidence gate
- after `WS16`: deployment gate

## Sign-off Standard

The remediation program is complete only when:

- live operator commands work reliably again
- canonical benchmark claims match actual execution
- reproducibility fails closed instead of drifting
- claim acceptance is trial-local and evidence-bound
- runtime write scope is materially constrained
- inference endpoints are diagnosable and safer
- workstation validation confirms the repaired stack under real conditions

## Post-WS16 Follow-On Backlog

The following items are intentionally out of the current remediation critical
path, but they are explicitly retained for the next roadmap phase after `WS16`.
They were raised as open design questions during the audit and should be
treated as first-class future work, not informal notes.

### Candidate `WS17`: Autonomous Portfolio Control And Evidence Budgeting

**Purpose**

Add a true research-portfolio controller so TAR can decide how long to stay on a
problem, when to pause it, when to escalate falsification effort, and when to
switch to another queued problem under explicit evidence and utility rules.

**Questions saved for this workstream**

- Is there a per-problem time budget or evidence budget?
- What explicit stopping criteria determine when TAR should put a problem down?
- How should TAR prioritize among multiple open problems?
- Should TAR automatically switch to the next problem when expected information
  gain or priority is higher?
- Should TAR hold meta-tests or micro-experiments specifically to challenge its
  own claims before promotion?
- How should falsification pressure, replication effort, and benchmark cost be
  balanced against exploration speed?

**Saved output standards**

The following output requirements are also explicitly saved for the post-`WS16`
phase. They should become part of TAR's reporting contract, not just its
internal design goals.

- TAR output must make clear whether it has produced a credible finding or only
  a provisional signal.
- TAR output must explain and verify findings, showing why a result should be
  believed and what evidence, checks, or replication support it.
- TAR output must stay honest when capability is incomplete, explicitly saying
  when a benchmark, environment, model, or evidence base is not strong enough
  for a stronger claim.

**Expected design themes**

- per-problem wall-clock and compute budgets
- evidence-budget accounting
- explicit stop / continue / retry / pivot decisions
- portfolio scheduling beyond simple queue priority
- automatic escalation from heuristic planning to targeted falsification tests
- meta-experiment generation for truth-proving
- expected-information-gain or uncertainty-reduction based prioritization

**Closure standard**

This future workstream closes only when TAR can honestly say:

- why it stayed on a problem
- why it stopped
- why it switched
- what evidence budget it spent
- what falsification or replication tests it ran before accepting a claim

### Deferred Extension: Academic Writing Handoff

This is a separate future extension and is not part of the current remediation
sequence.

**Intent**

Once TAR is producing credible, verification-backed findings, it should be able
to hand structured evidence, results, provenance, and contradictions to a
writer subsystem that can produce publication-grade academic drafts.

**Constraint**

This must remain downstream of:

- benchmark honesty
- verdict integrity
- reproducibility integrity
- workstation-scale validation

The writer should consume verified findings; it should not be used to create the
appearance of rigor ahead of the actual evidence base.
