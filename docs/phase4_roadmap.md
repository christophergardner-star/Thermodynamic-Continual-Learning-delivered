# Phase 4 Roadmap

Authoritative follow-on planning after the Phase 4 audit is now tracked in:

- [phase5_roadmap.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/phase5_roadmap.md)

Use this file for the Phase 4 system-integration context and the current
WS32/WS32.5/WS33 dependency chain. Use `phase5_roadmap.md` for the detailed
post-audit execution plan from `WS32-close` through `WS39`.

This is the authoritative forward roadmap after completion of:

- `WS1-WS7` foundation phase
- `WS8-WS16` remediation phase
- `WS17-WS20` first post-remediation autonomy/control phase
- `WS21-WS27` operator-model and large-branch phase

`Phase 3` proved that TAR can be:

- a disciplined research operating system
- a benchmark-honest operator
- a portfolio-aware project manager
- a publication-handoff producer
- a trained 7B TAR/TCL operator line
- a viable larger-model research branch with bounded refinement discipline

That changes the bottleneck.

The project is no longer primarily blocked on:

- control logic
- dataset scale
- baseline model viability
- basic TCL specialization

The remaining frontier work is now system integration:

- make the trained operator a first-class runtime component
- make real experiment backends first-class and resumable
- make reproducible payloads actually govern live runs
- make benchmark validation canonical and statistical
- deepen literature and retrieval into claim-graph quality
- harden runtime durability and sandbox safety

## Current State At Phase 4 Start

TAR now has:

- continuity, prioritization, falsification, and portfolio management
- operator-facing views and dashboards
- publication handoff packages
- a private TAR/TCL training corpus
- a held-out TAR/TCL evaluation harness
- a validated `WS25` TAR operator line
- a validated `WS26` TCL-deepened line
- a closed `WS27` branch with successful bounded refinement
- a closed `WS28` operator-serving integration layer
- a closed `WS29` backend resume and runtime-state layer

Most recent branch closure:

- see
  [ws27r2_refine_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r2_refine_closeout.md)

Most recent backend closure:

- see
  [ws29_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws29_closeout.md)

## Guiding Principle

Phase 4 should be managed like a serious systems-research integration phase:

- use the strongest validated model line as infrastructure, not just an artifact
- convert remaining architecture sketches into first-class runtime capabilities
- prioritize closed-loop scientific execution over surface novelty
- spend pod budget only when local validation stops being the correct tool

## Archived Phases

### Completed Foundation

- `WS1-WS7`

### Completed Remediation

- `WS8`
- `WS9`
- `WS10`
- `WS11`
- `WS12`
- `WS13`
- `WS14`
- `WS15`
- `WS16`

### Completed Autonomy / Operator Phase

- `WS17`
- `WS18`
- `WS19`
- `WS20`
- `WS21`
- `WS22`
- `WS23`
- `WS24`
- `WS25`
- `WS26`
- `WS27`

## Active Forward Roadmap

### `WS28`: Native Inference Integration And Operator Serving

Purpose:

- make the best validated TAR operator adapter a first-class local model inside
  TAR

Core deliverables:

- checkpoint registry integration for the `WS26/WS27` line
- OpenAI-compatible serving path using the trained operator
- role-aware routing hooks for Director / Strategist / Scout style use later
- operator-side selection of prompt-only vs tuned local model
- serving manifests and versioned model selection

Why it matters:

- TAR now has a strong trained operator line, but that line is still treated
  mainly as an experiment artifact rather than a system component

Execution posture:

- laptop-first for integration and local serving tests
- pod only for throughput, memory, or multi-checkpoint comparison work

Status:

- completed
- closeout:
  [ws28_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws28_closeout.md)

### `WS29`: Real Experiment Backend And Resume Semantics

Purpose:

- convert `asc_full` and related backends from typed plans into resilient,
  resumable first-class experiment backends

Core deliverables:

- actual resume semantics for `asc_full`
- cleaner backend manifests and artifact lineage
- checkpoint-aware relaunch logic
- stable backend state transitions across interruptions

Why it matters:

- TAR cannot become a real lab runtime if its experiment backends remain weaker
  than its planning layer

Execution posture:

- laptop-first for backend and manifest logic
- pod required for real end-to-end backend execution and resume validation

Status:

- completed
- closeout:
  [ws29_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws29_closeout.md)

### `WS30`: Locked Payload Adoption And Build Attestation

Purpose:

- make live TAR runs use locked payload images and attach build provenance to
  every serious run

Core deliverables:

- automatic use of locked payload images in live runs
- image digest capture
- build provenance attached to trials
- reproducibility metadata that survives pod boundaries

Why it matters:

- reproducibility modules exist, but Phase 4 should make them operational
  policy, not optional machinery

Execution posture:

- laptop-first for manifest and policy work
- pod when validating real locked-image execution paths

Status:

- completed
- closeout:
  [ws30_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws30_closeout.md)

### `WS31`: Canonical Benchmark Harnesses And Statistical Validation

Status:

- completed
- QML canonical slice complete
- generic_ml canonical split complete
- graph_ml canonical split complete
- computer_vision canonical split complete
- deep_learning canonical split complete
- natural_language_processing canonical split complete
- reinforcement_learning canonical slice assessed and still refused
- closeout:
  [ws31_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws31_closeout.md)

Purpose:

- move from internal eval strength to domain-canonical benchmark validation

Core deliverables:

- canonical external dataset wiring per supported domain
- benchmark-specific artifact schemas
- multi-seed statistical reporting
- explicit significance / confidence reporting where appropriate

Why it matters:

- TAR can already evaluate its own operator behavior well; Phase 4 should also
  make external scientific claims harder to bluff

Execution posture:

- mixed
- local for harness design and reporting
- pod for real benchmark execution at scale
- bounded pod use only after
  [ws31_benchmark_scale_criteria.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws31_benchmark_scale_criteria.md)
  is satisfied

### `WS32`: Literature Engine Deepening

Status:

- completed
- closeout:
  [ws32_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws32_closeout.md)

Purpose:

- upgrade literature ingestion from section-and-claim extraction toward
  provenance-preserving paper understanding

Core deliverables:

- stable content-derived paper identity
- ingest manifests and repeat-ingest history
- richer table and figure metadata
- stronger claim typing and claim-to-source linkage
- operator-visible literature artifact and conflict inspection

Why it matters:

- TAR should reason over evidence with stronger document fidelity than abstract
  snippets and loose notes

Execution posture:

- laptop-first
- pod generally not required

### `WS32.5`: Loop Closure

Status:

- completed
- closeout:
  [ws32_5_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws32_5_closeout.md)

Purpose:

- close the remaining loop boundaries between literature, planning, operator
  serving, and evidence-debt enforcement

Core deliverables:

- literature signal into Director policy
- serving-state fallback into live hierarchy role resolution
- evidence debt as a hard scheduling invariant

Why it matters:

- without loop closure, stronger retrieval and stronger runtime automation
  would deepen disconnected components rather than a real autonomous research
  loop

Execution posture:

- laptop-first
- no pod required

### `WS33`: Scientific Retrieval And Claim Graph

Purpose:

- make retrieval evidence-aware, contradiction-aware, and claim-graph aware on
  top of the now-closed planning loop

Core deliverables:

- contradiction-aware retrieval scoring
- claim-graph indexing
- source-confidence weighting
- evidence-budgeted retrieval feedback into evidence debt
- no silent lexical downgrade for literature-grounded study work

Why it matters:

- retrieval deepening is only behavior-changing once literature can actually
  influence planning and evidence gating

Execution posture:

- laptop-first
- pod only for large-index experiments or heavy embedder comparisons

### `WS34-pre`: Claim Verdict Lifecycle

Purpose:

- add aging and escalation logic for unresolved claim verdict states before the
  runtime becomes more autonomous

Core deliverables:

- review deadlines on unresolved claim verdicts
- staleness checks for aged provisional / insufficient-evidence states
- scheduled escalation from unresolved verdicts into explicit falsification
  pressure

Why it matters:

- durable runtime automation should not operate indefinitely against stale
  unresolved claim states

Execution posture:

- laptop-first
- no pod required

### `WS34`: Durable Lab Runtime

Purpose:

- move TAR from a robust single-node runtime into a resilient lab service

Core deliverables:

- lease-based workers
- orphan-run recovery
- explicit recoverable-crash state
- retry / backoff policies
- clearer daemon and queue health surfaces

Why it matters:

- long-running research systems fail in operational seams unless runtime
  durability is treated as a first-class engineering problem

Execution posture:

- laptop-first for daemon logic
- pod or multi-process environment for realistic recovery validation

### `WS35`: Safe Execution Hardening

Purpose:

- harden generated-code and live experiment execution further so the system can
  be trusted under broader autonomy

Core deliverables:

- universal sandbox-policy enforcement
- read-only workspace policies where appropriate
- capability drops
- seccomp tightening
- stronger container policy defaults
- clearer security audit surfaces for autonomous execution

Why it matters:

- stronger autonomy without stronger isolation is a bad trade

Execution posture:

- laptop-first for policy and sandbox plumbing
- pod or container host required for realistic hardening validation

## Recommended Execution Order

The recommended professional order is:

1. `WS32`
2. `WS32.5`
3. `WS33`
4. `WS34-pre`
5. `WS34`
6. independent eval validation pass
7. `WS35`

## Why This Order Is Correct

`WS32` first:

- TAR needed stronger literature fidelity and retrieval telemetry before
  retrieval and planning could safely consume literature state

`WS32.5` before `WS33`:

- better retrieval without loop closure would still leave planning disconnected
- evidence debt had to become a hard scheduling gate before more durable
  runtime autonomy was justified

`WS33` after loop closure:

- retrieval and claim-graph deepening now changes behavior instead of just
  storage quality

`WS34-pre`, `WS34`, and `WS35` last:

- those workstreams harden autonomy and runtime behavior
- they should follow, not precede, the integrity and loop-closure invariants

## Pod Hiring Policy

Pod/GPU time in Phase 4 should stay narrow and explicit.

### Do not hire a pod for

- roadmap refreshes
- docs-only closeouts
- serving/config plumbing that can be tested locally
- literature parsing logic
- loop-closure wiring
- retrieval logic
- manifest or schema work

### Hire a pod for

- live local-model serving validation at meaningful size
- real backend resume testing
- benchmark execution at scale
- locked-image execution validation
- large-index or large-checkpoint inference comparisons
- durable-runtime recovery validation
- realistic sandbox-hardening validation

### Earliest justified pod point in Phase 4

The first clearly justified remaining Phase 4 pod point is:

- workload-specific and no longer immediate
- neither `WS32`, `WS32.5`, `WS33`, nor `WS34-pre` require a pod by default
- `WS34` and `WS35` are the first likely pod-backed validation points

## Immediate Next Step

The next logical workstream is:

- `WS33: Scientific Retrieval And Claim Graph`

It should start locally on top of the closed `WS32` and `WS32.5` foundation.

The immediate engineering goal is:

- move contradiction handling and source confidence into retrieval scoring
- promote claim/conflict structures into first-class retrieval objects
- feed contradiction-heavy retrieval back into evidence-debt computation
