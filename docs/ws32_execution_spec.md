# WS32 Execution Spec

## Title

`WS32: Literature Engine Deepening`

## Purpose

Turn TAR's existing literature parser from a useful heuristic ingest path into
a stronger provenance-preserving evidence layer.

By the end of `WS31`, TAR already had:

- benchmark-truth discipline
- claim-review and claim-verdict surfaces
- semantic retrieval hooks
- a local literature engine with:
  - PDF/text ingestion
  - section extraction
  - bibliography extraction
  - citation-edge extraction
  - claim extraction
  - table extraction
  - figure extraction
  - basic contradiction detection

What remained weak was literature fidelity:

- paper identity was path-derived rather than content-derived
- repeated ingest could duplicate artifacts silently
- table and figure artifacts were still lightly structured
- claim and citation provenance quality was only moderately reliable
- studies did not persist whether retrieval was semantic or lexical fallback
- cross-paper conflict labels were still too permissive for same-topic,
  different-scope papers
- operator-facing literature inspection was thin

## Core Hypothesis

If TAR deepens literature ingestion around stable paper identity, richer
structured artifacts, stronger provenance, and operator-visible inspection,
then later retrieval, claim review, and publication work will rest on stronger
paper evidence rather than flattened snippets.

## WS32 Standard

WS32 should reach the practical PhD-grade target for the current repo:

- provenance-preserving section parsing
- better table parsing
- better figure extraction
- stronger claim typing and source linkage
- explicit contradiction surfaces

WS32 does **not** attempt frontier-grade external citation graph integration.
That remains downstream of this workstream.

## Scope

### In Scope

- stable paper identity based on content, not only path
- literature ingest manifests and deduplicated storage
- richer table and figure metadata
- stronger section and claim provenance metadata
- better surfaced contradiction quality
- retrieval-mode telemetry on study outputs
- tighter cross-paper conflict gating before WS33 consumes conflict labels
- operator/control/CLI/dashboard literature inspection
- dedicated WS32 literature regression tests

### Out Of Scope

- external citation graph APIs
- embedding-heavy retrieval redesign
- large-batch corpus ingestion at scale
- pod-based OCR throughput work
- full WS33 claim-graph retrieval

## Deliverables

### 1. Stable Literature Identity

Each ingested paper must carry:

- content-derived identity
- source fingerprint
- stable dedupe key
- ingest-manifest linkage

Repeated ingest of the same content from different paths must not silently
create duplicate stored artifacts.

### 2. Literature Ingest Manifests

The literature layer must persist manifest records that capture:

- requested and resolved paths
- ingested artifact IDs
- deduplicated-existing count
- stored-total count
- conflict count
- parser-chain provenance

### 3. Richer Structured Artifacts

Sections, claims, tables, and figures must carry stronger structure:

- section text hashes and word counts
- claim citation counts and quality flags
- table headers, dimensions, numeric-cell counts, and metric hints
- figure labels, caption hashes, OCR presence, and nearby claim linkage

### 4. Better Conflict Surfaces

Claim clusters and claim conflicts must expose more explicit quality signals:

- paper counts
- polarity distribution
- conflict kind
- shared-token counts

Cross-paper conflicts must also be gated by:

- a stricter shared-token floor
- opposite polarity
- a simple scope / negation compatibility check

That keeps same-topic, different-scope papers from being mislabeled as
contradictions before WS33 retrieval consumes those labels.

### 5. Retrieval Telemetry

Every `ProblemStudyReport` produced from literature-grounded study work must
persist whether retrieval executed with:

- `semantic`
- `lexical_fallback`

This telemetry is required so later WS33 retrieval work has a truthful baseline
instead of silent degradation.

### 6. Operator Inspection Surfaces

TAR must expose literature inspection through:

- orchestrator methods
- control commands
- CLI commands
- dashboard visibility

At minimum, TAR must be able to show:

- literature status
- recent paper artifacts
- one paper artifact in detail
- literature conflicts

## Acceptance Criteria

WS32 is acceptable only if all are true:

- stable paper identity survives repeated ingest
- literature ingest manifests are persisted
- duplicate content is deduplicated truthfully
- tables and figures are extracted with stronger structure than the WS31
  baseline
- claim and citation provenance fields are richer and test-covered
- literature-grounded studies persist `retrieval_mode`
- cross-paper conflict labeling is tighter than the original overlap-only
  heuristic
- operator inspection surfaces exist and report truthful literature state
- dedicated WS32 regression tests pass

## Pod Policy

Do not use a pod for the WS32 local slice.

Pod becomes justified only when:

- OCR-heavy corpora need throughput validation
- very large paper batches exceed local practicality
- later retrieval/index experiments in `WS33` need heavier embedding work

## Execution Order

1. extend schemas and state for literature manifests and richer artifact
   structure
2. harden `literature_engine.py` for stable identity and deduplicated ingest
3. expose literature inspection through orchestrator/control/CLI/dashboard
4. add dedicated WS32 regression coverage
5. validate locally before any roadmap or pod decision changes
