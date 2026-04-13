# WS30 Execution Spec

## Title

`WS30: Locked Payload Adoption And Build Attestation`

## Purpose

Convert TAR's reproducibility machinery from optional preparation logic into
operational truth:

- locked payload images should be the default serious-run execution surface
- build provenance should survive process boundaries and pod boundaries
- payload and science-environment builds should carry auditable attestation
  records instead of only transient stdout and tags

## Why WS30 Exists

By the end of `WS29`, TAR already had:

- locked payload manifests
- locked science-bundle manifests
- resumable backend execution
- operator-serving integration

What remained weak was operational provenance:

- builds were not yet first-class attested records
- image digests were not surfaced as durable state
- status and dashboard surfaces could say an image was "built" without durable
  attestation truth

`WS30` closes that gap.

## Core Hypothesis

If TAR persists build attestations as first-class state and ties them to locked
payload/science environments, then:

- reproducibility becomes auditable instead of inferred
- live run provenance survives pod churn
- later benchmark and publication layers can attach immutable build lineage to
  serious claims

## Scope

### In Scope

- first-class `BuildAttestation` schema and persistence
- build attestation attachment for:
  - payload image rebuilds
  - science-bundle image builds
- image digest and image ID capture from real docker builds
- operator/runtime/status/dashboard surfaces for latest payload build
  attestation
- propagation of build attestation metadata into problem execution records

### Out Of Scope

- benchmark harness redesign
- new model training
- distributed build infrastructure
- remote attestation signing
- production registry push/pull policy

## Deliverables

### 1. Durable Build Attestation State

TAR must persist build attestations under `tar_state/build_attestations/` with:

- attestation ID
- scope kind
- image tag
- build command
- builder backend
- build status
- return code
- image manifest hash
- dependency lock hash
- environment fingerprint ID
- run manifest hash
- image digest
- image ID

### 2. Payload Build Attestation

`rebuild_locked_image()` must:

- build the locked payload image
- create and persist a build attestation
- rewrite the payload environment manifest with attestation linkage
- surface attestation ID and digest in status/runtime payloads

### 3. Science-Bundle Build Attestation

Science-environment build paths must:

- attach build attestation to the environment bundle
- preserve that attestation into downstream study/execution records
- carry image digest when available

### 4. Operator-Facing Truth

`status()`, `runtime_status()`, and the dashboard infrastructure view must show:

- latest payload build status
- latest payload build attestation
- attestation ID
- image digest

## File Scope

Primary implementation:

- `tar_lab/schemas.py`
- `tar_lab/state.py`
- `tar_lab/reproducibility.py`
- `tar_lab/docker_runner.py`
- `tar_lab/orchestrator.py`
- `dashboard.py`

Primary test scope:

- `tests/test_reproducibility.py`
- `tests/test_docker_runner.py`

## Acceptance Criteria

WS30 local slice is acceptable only if all are true:

- payload rebuild persists a real build attestation
- payload status/runtime surfaces expose attestation truth
- science-bundle build paths attach attestation metadata
- successful build results capture image digest and image ID when available
- problem-execution records can carry build attestation linkage
- local regression tests cover the new behavior

## Pod Policy

Do not use a pod for this first slice.

Pod is justified only when:

- the code path is validated locally
- the next step is a real locked-image execution validation
- local hardware is no longer the correct environment for truthful execution

## Local Slice Plan

### Slice 1

- add and persist `BuildAttestation`
- attach attestation to payload/science builds
- capture image metadata in `DockerRunner`
- surface build attestation in status/runtime/dashboard
- add targeted local regression tests

### Slice 2

- validate real locked payload execution on pod
- verify attestation survives pod boundary
- close WS30 with a runtime validation result
