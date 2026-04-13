# WS29 Execution Spec

`WS29` is **Real Experiment Backend And Resume Semantics**.

## Purpose

Convert experiment backends from typed launch-plan stubs into truthful,
resumable runtime components, starting with `asc_full`.

## Why WS29 Exists

Phase 4 no longer needs another model branch before it can move forward. The
weak link is backend execution truth:

- backend plans exist, but resume semantics were inconsistent
- `asc_full` was listed as executable but not actually checkpoint-aware at the
  TAR runtime layer
- backend state transitions were inferred from folders instead of persisted as
  first-class runtime records

## WS29 Scope

The first local slice covers:

1. real `asc_full` checkpoint resume support
2. resume-aware backend launch plans
3. persisted backend runtime state records under `tar_state/experiment_backends`
4. cleaner artifact lineage for training log and checkpoint paths

This slice does **not** close WS29 by itself. Pod-backed end-to-end backend
resume validation is still required later.

## Deliverables

### 1. `asc_full` Resume Semantics

`asc_train_full.py` must:

- accept `--resume_from_checkpoint`
- persist a `resume_state.pt` bundle with:
  - completed global step
  - completed epoch index
  - intra-epoch batch position
  - optimizer state
  - scheduler state
  - RNG state
  - latest checkpoint path
- restore the trainer from that bundle
- use deterministic per-epoch shuffling so resumed runs skip already-completed
  batches in the current epoch instead of silently restarting from an
  unrelated batch order

### 2. Resume-Aware Launch Plans

`ExperimentLaunchPlan` must surface:

- `backend_state_path`
- structured `resume`
- structured `artifact_lineage`

`ExperimentBackendRegistry.build_plan(...)` must:

- mark `asc_full` as resume-capable
- detect an existing resume bundle for the trial/backend
- emit a truthful `--resume_from_checkpoint` command when resume is requested
  or auto-detected

### 3. Backend Runtime State

TAR must persist one runtime record per `(trial_name, backend_id)`:

- status
- output directory
- manifest path
- resume state
- artifact lineage
- completed steps
- epoch position
- launch count
- last error

These records live under:

- `tar_state/experiment_backends/<trial_name>__<backend_id>.json`

### 4. Artifact Lineage

The runtime layer must stop guessing from directories alone. It should persist:

- `training_log.json`
- `resume_state.pt`
- latest checkpoint path
- final checkpoint path when available

## Design Rules

- resume semantics must be truthful, not optimistic
- `supports_resume=True` only if the backend can actually relaunch from saved
  state
- interrupted runs must leave enough state to resume or fail honestly
- backend runtime records are part of system state, not disposable sidecars
- the first WS29 slice should improve `asc_full` without pretending that
  `coding_asc` is ready

## Acceptance Criteria For This Local Slice

- `asc_full` is marked resumable in the backend registry
- `asc_train_full.py` can persist and reload checkpoint state bundles
- backend launch plans persist runtime records with resume metadata
- backend runtime records are written under `tar_state/experiment_backends`
- targeted resume/backend tests pass locally

## Deferred To Later WS29 Steps

- pod-backed end-to-end resume validation for `asc_full`
- richer CLI/dashboard surfaces for backend runtime inspection
- runtime lease/restart orchestration around backend relaunch
- any attempt to promote `coding_asc` beyond scaffold status
