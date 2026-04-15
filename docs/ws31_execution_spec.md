# WS31 Execution Spec

## Title

`WS31: Canonical Benchmark Harnesses And Statistical Validation`

## Purpose

Convert TAR's benchmark path from identity- and truth-aware execution into
statistically defensible benchmark reporting.

By the end of `WS30`, TAR already had:

- benchmark registry truth
- canonical/validation/smoke benchmark tiers
- availability and refusal logic
- benchmark identity persistence in study and execution records
- locked payload and build-attestation support

What remained weak was benchmark statistics:

- benchmark runs still collapsed to flat metric dicts
- seed evidence was not represented as first-class benchmark state
- benchmark reports could be truthful about benchmark identity while still
  being under-specified statistically

`WS31` closes that gap in stages.

## Core Hypothesis

If TAR attaches benchmark-specific statistical summaries to experiment and
execution reports, then:

- canonical benchmark claims become harder to bluff
- under-powered runs become explicit instead of implicit
- later benchmark and publication layers can distinguish between:
  - benchmark identity truth
  - benchmark availability truth
  - benchmark statistical sufficiency

## Local-First Slice

This first local WS31 slice does **not** attempt large benchmark execution or
pod-scale sweeps.

It must do these things:

- define first-class benchmark statistical summary schemas
- add default statistical policy to benchmark specs
- attach benchmark statistical summaries to experiment execution results
- attach aggregate statistical status to execution reports
- preserve "not statistically ready" as an honest first-class outcome

## Scope

### In Scope

- benchmark statistical summary schemas
- benchmark-tier statistical defaults
- local aggregation of benchmark statistical readiness
- persistence of statistical summaries inside problem execution reports
- CLI/status surfaces that already read execution reports automatically gaining
  those fields
- targeted tests for statistical summary behavior

### Out Of Scope

- large benchmark sweeps
- benchmark pod execution at scale
- canonical external dataset downloads at volume
- significance testing against external baselines
- publication-facing benchmark result packaging

## Deliverables

### 1. Benchmark Statistical Contract

TAR must define:

- per-metric statistical summaries
- per-experiment benchmark statistical summaries
- per-execution aggregate statistical summaries

These must represent:

- primary metric
- sample count
- recommended seed count
- statistical readiness
- confidence interval when enough information exists
- explicit notes when readiness is missing

### 2. Tier-Aware Benchmark Policy

Benchmark specs must carry default statistical policy:

- `smoke`: minimal statistical burden
- `validation`: multi-seed expected
- `canonical`: stronger statistical burden

The policy must be visible through existing benchmark listing/status surfaces.

### 3. Statistical Summary Attachment

`science_exec.execute_study_payload()` must attach:

- experiment-level benchmark statistical summary
- execution-level aggregate benchmark statistical summary

### 4. Honest Under-Powered Reporting

If a benchmark executor does not expose enough seed evidence, TAR must report:

- benchmark executed
- benchmark identity preserved
- statistical readiness not satisfied

That is the correct behavior for the first slice.

## Acceptance Criteria

WS31 local slice is acceptable only if all are true:

- benchmark specs expose seed-policy defaults
- experiment results can carry benchmark statistical summaries
- execution reports carry aggregate benchmark statistical summaries
- seeded local benchmark executors produce statistically ready summaries
- non-seeded local benchmark executors report statistical insufficiency
- targeted regression tests cover both ready and not-ready cases

## Pod Policy

Do not use a pod for this first slice.

Pod is justified only when:

- benchmark statistical contracts are in place locally
- the next step is real multi-seed benchmark execution at scale
- local runtime is no longer the correct execution substrate

## First Slice Plan

### Slice 1

- add benchmark statistical schemas
- normalize statistical defaults on benchmark specs
- add local benchmark statistics helper functions
- attach summaries to problem experiment and execution reports
- add targeted tests

### Slice 2

- decide whether real benchmark scale justifies a pod
- only then run bounded multi-seed benchmark sessions
