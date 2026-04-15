# WS31 Benchmark Scale Criteria

## Purpose

Define the exact conditions under which `WS31` is allowed to leave the local
statistical-contract phase and use a pod for bounded benchmark-scale execution.

This document exists to prevent premature benchmark pod use. A pod is justified
 only when the next step is a real, bounded, multi-seed benchmark session that
 cannot be defended as a laptop-only task.

## Current Baseline

As of commit `59cd315`, the first local `WS31` slice is complete:

- benchmark specs expose seed-policy defaults
- experiment results carry benchmark statistical summaries
- execution reports carry aggregate statistical summaries
- seeded local benchmark executors can report statistically ready outcomes
- non-seeded validation benchmarks report honest statistical insufficiency

That means the next step is no longer schema work. It is benchmark-scale
 execution, but only if the run is bounded and worth the cost.

## Required Preconditions Before Any Pod Hire

All of these must be true before a pod is justified:

1. `main` contains the `WS31` local slice.
2. The selected benchmark pack is frozen in writing.
3. Every selected benchmark has:
   - a fixed `profile_id`
   - a fixed `benchmark_id`
   - a fixed tier
   - a fixed primary metric
   - a fixed required seed count
4. Every selected benchmark is executable without proxy fallback.
5. Every selected canonical benchmark is `canonical_ready`.
6. The pack has a bounded wall-clock estimate and a stop rule.
7. The run outputs and statistical success criteria are written before launch.

If any of these are missing, the pod is not justified yet.

## Bounded Benchmark Pack Rules

The first benchmark-scale `WS31` pod session must stay small.

Hard limits:

- maximum `3` profiles
- maximum `6` benchmark suites
- maximum `18` experiment cells
- maximum `2` tiers in the same run
- target wall time: `<= 6 hours`

The first pod run should prefer `validation` tier benchmarks. Add canonical
 benchmarks only when their runtime and dataset/backend requirements are already
 explicit and available.

## Statistical Requirements

### Validation Tier

- minimum `3` seeds
- `statistically_ready = true`
- CI95 present for the primary metric
- `proxy_benchmark_used = false`
- `benchmark_alignment = aligned`

### Canonical Tier

- minimum `5` seeds
- `statistically_ready = true`
- CI95 present for the primary metric
- `proxy_benchmark_used = false`
- `benchmark_alignment = aligned`
- `canonical_comparable = true`

### Disallowed Outcomes

The pod run is not considered a valid `WS31` scale run if:

- a benchmark silently downgrades into proxy mode
- a canonical benchmark executes in a refused or unsupported state
- seed count is below policy
- CI reporting is missing for a benchmark that claims readiness

## Pod Trigger Criteria

A pod is justified only if the frozen benchmark pack meets both of these:

1. it already passes local harness validation and statistical-contract checks
2. the expected repeated multi-seed execution is no longer efficient or honest
   to run locally

Concrete triggers:

- expected local wall time exceeds `90 minutes`
- repeated multi-seed runs would saturate local memory or CPU/GPU
- at least one selected benchmark requires a runtime/backbone that is available
  on the pod but not stable on the workstation

If the pack still fits comfortably on local hardware, keep it local.

## Required Outputs From The First Pod Run

The first benchmark-scale `WS31` pod run must produce:

- frozen benchmark-pack manifest
- per-benchmark run manifest
- per-experiment statistical summaries
- execution-level aggregate statistical summaries
- refusal/downgrade notes where applicable
- artifact bundle copied back locally

## Stop Rules

Abort the pod run if any of these occur:

- more than `20%` of experiment cells fail structurally
- a supposedly real benchmark falls back to proxy mode
- a canonical benchmark cannot satisfy its declared availability/backend path
- the run exceeds the bounded wall-clock budget without reaching the frozen
  benchmark pack

## Closure Criteria For WS31

`WS31` can close after the first benchmark-scale run only if:

- the frozen benchmark pack executed as planned
- statistical readiness claims are backed by real seed evidence
- canonical claims, if present, remain canonical-comparable
- run artifacts and summaries are preserved locally

If those conditions are not met, `WS31` remains open and the next step is a
 local harness correction, not another automatic pod cycle.
