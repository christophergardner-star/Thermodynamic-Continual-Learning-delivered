# WS31 Benchmark Pack V1

## Purpose

Freeze the first bounded `WS31` benchmark pack against
[ws31_benchmark_scale_criteria.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws31_benchmark_scale_criteria.md)
before any pod hire.

Machine-readable manifest:

- [ws31_benchmark_pack_v1.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/configs/ws31_benchmark_pack_v1.json)

## Frozen Pack

This first pack is intentionally narrow:

- `3` profiles
- `3` benchmark suites
- `3` experiment cells
- validation tier only
- no proxy fallback
- no external downloads

Selected suites:

1. `generic_ml`
   - benchmark: `breast_cancer_validation`
   - template: `baseline_sweep`
   - primary metric: `accuracy`
   - required seeds: `3`
2. `computer_vision`
   - benchmark: `digits_transfer_validation`
   - template: `backbone_transfer`
   - primary metric: `top1_accuracy`
   - required seeds: `3`
3. `reinforcement_learning`
   - benchmark: `cartpole_exploration_validation`
   - template: `exploration_ablation`
   - primary metric: `episodic_return`
   - required seeds: `3`

Reserve candidate:

- `graph_ml`
  - benchmark: `karate_heterophily_validation`
  - template: `heterophily_ablation`
  - primary metric: `node_accuracy`
  - note: viable, but excluded to preserve the explicit three-profile cap

## Why These Suites

These suites satisfy the written `WS31` preconditions:

- fixed `profile_id`, `benchmark_id`, tier, and template
- fixed primary metric
- fixed seed policy
- `proxy_allowed = false`
- local validation path already works
- benchmark statistical summaries already report `statistically_ready = true`

They are also bounded enough to make the next execution decision honest.

## Local Measurement

Observed local single-pass wall times:

- `generic_ml / baseline_sweep`: `3.112s`
- `computer_vision / backbone_transfer`: `1.751s`
- `reinforcement_learning / exploration_ablation`: `0.413s`

Observed pack total:

- single pass: `5.276s`

Conservative linear upper bound under the `18`-cell hard limit:

- `31.656s`

That is nowhere near the `90 minute` pod trigger in
[ws31_benchmark_scale_criteria.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws31_benchmark_scale_criteria.md).

## Pod Decision

Pod is **not justified now**.

Reason:

- the frozen pack already passes local harness validation
- its measured local wall time is trivial
- it does not require an unstable runtime or unavailable backend
- hiring a pod for this pack would violate the purpose of the scale criteria

## Exclusions

Not included in V1:

- validation suites that are still statistically under-powered in the current
  harness, such as generic calibration-oriented paths
- canonical suites that would require external downloads or backend-specific
  runtime guarantees before the first bounded benchmark pack is established
- additional ready validation suites that would push the pack beyond the
  deliberate three-profile cap

## Next Correct Move

Run this exact frozen pack locally and preserve:

- per-benchmark manifests
- per-experiment statistical summaries
- execution-level aggregate statistical summaries

Only after the local run of this frozen pack should `WS31` consider a larger
pack or a pod-backed canonical run.
