# WS31 Canonical Pack V1

## Decision

The next meaningful `WS31` pack is the first canonical pack, not a larger
validation pack v2.

Reason:

- the current validation-ready surface does not support a materially larger pack
  within the written `3 profile` cap without adding suites that are still
  statistically under-powered
- the only locally canonical-ready profile today is `quantum_ml`

Machine-readable manifest:

- [ws31_canonical_pack_v1.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/configs/ws31_canonical_pack_v1.json)

## Frozen Canonical Candidate Pack

Profile:

- `quantum_ml`

Suites:

1. `pennylane_barren_plateau_canonical`
   - family: `depth_trainability_curve`
   - template: `ansatz_depth_sweep`
   - primary metric: `trainability_gap`
   - required seeds: `5`
2. `pennylane_init_canonical`
   - family: `initialization_trainability`
   - template: `initialization_variance`
   - primary metric: `gradient_norm_variance`
   - required seeds: `5`
3. `qml_noise_canonical`
   - family: `noise_trainability_ablation`
   - template: `noise_shot_ablation`
   - primary metric: `shot_noise_robustness`
   - required seeds: `5`

All three suites are:

- `canonical_ready`
- `canonical_comparable = true`
- `proxy_allowed = false`
- executable locally

## Local Canonical Run Result

Observed canonical run timings:

- `pennylane_barren_plateau_canonical`: `5.291s`
- `pennylane_init_canonical`: `0.666s`
- `qml_noise_canonical`: `13.738s`

Observed total:

- single pass: `23.205s`
- conservative `18`-cell linear upper bound: `139.23s`

These timings do **not** trigger pod use under
[ws31_benchmark_scale_criteria.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws31_benchmark_scale_criteria.md).

## Why Pod Is Still Not Justified

Pod is **not** justified now.

Not because the canonical pack is fake. It is real, and the local rerun now
shows:

- `3/3` suites completed
- `3/3` suites statistically ready
- `3/3` suites canonical-comparable
- no proxy fallback

The pod is still blocked because the pack is too small and too fast locally to
justify external runtime.

Using a pod now would spend runtime budget without crossing the written WS31
trigger.

## What This Means

`WS31` is now in this state:

- validation pack v1: frozen and executed locally
- first canonical pack: frozen and executed locally
- canonical path: real
- canonical path: statistically ready under the 5-seed policy

## Next Correct Move

Keep `WS31` local.

Do this before any pod hire:

1. preserve and review the canonical pack manifests
2. push the tracked pack runner and frozen pack definitions
3. only if a larger canonical pack no longer fits local constraints should a pod
   be reconsidered
