# WS31 Non-QML Canonical Alignment Plan

## Decision

`WS31` remains **open**.

What is closed inside `WS31`:

- benchmark statistical contracts
- bounded validation benchmark packs
- first canonical QML pack with real `5`-seed readiness
- first non-QML canonical split in `generic_ml`
- second non-QML canonical split in `graph_ml`
- third non-QML canonical split in `computer_vision`
- fourth non-QML canonical split in `deep_learning`
- `reinforcement_learning` assessed and still fully refused

What remains open:

- non-QML canonical executor alignment

That means `WS31` is no longer blocked by benchmark statistics. It is now
blocked by **domain-specific canonical benchmark coverage** outside `quantum_ml`.

## Current Canonical State

### Canonical-Proven

- `quantum_ml`
  - `pennylane_barren_plateau_canonical`
  - `pennylane_init_canonical`
  - `qml_noise_canonical`

These suites are now:

- `canonical_ready`
- locally executable
- statistically ready under the `5`-seed canonical policy
- canonical-comparable

### Partially Canonical-Aligned

- `generic_ml`
  - `openml_adult_calibration`: canonical-ready, real executor path, real
    `5`-seed statistical summary
  - `openml_cc18_classification`: still truthfully refused until the executor
    covers the full CC18 suite
- `graph_ml`
  - `roman_empire_heterophily_canonical`: canonical-ready, real executor path,
    real `5`-split statistical summary
  - `cora_depth_canonical`: still truthfully refused until the executor covers
    the real Cora depth benchmark
- `computer_vision`
  - `imagenette_transfer_canonical`: canonical-ready, real executor path, real
    `5`-seed statistical summary
  - `cifar10_c_corruption`: still truthfully refused until the executor loads
    the real CIFAR-10-C corruption suite
- `deep_learning`
  - `cifar10_optimizer_canonical`: canonical-ready, real executor path, real
    `5`-seed statistical summary
  - `cifar10_scaling_canonical`: still truthfully refused until the executor
    implements a real CIFAR-10 scaling benchmark

### Still Fully Refused After Assessment

- `reinforcement_learning`
  - `minari_cartpole_exploration`: still refused because the executor remains a
    synthetic exploration probe and does not yet load a specific Minari
    CartPole dataset
  - `minari_offline_online_transfer`: still refused because there is no
    truthful Minari offline-to-online CartPole benchmark path yet

### Still Truthfully Refused

- `computer_vision`
- `deep_learning`
- `graph_ml`
- `natural_language_processing`
- `reinforcement_learning`

These profiles still have canonical suites registered, but the registry
correctly marks them unsupported because the named canonical executors are not
yet truly aligned to the claimed datasets/backends.

## What “Non-QML Canonical Alignment” Means

For each remaining profile, TAR must move from:

- canonical suite exists in the registry
- canonical suite is truthfully refused

to:

- canonical suite has a real executor path
- canonical suite has truthful availability checks
- canonical suite runs without proxy fallback
- canonical suite produces benchmark-specific statistical summaries
- canonical suite is genuinely canonical-comparable

## Priority Order

The correct order is not “all domains at once.”

Use this priority:

1. `generic_ml`
2. `graph_ml`
3. `computer_vision`
4. `deep_learning`
5. `reinforcement_learning`
6. `natural_language_processing`

Reason:

- `generic_ml` and `graph_ml` are structurally simpler
- they are the best next canonical alignment targets without dragging in heavy
  external serving or large dataset stacks too early
- `natural_language_processing` should come later because its canonical claims
  are the easiest to overstate and the hardest to make honest cheaply

## Next Local Slice

The next exact `WS31` slice should be:

### `WS31-Slice-Next: Natural Language Processing Canonical Alignment`

Target:

- `beir_fiqa_canonical`
- `longbench_narrativeqa_canonical`
- `cnn_dailymail_summarization`

Required local deliverables:

- honest canonical availability contract
- real canonical executor path or explicit continued refusal
- canonical pack candidate definition
- regression tests proving:
  - canonical suite is still refused if unresolved
  - canonical suite becomes `canonical_ready` only when the executor is truly aligned

## Pod Policy

Still **no pod now**.

Reason:

- the next step is executor-alignment design and local truthfulness work
- that is code and harness work, not scale work
- hiring a pod before non-QML canonical alignment exists would repeat the same
  mistake the scale criteria were written to prevent

## Exit Condition For WS31

`WS31` should close only when:

- QML canonical is proven
- at least one non-QML profile has a real canonical-aligned executor path
- broader unsupported canonical suites remain explicitly refused rather than
  ambiguously named

If that non-QML milestone is not reached, `WS31` stays open.
