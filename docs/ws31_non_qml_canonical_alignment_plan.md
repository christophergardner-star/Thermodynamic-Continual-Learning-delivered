# WS31 Non-QML Canonical Alignment Plan

## Decision

`WS31` is now **closed**.

What is closed inside `WS31`:

- benchmark statistical contracts
- bounded validation benchmark packs
- first canonical QML pack with real `5`-seed readiness
- first non-QML canonical split in `generic_ml`
- second non-QML canonical split in `graph_ml`
- third non-QML canonical split in `computer_vision`
- fourth non-QML canonical split in `deep_learning`
- fifth non-QML canonical split in `natural_language_processing`
- `reinforcement_learning` assessed and still fully refused

What remains explicit after closure:

- a carry-forward refusal policy for the still-unresolved canonical suites

That means `WS31` is no longer blocked by benchmark statistics, and it no
longer needs another bounded alignment slice to make its benchmark-truth claim
credible.

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
- `natural_language_processing`
  - `cnn_dailymail_summarization`: canonical-ready, real executor path, real
    `5`-seed statistical summary
  - `beir_fiqa_canonical`: still truthfully refused until the executor covers
    a truthful BEIR FiQA retrieval benchmark path
  - `longbench_narrativeqa_canonical`: still truthfully refused until the
    executor covers a truthful LongBench NarrativeQA benchmark path

### Still Fully Refused After Assessment

- `reinforcement_learning`
  - `minari_cartpole_exploration`: still refused because the executor remains a
    synthetic exploration probe and does not yet load a specific Minari
    CartPole dataset
  - `minari_offline_online_transfer`: still refused because there is no
    truthful Minari offline-to-online CartPole benchmark path yet

### Profiles With Remaining Truthful Canonical Refusals

- `generic_ml`
- `graph_ml`
- `computer_vision`
- `deep_learning`
- `natural_language_processing`
- `reinforcement_learning`

These profiles still have canonical suites registered, but the registry
correctly marks the unresolved suites unsupported because the named canonical
executors are not yet truly aligned to the claimed datasets or backends.

## What Non-QML Canonical Alignment Means

For each remaining suite, TAR must move from:

- canonical suite exists in the registry
- canonical suite is truthfully refused

to:

- canonical suite has a real executor path
- canonical suite has truthful availability checks
- canonical suite runs without proxy fallback
- canonical suite produces benchmark-specific statistical summaries
- canonical suite is genuinely canonical-comparable

## Alignment Order Used

The local alignment order used so far is:

1. `generic_ml`
2. `graph_ml`
3. `computer_vision`
4. `deep_learning`
5. `reinforcement_learning` assessment
6. `natural_language_processing`

That order was correct because it forced TAR to prove easier non-QML canonical
paths before promoting the heavier NLP slice.

## Closure Basis

`WS31` closes on this basis:

- QML canonical is proven with real `5`-seed readiness
- five non-QML canonical-aligned paths are proven
- the remaining unsupported suites stay explicitly refused
- the repo no longer carries ambiguous benchmark-availability claims for the
  unresolved canonical paths

## Pod Policy

Still **no pod now**.

Reason:

- the next step is no longer WS31 benchmark execution
- that is code and benchmark-truth governance work, not scale work
- hiring a pod before the remaining canonical scope is frozen would repeat the
  same mistake the scale criteria were written to prevent

## Exit Condition For WS31

`WS31` closes because:

- QML canonical is proven
- multiple non-QML profiles now have real canonical-aligned executor paths
- broader unsupported canonical suites remain explicitly refused rather than
  ambiguously named
