# WS31 Closeout

## Outcome

`WS31` is complete.

Canonical benchmark handling is now statistically grounded and benchmark-truth
aware across the validated QML slice plus five non-QML canonical-aligned
paths.

## Scope

`WS31` was the benchmark-truth workstream of Phase 4. Its purpose was to turn
benchmark identity and availability reporting into statistically defensible
benchmark execution and refusal policy.

This included:

- first-class benchmark statistical summaries
- tier-aware seed-count policy for smoke, validation, and canonical runs
- bounded validation and canonical benchmark packs
- canonical QML benchmark proof with real `5`-seed readiness
- non-QML canonical executor alignment where a truthful local path existed
- explicit continued refusal where a truthful canonical executor did not exist

## What WS31 Added

- benchmark execution reports now carry:
  - experiment-level statistical summaries
  - aggregate benchmark statistical summaries
  - truthful readiness and under-powered states
- canonical benchmark packs can now be frozen and executed with:
  - per-suite manifests
  - per-suite statistical summaries
  - pack-level readiness and comparability signals
- canonical benchmark alignment is now proven for:
  - `quantum_ml/pennylane_barren_plateau_canonical`
  - `quantum_ml/pennylane_init_canonical`
  - `quantum_ml/qml_noise_canonical`
  - `generic_ml/openml_adult_calibration`
  - `graph_ml/roman_empire_heterophily_canonical`
  - `computer_vision/imagenette_transfer_canonical`
  - `deep_learning/cifar10_optimizer_canonical`
  - `natural_language_processing/cnn_dailymail_summarization`

## Validation

### Pack-level validation

Validation benchmark pack v1 was executed locally and persisted under:

- `eval_artifacts/ws31_validation_pack_v1`

Observed result:

- `3/3` suites completed
- `3/3` suites statistically ready
- `3/3` suites non-proxy
- total wall time about `5.8s`

Canonical QML pack v1 was executed locally and persisted under:

- `eval_artifacts/ws31_canonical_pack_v1`

Observed result:

- `3/3` suites completed
- `3/3` suites statistically ready
- `3/3` suites canonical-comparable
- total wall time about `21.1s`

### Alignment validation

The final NLP alignment slice validated:

- `cnn_dailymail_summarization` as canonical-ready with a real dataset-backed
  executor path
- `beir_fiqa_canonical` as explicitly refused
- `longbench_narrativeqa_canonical` as explicitly refused

Targeted local validation reached:

- focused NLP/canonical slice: `18 passed, 4 deselected`
- broader provenance/science-exec slice: `22 passed`

No pod was needed during WS31 because the benchmark packs stayed statistically
ready and runtime-safe on local hardware.

## Refusal Policy Preserved

`WS31` does not claim universal canonical coverage.

The following suites remain explicitly refused because the repo still lacks a
truthful canonical executor path for them:

- `generic_ml/openml_cc18_classification`
- `graph_ml/cora_depth_canonical`
- `computer_vision/cifar10_c_corruption`
- `deep_learning/cifar10_scaling_canonical`
- `natural_language_processing/beir_fiqa_canonical`
- `natural_language_processing/longbench_narrativeqa_canonical`
- `reinforcement_learning/minari_cartpole_exploration`
- `reinforcement_learning/minari_offline_online_transfer`

That is an intended WS31 outcome, not a defect. The workstream closes because
benchmark truth is now strong enough to support Phase 4, and the remaining
gaps are explicitly named rather than blurred into false readiness.

## Conclusion

`WS31` proved the benchmark claim that matters:

- TAR can execute canonical benchmark paths with real statistical readiness
- TAR can refuse unsupported canonical suites explicitly
- benchmark reporting is now much harder to bluff

This closes the canonical benchmark and statistical-validation phase without
pretending coverage that the current executor stack does not have.

## What Is Next

The next active workstream is `WS32`.

`WS32` should deepen literature ingestion so TAR can move from stronger
benchmark truth into stronger provenance-preserving scientific evidence
handling.
