# WS38 — Self-Improvement Loop

Status: local slice complete, pod slice pending

## Local Slice Delivered

- `FrozenAnchorPackManifest`, `TrainingSignalRecord`, `CuratedDeltaRecord`,
  `RetrainRecord`, `SelfImprovementCycleRecord`, `SelfImprovementPolicy`, and
  `SelfImprovementCycleStatus` schemas
- `SelfImprovementEngine` in
  [tar_lab/self_improvement.py](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/tar_lab/self_improvement.py)
- one-time anchor-pack sealing against
  [run_manifest.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/external_validation/run_manifest.json)
  SHA256
- signal curation with hard overclaim rejection and anchor-overlap exclusion
- delta assembly with diversity scoring and minimum signal-count gate
- consecutive gate-failure pausing (`3` failures -> `human_resume_required`)
- max auto-cycle cap (`5` cycles -> `paused_cycle_limit`)
- orchestrator/control/CLI wrappers for all `9` commands
- `run1` and deploy explicitly stubbed with `NotImplementedError` and a pod note
- `8` tests in
  [test_self_improvement.py](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/tests/test_self_improvement.py)

## Pod Slice Required For

- actual LoRA fine-tune of curated delta (`run1`)
- real anchor-pack eval with GPU inference
- gate evaluation on real probe/run1 scores
- adapter deployment into checkpoint registry
- trigger: when `list_training_signals()` returns `>= 20` clean signals from
  real TAR research runs

## Anchor Pack Baseline

- external `mean_score = 0.4625`
- external `overclaim_rate = 0.0`
- sealed from
  [run_manifest.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/external_validation/run_manifest.json)
