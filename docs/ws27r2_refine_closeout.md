# WS27R2 Refinement Closeout

## Correction Note

Corrected by independent eval validation pass `2026-04-17`.

The original closeout below reported internal probe and regression pack results.
Those internal figures remain historically true for the bounded refinement run,
but they are **not** the authoritative independent validation result.

The independently validated external-slice metrics are:

- `mean_score = 0.4625`
- `decision_accuracy = 0.4375`
- `parse_error_rate = 0.4375`
- `false_refusal_rate = 0.0`
- `overclaim_rate = 0.0`

The published `WS27R2` mean-score claim is therefore corrected by the
independent eval validation pass. The honesty claim on `overclaim_rate = 0.0`
is confirmed.

## Scope

This document closes the bounded `WS27-R2` refinement cycle that followed
[ws27r1_run1_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r1_run1_closeout.md).

The purpose of `WS27-R2` was narrow and explicit:

- continue from the proven `WS27R1 run1` line
- preserve the `WS26` non-regression posture
- remove the remaining structured-output caveat

This was not a new branch redesign. It was a bounded continuation run.

## Provenance

Continuation base:

- `training_artifacts/ws27r1_qwen25_7b_run1/final_adapter`

True `WS27R1 run1` adapter hash:

- `aa7c2fe7d558b7a8c18a19c3d67b6b0879e467665ff40d3e8cdcf5b74e8ce0f2`

`WS27R2` adapter output:

- `training_artifacts/ws27r2_qwen25_7b_refine_run1/final_adapter`

`WS27R2` adapter weight hash:

- `2808c2252a024133bd0bd4b326e855473f6684fff887ffee2ac20154e22b0dfd`

Configuration:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- dataset: `tar_master_dataset_ws27_branch_v1`
- continuation adapter:
  `training_artifacts/ws27r1_qwen25_7b_run1/final_adapter`
- train config:
  `configs/tar_operator_qwen25_7b_ws27r2_refine.json`

## Train Result

Train summary:

- `train_runtime = 929.2866`
- `train_loss = 0.001176916641978572`
- `train_records = 4767`
- `validation_records = 658`

This was a bounded one-epoch refinement, not a fresh branch run.

## Branch Eval Result At 192 Tokens

Eval pack:

- `tar-operator-eval-ws27r1-probe-v1`

Runtime:

- `max_new_tokens = 192`

Overall:

- `mean_score = 0.8926666666666669`
- `decision_accuracy = 0.88`
- `parse_error_rate = 0.07333333333333333`
- `overclaim_rate = 0.0`

Error buckets:

- `none = 131`
- `parse_error = 11`
- `tcl_reasoning_mismatch = 4`
- `falsification_or_verification_mismatch = 2`
- `governance_mismatch = 1`
- `honesty_mismatch = 1`

## WS26 Non-Regression Gate At 192 Tokens

Eval pack:

- `tar-operator-eval-ws27r1-ws26-regression-v1`

Runtime:

- `max_new_tokens = 192`

Overall:

- `mean_score = 0.9959409594095943`
- `decision_accuracy = 0.992619926199262`
- `parse_error_rate = 0.0`
- `overclaim_rate = 0.0`

Error buckets:

- `none = 268`
- `falsification_or_verification_mismatch = 2`
- `honesty_mismatch = 1`

## Comparison To WS27R1 Run1

`WS27R1 run1`:

- `mean_score = 0.823`
- `decision_accuracy = 0.8133333333333334`
- `parse_error_rate = 0.16`
- `overclaim_rate = 0.0`

`WS27R2`:

- `mean_score = 0.8926666666666669`
- `decision_accuracy = 0.88`
- `parse_error_rate = 0.07333333333333333`
- `overclaim_rate = 0.0`

`WS27R2` independent external slice:

- `mean_score = 0.4625`
- `decision_accuracy = 0.4375`
- `parse_error_rate = 0.4375`
- `false_refusal_rate = 0.0`
- `overclaim_rate = 0.0`

This is the correct closing signal:

- reasoning improved
- parse reliability improved materially
- honesty stayed intact
- the non-regression gate did not weaken

## Decision

`WS27` training and refinement work remains closed, but the published `WS27R2`
performance claim is corrected by the independent eval validation pass.

More precisely:

- the initial coder-backbone branch remains rejected
- the revised continuation branch is validated
- the bounded refinement cycle improved the internal probe packs, but the
  independent external slice showed materially weaker generalization than the
  original published claim
- there is no further justified `WS27` pod cycle at this time

The branch no longer needs to be described as "successful with a caveat." The
structured-output caveat has been reduced enough to close the workstream.

## Artifacts

Authoritative local result bundle:

- `C:\\Users\\Chris\\contLRN\\ws27r2_results_min_bundle.tar`

Bundle SHA256:

- `c5f4bd29d5c1004c1b21ef736355748996d091ba4ba22e7ae4b796b600bea3c8`

This bundle contains:

- `training_artifacts/ws27r2_qwen25_7b_refine_run1/final_adapter`
- `training_artifacts/ws27r2_qwen25_7b_refine_run1/run_manifest.json`
- `training_artifacts/ws27r2_qwen25_7b_refine_run1/run_summary.json`
- `eval_artifacts/tar_operator_eval_runs/ws27r2_refine_probe_eval_192`
- `eval_artifacts/tar_operator_eval_runs/ws27r2_refine_ws26_regression_eval_192`

## What Is Next

`WS27` is no longer the active bottleneck.

The next work is not another branch experiment. The next work is a roadmap
transition:

1. freeze `WS27` as complete
2. mark `Phase 3` historical
3. start the `Phase 4` roadmap based on the remaining frontier-stack gaps
