# WS27 True WS26 Adapter Recovery Audit

## Purpose

This note records the recovery audit for the intended `WS26` continuation
adapter:

- expected path:
  `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

The goal was to determine whether the **true** `WS26` adapter could be restored
from current local assets before any new `WS27` continuation work.

## Result

The true `WS26` adapter is **not recoverable from the current local artifact
set**.

This is a hard recovery result, not a guess.

## Evidence

### 1. `ws26_private_bundle.tar` does not contain the WS26 adapter

Archive inspection shows:

- `dataset_artifacts/tar_master_dataset_ws26_merged_v1`
- `eval_artifacts/tar_operator_eval_ws26_v1`
- `training_artifacts/ws25_qwen25_7b_run1/final_adapter`

It does **not** contain:

- `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

### 2. `ws26_private_stage` does not contain the WS26 adapter

The extracted stage directory contains:

- `dataset_artifacts/tar_master_dataset_ws26_merged_v1`
- `eval_artifacts/tar_operator_eval_ws26_v1`
- `training_artifacts/ws25_qwen25_7b_run1/final_adapter`

It does **not** contain:

- `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

### 3. Earlier workspace backups do not contain the WS26 adapter

Checked:

- `Thermodynamic-Continual-Learning-delivered_backup_20260410_000153.tar`
- `Thermodynamic-Continual-Learning-delivered_backup_20260410_001328_tcl.tar`

These backups only contain the early trainer workspace state and do not contain
the later `WS26` adapter output.

### 4. Later local artifact bundles contain other adapters, not the WS26 adapter

Checked:

- `ws25_artifacts_bundle.tar`
- `ws27_probe_artifacts_bundle.tar`
- `ws27r1_probe_artifacts.tar`

These contain:

- `training_artifacts/ws25_qwen25_7b_run1/...`
- `training_artifacts/ws27_qwen25_coder_7b_probe/...`
- `training_artifacts/ws27r1_qwen25_7b_probe/...`

They do **not** contain:

- `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

## Current Local `ws26` Path Is A Staged Proxy

To validate the revised `WS27` continuation mechanics, a compatible adapter was
staged locally at:

- `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

That staged adapter is **not** the true `WS26` adapter.

It is a copy of the surviving `WS25` adapter from:

- `ws26_private_stage/training_artifacts/ws25_qwen25_7b_run1/final_adapter`

Hash evidence:

- staged local `training_artifacts/ws26_qwen25_7b_run1/final_adapter/adapter_model.safetensors`
  - `3E00F475E098E5D1454990FFFF466E3A7DEEF0E87789AF5F2B3489315DF2FBCB`
- source `ws26_private_stage/training_artifacts/ws25_qwen25_7b_run1/final_adapter/adapter_model.safetensors`
  - `3E00F475E098E5D1454990FFFF466E3A7DEEF0E87789AF5F2B3489315DF2FBCB`

The adapter configs are byte-equivalent as well.

## Interpretation

The revised `WS27R1` probe was a valid engineering continuation probe, because:

- the continuation path was exercised
- the data and eval packs were real
- the branch mechanics were real

But it was **not** a clean scientific continuation from the true `WS26` model
line, because the original `WS26` adapter artifact is absent.

## Decision

The recovery step should be considered **closed with non-recoverability** from
current local assets.

The correct next choices are:

1. accept the current `WS27R1` result as a staged-adapter branch probe only, or
2. reconstruct the true `WS26` adapter by rerunning `WS26`, then rerun the
   revised `WS27` continuation probe against that true adapter

## Pod Guidance

Do **not** open a new pod for `WS27` continuation until one of the following is
true:

- the true `WS26` adapter has been restored from a missing archive not yet found
- or `WS26` has been deliberately rerun to regenerate the adapter cleanly

Until then, any further `WS27` continuation run would still be continuation from
the staged `WS25` proxy, not the true `WS26` result.
