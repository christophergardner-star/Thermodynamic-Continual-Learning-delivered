# WS27R1 True-Continuation Closeout

## Scope

This document closes the clean `WS27R1` continuation rerun against the **true
regenerated `WS26` adapter**.

This run replaced the earlier staged-proxy continuation probe.

## Provenance

### True WS26 Regeneration

`WS26` was rerun cleanly to regenerate:

- `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

Regenerated adapter hash:

- `25216ebe36a4278755facbcbc4749a156a2b7a008ed5635697784dddb6c1159d`

WS26 train summary:

- `train_runtime = 4027.4735`
- `train_loss = 0.03828030684181183`
- `train_records = 6659`
- `validation_records = 932`

### WS26 Validation

Prompt-only baseline on the frozen WS26 eval pack:

- `mean_score = 0.02607816711590293`
- `decision_accuracy = 0.01078167115902965`
- `parse_error_rate = 0.623989218328841`
- `overclaim_rate = 0.0`

Regenerated WS26 adapter on the same pack:

- `mean_score = 0.8806603773584903`
- `decision_accuracy = 0.876010781671159`
- `parse_error_rate = 0.11185983827493262`
- `overclaim_rate = 0.0`

That confirms the regenerated adapter is a valid continuation base.

## WS27R1 True-Continuation Probe

### Train Run

The `WS27R1` probe was run with:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- continuation adapter:
  `training_artifacts/ws26_qwen25_7b_run1/final_adapter`
- dataset: `tar_master_dataset_ws27_branch_v1`

Train summary:

- `train_runtime = 241.3015`
- `train_loss = 0.0057114289486586735`
- `train_records = 4767`
- `validation_records = 658`

The run manifest records:

- `resume_adapter_path = /workspace/Thermodynamic-Continual-Learning-delivered/training_artifacts/ws26_qwen25_7b_run1/final_adapter`

So this is a clean scientific continuation from the regenerated `WS26` line.

### Probe Eval Result

On `tar-operator-eval-ws27r1-probe-v1`:

- `mean_score = 0.8206666666666667`
- `decision_accuracy = 0.8133333333333334`
- `parse_error_rate = 0.16`
- `overclaim_rate = 0.0`

Error buckets:

- `none = 120`
- `parse_error = 24`
- `tcl_reasoning_mismatch = 3`
- `falsification_or_verification_mismatch = 2`
- `honesty_mismatch = 1`

### WS26 Non-Regression Gate

On `tar-operator-eval-ws27r1-ws26-regression-v1`:

- `mean_score = 0.8520295202952031`
- `decision_accuracy = 0.8487084870848709`
- `parse_error_rate = 0.14391143911439114`
- `overclaim_rate = 0.0`

Error buckets:

- `none = 229`
- `parse_error = 39`
- `falsification_or_verification_mismatch = 2`
- `honesty_mismatch = 1`

## Comparison To The Staged-Proxy Probe

Earlier staged-proxy `WS27R1` probe:

- `mean_score = 0.5480`
- `decision_accuracy = 0.4000`
- `parse_error_rate = 0.16`
- `overclaim_rate = 0.0`

True-continuation `WS27R1` probe:

- `mean_score = 0.8207`
- `decision_accuracy = 0.8133`
- `parse_error_rate = 0.16`
- `overclaim_rate = 0.0`

This is the decisive result:

- the branch value was real
- the staged-proxy continuation was the confounder
- the true `WS26` continuation line materially improves the branch outcome

## Decision

`WS27` is now a **go**.

More precisely:

- `WS27R1` is strong enough to justify `run1`
- the branch preserves the core truthfulness contract:
  - `overclaim_rate = 0.0`
  - non-regression gate remains strong
- the remaining weakness is structured output reliability, not branch validity

## Caveat

The true-continuation probe still misses the earlier probe parse target:

- target: `parse_error_rate <= 0.12`
- actual: `parse_error_rate = 0.16`

That is a real caveat and should be handled explicitly during `run1`, but it is
no longer a blocker to branch continuation.

## Professional Reading

The correct reading is:

- do not spend another pod cycle repeating probe logic
- proceed to `WS27R1 run1`
- keep the non-regression gate in place
- keep parse/output-contract behaviour under explicit watch during `run1`

## Pod Status

The regeneration pod cycle can be treated as complete once artifacts are copied
back locally.

The next pod, when opened, should be for:

- `WS27R1 run1`
- branch eval
- WS26 non-regression eval

