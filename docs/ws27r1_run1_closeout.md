# WS27R1 Run1 Closeout

## Scope

This document closes the first full `WS27R1 run1` execution on the true
`WS26` continuation line.

It supersedes the prior probe-only posture recorded in
[ws27r1_true_continuation_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r1_true_continuation_closeout.md).

## Provenance

Continuation base:

- `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

True regenerated `WS26` adapter hash:

- `25216ebe36a4278755facbcbc4749a156a2b7a008ed5635697784dddb6c1159d`

`WS27R1 run1` adapter output:

- `training_artifacts/ws27r1_qwen25_7b_run1/final_adapter`

`WS27R1 run1` adapter hash:

- `aa7c2fe7d558b7a8c18a19c3d67b6b0879e467665ff40d3e8cdcf5b74e8ce0f2`

## Train Result

Configuration:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- dataset: `tar_master_dataset_ws27_branch_v1`
- continuation adapter:
  `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

Train summary:

- `train_runtime = 1781.9269`
- `train_loss = 0.0011198080488648218`
- `train_records = 4767`
- `validation_records = 658`

The run manifest confirms:

- `resume_adapter_path = /workspace/Thermodynamic-Continual-Learning-delivered/training_artifacts/ws26_qwen25_7b_run1/final_adapter`

So this is a clean continuation from the true `WS26` line.

## Branch Eval Result

Eval pack:

- `tar-operator-eval-ws27r1-probe-v1`

Overall:

- `mean_score = 0.823`
- `decision_accuracy = 0.8133333333333334`
- `parse_error_rate = 0.16`
- `overclaim_rate = 0.0`

Error buckets:

- `none = 120`
- `parse_error = 24`
- `falsification_or_verification_mismatch = 3`
- `tcl_reasoning_mismatch = 2`
- `honesty_mismatch = 1`

## WS26 Non-Regression Gate

Eval pack:

- `tar-operator-eval-ws27r1-ws26-regression-v1`

Overall:

- `mean_score = 0.8503690036900371`
- `decision_accuracy = 0.8450184501845018`
- `parse_error_rate = 0.14391143911439114`
- `overclaim_rate = 0.0`

Error buckets:

- `none = 228`
- `parse_error = 39`
- `falsification_or_verification_mismatch = 3`
- `honesty_mismatch = 1`

This preserves the core `WS26` behavior envelope well enough to treat the branch
as non-regressive on the important honesty and governance axes.

## What Actually Failed

The branch is not failing broadly. It is failing narrowly and repeatedly in a
small set of structured-output families.

### Probe Eval Parse Failures

All `24` probe parse errors are concentrated in exactly two families:

- `falsification_planning = 12`
- `project_resume = 12`

### WS26 Regression Parse Failures

All `39` regression parse errors are concentrated in:

- `falsification_planning = 39`

### Failure Shape

The failures are not random malformed outputs. They are consistently truncated
JSON responses.

Observed shape:

- nested object starts correctly
- field values are mostly semantically plausible
- output is cut before JSON closure
- parser records `invalid_json`

This is important because it means the branch is not primarily failing on
reasoning quality in those families. It is failing on output completion under a
strict schema contract.

## Root-Cause Reading

The most likely immediate cause is the current eval/runtime output budget:

- `max_new_tokens = 128`

That budget is applied in:

- `configs/tar_operator_eval_ws27r1_probe_runtime.json`
- `configs/tar_operator_eval_ws27r1_ws26_regression_runtime.json`

The truncated responses in `falsification_planning` and `project_resume` are
long nested JSON objects. They are exactly the kinds of outputs most likely to
clip first under a hard generation cap.

This does **not** rule out training-side format weakness. It does mean the
highest-value next step is bounded structured-output hardening, not another
backbone or branch redesign.

## Decision

`WS27R1 run1` is a **success with a structured-output caveat**.

More precisely:

- the branch is viable
- the branch preserves `overclaim_rate = 0.0`
- the branch preserves strong TCL performance
- the branch preserves the WS26 non-regression posture
- the remaining weakness is bounded and interpretable

So the correct professional reading is:

- do **not** reopen backbone-selection work
- do **not** start another broad branch redesign
- do a short local parse-hardening pass next
- only rent another pod if that bounded refinement cycle is approved

## Immediate Next Step

The next step is defined in
[ws27_parse_hardening_plan.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27_parse_hardening_plan.md).

That next step is local first.
