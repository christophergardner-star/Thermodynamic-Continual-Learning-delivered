# WS27 Parse Hardening Plan

## Purpose

This plan defines the smallest correct next step after
[ws27r1_run1_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r1_run1_closeout.md).

The aim is not to redesign `WS27`.

The aim is to reduce the remaining structured-output failures in:

- `falsification_planning`
- `project_resume`

without disturbing the gains already established in:

- TCL reasoning
- benchmark honesty
- reproducibility refusal
- non-regression against the `WS26` line

## Problem Statement

Current `WS27R1 run1` branch eval:

- `mean_score = 0.823`
- `decision_accuracy = 0.8133`
- `parse_error_rate = 0.16`
- `overclaim_rate = 0.0`

Current `WS26` non-regression gate:

- `mean_score = 0.8504`
- `decision_accuracy = 0.8450`
- `parse_error_rate = 0.1439`
- `overclaim_rate = 0.0`

The parse-error concentration is narrow:

- probe pack:
  - `falsification_planning = 12`
  - `project_resume = 12`
- regression pack:
  - `falsification_planning = 39`

So the hardening target is bounded and concrete.

## Working Hypothesis

The dominant cause is a combination of:

1. insufficient generation budget for long nested JSON responses
2. insufficient schema-complete supervision for those long-form families

The current eval runtime uses:

- `max_new_tokens = 128`

That is likely too tight for:

- `project_resume`
- `falsification_planning`

The observed outputs are mostly valid in content but clipped before JSON
closure, which strongly supports the completion-budget hypothesis.

## Goal

Reduce parse failure without spending another broad training cycle prematurely.

The first hardening cycle should try to achieve:

- branch eval `parse_error_rate <= 0.08`
- WS26 regression `parse_error_rate <= 0.10`
- `overclaim_rate = 0.0`
- no material drop in:
  - branch `decision_accuracy`
  - branch `mean_score`
  - WS26 non-regression `decision_accuracy`
- WS26 non-regression `mean_score`

## Current Assessment Result

The first bounded local assessment is now complete and recorded in
[ws27_parse_variant_assessment.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27_parse_variant_assessment.md).

Current decision:

- runtime budget increase alone is **not** sufficient
- `falsification_planning` is likely fixable by `192/256`
- `project_resume` is not likely fixable by `192/256` alone

So the next step is a small format/contract refinement pass, not an immediate
new pod cycle.

## Boundaries

Do not do any of the following in this hardening pass:

- change backbone
- change continuation base
- rebuild the whole branch dataset from scratch
- start a new pod immediately
- rerun a full broad branch experiment before local checks are done

This should stay a small, evidence-driven refinement.

## Concrete Work Items

### 1. Freeze A Parse-Regression Slice

Create a small local regression slice built from the actual failing items:

- all `falsification_planning` parse failures from `ws27r1_run1_probe_eval`
- all `project_resume` parse failures from `ws27r1_run1_probe_eval`
- representative `falsification_planning` parse failures from
  `ws27r1_run1_ws26_regression_eval`

Purpose:

- prevent vague "it seems better" judgment
- keep the hardening target tied to real failures

This slice now exists locally at:

- `eval_artifacts/ws27_parse_regression_slice_v1`

Builder:

- `build_ws27_parse_regression_slice.py`

### 2. Add Long-Form Schema Fixtures

Add explicit local fixtures or test cases for the two failing families.

The fixtures should verify:

- JSON closes correctly
- required nested objects are present
- array/object shape is preserved
- parser accepts the result

This should be test-backed, not just doc-backed.

### 3. Introduce Runtime Budget Variants

Create bounded runtime variants for evaluation such as:

- `max_new_tokens = 192`
- `max_new_tokens = 256`

Do not replace the current configs immediately. Add candidate variants and test
them deliberately.

Reason:

- the current failures are clipped outputs
- generation budget is the most likely first lever

Candidate configs now exist locally:

- `configs/tar_operator_eval_ws27r1_probe_runtime_192.json`
- `configs/tar_operator_eval_ws27r1_probe_runtime_256.json`
- `configs/tar_operator_eval_ws27r1_ws26_regression_runtime_192.json`
- `configs/tar_operator_eval_ws27r1_ws26_regression_runtime_256.json`

### 4. Add Family-Aware Output Contract Checks

Audit the prompt/contract wording for:

- `falsification_planning`
- `project_resume`

If the contract is underspecified or encourages extra prose, tighten it so the
model is asked for:

- JSON only
- no commentary
- required closing structure

### 5. Add Delta Supervision Only If Needed

If runtime budget increase alone does not collapse the parse failures, add a
small targeted supervision delta:

- more long-form `falsification_planning` examples
- more long-form `project_resume` examples
- explicit schema-complete exemplars

Keep this delta small and local to the failure families.

### 6. Re-Evaluate Locally First

Before any new pod cycle:

- dry-check the new runtime variants
- run regression tests
- confirm the parse-regression slice improves

Only then decide whether a bounded pod refinement run is justified.

## Acceptance Criteria

This hardening plan is successful only if all are true:

- the remaining parse failures are reproduced locally on a fixed slice
- at least one bounded runtime or contract adjustment materially reduces them
- the JSON parser accepts the repaired outputs
- the change does not compromise honesty or non-regression logic
- the next pod, if any, is for a specific bounded refinement run, not
  exploratory guesswork

## Pod Policy

Do **not** open a new pod for this plan yet.

Open a pod only if:

1. the local parse-regression slice exists
2. candidate runtime/contract changes are implemented
3. local tests are green
4. the next immediate step is a bounded refinement run

That keeps the next GPU cycle disciplined.

## Recommended Immediate Sequence

1. build the parse-regression slice
2. add fixtures/tests for the two failing families
3. add candidate runtime configs with higher output budget
4. complete the bounded local budget assessment
5. tighten the `project_resume` format/contract path locally
6. then decide whether the branch should be closed as successful-with-caveat or
   sent to one final bounded pod refinement
