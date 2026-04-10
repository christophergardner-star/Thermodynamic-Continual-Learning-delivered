# WS24 Execution Spec

`WS24` is **TAR/TCL Evaluation Harness**.

Its purpose is to turn the private `WS23` dataset release into a held-out,
versioned evaluation layer that can judge whether a tuned TAR operator model is
actually better than a prompt-only baseline on TAR-native work.

## Core purpose

After WS24, TAR should have a frozen evaluation release that can score:

- project resume fidelity
- next-action quality
- benchmark honesty
- reproducibility honesty
- falsification planning quality
- contradiction and verification handling
- portfolio governance quality
- TCL regime diagnosis
- TCL recovery planning

## Why WS24 matters

Without WS24:

- `WS25` would produce a checkpoint without a strong measurement layer
- prompt-only and tuned models could not be compared fairly
- honesty regressions could slip through unnoticed
- TCL gains would remain anecdotal instead of measured

## Strategic principle

WS24 is not "run a few prompts and eyeball the outputs."

It is a held-out measurement instrument. The right standard is:

- frozen
- lineage-safe
- family-aware
- honesty-aware
- test-defended

## What WS24 must deliver

1. A frozen evaluation pack

- versioned eval release
- stable item IDs
- stable suite breakdown
- file hashes
- source dataset fingerprints

2. A runnable evaluation harness

- build eval pack from the WS23 held-out split
- run predictors against the eval pack
- emit per-item scores, family scores, suite scores, and run manifests

3. Family-specific scorers

The harness must score TAR-native behavior, not generic text overlap.

4. Baseline-ready runtime

The harness must support:

- prompt-only base model evaluation
- tuned adapter evaluation
- non-GPU local validation predictors

5. Regression coverage

- parse robustness
- scorer correctness
- eval-pack build integrity
- end-to-end harness execution

## Eval families in scope

- `project_resume`
- `prioritization`
- `falsification_planning`
- `portfolio_governance`
- `benchmark_honesty`
- `execution_diagnosis`
- `verification_judgement`
- `decision_rationale`
- `reproducibility_refusal`
- `sandbox_policy_reasoning`
- `endpoint_observability_diagnosis`
- `portfolio_staleness_recovery`
- `claim_lineage_audit`
- `evidence_debt_judgement`
- `tcl_regime_diagnosis`
- `tcl_trace_analysis`
- `tcl_recovery_planning`

## File-by-file scope

### `tar_lab/eval_schemas.py`

- typed evaluation items
- typed run results
- typed aggregate summaries

### `tar_lab/eval_scorers.py`

- family rubrics
- scoring-target extraction
- structured prediction parsing
- family-aware scoring and error buckets

### `tar_lab/eval_harness.py`

- frozen eval-pack build logic
- suite file generation
- predictor interfaces
- end-to-end evaluation runner

### `eval_tar_operator.py`

- secure config loading
- predictor selection
- eval-pack build command
- end-to-end evaluation command

### `configs/tar_operator_eval_ws24_v1.json`

- baseline WS24 eval configuration

### `tests/test_eval_scorers.py`

- scorer and parse regression tests

### `tests/test_eval_harness.py`

- eval-pack build and run regression tests

## Acceptance criteria

WS24 closes only if all are true:

- TAR has a versioned held-out eval release
- eval items are derived from the WS23 held-out split
- family-specific scoring is implemented and tested
- the harness emits per-item and aggregate outputs
- honesty and TCL behavior are explicitly measured
- prompt-only model evaluation is supported in the runtime
- local validation predictors can run end-to-end without GPU spend

## What counts as done

WS24 is done when TAR can truthfully say:

- "this is the held-out standard"
- "this is how base and tuned models will be judged"
- "these metrics reward correctness and honesty, not bluffing"
- "this harness is strong enough to justify `WS25`"
