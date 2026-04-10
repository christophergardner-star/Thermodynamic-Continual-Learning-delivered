# TAR Operator Eval Spec

This document defines the `WS24` evaluation contract for TAR operator models.

## Eval release layout

The canonical WS24 release is built under `eval_artifacts/` and contains:

- `eval_manifest.json`
- `eval_core.jsonl`
- `eval_resume.jsonl`
- `eval_honesty.jsonl`
- `eval_falsification.jsonl`
- `eval_portfolio.jsonl`
- `eval_tcl.jsonl`
- `scoring_rubrics.json`

## Source of truth

The eval release is carved from the `WS23` held-out split, not from train or
validation data.

Required source inputs:

- dataset manifest hash
- test split hash
- task-family counts
- lineage count

## Item contract

Each eval item carries:

- `item_id`
- `example_id`
- `task_family`
- `task_name`
- `suite_names`
- `lineage_key`
- `messages`
- `input_context`
- `gold_target`
- `scoring_target`
- `provenance`

`scoring_target` is the canonical distilled representation used by the scorer.
This keeps scoring stable even when the original target structure is nested.

## Suite layout

`core`

- all items

`resume`

- `problem_scoping`
- `project_resume`
- `decision_rationale`

`honesty`

- `benchmark_honesty`
- `reproducibility_refusal`
- `sandbox_policy_reasoning`
- `endpoint_observability_diagnosis`
- `claim_lineage_audit`
- `evidence_debt_judgement`

`falsification`

- `execution_diagnosis`
- `verification_judgement`
- `falsification_planning`

`portfolio`

- `prioritization`
- `portfolio_governance`
- `portfolio_staleness_recovery`

`tcl`

- `tcl_regime_diagnosis`
- `tcl_trace_analysis`
- `tcl_recovery_planning`

## Scoring policy

Scoring is family-aware.

The harness uses:

- exact or normalized text comparison for key decisions
- boolean comparison for support/refusal gates
- tolerant numeric comparison where needed
- set overlap for unordered evidence lists and status lists

Headline metrics:

- `mean_score`
- `decision_accuracy`
- `overclaim_rate`
- `false_refusal_rate`
- `parse_error_rate`

Each family also produces an error-bucket breakdown.

## Baseline policy

The harness supports three predictor modes:

- `gold`
  - validation-only predictor that echoes gold targets
- `heuristic`
  - local non-GPU sanity predictor for harness validation
- `hf_causal_lm`
  - prompt-only or adapter-backed model evaluation using `transformers`

The scientific baseline for `WS25` remains:

- prompt-only `Qwen/Qwen2.5-7B-Instruct`

But `gold` and `heuristic` predictors are retained so the harness can be tested
and validated without renting GPU time.

## Security posture

- `trust_remote_code` stays false
- dataset paths are local by default
- eval output is refused inside `dataset_artifacts/`
- approved remote model policy mirrors the secure training path

## Outputs

Each evaluation run emits:

- `results.jsonl`
- `family_breakdown.json`
- `suite_breakdown.json`
- `results.json`
- `errors.jsonl`
- `run_manifest.json`

## Why this matters

This contract makes WS24 a measuring instrument instead of a convenience
script. It is the gate that determines whether `WS25` produces a result worth
believing.
