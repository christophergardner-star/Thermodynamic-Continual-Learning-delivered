# WS27 Parse Variant Assessment

## Purpose

This document records the bounded local assessment requested in
[ws27_parse_hardening_plan.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27_parse_hardening_plan.md):

- use the new parse-regression slice
- test the `192` and `256` runtime variants
- decide whether runtime budget alone is enough before any new pod cycle

## Important Boundary

This workstation cannot run the true 7B branch model locally:

- no CUDA-capable execution path for the 7B adapter line
- local GPU is not suitable for the `Qwen/Qwen2.5-7B-Instruct` branch eval
- the exact 7B base is not cached locally

So this assessment uses the next-best rigorous local method:

- the exact failing item slice from the real `WS27R1 run1` eval outputs
- the exact filtered eval pack derived from those failing items
- Qwen2.5 tokenizer-family token-budget analysis on:
  - the actual truncated model outputs
  - the real family targets in the training corpus

This is an inference from local artifacts, not a fresh 7B generation run.

## Artifacts Used

Real failure slice:

- `eval_artifacts/ws27_parse_regression_slice_v1`

Exact filtered eval pack built from that slice:

- `eval_artifacts/tar_operator_eval_ws27_parse_regression_v1`

Candidate runtime configs:

- `configs/tar_operator_eval_ws27_parse_regression_runtime_192.json`
- `configs/tar_operator_eval_ws27_parse_regression_runtime_256.json`

## What The Slice Contains

The real failure slice contains `63` parse failures:

- `falsification_planning = 51`
- `project_resume = 12`

Those failures came directly from:

- `ws27r1_run1_probe_eval`
- `ws27r1_run1_ws26_regression_eval`

## Token-Budget Findings

Using a cached local Qwen2.5 tokenizer-family snapshot, the actual failing
outputs tokenise as:

### Actual Truncated Outputs

`falsification_planning`

- count: `51`
- min: `127`
- max: `128`
- avg: `127.08`

`project_resume`

- count: `12`
- min: `128`
- max: `128`
- avg: `128.0`

This is decisive evidence that the current parse failures are saturating the
existing `max_new_tokens = 128` cap.

### Real Training-Target Lengths

`falsification_planning`

- count: `433`
- min: `129`
- p50: `130`
- p95: `130`
- max: `130`
- avg: `129.93`

`project_resume`

- count: `386`
- min: `358`
- p50: `366`
- p95: `380`
- max: `387`
- avg: `368.55`

## Interpretation

### `falsification_planning`

This family is almost certainly budget-bound.

The learned target format is about `130` tokens, and the failed outputs stop at
`127-128`.

So:

- `192` should be sufficient
- `256` should be more than sufficient

### `project_resume`

This family is not fixable by runtime budget alone under the current learned
format.

The learned target format is roughly `358-387` tokens, with a median of `366`.

So:

- `192` is not enough
- `256` is still below the typical learned target length

That means the branch is still trying to emit a long verbose resume schema that
does not fit the existing bounded output budget.

## Decision

Runtime budget increase **alone** will not collapse the remaining parse failures.

More precisely:

- it will likely fix or nearly fix `falsification_planning`
- it will not fully fix `project_resume`

So the correct professional conclusion is:

- do **not** open a new pod yet
- do **not** treat `192` or `256` as a complete fix by themselves
- do one more local contract/supervision tightening step for
  `project_resume`

## Next Correct Step

The next bounded local step should be:

1. tighten the `project_resume` output contract so the model is steered toward a
   shorter schema-complete JSON answer
2. optionally tighten `falsification_planning` the same way
3. keep the `192` and `256` runtime variants ready
4. only then decide whether a final small pod refinement cycle is justified

## Bottom Line

The local assessment answered the question cleanly:

- `192/256` help
- `192/256` are not enough on their own
- the remaining work is a small format/contract refinement, not another broad
  branch redesign
