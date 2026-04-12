# WS27 Probe Closeout

## Scope

This document closes the first `WS27` feasibility probe:

- branch variant: `Qwen/Qwen2.5-Coder-7B-Instruct`
- training path: secure TAR operator LoRA probe
- dataset: `tar_master_dataset_ws26_merged_v1`
- eval pack: `tar-operator-eval-ws27-probe-v1`

This probe was designed to answer one question only:

- is the first coder-backbone branch strong enough to justify a larger `WS27`
  run?

## Probe Result

The answer is **no**.

This branch is a **no-go** for continuation in its current form.

That is a no-go on the **current branch variant**, not a no-go on `WS27`
overall.

## Measured Results

Prompt-only baseline on the frozen probe pack:

- `mean_score = 0.0079`
- `decision_accuracy = 0.0000`
- `parse_error_rate = 0.6849`
- `overclaim_rate = 0.0000`

Probe-trained adapter on the same pack:

- `mean_score = 0.2753`
- `decision_accuracy = 0.1644`
- `parse_error_rate = 0.1062`
- `overclaim_rate = 0.0068`

Training itself was stable:

- `6659` train records
- `932` validation records
- `train_loss = 0.6847`
- `eval_loss = 0.3263`
- probe runtime about `165s`

## Reading

The probe is not a trivial failure. It established three important facts:

1. the branch fits and trains cleanly on commodity pod hardware
2. the public-repo plus private-dataset workflow is workable
3. the coder backbone does improve materially over prompt-only generation

But those are **feasibility** wins, not **branch-justification** wins.

The branch fails where `WS27` needed it to succeed:

- TCL-heavy reasoning
- honesty-heavy judgement
- falsification / verification discipline

## Failure Pattern

Overall error buckets for the probe adapter:

- `tcl_reasoning_mismatch = 114`
- `honesty_mismatch = 42`
- `falsification_or_verification_mismatch = 24`
- `parse_error = 31`
- `overclaim = 2`

The key signal is that parse quality improved while TCL decision quality did
not.

That means the branch did **not** fail because the model could not emit JSON.
It failed because the backbone/supervision choice did not produce the right
domain judgements.

## Suite-Level Reading

What worked:

- `portfolio` suite:
  - `decision_accuracy = 0.8333`
  - `mean_score = 0.8194`
- `sandbox_policy_reasoning`:
  - `decision_accuracy = 1.0000`
  - `mean_score = 1.0000`
- `portfolio_staleness_recovery`:
  - `decision_accuracy = 1.0000`
  - `mean_score = 1.0000`

What did not work:

- `tcl` suite:
  - `decision_accuracy = 0.0000`
  - `mean_score = 0.0682`
  - `tcl_reasoning_mismatch = 114`
- `honesty` suite:
  - `decision_accuracy = 0.2424`
  - `mean_score = 0.3833`
  - `overclaim_rate = 0.0303`
- `falsification` suite:
  - `decision_accuracy = 0.0000`
  - `mean_score = 0.2056`
- `resume` suite:
  - `decision_accuracy = 0.0556`
  - `mean_score = 0.2819`

This is the decisive pattern:

- governance/control stayed strong
- TCL judgement stayed weak
- honesty did not stay at the standard set by the proven operator line

## Interpretation

The most likely interpretation is:

- the first major variable change was the wrong one

More concretely:

- swapping to the coder backbone as the first `WS27` branch variable preserved
  some control-style strengths
- but it degraded the calibrated research-operator behaviour that the
  `WS25`/`WS26` line had already established

This is an inference from the probe results, not a proof. But it is the most
coherent reading of the measured behaviour.

## Decision

### No-Go

The following branch variant is **rejected**:

- `WS27` first branch = `Qwen/Qwen2.5-Coder-7B-Instruct` backbone swap +
  otherwise standard TAR operator LoRA path

It is not strong enough to justify `run1`.

### What This Is Not

This is **not**:

- a rejection of `WS27` as a research workstream
- a rejection of future ASC-aware evaluation support
- a rejection of later experimental ASC scaffolds

It is a rejection of the **first branch choice**.

## Professional Next Move

The next move is:

1. stop GPU spending on this branch variant
2. preserve artifacts
3. redesign the branch locally
4. keep the proven operator backbone and change the supervision/objective plan
   instead

That redesigned plan is saved in
[ws27_revised_branch_plan.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27_revised_branch_plan.md).

## Pod Decision

The pod should be terminated after artifacts are secured locally.

No new pod should be opened for `WS27` until:

- the revised branch plan is frozen
- the revised configs are written
- the revised dataset/eval deltas are prepared
- local dry-runs are green
