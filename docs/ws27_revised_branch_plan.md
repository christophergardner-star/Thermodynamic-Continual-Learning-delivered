# WS27 Revised Branch Plan

## Status

This document replaces the initial `WS27` branch choice after the first probe
result recorded in
[ws27_probe_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27_probe_closeout.md).

## Revised Go / No-Go Call

- current coder-backbone variant: **no-go**
- `WS27` overall as a research branch: **go**

That go is grounded first in the true-continuation rerun recorded in
[ws27r1_true_continuation_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r1_true_continuation_closeout.md).

It is now further supported by the full `run1` result recorded in
[ws27r1_run1_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r1_run1_closeout.md).

Current status:

- current coder-backbone variant remains rejected
- revised `WS27R1` branch is validated
- `WS27R1 run1` is complete
- `WS27R2` refinement is complete
- `WS27` is now closed successfully

## Core Revision

The revised first-class `WS27` branch should be:

- keep the proven operator backbone
- keep the proven operator format and eval contract
- change the **supervision and continuation strategy**, not the whole backbone

Primary backbone for the revised branch:

- `Qwen/Qwen2.5-7B-Instruct`

Reason:

- this is the backbone that already produced the validated `WS25` and `WS26`
  operator results
- the failed probe suggests the main gap is not generic text emission or
  compute fit
- the gap is domain judgement under TCL/honesty/falsification pressure

## Revised Branch Type

Use a revised `Candidate A`, not the initial coder swap.

Working name:

- `WS27-R1: WS26-Continuation + Auxiliary TCL/ASC Supervision`

## Revised Hypothesis

The revised branch hypothesis is:

- a continuation from the proven `WS26` operator line, with targeted
  TCL/ASC-specific supervision and non-regression control, is more likely to
  produce real branch value than a first-step backbone swap

## Concrete Design

### 1. Keep The Backbone

Use:

- `Qwen/Qwen2.5-7B-Instruct`

Do not change the base model again until the revised branch either succeeds or
fails clearly.

### 2. Add Continuation Support

Before the next pod cycle, add trainer support for:

- optional initialization from an existing adapter path

The practical target is:

- continue from the `WS26` adapter rather than starting a fresh branch adapter
  from scratch

This is important because `WS26` already earned:

- honesty discipline
- portfolio control
- strong TCL improvement relative to `WS25`

The revised branch should build on those gains instead of discarding them.

### 3. Build A Delta Dataset Instead Of Swapping Backbones

Create a private `WS27` delta dataset focused on the failure families.

Mandatory emphasis families:

- `tcl_anchor_policy_judgement`
- `tcl_failure_mode_classification`
- `tcl_intervention_selection`
- `tcl_recovery_confidence_estimation`
- `tcl_regime_transition_forecast`
- `tcl_run_triage`
- `tcl_trace_anomaly_diagnosis`
- `falsification_planning`
- `verification_judgement`
- `benchmark_honesty`
- `reproducibility_refusal`
- `claim_lineage_audit`
- `project_resume`

### 4. Oversample Weak Families Intentionally

Do not feed the revised branch the full merged corpus with the same family mix
and hope it self-corrects.

Instead build:

- `tar_master_dataset_ws27_delta_v1`
- `tar_master_dataset_ws27_branch_v1`

Recommended composition for `ws27_branch_v1`:

- `50%` weak-family delta examples
- `30%` representative `WS26` merged examples
- `20%` honesty/falsification non-regression examples

This is the right way to target the actual observed failures.

### 5. Tighten Structured Output On The Weak Families

The probe showed that parse quality was not the main problem, but it was still a
problem in:

- `falsification_planning`
- `project_resume`
- parts of `decision_rationale`

So the revised branch should also include:

- schema-strict examples for those families
- more explicit gold JSON contracts
- more refusal-safe operator language examples

### 6. Add A Non-Regression Gate

The revised branch must be judged on two packs:

- a branch-specific `WS27` probe pack focused on TCL/ASC-heavy cases
- the proven `WS26` eval pack as a non-regression gate

This matters because `WS27` must not buy TCL gains by giving back:

- honesty
- reproducibility discipline
- structured output reliability

## Revised Success Criteria

The revised probe has now cleared the branch-justification bar on the true
`WS26` continuation line.

`WS27R1 run1` is now complete.

The branch should now be treated as **successful** if all of these remain true:

- it preserves `overclaim_rate = 0.0`
- it preserves strong WS26 non-regression behaviour
- it remains materially above the failed coder-backbone path
- it improves or at least stabilizes the true-continuation probe result
- parse reliability no longer blocks closure after `WS27-R2`

Reference result for the true-continuation probe:

- `mean_score = 0.8207`
- `decision_accuracy = 0.8133`
- `parse_error_rate = 0.16`
- `overclaim_rate = 0.0`

That closure is now confirmed in
[ws27r2_refine_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r2_refine_closeout.md).

## Revised Execution Order

### Phase 1: Local Redesign

Local only.

Tasks:

1. implement adapter-continuation support in the trainer
2. build `WS27` delta dataset generation rules
3. freeze `ws27_branch_v1`
4. freeze revised eval configs

### Phase 2: Local Validation

Local only.

Tasks:

1. dry-run the revised train config
2. dry-run the revised eval config
3. run regression tests

### Phase 3: Revised Probe

Pod required.

Tasks:

1. pull the public repo
2. sync the private revised dataset
3. run the revised probe
4. run the revised branch eval
5. run the `WS26` non-regression eval

### Phase 4: Decide Again

Local only after artifacts are secured.

Tasks:

1. compare the true-continuation probe vs failed probe
2. compare the true-continuation probe vs `WS26`
3. confirm whether `run1` is justified

This phase is now complete. `run1` is justified.

## Pod Usage

Do **not** open another pod for `WS27`.

The bounded hardening pod cycle is complete and no further `WS27` pod work is
currently justified.

That means:

- parse-regression fixtures are in place
- candidate runtime/contract adjustments are in place
- local tests are green
- a bounded refinement run is the next immediate step

Recommended next pod:

- `A100 80GB`
- persistent volume enabled
- `300GB+` disk

Reason:

- the revised branch should be judged against the `WS26` standard on a stable,
  repeatable pod class

## Immediate Next Step

The next concrete step is no longer inside `WS27`.

`WS27` is closed. The next active work is the roadmap transition into
`Phase 4`, starting from
[phase4_roadmap.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/phase4_roadmap.md).
