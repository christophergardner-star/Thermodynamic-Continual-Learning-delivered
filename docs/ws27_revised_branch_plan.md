# WS27 Revised Branch Plan

## Status

This document replaces the initial `WS27` branch choice after the first probe
result recorded in
[ws27_probe_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27_probe_closeout.md).

## Revised Go / No-Go Call

- current coder-backbone variant: **no-go**
- `WS27` overall as a research branch: **conditional go**

That conditional go means:

- continue `WS27`, but only with a redesigned branch that changes the right
  variable
- do **not** repeat the first coder-backbone probe as-is

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

The next `WS27` probe should only advance to a larger run if all of these are
true:

- `WS27` probe mean score is materially above the first failed probe
- `WS27` probe decision accuracy is materially above the first failed probe
- no major TCL family remains at `0.0` decision accuracy
- honesty overclaim returns to `0.0`
- parse error remains at or below the current probe level
- `WS26` non-regression checks stay inside an acceptable tolerance band

Practical target for the revised probe:

- `mean_score >= 0.60`
- `decision_accuracy >= 0.50`
- `parse_error_rate <= 0.12`
- `overclaim_rate = 0.0`

Those are still only probe targets. They are not the final branch success bar.

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

1. compare revised probe vs failed probe
2. compare revised probe vs `WS26`
3. decide whether a serious `WS27` run is finally justified

## Pod Usage

Do **not** open a new pod yet.

The next pod is justified only when:

- adapter-continuation support is implemented
- revised dataset artifacts are built
- revised eval configs exist
- local dry-runs are green

Recommended next pod:

- `A100 80GB`
- persistent volume enabled
- `300GB+` disk

Reason:

- the revised branch should be judged against the `WS26` standard on a stable,
  repeatable pod class

## Immediate Next Step

The next concrete step is local:

- implement adapter-continuation support in
  [train_tar_operator_sft.py](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/train_tar_operator_sft.py)

That is the right next engineering move before any new GPU spend.
