# WS27 Branch Design

Status:

- this document records the **initial** `WS27` branch choice that was taken to
  the first pod-backed feasibility probe
- that initial branch has now been superseded by
  [ws27_revised_branch_plan.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27_revised_branch_plan.md)
  after the probe result

## Primary Branch Choice

`WS27` will start with `Candidate A: Operator-Plus-Auxiliary-TCL`.

This is the correct first branch because it preserves the validated TAR
operator stack from `WS25` and `WS26` while changing one major variable at a
time:

- move from the general instruct backbone to a code/TCL-heavier backbone
- keep the secure TAR operator SFT path
- keep the same evaluation contract
- use the TCL-enriched `WS26` merged dataset as the first branch corpus

The first `WS27` cycle is therefore a disciplined branch probe, not an attempt
to declare the experimental ASC scaffold canonical.

## Branch Objective

The first `WS27` question is:

- does a coder-oriented 7B backbone improve TCL- and ASC-adjacent operator
  behaviour enough to justify a separate branch?

This is narrower and more defensible than trying to port the full ASC objective
 stack into a large model immediately.

## Backbone Choice

Primary backbone:

- `Qwen/Qwen2.5-Coder-7B-Instruct`

Reasons:

- already approved in the secure SFT trainer
- closer to the runtime-debugging and operator-control style needed for the
  branch
- cheaper and cleaner than jumping straight into a larger or less familiar
  backbone

## Dataset Choice

Primary branch dataset:

- `dataset_artifacts/tar_master_dataset_ws26_merged_v1`

Reason:

- this is the strongest private merged dataset currently available
- it already carries the deeper TCL task families introduced before `WS27`
- it avoids inventing a new branch-only dataset before the backbone probe is
  even justified

## Training Path

The first branch uses the existing secure TAR operator trainer:

- [train_tar_operator_sft.py](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/train_tar_operator_sft.py)

This keeps:

- LoRA/QLoRA discipline
- manifest hashing
- local dataset restrictions
- remote-model allowlist enforcement
- artifact-layout compatibility with the existing eval path

The first two branch configs are:

- [tar_operator_qwen25_coder_7b_ws27_probe.json](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/configs/tar_operator_qwen25_coder_7b_ws27_probe.json)
- [tar_operator_qwen25_coder_7b_ws27_run1.json](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/configs/tar_operator_qwen25_coder_7b_ws27_run1.json)

## Evaluation Contract

The branch must be judged against the same class of held-out behaviour as the
mainline operator path.

Primary runtime configs:

- [tar_operator_eval_ws27_probe_runtime.json](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/configs/tar_operator_eval_ws27_probe_runtime.json)
- [tar_operator_eval_ws27_runtime.json](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/configs/tar_operator_eval_ws27_runtime.json)

The fit probe uses the smaller probe pack. The first serious branch run uses
the full `WS27` runtime pack.

## Experimental ASC Compatibility

`WS27` still needs to be able to compare against an actual ASC-wrapped
checkpoint if Candidate C is explored later. That compatibility is now enabled
in the evaluation runtime through the `asc_causal_lm` predictor path.

This matters because the experimental ASC branch saves:

- base model weights
- `warp.pt`
- `asc_config.json`

Without explicit ASC-aware loading in the eval harness, the branch would be
impossible to compare cleanly against `WS25` and `WS26`.

This ASC-aware eval path is support infrastructure only. It does **not** mean
the experimental ASC scaffold is now the default `WS27` branch.

## Fit-Probe Success Conditions

The first pod-backed fit probe is a success only if all of these are true:

- prompt-only probe eval runs cleanly on the probe pack
- the coder-backbone probe train config fits and writes checkpoints
- the adapter-backed probe eval runs cleanly
- parse reliability is not materially worse than `WS26`
- honesty metrics do not materially regress
- TCL-specific metrics show enough promise to justify the first serious branch
  run

## Go / No-Go For Run 1

Move from the probe config to the first serious `WS27` run only if:

- the probe completes without environment instability
- the probe model beats the prompt-only probe baseline materially
- the result is not obviously inferior to `WS26` on honesty and control

If those conditions are not met, the branch should stop and be analyzed before
more GPU time is spent.
