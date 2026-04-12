# TAR Master Dataset Release WS27 Private v1

## Scope

This note records the private `WS27` continuation datasets and the revised
branch readiness state.

These artifacts are private and remain outside git:

- `dataset_artifacts/tar_master_dataset_ws27_delta_v1`
- `dataset_artifacts/tar_master_dataset_ws27_branch_v1`
- `eval_artifacts/tar_operator_eval_ws27r1_probe_v1`
- `eval_artifacts/tar_operator_eval_ws27r1_ws26_regression_v1`

## Delta Release

- dataset dir: `dataset_artifacts/tar_master_dataset_ws27_delta_v1`
- dataset version: `tar-master-ws27-delta-v1`
- records: `3200`
- splits:
  - train: `2536`
  - validation: `364`
  - test: `300`
- lineages: `1522`
- lineage safe: `true`

Focus families:

- `benchmark_honesty`: `467`
- `claim_lineage_audit`: `467`
- `falsification_planning`: `468`
- `project_resume`: `467`
- `reproducibility_refusal`: `112`
- `tcl_anchor_policy_judgement`: `112`
- `tcl_failure_mode_classification`: `112`
- `tcl_intervention_selection`: `112`
- `tcl_recovery_confidence_estimation`: `98`
- `tcl_regime_transition_forecast`: `110`
- `tcl_run_triage`: `98`
- `tcl_trace_anomaly_diagnosis`: `110`
- `verification_judgement`: `467`

## Branch Release

- dataset dir: `dataset_artifacts/tar_master_dataset_ws27_branch_v1`
- dataset version: `tar-master-ws27-branch-v1`
- records: `6000`
- splits:
  - train: `4767`
  - validation: `658`
  - test: `575`
- lineages: `1845`
- lineage safe: `true`

Component mix:

- delta component: `3000`
- representative component: `1800`
- non-regression component: `1200`

Notable family totals:

- `benchmark_honesty`: `545`
- `claim_lineage_audit`: `540`
- `falsification_planning`: `540`
- `project_resume`: `480`
- `reproducibility_refusal`: `172`
- `sandbox_policy_reasoning`: `1027`
- `verification_judgement`: `542`
- `tcl_anchor_policy_judgement`: `112`
- `tcl_failure_mode_classification`: `112`
- `tcl_intervention_selection`: `112`
- `tcl_recovery_confidence_estimation`: `98`
- `tcl_regime_transition_forecast`: `110`
- `tcl_run_triage`: `98`
- `tcl_trace_anomaly_diagnosis`: `110`

## Revised Eval Packs

### WS27 R1 Probe Pack

- eval dir: `eval_artifacts/tar_operator_eval_ws27r1_probe_v1`
- eval version: `tar-operator-eval-ws27r1-probe-v1`
- held-out items: `150`
- held-out lineages: `50`

Included families:

- `benchmark_honesty`: `12`
- `claim_lineage_audit`: `12`
- `falsification_planning`: `12`
- `project_resume`: `12`
- `reproducibility_refusal`: `8`
- `tcl_anchor_policy_judgement`: `12`
- `tcl_failure_mode_classification`: `12`
- `tcl_intervention_selection`: `12`
- `tcl_recovery_confidence_estimation`: `11`
- `tcl_regime_transition_forecast`: `12`
- `tcl_run_triage`: `11`
- `tcl_trace_anomaly_diagnosis`: `12`
- `verification_judgement`: `12`

### WS26 Non-Regression Gate

- eval dir: `eval_artifacts/tar_operator_eval_ws27r1_ws26_regression_v1`
- eval version: `tar-operator-eval-ws27r1-ws26-regression-v1`
- held-out items: `271`
- held-out lineages: `135`

Included families:

- `benchmark_honesty`: `40`
- `claim_lineage_audit`: `50`
- `falsification_planning`: `39`
- `reproducibility_refusal`: `8`
- `sandbox_policy_reasoning`: `89`
- `verification_judgement`: `45`

## Revised Continuation Configs

- `configs/tar_operator_qwen25_7b_ws27r1_probe.json`
- `configs/tar_operator_qwen25_7b_ws27r1_run1.json`
- `configs/tar_operator_eval_ws27r1_probe_runtime.json`
- `configs/tar_operator_eval_ws27r1_ws26_regression_runtime.json`

These configs continue from:

- `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

## Remaining Local Blocker

The revised branch scaffolding is complete, but the intended `WS26` adapter is
not currently present inside the repo workspace.

That means:

- dataset generation is complete
- eval-pack generation is complete
- config wiring is complete
- continuation mechanics are implemented and test-covered
- a true local dry-run of the revised `WS27` continuation config still requires
  the `WS26` adapter bundle to be restored under:
  `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

No pod is justified until that adapter bundle is restored and the revised train
config has passed a local dry-run.
