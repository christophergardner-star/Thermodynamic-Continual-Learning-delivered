# TAR Operator Eval Release WS26 v1

## Release

- eval pack dir: `eval_artifacts/tar_operator_eval_ws26_v1`
- manifest: `eval_artifacts/tar_operator_eval_ws26_v1/eval_manifest.json`
- eval version: `tar-operator-eval-ws26-v1`

## Source

- source dataset: `dataset_artifacts/tar_master_dataset_ws26_merged_v1`
- source split: `tar_master_dataset_test.jsonl`

## Summary

- held-out items: `742`
- held-out lineages: `207`

## Task Families

- `benchmark_honesty`: `40`
- `claim_lineage_audit`: `50`
- `decision_rationale`: `43`
- `endpoint_observability_diagnosis`: `10`
- `evidence_debt_judgement`: `39`
- `execution_diagnosis`: `39`
- `falsification_planning`: `39`
- `portfolio_governance`: `45`
- `portfolio_staleness_recovery`: `39`
- `prioritization`: `39`
- `problem_scoping`: `40`
- `project_resume`: `39`
- `reproducibility_refusal`: `8`
- `sandbox_policy_reasoning`: `89`
- `tcl_regime_diagnosis`: `19`
- `tcl_failure_mode_classification`: `12`
- `tcl_anchor_policy_judgement`: `12`
- `tcl_intervention_selection`: `12`
- `tcl_trace_analysis`: `19`
- `tcl_trace_anomaly_diagnosis`: `12`
- `tcl_regime_transition_forecast`: `12`
- `tcl_recovery_planning`: `18`
- `tcl_recovery_confidence_estimation`: `11`
- `tcl_run_triage`: `11`
- `verification_judgement`: `45`

## Local Sanity Result

Heuristic predictor run:

- items: `122`
- items: `742`
- mean score: `0.3972`
- decision accuracy: `0.2709`
- parse error rate: `0.0000`
- overclaim rate: `0.0081`

## Notes

- This release extends the WS24 harness rather than replacing it.
- It is intended to judge whether a WS26 retrain is actually more TCL-native
  than the WS25 model.
