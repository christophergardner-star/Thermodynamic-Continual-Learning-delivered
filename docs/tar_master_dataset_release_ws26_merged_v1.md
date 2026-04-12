# TAR Master Dataset Release WS26 Merged v1

## Release

- dataset dir: `dataset_artifacts/tar_master_dataset_ws26_merged_v1`
- manifest: `dataset_artifacts/tar_master_dataset_ws26_merged_v1/manifest.json`
- dataset version: `tar-master-ws26-merged-v1`

## Source Releases

- `dataset_artifacts/tar_master_dataset_ws23_v1`
- `dataset_artifacts/tar_master_dataset_ws26_v1`

## Summary

- total records: `8333`
- pre-dedup records: `8397`
- duplicates removed: `64`
- lineages: `2262`
- lineage safe: `true`

## Splits

- train: `6659`
- validation: `932`
- test: `742`

## Why This Is The Training Target

This merged release preserves the broader `WS23` governance, honesty,
verification, and portfolio coverage while adding the deeper TCL-native
families from `WS26`.

That makes it the correct dataset for the next retraining cycle. Using the
TCL-heavy delta release alone would risk domain deepening at the cost of
broader operator behavior.

## TCL Family Totals

- `tcl_regime_diagnosis`: `192`
- `tcl_failure_mode_classification`: `112`
- `tcl_anchor_policy_judgement`: `112`
- `tcl_intervention_selection`: `112`
- `tcl_trace_analysis`: `190`
- `tcl_trace_anomaly_diagnosis`: `110`
- `tcl_regime_transition_forecast`: `110`
- `tcl_recovery_planning`: `178`
- `tcl_recovery_confidence_estimation`: `98`
- `tcl_run_triage`: `98`
