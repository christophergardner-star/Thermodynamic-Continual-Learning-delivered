# TAR Master Dataset Release WS26 v1

## Release

- dataset dir: `dataset_artifacts/tar_master_dataset_ws26_v1`
- manifest: `dataset_artifacts/tar_master_dataset_ws26_v1/manifest.json`
- dataset version: `tar-master-ws26-v1`

## Build Inputs

- local `tar_state`
- controlled WS26 TCL campaign state:
  - `dataset_artifacts/ws26_tcl_campaign_state_v1/tar_state`

## Summary

- total records: `1105`
- duplicates removed: `1`
- lineages: `128`
- lineage safe: `true`

## Splits

- train: `865`
- validation: `118`
- test: `122`

## TCL Deepening Coverage

- `tcl_regime_diagnosis`: `112`
- `tcl_failure_mode_classification`: `112`
- `tcl_anchor_policy_judgement`: `112`
- `tcl_intervention_selection`: `112`
- `tcl_trace_analysis`: `110`
- `tcl_trace_anomaly_diagnosis`: `110`
- `tcl_regime_transition_forecast`: `110`
- `tcl_recovery_planning`: `98`
- `tcl_recovery_confidence_estimation`: `98`
- `tcl_run_triage`: `98`

## Notes

- This release is intentionally TCL-heavy. It is for WS26 domain deepening, not
  for replacing the broader WS23 dataset history.
- The actual WS26 retraining target is the merged release:
  - `dataset_artifacts/tar_master_dataset_ws26_merged_v1`
- The release remains private and is intended for local or controlled pod use.
