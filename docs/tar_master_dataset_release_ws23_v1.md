# TAR Master Dataset Release WS23 v1

## Release

- dataset version: `tar-master-ws23-v1`
- output: `dataset_artifacts/tar_master_dataset_ws23_v1`
- merged state roots:
  - `tar_state`
  - `ws23_campaign_state_v1/tar_state`

## Scale

- total records: `7292`
- train: `5833`
- validation: `832`
- test: `627`

This clears the WS23 close target of `5k+` curated examples.

## Task-Family Counts

- `problem_scoping`: `487`
- `benchmark_honesty`: `485`
- `execution_diagnosis`: `481`
- `verification_judgement`: `482`
- `decision_rationale`: `485`
- `project_resume`: `480`
- `falsification_planning`: `480`
- `prioritization`: `480`
- `portfolio_governance`: `480`
- `reproducibility_refusal`: `112`
- `sandbox_policy_reasoning`: `968`
- `endpoint_observability_diagnosis`: `161`
- `portfolio_staleness_recovery`: `480`
- `claim_lineage_audit`: `480`
- `evidence_debt_judgement`: `480`
- `tcl_regime_diagnosis`: `96`
- `tcl_trace_analysis`: `94`
- `tcl_recovery_planning`: `81`

## Release Notes

- This release is materially broader than the earlier seed corpus.
- The split is lineage-safe and deterministic.
- The release manifest includes output file hashes and source artifact fingerprints.
- The new controlled campaign state was used to strengthen:
  - verification
  - claim lineage
  - evidence debt
  - staleness recovery
  - endpoint observability
  - TCL recovery

## Remaining Cautions

- `sandbox_policy_reasoning` is intentionally overrepresented because both execution
  reports and run manifests contribute it. That is acceptable for now because
  sandbox and reproducibility honesty are core TAR behaviors, but it should be
  monitored in `WS24`.
- TCL families are now present at useful scale, but they are still smaller than
  the governance families. `WS26` should deepen them further.
- This release remains private and should not be pushed casually because it is a
  core bespoke training artifact.
