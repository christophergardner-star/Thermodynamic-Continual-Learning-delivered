# WS26 Execution Spec

`WS26` is `TCL-Native Operator Deepening`.

## Purpose

`WS25` proved that the 7B TAR operator path works. `WS26` takes the next
proper step: make the operator materially stronger in its home domain,
Thermodynamic Continual Learning, without giving back the honesty, refusal,
governance, and structured-output gains already earned.

## Core Question

Can TAR's operator model become better at TCL-specific diagnosis,
intervention, and recovery judgement while preserving benchmark honesty,
reproducibility honesty, and portfolio discipline?

## Hypotheses

- `H1`: deeper TCL-native supervision improves failure-mode classification,
  regime judgement, and intervention choice over the `WS25` adapter baseline.
- `H2`: richer recovery-state supervision improves resume confidence and run
  triage.
- `H3`: tighter JSON-only prompting reduces parse errors on structured TCL
  tasks.
- `H4`: a deeper TCL eval pack reveals improvements that the original `WS24`
  TCL suite could not measure cleanly.

## Deliverables

- expanded TCL task families in the master dataset builder
- a deterministic TCL campaign generator for high-value failure and recovery
  state
- a private `WS26` dataset release suitable for retraining
- a merged `WS26` training release that preserves the broader `WS23` base while
  adding deeper TCL coverage
- an extended eval pack with deeper TCL coverage
- `WS26` train and eval configs for the next retrain cycle
- regression tests covering the new data and scoring contracts

## New TCL Families

- `tcl_failure_mode_classification`
- `tcl_anchor_policy_judgement`
- `tcl_intervention_selection`
- `tcl_trace_anomaly_diagnosis`
- `tcl_regime_transition_forecast`
- `tcl_recovery_confidence_estimation`
- `tcl_run_triage`

## Data Sources

- `tar_runs/**/payload_summary.json`
- `tar_runs/**/thermo_metrics.jsonl`
- `tar_runs/**/config.json`
- `tar_state/recovery.json`
- `tar_state/recovery_history.jsonl`

## Implementation Order

1. expand TCL family generation in `build_tar_master_dataset.py`
2. generate controlled WS26 TCL campaign state
3. extend evaluation scorers and harness support
4. add regression tests for new TCL families
5. freeze a `WS26` dataset release
6. merge `WS23` and `WS26` releases into the actual retraining target
7. freeze a `WS26` eval release
8. prepare the `WS26` train/eval configs
9. only then start a pod for the next retraining cycle

## Quality Gates

- new TCL families emit deterministic structured targets
- lineage-safe splitting remains intact
- no absolute local paths leak into dataset output
- eval scoring covers the new TCL families
- honesty and governance families remain part of the non-regression pack
- local tests pass before any pod-backed run is started

## Pod Policy

Do not start a pod for `WS26` until all of the following are true:

- the `WS26` dataset release is frozen
- the `WS26` eval pack is frozen
- the `WS26` train config is committed
- the `WS26` eval runtime config is committed
- the relevant local tests are green

Once those are true, the next justified pod action is a single serious `WS26`
retraining/eval cycle, not exploratory data engineering.

## Acceptance Criteria

`WS26` is ready for pod-backed retraining only if:

- the new TCL task families are present in the private dataset release
- the eval harness scores the new TCL families correctly
- local regression tests pass
- the dataset and eval artifacts are versioned and documented

`WS26` itself closes only after a retrained model beats the `WS25` baseline on
the deeper TCL evals without material regression on honesty, refusal,
governance, or parse reliability.
