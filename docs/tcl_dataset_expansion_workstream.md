# TCL Dataset Expansion Workstream

## Purpose

This workstream extends the TAR master dataset so the first 7B operator model
does not only learn generic TAR governance, but also learns the real TCL
runtime language:

- thermodynamic regime state
- trace dynamics
- anchor-governed recovery state

## Why It Exists

The first TAR operator SFT path is now working, but its dataset is still
TAR-first and only indirectly TCL-aware. Much of the most valuable TCL
knowledge still lives in runtime artifacts:

- `payload_summary.json`
- `thermo_metrics.jsonl`
- `recovery.json`

This workstream lifts those artifacts into supervised examples.

## First TCL-Native Task Families

### `tcl_regime_diagnosis`

Source:

- `tar_runs/**/payload_summary.json`
- sibling `config.json`

Teaches:

- reading `D_PR`
- reading equilibrium state
- understanding governor action / reasons
- distinguishing equilibrium, stabilizing, warming-up, and quenching regimes

### `tcl_trace_analysis`

Source:

- `tar_runs/**/thermo_metrics.jsonl`
- sibling `config.json`

Teaches:

- summarizing full thermodynamic traces instead of only last-step snapshots
- detecting whether `D_PR`, drift, and equilibrium trends are improving,
  flat, or collapsing

### `tcl_recovery_planning`

Source:

- `tar_state/recovery.json`

Teaches:

- when to resume
- when to debug
- when to pivot
- when anchor reuse is appropriate

## Acceptance Criteria

This workstream is complete when:

- TCL-native task families are emitted by the master dataset builder
- the examples come from real TAR/TCL artifacts
- the builder still sanitizes repo/state paths
- regression tests defend the new families
- the manifest reports the new TCL task-family counts

## Professional Boundary

This is not yet full 7B TCL backbone training.

It is the correct first step:

- teach the TAR operator model to understand TCL
- then expand the dataset
- then run stronger TAR/TCL operator training
- only later decide whether a full large-backbone TCL algorithm port is worth it
