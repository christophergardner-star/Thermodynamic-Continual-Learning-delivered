# TAR Master Dataset Spec

## Purpose

This is the first serious TAR/TCL master corpus for training a TAR-native
operator model. It is built from TAR's own state artifacts so the model learns:

- project continuity
- benchmark honesty
- reproducibility refusal
- execution diagnosis
- verification judgement
- falsification planning
- prioritization and portfolio governance

## Output Files

The builder writes:

- `tar_master_dataset.jsonl`
- `tar_master_dataset_train.jsonl`
- `tar_master_dataset_validation.jsonl`
- `tar_master_dataset_test.jsonl`
- `manifest.json`

Each record contains:

- stable `example_id`
- `task_family`
- `task_name`
- sanitized `input_context`
- structured `target`
- chat-style `messages`
- deterministic split assignment
- provenance back to the TAR state artifact

## Source Artifacts

The builder currently consumes:

- `problem_studies.jsonl`
- `problem_executions.jsonl`
- `verification_reports.jsonl`
- `research_decisions.jsonl`
- `research_projects.json`
- `falsification_plans.jsonl`
- `project_priority_records.jsonl`
- `portfolio_decisions.jsonl`
- `recovery.json`
- `tar_runs/**/payload_summary.json`
- `tar_runs/**/thermo_metrics.jsonl`
- sibling `tar_runs/**/config.json` when present

Missing files are allowed. The builder emits examples only from artifacts that
exist.

## Task Families

Current task families:

- `problem_scoping`
- `benchmark_honesty`
- `execution_diagnosis`
- `verification_judgement`
- `decision_rationale`
- `project_resume`
- `falsification_planning`
- `prioritization`
- `portfolio_governance`
- `tcl_regime_diagnosis`
- `tcl_trace_analysis`
- `tcl_recovery_planning`

## Data Hygiene

The builder sanitizes machine-specific roots into placeholders:

- `<REPO_ROOT>`
- `<STATE_DIR>`
- `<WORKSPACE_ROOT>`

That avoids training the model on local absolute paths.

## Build Commands

From the repo root:

```bash
python build_tar_master_dataset.py --state-dir tar_state --output-dir dataset_artifacts/tar_master_dataset_v1
```

To merge multiple TAR state roots into one master corpus:

```bash
python build_tar_master_dataset.py --state-dir tar_state --state-dir ../old_pod_memory_transfer/tar_state --output-dir dataset_artifacts/tar_master_dataset_merged_v1
```

On the pod:

```bash
cd /workspace/Thermodynamic-Continual-Learning-delivered
source .venv/bin/activate
python build_tar_master_dataset.py --state-dir tar_state --output-dir dataset_artifacts/tar_master_dataset_v1
```

## Intended Use

This corpus is the right first supervised dataset for:

- TAR operator SFT
- TAR/TCL research-control adaptation
- benchmark/refusal honesty tuning
- falsification-aware process tuning

It is not yet a full preference dataset or RLHF dataset.

## Professional Boundary

This dataset is bespoke and serious, but it is still version one. Its quality
depends directly on TAR's state quality. The right practice is:

1. keep improving TAR state quality
2. rebuild this corpus regularly
3. maintain held-out evaluations
4. later layer preference/process supervision on top

## TCL-Native Expansion

The first TCL-native dataset expansion now lifts real thermodynamic runtime
artifacts into supervised examples:

- `tcl_regime_diagnosis`
  - from `payload_summary.json`
  - teaches the model to read `D_PR`, equilibrium state, governor action, and
    fallback-data posture
- `tcl_trace_analysis`
  - from `thermo_metrics.jsonl`
  - teaches the model to summarize thermodynamic trace trends instead of only
    final snapshots
- `tcl_recovery_planning`
  - from `recovery.json`
  - teaches the model to interpret pause/resume/pivot state for governed TCL
    runs

These families are grounded in actual TAR/TCL artifacts rather than synthetic
free-form descriptions.
