# TAR Operator Eval Release `WS24 v1`

This document records the first frozen WS24 evaluation release.

## Eval release

- eval version: `tar-operator-eval-ws24-v1`
- source dataset: `tar-master-ws23-v1`
- source held-out split: `627` items
- lineage count: `196`

Task-family counts:

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
- `tcl_recovery_planning`: `7`
- `tcl_regime_diagnosis`: `8`
- `tcl_trace_analysis`: `8`
- `verification_judgement`: `45`

## Validation runs

`gold` predictor:

- items: `627`
- mean score: `1.0`
- decision accuracy: `1.0`
- parse error rate: `0.0`
- overclaim rate: `0.0`
- false refusal rate: `0.0`

`heuristic` predictor:

- items: `627`
- mean score: `0.34704944178628105`
- decision accuracy: `0.20574162679425836`
- parse error rate: `0.0`
- overclaim rate: `0.009569377990430622`
- false refusal rate: `0.0`

The heuristic predictor is a local harness-validation utility, not the
scientific baseline for `WS25`.

## Remaining note

The prompt-only `Qwen/Qwen2.5-7B-Instruct` baseline path is implemented in the
runtime but was not executed in this release note. That belongs to the real
model-comparison phase leading into `WS25`.
