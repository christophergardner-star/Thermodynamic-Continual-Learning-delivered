# WS23 Execution Spec

`WS23` is **TAR Master Dataset Scale-Up**.

Its purpose is to turn TAR's current seed corpus into a serious, versioned,
private supervised dataset release that is strong enough to justify `WS24`
evaluation work and then `WS25` serious 7B TAR operator training.

## Core purpose

After WS23, TAR should have a dataset release that is:

- large enough to matter
- diverse enough to avoid one-mode overfitting
- rich in refusal, contradiction, and negative cases
- TCL-aware, not only governance-aware
- split safely for future evaluation
- versioned, reproducible, and inspectable

## Why WS23 matters

Without WS23:

- TAR's control stack is ahead of the training corpus
- `WS25` would be technically possible but scientifically weak
- split leakage and weak release metadata would make evaluation less honest
- the model would not learn enough refusal, evidence-debt, or TCL-runtime language

WS23 closes that gap by treating the dataset as a first-class research artifact.

## Strategic principle

WS23 is not "export some JSONL and train."

It is a release-engineering workstream for a bespoke TAR/TCL operator corpus.
The right standard is:

- state-grounded
- versioned
- lineage-safe
- security-aware
- test-defended

## What WS23 must deliver

1. Versioned dataset release mechanics

- deterministic dataset builds
- release manifests
- file hashes
- split counts
- family counts
- deduplication metadata
- source artifact fingerprints

2. Stronger split integrity

- split by project/thread/trial lineage where possible
- prevent train/test leakage across related inquiry families
- surface split provenance explicitly in the manifest

3. Expanded task-family coverage

By the end of WS23, the release must strongly cover:

- `project_resume`
- `prioritization`
- `falsification_planning`
- `portfolio_governance`
- `benchmark_honesty`
- `execution_diagnosis`
- `verification_judgement`
- `decision_rationale`
- `tcl_regime_diagnosis`
- `tcl_trace_analysis`
- `tcl_recovery_planning`

And it must add the following high-value families:

- `reproducibility_refusal`
- `sandbox_policy_reasoning`
- `endpoint_observability_diagnosis`
- `portfolio_staleness_recovery`
- `claim_lineage_audit`
- `evidence_debt_judgement`

4. Controlled state-generation workflow

WS23 must include a deliberate workflow for producing more high-value TAR state:

- continuity campaign
- benchmark honesty campaign
- falsification campaign
- portfolio campaign
- TCL campaign
- runtime and observability campaign

This workflow must be deterministic enough to rebuild and extend later.

5. Dataset QA gates

The dataset builder and tests must enforce:

- no raw absolute local paths in emitted examples
- no duplicate examples by stable dedupe key
- no split leakage across lineage keys
- deterministic train/validation/test assignment
- manifest file hashes and source artifact fingerprints

## File-by-file scope

### `build_tar_master_dataset.py`

- harden split assignment around lineage keys
- add stronger manifest output
- add source artifact fingerprinting
- add dedupe accounting
- add the new WS23 task families
- read more state sources:
  - `claim_verdicts.jsonl`
  - `evidence_debt_records.jsonl`
  - `project_staleness_records.jsonl`
  - `inference_endpoints.json`
  - `tar_state/manifests/*.json`

### `docs/tar_master_dataset_spec.md`

- document the new task families
- document lineage-safe split semantics
- document release manifest structure
- document the new controlled state-generation workflow

### `tests/test_tar_master_dataset.py`

- add split-integrity tests
- add manifest QA tests
- add regression tests for the new families
- keep sanitization guarantees explicit

### Dataset campaign script

Add a dedicated script to generate controlled high-value TAR state for WS23.

That script should create deterministic state for:

- continuity
- benchmark honesty
- falsification
- portfolio staleness and resume
- TCL runtime traces
- endpoint observability incidents

### Release documentation

Add a dataset release note for the first WS23 release with:

- version
- build inputs
- counts by family
- counts by split
- any remaining quality gaps

## Design decisions

1. Split integrity must be lineage-aware

Random row split is not acceptable for TAR's structured state.

2. Refusal and contradiction examples are first-class data

They are part of TAR's identity, not edge cases.

3. State-grounded examples outrank freeform synthetic chat

Synthetic examples are allowed only when schema-grounded and policy-correct.

4. Release manifests must be audit-friendly

The dataset should be inspectable like a software release.

5. WS23 should stay private by default

This is a core bespoke dataset. The secure local/default stance remains correct.

## Target scale

- minimum useful milestone: `1k-2k` examples
- WS23 close target: `5k+` curated examples
- stronger target if time allows: `10k+`

WS23 should not be considered fully closed below `5k` unless quality is
exceptionally strong and the shortfall is explicit.

## Acceptance criteria

WS23 closes only if all are true:

- TAR has a versioned, private dataset release suitable for serious operator training
- task-family coverage is materially broader than the current seed corpus
- refusal, contradiction, debt, and TCL-runtime cases are first-class
- split integrity is lineage-safe
- manifests include hashes and source artifact fingerprints
- regression tests defend the release mechanics

## Recommended implementation order

1. formalize the release targets and quotas
2. harden the builder and split logic
3. add the missing WS23 task families
4. add the controlled state-generation workflow
5. rebuild a WS23 dataset release
6. add QA tests and release documentation
7. validate that the release is strong enough to justify `WS24`

## What counts as done

WS23 is done when TAR can truthfully say:

- "this is a private, versioned dataset release"
- "its splits are lineage-safe"
- "it includes refusal, debt, contradiction, and TCL-runtime cases"
- "its release manifest is strong enough to trust"
- "it is now worth building a real evaluation harness on top of it"
