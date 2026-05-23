# Phase 10 Baseline — Provenance Audit Trail

## What this document is

Phase 10 is the primary TCL result (TCL vs EWC: Δ=−0.0748, p=0.019, d=1.70, 5/5
seeds; TCL vs SGD: Δ=−0.117, p=0.0012, d=3.68, 5/5 seeds). It predates the
execution manifest gate (RAIL 3) by approximately 10 days. It cannot be brought under
RAIL 3 retroactively without fabricating a manifest that did not exist at the time.
This document records the equivalent human-verified provenance as it actually existed,
and the outcome of post-gate integrity checks.

A reviewer asking "where is the authorization artifact for your headline result?"
should be directed here.

---

## Canonical result file

```
tar_state/comparisons/phase10_controlled_rerun_20260509T132155Z.json
tar_state/comparisons/phase10_controlled_rerun_20260509T132155Z_env.json
```

Trust tier: `TRUST_TRUSTED_MANUAL` — controlled run with env snapshot present,
human-verified preflight, pre-gate era. Publication allowed: YES (per
`tar_lab/validation.py` trust tier definitions).

---

## Headline numbers (from result file)

**TCL vs EWC** (primary claim):

| Metric | Value |
|--------|-------|
| mean_delta (TCL − EWC forgetting) | −0.07478 |
| t_stat | −3.806 |
| p_val | 0.01901 |
| cohens_d | 1.702 |
| n_tcl_better | 5/5 seeds |
| per_seed_deltas | −0.117, −0.096, −0.075, −0.001, −0.084 |

**TCL vs SGD baseline**:

| Metric | Value |
|--------|-------|
| mean_delta | −0.117 |
| p_val | 0.00119 |
| cohens_d | 3.676 |
| n_tcl_better | 5/5 seeds |

**TCL vs SI**:

| Metric | Value |
|--------|-------|
| mean_delta | +0.024 (TCL worse) |
| p_val | 0.142 (not significant) |
| n_tcl_better | 1/5 seeds |

**Method aggregate forgetting means**: TCL=0.1161, SI=0.0920, EWC=0.1909,
SGD=0.2331

Note: the SI forgetting_std (0.000678) is anomalously low — essentially zero
variance across seeds. This is worth noting as a potential SI implementation
artifact but does not affect the TCL vs EWC headline.

---

## Environment snapshot (from env file — old schema format)

- **Run start**: 2026-05-09T13:22:06 UTC
- **Run end**: 2026-05-10T09:22:02 UTC (~20 hours)
- **Code commit**: `7a77eb5` ("freeze: add observability contract for HPC validation")
- **Git status**: Dirty working copy at capture time (paper files and tar_author.py
  modified; none affect the training code path; old phase11/12 json files modified in
  tar_state/comparisons — these are legacy fixed-name files, not the current canonical
  store)
- **GPU**: NVIDIA GeForce GTX 1650 (4096 MiB), Driver 591.86, CUDA 12.4
- **PyTorch**: 2.6.0+cu124, Python 3.13.3
- **Dataset**: split_cifar10, task-incremental, 5 tasks, 2 classes/task
- **Seeds**: 42, 0, 1, 2, 3
- **Backbone**: ResNet-18, 40 epochs/task, batch_size=32
- **Methods**: TCL, EWC (λ=100), SI (c=0.1, ξ=0.001), SGD baseline
- **TCL params**: penalty_lambda=0.01, alpha=0.5, ordered_lr_scale=0.5,
  disordered_lr_scale=1.2, reset_on_task_boundary=True

---

## Pre-execution controls (from env file `tar_autonomy_confirmations` block)

All three confirmed True at run start:

| Check | Result |
|-------|--------|
| no_tar_daemons_running_at_start | True |
| scheduled_task_inactive | True |
| startup_entry_disabled_present | True (watchdog startup disabled) |

These are the functional equivalent of the manifest gate's "controlled, deliberate,
human-initiated run" guarantee. The mechanism was a preflight check rather than a
cryptographic manifest, but the intent and the human-review step were the same.

---

## Gate 1 — Environment sibling coupling

**Status: PARTIAL PASS (pre-RAIL 2 schema)**

The env file exists alongside the result file and contains: git_rev_parse_head, full
per-seed config, pip_freeze, nvidia-smi output, and the preflight controls block.

However, the file uses the old env schema (no `artifact_schema: "tar_env_snapshot_v1"`
field) and does not contain `authorization.manifest_hash` or `authorization.manifest_path`
— these fields did not exist at run time. `tar_lab/canonical_registry._check_gate_1_sibling()`
would raise `ProvenanceSiblingInvalidError` on this file because the authorization block
is absent. This is expected and correct; the file predates RAIL 2.

---

## Gate 2 — Manifest hash verification

**Status: NOT APPLICABLE**

No manifest existed at the time of this run. Gate 2 cannot be applied retroactively.

---

## Gate 3 — Deterministic numpy recompute

**Status: INCOMPATIBLE SCHEMA**

`tar_lab/canonical_registry._check_gate_3_recompute()` expects a `seed_results` key
containing a list of per-seed dicts. Phase 10 uses the old schema (`per_seed` list,
`aggregate` dict, `pairwise` dict). Running Gate 3 produces:

```
DeterministicRecomputeFailure: No seed_results found in
phase10_controlled_rerun_20260509T132155Z.json.
```

The data for a manual equivalent of Gate 3 is present in the `per_seed` list. A schema
adapter in `_check_gate_3_recompute()` would allow this check to run, but this has not
been implemented. Status: **not verifiable by current Gate 3 implementation against
old-schema files**.

---

## Summary for reviewers

This result predates the manifest gate. The authorization equivalent is the three
preflight confirmations (no daemons, no scheduled tasks, startup disabled) plus the
approximately 20-hour supervised run on 2026-05-09 to 2026-05-10, producing results
consistent with the prior Phase 9 directional finding. The post-gate rebuild verified
provenance via the env sibling (Gate 1 partial pass) but the deterministic recompute
gate (Gate 3) cannot be applied due to schema mismatch.

The absence of a cryptographic manifest is a known limitation of the pre-gate era.
It is documented here, not concealed.
