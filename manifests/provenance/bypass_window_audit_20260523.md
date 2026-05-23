# Bypass Window Audit — 2026-05-23

## Window

- **Opened**: 2026-05-19T14:38:55 UTC — commit `c72e293`
  ("Exempt autonomous mode from orchestrator manifest gate (RAIL 3)")
- **Closed**: Not yet closed at time of writing (this audit precedes the close)
- **Window duration at audit**: ~4 days

## Methodology

Determine which canonical results in `tar_state/comparisons/` were produced during
the bypass window.

**Primary indicator**: presence of `authorization.manifest_path` and
`authorization.manifest_hash` in each `*_env.json` file. Null values indicate no
manifest was loaded at execution time.

**Secondary indicator**: `git.head` in env file compared against git commit ordering
relative to `c72e293`. Commits at or after `c72e293` are in the bypass window.

**Timestamp limitation**: `created_at` is absent from env files using the old schema
format (pre-`artifact_schema: tar_env_snapshot_v1`). For old-format files, pre/post
window determination uses git.head ordering.

**Scan scope**: All `*.json` files in `tar_state/comparisons/` excluding `*_env.json`
and `canonical_results_index.jsonl`. Subdirectory `_untrusted/` excluded (already
quarantined).

## Scan Results

| File | authorization.manifest_path | authorization.manifest_hash | run_started_at | In bypass window |
|------|-----------------------------|-----------------------------|----------------|-----------------|
| phase10_controlled_rerun_20260509T132155Z.json | absent (old schema) | absent (old schema) | 2026-05-09T13:22:06 | NO — pre-gate era |
| phase11_ablation__20260511T113318Z.json | manifests/phase11_rerun_20260511.json | ceaddd85... | pre-bypass | NO |
| phase12_ewc_sweep__20260511T203414Z.json | manifests/phase12_rerun_20260511.json | 3bf26fdc... | pre-bypass | NO |
| phase13_si_sweep__20260512T061410Z.json | manifests/phase13_rerun_20260511.json | 6c702aae... | pre-bypass | NO |
| phase17_tinyimagenet__20260523T144752Z.json | null | null | 2026-05-20T15:43:50 | **YES** |

## Finding: One bypass-era result

**phase17_tinyimagenet__20260523T144752Z.json — Phase 17, TinyImageNet scale-up**

From `phase17_tinyimagenet__20260523T144752Z_env.json`:

```
artifact_schema:  tar_env_snapshot_v1
run_started_at:   2026-05-20T15:43:50.722020+00:00
run_ended_at:     2026-05-23T14:47:49.189342+00:00   (3-day run)
trigger:          manual_script
source_script:    phase17_tinyimagenet.py
git.head:         2b8ab5bc94cb76f1754e5054001e5b1716f04736  (post-bypass commit)
git.status_clean: false
authorization.manifest_path:  null
authorization.manifest_hash:  null
```

The `authorization` block is null, confirming no manifest was loaded or verified
before execution.

**Additional finding on phase17_tinyimagenet.py**: The committed version of the file
at run time (sole commit: `331f115`, "freeze: enable TAR stabilisation mode for HPC
validation") contains no manifest gate. The current working copy on disk has
`_require_manifest()` added at line 690 but this change was never committed. The
working-copy addition post-dates the run. The version that executed on 2026-05-20
had no RAIL 3 enforcement.

## Null finding

No other canonical result files exist in `tar_state/comparisons/` outside the five
listed above. The bypass window produced exactly one unmanifested canonical result.

## Recommended actions (proposed in session 2026-05-23)

1. **Quarantine Phase 17**: move result + env sibling to `_untrusted/`, quarantine
   reason `bypass_era_unmanifested`. Estimated rerun cost: ~3 days GTX 1650 compute
   (3 seeds, TinyImageNet, ResNet-18, 40 epochs/task).

2. **Commit `_require_manifest()` addition** in `phase17_tinyimagenet.py` working copy.

3. **Close bypass**: remove `_autonomous_ok` logic from
   `tar_experiment_orchestrator.py:1186-1208` so `ManifestGateError` is raised
   unconditionally when `self._active_manifest is None`.

4. **Integration test gap** (separate action): no test in the suite exercises the
   full manual execution path (phase script → `load_and_verify_manifest` →
   `ManifestGateError` on invalid manifest → execute with valid manifest). Add before
   bridge work begins.

---

*Audit conducted by Claude Sonnet 4.6 on 2026-05-23. Methodology and raw data above
are sufficient for independent verification.*
