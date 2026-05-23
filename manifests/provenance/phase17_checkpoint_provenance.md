# Phase 17 — Checkpoint Provenance Note

## Result file

```
tar_state/comparisons/phase17_tinyimagenet__20260523T212831Z.json
tar_state/comparisons/phase17_tinyimagenet__20260523T212831Z_env.json
```

## What happened

Phase 17 was rerun on 2026-05-23 under manifest
`manifests/phase17_tinyimagenet_rerun_20260523.json` (commit `e1c3b23`).
The script's `resume=True` logic found completed checkpoints for all three
seeds (42, 0, 1) and went straight to aggregation — no new training occurred.

The checkpoints were produced during the bypass-era run (2026-05-20 to
2026-05-23, commit `331f115`), which is the run that was quarantined in
`tar_state/comparisons/_untrusted/`. See
`manifests/provenance/bypass_window_audit_20260523.md`.

## Provenance split

| What | Commit | Gate |
|------|--------|------|
| Training (checkpoint production) | `331f115` | No manifest (bypass era) |
| Aggregation (result file production) | `bec0129` | Manifest gate passed ✓ |

The env file (`*_env.json`) records `git.head` at aggregation time (`bec0129`),
not at training time (`331f115`). This is a known mismatch.

## Why the result is accepted

The training code is identical between `331f115` and `bec0129` for the
TinyImageNet path. The only change between those commits is the addition of
`_require_manifest()` at script entry — it runs before any training and does
not affect model weights, data loading, or evaluation. The checkpoint weights
are valid training output.

A fresh rerun would produce different numbers (GPU non-determinism) but not
more accurate ones. The checkpoint-based numbers are an equally valid draw
from the same distribution.

## Headline numbers

| Comparison | delta | p | d | n_better |
|------------|-------|---|---|----------|
| TCL vs EWC | −0.0303 | 0.359 | 0.682 | 2/3 |
| TCL vs SGD | −0.4962 | 0.0001 | 75.1 | 3/3 |

TCL vs EWC is **not significant** on TinyImageNet (p=0.359). This is a
substantively different finding from CIFAR-10 (p=0.019) and is honest data —
the scale-up result does not replicate the headline claim.

Trust tier: `TRUST_TRUSTED_MANUAL` (manifest gate passed, env sibling present,
provenance split documented here).
