# Phase 8 Roadmap — Making TCL Competitive

## Context

Phase 7 produced an honest null result on task-incremental Split-CIFAR-10:

| Method | mean_forgetting | final_mean_accuracy |
|--------|----------------|---------------------|
| EWC (λ=100) | 0.019 ± 0.006 | 0.873 |
| SGD baseline | 0.120 ± 0.031 | 0.802 |
| TCL | 0.128 ± 0.052 | 0.792 |
| SI | 0.155 ± 0.140 | 0.644 |

TCL sits alongside SGD — no regularization benefit over the baseline.
EWC wins clearly (p=0.009, d=2.945).

## Root Cause Diagnosis

The governor is mechanically working: hooks registered, step() firing every
batch, LR adjustment applied. But all 5 seeds end in "disordered" (rho > 1.1),
meaning the LR brake never fires. The governor is boosting LR by 1.2x for
most of training — counterproductive for forgetting prevention.

**The specific bug: sigma_star miscalibration.**

sigma_star = alpha × median(running_sigma_window), with alpha=1.0.
In continuous supervised training the gradient signal is always "hot".
sigma_star calibrates to the hot distribution. Every subsequent batch reads
rho ≈ 1.0–1.2. The "ordered" threshold (rho < 0.9) is never crossed.

The governor is measuring the thermodynamic state correctly — it has no
reference point that distinguishes "converged" from "actively learning"
because both look the same relative to a running median of the same data.

Secondary issue: sigma_star persists across tasks. When task 2 starts and
gradients spike again, the governor compares against a sigma_star calibrated
on converged task 1. The spike reads as "extremely disordered" and LR gets
boosted exactly when you want consolidation.

## Locked-In Execution Order

### Phase 8A — Calibration Fix (no pod, ~2 hours)

**Hypothesis:** sigma_star miscalibration is the primary reason the ordered
phase is never reached. Fixing calibration will let the brake fire.

**Changes:**

1. **Lower alpha** — try {0.3, 0.5, 0.7} in ActivationThermoObserver.__init__
   - Sets sigma_star = alpha × median(window), making "ordered" reachable
     when training converges even slightly below the median activity level
   - alpha=0.5 means ordered fires when sigma drops to 50% of median activity

2. **Per-task sigma_star anchor** — reset sigma_window at each task boundary
   - Currently sigma_window persists: task 2's gradient spike compares against
     task 1's calibration
   - Fix: call a reset_task_calibration() at the start of each task
   - sigma_star re-calibrates on the new task's own dynamics

3. **Diagnostic trace** — record step-by-step regime histogram, not just
   final_regime
   - Need: % of steps in each regime per task, LR min/max/mean per task,
     first batch that hit "ordered", rho trajectory summary
   - Without this data the epoch sweep is blind

**Acceptance:** at least 1 seed shows "ordered" regime during late-task
training. rho drops below 0.9 on at least one task boundary for at least 2
seeds.

**8A validation results (2026-04-18, commit 7eda5a0):**

| Check | Result | Detail |
|-------|--------|--------|
| Regime transitions at task boundaries | 0/5 FAIL | Dominant regime = disordered throughout all tasks/epochs |
| Anchor live (dimensionality_ratio > 0) | 5/5 PASS | anchor_d = 4.8–6.2 across seeds |
| "ordered" phase reached | 5/5 PASS | Reached in 0–1% of batches (tasks 3–4 only) |

Benchmark: TCL forgetting=0.1247±0.029, SGD=0.1184±0.042. Delta +0.006 (TCL marginallyy worse).

Mean LR across all tasks ≈ 0.01197 — essentially `base_lr × 1.2` flat.
Governor fires "disordered" 98–99% of batches → constant LR boost, rare braking.

**Interpretation:** Implementation is now correctly wired. The ordered phase IS
reachable with alpha=0.5 but only fires for ~1% of batches. The network is
genuinely in a high-entropy exploratory state for the entire 5-epoch budget.
The 1% braking doesn't overcome 99% boosting. This is not a calibration bug —
it is a measurement confirming the volume hypothesis: the thermodynamic
transition from disordered → ordered requires more training time per task
than the current 5-epoch budget allows.

Proceed to 8B (epoch sweep).

---

### Phase 8B — Epoch Sweep (laptop or pod, ~4 hours)

**Gating condition:** run after 8A regardless of outcome.
- If 8A succeeded: run sweep with fixed calibration to measure improvement slope
- If 8A failed: run sweep with original calibration to isolate volume effect

**Hypothesis:** governor needs training volume to transition through
disordered → critical → ordered. At 5 epochs the transition never completes.

**Setting:** identical to Phase 7 (same architecture, same seeds, same 4
methods). Only change: train_epochs_per_task swept over {5, 10, 20}.

**What success looks like:**
- TCL forgetting drops disproportionately relative to SGD as epochs increase
- Specifically: if SGD forgetting drops from 0.120 → 0.100 at 20 epochs
  but TCL drops from 0.128 → 0.060, that implicates volume as the key factor
- EWC and SI should be roughly flat (regularization methods are not epoch-sensitive)

**What failure looks like:**
- All four methods improve proportionally with epochs
- TCL stays roughly aligned with SGD at all epoch budgets
- → Architecture-level issue: LR modulation alone is insufficient;
  need a weight-penalty term derived from the thermodynamic signal

**Acceptance:** sweep complete for all 3 epoch budgets × 4 methods × 5 seeds.
TCL vs SGD delta at each epoch level reported with p-values. Trend conclusion
written before checking whether it favours TCL.

---

### Phase 8C — Weight Penalty Term (if 8A + 8B both fail)

**Gating condition:** only if 8B shows no differential improvement for TCL
relative to SGD across all epoch budgets.

**Hypothesis:** LR modulation is a weak forgetting signal. EWC's advantage
is that the Fisher diagonal directly identifies which weights matter. TCL needs
an analogous term derived from the thermodynamic signal.

**Candidate mechanism:** thermodynamic importance = effective_dimensionality
per layer (from participation ratio). High-dimensionality layers are doing
distributed computation and should be protected more. Add a
dimensionality-weighted L2 penalty on weight change between tasks, similar
to EWC but with thermodynamic importance scores instead of Fisher diagonal.

This is a larger change: new term in the loss function, not just LR scheduling.
Requires design work before implementation.

---

### Phase 8D — Class-Incremental (after 8A/8B show positive result)

**Gating condition:** only after a version of TCL shows measurable improvement
over SGD in the task-incremental setting (8A or 8B must show signal first).

**Setting:** class-incremental Split-CIFAR-10 (task identity not provided at
test time), same architecture.

**Why gated:** class-incremental is the current hard benchmark standard and
the setting where forgetting matters most. But running it before we have a
working TCL calibration produces an uninformative result.

---

## Execution Summary

```
Phase 8A: Fix calibration (alpha sweep + per-task reset + diagnostic trace)
  → if ordered phase reached: re-run Phase 7 benchmark with fixed calibration
  → always proceed to:

Phase 8B: Epoch sweep {5, 10, 20} × 4 methods × 5 seeds
  → if TCL shows differential improvement: document mechanism, proceed to 8D
  → if flat: proceed to 8C

Phase 8C: Weight penalty term from dimensionality signal (if 8A+8B fail)

Phase 8D: Class-incremental Split-CIFAR-10 (after positive TCL signal)
```

## What a Good Phase 8 Result Looks Like

The bar is not "TCL beats EWC." The bar is:

1. **8A/8B produce a working TCL calibration** where the governor actually
   transitions through regimes during training (not stuck in "disordered")

2. **TCL measurably outperforms SGD** on forgetting (p < 0.05, d > 0.5)
   with the fixed calibration — proving thermodynamic regime-awareness adds
   signal beyond vanilla gradient descent

3. **The mechanism is interpretable** — diagnostic traces show regime changes
   at task boundaries, LR variance correlated with forgetting reduction

If those three are true, TCL is a real scientific contribution regardless of
whether it matches EWC exactly. An interpretable, computationally cheaper
alternative that works is publishable. A non-working governor is not.
