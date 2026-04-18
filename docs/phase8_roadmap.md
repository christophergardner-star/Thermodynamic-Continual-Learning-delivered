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

### Phase 8B — Epoch Scouting Pass (single seed, complete 2026-04-18)

**Scouting design:** single seed (42), epochs_per_task ∈ {5, 15, 40},
TCL + SGD only. Purpose: distinguish volume hypothesis from calibration
hypothesis before committing to a full multi-seed sweep.

**Results (5-epoch and 15-epoch confirmed; 40-epoch running):**

| epochs | TCL forgetting | SGD forgetting | delta  | ordered% |
|--------|---------------|----------------|--------|----------|
| 5      | 0.1921        | 0.1534         | +0.039 | ~1%      |
| 15     | 0.1081        | 0.1046         | +0.004 | ~1%      |
| 40     | (running)     |                |        |          |

**15-epoch rho trajectories (task 0 and late tasks):**

Task 0: `[2.84, 2.92, 2.64, 2.71, 2.60, 2.70, 2.76, 2.59, 2.53, 3.11, 2.72, 3.43, 2.71, 3.27, 5.86]`
Task 3: `[2.74, 2.87, 2.91, 3.21, 3.07, 3.30, 4.03, 5.92, 4.04, 6.97, 5.96, 5.98, 11.40, 13.75, 38.31]`
Task 4: `[2.84, 2.56, 2.80, 2.67, 2.77, 3.19, 3.64, 3.09, 4.55, 4.39, 5.08, 4.74, 10.64, 6.30, 32.77]`

Ordered threshold: rho < 0.9. The ordered regime never fires during steady
within-task training. Late tasks show rho monotonically increasing — the
governor is moving away from ordered, not toward it.

sigma/sigma_star EMA at task 3 epoch 14: 0.001015 / 0.000144 = 7.06.

**Verdict: CALIBRATION — volume hypothesis rejected.**

**Root cause (derived from 8B traces):**

sigma_star = alpha × median(sigma_window) where sigma_window is populated with
recent sigma values. As sigma decreases during training, sigma_window contracts
at the same rate, so median(sigma_window) ≈ sigma. Therefore:

    rho = sigma / (alpha × sigma) = 1/alpha

The ratio is asymptotically pinned to 1/alpha regardless of training duration
or epoch budget. With alpha=0.5 the theoretical floor is rho ≈ 2.0. The
ordered threshold at rho < 0.9 is mathematically unreachable from stationary
training. The 1% ordered batches seen are noise from within-batch variance,
not thermodynamic convergence.

**Critical distinction:** the rolling-median formulation measures whether sigma
is currently low *relative to recent sigma* — a stationarity detector. What the
governor needs to detect is whether sigma has fallen significantly below where it
was at the start of the task — a convergence detector. Different physical
quantity, different scientific meaning. Lowering alpha makes this worse (raises
1/alpha floor). More epochs cannot fix a definitional error.

---

### Phase 8C — sigma_star Redefinition as Fixed Per-Task Anchor

**Gating condition:** gates on 8B confirming calibration failure (confirmed above).

**Diagnosis:** The current formulation specifies sigma_star as the rolling median
of the running sigma window. This makes sigma_star co-contract with sigma during
training, pinning rho ≈ 1/alpha and making the ordered threshold unreachable
under any steady training regime. Phase 8A correctly implemented the specified
formulation. Phase 8B revealed the specified formulation is thermodynamically
wrong.

**The fix:** sigma_star must be a fixed per-task reference set from the network's
gradient magnitude during the early phase of each task, then frozen for the
remainder of that task. "Ordered" then means "sigma has fallen significantly
below where it was when training on this task began" — thermodynamically, the
network has cooled to a lower-entropy state than its initial thermal level on
this data. This is the physically correct definition of convergence.

Analogy: sigma_star is the reference temperature — the initial thermal energy
of the system. The governor fires when the system has genuinely cooled, not
merely when it is briefly calm relative to recent fluctuations.

**Implementation changes in `ActivationThermoObserver`:**

1. **Two separate state variables per group (not one):**
   - `sigma_window` — rolling window, used only to smooth current sigma estimate
   - `sigma_star_anchor` — scalar, set once per task, frozen until next task

2. **Anchor-setting logic in `step()`:**
   - After `reset_for_new_task()`, maintain a separate `_anchor_window` list
   - Accumulate first N=20 batch sigma values (or up to first full epoch)
   - After N batches: set `sigma_star_anchor = alpha × median(_anchor_window)`, mark anchor as set
   - Use `sigma_star_anchor` (not rolling median) for all rho computations until next reset

3. **Alpha semantics (now defensible):**
   - With fixed anchor: alpha=0.5 means "ordered when sigma has dropped to 50%
     of early-task gradient magnitude." Interpretable convergence criterion.
   - Sweep alpha ∈ {0.3, 0.5, 0.7} after anchor fix to find natural separation.
   - Any value in {0.3–0.7} now has a thermodynamically meaningful interpretation.

4. **`reset_for_new_task()` additions:**
   - Clear `sigma_star_anchor = None`
   - Clear `_anchor_window = []`
   - Set `_anchor_set = False`

5. **Diagnostic trace additions:**
   - Log `sigma_star_anchor` value per task (confirm it's being set and frozen)
   - Log `anchor_set_at_batch` (batch number when anchor locked in)
   - Log rho trajectory relative to the new fixed anchor

**Acceptance:** rho trajectories show systematic downward trend within tasks
as training progresses. Task 0 final rho meaningfully lower than task 0 initial
rho. ordered% present in convergence window (final half of task epochs) for at
least 1 task across 2+ seeds.

**Estimated scope:** ~50 lines in thermoobserver.py, minor additions to
multimodal_payloads.py trace logging. No architecture changes.

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
Phase 8A: Fix integration bugs (alpha=0.5, per-task reset, correct step() order,
           anchor_snapshot, richer trace) — COMPLETE 2026-04-18
  → Result: governor correctly implements specified formulation
  → ordered reached ~1% of batches (volume insufficient, not a bug)

Phase 8B: Scouting pass {5, 15, 40} epochs × 1 seed — COMPLETE 2026-04-18
  → Result: CALIBRATION — sigma_star co-tracks sigma, rho pinned to 1/alpha
  → Rolling median is a stationarity detector, not a convergence detector
  → Volume hypothesis rejected; formulation itself is wrong

Phase 8C: Redefine sigma_star as fixed per-task anchor (N=20 batch median)
  → Target: rho drops within-task, ordered fires in convergence window
  → Once working: re-run Phase 7 benchmark (TCL + SGD, 5 seeds, 15–20 epochs)

Phase 8D: Class-incremental Split-CIFAR-10 (after Phase 8C shows TCL > SGD)
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
