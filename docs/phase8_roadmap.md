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
The 1% braking doesn't overcome 99% boosting.

*Note — this interpretation stated "volume hypothesis confirmed" but Phase 8B
subsequently showed this was wrong.  The 1% ordered rate was calibration-limited,
not volume-limited.  The rolling-median sigma_star makes the ordered threshold
mathematically unreachable under any stationary training regime regardless of
epoch count.  See Phase 8B root cause section.*

Proceed to 8B (scouting pass).

---

### Phase 8B — Epoch Scouting Pass (single seed, complete 2026-04-18)

**Scouting design:** single seed (42), epochs_per_task ∈ {5, 15, 40},
TCL + SGD only. Purpose: distinguish volume hypothesis from calibration
hypothesis before committing to a full multi-seed sweep.

**Results (complete 2026-04-18, seed=42):**

| epochs | TCL forgetting | SGD forgetting | delta         | ordered% |
|--------|---------------|----------------|---------------|----------|
| 5      | 0.1921        | 0.1534         | +0.039        | ~1%      |
| 15     | 0.1081        | 0.1046         | +0.004        | ~1–3%    |
| 40     | 0.0824        | (see below)    | TBD (SGD run) | ~1–10%*  |

*ordered% at 40 epochs is inflated by numerical noise, not genuine convergence —
see numerical pathology note below.

**Rho trajectories — 15 epochs:**

Task 0: `[2.84, 2.92, 2.64, 2.71, 2.60, 2.70, 2.76, 2.59, 2.53, 3.11, 2.72, 3.43, 2.71, 3.27, 5.86]`
Task 3: `[2.74, 2.87, 2.91, 3.21, 3.07, 3.30, 4.03, 5.92, 4.04, 6.97, 5.96, 5.98, 11.40, 13.75, 38.31]`
Task 4: `[2.84, 2.56, 2.80, 2.67, 2.77, 3.19, 3.64, 3.09, 4.55, 4.39, 5.08, 4.74, 10.64, 6.30, 32.77]`

**Rho trajectory — 40 epochs, task 0:**

`[2.81, 2.89, 2.67, 2.76, 2.53, 2.72, 2.76, 2.59, 2.50, 2.75, 2.69, 3.24, 3.61, 3.57, 3.19, 3.45,`
` 3.34, 4.71, 5.54, 7.93, 8.14, 26.5, 27.7, 50.8, 15.2, 6.22, 11.3, 12.5, 31.2, 31.1, 41.0, 45.2,`
` 72.0, 185.3, 204.9, 243.8, 158.2, 16.2, 25.3, 63.1]`

rho_to_ordered_final (task 0 at 40 epochs) = **62.23**

**Numerical pathology note (40-epoch finding):**

At 40 epochs per task, sigma collapses to literal 0.0 in late tasks (e.g., task 3
sigma = 1e-06 → 0.0 by epoch 17; task 4 sigma = 0.0 by epoch 18).  Since
sigma_star = alpha × median(rolling sigma_window), sigma_star also contracts to 0.0.
The rho computation uses max(sigma_star, 1e-12) as the denominator, so when both
hit zero simultaneously, rho = sigma / 1e-12 → 50–243 across epochs 20–35 of
task 0.

The governor interprets successful training convergence (sigma → 0) as maximum
thermal disorder and applies 1.2× LR boost at the exact moment consolidation
should fire.  This is not a null effect — it is an active anti-effect.  TCL
under the rolling-median formulation is measurably worse than SGD during the
consolidation window because it is fighting convergence.

The Phase 8C fixed anchor directly inverts this: sigma → 0 with a frozen
sigma_star_anchor ≈ 0.003 gives rho → 0.0003, triggering 0.5× braking throughout
the convergence phase.  The same physical event that currently breaks the governor
becomes its clearest signal.

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

### Phase 8C Pre-Registered Success Criteria

**Pre-registered 2026-04-18, before Phase 8C validation run.**  These criteria
are locked before any experimental data is seen.  Outcome classification happens
against these definitions, not post-hoc.

**Diagnostic gate (phase8c_validate.py, single seed=42, 15 epochs):**
All three checks must pass before proceeding to the full 5-seed benchmark.
1. sigma_star CV < 0.05 within each task (anchor frozen, not co-tracking)
2. rho descends from early to late epochs in ≥ 2 tasks (convergence visible)
3. ordered% > 2% in convergence window (final half) of ≥ 1 task

If diagnostic gate fails: inspect which check failed; adjust anchor_n or alpha
before re-running diagnostic (not the benchmark).

**Full benchmark: 5 seeds × TCL + SGD, 15 epochs per task**

Three mutually exclusive outcomes, classified before unblinding the result:

---

**Outcome A — Clean Win**

Criteria (all required):
- mean(TCL forgetting) < mean(SGD forgetting) by > 0.01 (raw delta)
- Mann-Whitney p < 0.05, Cohen's d > 0.5
- Mechanism confirmed: ordered% in convergence window for ≥ 2 tasks across ≥ 3
  seeds

Interpretation: Fixed TCL is a working continual learning regulariser.  The
thermodynamic regime signal adds measurable benefit over vanilla gradient descent.
Action: Document mechanism, write up result, proceed to Phase 8D
(class-incremental).

---

**Outcome B — Partial Win**

Criteria (any one sufficient):
- Raw forgetting delta 0 < δ < 0.01 (TCL directionally better, under threshold)
- TCL forgetting variance significantly lower than SGD (std ratio < 0.6,
  Levene p < 0.05) — lower variance implies more consistent consolidation
- TCL accuracy higher by > 0.01 at comparable forgetting — better tradeoff
  without raw forgetting improvement

Interpretation: The fixed anchor makes the governor correctly calibrated.
The LR multipliers (1.2×/0.5×) may be too conservative to produce a strong
forgetting signal, or 15 epochs is insufficient for the full benefit to manifest.
The governor is working; the effect size needs tuning.
Action: Alpha sweep {0.3, 0.5, 0.7} or multiplier sweep (brake 0.5→0.3, boost
1.2→1.0) at 1 seed each to find the parameter combination that maximises delta.

---

**Outcome C — Mechanism Right, Signal Absent**

Criteria: all of the following:
- Diagnostic gate passes (anchor frozen, rho descends, ordered in conv window)
- Raw forgetting delta |δ| < 0.005 (TCL indistinguishable from SGD)
- No variance or accuracy advantage

Interpretation: The sigma_star redefinition correctly calibrates the governor —
ordered phase now fires at the right time and the LR brake is applied.  But LR
modulation alone is insufficient as a forgetting signal.  EWC's advantage derives
from directly penalising weight changes proportional to their Fisher importance.
TCL braking the LR during consolidation does not provide an equivalent constraint.
Action: Implement a dimensionality-weighted L2 penalty on weight change between
tasks (thermodynamic importance from participation ratio), analogous to EWC but
derived from the governor signal rather than Fisher diagonal.  This is a loss
function change, not just LR scheduling.  Design work required.

---

**Outcome framing for the paper:**

Regardless of which outcome obtains, the scientific story is:
- EWC (frozen network via Fisher importance) = degenerate success at forgetting
- TCL with rolling-median sigma_star = degenerate failure (anti-consolidation)
- TCL with fixed-anchor sigma_star = governor correctly calibrated → [A/B/C result]

The contribution is not contingent on Outcome A.  Outcome B is publishable as
"thermodynamic regime detection with correct calibration adds signal."  Outcome C
is publishable as "correct calibration is necessary but not sufficient; LR
modulation requires a complementary weight penalty term."  Both advance the
science.  The null result that would be unpublishable is "governor correctly
calibrated, no forgetting benefit, and no clear explanation why."  Outcome C
avoids that by providing the clear explanation (signal exists, mechanism
insufficient) and the path forward.

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
           anchor_snapshot, richer trace) — COMPLETE 2026-04-18, commit 7eda5a0
  → governor correctly implements specified rolling-median formulation
  → ordered reached ~1% of batches — calibration-limited (not volume-limited,
    as originally inferred; Phase 8B corrects this interpretation)

Phase 8B: Scouting pass {5, 15, 40} epochs × 1 seed — COMPLETE 2026-04-18
  → CALIBRATION verdict: rolling-median sigma_star pins rho ≈ 1/alpha
  → At 40 epochs: sigma→0 numerically, sigma_star→0, rho explodes to 243
  → Governor applies 1.2× boost during consolidation — active anti-effect
  → Volume hypothesis rejected; formulation is definitionally wrong

Phase 8C: Redefine sigma_star as fixed per-task anchor — IMPLEMENTED 2026-04-18
           commit e9ae604; validation script phase8c_validate.py on pod
  → Diagnostic gate: anchor frozen, rho descends, ordered in conv window
  → If gate passes: full 5-seed TCL+SGD benchmark at 15 epochs
  → Pre-registered outcomes A/B/C defined above — classified before unblinding

Phase 8D: Class-incremental Split-CIFAR-10 (only after Outcome A or B from 8C)
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
