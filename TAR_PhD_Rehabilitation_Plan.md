# TAR PhD Rehabilitation Plan
## A Complete, Phased Programme to Publication-Ready Standard

*Generated: 2026-06-01 | Status: ACTIVE PLAN*

---

## PART I — GROUND TRUTH DIAGNOSIS

Before the plan, a precise statement of what is actually true today, separated from what the system believes about itself.

### What is genuinely demonstrated

| Result | Dataset | N seeds | Trust tier | Publication status |
|---|---|---|---|---|
| TCL < EWC on forgetting (δ=−0.075, p=0.019, d=1.70) | Split-CIFAR-10 | 5 | trusted_manual_controlled | **Publication-allowed** |
| Penalty component essential; governor-alone WORSE than SGD | Split-CIFAR-10 | 5 | trusted_rerun | **Publication-allowed** |
| HPC forgetting 0.058 vs TCL baseline 0.126 (δ=−0.068, p=0.018, d=1.73) | Split-CIFAR-10 | 5 | trusted_rerun | **Underpowered — needs replication** |
| TCL < SGD on TinyImageNet (d=75.08) | TinyImageNet | 3 | trusted_rerun | **Gate B fails: n=3 < 5** |
| TCL < EWC on CIFAR-100 (directional) | CIFAR-100 | 3 | trusted_rerun | **Gate B fails: n=3 < 5** |

### What the system falsely believes it has demonstrated

- That the thermodynamic governor (regime detection, LR adjustment) is a primary mechanism. Phase 11 ablation shows governor-only is *worse* than SGD. The regime detector stays `"unknown"` in every trace examined — `rho=0, sigma=0` — meaning the LR adjustment mechanism has *never fired in production*.
- That 25 claim verdicts are meaningful. 96% read `insufficient_evidence`, all from April 29 (now 33 days old, expired under the 14-day aging policy).
- That TCL is a broadly applicable solution. The falsified frontier `fp-catastrophic-forgetting` (17 null, 4 adverse, 1 breakthrough) contradicts this.
- That HPC is a breakthrough. It is a 5-seed result in a 5-hypothesis multiple-comparisons study at uncorrected α=0.05. Bonferroni-corrected threshold is p=0.01; p=0.018 fails that.

### The four structural problems that underlie everything else

1. **The theory and the algorithm are decoupled.** The thermodynamic framing is post-hoc narrative, not a predictive theory. The mechanism that works (gradient-EMA elastic regularization) is not thermodynamic in any meaningful physics sense.
2. **The system is systematically underpowered.** 5 seeds at α=0.05 gives ~50% power for medium effects (d=0.5). The system's own preregistration acknowledges needing 32 seeds. Every published-looking result is statistically fragile.
3. **The core code has never been committed.** Six governance files governing all autonomous behaviour exist only in the working tree, unreviewed and unstaged. The system is running from unaudited code.
4. **No paper manuscript exists.** Every LaTeX file is a template. The publication handoff directory is empty. The system has been autonomously running for months without producing prose.

---

## PART II — THE PLAN

Organised into six phases running partially in parallel. Each phase has: objective, all tasks with rationale, success criteria, and what blocks the next phase.

---

## PHASE 0 — TRIAGE & GROUND-TRUTH LOCK
**Duration:** 1–2 weeks
**Objective:** Stop the bleeding, create a stable and honest foundation before any new experiments run.

### 0.1 — Halt and audit all running experiments

- Let `harder_domain_split_tinyimagenet` run to completion (it is actively computing, GPU at 87%, seed 0 still in-progress as of audit). Do not kill it.
- Once complete, record the result honestly: if TCL does not beat EWC at p<0.05 on TinyImageNet with 5 seeds, record it as DIRECTIONAL at best.
- After completion, halt all autonomous experiment submission. Do not start any new queue entries until Phase 1 is complete.
- **Rationale:** any new experiments submitted before statistical methodology is corrected will compound the existing underpowering problem.

### 0.2 — Commit the six governance files

The files `tar_research_director.py`, `tar_experiment_orchestrator.py`, `tar_living_research.py`, `tar_scheduler.py`, `tar_frontier.py`, `tar_autonomous_research.py` exist in the working tree but have never been committed. This means the system is running unaudited code that controls all autonomous behaviour.

- Read each file in full (as referenced in the stabilisation-off deliberation document, 2026-05-19).
- Classify each diff against HEAD: provenance (how did it change?), correctness (is it sound?), disposition (commit, revert, or rewrite?).
- Commit the files that pass review. For any file that contains unsafe autonomous overrides, revert or gate with an explicit flag.
- **Rationale:** until these are committed, `git blame` is meaningless for the most critical code in the system, and no peer can review the autonomy logic.

### 0.3 — Update active_session.json on daemon restart

The file still reads `DORMANT_NO_MANIFEST` despite the daemon running since 18:35 UTC today. The batch file resets service state files but not this one.

- Identify where `active_session.json` is written and why it was not updated on restart.
- Fix the restart logic so `active_session.json` always reflects actual running state.
- **Rationale:** this file is used by the manifest gate check in the watchdog; a stale DORMANT flag could cause incorrect gate decisions.

### 0.4 — Fix the queue maintainer NoneType error

The error `'<=' not supported between instances of 'NoneType' and 'int'` has fired every 30 seconds since 2026-05-28. This is a null priority field in the experiment queue sorting logic.

- Locate the queue priority comparison, add a null-guard (e.g., `priority or 0`).
- Verify fix by confirming the error stops appearing in `living_research.log`.
- **Rationale:** this is a known recurring failure. It does not crash the daemon but it pollutes logs and may silently skip priority ordering.

### 0.5 — Build the honest evidence inventory

Produce a single canonical document (a JSON file in `tar_state/`) listing every result with its actual trust tier, seed count, and honest publication status. This is the ground truth from which all paper planning flows.

Fields per entry: `experiment_id`, `dataset`, `method_comparison`, `n_seeds`, `trust_tier`, `t_stat`, `p_value`, `cohens_d`, `bonferroni_corrected_significant`, `publication_allowed`, `honest_verdict`.

- **Rationale:** the system's self-assessment of its evidence quality is optimistic. Having a cold, independent audit document prevents the paper-authoring logic from citing underpowered results as breakthroughs.

### 0.6 — Purge the financial econometrics contamination from gap scans

The gap scan reports repeatedly surface "mean-variance-skewness-kurtosis portfolio optimization" as a frontier gap. This is quantitative finance, not continual learning.

- Identify the literature source injecting this content (likely a misconfigured arXiv RSS feed or domain crossover in `research_ingest.py`).
- Add an explicit domain filter in the gap scanner to reject entries whose primary domain does not map to any active frontier problem's domain.
- Mark the contaminated gap entries with `status=rejected, reason=domain_mismatch_financial` so they do not resurface.
- **Rationale:** a PhD committee reviewing literature coverage would immediately reject a thesis whose related-work section includes portfolio optimization.

**Phase 0 success criteria:**
- [ ] `harder_domain_split_tinyimagenet` complete with result recorded
- [ ] Six governance files committed to git with review comments
- [ ] `active_session.json` updated correctly on restart
- [ ] Queue maintainer error eliminated from logs
- [ ] Honest evidence inventory JSON exists and is accurate
- [ ] Financial contamination removed from gap scans

**What Phase 0 unlocks:** Phases 1 and 5 can begin simultaneously once the evidence inventory exists and governance files are committed.

---

## PHASE 1 — STATISTICAL FOUNDATIONS
**Duration:** 2–3 weeks
**Objective:** Correct every statistical methodology issue before any new claim is generated or any paper is drafted.

### 1.1 — Replace z-critical with t-critical for small samples

In `benchmark_stats.py`, the CI95 formula uses `1.96 × (std / √n)` regardless of sample size. For n=3 this underestimates the interval by a factor of ~2.2 (t-critical = 4.303 for df=2).

Implementation:
```python
from scipy.stats import t as t_dist
df = sample_count - 1
t_crit = t_dist.ppf(0.975, df) if df > 0 else 1.96
ci95_half = t_crit * (std_dev / math.sqrt(sample_count))
```

- Apply to every place that computes CI95: `benchmark_stats.py`, `phase8c_benchmark.py`, `ewc_lambda_sweep.py`, `multimodal_payloads.py`.
- Re-run all existing comparisons with corrected CIs and record whether any previously "significant" results become non-significant.
- **Rationale:** reporting CIs that are ~50% too narrow is a fundamental statistical error. A reviewer who catches this will reject immediately.

### 1.2 — Implement Bonferroni correction for all pairwise comparisons

Every phase that compares TCL against multiple baselines conducts multiple tests at the same α=0.05 threshold. With 4 pairwise tests (TCL vs EWC, SI, SGD, and a fourth), the family-wise error rate is 18.5%.

- Implement a `bonferroni_correct(p_values, alpha=0.05)` utility that returns per-test corrected threshold `α/k`.
- Apply to all phase result computation: Phase 10 (4 methods → threshold 0.0125), Phase 12 EWC sweep (4 λ values → threshold 0.0125), Phase 13 SI sweep (4 c values → threshold 0.0125).
- Record which results survive Bonferroni correction. **The Phase 10 TCL vs EWC result (p=0.019) does NOT survive Bonferroni correction at α_corr=0.0125. This must be acknowledged honestly.**
- Consider whether Holm-Bonferroni (less conservative) is more appropriate for this use case.
- **Rationale:** every major venue now expects multiple-comparison correction. Without it, the Phase 10 headline result is not defensible.

### 1.3 — Conduct retrospective power analysis for all existing experiments

For each completed experiment, compute achieved power given the observed effect size, n=seeds, and α=0.05.

Record: `{experiment_id, n_seeds, observed_d, achieved_power, seeds_needed_for_80pct_power, seeds_needed_for_90pct_power}`.

Key results to expect:
- Phase 10 TCL vs EWC: d=1.70, n=5 → ~90% power (solid)
- HPC: d=1.73, n=5 → ~90% power (solid, but multiple-comparisons issue)
- Phase 17 TCL vs EWC: d=0.68, n=3 → ~30% power (grossly underpowered)
- Phase 16 TCL vs EWC: d~0.5 (estimated), n=3 → ~25% power (grossly underpowered)

- Write the power analysis results into the honest evidence inventory from Phase 0.5.
- **Rationale:** publishing underpowered results without disclosing the power deficit is a form of research misconduct. Acknowledging it and fixing it is the only path to credibility.

### 1.4 — Pre-registration protocol for all future experiments

Implement a mandatory pre-registration step before any experiment enters the queue.

The pre-registration document must contain:
- Hypothesis (precise, falsifiable, one-directional)
- Primary outcome measure (forgetting, accuracy, or both)
- Minimum detectable effect size (pre-specified, not observed)
- Required sample size from power analysis (not the fixed default of 5)
- Analysis plan (which test, paired or independent, one or two-tailed)
- Stopping rule (what constitutes a null result)
- Registration timestamp (before data collection begins)

This pre-registration is committed to git before the manifest is generated. Any post-hoc change requires a documented amendment.

- Build a `PreRegistrationRecord` Pydantic model and add it as a required field in `ManifestExperimentEntry`.
- The canonical registry's three-gate verification should add a fourth gate: `preregistration_present`.
- **Rationale:** the Phase 8B→8C scouting pattern is selective inference. Pre-registration makes this transparent and reviewable.

### 1.5 — Fix the confidence interval formula for non-normality

Add a Shapiro-Wilk normality test to the benchmark statistics pipeline:

- If p_shapiro > 0.05 (normally distributed): use t-distribution CI.
- If p_shapiro ≤ 0.05 (non-normal, n=5 is too small for Shapiro to have power): flag and use bootstrap CI with 1000 resamples.
- Record the normality test result in every comparison JSON.
- **Rationale:** CI validity requires either normality or large n. With n=5 and asymmetric forgetting distributions (bounded below at 0), normality is not guaranteed.

### 1.6 — Implement Wilcoxon signed-rank test as primary inferential test for paired comparisons

- For seed-matched comparisons (same seed, different method): use Wilcoxon signed-rank test (non-parametric paired test).
- For independent-group comparisons: use Mann-Whitney U.
- Report both parametric and non-parametric results for all major claims.
- **Rationale:** Wilcoxon is the appropriate test for small paired samples with non-normal distributions.

**Phase 1 success criteria:**
- [ ] All CI calculations use t-distribution
- [ ] Bonferroni correction applied to all multi-test comparisons
- [ ] Power analysis exists for every historical result
- [ ] Pre-registration schema implemented and enforced in manifest gate
- [ ] Normality checks added to statistics pipeline
- [ ] Wilcoxon signed-rank implemented as primary test

---

## PHASE 2 — EXPERIMENTAL RIGOUR
**Duration:** 4–8 weeks
**Objective:** Bring every key result up to publication-grade evidence by correct sample sizes, fair baseline tuning, and honest scope.

### 2.1 — Replicate the HPC result with adequate power

The `high_penalty_conservative` result (p=0.018, d=1.73) is the best single finding in the system, but it is:
- One of five simultaneously-tested hypotheses at uncorrected α=0.05 (false discovery risk)
- Based on only 5 seeds when the system's own preregistration acknowledges needing 32 for medium effects

Required action:
- Pre-register the replication as a confirmatory test of this specific hypothesis only.
- Run 20 additional seeds (total n=25) on Split-CIFAR-10 with HPC vs TCL baseline.
- Apply no multiple comparison correction (this is a pre-registered single-hypothesis test).
- Success criterion: p<0.05 in the replication, d≥0.5.
- If the replication fails: HPC is a false positive and must be recorded as such.
- **Rationale:** a single positive result from a 5-hypothesis battery is exactly the situation that produces false discoveries. Replication is the only scientific resolution.

### 2.2 — Rerun Phase 17 (TinyImageNet) with 5 seeds

The current result uses 3 seeds. Gate B (seed count ≥ 5) blocks publication.

Pre-registered design:
- Seeds: [42, 0, 1, 2, 3] (matching all other phases for cross-experiment consistency)
- Methods: TCL, EWC (λ=1000, the Phase 12 optimum), SI (c=0.01), SGD baseline, DER++ (new)
- Epochs: 40 per task
- Primary comparison: TCL vs EWC (the comparator established in Phase 10)
- Secondary: TCL vs SGD (expected large effect, hypothesis: d>2.0)
- Stopping rule: if TCL forgetting > EWC forgetting in ≥3/5 seeds, record as ADVERSE

### 2.3 — Rerun Phase 16 (CIFAR-100) with 5 seeds

Same situation as Phase 17. Current result: 3 seeds, not publication-allowed.

Pre-registered design mirrors Phase 17, adapted for CIFAR-100 (10 tasks × 10 classes).

### 2.4 — Add DER++ to all standard comparison phases

DER++ (Dark Experience Replay++) is implemented in `method_registry.py` but has never been included in any phase comparison.

- A NeurIPS reviewer will ask: "Why no replay baselines? DER++ (Buzzega 2020) consistently outperforms EWC and SI at low memory budgets."
- Add DER++ to Phases 10, 16, 17 reruns with `der_mem_size` swept over [100, 200, 500] to find fair operating point.
- **This may show DER++ outperforming TCL. That is the correct scientific finding if true. Acknowledging it strengthens, not weakens, the paper.**

### 2.5 — Add LwF (Learning without Forgetting) as a baseline

LwF (Li & Hoiem, 2016) is one of the canonical CL baselines. It is listed in `frontier_problems.json` as a known method but never appears in phase comparisons. Any CL paper without LwF will get a mandatory revision request.

- Implement `LwFMethod` in `method_registry.py` using knowledge distillation loss.
- Add to all comparison phases.

### 2.6 — Implement fair joint hyperparameter selection for all baselines

The current tuning situation:
- TCL: fixed configuration across all phases (fair, but potentially sub-optimal)
- EWC: swept in Phase 12 *after* Phase 10 results (post-hoc)
- SI: c=0.01 was chosen because c=0.1 collapses (post-hoc)
- DER++: will need tuning

Fair protocol:
- Designate Split-CIFAR-10 with **seed=999** (not in any existing seed set) as the validation split for hyperparameter selection.
- Tune all methods jointly on this split *before* running any confirmatory experiments.
- Lock the selected hyperparameters and pre-register them before running confirmatory phases.
- Report all tuning runs as "tuning results" in an appendix, not in the main results table.
- **Rationale:** TCL's fixed hyperparameters vs post-hoc-tuned baselines is an unfair comparison. Joint tuning on a held-out split is the standard in rigorous ML papers.

### 2.7 — Resolve the regime-detection activation failure

Phase 11 ablation showed governor-alone is worse than SGD. The trace files show `rho=0.0, sigma=0.0` across all epochs (regime never fires).

Two paths:

**Path A (Honest Ablation):** Document that the regime-detection LR adjustment contributes zero net benefit, and the elastic penalty term is the entire mechanism. Rename the system's core contribution honestly. Provide a rigorous ablation showing penalty-only ≥ full-TCL in most configurations.

**Path B (Fix the Regime Detector):** Investigate why `rho=0` persists. Likely causes: the sigma computation in `multimodal_payloads.py` using `loss` as a proxy for entropy rather than the activation-based sigma from `ActivationThermoObserver`; or the `warmup_batches=0` default causing the anchor to be set from a random-initialization state.

Path B is scientifically more interesting but requires more time. **Pursue Path B first (2-week time limit), then fall back to Path A if the governor cannot be made to activate reliably.**

### 2.8 — Conduct honest scope delimitation for every frontier problem

The falsified frontier `fp-catastrophic-forgetting` should be formally closed. The evidence record (17 null, 4 adverse, 1 breakthrough) is definitive.

- Write a formal null-result record for this frontier.
- Update all papers that reference this frontier to remove overclaiming language.
- Re-scope claims to what is actually supported.

**Phase 2 success criteria:**
- [ ] HPC replication with n=25 complete and honestly recorded
- [ ] Phase 17 rerun with 5 seeds and DER++/LwF complete
- [ ] Phase 16 rerun with 5 seeds and DER++/LwF complete
- [ ] Fair joint hyperparameter selection protocol run and locked
- [ ] Regime detector investigation resolved (Path A or Path B)
- [ ] Formal null-result for `fp-catastrophic-forgetting`

---

## PHASE 3 — MECHANISTIC CLARITY & THEORETICAL GROUNDING
**Duration:** 4–8 weeks (overlapping with Phase 2)
**Objective:** Build a coherent, defensible theoretical narrative that is predictive rather than post-hoc.

### 3.1 — Conduct the definitive mechanistic ablation

A full mechanistic ablation for publication requires:

| Condition | Description | Hypothesis |
|---|---|---|
| SGD baseline | No mechanism | Worst forgetting |
| Penalty only (λ fixed) | Elastic regularization, no thermal component | Should match full TCL |
| Governor only (LR adjust) | Regime-detection LR adjustment, no penalty | Hypothesis: ≈ SGD |
| TCL full | Both components | Should match or exceed penalty-only |
| TCL with anchor frozen at init | Tests whether per-task sigma_star matters | Controls for anchor mechanism |
| TCL with warmup_batches=60 | Tests whether warmup affects governor activation | Tests Path B from 2.7 |
| EWC (best lambda) | Standard comparator | Provides external benchmark |

All conditions run with n=5 seeds, Bonferroni-corrected across conditions, on Split-CIFAR-10 with pre-registered primary metric.

### 3.2 — Establish the theoretical connection between gradient-EMA importance and Fisher information

The core TCL mechanism is closely related to the online Fisher information estimate used in EWC. This connection should be made formal:

- Derive: `ThermalImportance.finalize()` is an online approximation to the diagonal Fisher under the assumption that gradient-squared concentrations persist over the task.
- Derive the bias introduced by early-stopping the EMA before task completion.
- Show: when `ema_beta → 1` (slow EMA), TCL importance approximates EWC Fisher; when `ema_beta → 0` (fast EMA), it approximates the final-batch Fisher only.
- **Publishable theorem:** "TCL importance is a temporally-smoothed online estimate of the task-specific Fisher information, with EWC as a limiting case."

### 3.3 — Position the contribution correctly

Given the mechanistic evidence, the honest contribution is:
- **Not:** "A thermodynamic theory of catastrophic forgetting"
- **Yes:** "Continuous gradient-energy EMA produces a richer importance estimate than a single-shot Fisher snapshot, leading to better forgetting-accuracy tradeoffs on standard CL benchmarks"

The thermodynamic framing can be preserved as motivating intuition (hot parameters = high gradient energy = important) while the technical contribution is stated precisely.

### 3.4 — Formally characterize HPC (High-Penalty Conservative)

The HPC variant differs from baseline TCL in two ways: higher regularization lambda and momentum disabled. Ablation needed:

- TCL with high lambda only vs TCL with no momentum only vs HPC (both) vs TCL baseline.
- 4-condition 5-seed study on Split-CIFAR-10. Pre-register primary comparison.
- **Rationale:** if the effect is entirely from lambda, then "HPC" is just "TCL with better hyperparameter tuning" — a finding about robustness, not a new mechanism.

### 3.5 — Build a real theoretical foundation for the participation ratio (D_PR)

The PR has known connections to:
- Intrinsic dimensionality of learned representations (Amsaleg et al., 2015)
- The double-descent and capacity arguments (Bartlett et al., 2020)
- Forgetting and representation stability (Ramasesh et al., 2021)

Build the theoretical connection and verify empirically: plot D_PR vs forgetting across all seeds for all methods. Does lower D_PR predict higher forgetting?

**Phase 3 success criteria:**
- [ ] Full 7-condition mechanistic ablation complete with pre-registered analysis
- [ ] Formal connection between ThermalImportance and diagonal Fisher derived and verified empirically
- [ ] HPC mechanism understood (lambda vs momentum)
- [ ] D_PR relationship to forgetting empirically tested
- [ ] Paper framing updated to match honest mechanistic picture

---

## PHASE 4 — CODE QUALITY & SYSTEM INTEGRITY
**Duration:** 3–4 weeks (overlapping with Phases 2–3)
**Objective:** Bring the codebase to the standard a PhD candidate would be comfortable sharing publicly.

### 4.1 — Fix the CI formula in all locations

Locations needing the t-distribution fix:
- `tar_lab/benchmark_stats.py` — primary CI computation
- `phase8c_benchmark.py`
- `ewc_lambda_sweep.py`
- `tar_lab/multimodal_payloads.py`

All locations must use the same formula and the same `scipy.stats` call.

### 4.2 — Fix the method synthesizer's baseline validation gap

`tar_lab/method_synthesizer.py` validates generated code only by running the sandbox without crashing.

Add a mandatory minibench validation step after sandbox success:
- Run the synthesized method on `split_cifar10` with `tiny_cnn` backbone, 3 tasks, seeds [42], 2 epochs.
- Verify: forgetting in range [0.0, 1.0], accuracy above random guessing (>0.2 for 5-class), no NaN or Inf in any metric.
- Only if this minibench passes does the method get saved to `synthesized_methods/`.

### 4.3 — Fix the generative director's leading prompt

The current prompt frames standard methods as failures and always proposes novelty. Replace with:

1. A diagnosis step: "classify the most likely root cause: (a) hyperparameter mis-specification, (b) architecture limitation, (c) dataset characteristic, (d) algorithmic limitation."
2. A proposal step only if the diagnosis is (c) or (d).
3. A rejection path: "If the root cause is (a) or (b), return `{action: 'tune_hyperparameters', suggestion: ...}` instead of proposing a new family."

### 4.4 — Fix the smoke-tier synthetic benchmark

The smoke-tier vision benchmark generates trivial geometric patterns (stripes, diagonal) as "images."

- Add `is_synthetic_smoke_bench: true` flag to all results from this benchmark.
- The validation pipeline must treat these as `trust_tier=smoke_only` and exclude them from publication-relevant evidence.
- Relabel from `"canonical_ready"` to `"smoke_only"` in `science_profiles.py`.

### 4.5 — Fix the silent benchmark downgrade

When a canonical-tier benchmark is unavailable, the system silently downgrades without disclosing this.

- Add `tier_requested` and `tier_executed` fields to every result JSON.
- The canonical registry must flag `tier_requested != tier_executed` as a publication-blocking issue.
- Log a WARNING-level alert when a downgrade occurs.

### 4.6 — Replace the keyword-based domain classifier

`tar_lab/science_profiles.py` uses brittle keyword scoring (e.g., "quantum" scores +5.0).

- Build a training set of 50 problem statements labeled with correct domains from existing frontier problems.
- Train a logistic regression classifier (or Claude with structured prompt) to assign domain labels.
- Report confidence as calibrated probability, not heuristic formula.
- Validate on held-out problems before deploying.

### 4.7 — Fill the knowledge graph

`knowledge_graph.json` has `"entries": []` despite 1,513 literature papers having been ingested.

- Identify the call path from `literature_engine.py` → `knowledge_graph.json` and determine why writes have not occurred.
- Run the synthesis step on the existing corpus.
- Minimum content: (a) 10–15 most relevant CL papers with key findings, (b) explicit connections between TCL and prior work (EWC, SI, DER++, LwF), (c) known failure modes of each method.

### 4.8 — Fix the invariant violations in TCL

- Add `assert penalty >= 0.0` after the penalty loop in `ThermalMemory.penalty()`.
- Add a unit test that verifies penalty is always ≥ 0 for edge cases (zero importance, zero drift, identical weights).

### 4.9 — Implement power-analysis-based sample size selection in the pre-registration gate

```python
from statsmodels.stats.power import TTestOneSamplePower

def required_seeds(target_effect_size_d: float, alpha: float = 0.05, power: float = 0.80) -> int:
    analysis = TTestOneSamplePower()
    n = analysis.solve_power(effect_size=target_effect_size_d, alpha=alpha, power=power)
    return max(5, math.ceil(n))
```

If required n exceeds `max_affordable_seeds`, the experiment is queued with note: "Underpowered — may not reach publication tier."

**Phase 4 success criteria:**
- [ ] All CI computations use t-distribution
- [ ] Method synthesizer has minibench validation
- [ ] Generative director prompt has diagnosis step and rejection path
- [ ] Smoke benchmarks flagged as non-evidence
- [ ] Benchmark tier downgrade is disclosed in result records
- [ ] Domain classifier replaced with validated ML classifier
- [ ] Knowledge graph populated
- [ ] TCL invariant assertion added and tested
- [ ] Pre-registration includes power-analysis-driven n calculation

---

## PHASE 5 — PAPER PIPELINE
**Duration:** 8–12 weeks (begins after Phase 0 and Phase 1 complete)
**Objective:** Produce at minimum one real, defensible, submission-ready manuscript.

### 5.1 — Identify the single strongest claim and build one paper

The honest situation after Phase 2: the core Phase 10 TCL vs EWC result at p=0.019 does not survive Bonferroni correction (corrected threshold: 0.0125). This means the headline result of the mechanism paper is currently undefended at standard statistical thresholds.

Path forward depends on Phase 2 outcomes:
- If Phase 2 replications show p<0.01 on CIFAR-10 and TinyImageNet with 5 seeds each: proceed to full paper.
- If Phase 2 shows p<0.05 on both datasets but not consistently <0.01: submit to a lower-tier venue with transparent limitations section.
- If Phase 2 HPC replication succeeds: a separate, stronger paper is possible focused specifically on the high-lambda no-momentum finding.

### 5.2 — Write the paper structure

Structure for the mechanism paper:
1. **Abstract** (150 words): Problem, approach, key result, implication.
2. **Introduction**: Catastrophic forgetting as a real deployment problem. Current solutions (EWC, SI, replay). The limitation: Fisher snapshot at task-end misses the temporal dynamics of learning. The proposal: accumulate gradient energy continuously.
3. **Related Work**: EWC (Kirkpatrick 2017), SI (Zenke 2017), LwF (Li 2016), DER++ (Buzzega 2020), PackNet, Progressive Neural Networks. Position CGEIA in this space.
4. **Method**: ThermalImportance algorithm precisely stated. ThermalMemory ring buffer. TCLRegularizer penalty. Connection to EWC as a limiting case (from Phase 3.2). Complexity analysis.
5. **Experiments**:
   - Phase 10 (CIFAR-10, 4-way comparison, 5 seeds)
   - Phase 16 rerun (CIFAR-100, 5 seeds)
   - Phase 17 rerun (TinyImageNet, 5 seeds)
   - Phase 11 ablation (governor vs penalty)
   - HPC ablation (lambda vs momentum)
6. **Analysis**: D_PR as a forgetting predictor (Phase 3.5). Power analysis table. Effect sizes with CIs.
7. **Limitations**: Regime detector non-activation. Bounded to split benchmarks with task IDs. Not evaluated on class-incremental. Single backbone (ResNet-18).
8. **Conclusion**: Honest statement of what was shown and not shown.

### 5.3 — Enable tar_author.py to draft real sections

- Verify the prompt structure in `tar_author.py` for each section renderer.
- Ensure prompts provide: exact numerical results from the evidence inventory, a pre-written outline, constraints on length and citation format.
- Run the author on each section against the completed evidence inventory.
- **Human review every output before accepting — the author is a drafting tool, not a final authority.**

### 5.4 — Build the related work section using the knowledge graph

Standard NeurIPS CL comparison table format:

| Method | Split-CIFAR-10 Forgetting | Split-CIFAR-100 Forgetting | Source |
|---|---|---|---|
| SGD baseline | 0.233 ± 0.003 | — | Our Phase 10 |
| EWC (λ=1000) | 0.160 ± 0.020 | — | Our Phase 12 |
| SI (c=0.01) | 0.092 ± 0.001 | — | Our Phase 13 |
| LwF | — | — | To be run |
| DER++ (mem=200) | — | — | To be run |
| TCL (ours) | 0.116 ± 0.029 | — | Our Phase 10 |

### 5.5 — Target venue and submission timeline

- **If HPC replication succeeds and TinyImageNet/CIFAR-100 achieve 5-seed results:** Target ICLR 2027 (abstract deadline ~September 2026). Achievable if Phase 2 completes by August 2026.
- **If results are solid but marginal (p<0.05 without Bonferroni):** Target ContinualAI workshop at NeurIPS 2026 (typically September deadline). Use workshop feedback to strengthen the full paper.
- **Do not submit to arXiv until Phase 2 replication results are in.** A preprint with underpowered claims that later fail to replicate damages credibility.

**Phase 5 success criteria:**
- [ ] LaTeX manuscript with real prose in all 8 sections
- [ ] Results table with honest effect sizes and corrected CIs
- [ ] Related work table with published numbers from literature
- [ ] Limitations section that addresses regime non-activation and scope
- [ ] Compilation succeeds and produces a real PDF (not a template)
- [ ] All claims in the manuscript traceable to the evidence inventory

---

## PHASE 6 — GOVERNANCE, SAFETY & LONG-TERM AUTONOMY
**Duration:** 4–6 weeks (parallel with Phases 3–5)
**Objective:** Bring the autonomous research infrastructure to the standard where it can safely be trusted to run unattended without producing false science.

### 6.1 — Cryptographically seal governance state files

Implementation:
- Generate an Ed25519 keypair at system setup, stored in a protected location.
- When any governance state file is written, append a signature: `{content_hash: SHA256(content), signature: Ed25519(content_hash)}`.
- When the watchdog or manifest gate reads any governance file, verify the signature before trusting the content. On failure: fail-closed.
- Private key must NOT be accessible to experiment sandbox processes.

### 6.2 — Enforce execution_policy.json at code generation time

`tar_state/policies/execution_policy.json` declares `"require_sandbox_for_generated_code": true` but no enforcement code raises `ExecutionPolicyViolation`.

- Add enforcement call in `method_synthesizer.py`: `assert_policy_allows_generated_code_execution(workspace)`.
- If policy is not readable: fail-closed.
- Log every policy enforcement check to the alert system.

### 6.3 — Make Docker non-bypassable for experiment execution

- Remove the `allow_host_fallback` path for experiment execution.
- If Docker is unavailable, fail with: `ExecutionPolicyViolation("Docker required for experiment execution; subprocess fallback is disabled.")`.
- Allow host execution only for non-experiment operations (daemon startup, health checks).

### 6.4 — Build a tamper-proof audit log

- Implement an append-only JSONL audit log at `tar_state/audit.jsonl` (distinct from alerts).
- Every manifest authorization, verdict generation, governance file write, experiment start and stop is appended with timestamp, PID, and SHA256 of the state file at that moment.
- The audit log is never trimmed. Gzip-compressed monthly.

### 6.5 — Close the autonomous mode human-review bypass

`human_review.py` has `_UniversalApproval` — a sentinel that approves all experiments in autonomous mode.

The correct architecture:
- In autonomous mode, the Director proposes experiments; they enter a `pending_director_review` queue.
- A new `auto_approve_after_hours` setting (default: 24 hours) allows experiments to proceed if no veto is received.
- This creates a genuine veto window rather than instant approval.
- The operator sees: "3 experiments will start in 23 hours unless vetoed."
- **Rationale:** the current system gives the appearance of human oversight but provides none.

### 6.6 — Implement per-session API cost budget with hard stop

- Add a `session_budget_usd` field to the daily cost tracker.
- If cumulative session cost exceeds the budget: all LLM calls degrade to Haiku and an alert is raised.
- If session cost reaches 2× the budget: all LLM calls are blocked and the operator is notified.

### 6.7 — Establish a recurring integrity check routine

Build `tar_health_check.py` into a comprehensive integrity suite that runs:
- On startup (before any execution)
- Daily at a fixed time
- On demand via `tar_cli.py --health-check`

Checks:
- All governance file signatures valid
- All canonical comparison results pass the three-gate registry check
- All evidence in the evidence inventory matches the raw comparison JSON files
- `active_session.json` reflects actual running state
- GPU temperature within safe range
- API key reachable (ping with minimal tokens)
- No orphan processes (PIDs in state files that no longer exist)
- No stale leases (leases >24 hours old)

Output: health report JSON written to `tar_state/health_report.json`.

**Phase 6 success criteria:**
- [ ] Ed25519 signatures on all governance files
- [ ] `ExecutionPolicyViolation` raised when policy requires sandbox
- [ ] Docker fallback for experiments eliminated
- [ ] Append-only audit log implemented
- [ ] Auto-approve-after-24h veto window replaces `_UniversalApproval`
- [ ] Session API cost budget enforced
- [ ] Health check routine operational and running daily

---

## PART III — TIMELINE & DEPENDENCY MAP

```
WEEK  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
      ├───Phase 0: Triage (2w)────┤
                    ├───Phase 1: Statistical Foundations (3w)────┤
                    ├───Phase 4: Code Quality (4w)───────────────────┤
                          ├───Phase 2: Experimental Rigour (8w)──────────────────────┤
                          ├───Phase 3: Mechanistic Clarity (8w)──────────────────────┤
                          ├───Phase 6: Governance (6w)─────────────────────┤
                                    ├───Phase 5: Paper Pipeline (12w)────────────────────────────────┤
```

**Critical path:** Phase 0 → Phase 1 → Phase 2 (HPC replication) → Phase 5 (paper)

**Parallelizable:** Phase 3, Phase 4, Phase 6 can all run while Phase 2 experiments execute.

---

## PART IV — WHAT WOULD A PhD EXAMINER ACCEPT

After all phases complete, the system should be able to defend:

**Defensible claim 1 (mechanism):**
> "Continuous gradient-energy EMA (ThermalImportance) produces a richer importance estimate than the EWC single-shot Fisher snapshot. On Split-CIFAR-10 (n=5 seeds, Holm-corrected p<0.025), this reduces catastrophic forgetting by [X] percentage points while maintaining equivalent accuracy. The effect generalizes to Split-CIFAR-100 and Split-TinyImageNet (n=5 seeds each)."

**Defensible claim 2 (HPC variant, if replication succeeds):**
> "A high-penalty conservative variant of CGEIA, with elevated λ and SGD momentum disabled, reduces forgetting by ~50% compared to the baseline CGEIA configuration. This appears to be primarily attributable to [lambda / momentum removal / both — resolved in Phase 3.4]. The mechanism is [explained / still under investigation]."

**Honest limitations section:**
- The thermodynamic regime-detection LR adjustment does not contribute to the performance improvement (Phase 11 ablation; governor-alone < SGD).
- Results are restricted to task-incremental scenarios where task IDs are provided at test time. Class-incremental performance is untested.
- With DER++ included as a baseline, TCL [beats / does not beat] replay-based methods — the honest result from Phase 2.4.
- Sample sizes remain below what would be required to claim that null results are true negatives (power for null hypotheses not tested).

---

## PART V — WHAT MUST NEVER HAPPEN AGAIN

**The five failure modes this plan prevents:**

1. **Running experiments that cannot reach publication tier.** Every experiment now requires a pre-registered power analysis before queueing. If `n_required > n_affordable`, the experiment is marked `exploration_grade` and cannot be cited as evidence in a paper.

2. **Falsely attributing results to a mechanism that doesn't work.** The mechanistic ablation (Phase 3.1) and the governor activation investigation (Phase 2.7) ensure that the paper's mechanism section describes what actually happens in the code.

3. **Post-hoc hyperparameter tuning of baselines.** Joint tuning on a held-out validation split (Phase 2.6) eliminates the EWC lambda=100→1000 retroactive adjustment.

4. **Autonomous operation from unreviewed code.** The six uncommitted governance files (Phase 0.2) are committed and reviewed before the next autonomous cycle begins.

5. **Calling underpowered results breakthroughs.** The honest evidence inventory (Phase 0.5) and Bonferroni correction (Phase 1.2) mean that the system's internal claim status reflects what a reviewer would accept, not what the system wishes were true.

---

*This plan gives TAR a clear, executable path from its current state — where the best result is a 5-seed CIFAR-10 comparison that is technically not Bonferroni-significant — to a system capable of producing and defending a legitimate, peer-reviewed contribution to the continual learning literature.*
