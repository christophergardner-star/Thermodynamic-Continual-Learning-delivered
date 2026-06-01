# TAR PhD Rehabilitation Plan
## A Complete, Phased Programme to Full Competency — Research, Engineering & Operations

*Generated: 2026-06-01 | Engineering audit merged: 2026-06-01 | Status: ACTIVE PLAN*

---

## PART I — GROUND TRUTH DIAGNOSIS

### A. Research ground truth — what is scientifically demonstrated today

| Result | Dataset | N seeds | Trust tier | Publication status |
|---|---|---|---|---|
| TCL < EWC on forgetting (δ=−0.075, p=0.019, d=1.70) | Split-CIFAR-10 | 5 | trusted_manual_controlled | **Publication-allowed** |
| Penalty component essential; governor-alone WORSE than SGD | Split-CIFAR-10 | 5 | trusted_rerun | **Publication-allowed** |
| HPC forgetting 0.058 vs TCL baseline 0.126 (δ=−0.068, p=0.018, d=1.73) | Split-CIFAR-10 | 5 | trusted_rerun | **Underpowered — needs replication** |
| TCL < SGD on TinyImageNet (d=75.08) | TinyImageNet | 3 | trusted_rerun | **Gate B fails: n=3 < 5** |
| TCL < EWC on CIFAR-100 (directional) | CIFAR-100 | 3 | trusted_rerun | **Gate B fails: n=3 < 5** |

### B. Engineering ground truth — layer-by-layer grades

| Layer | Grade | One-line verdict |
|---|---|---|
| ML Algorithm Engine | **B** | Correct algorithms, sound math; monolithic functions, regime detector never fires |
| Orchestration & Daemons | **C+** | Good manifest gate; restart loops possible, process lifecycle fragile |
| Infrastructure, API & Deployment | **D+** | Critical credential leak in repo, dev server in production, no CI/CD |
| State Management & Storage | **B−** | Atomic writes solid; no schema migrations, unbounded JSONL, missing fsync |
| Literature & Paper Pipeline | **B−** | Well-designed; active learner orphaned, self-improvement never run, zero prose |
| **System Overall** | **C+** | Research-grade science; development-grade operations |

### C. What the system falsely believes it has demonstrated

- That the thermodynamic governor (regime detection, LR adjustment) is a primary mechanism. Phase 11 ablation shows governor-only is *worse* than SGD. The regime detector stays `"unknown"` in every trace examined — `rho=0, sigma=0` — meaning the LR adjustment mechanism has *never fired in production*.
- That 25 claim verdicts are meaningful. 96% read `insufficient_evidence`, all from April 29 (now 33 days old, expired under the 14-day aging policy).
- That TCL is a broadly applicable solution. The falsified frontier `fp-catastrophic-forgetting` (17 null, 4 adverse, 1 breakthrough) contradicts this.
- That HPC is a breakthrough. It is a 5-seed result in a 5-hypothesis multiple-comparisons study at uncorrected α=0.05. Bonferroni-corrected threshold is p=0.01; p=0.018 fails that.

### D. The five structural problems that underlie everything

1. **The theory and the algorithm are decoupled.** The thermodynamic framing is post-hoc narrative, not a predictive theory. The mechanism that works (gradient-EMA elastic regularization) is not thermodynamic in any meaningful physics sense. The regime detector never activates in production.

2. **The system is systematically underpowered.** 5 seeds at α=0.05 gives ~50% power for medium effects (d=0.5). The system's own preregistration acknowledges needing 32 seeds. The CI formula uses z-critical (1.96) regardless of sample size — for n=3 this makes confidence intervals ~2.2× too narrow. Every published-looking result is statistically fragile.

3. **The core code has never been committed.** Six governance files governing all autonomous behaviour exist only in the working tree, unreviewed and unstaged. The system is running from unaudited code.

4. **No paper manuscript exists.** Every LaTeX file is a template. The publication handoff directory is empty. The active learner that should populate the knowledge graph has never been wired to the orchestrator and has therefore never run. The self-improvement anchor pack has never been initialized. The system has been autonomously running for months without producing prose.

5. **The operational infrastructure is development-grade.** Live API credentials are committed to the public GitHub repository. There is no CI/CD pipeline, no pinned dependency versions, no production web server, no JSONL size management, and no schema migration framework. The watchdog can enter infinite restart loops. Shared JSON state files are written by multiple processes without locking.

---

## PART II — THE PLAN

Nine phases, ordered by urgency. Phase E0 is a same-day emergency action. Phases 0–6 are the research rehabilitation programme. Phase 7 is the engineering infrastructure uplift. All phases have explicit success criteria.

---

## PHASE E0 — EMERGENCY SECURITY
**Duration:** Same day — complete before any other work
**Objective:** Eliminate the live credential exposure before anything else.

### E0.1 — Rotate all exposed credentials immediately

The file `tar_state/api_secrets.json` contains live credentials committed to the public GitHub repository `christophergardner-star/Thermodynamic-Continual-Learning-delivered`. The file `publish_config.json` in the repository root contains FTP credentials. Both were pushed in the commit made during this session.

Credentials to rotate:
- Anthropic API key (sk-ant-api03-... pattern)
- RunPod API key
- RunPod S3 access key and secret key
- FTP username/password in `publish_config.json`

**Do this now: go to each provider's console and invalidate/regenerate every key in those files before any other step in this plan.**

### E0.2 — Remove credentials from git history

Rotating keys stops future harm but the old keys remain in git history. Remove them:

```bash
# Install BFG Repo-Cleaner or use git-filter-repo
# BFG approach:
bfg --delete-files api_secrets.json
bfg --replace-text passwords.txt  # list of old key strings
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
git push --force --tags
```

Notify any collaborators with clones of the repo that they must re-clone.

### E0.3 — Add .gitignore entries for all secrets

Add to `.gitignore` immediately:
```
tar_state/api_secrets.json
publish_config.json
.env
*.key
*_secrets.json
*credentials*.json
```

### E0.4 — Move credentials to environment variables

Replace all hardcoded credential reads with environment variable lookups. `api_secrets.json` should not exist as a file — keys should live only in the process environment (set via `.env` which is gitignored, or via Windows environment variables).

Update `llm_bridge.py`, `tar_api.py`, and `sync_research.py` to read exclusively from `os.environ`. Provide a `.env.example` file (committed, with placeholder values only) so the startup procedure is documented.

### E0.5 — Add pre-commit secrets detection

Install `detect-secrets` and add a pre-commit hook so this class of leak cannot happen again:

```bash
pip install detect-secrets pre-commit
detect-secrets scan > .secrets.baseline
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

**Phase E0 success criteria:**
- [ ] All exposed credentials rotated at provider consoles
- [ ] Git history cleaned; force-push complete
- [ ] `.gitignore` excludes all secrets files
- [ ] Credentials read from environment variables only
- [ ] pre-commit detect-secrets hook installed and passing

**What Phase E0 unlocks:** Everything else. No other work should proceed until this is done.

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

The files `tar_research_director.py`, `tar_experiment_orchestrator.py`, `tar_living_research.py`, `tar_scheduler.py`, `tar_frontier.py`, `tar_autonomous_research.py` exist in the working tree but have never been committed. The engineering audit confirms: four are safe, one (`tar_autonomous_research.py`) is intentionally a stub that prints "retired," and one (`tar_experiment_orchestrator.py`) contains an autonomous manifest generation pathway that deserves explicit review.

- Read each file against HEAD and classify: provenance, correctness, disposition (commit / revert / gate).
- For `tar_experiment_orchestrator.py`: confirm the `set_autonomous(True)` path produces git-committed manifests and document this explicitly.
- Commit all six. Add inline review comments for any non-obvious design decision.
- **Rationale:** until these are committed, `git blame` is meaningless for the most critical code in the system.

### 0.3 — Update active_session.json on daemon restart

The file still reads `DORMANT_NO_MANIFEST` despite the daemon running since 18:35 UTC. The batch file resets service state files but not this one.

- Identify where `active_session.json` is written and fix the restart logic so it always reflects actual running state.
- **Rationale:** this file is used by the manifest gate check in the watchdog; a stale DORMANT flag could cause incorrect gate decisions.

### 0.4 — Fix the queue maintainer NoneType error

The error `'<=' not supported between instances of 'NoneType' and 'int'` has fired every 30 seconds since 2026-05-28. This is a null priority field in the experiment queue sorting logic.

- Locate the queue priority comparison, add a null-guard (e.g., `priority or 0`).
- Verify fix by confirming the error stops appearing in `living_research.log`.

### 0.5 — Build the honest evidence inventory

Produce a single canonical JSON file in `tar_state/honest_evidence_inventory.json` listing every result with its actual trust tier, seed count, and honest publication status.

Fields per entry: `experiment_id`, `dataset`, `method_comparison`, `n_seeds`, `trust_tier`, `t_stat`, `p_value_uncorrected`, `p_value_bonferroni`, `cohens_d`, `achieved_power`, `bonferroni_corrected_significant`, `publication_allowed`, `honest_verdict`.

- **Rationale:** the system's self-assessment of its evidence quality is optimistic. This document is the ground truth from which all paper planning flows.

### 0.6 — Purge the financial econometrics contamination from gap scans

The gap scan reports repeatedly surface "mean-variance-skewness-kurtosis portfolio optimization" as a frontier gap. This is quantitative finance, not continual learning.

- Identify the literature source (likely a misconfigured arXiv RSS feed or domain crossover in `research_ingest.py`).
- Add an explicit domain filter in the gap scanner to reject entries whose primary domain does not map to any active frontier problem's domain.
- Mark all contaminated gap entries with `status=rejected, reason=domain_mismatch_financial`.

### 0.7 — Fix the watchdog restart loop

The watchdog has a 30-second restart cooldown but no maximum restart count, no exponential backoff, and no circuit breaker. A daemon that fails immediately on startup will be respawned every 30 seconds forever.

- Add a restart counter per service with a rolling 1-hour window.
- If a service restarts more than 6 times in one hour: stop respawning, write `CIRCUIT_OPEN` to the service's health field, raise a CRITICAL alert.
- The circuit resets only when manually cleared by the operator (`tar_cli.py --reset-service-circuit <name>`).
- Add exponential backoff to the cooldown: 30s, 60s, 120s, 240s, cap at 300s.
- **Rationale:** the current system will consume resources and fill logs indefinitely on a broken daemon. A circuit breaker converts an invisible runaway into a visible operator action.

### 0.8 — Fix process lifecycle: prevent orphaned experiments

Daemons are launched with `DETACHED_PROCESS` flags and are not killed when the watchdog stops. If the watchdog crashes while an experiment is running, the experiment subprocess continues unmonitored; on daemon restart a duplicate can be launched.

- Track actual experiment subprocess PIDs separately from the daemon PID in `process_registry.json`.
- When marking an experiment as `STALLED` or `FAILED` in `reconcile_runtime_state()`, attempt `os.kill(subprocess_pid, signal.SIGTERM)` before resetting the spec.
- On watchdog shutdown (clean or via `STOP_TAR.bat`), send `SIGTERM` to all registered experiment subprocesses before exiting.
- Add a startup check: if `process_registry.json` shows a running experiment PID that still exists, confirm it matches the expected experiment before accepting it as valid.
- **Rationale:** orphaned subprocesses running alongside a newly launched duplicate produce corrupted or overwritten results with no error.

### 0.9 — Add lockfiles to all shared JSON state files

Only `runtime_ledger.json` currently has a proper lockfile. `experiment_queue.json`, `experiment_archive.json`, `research_director_state.json`, and `research_coordination_state.json` are written by multiple daemons without locking.

- Apply the same lockfile pattern already used in `result_artifacts.py` (atomic `open(path, "x")` with 10-second timeout) to all shared multi-writer state files.
- Centralise the lock acquisition helper so it is not re-implemented per module.
- **Rationale:** under normal operation (staggered 30s poll intervals) races are unlikely; under rapid restart or high-load scenarios they are not.

**Phase 0 success criteria:**
- [ ] `harder_domain_split_tinyimagenet` complete with result recorded
- [ ] Six governance files committed to git with review comments
- [ ] `active_session.json` updated correctly on restart
- [ ] Queue maintainer NoneType error eliminated from logs
- [ ] Honest evidence inventory JSON exists and is accurate
- [ ] Financial contamination removed from gap scans
- [ ] Watchdog restart loop circuit breaker implemented
- [ ] Experiment subprocess PIDs tracked and cleaned up on shutdown
- [ ] Lockfiles on all shared multi-writer JSON state files

**What Phase 0 unlocks:** Phases 1 and 5 can begin simultaneously.

---

## PHASE 1 — STATISTICAL FOUNDATIONS
**Duration:** 2–3 weeks
**Objective:** Correct every statistical methodology issue before any new claim is generated or any paper is drafted.

### 1.1 — Replace z-critical with t-critical for small samples

In `benchmark_stats.py`, the CI95 formula uses `1.96 × (std / √n)` regardless of sample size. For n=3 this underestimates the interval by a factor of ~2.2 (t-critical = 4.303 for df=2).

Implementation to apply at every CI computation location:
```python
from scipy.stats import t as t_dist
df = sample_count - 1
t_crit = t_dist.ppf(0.975, df) if df > 0 else 1.96
ci95_half = t_crit * (std_dev / math.sqrt(sample_count))
```

Affected files: `tar_lab/benchmark_stats.py`, `phase8c_benchmark.py`, `ewc_lambda_sweep.py`, `tar_lab/multimodal_payloads.py`.

- Re-run all existing comparisons with corrected CIs. Record whether any previously "significant" results become non-significant.
- **Rationale:** reporting CIs that are ~50% too narrow is a fundamental statistical error. A reviewer who catches this will reject immediately.

### 1.2 — Implement Bonferroni correction for all pairwise comparisons

Every phase that compares TCL against multiple baselines conducts multiple tests at the same α=0.05 threshold. With 4 pairwise tests, the family-wise error rate is 18.5%.

- Implement `bonferroni_correct(p_values, alpha=0.05)` utility returning per-test corrected threshold `α/k`.
- Apply to all phase result computation: Phase 10 (4 methods → threshold 0.0125), Phase 12 EWC sweep, Phase 13 SI sweep.
- Record which results survive. **The Phase 10 TCL vs EWC result (p=0.019) does NOT survive Bonferroni correction at α_corr=0.0125. This must be acknowledged honestly.**
- Evaluate Holm-Bonferroni as a less conservative alternative.

### 1.3 — Conduct retrospective power analysis for all existing experiments

For each completed experiment, compute achieved power given observed effect size, n=seeds, and α=0.05.

Record: `{experiment_id, n_seeds, observed_d, achieved_power, seeds_needed_for_80pct_power, seeds_needed_for_90pct_power}`.

Key expected results:
- Phase 10 TCL vs EWC: d=1.70, n=5 → ~90% power (solid)
- HPC: d=1.73, n=5 → ~90% power (solid, but multiple-comparisons issue)
- Phase 17 TCL vs EWC: d=0.68, n=3 → ~30% power (grossly underpowered)
- Phase 16 TCL vs EWC: d~0.5 estimated, n=3 → ~25% power (grossly underpowered)

Write results into the honest evidence inventory from Phase 0.5.

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

The pre-registration is committed to git before the manifest is generated. Any post-hoc change requires a documented amendment.

- Build a `PreRegistrationRecord` Pydantic model; add as required field in `ManifestExperimentEntry`.
- Add fourth gate to canonical registry verification: `preregistration_present`.

### 1.5 — Fix confidence intervals for non-normality

Add a Shapiro-Wilk normality test to the benchmark statistics pipeline:
- If p_shapiro > 0.05: use t-distribution CI.
- If p_shapiro ≤ 0.05 (or n<8, too small for Shapiro to have power): flag and use bootstrap CI with 1000 resamples.
- Record normality test result in every comparison JSON.

### 1.6 — Implement Wilcoxon signed-rank as primary inferential test for paired comparisons

- Seed-matched comparisons (same seed, different method): Wilcoxon signed-rank (non-parametric paired test).
- Independent-group comparisons: Mann-Whitney U.
- Report both parametric and non-parametric results for all major claims.

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

The `high_penalty_conservative` result (p=0.018, d=1.73) is the best single finding in the system, but it is one of five simultaneously-tested hypotheses at uncorrected α=0.05.

- Pre-register the replication as a confirmatory single-hypothesis test.
- Run 20 additional seeds (total n=25) on Split-CIFAR-10 with HPC vs TCL baseline.
- Success criterion: p<0.05 in the replication, d≥0.5. If it fails: record as false positive.

### 2.2 — Rerun Phase 17 (TinyImageNet) with 5 seeds

Current result: 3 seeds, Gate B fails.

Pre-registered design:
- Seeds: [42, 0, 1, 2, 3]
- Methods: TCL, EWC (λ=1000), SI (c=0.01), SGD baseline, DER++ (new)
- Epochs: 40 per task
- Primary comparison: TCL vs EWC
- Stopping rule: if TCL forgetting > EWC forgetting in ≥3/5 seeds, record as ADVERSE

### 2.3 — Rerun Phase 16 (CIFAR-100) with 5 seeds

Same situation as Phase 17. Pre-registered design mirrors 2.2, adapted for CIFAR-100 (10 tasks × 10 classes).

### 2.4 — Add DER++ to all standard comparison phases

DER++ (Buzzega 2020) is implemented in `method_registry.py` but never used. A NeurIPS reviewer will require replay baselines.

- Add DER++ to Phases 10, 16, 17 reruns with `der_mem_size` swept over [100, 200, 500] on the held-out validation split (seed=999) from Phase 2.6.
- **This may show DER++ outperforming TCL. That is the correct scientific finding if true.**

### 2.5 — Add LwF (Learning without Forgetting) as a baseline

LwF (Li & Hoiem, 2016) appears in `frontier_problems.json` but never in phase comparisons. Any CL paper without it will receive a mandatory revision request.

- Implement `LwFMethod` in `method_registry.py` using knowledge distillation loss.
- Add to all comparison phases.

### 2.6 — Implement fair joint hyperparameter selection for all baselines

Current problem: TCL uses fixed config (fair), EWC lambda swept post-hoc after Phase 10 (unfair), SI c=0.01 chosen because c=0.1 collapses (post-hoc).

Fair protocol:
- Designate Split-CIFAR-10 with **seed=999** (not in any existing seed set) as the validation split.
- Tune all methods jointly on this split before running any confirmatory experiments.
- Lock and pre-register hyperparameters before confirmatory phases.
- Report all tuning runs in an appendix, not in the main results table.

### 2.7 — Resolve the regime-detection activation failure

Every production trace shows `rho=0.0, sigma=0.0, regime="unknown"`. Two paths:

**Path B (Fix):** Investigate why `rho=0` persists. Root cause candidates: `multimodal_payloads.py` uses `loss` as proxy for entropy rather than activation-based sigma from `ActivationThermoObserver`; the two sigma measurements exist in parallel and are never reconciled; `warmup_batches=0` sets the anchor from random-initialization noise. Allocate 2 weeks to resolve.

**Path A (Honest Ablation):** If Path B fails, document that the LR adjustment contributes zero net benefit and the elastic penalty is the entire mechanism. Provide rigorous penalty-only ≥ full-TCL ablation.

**Pursue Path B first. Fall back to Path A after 2 weeks if the governor cannot be made to activate reliably.**

### 2.8 — Conduct honest scope delimitation for every frontier problem

The falsified frontier `fp-catastrophic-forgetting` (17 null, 4 adverse, 1 breakthrough) must be formally closed.

- Write a formal null-result record. Update all papers referencing this frontier to remove overclaiming language.
- Re-scope claims to what is actually supported.

**Phase 2 success criteria:**
- [ ] HPC replication with n=25 complete and honestly recorded
- [ ] Phase 17 rerun with 5 seeds, DER++, and LwF complete
- [ ] Phase 16 rerun with 5 seeds, DER++, and LwF complete
- [ ] Fair joint hyperparameter selection protocol run and locked
- [ ] Regime detector investigation resolved (Path A or Path B)
- [ ] Formal null-result for `fp-catastrophic-forgetting`

---

## PHASE 3 — MECHANISTIC CLARITY & THEORETICAL GROUNDING
**Duration:** 4–8 weeks (overlapping with Phase 2)
**Objective:** Build a coherent, defensible theoretical narrative that is predictive rather than post-hoc.

### 3.1 — Conduct the definitive mechanistic ablation

| Condition | Description | Hypothesis |
|---|---|---|
| SGD baseline | No mechanism | Worst forgetting |
| Penalty only (λ fixed) | Elastic regularization, no thermal component | Should match full TCL |
| Governor only (LR adjust) | Regime-detection LR adjustment, no penalty | Hypothesis: ≈ SGD |
| TCL full | Both components | Should match or exceed penalty-only |
| TCL with anchor frozen at init | Tests whether per-task sigma_star matters | Controls for anchor mechanism |
| TCL with warmup_batches=60 | Tests whether warmup affects governor activation | Tests Path B from 2.7 |
| EWC (best lambda) | Standard comparator | Provides external benchmark |

All conditions: n=5 seeds, Bonferroni-corrected, Split-CIFAR-10, pre-registered primary metric.

### 3.2 — Establish the theoretical connection between gradient-EMA importance and Fisher information

- Derive: `ThermalImportance.finalize()` is an online approximation to the diagonal Fisher.
- Show: when `ema_beta → 1`, TCL importance approximates EWC Fisher; when `ema_beta → 0`, it approximates the final-batch Fisher only.
- **Publishable theorem:** "TCL importance is a temporally-smoothed online estimate of the task-specific Fisher information, with EWC as a limiting case."

### 3.3 — Position the contribution correctly

- **Not:** "A thermodynamic theory of catastrophic forgetting"
- **Yes:** "Continuous gradient-energy EMA produces a richer importance estimate than a single-shot Fisher snapshot, leading to better forgetting-accuracy tradeoffs on standard CL benchmarks"

The thermodynamic framing can be preserved as motivating intuition only.

### 3.4 — Formally characterize HPC (High-Penalty Conservative)

Ablation: TCL with high lambda only vs TCL with no momentum only vs HPC (both) vs TCL baseline. 4-condition 5-seed study on Split-CIFAR-10. Pre-register primary comparison. If the effect is entirely from lambda, HPC is a hyperparameter finding, not a new mechanism.

### 3.5 — Build a real theoretical foundation for the participation ratio (D_PR)

Connect D_PR to known literature (Amsaleg 2015, Bartlett 2020, Ramasesh 2021). Verify empirically: plot D_PR vs forgetting across all seeds for all methods. Does lower D_PR predict higher forgetting?

**Phase 3 success criteria:**
- [ ] Full 7-condition mechanistic ablation complete with pre-registered analysis
- [ ] Formal Fisher-EMA connection derived and verified empirically
- [ ] HPC mechanism understood (lambda vs momentum)
- [ ] D_PR–forgetting relationship empirically tested
- [ ] Paper framing updated to match honest mechanistic picture

---

## PHASE 4 — CODE QUALITY & ALGORITHM CORRECTNESS
**Duration:** 3–4 weeks (overlapping with Phases 2–3)
**Objective:** Bring the algorithm codebase to the standard a PhD candidate would be comfortable sharing publicly.

### 4.1 — Fix the CI formula in all locations

All four locations must use the same t-distribution formula from Phase 1.1:
- `tar_lab/benchmark_stats.py`
- `phase8c_benchmark.py`
- `ewc_lambda_sweep.py`
- `tar_lab/multimodal_payloads.py`

### 4.2 — Fix TCL device placement edge cases

`ThermalMemory.penalty()` returns `torch.zeros(1, requires_grad=False)` without specifying a device. If the model is on CUDA, the tensor is on CPU, and the subsequent addition will either fail or silently copy data across devices.

- Fix: `torch.zeros(1, device=device, requires_grad=False)` wherever device-agnostic zeros are returned.
- Same fix in `ThermalCheckpoint.effective_importance()` (returns `torch.zeros(1)` without device).
- Add a unit test that runs the full penalty computation with a CUDA model and verifies no device mismatch.

### 4.3 — Fix the method synthesizer's baseline validation gap

`tar_lab/method_synthesizer.py` validates generated code only by running the sandbox without crashing. An incorrect method (e.g., zero penalty always) passes validation but produces invalid results.

Add a mandatory minibench validation step after sandbox success:
- Run the synthesized method on `split_cifar10` with `tiny_cnn` backbone, 3 tasks, seed [42], 2 epochs.
- Verify: forgetting in [0.0, 1.0], accuracy above random guessing (>0.2 for 5-class), no NaN or Inf.

### 4.4 — Fix the generative director's leading prompt

The current prompt frames standard methods as failures and always proposes novelty. Replace with:

1. A diagnosis step: "classify root cause: (a) hyperparameter mis-specification, (b) architecture limitation, (c) dataset characteristic, (d) algorithmic limitation."
2. A proposal step only if the diagnosis is (c) or (d).
3. A rejection path returning `{action: 'tune_hyperparameters'}` for (a) or (b).

### 4.5 — Fix the smoke-tier synthetic benchmark

The smoke-tier vision benchmark generates trivial geometric patterns (stripes, diagonal) as "images." These are plumbing tests, not evidence.

- Add `is_synthetic_smoke_bench: true` flag to all results from this benchmark.
- Treat as `trust_tier=smoke_only`; exclude from all publication-relevant evidence.
- Relabel from `"canonical_ready"` to `"smoke_only"` in `science_profiles.py`.

### 4.6 — Fix the silent benchmark downgrade

When a canonical-tier benchmark is unavailable the system silently downgrades without disclosure.

- Add `tier_requested` and `tier_executed` fields to every result JSON.
- Flag `tier_requested != tier_executed` as a publication-blocking issue.
- Log a WARNING-level alert when a downgrade occurs.

### 4.7 — Replace the keyword-based domain classifier

`tar_lab/science_profiles.py` uses brittle keyword scoring ("quantum" scores +5.0). The financial literature contamination traces back partly to this classifier routing arXiv papers into wrong domains.

- Build a training set of 50 labeled problem statements from existing frontier problems.
- Train a logistic regression classifier (or use Claude with a structured classification prompt).
- Report confidence as calibrated probability. Validate on held-out problems before deploying.

### 4.8 — Fill the knowledge graph

`knowledge_graph.json` has `"entries": []` despite 1,513 literature papers being ingested. The `ActiveLearner` daemon that should populate it has never been called by the orchestrator.

- Identify why `LiteratureBrain.start()` is never invoked in the operational loop.
- Wire `LiteratureBrain.start()` into orchestrator startup (call once at system init).
- Verify papers are ingesting via `brain.corpus_summary()` output.
- Minimum required content: 10–15 most relevant CL papers with key findings, explicit connections between TCL and EWC/SI/DER++/LwF, known failure modes of each method.

### 4.9 — Fix the invariant violations in TCL

- Add `assert penalty >= 0.0` (or equivalent runtime check with descriptive error) after the penalty loop in `ThermalMemory.penalty()`.
- Add a unit test verifying penalty ≥ 0 for edge cases: zero importance, zero drift, identical weights.

### 4.10 — Wire the self-improvement anchor pack initialization

`tar_state/self_improvement/` is empty. `SelfImprovementEngine.initialize_anchor_pack()` has never been called, so the improvement gate has no reference baseline and no adapter has ever been deployed.

- Run `initialize_anchor_pack()` with the best current evaluation results as the baseline.
- Store the anchor manifest in version control.
- Add a health check assertion: warn if anchor manifest is absent at startup.

### 4.11 — Fix the floating-point equality bug in self-improvement gate

`self_improvement.py` line 167: `if probe_overclaim_rate != 0.0` — floating-point equality comparison on a rate derived from division.

- Fix: `if probe_overclaim_rate > 1e-6`
- Add constant `_FLOAT_TOLERANCE = 1e-9` for all similar comparisons in this file.

### 4.12 — Implement power-analysis-based sample size selection in the pre-registration gate

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
- [ ] TCL device placement edge cases fixed with tests
- [ ] Method synthesizer has minibench validation
- [ ] Generative director prompt has diagnosis step and rejection path
- [ ] Smoke benchmarks flagged as non-evidence
- [ ] Benchmark tier downgrade disclosed in result records
- [ ] Domain classifier replaced with validated ML classifier
- [ ] Knowledge graph populated via wired ActiveLearner
- [ ] TCL invariant assertion added and tested
- [ ] Self-improvement anchor pack initialized
- [ ] Float equality bug in self-improvement gate fixed
- [ ] Pre-registration includes power-analysis-driven n calculation

---

## PHASE 5 — PAPER PIPELINE
**Duration:** 8–12 weeks (begins after Phase 0 and Phase 1 complete)
**Objective:** Produce at minimum one real, defensible, submission-ready manuscript.

### 5.1 — Identify the single strongest claim and build one paper

The Phase 10 result (p=0.019) does not survive Bonferroni correction. Path forward depends on Phase 2:
- If Phase 2 replications show p<0.01 on CIFAR-10 and TinyImageNet: proceed to full paper.
- If p<0.05 on both but not consistently <0.01: submit to ContinualAI workshop at NeurIPS 2026 with transparent limitations.
- If HPC replication succeeds: a separate, stronger paper on the high-lambda no-momentum finding.

### 5.2 — Write the paper structure

1. **Abstract** (150 words): Problem, approach, key result, implication.
2. **Introduction**: Catastrophic forgetting as a deployment problem. EWC/SI/replay limitations. The proposal: continuous gradient-energy accumulation.
3. **Related Work**: EWC (Kirkpatrick 2017), SI (Zenke 2017), LwF (Li 2016), DER++ (Buzzega 2020), PackNet, Progressive Neural Networks.
4. **Method**: ThermalImportance algorithm. ThermalMemory ring buffer. TCLRegularizer penalty. Connection to EWC as a limiting case. Complexity analysis.
5. **Experiments**: Phase 10 (CIFAR-10, 5 seeds), Phase 16 rerun (CIFAR-100, 5 seeds), Phase 17 rerun (TinyImageNet, 5 seeds), Phase 11 ablation (governor vs penalty), HPC ablation (lambda vs momentum).
6. **Analysis**: D_PR as forgetting predictor. Power analysis table. Effect sizes with corrected CIs.
7. **Limitations**: Regime detector non-activation. Task-incremental only. Not class-incremental. Single backbone.
8. **Conclusion**: Honest statement of what was shown and what was not.

### 5.3 — Enable tar_author.py to draft real sections

- Verify the prompt structure in `tar_author.py` for each section renderer.
- Ensure prompts provide: exact numerical results from the evidence inventory, a pre-written outline, constraints on length and citation format.
- Add LLM call cost tracking to `tar_author.py` (currently absent — Sonnet-4-6 at 4096 tokens per section call, across 8 sections, on multiple paper attempts, accumulates significant cost with no budget gate).
- Parameterize all prompts: remove hardcoded TCL-specific numbers and CL-specific references so the authoring pipeline generalises to future papers.
- **Human review every output before accepting — the author is a drafting tool, not a final authority.**

### 5.4 — Build the related work section using the knowledge graph

Standard NeurIPS CL comparison table:

| Method | Split-CIFAR-10 Forgetting | Split-CIFAR-100 Forgetting | Source |
|---|---|---|---|
| SGD baseline | 0.233 ± 0.003 | — | Our Phase 10 |
| EWC (λ=1000) | 0.160 ± 0.020 | — | Our Phase 12 |
| SI (c=0.01) | 0.092 ± 0.001 | — | Our Phase 13 |
| LwF | — | — | To be run |
| DER++ (mem=200) | — | — | To be run |
| TCL (ours) | 0.116 ± 0.029 | — | Our Phase 10 |

### 5.5 — Target venue and submission timeline

- **HPC replication succeeds + TinyImageNet/CIFAR-100 at 5 seeds:** Target ICLR 2027 (abstract deadline ~September 2026). Achievable if Phase 2 completes by August 2026.
- **Results solid but marginal:** Target ContinualAI workshop at NeurIPS 2026 (typically September deadline).
- **Do not submit to arXiv until Phase 2 replication results are in.**

**Phase 5 success criteria:**
- [ ] LaTeX manuscript with real prose in all 8 sections
- [ ] Results table with honest effect sizes and corrected CIs
- [ ] Related work table with published numbers from literature
- [ ] Limitations section addresses regime non-activation and scope
- [ ] Compilation succeeds and produces a real PDF
- [ ] All claims traceable to the honest evidence inventory

---

## PHASE 6 — GOVERNANCE, SAFETY & LONG-TERM AUTONOMY
**Duration:** 4–6 weeks (parallel with Phases 3–5)
**Objective:** Bring the autonomous research infrastructure to the standard where it can safely be trusted to run unattended without producing false science.

### 6.1 — Cryptographically seal governance state files

- Generate an Ed25519 keypair at system setup, stored in a protected location.
- When any governance state file is written, append `{content_hash: SHA256(content), signature: Ed25519(content_hash)}`.
- On read: verify signature before trusting content. On failure: fail-closed.
- Private key must NOT be accessible to experiment sandbox processes.

### 6.2 — Enforce execution_policy.json at code generation time

`tar_state/policies/execution_policy.json` declares `"require_sandbox_for_generated_code": true` but the `ExecutionPolicyViolation` error class exists without any enforcement code raising it.

- Add `assert_policy_allows_generated_code_execution(workspace)` call in `method_synthesizer.py` before any generated code runs.
- If policy file is unreadable: fail-closed.
- Log every policy check to the alert system.

### 6.3 — Make Docker non-bypassable for experiment execution

The current `allow_host_fallback` path allows uncontrolled code to run on the host when Docker is unavailable.

- Remove this fallback path for experiment execution.
- If Docker is unavailable: `ExecutionPolicyViolation("Docker required; subprocess fallback disabled.")`.
- Allow host execution only for non-experiment operations.

### 6.4 — Build a tamper-proof audit log

The current alert ring buffer (500 entries) loses history.

- Implement append-only JSONL audit log at `tar_state/audit.jsonl`.
- Every manifest authorization, verdict generation, governance file write, experiment start/stop appended with timestamp, PID, and SHA256 of state file at that moment.
- Never trimmed. Gzip-compressed monthly.

### 6.5 — Close the autonomous mode human-review bypass

`human_review.py` has `_UniversalApproval` — a sentinel that approves all experiments in autonomous mode, bypassing the human review layer entirely.

- In autonomous mode: Director proposals enter `pending_director_review` queue in `human_review_state.json`.
- New `auto_approve_after_hours` setting (default: 24 hours) allows experiments to proceed if no veto is received.
- Operator notification: "3 experiments will start in 23 hours unless vetoed."

### 6.6 — Implement per-session API cost budget with hard stop

`tar_author.py` makes multiple Sonnet-4-6 calls with 4096 max tokens, with no cost tracking or budget gate.

- Add `session_budget_usd` to the daily cost tracker in `model_router.py`.
- If cumulative cost exceeds budget: degrade all calls to Haiku and raise an alert.
- If cost reaches 2× budget: block all LLM calls and notify operator.
- Add per-section cost attribution to `tar_author.py`.

### 6.7 — Establish a recurring integrity check routine

Build `tar_health_check.py` to run on startup, daily, and on demand via `tar_cli.py --health-check`.

Checks:
- All governance file signatures valid
- All canonical comparison results pass the three-gate registry check
- All evidence in the honest evidence inventory matches raw comparison JSON files
- `active_session.json` reflects actual running state
- GPU temperature within safe range
- API key reachable (ping with minimal tokens)
- No orphan processes
- No stale leases (>24 hours old)
- All JSONL files within size thresholds (alert if >50 MB)

Output: `tar_state/health_report.json`.

**Phase 6 success criteria:**
- [ ] Ed25519 signatures on all governance files
- [ ] `ExecutionPolicyViolation` raised when policy requires sandbox
- [ ] Docker fallback for experiments eliminated
- [ ] Append-only audit log implemented
- [ ] Auto-approve-after-24h veto window replaces `_UniversalApproval`
- [ ] Session API cost budget enforced with per-author attribution
- [ ] Health check routine operational and running daily

---

## PHASE 7 — ENGINEERING INFRASTRUCTURE
**Duration:** 4–6 weeks (parallel with Phases 3–5)
**Objective:** Bring the operational infrastructure from development-grade to a standard that supports reliable autonomous long-term operation.

### 7.1 — Add GitHub Actions CI/CD pipeline

There is currently no automated test runner. Tests can only be run manually. Code can be pushed that breaks 482 tests with no detection.

Create `.github/workflows/ci.yml`:
```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements-dev.txt
      - run: pytest --tb=short -q
      - run: detect-secrets scan --baseline .secrets.baseline
```

- Block merges to `main` unless CI passes.
- Add coverage reporting with threshold: fail if coverage falls below 70%.
- **Rationale:** 482 tests provide zero protection without an automated runner.

### 7.2 — Pin all dependency versions and generate a lock file

`requirements.txt` contains no version pins. Every `pip install` resolves to the latest version at that moment, making the system non-reproducible.

- Pin every direct dependency to an exact version: `torch==2.3.1`, `transformers==4.41.2`, etc.
- Generate `requirements.lock` via `pip freeze` after confirming all tests pass.
- Commit the lock file. `requirements.txt` becomes the intent; `requirements.lock` is the ground truth.
- Remove unused packages: three PDF parsing libraries (PyMuPDF, pdfplumber, pypdf are all present — choose one primary), and quantum/RL packages that are barely used.
- Add `extras_require` grouping: `requirements-dev.txt` for test/lint tools, `requirements-research.txt` for heavy ML deps, `requirements-core.txt` for daemon operation.

### 7.3 — Pin the Docker base image digest

`Dockerfile.experiment` uses `FROM python:3.13.3-slim` without a digest. Docker registries can re-tag, making builds non-reproducible.

- Add: `FROM python:3.13.3-slim@sha256:<digest>`
- Add a post-build health check: `RUN python -c "import torch; print(torch.__version__)"`
- Generate and commit a Software Bill of Materials (SBOM) for the image.
- Add a container health check to `docker_runner.py` that verifies key imports before accepting the container as ready.

### 7.4 — Replace the Flask development server with a production WSGI server

`tar_dashboard.py` runs Flask's built-in development server — single-threaded, explicitly documented as "do not use in production," and not designed for concurrent requests or long-running connections.

- Install Gunicorn: `pip install gunicorn`
- Replace `flask run` with `gunicorn --workers 2 --bind 127.0.0.1:7860 tar_dashboard:app`
- Add a `SIGTERM` handler so in-flight requests complete before shutdown.
- Add `/readiness` and `/liveness` endpoints for health-check use.

### 7.5 — Add fsync to all critical JSONL writes

`state.py`'s `_safe_jsonl_append()` calls `handle.flush()` but not `os.fsync()`. On Linux and Windows, `flush()` pushes data to the OS buffer but does not guarantee disk write. `canonical_registry.py` correctly calls `os.fsync()` after its critical appends; all other JSONL writes should match.

- Add `os.fsync(handle.fileno())` after every `handle.flush()` in the JSONL append path.
- Confirm that the existing lockfile-based append in `result_artifacts.py` also has this fsync.
- Add a comment noting why fsync is required here (NTFS/ext4 page cache and power-loss durability).

### 7.6 — Implement JSONL rotation and archival

11 JSONL files grow without bound. `research_intel.jsonl` is already 7.97 MB after ~6 months. At current growth rates several will exceed 50 MB within a year.

- Implement monthly rotation: at the start of each month, rename `foo.jsonl` to `foo_YYYY-MM.jsonl.gz` (gzip compressed) and start a fresh `foo.jsonl`.
- The reader (`_iter_jsonl()`) should transparently read both current and rotated archives.
- Add a size threshold alert (warn if any JSONL exceeds 50 MB unrotated).
- Add this check to the health check routine from Phase 6.7.

### 7.7 — Implement schema versioning and a migration framework

No JSON state file carries a version field that is checked on read. Renaming or removing a Pydantic field causes silent data loss — the old file is parsed, the field is absent, the default is used, and no warning is logged.

- Add `_schema_version: str` field to every Pydantic model that maps to a persisted file.
- On load: compare file's `_schema_version` against the current model's version. If they differ, run the registered migration function.
- Implement a migration registry: `{("v1", "v2"): migrate_v1_to_v2}` per state file.
- Write migrations for all schema changes made since the first committed version.
- Log a WARNING whenever a migration runs so it is auditable.

### 7.8 — Back up ChromaDB and enforce a backup schedule

The vector memory store at `tar_state/memory/` (ChromaDB embedded database) is not version-controlled, not synced to FTP, and not backed up anywhere. If the E: drive fails, all literature embeddings are permanently lost.

- Add `tar_state/memory/` to the FTP sync in `sync_research.py` (or better: replace FTP with S3/SFTP from task 7.9).
- Run a weekly backup job that tarballs `tar_state/memory/` and stores it to a secondary location.
- Add a health check assertion that the backup is not more than 8 days old.

### 7.9 — Replace plaintext FTP publishing with S3 or SFTP

`sync_research.py` uses cleartext FTP (credentials in plaintext, data in plaintext, no verification after upload, no retry logic, blocks main thread on slow connections).

- Replace with `boto3` (S3) or `paramiko` (SFTP).
- Move credentials to environment variables.
- Implement exponential backoff retry (up to 3 attempts, 5s/25s/125s).
- After every upload, verify the remote object checksum matches the local file.
- Run uploads in a background thread so the main state-polling loop is never blocked.

### 7.10 — Add tar_api.py authentication enforcement and input validation

The API requires an API key only if `TAR_API_KEY` is set in the environment. If not set, the entire API is unauthenticated.

- Make the API key mandatory: if absent, raise a configuration error at startup.
- Validate all path parameters: `project_id` and similar must match `^[a-zA-Z0-9_-]{1,100}$`.
- Sanitize error responses: do not expose backend command names or internal exception messages verbatim.
- Add structured logging of all API requests (method, path, status code, latency).

### 7.11 — Refactor tar_author.py from god-class into modules

`tar_author.py` is 6,437 lines. It combines evidence collection, LaTeX section generation, citation handling, bibliography management, originality auditing, compilation orchestration, and LLM prompt management in a single file. This makes it hard to test, hard to understand, and impossible to reuse components independently.

Split into:
- `tar_author_engine.py` — orchestration and evidence loading (~500 lines)
- `tar_author_sections.py` — per-section LaTeX generation with domain-agnostic prompts (~1,500 lines)
- `tar_author_citations.py` — citation validation, bibliography, deduplication (~700 lines)
- `tar_author_compiler.py` — LaTeX compilation, PDF generation, error handling (~400 lines)

Each module should be independently importable and independently testable.

**Phase 7 success criteria:**
- [ ] GitHub Actions CI running on every push; merges blocked unless passing
- [ ] `requirements.lock` committed with exact version pins
- [ ] Docker base image pinned to digest
- [ ] Flask dev server replaced with Gunicorn
- [ ] `os.fsync()` added to all JSONL append paths
- [ ] JSONL rotation implemented with monthly archival
- [ ] Schema version field on all persisted models; migration framework in place
- [ ] ChromaDB backup schedule running
- [ ] FTP replaced with S3/SFTP with retry and verification
- [ ] tar_api.py authentication mandatory; input validation in place
- [ ] tar_author.py split into four focused modules

---

## PART III — TIMELINE & DEPENDENCY MAP

```
DAY   1
      ├─ Phase E0: Emergency Security (1 day — FIRST)
      
WEEK  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
      ├───Phase 0: Triage + Engineering Triage (2w)────┤
                    ├───Phase 1: Statistical Foundations (3w)────┤
                    ├───Phase 4: Code Quality (4w)───────────────────┤
                          ├───Phase 2: Experimental Rigour (8w)──────────────────────┤
                          ├───Phase 3: Mechanistic Clarity (8w)──────────────────────┤
                          ├───Phase 6: Governance (6w)─────────────────────┤
                          ├───Phase 7: Engineering Infrastructure (6w)──────────────────┤
                                    ├───Phase 5: Paper Pipeline (12w)────────────────────────────────┤
```

**Critical path:** Phase E0 → Phase 0 → Phase 1 → Phase 2 (HPC replication) → Phase 5 (paper)

**Parallelizable:** Phases 3, 4, 6, and 7 can all run while Phase 2 experiments execute.

**Engineering critical path (independent):** Phase E0 → Phase 7.1 (CI) → Phase 7.2 (dependency pins) → all subsequent engineering tasks are unblocked.

---

## PART IV — WHAT WOULD A PhD EXAMINER ACCEPT

After all phases complete, the system should be able to defend:

**Defensible claim 1 (mechanism):**
> "Continuous gradient-energy EMA (ThermalImportance) produces a richer importance estimate than the EWC single-shot Fisher snapshot. On Split-CIFAR-10 (n=5 seeds, Holm-corrected p<0.025), this reduces catastrophic forgetting by [X] percentage points while maintaining equivalent accuracy. The effect generalizes to Split-CIFAR-100 and Split-TinyImageNet (n=5 seeds each)."

**Defensible claim 2 (HPC variant, if replication succeeds):**
> "A high-penalty conservative variant of CGEIA, with elevated λ and SGD momentum disabled, reduces forgetting by ~50% compared to baseline CGEIA. The mechanism is primarily attributable to [lambda / momentum removal / both — resolved in Phase 3.4]."

**Honest limitations section:**
- The thermodynamic regime-detection LR adjustment does not contribute to performance improvement (Phase 11 ablation; governor-alone < SGD).
- Results restricted to task-incremental scenarios with task IDs at test time. Class-incremental is untested.
- With DER++ included, TCL [beats / does not beat] replay-based methods — the honest result from Phase 2.4.
- Sample sizes remain below what would be required to claim that null results are true negatives.

---

## PART V — WHAT MUST NEVER HAPPEN AGAIN

**Eight failure modes this programme prevents:**

1. **Running experiments that cannot reach publication tier.** Every experiment now requires a pre-registered power analysis before queueing. If `n_required > n_affordable`, the experiment is marked `exploration_grade` and cannot be cited as evidence in a paper.

2. **Falsely attributing results to a mechanism that doesn't work.** The mechanistic ablation (Phase 3.1) and the governor activation investigation (Phase 2.7) ensure that the paper's mechanism section describes what actually happens in the code.

3. **Post-hoc hyperparameter tuning of baselines.** Joint tuning on a held-out validation split (Phase 2.6) eliminates the EWC lambda=100→1000 retroactive adjustment.

4. **Autonomous operation from unreviewed code.** The six uncommitted governance files (Phase 0.2) are committed and reviewed before the next autonomous cycle begins.

5. **Calling underpowered results breakthroughs.** The honest evidence inventory (Phase 0.5) and Bonferroni correction (Phase 1.2) mean that the system's internal claim status reflects what a reviewer would accept.

6. **Credentials committed to a public repository.** Phase E0 rotates all exposed keys, removes them from git history, adds `.gitignore` protection, and installs `detect-secrets` as a pre-commit hook. This class of error is structurally prevented going forward.

7. **Infrastructure failures silently contaminating results.** Phase 7 adds CI/CD (broken tests are caught before merge), pinned dependencies (the environment is reproducible), fsync on all writes (data is durable), and schema versioning (field renames do not cause silent data loss).

8. **Subsystems designed but never activated running as dead code.** The ActiveLearner and self-improvement pipeline are wired to the orchestrator startup (Phase 4.8 and 4.10). The knowledge graph will be populated. The anchor pack will be initialized. Future development must verify that each new subsystem has a call path from the operational loop before it is considered complete.

---

*This programme gives TAR a clear, executable path from its current state — a 5-seed CIFAR-10 comparison that is not Bonferroni-significant, running from unreviewed daemon code with live API keys committed to GitHub — to a system capable of producing and defending a legitimate, peer-reviewed contribution while operating reliably and securely over the long term.*
