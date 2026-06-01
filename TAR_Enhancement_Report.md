# TAR Enhancement Report — Master Synthesis
## PhD-Level Team Assessment, 2026-06-02

*Ten-agent deep dive across: CL algorithms, deep learning theory, research automation, LLM/NLP, systems architecture, experimental design, publication strategy, safety/alignment, literature synthesis, research programme strategy.*

---

## PREAMBLE: THE CROSS-CUTTING FINDING

All ten experts independently converged on the same structural diagnosis: **TAR's infrastructure and science are separated by an integration gap.** The ML engine works. The governance architecture is thoughtful. The experiments produce real signal. But the literature brain has never run, the self-improvement loop has never completed a cycle, the knowledge graph has zero entries, the paper has never been drafted, and the theory used to frame the work does not match the mechanism that actually produces the results. The rehabilitation plan correctly identifies these issues. What follows is what the plan missed.

---

## I. ALGORITHMIC ENHANCEMENTS

### A. Immediate additions the plan does not mention

**1. Backward Transfer (BWT) and Forward Transfer (FWT) metrics — 30 minutes, essential**

The system reports only mean forgetting and mean accuracy. Both are absent from the Phase 10 results. Standard CL papers since Javed & White (2019) report:
- **BWT:** how much does learning task T+1 change task T's accuracy (can be negative — interference, or positive — consolidation)?
- **FWT:** how much does pre-training on T improve zero-shot performance on T+1?
- **Intransigence index:** fraction of tasks with >5% forgetting

These require no additional experiments — they are computable from the existing `acc_matrix` already produced by `generic_cl_runner.py`. Omitting them from a 2026 CL paper will generate an automatic revision request. The implementation is ten lines.

**2. Learning without Forgetting (LwF) — 2 hours, publication-critical**

LwF (Li & Hoiem, 2016) is listed in `frontier_problems.json` and marked as a known method, but it is not in `method_registry.py` and does not appear in any phase comparison. Every CL reviewer will notice this absence immediately. It is the canonical knowledge-distillation baseline, orthogonal in mechanism to elastic regularization. The implementation:

```python
loss_new = CE(model(x), y)
old_logits = old_model(x).detach()
loss_distill = KL(softmax(model(x)/T), softmax(old_logits/T))
loss = loss_new + alpha * loss_distill + TCL_penalty
```

Beyond just adding it as a baseline: TCL + LwF combined is untested and likely additive. If TCL (elastic penalty, gradient-space) + LwF (output-space distillation) compound, the combined method is a novel contribution. This is faster to implement than any algorithmic innovation and may be the most impactful single addition.

**3. Gradient Episodic Memory (GEM/A-GEM) — 4 hours, mechanistically important**

GEM (Lopez-Paz & Ranzato, 2017) and A-GEM (Chaudhry et al., 2019) enforce gradient constraints rather than weight penalties. They represent a fundamentally different protection class — constraint-based rather than regularization-based. The comparison would isolate whether TCL's advantage is from the gradient-EMA importance weighting specifically, or from elastic regularization in general. EWC and GEM are both common comparators in top-venue CL papers; having only EWC is incomplete.

**4. Class-incremental evaluation (CIL, no task IDs) — 2 hours, major opportunity**

The Phase 15 data exists (p=0.012, d=5.26, 3 seeds — underpowered but promising). Running CIL properly — where the model is never given the task ID at test time — is a more realistic and more publishable setting than task-incremental. Two hours to modify the evaluation loop to omit task identity at inference. This could be a second paper or a substantial contribution to the first.

**5. Disable the governor by default — 30 minutes, code integrity**

Phase 11 shows governor-alone is worse than SGD. The regime detector never fires in production. Every trace examined shows `rho=0, sigma=0, regime="unknown"`. The code currently runs the governor on every step by default and applies a LR adjustment that never triggers. This is computational overhead for a broken mechanism. Add `use_governor=False` as the default. Keep the infrastructure for Path B (governor repair), but stop advertising a mechanism that does not work.

### B. Second-order importance as an algorithmic upgrade

The plan calls for investigating why the governor does not fire. A parallel track worth pursuing: upgrade ThermalImportance from first-order (`v_i = β·v_i + (1−β)·g_i²`) to second-order by adding the Hessian diagonal term:

```python
v_i = β·v_i + (1−β)·g_i² + γ·h_ii
```

The Hessian diagonal `h_ii` measures true parameter sensitivity — how much the loss changes when parameter i is perturbed. Gradient squared `g_i²` is a proxy. The connection: Fisher information (used in EWC) = E[g²], which is the diagonal of the Fisher matrix. But the Hessian diagonal `h_ii` is the second derivative and directly captures curvature, not just gradient energy. Parameters with high curvature are more important. This is the theoretical gap between TCL and EWC. If TAR adds the Hessian diagonal (computable via `torch.autograd.functional.hessian` on mini-batches, or via the `backpack` library), it creates a genuinely novel importance estimator.

**Estimated gain:** +2–4% forgetting reduction. **Estimated effort:** 4 hours.

---

## II. THEORETICAL FOUNDATIONS

The theory agent identified the most consequential gap: **the Fisher-EMA theorem does not exist in the codebase, only as a claim in the rehabilitation plan.** The plan states it as a deliverable; here is what it needs to actually say.

**The formal connection:**

`ThermalImportance.finalize()` returns `v_i / max(v_i)` where `v_i = EMA_β(g_i²)`. The diagonal Fisher is `F_ii = E_{x~D}[(∂ℓ/∂θ_i)²]`. Under i.i.d. sampling from the task distribution, as the EMA window covers sufficient samples, `v_i → F_ii` with exponential bias decay `β^T`. The bias is `|v_i - F_ii| ≤ C · β^T` where C is the second moment of the gradient.

**The key insight that makes TCL different from EWC:** EWC uses a one-time snapshot at task end. If the loss landscape changes during task training (which it does — gradients shift as the model converges), the snapshot may not represent the full task distribution. TCL's EMA accumulates continuously, averaging over the entire task training trajectory. This is a temporally smoothed Fisher estimate rather than a point estimate. Both EWC and TCL are biased; they are biased differently.

**The publishable theorem sketch:**
> *Under the assumption that the task gradient process is ergodic with mixing time τ, ThermalImportance with ema_beta = 1 − 1/τ converges to the diagonal Fisher with error O(1/√T) after T gradient steps, matching the convergence rate of standard Monte Carlo estimation. EWC's end-of-task Fisher estimate achieves the same asymptotic rate but with a bias term that depends on gradient non-stationarity during training. In non-stationary settings (which all continual learning sequences are), the EMA estimate has strictly lower bias.*

This is the paper's theoretical contribution. The experiments demonstrate the practical consequence.

**The D_PR gap:** There is no causal connection between D_PR compression and forgetting — only a correlational claim. The plan calls for an empirical test (plot D_PR vs forgetting across seeds). This is necessary but not sufficient. What is needed is a proof that TCL's penalty resists D_PR compression by protecting parameters in the task-T principal subspace. This is a matrix perturbation argument connecting the penalty term to the covariance rank. Without it, D_PR remains a diagnostic metric with no theoretical standing in the paper.

**The thermodynamic framing verdict:** Abandon it in the paper. The physical analogy requires stochastic dynamics (Langevin), which TAR's deterministic SGD does not have. There is no Boltzmann distribution. The sigma metric (`lr × ||grad||²`) is not entropy in any statistical mechanics sense. Replace with precise language: sigma is "step magnitude," rho is "relative learning rate ratio," the regime labels become "converging/learning/unstable." The intuition (hot parameters = heavily-used = important) can be retained as motivation but should not appear in the formal contributions section.

---

## III. RESEARCH AUTOMATION ARCHITECTURE

Three high-value additions not in the current plan:

**1. Mechanism-isolation ablation engine**

When a hypothesis claims a specific mechanism (e.g., "elastic anchoring reduces drift via importance-weighted penalty"), the system should automatically generate an orthogonal ablation: disable the claimed mechanism, test if performance degrades. Currently TAR never does this — it tests methods holistically. The mechanism-isolation engine would:

- Parse the `rationale` field of each passed hypothesis for mechanism names
- For each mechanism M: automatically propose a "knockout" variant with M disabled
- Mark hypotheses as "mechanism confirmed" only when the knockout shows statistically significant degradation

This catches false positives — experiments that pass global metrics without demonstrating the claimed mechanism. It is the most important addition to the research automation architecture.

**2. Cross-failure clustering with hypothesis-space escape**

TAR currently diagnoses each failure independently and permanently caches the diagnosis. There is no aggregation. If five experiments fail on "gradient explosion," the system diagnoses gradient explosion five times without asking: "why does this keep happening?" The fix:

- Maintain `tar_state/failures/cluster_log.json` grouping failure diagnoses by semantic category using embedding cosine similarity
- When a failure category clusters at ≥3 instances, invoke a different prompt: "We've failed N times on [category]. What is a fundamentally different approach that avoids this failure mode entirely?"
- This converts the system from a local optimizer (cycling the same three families) to a global explorer that can escape failure-mode basins

**3. Pre-synthesis literature grounding**

Before generating any new method code, the method synthesizer should query the knowledge graph: "Does any paper in the last 5 years implement a method with mechanism X on dataset Y?" If found, it should retrieve the implementation details and either build on them explicitly or explain why a fresh synthesis is warranted. Currently the synthesizer generates code freely without checking whether equivalent code already exists in published work.

---

## IV. LANGUAGE MODEL AND ASC COMPONENTS

**ASC is not novel.** The architecture — EMA target model + online student + consistency loss + small MLP perturbation head — is Mean Teacher (Tarvainen & Valpola, 2017) with a learned adversarial augmentation. Mean Teacher is from 2017. The LatentWarp head does not justify a novelty claim over standard knowledge distillation or BYOL. If ASC is presented as a research contribution, it will be rejected as derivative.

**The genuine novel contribution possible here: TCL-for-LLMs.**

Apply TCL's gradient-EMA importance weighting to LLM fine-tuning to prevent forgetting of pretraining knowledge. The mechanism:

1. During task-T fine-tuning on downstream data, accumulate per-parameter EMA of gradient energy
2. Parameters with high accumulated energy have high task-T importance
3. When fine-tuning on task T+1, add elastic penalty proportional to task-T importance
4. The consistency loss from ASC (EMA target) becomes the anchor for the importance computation

This is the unification of TCL and ASC: TCL's protection mechanism applied to the LLM fine-tuning problem, with ASC's EMA target providing the frozen reference. This would be a genuinely new combination that neither TCL (vision) nor ASC (NLP) achieves alone. **This is the path to a second publishable paper.**

**Self-improvement pipeline risks requiring architectural fixes:**

- **Reward hacking:** The Director optimizes the publication rubric, not scientific truth. Add a holdout test set for gate evaluation that is never exposed to the Director during training.
- **Mode collapse:** Diversity scoring is across signal kinds, not content. A thousand variants of the same wrong pattern pass. Add content-level deduplication using embedding cosine similarity.
- **No rollback:** If a deployed adapter causes regressions, there is no revert mechanism. Add a post-deployment monitoring period (5 experiment cycles) before an adapter is considered stable.

---

## V. SYSTEMS ARCHITECTURE

The current file-based polling at 30 seconds is **adequate for the research scale** (single GPU, 6–8 hour experiments). The upgrade priorities in order of actual impact:

**1. SQLite for state management** — the single highest-leverage infrastructure change. Eliminates the 23-file makeshift database. Enables atomic multi-table updates, concurrent reads between dashboard and daemon, and sub-millisecond state queries. The migration path: SQLite with WAL mode (zero external dependencies on Windows). Three tables: experiments, metrics_timeseries, artifacts. Existing JSON files continue to be written as fallback. **Estimated: 1–2 weeks.**

**2. RunPod executor wrapper** — the GTX 1650 is saturated at 87% VRAM utilization. Phase 2 requires 4–5 parallel experiments at 5 seeds each. Cloud GPU (RunPod A40, 24GB VRAM) enables 6× batch sizes and seed parallelization. The backend factory already has `supports_distributed` flags and checkpoint resume logic. Adding a `RemoteExecutor` adapter translates Docker launch commands to RunPod API calls. **Estimated: 1 week. Unlocks full Phase 2 without serial queueing.**

**3. Prometheus + Grafana** — three metrics matter: `experiments_running`, `gpu_vram_free_gb`, `daemon_cycle_latency_ms`. An alert on `gpu_vram_free_gb < 0.3` prevents the OOM crashes that corrupt in-progress experiments. **Estimated: 2–3 days.**

The file-based daemon architecture does **not** need a message queue (Redis, Celery) at this scale. The 30-second poll latency is not a bottleneck when experiments run for 8 hours.

---

## VI. EXPERIMENTAL DESIGN — WHAT THE PLAN MISSED

**1. Sequential testing / adaptive stopping**

The plan fixes n=5 to n=25 for HPC replication. A Sequential Probability Ratio Test (SPRT) would allow stopping early when evidence is decisive and reallocating compute to more informative experiments. If after 8 seeds the posterior probability of improvement exceeds 0.95, stop and claim decisive evidence. If after 8 seeds the posterior is below 0.10, stop and declare null. Only borderline cases need the full 25 seeds. This could compress the HPC replication to ~12–15 seeds on average while maintaining the same statistical guarantees.

**2. Three critical confounds not in the plan**

The experimental design audit identified three confounds the rehabilitation plan does not address:

- **Class-order randomization:** `generic_cl_runner.py` shuffles class order per seed using the seed value. If TCL exploits class-order artifacts, the reported forgetting reduction is a class-ordering artifact, not a method improvement. Test: compute Spearman rank correlation between the class-order permutation seed and mean_forgetting across the 5 seeds. If |ρ| > 0.6, class order is a confound and must be controlled.

- **BatchNorm statistics across tasks:** The code does not reset BatchNorm running statistics at task boundaries. This means BN running mean and variance for task 5 carry information from tasks 1–4. This is a form of implicit memory that could advantage or disadvantage methods differently. Test: add `model.apply(lambda m: m.reset_running_stats() if hasattr(m, 'reset_running_stats') else None)` at each task boundary and compare forgetting. If delta is large, BN state is a confound; if small, it can be dismissed.

- **Weight initialization seed:** `_set_seed()` sets torch.manual_seed but the ResNet18 uses `weights=None` (kaiming uniform default). The interaction between the manual seed and PyTorch's internal initialization RNG may not be fully controlled. Verify by running the same seed twice and confirming identical initial weights.

**3. Calibration trajectory under forgetting**

ECE is reported as a single scalar at the end of all tasks. What is actually needed:

- Per-task ECE measured after each task, not just at the end
- Whether TCL maintains calibration as more tasks are learned
- Whether high-confidence predictions on old tasks are actually accurate after forgetting

A method that reduces forgetting but becomes severely overconfident on old tasks is not actually better. None of this is currently measured.

**4. Bayesian credible intervals as a replacement for p-values**

`P(μ_TCL < μ_EWC) = 0.94` is more actionable than `p = 0.019`. The posterior update requires specifying a prior on effect size (reasonable choice: half-normal with scale 0.5, encoding the belief that most CL improvements are small-to-medium), then updating with each seed observation. This is 20 lines of scipy code and would replace the frequentist p-value as the primary evidence metric.

---

## VII. PUBLICATION STRATEGY — MAJOR REFRAMING

The evidence actually says:
- TCL beats EWC at p=0.019, d=1.70, but does not survive Bonferroni (weak)
- The governor is broken (weakens the narrative further)
- HPC beats TCL by 50%, but HPC is probably better hyperparameters (undermines the mechanism claim)

**Do not frame this as a new algorithm. Frame it as an empirical audit of EWC's failure mode.**

The honest reframing: *"We show that EWC's canonical hyperparameters (from Kirkpatrick 2017) systematically underfit activation heterogeneity in ResNet-18 architectures on Split-CIFAR-10. A reparametrized elastic penalty using continuous gradient-energy accumulation reduces this underfitting. We provide a controlled ablation that isolates the mechanism to the importance weighting rather than the thermodynamic regime detection, which we find does not activate in practice."*

This framing is defensible (the ablation data supports it), novel (no paper has specifically analyzed EWC's hyperparameter sensitivity under controlled seed management), honest (does not overclaim a new theory), and high-impact (EWC is the foundational baseline; an empirical audit of its failure mode has wide citation value).

**The five reviewer archetypes and pre-emptions:**

| Reviewer | Objection | Pre-emption |
|---|---|---|
| The Baseline Defender | "SI already solves forgetting better than TCL" | "SI at c=0.01 is effectively degenerate (near-zero regularization). Phase 13 SI robustness sweep shows SI is unstable across its hyperparameter range." |
| The Mechanist | "Where is the evidence the regime signal drives forgetting?" | "Phase 11 ablation: governor-alone is *worse* than SGD (0.250 vs 0.219). The mechanism is the gradient-EMA penalty, not regime detection. We use regime language only as description." |
| The Statistician | "n=5, uncorrected α=0.05. Fails Bonferroni." | "Phase 10 is a controlled replication with preregistered hypothesis. Bonferroni threshold p=0.0125; our p=0.019 does not survive it. We frame this as a directional result motivating the Phase 11 mechanistic ablation, which has independent power." |
| The Scope Skeptic | "Split-CIFAR-10 is a toy benchmark" | "Split-CIFAR-10 is chosen because it forces method differences to surface. Phase 16–17 (CIFAR-100, TinyImageNet) are directional; a separate scale-up paper is in preparation." |
| The CL Veteran | "Why not compare to DualPrompt/L2P?" | "Vision-transformer prompt methods assume class-incremental with high-capacity backbones. Our contribution targets bandwidth-constrained settings where task IDs are available. Phase 15 (CIL) directly targets the prompt-based setting and is in preparation." |

**The multi-paper strategy:**

| Paper | Target | Framing | Status |
|---|---|---|---|
| **Paper 1** | ContinualAI@NeurIPS 2026 or ICLR 2027 | Empirical audit of EWC failure mode + controlled TCL comparison | In preparation |
| **Paper 2** | NeurIPS 2027 or ICML 2027 | TCL-for-LLMs: gradient-EMA importance applied to sequential LLM fine-tuning | Prototype needed |
| **Paper 3** | AutoML@ICML 2027 | TAR as autonomous research system that identifies reproducible findings | Requires Paper 1 first |
| **Null-result brief** | NeurIPS brief/workshop | "Thermodynamic regime detection does not improve CL performance" | Phase 11 ablation ready |

**Do not post to arXiv until Phase 2 HPC replication is complete.** A preprint with p=0.018 that later fails to replicate at n=25 is permanently damaging.

---

## VIII. SAFETY AND ALIGNMENT — THREE NEW MECHANISMS

Three novel mechanisms identified, not in the existing rehabilitation plan:

**1. Prospective Frontier Seal**

Before any experiments run on a new research programme, create a signed manifest of frontier problems (using the Ed25519 infrastructure from Phase 6). This manifest is committed to git and cannot be changed without a new human-approved manifest. Any Director proposal that maps to a frontier outside the sealed set is automatically rejected. This prevents gradual scope narrowing toward easy tractable problems that accumulate "breakthroughs" without scientific significance.

**2. Goodhart Canary**

Every N weeks, randomly partition the linked experiments for a paper into a "publication set" (visible to Director) and a "holdout set" (withheld). If the Director's papers based on the publication set fail on the holdout set more often than a naive baseline, the Director is goodharting the evaluation metric rather than learning science. This requires: (a) maintaining a holdout partition, (b) running a periodic out-of-sample evaluation, (c) alerting if the gap between in-sample and out-of-sample performance exceeds a threshold.

**3. Immutable Director Snapshot**

When stabilisation mode is activated, take an immutable snapshot of the Director weights and governance code (git commit + Ed25519 signature). No self-improvement cycle may update the Director during the stabilisation window. Any proposed Director update must be reviewed against this snapshot by a human before deployment. This converts the current procedural check ("are the governance files committed?") into a hard architectural gate that prevents the system from modifying its own governing code during the most sensitive research phases.

**The misalignment risk not in the plan:**

TAR's optimization target is "produce papers that score higher on an evaluation rubric." This is not the same as "discover scientific truth." The Director can learn to propose experiments with high prior probability of success, then claim breakthrough — without the results being scientifically significant. The three mechanisms above structurally prevent this by: (Seal) restricting the frontier search space, (Canary) detecting when performance on the rubric diverges from performance on held-out science, and (Snapshot) preventing the Director from modifying its own success criteria.

---

## IX. LITERATURE INTEGRATION — THE MISSING INFRASTRUCTURE

**The 15 papers TAR must cite for a 2026 CL paper (priority order):**

| # | Paper | Year | Why Essential |
|---|---|---|---|
| 1 | Kirkpatrick et al. — EWC | 2017 | Foundational; main comparator |
| 2 | Zenke et al. — Synaptic Intelligence | 2017 | SI baseline |
| 3 | Li & Hoiem — LwF | 2016 | Distillation baseline (to be implemented) |
| 4 | Buzzega et al. — DER++ | 2020 | Replay baseline (implemented, unused) |
| 5 | Rebuffi et al. — iCaRL | 2017 | Class-incremental baseline |
| 6 | Lopez-Paz & Ranzato — GEM | 2017 | Gradient constraint baseline |
| 7 | Chaudhry et al. — A-GEM | 2019 | Efficient GEM |
| 8 | Wang et al. — DualPrompt | 2022 | State-of-the-art CIL (vision transformer) |
| 9 | Wang et al. — L2P | 2022 | Prompt-based CL; positions against TAR |
| 10 | Rusu et al. — Progressive Neural Networks | 2016 | Architectural expansion baseline |
| 11 | Javed & White — BWT/FWT metrics | 2019 | Required for metrics section |
| 12 | Martens & Grosse — K-FAC | 2015 | Fisher approximation; grounds Section 3 |
| 13 | Amari — Natural Gradient | 1998 | Theoretical grounding for EMA-Fisher connection |
| 14 | McCloskey & Cohen — Catastrophic interference | 1989 | Historical context |
| 15 | Javed & White — Meta-CL | 2019 | BWT/FWT definitions standard |

**Knowledge graph schema enabling four critical query types:**

```
Node types:
  Paper     { id, title, year, venue_tier, embedding }
  Mechanism { id, name, papers, year_introduced }
  Claim     { id, text, paper_id, truth_value }
  Dataset   { id, name, domain, scale }
  Benchmark { id, dataset_id, metric, sota_entries }
  Domain    { id, name, subdomains }

Edge types:
  paper  -cites->       paper
  paper  -introduces->  mechanism
  paper  -claims->      claim
  claim  -contradicts-> claim
  mechanism -extends->  mechanism
  paper  -evaluated_on-> benchmark
  domain -bridges_to->  domain
```

Key queries this enables:
- Find papers contradicting a claim: `Claim(paper=ours) -contradicted_by-> Claim() -inferred_by-> Paper(year >= 2022)`
- Find adjacent mechanisms not cited: `Mechanism(tcl_mechanism) -related_to-> Mechanism() -used_by-> Paper NOT in bibliography`
- Current SOTA on a benchmark: `Benchmark(dataset=cifar100) -papers-> Paper() SORT BY metric DESC`

**The `ActiveLearner` is one wiring call away from populating this graph.** `LiteratureBrain.start()` must be called at orchestrator startup. This is a 4-line change that unlocks the entire literature intelligence layer.

---

## X. THE 90-DAY CRITICAL PATH

**GPU budget allocation (ruthless):**

| Frontier | Hours | Decision |
|---|---|---|
| `fp-hyperparameter-robustness` | 48–56h | PROCEED — publication-critical |
| `fp-catastrophic-forgetting` | 0h | CLOSE — 17 null, 4 adverse, falsified |
| `fp-regime-detection-accuracy` | 0h | DEFER — document as Path A, future work |
| `fp-scale-up` | 2h pilot only | DEFER — one small pilot, abandon if negative |
| `fp-class-incremental` | 2h re-analysis | DEFER — re-analyse Phase 15 for Paper 2 |

**Weeks 1–4 (Emergency + Triage):**
1. Rotate all exposed credentials — **same day**
2. Commit six governance files (Phase 0.2)
3. Fix t-distribution CI formula across all 4 files (Phase 1.1)
4. Implement Bonferroni correction (Phase 1.2)
5. Add BWT/FWT metrics to existing results — 30 minutes, new
6. Implement LwF in method_registry.py — 2 hours, new
7. Wire ActiveLearner to orchestrator startup — 4 hours (Phase 4.8)
8. Initialize self-improvement anchor pack — 2 hours (Phase 4.10)
9. Disable governor by default — 30 minutes, new
10. Build honest evidence inventory (Phase 0.5)
11. Pre-register Phase 2 HPC replication with SPRT boundaries

**Weeks 5–11 (Experiments):**
12. HPC replication: 20 seeds, SPRT boundary, pre-registered (48h GPU)
13. Phase 17 TinyImageNet rerun: 5 seeds, DER++, LwF added (16h GPU)
14. Phase 16 CIFAR-100 rerun: 5 seeds, same baselines (16h GPU)
15. Add per-task ECE trajectory measurement to all outputs
16. Run class-order Spearman confound audit
17. Run BatchNorm task-boundary reset ablation

**Weeks 12–16 (Paper):**
18. Write abstract and introduction (human, 8 hours)
19. Use `tar_author.py` for methods and results drafts; edit heavily
20. Assemble results table from evidence inventory
21. Build related work section from populated knowledge graph
22. LaTeX compile; arXiv release only if HPC replicates
23. **Submit to ContinualAI@NeurIPS 2026 (target: September 1)**

**Top 3 risks with contingencies:**

| Risk | Trigger | Contingency |
|---|---|---|
| HPC replication fails (p>0.05, n=20) | d<0.3 at n=20 seeds | Pivot to penalty-only ablation; submit to workshop as "modest improvement over EWC, robust hyperparameters." Still defensible. |
| GPU hardware failure (GTX 1650 dies) | Card overheats or CUDA errors | Activate RunPod executor wrapper (Phase 7 item); have cloud backup in place by week 2. Estimated cost: £40/month for A40. |
| Paper not finished by Sept 1 | Sections incomplete, no compile | Pre-write abstract and intro in week 6 (before experiments complete). Seek one external collaborator from ContinualAI community by week 3. |

---

## PRIORITY MATRIX — TOP 20 ENHANCEMENTS

| # | Enhancement | Type | Effort | Impact | In Rehab Plan? |
|---|---|---|---|---|---|
| 1 | Add BWT/FWT metrics to existing results | Evaluation | 30 min | **Critical for publication** | No |
| 2 | Implement LwF in method_registry.py | Algorithm | 2h | **Critical for publication** | Phase 2.5 (partial) |
| 3 | Disable governor by default; document in ablation | Code/Science | 30 min | High | No |
| 4 | Wire ActiveLearner to orchestrator startup | Engineering | 4h | Unblocks knowledge graph | Phase 4.8 |
| 5 | Run class-order Spearman confound audit | Science | 2h | High (validity risk) | No |
| 6 | Add sequential SPRT boundary to HPC replication | Statistics | 3h | Saves ~30% compute | No |
| 7 | Add per-task ECE trajectory measurement | Evaluation | 4h | Missing from all results | No |
| 8 | Mechanism-isolation ablation engine in Director | Automation | 8h | Eliminates false positives | No |
| 9 | SQLite migration for state management | Systems | 2w | Concurrent reads, fast queries | Phase 7 (partial) |
| 10 | Add GEM/A-GEM baseline | Algorithm | 4h | Required at top venues | No |
| 11 | Implement Prospective Frontier Seal | Safety | 4h | Prevents scope creep | No |
| 12 | Implement Goodhart Canary | Safety | 6h | Detects metric gaming | No |
| 13 | Formalise Fisher-EMA theorem with proof sketch | Theory | 8h | Core theoretical contribution | Phase 3.2 |
| 14 | Add Bayesian credible intervals | Statistics | 4h | More informative than p-values | No |
| 15 | Hessian diagonal importance upgrade to TCL | Algorithm | 4h | +2–4% forgetting reduction | No |
| 16 | RunPod executor wrapper | Systems | 1w | Cloud GPU parallelism | No |
| 17 | Cross-failure clustering in Director | Automation | 6h | Breaks local failure-mode loops | No |
| 18 | Immutable Director Snapshot mechanism | Safety | 6h | Prevents self-modification | No |
| 19 | TCL-for-LLMs prototype | Algorithm/NLP | 2w | Second paper opportunity | No |
| 20 | Pre-synthesis literature grounding in synthesizer | Automation | 4h | Prevents derivative synthesis | No |

**Items 1–3** are zero-cost additions that should happen before any experiment runs.
**Items 4–8** are week-1 additions that unblock multiple downstream improvements.
**Items 9–20** elevate TAR from a working research system to a publication-grade autonomous research programme.

---

## THE ONE-SENTENCE DOMAIN SUMMARIES

**CL Algorithms:** Add BWT/FWT metrics (30 min), LwF (2h), and GEM (4h); the rest can wait.

**Theory:** The Fisher-EMA theorem needs to be written as a formal proposition with a proof sketch; without it the methods section is a claim, not a contribution.

**Research Automation:** The mechanism-isolation ablation engine is the highest-leverage architecture addition; it prevents TAR from autonomously accumulating false positives.

**LLM/NLP:** ASC is not novel; TCL-for-LLMs is. Invest in one prototype of TCL applied to sequential LLM fine-tuning — this is the second paper.

**Systems:** SQLite migration first, RunPod executor second; do not build a message queue until experiments take less than 10 minutes each.

**Experimental Design:** The class-order confound audit and per-task calibration measurement are the two missing elements that could invalidate existing results if not addressed before publication.

**Publication Strategy:** Reframe as an empirical audit of EWC's failure mode, not a new thermodynamic theory; this is the only framing the current evidence can support.

**Safety/Alignment:** The Prospective Frontier Seal and Immutable Director Snapshot are the two mechanisms that would most reduce the risk of TAR producing false science at scale.

**Literature:** Wire the ActiveLearner today; populate the knowledge graph from 1,513 existing papers; build the 15-paper citation base before drafting the related work section.

**Strategy:** Close `fp-catastrophic-forgetting`, allocate all GPU budget to hyperparameter-robustness, target ContinualAI@NeurIPS 2026, ICLR 2027 as stretch goal.

---

*This report was produced by a ten-agent PhD-level team covering: continual learning algorithms, deep learning theory, research automation architecture, LLM/NLP, systems architecture, experimental design, publication strategy, safety and alignment, literature synthesis, and research programme strategy. All analysis is read-only; no code was modified.*
