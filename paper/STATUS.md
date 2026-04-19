# Paper Status

**Target venue:** ICLR 2027
**Lead researcher:** Christopher Gardner
**Current phase:** Phase 10 complete — Outcome A confirmed. Extension experiments next.

## Contributions (locked — do not expand without discussion)
1. TCL method — thermodynamic governor with per-task entropy anchoring + D_PR-weighted L2 penalty
2. Pareto-optimal accuracy-forgetting tradeoff — TCL dominates on JAF metric across all seeds
3. Methodological observation — forgetting-only ranking breaks when importance-weighted methods collapse

## Section status

| File | Status | Lock condition |
|------|--------|----------------|
| s1_introduction.tex | SCAFFOLD | After ablation + SI sweep complete |
| s2_background.tex | NOT STARTED | — |
| s3_method.tex | NOT STARTED | — |
| s4_experiments.tex | PHASE 10 LOCKED | After ablation + SI sweep → extend with new subsections |
| s5_discussion.tex | NOT STARTED | — |
| references.bib | NOT STARTED | — |

## Data in hand
- **Phase 9:** TCL vs SGD, ResNet-18, 5 seeds. p=0.0113, d=1.99. CLEAN.
- **Phase 10:** TCL vs EWC(λ=100) vs SI(c=0.1,ξ=0.001) vs SGD, ResNet-18, 5 seeds. COMPLETE.
  - **Primary analysis (TCL vs EWC):** 5/5 seeds. Mean delta -0.0656, t(4)=-3.26, p=0.0310, d=1.46. **Outcome A confirmed.**
  - **Secondary (TCL vs SGD):** 5/5 seeds. Mean delta -0.1322, t(4)=-3.75, p=0.0208, d=1.68. Reproduces Phase 9.
  - **SI degeneracy:** acc=0.500 exactly on all 5 seeds at default hyperparameters.
  - **JAF dominance:** TCL first on all 5 seeds. Mean 0.672±0.042 (TCL) vs 0.590±0.074 (EWC).
  - **JAF-gap-vs-EWC-acc pattern:** gap >0.10 when EWC acc <0.70 (seeds 1,2); gap <0.06 when EWC acc >0.70 (seeds 42,0,3). Clean separation at threshold, all 5 seeds.

## Pending experiments (for preprint / paper acceptance)
- [ ] **SI robustness sweep:** c∈{0.01, 0.1, 0.5}, ξ∈{1e-4, 1e-3, 1e-2}, 3 seeds. Appendix. Scopes the SI degeneracy claim.
- [ ] **EWC λ sweep on ResNet-18:** λ∈{10, 100, 1000, 10000}, 3 seeds. Appendix. Closes the "did you pick a weak λ" reviewer gap.
- [ ] **Ablation:** 4 conditions (SGD / governor-only / penalty-only / full TCL), 5 seeds. Critical for contribution 1 — isolates what's load-bearing.
- [ ] **Split-CIFAR-100:** 10 or 20 tasks, ResNet-18, TCL + EWC + SGD + SI baseline, 5 seeds. Main extension.
- [ ] **Class-incremental CIFAR-10:** TCL + baselines, 5 seeds. Tests whether regime signal transfers to class-IL setting.

## Pending paper work
- [ ] Related work (s2) — lit review, cite EWC, SI, MAS, iCaRL, GEM, A-GEM, thermodynamic-analogy papers
- [ ] Method (s3) — translate tcl.py and thermoobserver.py into prose + algorithm blocks
- [ ] Discussion (s5) — interpret JAF pattern, variance compression, limits of thermodynamic analogy
- [ ] Figures: (i) forgetting-accuracy Pareto scatter per method per seed, (ii) JAF-gap vs EWC-accuracy scatter, (iii) governor regime timeline for one TCL run
- [ ] Bibliography (references.bib)
- [ ] Abstract with final Phase 10 numbers

## Standing instructions (lead researcher)
- Claims match evidence exactly. No creeping optimism.
- No scope expansion mid-draft. Future ideas go to future work only.
- No invented citations. Unverified refs marked \needsref{}.
- No hyperparameter retrofitting. Post-hoc sweeps clearly labelled.
- Every baseline comparison scoped to exact hyperparameter condition (e.g. "EWC (λ=100)", "SI (c=0.1, ξ=0.001)").
- Limitations are stated, not softened.

## Update protocol
- Seed lands → numbers into s4_experiments.tex table only
- Experiment complete → aggregate stats pass, lock primary analysis
- Structural change → ask before doing
- Section marked done only when zero placeholders remain
