# Paper Status

**Target venue:** ICLR 2027
**Lead researcher:** Christopher Gardner
**Current phase:** Phase 11 complete — ablation done. EWC λ sweep and SI sweep next.

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
| s4_experiments.tex | PHASE 11 LOCKED | After EWC λ sweep + SI sweep → extend with appendix subsections |
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
- **Phase 11:** Ablation (4 conditions × 5 seeds). COMPLETE. Verdict: **PENALTY_DOMINANT**.
  - SGD:           F=0.2354±0.079, A=0.6929±0.080, JAF=0.5348±0.114
  - Governor-only: F=0.2354±0.034, A=0.6931±0.035, JAF=0.5309±0.050
  - Penalty-only:  F=0.1905±0.065, A=0.7169±0.067, JAF=0.5839±0.097
  - Full TCL:      F=0.1148±0.008, A=0.7770±0.006, JAF=0.6879±0.011
  - Governor-only vs SGD: Δ=-0.0001, p=0.999, d=0.001 (zero effect).
  - Penalty-only vs SGD:  Δ=-0.045,  p=0.330, d=0.496 (directional, not significant).
  - Full TCL vs SGD:      Δ=-0.121,  p=0.030, d=1.48, 5/5 (significant).
  - Full TCL vs governor: Δ=-0.121,  p=0.002, d=3.18, 5/5 (highly significant).
  - Full TCL vs penalty:  Δ=-0.076,  p=0.056, d=1.20, 5/5 (directional; misses α=0.05).
  - Variance compression: full TCL std=0.008 vs penalty-only std=0.065 (governor role).
  - Ablation subsection added to s4_experiments.tex (locked).

## Pending experiments — SEQUENCED (reviewer defence before scope extension)

### Reviewer defence (run these first — gate on ablation result before extending)
- [x] **[1] Ablation:** DONE. Verdict: PENALTY_DOMINANT. Subsection added to s4. See Phase 11 data above.
- [ ] **[2] EWC λ sweep ResNet-18:** λ∈{10, 1000, 10000}, 3 seeds = 9 runs. (λ=100 already in Phase 10.) Appendix. Closes "you picked a weak λ" gap.
- [ ] **[3] SI robustness sweep (minimal):** c∈{0.01, 0.1, 0.5}, ξ=0.001 fixed, 3 seeds = 9 runs. Appendix. Scopes SI degeneracy claim. Full 2D grid only if surprising interaction effects in this pass.

### Scope extension (only after all three defence experiments land clean)
- [ ] **[4] Class-incremental CIFAR-10:** TCL + baselines, 5 seeds = 20 runs. Tests whether regime signal transfers to class-IL setting.
- [ ] **[5] Split-CIFAR-100:** 20 tasks, ResNet-18, TCL + EWC + SGD + SI, 5 seeds = 20 runs (much longer per seed). Main extension. Last.

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
