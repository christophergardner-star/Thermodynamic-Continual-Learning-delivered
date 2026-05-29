# Paper Status

**Target venue:** ICLR 2027
**Lead researcher:** Christopher Gardner
**Current phase:** Controlled Phase 10 rerun complete and canonical. Phase 11 rerun complete under rails. Phase 12 and Phase 13 corrected artifacts are available; interpretation is being tightened before final lock.

## Contributions (locked — do not expand without discussion)
1. TCL method — thermodynamic governor with per-task entropy anchoring + D_PR-weighted L2 penalty
2. Pareto-aware accuracy-forgetting reporting — JAF exposes collapse that forgetting-only rankings hide
3. Methodological observation — default importance-weighted baselines can look competitive on forgetting while failing on accuracy

## Section status

| File | Status | Lock condition |
|------|--------|----------------|
| s1_introduction.tex | REVISED | Keep aligned to canonical Phase 10 / 11 / 13 framing |
| s2_background.tex | NOT STARTED | — |
| s3_method.tex | NOT STARTED | — |
| s4_experiments.tex | REVISED | Align to controlled Phase 10 Outcome A + corrected Phase 11/13 framing |
| s5_discussion.tex | REVISED | Update mechanistic framing to current Phase 11 / Phase 13 evidence |
| sA_ewc_sweep.tex | REVISED | Phase 12 corrected artifact integrated |
| sB_si_robustness.tex | NOT STARTED | Optional appendix if a clean rerun replaces the corrected artifact |
| references.bib | NOT STARTED | — |

## Data in hand

- **Phase 9:** TCL vs SGD, ResNet-18, 5 seeds. p=0.0113, d=1.99. CLEAN historical precursor.
- **Phase 10:** Controlled rerun is canonical: TCL vs EWC(λ=100) vs SI(c=0.1, ξ=0.001) vs SGD, ResNet-18, 5 seeds.
  - **Primary analysis (TCL vs EWC):** 5/5 seeds. Mean delta -0.0748, p=0.0190, d=1.70. **Outcome A reproduced under full provenance.**
  - **Secondary (TCL vs SGD):** 5/5 seeds. Mean delta -0.1170, p=0.0012, d=3.68.
  - **SI degeneracy:** acc=0.500 exactly on all 5 seeds at default hyperparameters.
  - **JAF dominance:** TCL mean 0.6436±0.0541 vs EWC 0.5387±0.0454.
- **Phase 11:** Controlled rerun under rails.
  - SGD:           F=0.2191±0.0608, A=0.7093±0.0629, JAF=0.4903±0.1238
  - Governor-only: F=0.2500±0.0688, A=0.6794±0.0702, JAF=0.4294±0.1390
  - Penalty-only:  F=0.1429±0.0177, A=0.7650±0.0171, JAF=0.6222±0.0342
  - Full TCL:      F=0.1272±0.0216, A=0.7657±0.0228, JAF=0.6385±0.0443
  - Full TCL vs SGD:      Δ=-0.0919, p=0.0198, d=1.68, 5/5
  - Full TCL vs governor: Δ=-0.1229, p=0.0232, d=1.60, 5/5
  - Full TCL vs penalty:  Δ=-0.0157, p=0.3154, d=0.51, 4/5
  - Current reading: anchor penalty is load-bearing; governor-only fails; full-vs-penalty remains unresolved at n=5.
- **Phase 12:** Corrected EWC λ sweep artifact available.
  - λ=10:    F=0.1804±0.0372, A=0.7454±0.0354 | vs TCL: Δ=-0.0644, p=0.0287, d=1.92, 5/5
  - λ=1000:  F=0.1151±0.0254, A=0.7388±0.1344 | vs TCL: Δ=+0.0010, p=0.9230, d=0.04, 2/5
  - λ=10000: F=0.1593±0.0053, A=0.5000±0.0000 | vs TCL: Δ=-0.0432, p=0.0204, d=2.04, 5/5
  - Current reading: report per-λ; λ=1000 is effectively tied on forgetting, λ=10 and λ=10000 favour TCL, and λ=10000 collapses.
- **Phase 13:** Corrected SI sweep artifact available.
  - c=0.01: F=0.0519±0.0039, A=0.7904±0.0025 (non-collapsed)
  - c=0.1:  F=0.1298±0.0125, A=0.5000±0.0000 (collapsed)
  - c=0.5:  F=0.0918±0.0006, A=0.5000±0.0000 (collapsed)
  - Current reading: partial recovery only; the collapse is specific to part of the default neighbourhood, not the entire SI family.

## Pending experiments — sequenced

### Reviewer defence
- [x] **[1] Ablation:** complete under the new rails.
- [x] **[2] EWC λ sweep ResNet-18:** corrected artifact available and appendix revised.
- [x] **[3] SI robustness sweep (minimal):** corrected artifact available; a clean env-coupled rerun remains optional if the corrected artifact is deemed insufficient for publication.

### Scope extension
- [ ] **[4] Class-incremental CIFAR-10:** TCL + baselines, 5 seeds.
- [ ] **[5] Split-CIFAR-100:** scale-up benchmark.
- [ ] **[6] Split-TinyImageNet:** harder visual scale-up benchmark.

## Pending paper work

- [ ] Related work (s2)
- [ ] Method (s3)
- [ ] Figures: forgetting/accuracy scatter, JAF comparison, regime timeline
- [ ] Bibliography (references.bib)
- [ ] Optional SI appendix if a clean rerun replaces the corrected artifact

## Standing instructions (lead researcher)

- Claims match evidence exactly.
- No cross-domain evidence mixing.
- No invented citations.
- No hyperparameter retrofitting without explicit post-hoc labelling.
- Every baseline comparison stays scoped to its exact hyperparameter condition.
- Limitations are stated directly, not softened.

## Update protocol

- Canonical result changes propagate from the timestamped artifact plus env snapshot.
- Section text is revised only from canonical, domain-matched evidence.
- Structural changes still require explicit review before lock.
