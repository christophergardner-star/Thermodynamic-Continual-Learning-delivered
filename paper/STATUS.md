# Paper Status

**Target venue:** ICLR 2027
**Lead researcher:** Christopher Gardner
**Current phase:** Phase 10 complete — aggregate stats locked, writing sections

## Contributions (locked — do not expand without discussion)
1. TCL method — thermodynamic governor with per-task entropy anchoring + D_PR-weighted L2 penalty
2. Pareto-optimal accuracy-forgetting tradeoff — TCL dominates on JAF metric across all seeds
3. Methodological observation — forgetting-only ranking breaks when importance-weighted methods collapse

## Section status

| File | Status | Lock condition |
|------|--------|----------------|
| s1_introduction.tex | SCAFFOLD | After Phase 10 locked + SI sweep done |
| s2_background.tex | NOT STARTED | — |
| s3_method.tex | NOT STARTED | — |
| s4_experiments.tex | COMPLETE | LOCKED — all 5 seeds + aggregate stats + stats subsection |
| s5_discussion.tex | NOT STARTED | — |
| references.bib | NOT STARTED | — |

## Data in hand
- Phase 9: TCL vs SGD, ResNet-18, 5 seeds. p=0.0113, d=1.99. CLEAN.
- Phase 10: TCL vs EWC vs SI vs SGD, ResNet-18, 5 seeds. COMPLETE.
  - TCL vs EWC: p=0.0310, d=1.46, 5/5 TCL better. Outcome A confirmed.
  - TCL vs EWC deltas: -0.037, -0.024, -0.082, -0.136, -0.049. Mean=-0.0656.
  - SI collapsed to acc=0.500 on all 5 seeds (deterministic, n=5).
  - SGD Phase 9 acc 0.60-0.74 — not collapsed, Phase 9 d=1.99 is clean.
  - Aggregate: TCL F=0.1275±0.026, A=0.770±0.025, JAF=0.672±0.042
  - Aggregate: EWC F=0.1931±0.047, A=0.728±0.050, JAF=0.590±0.074

## Pending experiments
- [x] Phase 10 seed 3 (completed 2026-04-19T22:22:43)
- [ ] SI robustness sweep: c={0.01, 0.1, 0.5}, 3 seeds (post-hoc, appendix)
- [ ] EWC lambda sweep already done (Phase 7) — needs pulling into appendix
- [ ] Split-CIFAR-100 (future work / extension)

## Standing instructions (lead researcher)
- Claims match evidence exactly. No creeping optimism.
- No scope expansion mid-draft. Future ideas go to future work only.
- No invented citations. Unverified refs marked \needsref{}.
- No hyperparameter retrofitting. Post-hoc sweeps clearly labelled.
- Limitations are stated, not softened.

## Update protocol
- Seed lands → numbers into s4_experiments.tex table only
- Experiment complete → aggregate stats pass, lock primary analysis
- Structural change → ask before doing
- Section marked done only when no placeholders remain
