# TAR — Five-Domain Autonomous-Research Roadmap

**Authored:** 2026-06-03 · **Source:** independent read-only audit (9 specialist agents + direct source
verification). **Provenance note:** this roadmap was derived bottom-up from the *code, runtime state, and git*,
**not** from `TAR_Master_Implementation_Plan.md` / `TAR_PhD_Rehabilitation_Plan.md`. Where it converges with those
documents that is corroboration, not inheritance. Companion document:
`TAR_Phase0_and_StackBridge_Implementation_Plan.md` (the build-ready keystone).

---

## 0. The question this answers

> Will TAR ever autonomously do self-research, experimentation, and frontier-gap search across
> **continual learning · thermodynamic ML · medical AI · quantitative finance · quantum ML** — and what is the
> full team plan to make it one of the most advanced autonomous research systems in those domains?

**Answer: yes, and the machinery is ~70% already present.** The work is *unification + loop-closure + policy +
scientific integrity*, not greenfield. The constraints are honest and specific (below).

---

## 1. The decisive architectural fact — TAR has TWO research stacks

| | **Stack A — the live brain** | **Stack B — the dormant polymath** |
|---|---|---|
| Entry | `tar_living_research.py` daemon (`while True` loop) | `tar_lab.orchestrator.TAROrchestrator` → `ProblemResearchEngine` → `tar_lab/problem_runner.py` → `tar_lab/science_exec.py` |
| Triggered by | **the autonomous daemon** | CLI / chat / dashboard / `phase14` — **NOT the daemon** |
| Domains | **CL / vision only**, hardcoded `_dispatch` switch (`tar_experiment_orchestrator.py:2171`) | **7 domains, config-driven**: `quantum_ml` (real PennyLane), `graph_ml` (real torch_geometric), `generic_ml` (real sklearn/OpenML), `deep_learning`, `computer_vision`, `nlp`, `reinforcement_learning` (`science_exec.py:343-358`) |
| Add a domain | hand-edit director + orchestrator | drop `science_profiles/<name>.json` (auto-loaded) + write one `_execute_<domain>` |

**Implication:** the multi-domain, polymath engine *already exists and partly works* — it is simply not the
orchestrator the autonomous daemon drives. The single highest-leverage change in the whole project is the
**Stack-A ↔ Stack-B bridge** (the keystone).

### Frontier-gap search already exists — but is fenced off
- `literature/gap_detector.py` implements 7 gap types (benchmark coverage, scale, temporal, conflict,
  cross-domain, replication), scored `0.40·impact + 0.35·novelty + 0.25·tractability`, and runs live
  (`frontier_gaps.jsonl`, `gap_scan_reports.jsonl`, 91 records).
- It is **short-circuited**: `tar_research_director.py:34` `_STRICT_REAL_WORLD_FRONTIER_ONLY = True` makes the
  novel-problem path hard-return `None`. Gaps currently *advise prioritization* but cannot become experiments.
- `FrontierRegistry.register` (`tar_frontier.py:372`) **can** mint a frontier autonomously but enforces a real-science
  guard (`well_known_problem=True` + named external baselines/datasets/backbones). **Keep this guard.**
- Quant finance is **policy-blocked** (`_FINANCE_HARD_KEYWORDS`, `_ACCEPTED_DOMAINS` excludes it).

---

## 2. The synergy thesis — what "breathing synergistically" means

Five domains bolted on is a gimmick. **One spine with applied limbs is a research program.** TAR's spine is the
union of its two namesakes:

> **Thermodynamic Continual Learning — a free-energy / entropy account of catastrophic forgetting.**
> Stability–plasticity as an energy budget; importance as gradient-energy; forgetting as entropy production.

Everything hangs off that through-line:
- **Thermodynamic-ML** = the theory half of the spine (free-energy descent, diffusion-as-non-equilibrium-thermo, EBMs). *Most original; fully local.*
- **Quant-finance (FI-2010)** = the spine *applied*: continual learning under concept drift **is** the finance problem. *Highest-value applied test; fits 4 GB; gated on leakage discipline.*
- **Quantum-ML** = a Fisher-geometry / loss-landscape side-study (barren plateaus ≈ gradient-energy collapse). *Legitimate, local, low-synergy; expect honest "no advantage."*
- **Medical-AI** = the spine applied to cross-hospital distribution shift. *High impact; human-gated.*

---

## 3. Domain feasibility & sequencing (grounded in current benchmarks)

Ranked by **autonomy-tractability × synergy × compute-fit**:

| Rank | Domain | Tractable now? | Local 4 GB? | First runnable benchmark | Hard constraint |
|---|---|---|---|---|---|
| 1 | **Continual learning** | ✅ production | ✅ | Split-CIFAR-10/100, Split-MNIST | none (home turf) |
| 2 | **Thermodynamic ML** | ✅ (theory + toy/MNIST) | ✅ | small EBM/DDPM on MNIST, entropy-production probes | metaphor-vs-measurement rigor; no hardware-energy claims |
| 3 | **Quant finance** | ✅ (open data) | ✅ | **FI-2010** LOB (CC-BY-4.0) | walk-forward/purged splits; no tradeable-alpha claims; licensed feeds off-limits |
| 4 | **Quantum ML** | ✅ (simulator, CPU) | ✅ (CPU-bound) | PennyLane VQC on Iris/MNIST-PCA; barren-plateau gradient-variance | no real-QPU "advantage" claims |
| 5 | **Medical AI** | ⚠️ MedMNIST yes; MIMIC no | MedMNIST-small ✅; 3D/224px → RunPod | MedMNIST / MedMNIST-C | **MIMIC needs PhysioNet DUA + CITI — a human must clear it; autonomy must defer** |

**Recommended order:** consolidate **CL** → push **Thermodynamic-ML** (the novel crown) → **Finance/FI-2010** (the
applied crown) → **Quantum** (local side-study) → **Medical** (human-gated, last).

**Feasible-soon vs needs-human-scaffolding:** autonomous science is realistic *now* for CL, Thermo-ML,
Finance (open data), and the simulator slice of Quantum. **Needs humans:** credentialed medical data, any
thermodynamic-*hardware* claim, real-money finance, real-QPU quantum advantage.

---

## 4. The Lab — a standing team of specialist agents

Each agent owns real TAR components and is invoked (as in this audit) for the work in its lane.

| # | Specialist agent | Mandate | Owns |
|---|---|---|---|
| 1 | **Principal Investigator** (orchestrator) | sets agenda, sequences phases, arbitrates | the roadmap, frontier priorities |
| 2 | **Methodologist / Statistician** | preregistration, seeds, SPRT, power, correction | `stat_utils.py`, `run_hpc_replication.py`, preregs |
| 3 | **CL/Thermo Scientist** | the spine: `tcl_canonical`/`tcl_full`, free-energy framing | `tcl.py`, `multimodal_payloads.py`, phase scripts |
| 4 | **Domain-Extension Engineer** | Stack-A↔B bridge; new profiles + executors | `science_exec.py`, `science_profiles/`, orchestrators |
| 5 | **Frontier Scout** | literature + gap detection + novelty gating | `gap_detector.py`, literature graph, `FrontierRegistry` |
| 6 | **Self-Improvement / MLOps Engineer** | close GAP-1, eval harness, RunPod scaling | `self_improvement.py`, `eval_harness.py`, `tar_runpod_*` |
| 7 | **Alignment & Safety Officer** | rails armed, veto used, ramp staged, no over-claiming | `tar_autonomy_ramp.py`, `human_review.py`, kill-switch flags |
| 8 | **Reliability / Infra Engineer** | state hygiene, CI, disk, atomic writes | `tar_health_check.py`, watchdog, E:/C: disk |
| 9 | **Provenance Auditor** | manifests, anchors, append-only integrity | `manifests/`, `anchors/`, `append_only_guard.py` |

---

## 5. The six-phase program

> **Ordering law:** truth and safety gate capability. You cannot be "advanced autonomous" on a flagship result
> that benchmarks a proxy. Phase 0 precedes all expansion.

### Phase 0 — Truth & Safety Foundation *(detailed in the keystone doc)*
*Owners: Methodologist, CL/Thermo Scientist, Alignment Officer, Provenance Auditor, Infra.*
- Fix the flagship science: `tcl_canonical` + new `tcl_full` (importance **and** regime observer) at full Phase-10 protocol vs EWC/SI/SGD.
- Fix the confirmatory test: replace the invalid SPRT input; log a formal prereg amendment.
- Protect the work: gitignore `xdg/`+`pip/`; selectively commit the untracked SI stack the running director imports; push the 44 unpushed commits.
- Arm alignment: initialize the autonomy ramp (0-byte no-op today); actually use the 24 h veto.
- Hygiene: free C: (10 GB left); reconcile stale `DORMANT.json`/PID markers.

### Phase 1 — Close the loops (make it breathe)
*Owners: Self-Improvement Engineer, Frontier Scout, Domain-Extension Engineer.*
- Wire the two orphan CPU learners (`operational_learner`, `authoring_learner`) — free signal, no GPU.
- Have the director **query the VectorVault it already builds** (semantic memory is write-only to the reasoning loop today).
- Close **GAP-1**: wire the existing eval harness into `run1()` so the SI gate can pass functionally (even before cloud GPU).
- Carefully relax `_STRICT_REAL_WORLD_FRONTIER_ONLY` (per-domain) so `gaps_to_problems` → `FrontierRegistry.register`, **keeping** the `well_known_problem` guard.

### Phase 2 — Unify the stacks (one body) — THE KEYSTONE
*Owner: Domain-Extension Engineer.* Bridge the daemon to `science_exec` so the `while True` loop can dispatch
*multi-domain* experiments through the same archive/verdict/provenance path. Everything multi-domain depends on
this one seam. *(Full spec in the keystone doc.)*

### Phase 3 — Light up the domains, in synergy order
*Owners: Domain-Extension Engineer + CL/Thermo Scientist + Methodologist.*
1. **Thermodynamic-ML** profile + executor (toy EBM/diffusion + entropy probes) — all local.
2. **Quant-finance** — remove the hard-block, add FI-2010 adapter, **enforce walk-forward/purged splits**.
3. **Quantum-ML** — promote the existing PennyLane path onto the daemon (barren-plateau / gradient-variance).
4. **Medical-AI** — MedMNIST autonomously; **MIMIC-grade gated behind explicit human DUA/CITI** (hard stop).

### Phase 4 — Scale & genuinely self-improve
*Owners: MLOps Engineer, Infra Engineer.*
- RunPod for ≥24 GB work (operator-LoRA, ViT, CIFAR-diffusion).
- Feed **real** signals (the 129 `portfolio_decisions.jsonl` records) into self-improvement instead of 29 synthetic.
- Add 2–3 parallel seed workers locally (16 threads, 64 GB RAM idle) → 2–3× experiment throughput.

### Phase 5 — Frontier autonomy with guardrails
*Owner: PI + Alignment Officer.* The full unattended loop across domains: gap-scan → mint frontier (guarded) →
preregister → run → evaluate → ingest → draft → human-review — within *armed* rails. This is "alive."

---

## 6. Alignment — four non-negotiable axes

1. **Aligned to truth.** Integrity gates stay hard (Bonferroni, append-only results, never claim the proxy is the method). The Phase-0 science fix *is* an alignment act.
2. **Aligned to the human.** Ramp staged, veto armed, `well_known_problem` guard kept. Default must become *ask*, not *proceed*.
3. **Aligned to feasibility (no over-claiming).** No thermodynamic-*hardware* energy claims; no real-QPU "quantum advantage"; no tradeable financial alpha; no clinical-validity claims; medical data human-gated. The system must **say what it cannot prove.**
4. **Aligned to provenance.** Every cross-domain result carries a manifest + env snapshot + anchor; fill the empty `anchors/`.

---

## 7. Compute strategy (PC vs potential)

- **Local (Ryzen 7 5700G, 64 GB RAM, GTX 1650 4 GB):** CL core, Thermo-ML theory/toy/MNIST, Finance FI-2010, Quantum simulators (CPU-bound). The 4 GB VRAM is the hard ceiling; 64 GB RAM and 16 threads are *under-used*.
- **RunPod (24 GB+):** operator-LoRA self-improvement, ViT/large backbones, CIFAR-scale diffusion, MIMIC notes.
- **Throughput win available now:** parallel seed workers (the HPC suite runs seeds sequentially at ~2.8 h each).
- **Disk:** C: only ~10 GB free — move builds to E:; never spill checkpoints to C:.

---

## 8. Honest bottom line

TAR can become one of the most advanced **locally-runnable** autonomous research systems on the
**CL ↔ Thermodynamic-ML spine, with Finance and Quantum as fully-feasible extensions** — because most of the
machinery already exists. The work is: **(0)** make the science true, **(1)** close the loops, **(2)** bridge the two
stacks, **(3)** light up domains in synergy order, **(4)** scale, **(5)** run the guarded frontier loop.
**Medical-AI and any "advantage / alpha / hardware" claim are where honest autonomy must defer to a human.**
Build truth and safety first — then it breathes.
