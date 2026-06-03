# Untapped Self-Improvement — Execution Spec

*Generated: 2026-06-03 | Source: 7-agent read-only survey of TAR self-improvement surfaces | Status: ACTIVE PLAN (Phase 1.1 shipped)*

---

## 1. Motivation — the one-line finding

TAR collects rich signal everywhere but **closes almost no loops**. Its only explicit "self-improvement" is a narrow operator-LoRA retrain (`tar_lab/self_improvement.py`) that is double-blocked (GAP-1: no eval step; GPU < 14 GB) and fed by **29 hand-written synthetic signals, not real system signal**. Every other signal source — human decisions, experiment outcomes, falsifications, operational failures, method results, TAR's own published knowledge — is **write-only: generated, stored, displayed, then forgotten.**

**Strategic insight:** 5 of the 6 untapped surfaces need **no GPU and no model retraining** — they are deterministic prior/registry loops. The highest-ROI self-improvement is therefore **unblocked today**, independent of GAP-1/GPU.

---

## 2. The untapped surfaces (evidence map)

| # | Surface | Signal that exists (file/state) | Today | The gap | GPU? |
|---|---|---|---|---|---|
| 1 | **Human feedback** | vetoes/approvals/answers/`veto_reason`/`human_notes`; `human_review_state.json` history (58+); `director_priority_overlay.json` | history **never read**; overlay decays in 48h | no durable preference model — identically-shaped vetoed proposals re-generated forever | No |
| 2 | **Experiment outcomes** | 40k+ executions; `generate_findings_memo`/`generate_failure_diagnosis` (`tar_lab/llm_bridge.py`); `claim_verdicts.jsonl`; `evidence_debt_records.jsonl` | findings/diagnoses **read only by the dashboard** | no learned priors on what works; a failed run's diagnosis never adjusts the next run | No |
| 3 | **Scientific process** | frontier `truth_status` + verdict counts (`frontier_problems.json`); pre-registration | displayed, not gated | **falsified frontiers still generate experiments** (fp-catastrophic-forgetting: 21 null/adverse vs 2 positive); no prediction-vs-outcome calibration; static n=5 | No |
| 4 | **Operational** | watchdog restarts (`watchdog_state.json`, 81 dashboard restarts); `anomalies/` (empty); `circuit_breakers.json` written-never-read; refuse notes | recorded in isolation | no aggregation / trend detection → same failures recur | No |
| 5 | **Method synthesis** | `tar_lab/method_synthesizer.py`; `synthesized_methods/` | **one-shot** generate→validate→quarantine | no variant→validate→adopt loop; validated methods never promoted to canonical | Light |
| 6 | **Knowledge / authoring** | `knowledge_graph.json`; novelty gate; `tar_author.py` drafts/revisions | KG drives proposals (input only) | TAR's own results never update the KG; authoring never learns from cut/rejected claims; arXiv ingest degraded & non-adaptive | No |

Cross-cutting: the existing SI pipeline has **defined-but-unused** `TrainingSignalKind`s (`research_decision`, `falsification_plan`, `claim_verdict`, `problem_study`) and a dormant `AgendaEngine._recycle_to_training_signal` — the wiring points for Phase 4.

---

## 3. The unifying pattern (the spec)

Every loop is the same safe shape:

```
SIGNAL → AGGREGATOR (read-only, periodic)
       → DURABLE PRIOR / REGISTRY (plain JSON, inspectable, sliding window)
       → FEEDBACK HOOK (adjusts a decision: proposal score / retirement / param)
       → SAFETY RAILS
```

### Safety envelope — non-negotiable (every agent independently insisted)
1. **Suggest, don't seize** — learned priors *adjust scores / propose*; they never silently act.
2. **Human-reviewable + audit-logged** — plain JSON; every adjustment logged with its reason; operator override available.
3. **Never weaken integrity** — must not lower evidence standards; no post-hoc hypothesis edits; sample-size changes are amendment-logged **before** running (preserve pre-registration honesty).
4. **Never self-modify governance/execution code** — operational tuning is limited to *parameters* (timeouts/cooldowns/thresholds), never manifest/review/execution/auth logic.
5. **Preserve exploration** — keep ≥ ~30% of proposals novelty-driven so learned priors cannot collapse the search space.
6. **Append-only / additive (Phase 3)** — existing algorithms and knowledge are READ-ONLY to self-improvement. It may CREATE new candidate artifacts (method variants, internal knowledge entries, memories) but NEVER overwrite/edit/delete a canonical method, algorithm, knowledge entry, or prior result (the methods/knowledge analogue of RAIL 1). Enforced by `tar_lab/append_only_guard.py` and proven by `tests/test_append_only_self_improvement.py`.

---

## 4. Phased plan

### Phase 1 — Stop repeating known mistakes (unblocked, highest ROI)
- **1.1 Falsified-direction retirement** — ✅ **SHIPPED 2026-06-03.** `tar_research_director.py` (`_build_experiment_directives`, ~line 2576): skip `_frontier_experiment_catalog` for a frontier with `truth_status=="falsified"` and no `waiting_on_experiment_ids`. Reversible; existing/running experiments unaffected. *Effect: TAR stops generating fresh probes on fp-catastrophic-forgetting.*
- **1.2 Human-feedback priors** — ✅ **SHIPPED 2026-06-03.** NEW module `tar_lab/human_feedback_learning.py` (read-only aggregator over `human_review_state.json` + `director_proposals.json`) → durable `tar_state/human_feedback_learned_priors.json` (90-day window). Director hook in `_priority_for` (`tar_research_director.py`) applies a **penalty only**: an explicitly human-vetoed/rejected experiment is sunk (−1000, sorts last, reversible on re-approval); a frontier with ≥3 decisions at ≥60% veto-rate softly penalises new proposals (≤40 × rate). Only explicit human actions count (passive 24h auto-approval ignored). Operator off-switch: `tar_state/human_feedback_priors.disabled`. Verified by unit test + live-state run (currently 0 vetoes → 0 penalty; armed).
- **1.3 Outcome registry** — ✅ **SHIPPED 2026-06-03.** NEW module `tar_lab/outcome_learner.py` (read-only over `experiment_archive.json` + the `llm_cache` findings/failure memos) → durable `tar_state/outcome_learning/outcome_priors.json` with (a) per **method×dataset** reliability stats (n, completed/failed, forgetting mean/std, `reproducible` when σ<0.08) and (b) a per-experiment digest that **pulls in the previously write-only findings-memo & failure-diagnosis**. Director hooks: rebuild each cycle; `_priority_for` subtracts a penalty-only `failure_penalty` (deprioritise re-proposing an operationally-FAILED experiment — never scientific NULLs, which the plan re-runs); directives now carry an `outcome_digest` so findings/diagnoses reach the director state + LLM proposer. Deliberately NO scoring boost for "good" combos (preserve exploration). Off-switch: `tar_state/outcome_learning.disabled`. Verified unit + live (11 combos, 11 findings + 5 diagnoses consumed, 0 failures → 0 penalty, armed).

### Phase 2 — Get better at science + ops (unblocked) — ✅ SHIPPED 2026-06-03
- **2.1 Calibration loop** — ✅ NEW `tar_lab/calibration_learner.py` → `tar_state/calibration/calibration_registry.json` (ADVISORY ONLY). (a) Effect-size calibration: per result with a Cohen's d → observed |d|, achieved power, and **seeds-needed-for-80%-power** (via `stat_utils.power_analysis`), with the integrity note to pre-register the larger n via amendment first. (b) Frontier calibration: flags optimistic miscalibration (ranked-high-but-falsified — e.g. fp-catastrophic-forgetting). Director refreshes it each cycle; changes NO score, NO pre-registration (rail #3). Off-switch `tar_state/calibration.disabled`. Verified (HPC flagged underpowered → 7 seeds; fp-catastrophic-forgetting flagged).
- **2.2 Operational self-tuning** — ✅ NEW `tar_lab/operational_learner.py` → `tar_state/operational/operational_recommendations.json` (ADVISORY ONLY). Aggregates watchdog restart pressure, health-check failures, and recurring lease-termination reasons into recommendations with **suggested** parameter tunings for **human approval** (params-only — never code/governance; rail #4). Off-switch `tar_state/operational.disabled`. Verified (dashboard 81-restart HIGH → suggest ↑`stale_after_s`; recurring dead-PID cleanup flagged). Periodic rebuild (watchdog/health-check hook) is the small follow-up; runnable now via `python -m tar_lab.operational_learner`.

### Phase 3 — Compounding loops (append-only / additive — rail #6)
- **Append-only rail** — ✅ **SHIPPED 2026-06-03.** `tar_lab/append_only_guard.py`: `guarded_create_new()` refuses to overwrite an existing artifact or any protected canonical algorithm / knowledge path; `snapshot()`/`assert_unchanged()` prove canonical methods + KG are byte-for-byte unchanged after self-improvement runs. Verified by `tests/test_append_only_self_improvement.py` (4 properties) + live check (5 protected files unchanged).
- **3.1a Method-refinement PROPOSER** — ✅ **SHIPPED 2026-06-03 (propose-only).** `tar_lab/method_refinement_engine.py` reads the outcome registry → advisory `tar_state/method_refinement/variant_proposals.json` (targeted variant suggestions for under-performing/failed method×dataset combos, e.g. stronger penalty / LR annealing). Generates NO code, runs nothing, adopts nothing; each proposal records the actuation rails (new-file-only via append_only_guard, RAIL-3 sandbox, pre-registration + MC-correction, human ≥2-dataset adoption gate). Director refreshes each cycle. Off: `method_refinement.disabled`. Live: 6 proposals (tcl, der++).
- **3.2a Authoring style memory** — ✅ **SHIPPED 2026-06-03 (advisory, style-only).** `tar_lab/authoring_learner.py` aggregates paper-claim review outcomes → `tar_state/authoring/authoring_style_memory.json` (accepted-vs-cut claim styles). Shapes STYLE only — never invents/alters a fact, never edits a paper. Off: `authoring.disabled`. Live: 8 accepted claim decisions, armed.
- **Deferred (full closed loop, human-gated):** 3.1b auto-generate variant code → sandbox-run → head-to-head → adopt-to-canonical (all via append_only_guard + human gate); 3.2b KG self-update on an approved paper (`tar_internal`, low weight, human gate) + adaptive arXiv ingestion. These cross into "TAR creates runnable code / edits its world-model" and stay behind the autonomy-ramp `awaiting_confirm` gate.

### Phase 4 — Feed real signal into the model (when GPU/GAP-1 resolved)
- Wire the now-flowing real signals from Phases 1–3 into the existing `TrainingSignalRecord` kinds (close GAP-1 eval step first), so the operator-LoRA finally trains on genuine system experience rather than 29 synthetic items.

---

## 5. Success criteria

- [x] **1.1** Falsified frontiers no longer generate fresh proposals (verified: fp-catastrophic-forgetting). 
- [x] **1.2** Human-vetoed experiments sink and high-veto frontiers penalise new proposals; durable prior is the audit. (Shipped; armed — 0 vetoes in current history.)
- [x] **1.3** Director reads a prior derived from past outcomes (method×dataset registry); findings/diagnoses are consumed into the digest, not just displayed; failed experiments are deprioritised. (Shipped; armed — 0 failures in current archive.)
- [x] **2.1** Calibration registry computes predicted-vs-observed + power-based recommended sample size (HPC: 7 seeds for 80% power) and flags optimistic frontiers — advisory; human pre-registers any n change via amendment. (Shipped.)
- [x] **2.2** Operational recommendations populated from aggregated failure signals (dashboard 81 restarts, recurring dead-PID cleanup); suggested param tunings await human approval. (Shipped; actuation human-gated.)
- [ ] **3.x** ≥1 synthesised method variant validated head-to-head; ≥1 KG self-update on an approved paper.
- [ ] **4** Operator-LoRA delta assembled from real (non-synthetic) signals.

## 6. Risks
- Overfitting to noise → reproducibility thresholds + sliding windows.
- Search-space collapse → exploration floor (rail #5).
- Integrity erosion → rails #3/#4 are hard gates, human-reviewed, audit-logged.
- Falsified-retirement false-negative (a method that fails on one frontier but works on another) → guard is per-frontier and reversible; canonical methods are unaffected.
