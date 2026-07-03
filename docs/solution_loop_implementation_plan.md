# TAR Solution-Finding Loop — Implementation Plan

**Date:** 2026-07-03 · **Status:** PLAN (approved scope: the "four ingredients"; no implementation yet)
**Goal:** turn TAR from a self-referential hypothesis generator ("test another TCL variant") into a
disciplined **generate-and-test engine**: real prior art in, anomaly-targeted candidate mechanisms out,
every candidate killed or validated by the existing governance stack (prereg → veto → ramp → n-seeds →
Holm → truth-lock). Governance is NOT modified — it is the selector.

Grounded by four read-only code scouts (2026-07-03). Every anchor below is a verified file:line.

---

## Phase 0 — Integrity pre-work (BLOCKING: the loop is unsound without these)

The scouts found four defects that would silently corrupt the loop's results. Fix first.

**0.1 The `"tcl"` key collision (truth-lock hazard).**
`method_registry.py:586` binds key `tcl` to the CANONICAL tcl.py algorithm in Harness B (generic_cl),
but Harness A (`multimodal_payloads.py:1284-1296`) and the truth-lock layer
(`method_identity.py:26`, `_TCL_PROXY = {"tcl", "tcl_penalty_only"}`) treat `tcl` as the uniform-L2 PROXY.
A generic_cl run of `method="tcl"` runs canonical code yet gets fingerprinted `uses_uniform_l2_proxy=True`.
*Fix:* rename the registry key (e.g. `tcl_canonical_generic`) or teach method_identity harness-aware
identity. Acceptance: no key maps to two algorithms.

**0.2 `internal_source_tag` misclassifies novel methods as baselines (circular-validation hazard).**
`method_identity.py:51-64` tags anything outside `_TCL_FAMILY` as `tar_internal_baseline` — the
NoveltyGate comparison bar (`novelty_gate.py:356-360`). A NOVEL composed candidate would become its own
external bar. *Fix:* add a third class `tar_novel` (methods minted by the proposer/synthesizer), excluded
from the bar like `tar_internal`. Mirror at `multimodal_payloads.py:903-904`.

**0.3 Preregistered criteria are WRITE-ONLY (governance gap; also blocks Ingredient 3).**
Prereg entries carry `criteria` dicts (`tar_living_research.py:879`), but repo-wide nothing reads them
back; verdicts use HARDCODED thresholds at `tar_experiment_orchestrator.py:2986`
(`mean_delta < -0.01 and p_val < alpha_bonf and cohens_d > 0.5`); accuracy never enters the verdict;
no collapse detection. Gate 4 (`tar_lab/validation.py:132-182`) checks only that a prereg record EXISTS.
*Fix:* a criteria evaluator in/after `_build_result` that loads the prereg entry for `spec.id` and
enforces its keys (incl. new ones below). This is also a north-star governance win in its own right.

**0.4 Silent-SGD + registry-set drift hazards.**
(a) Harness A trains an UNRECOGNIZED method string as plain SGD without error (if/elif fallthrough,
`multimodal_payloads.py`); add a strict allowlist raise. (b) `_NATIVE_METHODS`
(`tar_living_research.py:1427-1430`) omits `tcl_full` and `agem` — a directive proposing either wrongly
enters the LLM-synthesis path. Add them.

*Effort: ~1 day. All local, test-covered, no behavior change for existing runs.*

---

## Phase 1 — Ingredient 1: load the material (`literature/method_catalog.json`)

**What:** a hand-curated catalog of ~40 real CL methods — the proposer's recombination material.
Schema (per Scout D): `method_key, full_name, mechanism_class` (enum: regularization | replay |
distillation | architectural | parameter_isolation | prompt_based | subspace_projection |
optimizer_based | bayesian), `mechanism_summary, protects, failure_modes[], citation{paper_title,
arxiv_id|doi|url, year, venue}, paper_id_in_db (null-honest), implemented_in_tar, combinable_with[],
provenance (human_curated | db_extracted)`. **NO metric numbers** — cited numbers go exclusively through
`curated_external_sota.json` (existing refuse-uncited loader, `curated_sota.py:67`).

**Why a NEW file (not knowledge_graph.json):** the existing 12-method registry lives in a file that
(a) fails its own schema validation (`KnowledgeGraphState`, `tar_lab/state.py:216`; vault indexer skips it
as drift, `vault.py:1185-1202`), (b) is append-only-protected (`append_only_guard.py:28,124`), and
(c) has demonstrably WRONG mechanism labels (DualPrompt and O-LoRA tagged "replay"). New loader
`literature/method_catalog.py` clones curated_sota.py's guards: refuse entries missing
`method_key | full_name | mechanism_class | citation`.

**Curation split (honest):** the literature DB (1,954 papers) contains essentially ZERO pre-2022
method-defining papers — EWC, SI, MAS, GEM, A-GEM, iCaRL, GDumb, DER++, ER, LwF originals all absent.
Auto-extractable: only ~10-15 modern methods (DualPrompt, CODA-Prompt, O-LoRA, InfLoRA, EASE…), each
still human-verified (the old registry's mislabels prove unsupervised extraction fails).
~25-30 classics are typed in by hand — optionally after ingesting their known arXiv ids through the
existing semantic_scholar source so `paper_id_in_db` becomes non-null. **Human gate: the lead verifies
every entry (anti-fabrication).**

**Consumption wiring (three points):**
1. **Proposer prompt** — serialize catalog (mechanism_class + summary + failure_modes) into
   `llm_bridge.py:733-755`, which today contains ZERO method knowledge.
2. **Gap detector** — its method universe comes only from `sota_entries` (`gap_detector.py:183,193` via
   `methods_on_benchmark`), i.e. 6 internal rows; optionally upsert method-coverage stubs so coverage
   gaps see the real space.
3. **NoveltyGate** — untouched; it consumes cited numbers, which stay in curated_external_sota.json.
   (Populating THAT file with 5-10 cited Split-CIFAR-10/100 numbers is a sibling task with the same
   human gate — it finally gives the novelty verdicts an external bar.)

*Effort: loader + wiring ~half day code; curation ~1-2 days human-in-the-loop.*

---

## Phase 2 — Ingredient 3: the SI anomaly becomes a first-class research object

**The anomaly (fully evidenced, all TAR-generated):** on the same dataset/seeds, SI is EITHER the best
and by far the most stable method in the suite OR exactly at chance, controlled solely by one
hyperparameter:
- Stability: phase18 (n=5) SI forgetting **0.0474 ± 0.00753** — std 3.2-9.3× tighter than every other
  method, best mean on both metrics, beats TCL on 5/5 seeds (d=2.52-4.23)
  (`comparisons/phase18_tcl_canonical_fullprotocol__20260605T093223Z.json`).
- Collapse: phase10 (si_c=0.1) accuracy **exactly 0.500 on all 5 seeds** (std 0.0) — chance on 2-class
  tasks; the low forgetting is an artifact of learning nothing. Phase13 sweep: c=0.01 fine; c∈{0.1,0.5}
  → 3/3 collapse (threshold 0.55).
- **Nothing in the suite explains the cliff.** And `honest_evidence_inventory.json` has NO SI record —
  the anomaly is invisible to the paper-planning source of truth.

**Steps:**
1. **Add the SI anomaly to `honest_evidence_inventory.json`** (with the numbers above; human sign-off).
2. **Mint the gap:** `ResearchGap` (gap_type `"theoretical"` — the enum has no "anomaly";
   `literature/schemas.py:190-201`), domain `continual_learning`, description carrying the full anomaly
   statement (it becomes the probe's `experiment_goal` via `frontier_problem_from_gap`,
   `tar_frontier.py:326-329`). **Constraint:** director picks `get_top_gaps(n=1)` — the anomaly gap must
   be the top-composite open CL gap (score it accordingly, honestly).
3. **Preregister the JOINT criterion** for any candidate: `max_forgetting_std ≤ 0.00753 (SI's)`,
   `min_mean_acc ≥ 0.794 (SI's)`, `min_seed_acc ≥ 0.55 (phase13 collapse threshold — kills the
   "stability by not learning" cheat)`, plus the existing delta/p/d keys. Enforced by the Phase-0.3
   evaluator. **Honest power note:** a variance criterion at n=5 is underpowered — screening at n=5 may
   only kill; confirmation of a variance claim needs an explicit preregistered variance-ratio test at
   n≥20 (TAR's own prereg power table says n=5 covers only d ≥ 1.26).

*Effort: ~1 day + human sign-off on framing.*

---

## Phase 3 — Ingredient 2: widen the proposer (three arms + the implementability bridge)

**The mandate to replace** (all verified): the catalog arm pins `"method": "tcl"` at ~10 sites
(`tar_research_director.py:2034…2506`) with self-referential `mechanism_focus`/`internal_method_role`
text (`:2003-2022`); the LLM arm hard-pins `"method": "tcl"` at `:3277` and its prompt
(`llm_bridge.py:747-752`) mandates "compare TCL against baselines"; the response schema has NO method
field at all.

**Changes:**
1. **fp-gap-* probes (the anomaly path):** replace `_method = "tcl"` (`:2506`) with a
   mechanism-proposer step: input = anomaly record + method catalog + kill-ledger (Phase 4); output =
   N candidates, each REQUIRED to (a) name its mechanism_class, (b) cite which catalog methods it
   recombines/differs from, (c) state a falsifiable prediction against the joint criterion, (d) resolve
   to a RUNNABLE method key (see bridge below).
2. **LLM arm:** add `method` to the response schema + validation (validate against
   registry ∪ catalog-implementable set); replace the TCL-only rules; inject the catalog block.
3. **Host it in H2:** `experiment_design.py` is live (flag present) and has an explicit `[RESEARCH]`
   proposer/critic hook (`:14-16`, advisory critic `:111`); a proposer emitting
   `HypothesisSpec(claim, primary_method=<candidate>, mechanism_components=[…])` slots directly upstream
   of `design_experiment`, which already computes falsifying baselines, powered n, and frozen prereg
   for ANY method the registry knows.

**The implementability bridge (proposable ⇒ runnable), strictly tiered:**
- **Tier 1 (default): composition over existing primitives in Harness B** — importance estimators
  (Fisher `method_registry.py:122`, SI path-integral `:183`, gradient-EMA `tcl.py:138`, Fisher-Rao 2nd
  order `tcl.py:152`), penalty forms (+ ring-buffer decay/anneal knobs `tcl.py:306-461`, default-off),
  reservoir buffers ×2, logit distill, A-GEM-style gradient surgery (proven in the `augmented_loss`
  slot), LwF KL, BN reset. A composed candidate = a ~40-80-line `CLMethod` subclass
  (`@register_method`), hours not days. Known hole: Harness B hooks can't touch the optimizer/LR
  (`generic_cl_runner.py:486`) — LR-schedule mechanisms need a one-line optimizer-aware hook first.
- **Tier 2 (gated): the EXISTING LLM synthesis pipeline** — `method_synthesizer.py` (AST allowlist →
  Docker sandbox → CPU minibench with collapse floor) + `synthesis_loop.py` quarantine with
  **human approval** (`approve_pending_candidate`). Currently OFF (`method_synthesis.enabled` absent,
  zero synthesized methods). Enable ONLY with the quarantine flow; free-form training code never runs
  unreviewed.
- **Protocol-equivalence caveat (must appear in any claim):** everything published to date ran
  Harness A (`run_split_cifar10_benchmark`); candidates run Harness B. Bridge with a one-time
  calibration run (same method/config on both harnesses) and state the caveat in the prereg; defaults
  differ today (e.g. ewc_lambda 1000 registry vs 100 phase18).

*Effort: ~1-2 days. Human gates unchanged: proposals still enter the 24h veto window; ramp still
gates execution; synthesis adoption is human-approved.*

---

## Phase 4 — Ingredient 4: close the learning loop at the DATA level

**No GPU/LoRA self-improvement in v1** (never closed a cycle; not needed). The loop closes with data:

1. **Kill-ledger** (`tar_state/solution_loop/kill_ledger.jsonl`, append-only): per candidate — config
   fingerprint (method_key + mechanism components + key HPs), prediction, outcome verdict, kill reason,
   design-space region eliminated. Written by the Phase-0.3 criteria evaluator on every terminal verdict.
2. **Deterministic pruning:** the proposer receives the ledger AND a hard check rejects any new proposal
   whose fingerprint matches a killed region (code check, not prompt-hope). Extend the live recall-bite
   (repeat-deprioritization) to candidate fingerprints.
3. **Existing loop-closure plumbing to reuse:** verdicts already flow to
   `FrontierProblem.breakthroughs_found/adverse_count/null_count` (`tar_frontier.py:57-59`); the
   write-back → VectorVault → recall path is live from the remediation. The `SeedVarianceReport` /
   `ClaimAcceptancePolicy.max_seed_loss_std` schemas exist (`schemas.py:1401-1442`) but have never
   fired — wire them as the stability-gate implementation rather than inventing new schema.

*Effort: ~1 day.*

---

## Phase 5 — First campaign: "SI-level stability without SI collapse"

1. **Generate:** ~15-20 candidates from the widened proposer (Tier-1 compositions targeting the
   stability/collapse cliff: e.g. path-integral importance × clamped penalty × decay; SI importance +
   collapse-guard reset; hybrid SI/EMA importance; buffer-assisted regularization…). Each preregistered
   with the joint criterion; each must cite its catalog lineage.
2. **Screen (local 4GB, cheap kills):** n=5, kill on collapse (any seed acc < 0.55), kill on clear
   underperformance. GPU-scheduled AFTER the current phase-2 confirmatory queue drains (HPC replication
   + mechanistic ablation own the card now).
3. **Confirm (RunPod, capped):** survivors → n≥20 with the preregistered variance-ratio test + Holm.
   Per-run cost cap unchanged.
4. **Report through governance:** survivors get the full truth-lock treatment. **Both outcomes are
   genuine outputs:** a validated mechanism meeting the joint criterion, OR a mapped, eliminated design
   space + the honest negative ("nothing in this space delivers SI-stability without collapse") — which
   directly feeds the evidence-governance north star.

---

## Sequencing, effort, gates

| Phase | Effort | Blocking? | Human gates |
|---|---|---|---|
| 0 Integrity pre-work | ~1 day | YES — loop unsound without it | review of identity changes |
| 1 Method catalog | ~0.5 day code + 1-2 days curation | needed by 3 | verify every entry (anti-fabrication) |
| 2 Anomaly + joint prereg | ~1 day | needed by 5 | sign off anomaly framing + inventory entry |
| 3 Proposer widening | ~1-2 days | needed by 5 | veto window unchanged; synthesis approval |
| 4 Kill-ledger loop | ~1 day | needed by 5 | — |
| 5 Campaign | compute-bound | after phase-2 queue drains | ramp confirm for autonomous execution (else launched like phase-2 runs) |

**Total build: roughly one focused week + curation.**

## Explicit non-goals (v1)
- No free-form LLM-generated training code outside the sandboxed, human-approved synthesis quarantine.
- No LoRA/weights-level self-improvement.
- No change to the autonomy ramp, veto window, truth-lock, or anti-fabrication rules — the loop's value
  IS that candidates die honestly.
- No claim of external SoTA without a cited bar in curated_external_sota.json.

## ACTIVATION RUNBOOK (run in order, only when safe)

The code is BUILT + committed + tested; it activates at the next safe daemon restart.
Nothing below runs while the HPC confirmatory experiment is training.

1. **Wait for the GPU to free.** The HPC replication + mechanistic ablation own the card.
   Do NOT restart the daemon or run `--apply` while they train. Check: `nvidia-smi`, and the
   phase-2 sequencer log.
2. **Restart the daemon onto the new code** (standard clean restart once GPU idle; the watchdog
   supervisor + git-on-PATH launch). This loads Phase 0-4.
3. **(Optional but recommended) verify the catalog.** Open `literature/method_catalog.json`,
   confirm each `citation` resolves to the named paper, set `"_verified": true`. Until then the
   catalog is usable proposer material but flagged unverified.
4. **Seed the anomaly:** `python scripts/seed_si_anomaly.py --apply`
   (writes the inventory record, the top-composite gap, and the joint-criterion prereg).
5. **Verify readiness:** `python scripts/solution_loop_status.py` — expect
   `anomaly is the top gap = True`, `prereg_criterion=True`.
6. The director (new code) then: picks the anomaly gap -> mints the fp-gap frontier -> the
   WIDENED proposer composes candidates from the catalog -> each enters the 24h veto window.
7. **When ready for autonomous execution:** `python tar_autonomy_ramp.py confirm` (a FRESH
   reauth is required). Candidates screen at n=5 locally (collapse-killed cheaply), survivors go
   to n>=20 on RunPod (per-run cost cap). Kills are pruned via the ledger automatically.
8. **Monitor:** `scripts/solution_loop_status.py` (candidates proposed / killed / survived) and
   the kill-ledger at `tar_state/solution_loop/kill_ledger.jsonl`.

Rollback: everything new is additive or behind flags; reverting the commits + not running
`--apply` returns TAR to its prior behaviour.

## Honest expectations
This makes TAR capable of producing **genuine, validated, usually-incremental solutions** — real
recombinations that survive an incorruptible selector — and calibrated negative maps when nothing
survives. It does not manufacture paradigm shifts. The SI-stability campaign is a real, well-posed,
TAR-discovered problem; either outcome is publishable and feeds the evidence-governance north star.
