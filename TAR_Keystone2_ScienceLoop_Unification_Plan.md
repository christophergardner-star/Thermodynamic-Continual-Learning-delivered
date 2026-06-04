# TAR — Keystone #2: Science-Loop Unification (SCOPE / build-ready plan)

**Authored:** 2026-06-04 · **Status:** SCOPED, not implemented (hold) · **Branch:** phase0-stackbridge
**Depends on:** Keystone #1 (truth-lock) ACTIVE. **Companions:** `TAR_Autonomy_Unification_Roadmap.md`,
`TAR_TruthLock_Implementation_Plan.md`, `TAR_Phase0_and_StackBridge_Implementation_Plan.md`.
**Derivation:** read-only; all file:line anchors verified against current source 2026-06-04.

---

## 0. Goal, scope, guardrails

**Goal:** collapse TAR's two disconnected research loops into ONE closed, self-driving, *truthful*, multi-domain
loop — so the live daemon can discover real gaps, run experiments in any supported domain, verify them, write the
results back into the memory it recalls, and let the next cycle build on them. This is the roadmap's move #2.

**The gap today (verified by the 10-agent audit):**
- The tested `science_exec` bridge exists but **no production code emits `runner_key="science_exec"`** (only the test).
- The real idea engine (`scan_frontier_gaps`) lives in the unsupervised Stack-B; the daemon never calls it.
- Experiment results are **never written back** to the store the director recalls (its decision tables are 0 rows);
  the VectorVault is built but never queried by the live director. So **research does not compound.**

### Guardrails (apply to the eventual implementation)
- 🚫 Do not implement until **truth-lock is ACTIVE** (else unverified multi-domain results would count). K2 feeds
  results into the publication path; that path must be truth-locked first.
- 🚫 Keep autonomy increases behind the existing rails: per-domain `_FRONTIER_AUTONOMY_DOMAINS` (default empty),
  the `FrontierRegistry.register` `well_known_problem` guard, the autonomy ramp, the 24 h veto, and the kill-switch flags.
- 🟢 Each sub-keystone lands behind a flag / per-domain opt-in and is inert until explicitly enabled.
- ⚠️ **Integrity rule (critical):** internal TAR results written back to the recall store MUST be tagged
  `source="tar_internal"` and excluded from the external-SoTA used by `NoveltyGate`, to prevent **circular
  self-validation** (TAR citing its own result as the prior art it must beat).

### Dependency ordering
Truth-lock ACTIVE → **K2.1 emit bridge** → **K2.2 write-back + recall** → **K2.5 truth-lock integration check** →
**K2.3 idea source** (the hardest; needs the literature DB repopulated) → **K2.4 one entrypoint**. Per-domain
enablement is last and human-gated.

---

## K2.1 — Emit the bridge (unify dispatch across domains)

**Today:** `_run_science_exec` + `_build_study_payload_from_spec` + `_adapt_science_report` exist and are tested
(`tar_experiment_orchestrator.py:2222+`), reached by `_dispatch` when `spec.runner_key == "science_exec"`
(`:2200`). But nothing sets that runner_key in production.

**Changes (exact anchors):**
1. **Science profiles carry the runner + an experiment plan.** `science_profiles/*.json` currently have
   `experiment_templates` (list of `{template_id, name, hypothesis, benchmark, metrics, parameter_grid}`) but **no
   `runner_key`**. Add `"runner_key": "science_exec"` to each domain profile that should route through the bridge
   (quantum_ml, graph_ml, generic_ml, computer_vision, …). The director already propagates it:
   `tar_research_director.py:2387` (`if profile.get("runner_key"): exp_entry["runner_key"] = profile["runner_key"]`).
2. **Thread the experiment plan into config_overrides.** `_build_study_payload_from_spec`
   (`tar_experiment_orchestrator.py`) requires `config_overrides["experiments"]` (ProblemExperimentPlan-shaped). The
   directive builder must map the profile's `experiment_templates` → `config_overrides["experiments"]` (template_id,
   name, benchmark, metrics, parameter_grid map almost 1:1) plus `domain`, `benchmark_tier`, optional
   `baseline_metric_value` (so `_accuracy_verdict` can yield a non-NULL verdict).
3. **Stop the generic_cl override for science_exec.** `tar_living_research.py:_specs_from_directives` (~:1442-1509)
   reads `directive["runner_key"]` (`:1447`) but **overrides it to `generic_cl` for any non-native method**
   (`:1448-1482`) — and a science_exec directive's "method" is a metric, not a CL method. Add a guard: if
   `directive.get("runner_key") == "science_exec"`, keep it and skip the native-method/synthesis branch. Pass through
   at `:1509` (already passes `runner_key=_runner_key`).

**Acceptance:** with a quantum_ml (CPU) profile opted in, the daemon emits a `runner_key="science_exec"` directive →
`_dispatch` → `_run_science_exec` → real PennyLane run → `ExperimentResult` (accuracy-domain verdict) →
`_save_result`. (Test mirrors `tests/test_science_exec_bridge.py` but from a directive.)

**Compute note:** science_exec runs in-process in the daemon → real torch/PennyLane. On the 4 GB card keep to
`validation` tier + CPU domains (quantum sim, tabular, tiny-CV); route heavy domains to RunPod (K2.4 / roadmap).

---

## K2.2 — Write results back to memory + recall it (make research compound)

**Today:** `finalize_autonomous_results` (`tar_living_research.py:1661`) ingests results → archive + `FrontierRegistry.record_*`,
but **nothing writes the result into the store the director recalls.** The director recalls only
`LiteratureKnowledgeGraph` (`tar_research_director.py:1359-1371` NoveltyGate, `:1392-1400` GapDetector), whose
`research_gaps`/`sota_entries` tables are **0 rows**. The VectorVault (`tar_lab/memory/vault.py:VectorVault.search` :612)
is searched only by Stack-B (`tar_lab/orchestrator.py`), never by the live director.

**Changes:**
1. **Write-back on finalize.** In `finalize_autonomous_results` (`:1661`), for each terminal result, upsert into the
   SAME db the director queries via `LiteratureKnowledgeGraph`:
   - `upsert_sota_entry(SoTAEntry(...))` (`literature/knowledge_graph.py:463`) for the (method × benchmark × metric)
     result — **tagged `source="tar_internal"`** (integrity rule).
   - `upsert_gap(ResearchGap(...))` (`:580`) when a frontier is opened/closed by the result.
   This makes the director's NoveltyGate/GapDetector see TAR's own results next cycle.
2. **Recall the VectorVault in the director.** Wire `VectorVault.search` (`:612`) /
   `search_similar_trials` (`:661`) into `update_state` / `_build_experiment_directives`
   (`tar_research_director.py:220` / `:2424`) so prior trials inform the next directive (e.g. avoid repeating a
   configuration that already failed). Also re-sync the vault (stale since May 3) and index all 1,514 papers.
3. **Reward, not just penalize (compounding).** Wire the dormant CPU learners' `load_*` priors (calibration,
   method_refinement, operational, authoring) into a *positive* signal for the next directive, complementing the
   existing penalty-only loops (roadmap move #4 first step).

**Acceptance:** after an experiment finalizes, `get_top_gaps(domain=…)` / NoveltyGate reflect the new internal result
(tagged internal); a repeated-config directive is de-prioritized by a vault hit. Integrity test: an internal SoTA
entry is **excluded** from the external-SoTA NoveltyGate compares against.

---

## K2.5 — Truth-lock integration (counts honestly)

science_exec results already flow `_adapt_science_report → ExperimentResult → _save_result`, so **TL-1**
(verify + quarantine) and **TL-4** (publication gate) apply automatically. Confirm:
- The bridge result is written with a `result_env.json` carrying `git.head` + `authorization.manifest_hash` (it
  inherits `_execute`'s manifest handling / autonomous self-manifest) so `verify_canonical_3gate` (gate-1/2) passes.
- Accuracy-domain results carry `family_wise_significant` only when a corrected comparison is actually done (else
  they're trusted-but-not-publication-grade — correct).
- Method-identity is non-TCL for these domains → `method_ok` True.
**Acceptance:** a bridged result that verifies is `canonical_verified=True`; one lacking provenance is quarantined.

---

## K2.3 — Unify the idea source (real autonomous gap discovery) — HARDEST

**Today:** `scan_frontier_gaps` (`tar_lab/orchestrator.py`) is the real, arXiv-grounded gap engine but the daemon
never calls it; the literature decision tables are empty; novelty runs in lexical-fallback (no embedder); 91 scans
produced 1 (rejected) gap.

**Changes:**
1. **Repopulate + schedule the literature DB.** Make `ExternalEvidenceIngestor` actually write
   `papers/benchmarks/sota_entries/research_gaps` with **retry/backoff** (the daily cycle currently ends `ok:false`
   on `http_429`). Seed per-domain `_EXPECTED_BENCHMARKS` (`literature/gap_detector.py:68` only covers CL/vision/NLP/GNN).
2. **Engage the embedder** (all scans are lexical-fallback) so novelty/similarity are semantic.
3. **Feed gaps → frontiers, guarded.** Call `scan_frontier_gaps` from the daemon (or inside `update_state`); route
   surviving candidates through `_pick_novel_problem_for_domain` (`tar_research_director.py:1384`) →
   `FrontierRegistry.register`. **Extend the candidate to populate the registry's required fields**
   (`external_baselines`, `candidate_datasets`, `candidate_backbones`) so it satisfies the `well_known_problem` guard
   (`tar_frontier.py:372`) — today gap candidates lack these and would be rejected. Enable per-domain via
   `_FRONTIER_AUTONOMY_DOMAINS` (`tar_research_director.py:44`), one domain at a time, behind the veto window.

**Acceptance:** with `continual_learning` opted in and the DB repopulated, a real arXiv-grounded gap becomes a
registered frontier (guard satisfied) → a queued experiment, end to end, under the veto window.

**Risk:** this is the most uncertain part (depends on external-API ingestion quality). De-risk by ALSO deriving gaps
from internal results (K2.2 write-back) so the loop compounds even before external ingestion is rich.

---

## K2.4 — One supervised platform entrypoint

**Today:** control is fragmented — the watchdog (`tar_watchdog.py`) supervises exactly 3 services
(living_research_daemon, queue_maintainer, dashboard); Stack-B's `serve_forever_full` is unsupervised; there's no
single "run the platform" command.

**Changes:** add a `tar_living_research.py --platform` (or a thin launcher) that starts watchdog → daemon →
queue-maintainer → dashboard as one supervised unit; fold Stack-B's `science_exec` + gap engine into the daemon as a
**library** (retire the parallel `TAROrchestrator` daemon) so there is one orchestrator + one runner registry.

**Acceptance:** one command brings up the whole platform under the watchdog; only one orchestrator drives experiments.

---

## Risks & integrity summary
- **Circular self-validation** — the #1 risk: tag internal results `source="tar_internal"` and exclude from
  external-SoTA novelty. (K2.2)
- **GPU ceiling** — science_exec runs in-process; keep heavy domains on RunPod/3090 (4 GB does quantum-CPU/tabular/tiny-CV).
- **Idea quality** — K2.3 depends on repopulating the literature DB; de-risk with internal-result-derived gaps.
- **Autonomy** — every enablement is per-domain, behind the ramp + veto + frontier guard; truth-lock must be active first.

## Definition of done (Keystone #2)
One supervised daemon that: scans real gaps → mints guarded frontiers (per-domain opt-in) → runs experiments in any
supported domain via the bridge → truth-lock-verifies them → writes results back into the recalled memory →
de-prioritizes repeats and rewards promising directions next cycle — i.e., **research that compounds, across domains,
truthfully.** Then the roadmap continues to synthesis (move #4) and the guarded frontier loop (move #5).

---

## STATUS: SCOPED ONLY — HOLD
No K2 code written. Implement only after truth-lock is active (phase18 result → dry-run → dashboard restart) and on
explicit go, sub-keystone by sub-keystone, each behind a per-domain flag.
