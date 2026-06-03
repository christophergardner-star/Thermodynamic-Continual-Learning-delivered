# TAR — Phase-0 + Stack-Bridge Implementation Plan (Keystone)

**Authored:** 2026-06-03 · **Status:** build-ready · **Companion:** `TAR_Five_Domain_Autonomy_Roadmap.md`
**Derivation:** independent read-only audit (line numbers verified against current source).
**Code root:** `C:\Users\cgard\TAR\Thermodynamic-Continual-Learning-delivered`
**State root (symlinked from C:):** `E:\TAR\Thermodynamic-Continual-Learning-delivered`

---

## 0. Scope, guardrails, and execution windows

This plan delivers two things, in order:
- **Part A — Phase 0: Truth & Safety Foundation** (must precede any domain expansion).
- **Part B — The Stack-A↔Stack-B Bridge** (the keystone that unlocks multi-domain autonomy).

### Hard guardrails (apply to every task here)
- 🚫 **Do not disturb the live HPC run** (PID 17896, `run_hpc_replication.py`, ~seed 15/20). Do not touch the GPU,
  `tar_state/experiment_queue.json`, `tar_state/execution_enabled.flag`, `tar_state/daemon_paused.flag`, or
  `tar_state/comparisons/hpc_replication_checkpoint.json`.
- 🟢 **CPU/IO/code tasks are safe now.** **GPU experiments are NOT** — design now, run on RunPod or after the local
  HPC run completes (ETA ~39 h from 2026-06-03 ~18:00).
- 🟢 Keep the daemon **down** (it is, deliberately) until Part B is tested.
- 🚫 Per standing rule, do not `git add -A`; stage explicitly (see A0.1).

### Execution-window legend (used in the task board, Part C)
- **NOW-SAFE** = pure code/IO/CPU, no GPU, no live-run contact.
- **AFTER-HPC** = needs the local GPU; run after PID 17896 finishes (or on RunPod).
- **GATED** = requires an explicit human decision/approval.

---

# PART A — Phase 0: Truth & Safety Foundation

## A0.1 — Protect the uncommitted autonomy work  *(NOW-SAFE · Owner: Infra + Provenance)*

**Why:** the *running* director already imports four untracked modules
(`tar_research_director.py:2434-2464` → `human_feedback_learning`, `outcome_learner`, `calibration_learner`,
`method_refinement_engine`). Losing the working tree loses functional code. Local `main` is **44 commits unpushed**.
A blanket `git add -A` would also sweep 16 MB `xdg/` + 1.6 MB `pip/` (un-ignored) and the dirtied runtime file
`manifests/active_session.json` (flipped to `AUTONOMOUS_MODE`).

**Steps**
1. Append to `.gitignore`: `xdg/`, `pip/` (vendored tool caches; confirm they are not referenced by code first).
2. Restore the runtime-state file out of the commit set: leave `manifests/active_session.json` unstaged (it is
   generated). Do **not** commit `AUTONOMOUS_MODE`.
3. **Selectively** stage and commit the self-improvement stack (these are real, imported code):
   - `tar_autonomy_ramp.py`
   - `tar_lab/human_feedback_learning.py`, `tar_lab/outcome_learner.py`, `tar_lab/calibration_learner.py`,
     `tar_lab/method_refinement_engine.py`, `tar_lab/operational_learner.py`, `tar_lab/authoring_learner.py`,
     `tar_lab/append_only_guard.py`
   - `tests/test_append_only_self_improvement.py`
   - `docs/self_improvement_untapped_execution_spec.md`
4. Review the 12 modified tracked files; commit the genuine code changes (dashboard/director/orchestrator/
   scheduler/living_research/schemas/run_hpc_replication) in a focused commit. **Exclude** generated/auto files.
5. `git push` the backlog (44 + new) once the working tree is clean of junk.

**Acceptance:** `git status` shows no `xdg/`, `pip/`, or `active_session.json` staged; the 6 learner modules +
`append_only_guard` + tests are tracked; `git log origin/main..HEAD` is empty after push. Per the standing
**unknown-files rule**, get explicit user confirmation before staging any file >100 lines not authored this session.

## A0.2 — Disk + stale-state hygiene  *(NOW-SAFE · Owner: Infra)*

- C: has **~10 GB free**. Move paper LaTeX builds to E:; prune duplicate C:-side `paper/` artifacts; ensure no
  checkpoints write to C:.
- Reconcile stale markers so the operator console stops lying about dormancy:
  - `tar_state/DORMANT.json` + `tar_state/active_session.json` claim "dormant since 2026-05-30" while a real run is
    live. Update/retire them to reflect "manual Phase-2/3 execution active."
  - Stale PID fields: `active_preregistration.json` (names old pid 26760; live is 17896); `watchdog_state.json`
    (last 12:57). Refresh or annotate.
  - `autonomy_ramp.json` is **0 bytes** — see A0.3.

## A0.3 — Arm alignment (ramp + veto)  *(NOW-SAFE · Owner: Alignment Officer)*

Today the safety rails are *passive*: the autonomy ramp is a 0-byte no-op
(`tar_autonomy_ramp.py:116-117` → `is_full_autonomy` returns `True` when no `autonomy_ramp.json`), and the human
review is opt-out (24 h auto-approve).

- Initialize the ramp (`init_ramp`) at a conservative tier so `is_full_autonomy` gates director-generated
  experiments at `tar_scheduler.py:332-357` until an explicit `confirm_promotion()` / `autonomy_ramp_confirm.flag`.
- Decide and document the veto policy (who watches the 24 h window; what auto-approve means). This is an alignment
  decision, not code — record it in `manifests/provenance/`.

## A0.4 — Fix the flagship science: `tcl_canonical` + new `tcl_full`  *(DESIGN now-safe; RUN after-HPC/RunPod · Owners: CL/Thermo Scientist + Methodologist)*

**Problem (verified in source):** every published "TCL" number uses `method="tcl"`, which
`multimodal_payloads.py:1058-1072` documents as a **D_PR-scaled uniform L2 anchor to the last task** (+ a regime
observer LR scaler). The real per-element importance method runs only under `method="tcl_canonical"`
(`:1074-1077, 1135-1137`) — which has **no** observer. **No single path runs the full intended algorithm
(importance + thermodynamic regime LR control).** Fix = add `method="tcl_full"` and benchmark all variants at the
protocol of record.

### A0.4.1 Add a decoupled canonical lambda  *(NOW-SAFE)*
- `tar_lab/schemas.py:251` `ContinualLearningBenchmarkConfig(StrictModel)` forbids extra fields. Add:
  `tcl_canonical_lambda: float = 0.01`. Today canonical penalty silently rides `tcl_penalty_lambda`
  (`multimodal_payloads.py:940`). Point the canonical penalty at the new field.

### A0.4.2 Thread a new `tcl_full` method  *(NOW-SAFE)*
In `multimodal_payloads.py`, introduce helper sets near the method dispatch:
`_TCL_OBS = {"tcl", "tcl_full"}` and `_TCL_CANON = {"tcl_canonical", "tcl_full"}`. Then change the **exact**
method-name guards (verified line numbers):

| Line | Current guard | Change to | Purpose |
|---|---|---|---|
| `:896` observer creation | `method == "tcl"` (and `tcl_governor_enabled`) | `method in _TCL_OBS` (+ force observer on for `tcl_full`) | give `tcl_full` an observer |
| `:962` task-boundary reset | `method == "tcl"` | `method in _TCL_OBS` | |
| `:969` canonical importance init | `method == "tcl_canonical"` | `method in _TCL_CANON` | give `tcl_full` importance |
| `:1076` canonical penalty | `method == "tcl_canonical"` | `method in _TCL_CANON` | canonical penalty for `tcl_full` |
| `:1136` importance accumulate | `method == "tcl_canonical"` | `method in _TCL_CANON` | |
| `:1142` observer LR block | `method == "tcl"` | `method in _TCL_OBS` | regime LR for `tcl_full` |
| `:1183` per-epoch regime trace | `method == "tcl"` | `method in _TCL_OBS` | |
| `:1267` canonical commit | `method == "tcl_canonical"` | `method in _TCL_CANON` | |

**DO NOT change** the uniform-anchor blocks `:1066` (D_PR L2 penalty) and `:1273/:1279` (`anchor_snapshot()` +
D_PR anchor) — keep them `"tcl"`/`"tcl_penalty_only"`-only. Otherwise `tcl_full` would stack the canonical penalty
**and** the uniform anchor. (This is the #1 correctness risk for this task.)

### A0.4.3 New phase script `phase18_tcl_canonical_fullprotocol.py`  *(DESIGN now; RUN after-HPC/RunPod)*
Model on `phase10_baseline.py`. Hold the **protocol of record** constant so results compare to the published
0.1161 TCL-in-suite value:
- `SEEDS=[42,0,1,2,3]`, `BACKBONE="resnet18"`, `EPOCHS=40`, default fixed class order `[[0,1]…[8,9]]`,
  `ewc_lambda=100.0`, `si_c=0.01`, `si_xi=0.001` (Phase-13 best — **not** schema defaults 1000/0.1 which collapse).
- `METHODS=["sgd_baseline","ewc","si","tcl_canonical","tcl_full"]`; set `tcl_governor_enabled=True` so `tcl_full`
  gets its observer.
- **Lambda sweep** for the untuned canonical penalty: `tcl_canonical_lambda ∈ {0.001, 0.01, 0.1, 1.0}` for both
  `tcl_canonical` and `tcl_full` (headline can fix 0.01; sweep as secondary).
- Add `"phase18_tcl_canonical_fullprotocol"` to the manifest-gate accepted ids (mirror `phase10_baseline.py:62`).
- Write results via `write_canonical_comparison_result(logical_name="phase18_tcl_canonical_fullprotocol",
  phase_number=18)` → `tar_state/comparisons/`.
- **Stats:** primary metric `mean_forgetting`; comparators = EWC and the published 0.1161. With 2 TCL variants ×
  {EWC, SI, SGD} = 6 primary tests, **apply Holm–Bonferroni (`bonferroni_k=6`) and pre-register it.**
- **Power:** n=5 detects only d≈1.5+. Pre-register **n=10–20** for any confirmatory canonical-vs-EWC claim; n=5 is a
  directional pilot only — state this limit in the prereg.
- **New prereg:** `tar_state/preregistrations/phase18_tcl_canonical_fullprotocol.json` (PreRegistrationRecord v1.0):
  hypothesis "canonical/full TCL ≤ EWC forgetting at full protocol", `primary_outcome: mean_forgetting`,
  `test_type: wilcoxon_signed_rank`, seeds, `bonferroni_k: 6`, `min_detectable_effect_d`, empty `amendment_log`.
- **Runtime:** ~2.8 h/seed (confirm against HPC timing log). Headline ≈ ~14 h; full λ-sweep ≈ 2–3 days.
  **Run on RunPod or after PID 17896.**

**Why this is the highest-value experiment in the project:** it is the first time the *named* algorithm is the
*measured* algorithm. Every downstream domain claim inherits credibility (or not) from this.

## A0.5 — Fix the SPRT input + log a prereg amendment  *(NOW-SAFE code; applies to the NEXT run)*

**Problem (verified):** `run_hpc_replication.py:618-631` rebuilds `batch_p_values` as **nested cumulative**
Wilcoxon p-values on prefixes `hpc_vals[:i]`, then `sprt_boundary` (`stat_utils.py:792-796`) treats each as an
independent Wald increment. The elements are strongly correlated (each early seed appears in all later prefixes), a
single Wilcoxon p is not a likelihood ratio, and early prefixes are structurally floored — biasing toward
`accept_H0` (this caused the n=4 false accept at d=−1.17). The `SPRT_MIN_SEEDS_FOR_CHECK=8` floor (`:66-77`) was
itself added *after* seeing that outcome.

**Fix (minimal, recommended):** feed `sprt_boundary` **one independent per-seed increment** — for seed k,
`delta_k = hpc_k − baseline_k`; emit a per-seed p-proxy of `alpha/2` when `delta_k < 0` else `1−alpha/2`. Each seed
contributes exactly one Wald increment (the i.i.d. sign-test structure the comment at `:601-605` intends). Keep the
cumulative Wilcoxon only as a reported diagnostic (`running_wilcoxon_p`, `:645`), not as SPRT input.

**Fix (rigorous default for the writeup):** abandon sequential early-stop; run **fixed-n Wilcoxon at the
pre-registered n=20** (`REPLICATION_SEEDS`, `:40`) and report one p + Cohen's d. The prereg already names
`wilcoxon_signed_rank`, n=20, α=0.05, direction `less` — fully consistent, removes the SPRT burden.

**Amendment:** append `A2` to `amendment_log` in
`tar_state/preregistrations/hpc_replication.json` (`PREREG_FILE`, `:79`): `field_changed: "stopping_rule"`
(or `"test_type"` if switching to fixed-n), `old_value`/`new_value`, reason = nested-cumulative-p SPRT bug.
**Do not apply to the in-flight run**; apply to the next replication or the fixed-n re-analysis.

---

# PART B — The Stack-A ↔ Stack-B Bridge (Keystone)

**Goal:** let the live daemon dispatch Stack-B multi-domain experiments **through the same `_execute` path** so they
inherit RAIL-3 manifest, RAIL-4 veto, runtime lease, append-only archive, and env-snapshot provenance for free.

## B1 — Schemas (verified)

**`ExperimentSpec`** — `tar_experiment_orchestrator.py:217-284` (`@dataclass`). Key fields:
`name, project_id, hypothesis_name, dataset, method, seeds, config_overrides, runner_key (PRIMARY selector),
runtime_context (carries domain_id into the lease), frontier_problem_id, context`. **There is no `domain` field** —
carry the Stack-B domain via `config_overrides["domain"]` (and `runtime_context["domain_id"]`).

**`ExperimentResult`** — `tar_experiment_orchestrator.py:289-319` (`@dataclass`):
`seed_results: list[dict]`, `mean_forgetting/std_forgetting/mean_accuracy/std_accuracy`,
`baseline_forgetting, mean_delta, t_stat, p_val, cohens_d, n_better`,
`verdict ∈ {BREAKTHROUGH,DIRECTIONAL,NULL,ADVERSE,ERROR}`, `notes`, `power_analysis`, `confidence_score`.
**Forgetting-centric (lower-is-better).**

**Stack-B return** — `ProblemExecutionReport` (`schemas.py:1271-1310`): `status ∈
{completed,dependency_failure,partial_failure,failed}`, `domain`, `experiments: List[ProblemExperimentResult]`,
`benchmark_statistical_summary`, `manifest_path/hash`, `summary`. Each `ProblemExperimentResult`
(`schemas.py:1248-1269`): `status`, **`metrics: Dict[str,float]`** (free-form), `statistical_summary`,
`canonical_comparable`, `proxy_benchmark_used`, `execution_mode`.

## B2 — Stack-B entry contract (call the pure core, not the subprocess)

`tar_lab/science_exec.py:303` —
`execute_study_payload(payload: dict, artifact_path: Path) -> ProblemExecutionReport`.
- **INPUT** `payload` keys: `domain` (`:306`, dispatch key), `problem_id, problem, profile_id, benchmark_tier,
  requested_benchmark, canonical_only, no_proxy_benchmarks, environment.validation_imports,
  benchmark_availability, experiments: list[dict]` (each `template_id, name, benchmark, metrics, parameter_grid,
  benchmark_spec`).
- **Dispatch** `:343-358`: `_execute_<domain>` for `quantum_ml | deep_learning | computer_vision | graph_ml |
  natural_language_processing | reinforcement_learning | generic_ml`, else `_execute_generic_domain` (`:411`,
  refuses with `status="failed"`).
- Calling in-process is **acyclic** (`science_exec` imports only `tar_lab.schemas/benchmark_stats/thermoobserver`).

## B3 — Bridge design (insertion point + skeletons)

**Insertion point:** one new branch in `_dispatch` (`tar_experiment_orchestrator.py:~2199`, before the `dataset`
fallback):
```python
if spec.runner_key == "science_exec":
    return self._run_science_exec(spec)
```

**New methods** (mirror `_run_nlp_continual`; return an `ExperimentResult`):
```python
def _run_science_exec(self, spec: ExperimentSpec) -> ExperimentResult:
    from tar_lab.science_exec import execute_study_payload
    payload  = self._build_study_payload_from_spec(spec)
    artifact = self._exp_dir / spec.id / "environment_probe.json"
    report   = execute_study_payload(payload, artifact)      # in-process Stack-B core
    return self._adapt_science_report(spec, report)

def _build_study_payload_from_spec(self, spec) -> dict:
    ov = dict(spec.config_overrides or {})
    return {
        "problem_id": spec.frontier_problem_id or spec.id,
        "problem": spec.hypothesis_name,
        "profile_id": ov.get("profile_id", spec.project_id),
        "domain": ov.get("domain") or (spec.runtime_context or {}).get("domain_id", "generic_ml"),
        "benchmark_tier": ov.get("benchmark_tier", "validation"),
        "requested_benchmark": ov.get("requested_benchmark"),
        "canonical_only": ov.get("canonical_only", False),
        "no_proxy_benchmarks": ov.get("no_proxy_benchmarks", False),
        "environment": {"validation_imports": ov.get("validation_imports", [])},
        "benchmark_availability": ov.get("benchmark_availability", []),
        "experiments": ov["experiments"],   # list[dict], ProblemExperimentPlan-shaped
    }

def _adapt_science_report(self, spec, report) -> ExperimentResult:
    completed = [e for e in report.experiments if e.status == "completed"]
    if not completed:
        return ExperimentResult(..., verdict="ERROR", notes=f"science_exec status={report.status}")
    primary = (spec.config_overrides or {}).get("primary_metric", "accuracy")
    vals = [e.metrics.get(primary, float("nan")) for e in completed]
    # accuracy-domain verdict — DO NOT route through _build_result (forgetting-only)
    return ExperimentResult(
        experiment_id=spec.id, ..., dataset=report.domain, method=primary,
        seed_results=[{"metric": primary, "value": v} for v in vals],
        mean_accuracy=_mean(vals), std_accuracy=_std(vals),
        mean_forgetting=0.0, std_forgetting=0.0,     # not applicable; see field table
        verdict=_accuracy_verdict(vals, spec), notes=report.summary,
        power_analysis=report.benchmark_statistical_summary or {},
    )
```
Add a small **accuracy-domain verdict helper** (`_accuracy_verdict`) — higher-is-better, vs an external baseline
with p<α → `BREAKTHROUGH`/`DIRECTIONAL`; else `NULL`. **Never** call `_build_result` from the bridge (it computes
deltas vs the TCL *forgetting* baseline, which would mislabel accuracy results).

## B4 — Field mapping (Stack-B → `ExperimentResult`)

| Stack-B | `ExperimentResult` | Note |
|---|---|---|
| `report.domain` | `dataset` / `notes` | no dedicated domain field |
| `e.metrics[primary]` per completed exp | `seed_results[i]`, `mean_accuracy`, `std_accuracy` | choose `primary_metric` from spec |
| — (no forgetting concept) | `mean_forgetting`/`std_forgetting` = `0.0` | **MISSING on Stack-B**; verdict from accuracy gate |
| `report.benchmark_statistical_summary` / `e.statistical_summary` | `p_val`, `cohens_d`, `power_analysis` | adapter computes from per-seed metric list |
| `report.status` + `canonical_comparable` | `verdict` | accuracy-oriented map |
| `report.manifest_path/hash`, `e.metrics`, `report.summary` | `notes` + persisted raw report | see B5 |

## B5 — Provenance (mostly free, one gap to close)

Because the bridge dispatches through `_execute`, every Stack-B run **automatically** gets: RAIL-3 manifest
(auto-generated+git-committed in autonomous mode, `:1711`), RAIL-4 veto re-check (`:1778`), runtime lease (`:1832`,
records `runtime_context["domain_id"]`), append-only archive (`_archive_terminal_experiment`, `:715`), and the
`collect_environment_snapshot` env snapshot in `_save_result` (`:3132`).
**Gap to close:** Stack-B's own `report.manifest_hash`/attestation is richer than `ExperimentResult` exposes —
extend `_save_result` to also dump the raw `report.model_dump_json()` next to `result.json`.

## B6 — Policy/gate changes (exact lines; keep the frontier guard)

To let the daemon *propose + run* a non-CL domain:
- `tar_lab/science_profiles.py:35` `_ACCEPTED_DOMAINS = frozenset({"continual_learning","thermodynamics_ml",
  "other_ml"})` (checked `:66`) — **add** target domains (e.g. `"computer_vision"`, `"graph_ml"`, later
  `"quantum_ml"`).
- `tar_lab/domain_classifier.py:27-31` finance hard-block — **leave intact** for now (new domains are non-financial).
  Finance gets unblocked only in Phase 3 with its own data/adapter + leakage discipline.
- `tar_research_director.py:34` `_STRICT_REAL_WORLD_FRONTIER_ONLY = True` (short-circuits at `:1157, :1363, :2220`)
  — **gate per-domain**, do not globally flip. Relax only after confirming the registry guard still holds.
- **KEEP** `tar_frontier.py:372` `register()` guard (`well_known_problem=True` + named external baselines /
  candidate datasets / candidate backbones + explicit `domain`). This is orthogonal to dispatch and must not be
  relaxed — it is what prevents TAR inventing ungrounded "frontiers."

## B7 — Risks

1. **Metric-semantics mismatch (highest):** Stack-A verdict = forgetting/lower-better w/ TCL baseline; Stack-B =
   accuracy/higher-better, no forgetting. Mitigation: dedicated `_accuracy_verdict`; never reuse `_build_result`.
2. **Result-schema impedance:** `metrics` is free-form `Dict[str,float]`; missing primary → NaN → set
   `verdict="ERROR"`.
3. **Generic/benchmark refusal:** `_execute_generic_domain` / `_benchmark_gate` return `status="failed"` for
   unaligned benchmarks — surface as `ERROR`/`NULL`, do not crash.
4. **Provenance gap:** Stack-B manifest not propagated — fix in B5.
5. **GPU/VRAM:** Stack-B executors run real torch in-process; on the 4 GB GTX 1650 keep to `validation`/smoke tiers,
   honor `hardware_budget`. Heavy domains → RunPod.
6. **Import cycle:** none (verified acyclic).

## B8 — Safe implementation order

1. Add pure functions `_build_study_payload_from_spec`, `_adapt_science_report`, `_accuracy_verdict`
   (unit-testable, no rails).
2. Add the `runner_key == "science_exec"` branch in `_dispatch`. Test with a hand-built spec, `_autonomous=False`,
   manifest supplied — exercises the full `_execute` path incl. provenance.
3. Extend `_save_result` to persist the raw `ProblemExecutionReport` + manifest hash (B5).
4. Then relax gates: add domains to `_ACCEPTED_DOMAINS` (B6).
5. **Last**, enable director auto-proposal of non-CL frontiers (per-domain gating of `_STRICT_…`), keeping
   `tar_frontier.py:372` untouched.

## B9 — Acceptance tests

- **Unit:** `_adapt_science_report` maps a synthetic `ProblemExecutionReport` (completed `generic_ml`, `metrics`
  with `accuracy`) to a valid `ExperimentResult` with a correct `verdict`; an all-`failed` report → `verdict="ERROR"`.
- **Integration (CPU, NOW-SAFE):** dispatch a `runner_key="science_exec"` spec for `generic_ml` (sklearn
  breast-cancer / OpenML Adult — already implemented in Stack-B) through `_execute` with `_autonomous=False` + a
  supplied manifest; assert `result.json`, the raw report dump, and an env snapshot are written, and the archive +
  `FrontierRegistry.record_*` are exercised.
- **Provenance:** assert the run produced a manifest hash and that append-only refused an overwrite.
- **No regression:** existing CL `_dispatch` branches and `_build_result` paths unchanged (run the CL smoke tests).

---

# PART C — Task board mapped to the Lab

| ID | Task | Owner | Window | Depends on | GPU |
|---|---|---|---|---|---|
| A0.1 | gitignore + selective commit + push SI stack | Infra + Provenance | NOW-SAFE | — | no |
| A0.2 | free C: + reconcile stale state | Infra | NOW-SAFE | — | no |
| A0.3 | arm autonomy ramp + veto policy | Alignment | NOW-SAFE | A0.1 | no |
| A0.4.1 | add `tcl_canonical_lambda` to config | CL/Thermo Sci | NOW-SAFE | — | no |
| A0.4.2 | thread `tcl_full` method (8 guards) | CL/Thermo Sci | NOW-SAFE | A0.4.1 | no |
| A0.4.3 | `phase18` script + prereg (design) | CL/Thermo + Methodologist | DESIGN NOW-SAFE; RUN AFTER-HPC/RunPod | A0.4.2 | yes (run) |
| A0.5 | SPRT input fix + amendment | Methodologist | NOW-SAFE (next run) | — | no |
| B1–B5 | bridge code (adapters + `_run_science_exec` + `_save_result` dump) | Domain-Extension Eng | NOW-SAFE | — | no |
| B6 | gate relaxations (per-domain) | Domain-Extension + Alignment | NOW-SAFE | B1–B5 | no |
| B9 | acceptance tests (unit + CPU integration) | Domain-Extension + Infra | NOW-SAFE | B1–B5 | no |
| — | enable director non-CL proposals | PI + Alignment | GATED | B6, B9 | no |

---

# PART D — Sequencing (what runs now vs after the HPC run)

**Do now (no GPU, no live-run contact):** A0.1 → A0.2 → A0.3; A0.4.1 → A0.4.2; A0.5 (code); B1–B5 → B6 → B9. This
protects the work, arms safety, makes the code true, and lands the keystone bridge — all without touching the GPU.

**Do after the local HPC run finishes (or on RunPod):** A0.4.3 (run `phase18` — the canonical/full head-to-head),
then a fixed-n re-analysis of the HPC replication per A0.5.

**Then enter Phase 1–2 of the roadmap:** close the loops, and turn on the bridged daemon for the first multi-domain
(CV/graph/generic, then thermo) autonomous experiment — under armed rails.

**Definition of done for this keystone:** the daemon can autonomously propose, run (via `science_exec`), evaluate,
and archive a non-CL experiment with full provenance and a correct accuracy-domain verdict — and the flagship CL
science finally benchmarks the algorithm it claims (`tcl_full`).
