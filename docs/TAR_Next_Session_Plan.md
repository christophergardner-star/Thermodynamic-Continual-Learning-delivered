# TAR — Next Session Plan (from 2026-07-08 handoff)

**One-line goal:** close the last gap so the autonomous loop actually runs one end-to-end cycle
on the SI anomaly — everything upstream (key, proposer, synthesis) is fixed and proven; the
daemon's *director* just isn't chaining them yet.

---

## 0. Current state (verified 2026-07-08 — do not re-derive)

- **TAR is intentionally DOWN** (to save API credit): supervisor task disabled, watchdog/daemon/
  queue-maintainer/operator-agent/dashboard stopped, `tar_state/daemon_paused.flag` set. Only
  `sync_research.py` runs (website sync, no API). API key valid + persisted in HKCU.
- **Ramp = full_autonomy** (confirmed_by_human=True, reauth cleared). Gate was reached legitimately
  (phase16 done + phase17 de-scoped). Branch `phase0-stackbridge` (many commits ahead of `main`).
- **Fixed + PROVEN this session:**
  - API key was invalid (401) → replaced; proposer now returns real non-TCL SI candidates
    (`si_adaptive_damping`, `si_er_hybrid`).
  - Method synthesis was 100% broken → **6 fixes, proven end-to-end** (an EWC idea synthesized →
    AST → Docker sandbox `VALIDATION_PASSED` → minibench → registered). Commits `bea4814`, `236599f`;
    image `tar-sandbox:torch-cpu` (built from `scripts/sandbox_torch_cpu.Dockerfile`); Docker Desktop
    engine 29.6.1 installed (`docker.exe` at `C:\Users\cgard\AppData\Local\Programs\DockerDesktop\resources\bin`).
- **THE remaining blocker:** the live daemon's **director never invokes the proposer/synthesis path** —
  logs `submitted=0 experiments`, and the whole boot log has zero `[Director]`/`[method_synthesizer]`/
  `gap_probe` lines. Direct calls of the proposer + synthesis WORK; the director isn't calling them for
  the SI anomaly frontier (`fp-gap-tar-anomaly-si-stability-without-collapse`, top gap 0.907, CL domain,
  truth_status `weak`).

## DEFINITION OF DONE for the next session
The daemon, unattended, completes **one** solution-loop cycle on the SI anomaly:
director → non-TCL proposal → Docker-sandbox synthesis → n=5 screen (kill-only) → (survivor →
n≥20 confirm escalation) → governed verdict — visible on the dashboard, no manual patching mid-flow,
honesty rails (source tags, collapse-veto, kill-ledger, truth-lock default-deny) firing.

---

## Phase 1 — Bring TAR up, metered (~15 min)
1. From a shell with docker+key on PATH:
   `$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY","User")`;
   add the DockerDesktop bin to `$env:Path`.
2. (Optional but recommended for debugging) cap spend: set `TAR_API_BUDGET_USD` low (e.g. `2.00`) in the
   launch env so runaway calls self-limit (`tar_lab/llm_bridge.py:62`).
3. Remove `tar_state/daemon_paused.flag`; `Enable-ScheduledTask "TAR Platform Supervisor"` OR launch
   `python tar_living_research.py --platform --poll-interval-s 15` (Start-Process, detached).
4. Verify: 5 services up, daemon heartbeat fresh, `is_full_autonomy=True`, `docker version` works from
   the daemon's PATH, key valid (`call_claude('ping')`).

## Phase 2 — Fix the director experiment-generation gap  ← THE critical path
The proposer/synthesis work in isolation; the director isn't calling them. Instrument, then fix.
1. **Trace** `tar_research_director.py` experiment-generation (the frontier loop with the a2 anomaly
   branch, ~L3188 `_falsified_dead`/`_is_anomaly_frontier`, ~L3256 `_anomaly_first_probe`, and
   `_frontier_experiment_catalog` ~L1985/2491). Add temporary `print`/`_log` at each decision point,
   restart, read one cycle in the daemon service log (`tar_state/logs/watchdog-living-research-*.log`).
2. **Check, in order, which is false for the SI frontier:**
   - Is the experiment-directive fn even reached each cycle (vs only `_register_frontiers_from_gaps`)?
   - Does the frontier dict carry `domain=="continual_learning"` + non-empty `candidate_datasets` +
     `external_baselines`? (`frontier_problem_from_gap` in `tar_frontier.py` should source these from
     the well-known CL catalog; if empty, the gap-probe guard at director ~L2500-2502 skips it AND
     `_frontier_autonomy_allowed(domain)` may fail.)
   - Is `is_anomaly_frontier(frontier_id)` True (prefix `fp-gap-tar-anomaly`) AND
     `_frontier_autonomy_allowed(domain)` True? (both verified True in isolation — confirm live.)
   - Is `n_pending_for_frontier == 0`? Is an exception being swallowed (wrap + log)?
3. **Fix** the identified condition. Likely candidates: the frontier's domain/datasets not populated for
   a `tar_anomaly::` gap; or the experiment-gen fn gated behind something; or `completed_for_frontier`/
   `n_pending` logic. Keep the fix tightly scoped + add a test.
4. **Acceptance:** daemon log shows `[Director]` proposing for the SI frontier + `[method_synthesizer]`
   running the sandbox + a spec queued (`experiment_queue.json` count > 0) OR a synthesized method in
   `tar_state/synthesized_methods/`.

## Phase 3 — Verify the full loop end-to-end
- Watch one candidate flow: proposal → 24h veto window (may need to approve/shorten for testing) →
  n=5 screen → `_build_result` (joint criteria + collapse-veto + kill-ledger) → survivor →
  n≥20 confirm escalation, OR a NULL/COLLAPSED kill recorded to `tar_state/solution_loop/kill_ledger.jsonl`.
- Confirm honesty rails: candidate source-tagged `tar_novel` (excluded from bar), truth-lock keeps
  `publication_allowed=0`, veto window honored.
- **Acceptance:** at least one candidate fully screened with a governed verdict; dashboard
  (`http://127.0.0.1:7860`) shows real autonomous activity.

## Phase 4 — Let it run: SI-stability campaign + STRATA
- Let the loop work the SI anomaly, then the STRATA frontiers (#2 replay-free, #3 sparse-routing).
- Note: some proposer candidates (adaptive, forgetting-gated) legitimately fail the minibench
  ("~0 reg in a benign minibench") — that's correct rejection; the loop will try others.
- Optional high-value: hook the STRATA attribution-probe port (`tar_lab/forgetting_attribution.py`, the
  `scripts/run_forgetting_attribution.py` demo) onto real `generic_cl_runner` runs to diagnose *why*
  TAR's models forget, and reproduce STRATA's Split-MNIST adjudication as external adjudicator.

## Phase 5 — Debt + consolidation (as time permits)
- Quiet the repeating **blocked paper-author** loop (`autonomous_high_penalty_conservative` warns every
  cycle) — it's the human-review gate working, but it's noisy; make it back off.
- **Parallel-session WIP disposition** (still uncommitted: `main_paper/main.tex` rewrite, `tar_frontier.py`,
  `tar_dashboard.py`, `tar_lab/honest_evidence.py`, `tar_watchdog.py`, `tar_operator_agent.py`) → commit
  with the operator, then **fast-forward `phase0-stackbridge` → `main`** (was 23/0 clean; re-check).
- Finish the 4 unreviewed adversarial bug-hunt areas (method_guard, catalog, killledger_director,
  seed_integration) from `wf_172c51fa-9a7`.
- Coherence-plan carryovers (`docs/TAR_Coherence_Master_Plan.md`): WS1.4 cost-watchdog tests,
  WS4.1 single review-queue view.

## API-cost discipline (lead cares about this)
- The director/proposer/synthesis calls cost credit; the embedder/recall is local (free).
- Prefer **bursts**: bring up → do Phase 2/3 → bring down. Or set `TAR_API_BUDGET_USD` low while debugging.
- **Bring down** (zero API) any time: `Disable-ScheduledTask "TAR Platform Supervisor"`; stop watchdog +
  daemon + queue_maintainer + operator_agent (Stop-Process); set `daemon_paused.flag`. (Backup task is
  file-only, safe to leave.)

## Pointers
- Memory: `project_tar_solution_loop_built` (full fix history + the director gap), `project_tar_strata_integration`,
  `feedback_python313_training_retired`, `project_tar_coherence_master_plan`.
- Commits this session: `1d3c7e2`, `bc13bcd`, `fd277d8`, `b00d213`, `c8011e6`, `a4ffc52`, `547792b`,
  `37dba33`, `a636d16`, `2a2f093`, `bea4814`, `236599f` (all on `phase0-stackbridge`, pushed).
