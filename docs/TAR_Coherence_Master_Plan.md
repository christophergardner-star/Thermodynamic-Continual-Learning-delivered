# TAR Coherence Master Plan

**Date:** 2026-07-04 · **Status:** APPROVED-FOR-EXECUTION plan (execution by a future session; human gates marked ⛔)
**Repo:** branch `phase0-stackbridge` @ `1d3c7e2` (23 ahead of `main`, 0 behind — clean fast-forward)
**Grounding:** the 12-dimension state audit, the remediation + solution-loop builds, two adversarial
bug-hunts, and five deep-dive scouts (harness-calibration + knowledge-feed reports verified; paper/ops/
safety areas verified by direct checks). Every anchor cited was checked on 2026-07-03/04.

---

## 0. Governing principles (read first, apply to every task)

1. **Prove it in anger.** The system's repeated failure pattern: machinery outruns output, and integrity
   holes surface late. Therefore: *no new capability is built until the capability it depends on has been
   exercised end-to-end.* Every workstream below terminates in something RUN, not something written.
2. **The product is honesty, so the pitch cannot be dishonest.** Applies to plan claims, paper text,
   the website, and this document. If a task's acceptance can't be verified, say UNVERIFIED.
3. **Standing safety rules (unchanged):** never touch a running experiment; the ramp/veto/manifest
   gates are never bypassed; dry-run defaults; no fabricated numbers ever; RunPod spends nothing
   without ⛔ operator go; all commits carry the Co-Authored-By trailer; files >100 lines not authored
   in-session are never staged without ⛔ operator confirmation.
4. **Definition of COHERENT (the finish line):** TAR has run ONE complete campaign end-to-end —
   anomaly → proposed candidates → veto window → ramp-released execution → governed verdicts →
   kill-ledger/write-back → a reviewer-grade artifact — with zero manual patching mid-flow, on a
   substrate that survives a reboot, from a repo whose `main` is the truth, with a paper/public layer
   containing no claim the evidence layer denies.

---

## 1. Current state (verified facts the plan builds on)

- **Live:** HPC replication training (pid 12560, seed ≥28, SPRT=continue); phase-2 sequencer armed to
  auto-launch the mechanistic ablation; daemon on OLD code (solution-loop code is on disk, inert);
  execution held at the ramp; dashboard honest; website reframed + deployed.
- **Solution loop BUILT** (commits `33ad4a6..1d3c7e2`): integrity pre-work (tcl_canonical key,
  tar_novel class, enforced prereg criteria + collapse-veto, fail-loud method guard), method catalog
  (20 methods, `_verified:false`), SI-anomaly seeder (dry-run), widened proposer, kill-ledger,
  status reporter, activation runbook. 31 dedicated tests green; 2 confirmed bug-hunt findings fixed
  (`1d3c7e2`).
- **Bug-hunt status:** 2 of 6 review areas fully verified (identity_bar, criteria_eval);
  4 areas (method_guard, catalog, killledger_director, seed_integration) **cut off by session limits
  twice** — their review is INCOMPLETE. This is open verification debt (→ WS1.1).
- **Narrative contradiction CONFIRMED live:** `frontier_problems.json` holds
  `fp-catastrophic-forgetting: status=active, truth_status=falsified` simultaneously.
- **Dirty tree (29 items) inventoried:** (a) daemon churn — 6 `paper_plan.json` + `manifests/auto/*` +
  `manifests/human_review/*` (safe to ignore/commit-as-churn); (b) **parallel-session real work,
  DO NOT STAGE without ⛔:** `paper/main_paper/main.tex` (904-line rewrite in progress) + `main.pdf`,
  new `paper/tcl_phd_rehabilitation/` sections, `paper/main_paper/CLAIM_TRACEABILITY.md`,
  `tar_lab/honest_evidence.py` (+76), `tar_frontier.py` (+38), `tar_dashboard.py` (−63),
  `tar_watchdog.py` (operator-agent supervision), `tar_operator_agent.py` (untracked, 505 lines).
- **Knowledge loop root causes found:** arXiv 429s persist because `_run_fast_cycle`
  (tar_evidence_ingest.py:1276-1321) **bypasses the cooldown gate** (daily cycle honors it at
  :1182-1194) → 119 consecutive failures; Semantic Scholar is keyless (0.9 rps) and burst-queried in
  parallel (:1205-1234). By-ID ingest capability EXISTS (`SemanticScholarClient.batch_fetch`,
  semantic_scholar.py:303-331 — 500 ids/request) but is wired to nothing.
- **Harness A/B are NOT protocol-equivalent** (full diff in scout report): forgetting formula differs
  by a deterministic ×5/4; task-IL impossible in B; regime observer impossible in B; EWC/SI estimator +
  scope + factor-2 penalty differences; `tcl_canonical` λ defaults 100× apart on different keys.
  Equivalence is only establishable on {sgd, ewc, si, der++, lwf, tcl_canonical} × class-incremental ×
  resnet18.
- **Ops:** `main` exists locally + on origin (origin/HEAD→main); fast-forward merge available.
  Non-git state worth backing up: `tar_state/literature` 560 MB + `tar_state/memory` 21 MB.
  Daemon boot ~3 min vs watchdog `stale_after_s=180` → exactly one wasted respawn per boot.
  "TAR Platform Supervisor" task = current-user → does NOT run before login (residual reboot gap).

---

## 2. Workstreams

### WS1 — VERIFY IN ANGER (pitfalls: under-exercised honesty machinery; unexercised safety stack)

**WS1.1 Finish the adversarial bug-hunt (verification debt).** Resume run `wf_172c51fa-9a7`
(cached agents replay) when subagent capacity is available; the 4 unfinished areas are method_guard,
catalog, killledger_director, seed_integration. Fix every CONFIRMED finding with a test.
*Acceptance:* all 6 areas verified; zero unfixed CONFIRMED findings. *Effort: 1-2 h + fixes.*

**WS1.2 Harness A/B calibration run** (spec from the scout report, `cscout_harness_calib.md` —
reproduce into `docs/` as part of this task):
- Pairs: {sgd, ewc, si} (+optional tcl_canonical with λ pinned 0.01 BOTH sides) × class_incremental ×
  resnet18 × n=10 seeds `[42,0,1,2,3,123,456,789,1337,7]`; B-side forced `lr=0.01, batch=64,
  ewc_lambda=1000, si_c=0.01, si_xi=0.001` + **prebuilt loaders** with A's fixed class order and
  transforms; A-side `setting="class_incremental"`.
- Apply `F_A_corrected = F_A × 5/4` before comparison (different forgetting denominators).
- **Acceptance criterion (preregister it):** per-method TOST with ε=0.02 absolute forgetting (both
  one-sided p<0.05) AND identical method ranking (Kendall τ=1.0); fallback "offset-equivalent" iff
  sd(diff)<0.01 with recorded per-method offset b_m; else cross-harness numeric comparison is
  PROHIBITED and every campaign claim carries the harness tag.
- Budget ≈ 12 h (E=5 grid) + optional 19 h (E=40 spot-check) on the 1650. **Runs only after the
  phase-2 queue drains** (never concurrent with HPC).
*Acceptance:* a written calibration verdict (equivalent / offset-equivalent / not-equivalent) stored as
a canonical comparison artifact. *This unblocks the campaign's cross-harness honesty.*

**WS1.3 Safety-stack rehearsal (the first L4 flow, cheap by construction).**
1. Kill-switch drills first (no ramp change): (a) create `daemon_paused.flag` mid-cycle → verify
   planning-only; (b) veto a pending proposal → verify it never runs; (c) corrupt a COPY of
   `autonomy_ramp.json` in a test workspace → verify `is_full_autonomy=False` (tests exist; run them
   against the live code path once more).
2. ⛔ Operator confirms the ramp (`python tar_autonomy_ramp.py confirm` — FRESH confirm required by
   the reauth guard).
3. The rehearsal experiment: a **tiny 2-task × 2-epoch × tiny-backbone Split-CIFAR-10 spec**
   (minutes on the 1650; there is no cpu_only runner — scheduler supports the flag but no runner
   consumes it, so smallest-GPU is the floor) flowing the ENTIRE pipeline: proposal → veto window →
   auto-approve → ramp-released launch → truth-lock verdict → write-back → recall visible.
4. Immediately after: ⛔ operator decides whether the ramp STAYS released for the campaign or is
   re-held (`disable`) until WS2.3.
*Acceptance:* one experiment launched autonomously end-to-end with every gate observed firing, and a
written log of which gates fired in what order. *Effort: half a day incl. drills.*

**WS1.4 Cost-watchdog unit validation (no money).** Extract `_cost_watchdog`'s tick logic into a
testable function (injected clock); unit-test: warn fires at 50%, hard-kill at 100% of
`max_experiment_cost_usd`, price-unknown fallback engages `assumed_price_per_hour_when_unknown`.
*Acceptance:* tests green; NO live pod. *Effort: ~2 h.*

### WS2 — OUTPUT FIRST (pitfall: infrastructure outruns output)

**WS2.1 Drain the phase-2 gate (already in motion — protect it).** HPC replication finishes (SPRT/seed
target), sequencer auto-launches the mechanistic ablation. Do not restart the daemon until BOTH are
terminal. Then evaluate_ramp sees hp_selection+hpc_replication+mechanistic_ablation terminal; the
remaining not_found runners (phase16/17 reruns, hpc_lambda) are ⛔ operator decisions: run on RunPod
(cost), run locally (days), or formally de-scope them from `phase2_runner_keys` (an honest,
documented reduction — preferable to a permanently-stuck gate).
*Acceptance:* ramp reaches `awaiting_confirm` on its own gate report.

**WS2.2 Narrative-layer cleanup (the paper/public layer must not contradict the evidence layer).**
- Fix the live registry contradiction: `fp-catastrophic-forgetting` must carry a closed/falsified
  status consistent with its closure file (the public site already drops it; the registry itself
  still says `active`). One-line data fix + a regression test that registry status and closure
  verdicts can never disagree.
- ⛔ Coordinate with the parallel session's in-progress work before touching `paper/`:
  `main_paper/main.tex` (904-line rewrite), `tcl_phd_rehabilitation/`, CLAIM_TRACEABILITY.md, and the
  modified `honest_evidence.py`/`tar_frontier.py`/`tar_dashboard.py` diffs are ANOTHER session's
  work-in-progress. Disposition each with the operator: commit theirs first, then rebase this plan's
  tasks on it. The proxy-as-TCL and orphaned-phase18-negative fixes belong to that thread — verify
  they land; do not duplicate.
- Write the phase18 negative result into the rehabilitation paper (it is currently orphaned).
*Acceptance:* zero contradictions between paper text / registry / closure files / honest inventory,
verified by a grep-able consistency check script committed to `scripts/`.

**WS2.3 THE CAMPAIGN (the deliverable that makes everything else matter).** After WS1.2 + WS1.3 +
Phase-B activation: run the SI-stability-without-collapse campaign per the activation runbook —
seed anomaly (`--apply`), widened proposer generates candidates (catalog + kill-ledger context),
⛔ veto window, n=5 local screen (kill-only: collapse or clear underperformance — a screen pass is
NOT evidence), survivors → n≥20 with the preregistered variance-ratio test (RunPod ⛔ if scale needed).
Every claim through governance; ALL outcomes reported (validated mechanism OR mapped eliminated space).
*Acceptance (Definition of COHERENT):* the full chain runs without manual patching; the artifact is
reviewer-grade (claims → evidence JSONs traceable end-to-end).

### WS3 — FEED THE KNOWLEDGE LOOP (pitfall: data starvation)

**WS3.1 The one script that closes three gaps (~1-2 h).** `scripts/ingest_catalog_papers.py`:
read ids from `method_catalog.json` → `SemanticScholarClient.batch_fetch` (ONE request, prefixed
`ARXIV:`/`DOI:`) → `upsert_paper` into the graph → emit a **title-diff report** (claimed title vs
resolved title per method_key). That report IS the citation-verification artifact.
**WS3.2 ⛔ Operator verifies the title-diff** (~20 min scripted vs 45-60 min manual; checklist table
already generated in `cscout_knowledge_feed.md` — copy into docs/). Flip `citation_status` per entry
and `_verified:true` only when all pass. Any mismatch → fix or delete the entry (never keep).
**WS3.3 Fix the ingest defects:** (a) route `_run_fast_cycle`'s arXiv/OpenAlex calls through the same
cooldown gate the daily cycle uses (tar_evidence_ingest.py:1276-1321 vs :1182-1194) — ~1 h, kills the
429 loop; (b) ⛔ operator obtains + sets `SS_API_KEY` (5 min) → 10 rps.
**WS3.4 ⛔ External SoTA bar — human transcription only.** There is NO in-DB shortcut (verified: 0
abstracts state a usable Split-CIFAR-10 number). Operator transcribes 5-10 cited rows from the actual
papers (DER++ 2004.07211, GEM 1706.08840, iCaRL 1611.07725, or a benchmark survey) into
`curated_external_sota.json`. ~1-2 h. NEVER automated from abstracts.
*Acceptance:* NoveltyGate returns a real external verdict; gap detector sees a method universe larger
than 6 internal rows; catalog `_verified:true`.

### WS4 — HUMAN-BOTTLENECK HARDENING (pitfall: one fatigable human)

**WS4.1 Single review queue.** One dashboard view (or one CLI report) listing EVERY pending ⛔ gate:
proposals in veto, catalog verification state, ramp stage, RunPod toggle, unverified-citation count,
paper approvals. The operator should never have to remember where the gates live.
**WS4.2 Checklists over rubber stamps.** Each gate gets a 3-5 line checklist in the UI/report (what
to actually verify), starting with the citation title-diff (WS3.1 makes it mechanical).
**WS4.3 Bus-factor mitigations (honest scope):** a RUNBOOK.md covering start/stop/restart, activation,
the gates, backup/restore; plus WS5.1 backups. A second maintainer is out of scope for software to fix.
*Acceptance:* operator can enumerate all pending decisions in one place in <1 min.

### WS5 — SUBSTRATE (pitfall: hobbyist foundation for an infrastructure thesis)

**WS5.1 Backups (first — cheap, highest value).** Nightly scheduled copy of
`tar_state/literature` (560 MB), `tar_state/memory` (21 MB), `comparisons/`, `experiments/`,
`anchors/`, and the prereg/inventory JSONs to a second physical drive (C: task already exists for
supervision; add a backup task). *Effort: ~1 h.*
**WS5.2 Branch → main.** `phase0-stackbridge` is a clean fast-forward (23 ahead, 0 behind).
⛔ After the parallel-session work is committed (WS2.2 disposition), fast-forward `main`, push, and
make `main` the working branch. *Effort: minutes, once the tree is clean.*
**WS5.3 Boot vs watchdog.** Either raise the daemon ServiceConfig `stale_after_s` 180→360 (one line in
tar_watchdog.py — note the file currently carries ANOTHER session's uncommitted diff; coordinate) or
emit a heartbeat during boot (touch the state file before torch import and after embedder load).
*Acceptance:* one daemon restart with ZERO watchdog respawns during boot.
**WS5.4 Reboot drill.** After WS5.1-5.3: reboot the box once, log in, verify the supervisor task
resurrects the platform within its 10-min interval. Document the honest residual: the current-user
task does NOT run before login (fix = auto-login or a SYSTEM-level task, ⛔ operator choice).
**WS5.5 Deploy path.** Already hardened to the host's ceiling (the host refuses FTPS). Standing items:
⛔ rotate the FTP password (it transited plaintext + sits in a chat log); do not bulk-upload while the
local research.json is stale (restart first); longer-term ⛔ consider an SFTP-capable host.

### WS6 — LOOP CORRECTIONS (pitfall: traps in the new machinery)

**WS6.1 Screening-power policy (write into the campaign prereg):** n=5 screen is KILL-ONLY
(collapse or catastrophic underperformance); passing the screen carries zero evidential weight; the
false-negative risk (killing a real winner whose d<1.26) is ACCEPTED and stated.
**WS6.2 Kill-ledger hygiene:** add `protocol_version` + `harness` fields to kill records (one-line
change in `record_kill` callers); a killed fingerprint only prunes proposals under the SAME protocol
version; ⛔ operator may amnesty a region after a protocol change. Near-duplicate pruning (HP-distance)
is explicitly OUT of scope v1 — revisit only if the campaign shows churn.
**WS6.3 Synthesis stays OFF for campaign #1.** Tier-1 composed classes only (hand-written, ~40-80
lines each; budget ~5-8 candidate classes for the campaign). Enabling `method_synthesis.enabled` is a
⛔ decision AFTER the campaign proves the Tier-1 path, never before. The baseline-name guard
(`forbidden_synthesis_name`, commit 1d3c7e2) is already in place for when it comes.
**WS6.4 Harness tags everywhere:** every result/claim records which harness produced it (spec.runner_key
already distinguishes; surface it in notes + paper tables).

---

## 3. Sequencing (dependency-ordered; ✅ = can start now)

| Phase | Trigger | Tasks |
|---|---|---|
| **A ✅ (now — GPU busy is fine)** | none | WS1.1 bug-hunt completion; WS3.1 ingest script + WS3.3a fast-cycle fix; WS5.1 backups; WS1.4 cost-watchdog tests; WS4.1-4.2 review queue; WS2.2 registry-contradiction fix + ⛔ parallel-work disposition |
| **B (HPC + ablation terminal)** | phase-2 queue drained | daemon restart onto new code (git on PATH); WS5.3 boot fix verified; `seed_si_anomaly.py --apply`; `solution_loop_status.py` check; WS1.2 calibration run; WS3.2/3.4 ⛔ verifications |
| **C (rehearsal)** | B complete | WS1.3 drills + ⛔ ramp confirm + tiny end-to-end rehearsal; WS5.4 reboot drill |
| **D (the campaign)** | C green | WS2.3 SI-stability campaign under WS6 policies |
| **E (the artifact + consolidation)** | D terminal | reviewer-grade write-up (all outcomes); WS5.2 ⛔ merge to main; retro: which machinery failed in anger → fix list |

**Estimated effort:** Phase A ≈ 2-3 focused days; B ≈ 1 day + 12-31 GPU-hours; C ≈ half day;
D = compute-bound (days of unattended GPU) + operator gate touches; E ≈ 1-2 days.

## 4. STOP points (⛔ operator-only, consolidated)

ramp confirm (twice: rehearsal + campaign posture) · catalog `_verified` flip · external SoTA numbers ·
RunPod enable / any spend · phase16/17/lambda de-scope decision · parallel-session work disposition ·
merge to main · synthesis enablement · FTP password rotation / host change · publishing anything.

## 5. What this plan deliberately does NOT do

No new subsystems. No multi-domain expansion. No LoRA/weights self-improvement. No RunPod spend
without a gate. No paper claims beyond what the campaign yields. The point is to make the EXISTING
system coherent by running it, not to grow it.
