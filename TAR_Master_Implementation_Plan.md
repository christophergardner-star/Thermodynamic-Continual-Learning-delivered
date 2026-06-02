# TAR Master Implementation Plan
## The Complete Unified Programme — Research, Engineering, Science & Publication
### PhD-Level Team Implementation | 10 Phases | 104 Tasks

*Sources: TAR_PhD_Rehabilitation_Plan.md + TAR_Enhancement_Report.md*
*Compiled: 2026-06-02 | Status: ACTIVE MASTER PLAN*

---

## HOW TO USE THIS DOCUMENT

This is the single authoritative plan for bringing TAR to full PhD-level competency across continual learning research, engineering infrastructure, autonomous research architecture, and academic publication. It supersedes the individual documents that preceded it.

**Every task has a unique ID** (e.g., E0.1, 0.4, 1.7, 8.2) and is never repeated elsewhere in the plan. Phases are ordered by urgency; within a phase tasks are ordered by dependency. Nothing is optional — items marked LOW priority are lower urgency, not skippable.

**Phases run on three tracks in parallel:**
- **Track A (Science):** E0 → 0 → 1 → 2 → 3 → 5
- **Track B (Code & Engineering):** E0 → 0 → 4 → 7
- **Track C (Automation & Literature):** 0 → 9 → 8 → 6

Phase 5 (paper) cannot start until Track A Phases 1–2 are complete. All other phases overlap.

---

## PART I — SYSTEM BASELINE

### A. Research ground truth — what is scientifically demonstrated today

| Result | Dataset | N seeds | Trust tier | Publication status |
|---|---|---|---|---|
| TCL < EWC on forgetting (δ=−0.075, p=0.019, d=1.70) | Split-CIFAR-10 | 5 | trusted_manual_controlled | **Publication-allowed but fails Bonferroni** |
| Penalty component essential; governor-alone WORSE than SGD | Split-CIFAR-10 | 5 | trusted_rerun | **Publication-allowed** |
| HPC forgetting 0.058 vs TCL baseline 0.126 (δ=−0.068, p=0.018, d=1.73) | Split-CIFAR-10 | 5 | trusted_rerun | **Underpowered — needs replication** |
| TCL < SGD on TinyImageNet (d=75.08) | TinyImageNet | 3 | trusted_rerun | **Gate B fails: n=3 < 5** |
| TCL < EWC on CIFAR-100 (directional) | CIFAR-100 | 3 | trusted_rerun | **Gate B fails: n=3 < 5** |

### B. Engineering layer grades

| Layer | Grade | Critical finding |
|---|---|---|
| ML Algorithm Engine | **B** | Regime detector never fires; train_template.py is 1,600-line monolith |
| Orchestration & Daemons | **C+** | Restart loops possible; orphaned processes on crash |
| Infrastructure, API & Deployment | **D+** | Live credentials committed to public GitHub repo |
| State Management & Storage | **B−** | Missing fsync; no schema migrations; unbounded JSONL growth |
| Literature & Paper Pipeline | **B−** | ActiveLearner orphaned; knowledge graph empty; zero prose written |
| **System Overall** | **C+** | Research-grade science; development-grade operations |

### C. The five structural problems

1. **Theory–algorithm decoupling.** The thermodynamic framing is post-hoc narrative. The regime detector stays `rho=0, sigma=0, regime="unknown"` in every production trace. It has never fired. The mechanism that works is gradient-EMA elastic regularization, not thermodynamics.

2. **Systematic underpowering.** 5 seeds at α=0.05 gives ~50% power for medium effects. The CI formula uses z-critical (1.96) regardless of n — for n=3 this makes intervals ~2.2× too narrow. The headline Phase 10 result (p=0.019) does not survive Bonferroni correction.

3. **Unreviewed governance code.** Six files governing all autonomous behaviour have never been committed. The system is running from working-tree code with no git audit trail.

4. **No manuscript, no active literature intelligence.** Every LaTeX file is a template. The ActiveLearner that should populate the knowledge graph has never been wired to the orchestrator. The self-improvement anchor pack has never been initialized.

5. **Development-grade infrastructure.** Live API credentials committed to the public repository. No CI/CD. No pinned dependencies. Flask dev server in production. 11 JSONL logs growing without bound.

### D. What the system falsely believes it has demonstrated

- That the thermodynamic governor is a primary mechanism (Phase 11 ablation disproves this)
- That 25 claim verdicts are meaningful (96% are `insufficient_evidence`, all expired >14 days)
- That TCL is a broadly applicable forgetting solution (`fp-catastrophic-forgetting` is formally falsified: 17 null, 4 adverse, 1 breakthrough)
- That HPC is a confirmed breakthrough (5 seeds from a 5-hypothesis battery at uncorrected α)

---

## PART II — TEAM & GOVERNANCE

### Roles

| Role | Responsibilities |
|---|---|
| **Lead Researcher** | Scientific direction, phase gate decisions, paper authorship, reviewer of all claims |
| **Algorithm Engineer** | ML code (tcl.py, method_registry.py, generic_cl_runner.py), baseline implementations |
| **Systems Engineer** | Infrastructure (Docker, CI/CD, SQLite, RunPod, state management) |
| **Statistician** | Statistical methodology, pre-registration, power analysis, CI corrections |
| **Paper Author** | tar_author.py management, LaTeX drafting, citation management |

*In a solo setting, the Lead Researcher performs all roles. The role labels define context-switching — put on each hat deliberately.*

### Phase gate authority

- **Phases E0, 0:** Lead Researcher must personally confirm completion before proceeding
- **Phase 1 exit:** Statistician signs off that all existing results have been re-analysed with corrected methodology
- **Phase 2 exit:** Lead Researcher reviews all replication results; records honest verdict in evidence inventory
- **Phase 5 entry:** All Phase 2 experimental results must be publication-allowed (trust tier + seed count)
- **Phase 5 exit:** At least one external human has reviewed the draft before submission

### Decision rules

- Any result that does not survive Bonferroni correction is recorded as DIRECTIONAL, not BREAKTHROUGH
- Any experiment with n < 5 seeds is marked exploration_grade and cannot be cited in a paper
- No new experiments are queued until Phase 1 statistical corrections are complete
- arXiv submission only after Phase 2 HPC replication result is recorded

---

## PART III — MASTER PHASE PLAN

---

## PHASE E0 — EMERGENCY SECURITY
**Duration:** Same day — complete before any other work begins
**Track:** All tracks blocked until E0 is done
**Owner:** Lead Researcher
**Objective:** Eliminate the live credential exposure. This is non-negotiable and non-deferrable.

---

### Task E0.1 — Rotate all exposed credentials immediately

**Priority:** CRITICAL — SAME DAY

The file `tar_state/api_secrets.json` contains live credentials committed to the public GitHub repository. `publish_config.json` in the repository root contains FTP credentials. Both were pushed and are publicly readable now.

**Credentials to rotate at each provider's console:**
- Anthropic API key (`sk-ant-api03-...` pattern) → anthropic.com console
- RunPod API key → runpod.io console
- RunPod S3 access key and secret key → RunPod storage settings
- FTP username/password from `publish_config.json` → hosting panel

**Do not proceed to E0.2 until every key above has been invalidated and replaced.**

**Verification:** Attempt to use the old keys via API call; confirm they return 401/403.

---

### Task E0.2 — Remove credentials from git history

Rotating keys stops future harm; old keys remain in all git clones until history is cleaned.

```bash
# Option A: BFG Repo-Cleaner (fastest)
java -jar bfg.jar --delete-files api_secrets.json
java -jar bfg.jar --replace-text passwords.txt   # one old key per line
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
git push --force --tags

# Option B: git-filter-repo (no Java dependency)
pip install git-filter-repo
git filter-repo --path tar_state/api_secrets.json --invert-paths
git filter-repo --path publish_config.json --invert-paths
git push --force --all
```

Notify any collaborators with existing clones to delete and re-clone.

**Verification:** `git log --all --full-history -- tar_state/api_secrets.json` returns nothing.

---

### Task E0.3 — Add .gitignore entries for all secrets

Add to `.gitignore` at repo root:
```
tar_state/api_secrets.json
publish_config.json
.env
*.key
*_secrets.json
*credentials*.json
*_passwords.json
```

Commit this change immediately after E0.2.

**Verification:** `git check-ignore -v tar_state/api_secrets.json` shows the file is ignored.

---

### Task E0.4 — Move credentials to environment variables

Replace all hardcoded credential reads with environment variable lookups:

Files to update:
- `tar_lab/llm_bridge.py` — reads `ANTHROPIC_API_KEY` or `TAR_LLM_API_KEY`
- `tar_api.py` — reads `TAR_API_KEY`
- `sync_research.py` — reads `TAR_FTP_HOST`, `TAR_FTP_USER`, `TAR_FTP_PASSWORD`

Create `.env.example` (committed, placeholder values only):
```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
RUNPOD_API_KEY=YOUR-RUNPOD-KEY-HERE
TAR_FTP_HOST=your-ftp-host.com
TAR_FTP_USER=your-username
TAR_FTP_PASSWORD=your-password
```

**Verification:** Delete any local `.env` file; confirm `llm_bridge.py` fails gracefully (returns empty string) when key is absent.

---

### Task E0.5 — Add pre-commit secrets detection hook

```bash
pip install detect-secrets pre-commit
detect-secrets scan > .secrets.baseline
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-added-large-files
      - id: check-merge-conflict
```

```bash
pre-commit install
pre-commit run --all-files   # must pass clean
```

**Verification:** Attempt to commit a file containing `sk-ant-api03-test`; confirm pre-commit blocks it.

---

**Phase E0 exit gate — ALL must be true before any other work:**
- [ ] Every credential in `api_secrets.json` and `publish_config.json` rotated at provider console
- [ ] Old credentials verified non-functional (API call returns 401)
- [ ] Git history cleaned; `git log` shows no trace of old secret files
- [ ] `.gitignore` entries committed and verified
- [ ] All credential reads use `os.environ`; `.env.example` committed
- [ ] `detect-secrets` pre-commit hook installed and passing on clean run

---

## PHASE 0 — TRIAGE & GROUND-TRUTH LOCK
**Duration:** 1–2 weeks
**Track:** All tracks
**Owner:** Lead Researcher
**Objective:** Stabilise the running system, audit what actually exists, and create the honest foundation from which all subsequent work flows. No new experiments until this phase is complete.

---

### Task 0.1 — Audit the running TinyImageNet experiment to completion

The experiment `harder_domain_split_tinyimagenet` (ResNet-18, Split-TinyImageNet, 5 seeds, 40 epochs) was active at audit time with 0/5 seeds complete and GPU at 87% utilisation.

- Monitor until completion. Do not kill it.
- Once complete, run the honest result recording: compare TCL vs EWC with Bonferroni-corrected threshold (α=0.0125 for 4 comparisons). If TCL does not beat EWC at p<0.0125, record as DIRECTIONAL.
- Add the result to `tar_state/honest_evidence_inventory.json` (created in Task 0.5).
- After recording: halt all autonomous experiment submission. Set `execution_enabled.flag` to absent. No new queue entries until Phase 1 statistical corrections are complete.

**Verification:** `experiment_queue.json` shows 0 running experiments and `execution_enabled.flag` does not exist.

---

### Task 0.2 — Commit and review the six governance files

The files `tar_research_director.py`, `tar_experiment_orchestrator.py`, `tar_living_research.py`, `tar_scheduler.py`, `tar_frontier.py`, `tar_autonomous_research.py` exist only in the working tree.

**Review each file against HEAD:**

| File | Confirmed status | Action |
|---|---|---|
| `tar_autonomous_research.py` | Stub: prints "retired", exits 2 | Commit as-is |
| `tar_frontier.py` | Clean frontier registry, safe | Commit as-is |
| `tar_scheduler.py` | Clean hardware-aware scheduler | Commit as-is |
| `tar_living_research.py` | Portfolio coordinator, approved-ids check present | Commit with review comment |
| `tar_research_director.py` | Decision engine, LLM calls exception-safe | Commit with review comment |
| `tar_experiment_orchestrator.py` | Contains `set_autonomous(True)` auto-manifest path | Commit with explicit comment documenting the bypass |

For `tar_experiment_orchestrator.py`, add a code comment at the `set_autonomous` method explicitly stating: "Autonomous mode generates and git-commits manifests without human review. The commit is permanent and auditable. This path requires execution_enabled.flag to be present."

**Verification:** `git log --oneline -- tar_experiment_orchestrator.py` shows a commit. `git status` shows no untracked governance files.

---

### Task 0.3 — Fix active_session.json stale DORMANT state

The file reads `DORMANT_NO_MANIFEST` despite the daemon having restarted today. The batch file `START_TAR.bat` resets service state files but not this one.

- Locate the write path for `active_session.json` in the daemon startup sequence.
- Add a write to `active_session.json` during daemon startup that sets: `{"manifest_path": "AUTONOMOUS_MODE", "note": "Daemon running in autonomous mode", "started_at": <iso_timestamp>}`.
- Add the same write to `STOP_TAR.bat` shutdown: set `{"manifest_path": "DORMANT_NO_MANIFEST", "note": "System stopped", "stopped_at": <iso_timestamp>}`.

**Verification:** Stop and start the daemon; confirm `active_session.json` reflects the correct state at each step.

---

### Task 0.4 — Fix the queue maintainer NoneType error

The error `'<=' not supported between instances of 'NoneType' and 'int'` has fired every 30 seconds since 2026-05-28. It is a null `priority` field in the queue sorting logic.

- Locate the queue priority sort (in `tar_living_research.py` or `tar_experiment_orchestrator.py`).
- Apply null-guard: wherever priority is compared, replace raw access with `(experiment.priority or 0)`.
- Restart the queue maintainer.

**Verification:** `living_research.log` shows no `queue_maintainer_error` entries for 10 minutes after restart.

---

### Task 0.5 — Build the honest evidence inventory

Create `tar_state/honest_evidence_inventory.json`. This is the single source of truth for all paper planning. It must be built by reading raw comparison JSON files, not by trusting the system's own verdicts.

**Schema per entry:**
```json
{
  "experiment_id": "phase10_controlled_rerun",
  "dataset": "split_cifar10",
  "method_comparison": "tcl_vs_ewc",
  "n_seeds": 5,
  "trust_tier": "trusted_manual_controlled",
  "p_value_uncorrected": 0.019,
  "p_value_bonferroni": 0.019,
  "bonferroni_k": 4,
  "bonferroni_threshold": 0.0125,
  "bonferroni_significant": false,
  "cohens_d": 1.70,
  "achieved_power_pct": 90,
  "seeds_needed_80pct_power": 5,
  "publication_allowed_gate_a": true,
  "publication_allowed_gate_b": true,
  "honest_verdict": "DIRECTIONAL — does not survive Bonferroni at k=4"
}
```

Build an entry for every experiment in `tar_state/comparisons/` and `tar_state/autonomous_research/`.

**Verification:** File exists; every phase result has a corresponding entry; no entry reads "BREAKTHROUGH" for results that fail Bonferroni.

---

### Task 0.6 — Purge financial econometrics contamination from gap scans

Gap scan reports repeatedly surface "mean-variance-skewness-kurtosis portfolio optimization" as a frontier gap. This is quantitative finance.

- Trace the source in `tar_lab/research_ingest.py`: identify which arXiv RSS feed or domain keyword is routing financial papers into the CL domain.
- Add an explicit domain allowlist to the gap scanner: only accept gaps whose `primary_domain` maps to one of the active frontier problem domains (`continual_learning`, `thermodynamics_ml`, `computer_vision`).
- Mark all existing contaminated gap entries: `{status: "rejected", reason: "domain_mismatch_financial", rejected_at: <timestamp>}`.
- Rebuild `frontier_gaps.jsonl` excluding contaminated entries.

**Verification:** Run `tar_cli.py list_frontier_gaps`; no entries mention portfolio, variance, financial, or econometric.

---

### Task 0.7 — Fix the watchdog restart loop

The watchdog has a 30-second cooldown but no maximum restart count. A daemon that fails immediately on startup respawns every 30 seconds indefinitely.

**Implementation in `tar_watchdog.py`:**
```python
# Per-service restart tracking
_restart_history: dict[str, list[float]] = {}  # service_id -> list of restart timestamps

def _restart_allowed(self, service_id: str, cooldown_s: float) -> bool:
    now = time.time()
    history = _restart_history.setdefault(service_id, [])
    # Evict entries older than 1 hour
    history[:] = [t for t in history if now - t < 3600]
    if len(history) >= 6:
        # Circuit open: too many restarts in 1 hour
        self._set_circuit_open(service_id)
        return False
    # Exponential backoff: 30s, 60s, 120s, 240s, capped at 300s
    cooldown = min(300, cooldown_s * (2 ** len(history)))
    last = history[-1] if history else 0
    return (now - last) >= cooldown

def _set_circuit_open(self, service_id: str) -> None:
    # Write CIRCUIT_OPEN to health field; raise CRITICAL alert
    ...
```

Circuit resets only via `tar_cli.py --reset-service-circuit <name>`.

**Verification:** Kill a daemon 7 times in rapid succession; confirm watchdog raises CRITICAL alert and stops respawning after the 6th restart within 1 hour.

---

### Task 0.8 — Fix process lifecycle to prevent orphaned experiments

Daemons are launched with `DETACHED_PROCESS` flags. On daemon crash, running experiment subprocesses continue unmonitored. On restart a duplicate can launch.

**Changes to `tar_experiment_orchestrator.py` and `tar_watchdog.py`:**
- Track actual experiment subprocess PIDs in `process_registry.json` separately from the daemon PID.
- In `reconcile_runtime_state()`: before marking an experiment STALLED, attempt `os.kill(subprocess_pid, signal.SIGTERM)` with a 5-second grace period, then `signal.SIGKILL`.
- In watchdog shutdown (`STOP_TAR.bat`): send `SIGTERM` to all PIDs in `process_registry.json` before exiting.
- On daemon startup: for each PID in `process_registry.json` that still exists, verify it matches the expected experiment before accepting it as valid (check start time within ±5 minutes).

**Verification:** Kill the daemon mid-experiment; confirm the experiment subprocess also terminates within 10 seconds. Restart daemon; confirm no duplicate experiment launches.

---

### Task 0.9 — Add lockfiles to all shared multi-writer JSON state files

Only `runtime_ledger.json` has a proper lockfile. `experiment_queue.json`, `experiment_archive.json`, `research_director_state.json`, and `research_coordination_state.json` are written concurrently without locking.

**Implementation:** Create a centralised `acquire_file_lock(path, timeout_s=10)` helper in `tar_lab/state.py` using the atomic `open(lock_path, "x")` pattern already used in `result_artifacts.py`. Apply to all multi-writer state file writes across:
- `tar_experiment_orchestrator.py` (writes to queue and archive)
- `tar_living_research.py` (writes to coordination state)
- `tar_research_director.py` (writes to director state)

**Verification:** Use two concurrent Python processes attempting to write `experiment_queue.json`; confirm only one succeeds; confirm the other waits and then succeeds after the lock is released.

---

### Task 0.10 — Disable the governor by default

Phase 11 ablation shows governor-alone is worse than SGD (0.250 vs 0.219 forgetting). The regime detector produces `rho=0, sigma=0, regime="unknown"` in every production trace. The LR adjustment never fires. Running it on every training step is wasted computation.

**Change in `tar_lab/multimodal_payloads.py` and `tar_lab/train_template.py`:**
- Add `use_governor: bool = False` to the experiment configuration schema.
- When `use_governor=False` (the new default), skip all regime detection and LR adjustment calls.
- Keep the infrastructure intact for Path B (governor repair in Phase 2.7).
- Add a config comment: `"Governor disabled by default pending activation investigation (Phase 2.7). Re-enable with use_governor=True."`

**Verification:** Run a smoke test with `use_governor=False`; confirm training completes normally. Run with `use_governor=True`; confirm governor code path is still reachable.

---

**Phase 0 exit gate — ALL must be true:**
- [ ] `harder_domain_split_tinyimagenet` complete, result honestly recorded in evidence inventory
- [ ] Six governance files committed with review comments
- [ ] `active_session.json` correctly reflects daemon state on start/stop
- [ ] Queue maintainer NoneType error absent from logs for 10+ minutes
- [ ] `tar_state/honest_evidence_inventory.json` exists with all historical results
- [ ] Financial gap scan contamination removed
- [ ] Watchdog circuit breaker operational (tested with forced rapid restarts)
- [ ] Experiment subprocess lifecycle managed (SIGTERM on crash, no duplicates)
- [ ] Centralised file lock helper in place for all shared state files
- [ ] Governor disabled by default with `use_governor=False`

**What Phase 0 unlocks:** Phases 1, 4, 7, and 9 can all begin in parallel.

---

*Stage 1 complete — Phases E0 and 0 with 15 tasks (E0.1–E0.5, 0.1–0.10)*
