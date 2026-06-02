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

---

## PHASE 1 — STATISTICAL & EVALUATION FOUNDATIONS
**Duration:** 2–4 weeks (begins immediately after Phase 0 exit gate)
**Track:** A (Science)
**Owner:** Statistician / Lead Researcher
**Objective:** Correct every statistical methodology error before any new claim is generated or paper drafted. Add the evaluation metrics that peer reviewers will require. Nothing in the paper pipeline can advance until this phase is complete.

---

### Task 1.1 — Replace z-critical with t-critical for small samples

**Priority:** CRITICAL — affects validity of all existing results

`benchmark_stats.py` and three other files use `1.96 × (std / √n)` regardless of sample size. For n=3 this makes confidence intervals ~2.2× too narrow (t-critical = 4.303 for df=2). This is a fundamental statistical error that any quantitative reviewer will catch.

**Implementation — apply to all four locations:**
```python
from scipy.stats import t as t_dist
df = sample_count - 1
t_crit = t_dist.ppf(0.975, df) if df > 0 else 1.96
ci95_half = t_crit * (std_dev / math.sqrt(sample_count))
```

**Files to update:**
- `tar_lab/benchmark_stats.py` — primary CI computation
- `phase8c_benchmark.py` — standalone benchmark script
- `ewc_lambda_sweep.py` — sweep result reporting
- `tar_lab/multimodal_payloads.py` — result JSON generation

After fixing: re-run all existing comparisons with corrected CIs. Record in the honest evidence inventory whether any previously "significant" results become non-significant. The answer for Phase 17 and Phase 16 (n=3) is: yes, their CIs widen substantially.

**Verification:** For n=3, std=0.01: old CI = ±0.011, new CI = ±0.025. Confirm the ratio is ~2.2×.

---

### Task 1.2 — Implement Bonferroni correction for all pairwise comparisons

**Priority:** CRITICAL — headline Phase 10 result fails this correction

Every phase comparing TCL against k baselines simultaneously tests at α=0.05 per test. With k=4 tests, the family-wise error rate is 18.5%. The corrected per-test threshold for k=4 is α/k = 0.0125. The Phase 10 TCL vs EWC result (p=0.019) does NOT survive this threshold. This must be acknowledged in the evidence inventory and in the paper.

**Implementation:**
```python
def bonferroni_correct(p_values: list[float], alpha: float = 0.05) -> dict:
    k = len(p_values)
    threshold = alpha / k
    return {
        "k": k,
        "threshold": threshold,
        "significant": [p < threshold for p in p_values],
        "p_values": p_values,
    }
```

Also implement Holm-Bonferroni (step-down, less conservative) as an alternative:
```python
def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    k = len(p_values)
    sorted_idx = sorted(range(k), key=lambda i: p_values[i])
    results = [False] * k
    for rank, idx in enumerate(sorted_idx):
        if p_values[idx] < alpha / (k - rank):
            results[idx] = True
        else:
            break  # stop at first non-significant (Holm step-down)
    return results
```

**Apply to:** Phase 10 (4 methods), Phase 11 ablation (4 conditions), Phase 12 EWC sweep (4 λ values), Phase 13 SI sweep (4 c values), HPC study (5 hypotheses → threshold = 0.01).

Update `honest_evidence_inventory.json` with `bonferroni_significant` field for every entry.

**Verification:** Phase 10 TCL vs EWC: p=0.019 > threshold 0.0125 → `bonferroni_significant: false`. Phase 10 TCL vs SGD: p=0.001 < threshold → `bonferroni_significant: true`.

---

### Task 1.3 — Conduct retrospective power analysis for all existing experiments

**Priority:** HIGH — quantifies how underpowered each result is

```python
from statsmodels.stats.power import TTestOneSamplePower

def power_analysis(effect_size_d: float, n_seeds: int, alpha: float = 0.05) -> dict:
    analysis = TTestOneSamplePower()
    achieved_power = analysis.power(effect_size=effect_size_d, nobs=n_seeds, alpha=alpha)
    n_for_80 = math.ceil(analysis.solve_power(effect_size=effect_size_d, power=0.80, alpha=alpha))
    n_for_90 = math.ceil(analysis.solve_power(effect_size=effect_size_d, power=0.90, alpha=alpha))
    return {"achieved_power_pct": round(achieved_power * 100, 1),
            "seeds_needed_80pct": n_for_80, "seeds_needed_90pct": n_for_90}
```

**Key results to record:**

| Experiment | d | n | Achieved power | Seeds for 80% |
|---|---|---|---|---|
| Phase 10 TCL vs EWC | 1.70 | 5 | ~90% | 5 |
| HPC autonomous | 1.73 | 5 | ~90% | 5 (but multiple-comparisons issue) |
| Phase 17 TCL vs EWC | 0.68 | 3 | ~30% | 18 |
| Phase 16 TCL vs EWC | ~0.50 est. | 3 | ~25% | 32 |
| Phase 13 SI sweep | varies | 3 | ~25% | 32 |

Write all results into `honest_evidence_inventory.json`.

**Verification:** Every entry in the inventory has `achieved_power_pct` and `seeds_needed_80pct` populated.

---

### Task 1.4 — Implement pre-registration protocol for all future experiments

**Priority:** HIGH — prevents further selective inference

Every future experiment must be pre-registered before the manifest is generated. The pre-registration is committed to git before data collection begins; any post-hoc change requires a documented amendment commit.

**`PreRegistrationRecord` Pydantic model** (add to `tar_lab/schemas.py`):
```python
class PreRegistrationRecord(BaseModel):
    experiment_id: str
    registered_at: str          # ISO timestamp, before manifest
    hypothesis: str             # Precise, falsifiable, one-directional
    primary_outcome: str        # "mean_forgetting" | "mean_accuracy" | both
    min_detectable_effect_d: float   # Pre-specified, not observed
    required_seeds: int         # From power analysis
    alpha: float = 0.05
    power_target: float = 0.80
    test_type: str              # "wilcoxon_paired" | "mann_whitney" | "t_one_sample"
    comparison_direction: str   # "less" | "greater"
    stopping_rule: str          # What constitutes a null result
    bonferroni_k: int = 1       # Number of simultaneous comparisons
    amendment_log: list[str] = []
```

**Gate:** Add `preregistration_present` as a fourth verification gate in `tar_lab/canonical_registry.py`. Results without a pre-registration record cannot progress past `trust_tier=exploration_grade`.

**Verification:** Attempt to register a result without a pre-registration; confirm the canonical registry raises a gate failure.

---

### Task 1.5 — Fix confidence intervals for non-normality

**Priority:** MEDIUM

Forgetting metrics are bounded below at 0 and can be right-skewed, particularly at small n. Add a Shapiro-Wilk normality test:

```python
from scipy.stats import shapiro

def compute_ci(values: list[float], alpha: float = 0.05) -> dict:
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if n >= 8:
        stat, p_shapiro = shapiro(values)
        is_normal = p_shapiro > 0.05
    else:
        is_normal = None  # Shapiro has no power at n<8; flag as unknown
    if is_normal or is_normal is None:
        # t-distribution CI (Task 1.1 formula)
        ci = t_dist_ci(mean, std, n, alpha)
        ci_method = "t_distribution"
    else:
        # Bootstrap CI (1000 resamples)
        ci = bootstrap_ci(values, n_resamples=1000, alpha=alpha)
        ci_method = "bootstrap"
    return {"mean": mean, "std": std, "ci_low": ci[0], "ci_high": ci[1],
            "ci_method": ci_method, "p_shapiro": p_shapiro if n >= 8 else None}
```

Record `ci_method` and `p_shapiro` in every comparison JSON output.

**Verification:** Inject a known right-skewed sample (exponential); confirm bootstrap CI is selected and is wider than t-distribution CI.

---

### Task 1.6 — Implement Wilcoxon signed-rank as the primary inferential test for paired comparisons

**Priority:** MEDIUM

The current code uses Mann-Whitney U (unpaired) and paired t-test inconsistently. Standardise:

- **Seed-matched comparisons** (same seed index, two methods): Wilcoxon signed-rank test
- **Independent-group comparisons**: Mann-Whitney U
- **Report both** parametric (paired t-test / independent t-test) and non-parametric results for all major claims

```python
from scipy.stats import wilcoxon, mannwhitneyu

def compare_methods(a: list[float], b: list[float], paired: bool = True) -> dict:
    if paired:
        stat, p_np = wilcoxon(a, b, alternative='less')
        test_name = "wilcoxon_signed_rank"
    else:
        stat, p_np = mannwhitneyu(a, b, alternative='less')
        test_name = "mann_whitney_u"
    # Also compute parametric
    from scipy.stats import ttest_rel, ttest_ind
    _, p_param = (ttest_rel if paired else ttest_ind)(a, b, alternative='less')
    return {"test": test_name, "p_nonparametric": p_np, "p_parametric": p_param,
            "statistic": stat}
```

Update all phase comparison scripts to use this function.

**Verification:** On Phase 10 data: both Wilcoxon and paired t-test return same significance direction; confirm both results are recorded in output JSON.

---

### Task 1.7 — Add SPRT sequential testing boundary to HPC replication

**Priority:** HIGH — saves ~30% compute on decisive results

The HPC replication (Phase 2.1) will run 20+ seeds sequentially. A Sequential Probability Ratio Test (SPRT) allows early stopping when evidence is decisive, reallocating GPU hours to other experiments.

**Implementation:**
```python
def sprt_boundary(n: int, p_values_so_far: list[float], 
                  alpha: float = 0.05, beta: float = 0.10) -> str:
    """Returns 'accept_H1', 'accept_H0', or 'continue'."""
    # Wald's SPRT approximation
    A = math.log((1 - beta) / alpha)   # upper boundary
    B = math.log(beta / (1 - alpha))   # lower boundary
    # Log-likelihood ratio based on observed p-values
    log_lr = sum(math.log(0.05 / p) if p < 0.05 else math.log(0.95 / (1 - p))
                 for p in p_values_so_far)
    if log_lr >= A:
        return "accept_H1"  # Stop: effect is real
    elif log_lr <= B:
        return "accept_H0"  # Stop: no effect
    return "continue"
```

The HPC replication checks this boundary after every 4 seeds. If `accept_H1` or `accept_H0` is returned, halt and record the decision. If evidence is borderline, continue to n=25.

**Verification:** Simulate a run where all p-values are 0.01; confirm early stopping at seed 8–10.

---

### Task 1.8 — Add BWT, FWT, and intransigence metrics to all results

**Priority:** CRITICAL — reviewers will ask for these; they cost nothing to add

These three metrics are computable from the `acc_matrix` already produced by `generic_cl_runner.py`. Not adding them to a 2026 CL paper will generate mandatory revision requests.

**Definitions (Javed & White, 2019; Lopez-Paz & Ranzato, 2017):**
```python
def compute_transfer_metrics(acc_matrix: list[list[float]]) -> dict:
    T = len(acc_matrix)
    # Backward Transfer: Σ (acc[t][T-1] - acc[t][t]) / (T-1)
    bwt = sum(acc_matrix[t][-1] - acc_matrix[t][t]
              for t in range(T - 1)) / max(T - 1, 1)
    # Forward Transfer: Σ (acc[t][t] - acc_random[t]) / T
    # acc_random = 1/n_classes (chance level)
    fwt = sum(acc_matrix[t][t] for t in range(T)) / T  # simplified
    # Intransigence: fraction of tasks with >5% forgetting
    forgetting_per_task = [max(0, max(acc_matrix[t]) - acc_matrix[t][-1])
                           for t in range(T - 1)]
    intransigence = sum(1 for f in forgetting_per_task if f > 0.05) / max(len(forgetting_per_task), 1)
    return {"bwt": round(bwt, 4), "fwt": round(fwt, 4),
            "intransigence_index": round(intransigence, 4),
            "forgetting_per_task": [round(f, 4) for f in forgetting_per_task]}
```

Add `compute_transfer_metrics()` call at the end of `_run_one_seed()` in `generic_cl_runner.py`. Include in all result JSONs and in the honest evidence inventory.

Re-compute for all existing experiments (Phase 10, 11, 16, 17) from the raw `acc_matrix` stored in comparison JSONs.

**Verification:** Phase 10 Phase 10 results now include `bwt`, `fwt`, `intransigence_index` fields.

---

### Task 1.9 — Add per-task calibration (ECE) trajectory measurement

**Priority:** HIGH — a method that reduces forgetting but becomes overconfident is not actually better

ECE is currently reported as a single scalar at the end of all tasks. Add per-task ECE after each task is trained:

```python
def compute_per_task_ece(model, task_loaders: list, device, n_bins: int = 15) -> list[float]:
    ece_per_task = []
    for loader in task_loaders:
        confidences, accuracies = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                probs = torch.softmax(model(x.to(device)), dim=-1)
                conf, pred = probs.max(dim=-1)
                confidences.extend(conf.cpu().tolist())
                accuracies.extend((pred.cpu() == y).float().tolist())
        ece = _compute_ece(confidences, accuracies, n_bins)
        ece_per_task.append(round(ece, 4))
    return ece_per_task
```

Store `ece_trajectory` (list of floats, one per task after training) in the result JSON alongside `acc_matrix`.

**What to look for:** If ECE increases across tasks while forgetting decreases, TCL is trading calibration for retention — that is a genuine limitation that must be disclosed.

**Verification:** Phase 10 rerun includes `ece_trajectory: [0.12, 0.14, 0.18, 0.22, 0.25]` (expected upward trend as tasks accumulate).

---

### Task 1.10 — Add Bayesian credible intervals as the primary evidence metric

**Priority:** MEDIUM — more informative than p-values; enables adaptive stopping

Replace the frequentist p-value as the primary evidence metric with the posterior probability that the treatment effect is positive:

```python
from scipy.stats import t as t_dist
import numpy as np

def bayesian_evidence(deltas: list[float], prior_scale: float = 0.5) -> dict:
    """
    deltas: list of (method_A - method_B) forgetting differences, one per seed.
    Negative delta = A is better (lower forgetting).
    Prior: half-normal(0, prior_scale) on effect size d.
    Returns posterior P(mean_delta < 0) = P(A better than B).
    """
    n = len(deltas)
    mean_d = np.mean(deltas)
    std_d = np.std(deltas, ddof=1)
    se = std_d / np.sqrt(n)
    # t-posterior: P(delta < 0 | data)
    t_stat = mean_d / se
    df = n - 1
    p_better = float(t_dist.cdf(-abs(t_stat), df=df))  # one-sided P(A<B)
    # 95% credible interval
    ci_half = t_dist.ppf(0.975, df=df) * se
    return {
        "posterior_p_better": round(p_better, 4),
        "posterior_mean_delta": round(mean_d, 4),
        "credible_interval_95": [round(mean_d - ci_half, 4), round(mean_d + ci_half, 4)],
        "interpretation": f"P(TCL beats EWC) = {p_better:.1%}",
    }
```

Include `bayesian_evidence` output in all comparison JSONs and in the honest evidence inventory. Report it alongside (not instead of) the frequentist p-value.

**Verification:** Phase 10 TCL vs EWC: `posterior_p_better ≈ 0.98`, interpretation "P(TCL beats EWC) = 98%". This is more intuitive than p=0.019.

---

**Phase 1 exit gate — ALL must be true:**
- [ ] t-distribution CI formula applied in all four files; old results recomputed
- [ ] Bonferroni correction applied to all multi-comparison phases; honest_evidence_inventory updated
- [ ] Power analysis table complete for all historical results
- [ ] `PreRegistrationRecord` schema in schemas.py; fourth canonical gate enforced
- [ ] Shapiro-Wilk normality check in statistics pipeline; bootstrap CI fallback working
- [ ] Wilcoxon signed-rank implemented and used as primary test
- [ ] SPRT boundary logic implemented for HPC replication
- [ ] BWT, FWT, intransigence index computed for all existing experiments
- [ ] Per-task ECE trajectory added to result schema; existing phases recomputed
- [ ] Bayesian credible intervals included in all comparison outputs

**What Phase 1 unlocks:** Phase 2 experiments can begin; Phase 5 paper drafting can start with stubs.

---

## PHASE 2 — EXPERIMENTAL RIGOUR
**Duration:** 4–8 weeks
**Track:** A (Science)
**Owner:** Lead Researcher / Algorithm Engineer
**Objective:** Bring every key result to publication-grade evidence through correct sample sizes, fair baseline tuning, added baselines, and honest scope delimitation. GPU budget is the constraint; allocate ruthlessly.

**GPU budget allocation:**

| Frontier | Hours | Decision |
|---|---|---|
| `fp-hyperparameter-robustness` | 48–56h | PROCEED — publication-critical |
| `fp-catastrophic-forgetting` | 0h | CLOSE — 17 null, 4 adverse, falsified |
| `fp-regime-detection-accuracy` | 0h | DEFER — document as Path A, future work |
| `fp-scale-up` | 2h pilot only | DEFER — one small pilot, abandon if negative |
| `fp-class-incremental` | 2h re-analysis | DEFER — re-analyse Phase 15 data for Paper 2 |

---

### Task 2.1 — Replicate the HPC result with adequate power

**Priority:** CRITICAL — this is the system's single strongest finding; it must be confirmed or disproved

The `high_penalty_conservative` result (p=0.018, d=1.73) emerged from a 5-hypothesis simultaneous battery at uncorrected α. A single positive result from 5 tests has a false discovery probability of 23% (1−0.95⁵). Replication on fresh data at a pre-registered single hypothesis is the only scientific resolution.

**Pre-registered protocol:**
- Hypothesis: HPC mean_forgetting < TCL_baseline mean_forgetting (one-tailed, paired, pre-registered before data collection)
- Seeds: 20 new seeds beyond the original 5 (total n=25)
- Dataset: Split-CIFAR-10
- SPRT boundary: check after every 4 seeds (Task 1.7); stop early if decisive
- No multiple comparison correction (this is a single pre-registered test)
- Success criterion: p<0.05 (Wilcoxon signed-rank, one-tailed), d≥0.5
- Failure criterion: p>0.10 at n=20 seeds → record as false positive

**If replication fails:** Record HPC as a false positive in the honest evidence inventory. The paper then focuses on Phase 10 (TCL vs EWC) as the primary result. This is the honest outcome.

**Verification:** Pre-registration commit exists with timestamp before first seed run; result JSON includes SPRT decision log.

---

### Task 2.2 — Rerun Phase 17 (TinyImageNet) with 5 seeds and full baseline set

**Priority:** HIGH — current result (n=3) fails Gate B

**Pre-registered design:**
- Seeds: [42, 0, 1, 2, 3] (consistent with all other phases)
- Methods: TCL, EWC (λ=1000 from Phase 12 optimum), SI (c=0.01), SGD baseline, **DER++ (new, mem=200)**, **LwF (new)**
- Epochs: 40 per task; backbone: ResNet-18
- Primary comparison: TCL vs EWC (Wilcoxon signed-rank, paired, one-tailed, α=0.05/4=0.0125 Bonferroni)
- Secondary comparisons: TCL vs SGD, TCL vs DER++, TCL vs LwF
- Stopping rule: if TCL forgetting > EWC forgetting in ≥3/5 seeds → record as ADVERSE

**New metrics to record:** BWT, FWT, intransigence index, per-task ECE trajectory, Bayesian credible interval (all from Phase 1 tasks).

**Verification:** Result JSON contains all new fields. Trust tier classification: `trusted_rerun_with_env` (requires env snapshot). Seed count: 5 → Gate B passes.

---

### Task 2.3 — Rerun Phase 16 (CIFAR-100) with 5 seeds and full baseline set

**Priority:** HIGH — current result (n=3) fails Gate B

**Pre-registered design:** Mirrors Phase 2.2 but for Split-CIFAR-100 (10 tasks × 10 classes, epochs=40, same backbone).

All same metrics and stopping rules apply.

**Verification:** Result JSON contains all new fields; seed count 5; trust tier `trusted_rerun_with_env`.

---

### Task 2.4 — Implement and add DER++ to all comparison phases

**Priority:** HIGH — every top-venue CL paper requires a replay baseline; omitting it is an automatic reviewer flag

DER++ (Buzzega et al., 2020) is already in `method_registry.py` but has never appeared in any phase comparison. It represents a fundamentally different protection class (replay-based) versus regularization-based methods.

**Hyperparameter tuning (Task 2.6 dependency):** Sweep `der_mem_size` over [100, 200, 500] on the held-out validation split (seed=999) from Task 2.6. Select best performing. Report the sweep results in paper appendix.

**Integration:** Add `der_plus_plus` to the methods list in all comparison experiments (Phases 10, 16, 17 reruns).

**Critical honesty principle:** If DER++ outperforms TCL, record it honestly. A paper that shows DER++ > TCL > EWC > SGD is more publishable (honest comparison) than one that omits DER++ and claims TCL is best.

**Verification:** Phase 10 rerun result JSON includes `der_plus_plus` results. `method_registry.py` DER++ entry passes minibench validation (Task 4.3).

---

### Task 2.5 — Implement LwF and add to all comparison phases

**Priority:** CRITICAL — absence of LwF will generate a mandatory revision request

LwF (Li & Hoiem, 2016) is the canonical knowledge-distillation CL baseline. It is listed in `frontier_problems.json` but not in `method_registry.py`.

**Implementation in `method_registry.py`:**
```python
@register_method("lwf")
class LwFMethod(CLMethod):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = float(getattr(config, "lwf_alpha", 0.5))
        self.temperature = float(getattr(config, "lwf_temperature", 2.0))
        self._old_model: Optional[nn.Module] = None

    def pre_task(self, task_id: int, model: nn.Module, device: torch.device) -> None:
        if task_id > 0:
            import copy
            self._old_model = copy.deepcopy(model).eval().to(device)

    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        return torch.tensor(0.0)  # handled in augmented_loss

    def augmented_loss(self, model, x, y, task_id, device) -> torch.Tensor:
        if self._old_model is None or task_id == 0:
            return torch.tensor(0.0, device=device)
        with torch.no_grad():
            old_logits = self._old_model(x.to(device)) / self.temperature
        new_logits = model(x.to(device)) / self.temperature
        return self.alpha * F.kl_div(
            F.log_softmax(new_logits, dim=-1),
            F.softmax(old_logits, dim=-1),
            reduction="batchmean"
        )
    
    def post_task(self, task_id, model, loader, device):
        import copy
        self._old_model = copy.deepcopy(model).eval().to(device)
```

**Bonus:** Also test TCL + LwF combined (elastic penalty from Task gradient-EMA + distillation from LwF). This combination is untested and potentially additive. If it outperforms either alone, it is a novel contribution.

**Verification:** `method_registry.py` contains `lwf` key; LwF passes minibench validation (Task 4.3); Phase 10 rerun includes `lwf` results.

---

### Task 2.6 — Implement fair joint hyperparameter selection for all baselines

**Priority:** HIGH — current tuning is demonstrably post-hoc (EWC λ swept after Phase 10; SI c=0.01 chosen because c=0.1 collapses)

**Protocol:**
1. Designate **Split-CIFAR-10, seed=999** as the held-out validation split. This seed is not in any existing experiment.
2. Tune all methods jointly on this split before running any confirmatory experiments:
   - EWC: λ ∈ {10, 100, 1000, 5000} → select lowest forgetting with accuracy > 0.6
   - SI: c ∈ {0.001, 0.01, 0.1, 1.0} → select lowest forgetting without collapse (accuracy > 0.5)
   - DER++: mem_size ∈ {100, 200, 500} → select lowest forgetting
   - LwF: α ∈ {0.3, 0.5, 1.0}, T ∈ {1.0, 2.0} → select lowest forgetting
   - TCL: keep fixed (λ=1.0, ema_beta=0.99) — do not tune post-hoc
3. Lock selected hyperparameters. Commit the selection decision to git with a pre-registration commit before any confirmatory experiments run.
4. Report all tuning runs in a paper appendix clearly labelled "Validation Sweep (seed=999)".

**Verification:** A `hyperparameter_selection.json` file exists in `tar_state/` with the selected values and the seed=999 validation results. Commit timestamp predates any confirmatory experiment runs.

---

### Task 2.7 — Resolve the regime-detection activation failure

**Priority:** HIGH — the governor never fires; the paper cannot claim a thermodynamic mechanism

**Time budget: 2 weeks for Path B. If unresolved after 2 weeks, execute Path A.**

**Path B (Fix the Governor):**

Root cause investigation:
1. Log actual `sigma` and `sigma_star` values per layer per batch during training on Split-CIFAR-10.
2. Check whether `multimodal_payloads.py` uses `loss` (scalar) as the sigma proxy instead of the activation-based sigma from `ActivationThermoObserver`.
3. Check the `warmup_batches=0` default: the anchor is being set from random-initialization noise (first 20 batches), which may produce a sigma_star that is never crossed.
4. Hypothesis: set `warmup_batches=60` (approximately 2 epochs) to let the model stabilize before locking sigma_star.

If fix is found: run the 7-condition ablation (Task 3.1) with the fixed governor to quantify its contribution.

**Path A (Honest Ablation, fallback):**
- Document that the regime-detection LR adjustment contributes zero net benefit.
- The elastic penalty term is the entire mechanism.
- Provide rigorous ablation showing penalty-only ≥ full-TCL performance.
- Rename the contribution: not "thermodynamic governor" but "gradient-EMA elastic regularization."
- Update paper framing accordingly (Task 3.3).

**Verification (Path B):** At least one experiment shows `regime != "unknown"` during training; governor LR adjustment fires at least once per 100 batches.
**Verification (Path A):** A formal ablation document in `tar_state/stat_audit/` records governor contribution as zero with confidence intervals.

---

### Task 2.8 — Formally close the falsified frontier

**Priority:** MEDIUM — necessary for honest scope

The frontier `fp-catastrophic-forgetting` has 17 null results, 4 adverse results, and 1 breakthrough across diverse conditions. This is not a research frontier in progress; it is a falsified hypothesis.

- Write a formal null-result record in `tar_state/validation/fp_catastrophic_forgetting_closure.json`:
  ```json
  {
    "frontier_id": "fp-catastrophic-forgetting",
    "verdict": "FALSIFIED",
    "evidence_summary": "17 null, 4 adverse, 1 breakthrough across 22 experiments",
    "honest_conclusion": "TCL does not solve general catastrophic forgetting; results are configuration-specific",
    "closed_at": "<timestamp>",
    "papers_affected": ["integration-test"]
  }
  ```
- Update all papers that reference this frontier to remove overclaiming language.
- Set `truth_status: "falsified"` in `frontier_problems.json` for this entry.

**Verification:** `tar_cli.py frontier_status` shows `fp-catastrophic-forgetting` with `truth_status: "falsified"`.

---

### Task 2.9 — Add GEM/A-GEM as a baseline

**Priority:** HIGH — constraint-based methods are a different protection class; their absence weakens the paper

GEM (Lopez-Paz & Ranzato, 2017) enforces that new task gradients do not increase old task losses. A-GEM (Chaudhry et al., 2019) is the memory-efficient variant.

**Implementation in `method_registry.py`:**

A-GEM is the priority (lower memory cost):
```python
@register_method("agem")
class AGEMMethod(CLMethod):
    def __init__(self, config):
        super().__init__(config)
        self.mem_size = int(getattr(config, "agem_mem_size", 200))
        self._mem_x: list[torch.Tensor] = []
        self._mem_y: list[torch.Tensor] = []

    def post_task(self, task_id, model, loader, device):
        # Reservoir sample from task data
        for x, y in loader:
            for xi, yi in zip(x, y):
                if len(self._mem_x) < self.mem_size:
                    self._mem_x.append(xi.cpu())
                    self._mem_y.append(yi.cpu())
                else:
                    idx = random.randint(0, task_id * len(x))
                    if idx < self.mem_size:
                        self._mem_x[idx] = xi.cpu()
                        self._mem_y[idx] = yi.cpu()

    def regularization_loss(self, model):
        return torch.tensor(0.0)  # handled via gradient projection

    def project_gradients(self, model, device):
        """Project new task gradients to not conflict with memory gradients."""
        if not self._mem_x:
            return
        mx = torch.stack(self._mem_x[:32]).to(device)
        my = torch.stack(self._mem_y[:32]).to(device)
        mem_loss = F.cross_entropy(model(mx), my)
        mem_loss.backward()
        # Project current gradient onto half-space defined by memory gradient
        for p in model.parameters():
            if p.grad is not None and hasattr(p, '_mem_grad'):
                dot = (p.grad * p._mem_grad).sum()
                if dot < 0:
                    p.grad -= dot / (p._mem_grad.norm() ** 2 + 1e-8) * p._mem_grad
```

Add to all comparison phases (10, 16, 17 reruns).

**Verification:** A-GEM passes minibench validation (Task 4.3); Phase 10 rerun includes `agem` results.

---

### Task 2.10 — Run class-order Spearman confound audit

**Priority:** HIGH — if significant, it invalidates the headline result

`generic_cl_runner.py` shuffles class order per seed using the seed value. If TCL exploits class-order-specific artifacts, the observed forgetting reduction is a confound, not a method effect.

**Test:**
```python
import scipy.stats

# For each method across all seeds in Phase 10:
class_order_seeds = [42, 0, 1, 2, 3]
tcl_forgetting = [result[seed]["tcl"]["mean_forgetting"] for seed in class_order_seeds]
# Compute Spearman correlation between seed index and TCL forgetting
rho, p_value = scipy.stats.spearmanr(class_order_seeds, tcl_forgetting)
print(f"Spearman rho = {rho:.3f}, p = {p_value:.3f}")
```

**Decision rule:**
- If |ρ| > 0.6: class order is a confound. Must control for it by running all seeds with multiple fixed class orders and computing forgetting averaging over orders.
- If |ρ| ≤ 0.6: confound is negligible; document as a limitation.

**Verification:** A `confound_audit.json` in `tar_state/stat_audit/` records the Spearman ρ for each method and the decision.

---

### Task 2.11 — Run BatchNorm task-boundary reset ablation

**Priority:** MEDIUM — if BatchNorm memory is a confounder, it must be controlled

`generic_cl_runner.py` does not reset BatchNorm running statistics at task boundaries. BN statistics for task 5 implicitly carry information from tasks 1–4.

**Test:**
```python
# Add at each task boundary:
def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()

# Compare: with BN reset vs without BN reset for each method
```

Run both variants on Phase 10 (5 seeds each). If mean_forgetting differs by >0.01 between variants, BN state is a meaningful confounder.

**Verification:** Result shows Δforgetting between reset and non-reset variants. Record in `confound_audit.json`. If significant, add BN reset as default and re-run main comparisons.

---

### Task 2.12 — Verify weight initialization seed reproducibility

**Priority:** LOW — necessary for reproducibility claims

`_set_seed()` sets `torch.manual_seed` but `_ResNet18Trunk` uses `weights=None` (kaiming uniform). The interaction between manual seeds and PyTorch's RNG may not be fully deterministic across PyTorch versions.

**Test:**
```python
_set_seed(42)
model_a = _build_model("resnet18", 10, device)
weights_a = {n: p.data.clone() for n, p in model_a.named_parameters()}

_set_seed(42)
model_b = _build_model("resnet18", 10, device)
weights_b = {n: p.data.clone() for n, p in model_b.named_parameters()}

for name in weights_a:
    assert torch.allclose(weights_a[name], weights_b[name]), f"Mismatch in {name}"
print("Weight initialization is deterministic under the same seed.")
```

If any assertion fails: document which layers are non-deterministic and add an explicit `torch.manual_seed` call before weight initialization.

**Verification:** Test passes; commit it as `tests/test_reproducibility.py`.

---

### Task 2.13 — Re-analyse Phase 15 data for class-incremental Paper 2 scoping

**Priority:** LOW (for Paper 1) — HIGH (for Paper 2 planning)

Phase 15 data shows p=0.012, d=5.26 for TCL in class-incremental (CIL) setting with only 3 seeds. This is promising but underpowered.

- Load Phase 15 raw data; apply t-distribution CI correction (Task 1.1) and compute Bonferroni-corrected threshold.
- Compute BWT, FWT, intransigence for the CIL setting.
- Produce a brief scoping document: "What would a full CIL paper require?" (n seeds, which benchmarks, which comparators including DualPrompt and L2P).
- Document this as the foundation for Paper 2 (class-incremental focus).

**Verification:** A `tar_state/papers/paper_2_scoping.md` document exists outlining the CIL paper plan.

---

**Phase 2 exit gate — ALL must be true:**
- [ ] HPC replication (n=25) complete; result honestly recorded (confirmed or falsified)
- [ ] Phase 17 rerun: 5 seeds, DER++, LwF, A-GEM complete
- [ ] Phase 16 rerun: 5 seeds, DER++, LwF, A-GEM complete
- [ ] Hyperparameter selection protocol run on seed=999; locked and pre-registered
- [ ] Regime detector investigation resolved (Path A or Path B documented)
- [ ] `fp-catastrophic-forgetting` formally closed with evidence record
- [ ] DER++, LwF, A-GEM baselines added and validated
- [ ] Class-order and BatchNorm confound audits complete; results recorded
- [ ] Weight initialization reproducibility test passing
- [ ] All results include BWT, FWT, intransigence, per-task ECE

**What Phase 2 unlocks:** Phase 5 paper pipeline can begin in earnest.

---

---

## PHASE 3 — MECHANISTIC CLARITY & THEORETICAL GROUNDING
**Duration:** 4–8 weeks (overlaps with Phase 2)
**Track:** A (Science)
**Owner:** Lead Researcher
**Objective:** Produce a coherent, defensible theoretical narrative grounded in what the experiments actually show. Retire the thermodynamic framing where it is not grounded. Build the Fisher-EMA theorem that constitutes the paper's theoretical contribution.

---

### Task 3.1 — Conduct the definitive 7-condition mechanistic ablation

**Priority:** HIGH — required to understand what actually drives TCL's performance

Phase 11 established that penalty > governor. This ablation must determine: why does the penalty work? Does the governor contribute at all under ideal conditions? Does the per-task anchor (sigma_star freezing) matter?

**Pre-registered conditions (all on Split-CIFAR-10, n=5 seeds, Bonferroni-corrected across 6 inter-condition comparisons):**

| Condition ID | Description | Primary hypothesis |
|---|---|---|
| `sgd_baseline` | No regularization | Worst forgetting (positive control) |
| `penalty_only` | Gradient-EMA elastic penalty, no governor LR adjustment | Should match or beat full TCL |
| `governor_only` | Regime-detection LR adjustment, no penalty | Hypothesis: ≈ SGD (from Phase 11) |
| `full_tcl` | Both penalty and governor | Should match or slightly exceed penalty-only |
| `anchor_frozen_at_init` | Penalty with sigma_star fixed from random init (no per-task reset) | Controls for per-task anchor mechanism |
| `warmup_batches_60` | Penalty with governor, warmup_batches=60 | Tests whether warmup enables governor activation |
| `ewc_best_lambda` | EWC at Phase 12-tuned λ | External comparator establishing reference |

**Pre-registered primary comparison:** penalty_only vs sgd_baseline (tests whether gradient-EMA importance adds value at all).

**Pre-registered secondary comparisons (Bonferroni threshold α=0.05/6=0.0083):**
- full_tcl vs penalty_only (tests governor marginal contribution)
- anchor_frozen_at_init vs penalty_only (tests per-task anchor value)
- warmup_batches_60 vs governor_only (tests whether warmup fixes governor)
- penalty_only vs ewc_best_lambda (establishes paper's theoretical claim)

**Verification:** All 7 conditions complete with n=5 seeds; result JSON includes all conditions; Bonferroni-corrected significance table produced.

---

### Task 3.2 — Formalise the Fisher-EMA theorem

**Priority:** HIGH — without this, the Methods section makes a claim but not a contribution

The plan names this as a deliverable but does not contain the actual derivation. Here is what it must say:

**Theorem (Temporally-Smoothed Fisher Estimation):**

Let `g_i(t)` denote the gradient of parameter `θ_i` at training step t. Define the EMA importance estimate: `v_i(T) = (1−β) Σ_{t=1}^{T} β^{T−t} g_i(t)²`.

Define the true diagonal Fisher: `F_ii = E_{x~D_task}[(∂ℓ/∂θ_i)²]` where D_task is the task data distribution.

**Claim:** Under i.i.d. sampling with replacement from D_task, as T → ∞:

`|v_i(T) − F_ii| ≤ C · β^T`

where `C = F_ii` (bounded by the Fisher value itself), and the convergence rate is controlled by the EMA decay β.

**Corollary (EWC as a limiting case):** EWC's importance estimate is `w_i^EWC = (1/N) Σ_{t=T-N}^{T} g_i(t)²` (uniform average over the final N batches). When β → 0 (fast decay), `v_i(T) ≈ g_i(T)²` (last-batch only). When β → 1 (slow decay), `v_i(T) ≈ (1/T) Σ_{t=1}^{T} g_i(t)²` (uniform average = time-discounted EWC). EWC's snapshot is thus recovered as a special case.

**Key difference:** EWC's snapshot is biased by gradient non-stationarity within the task (gradients at task start vs end differ substantially during convergence). TCL's EMA over-weights recent gradients but captures the full training trajectory. In non-stationary settings, TCL's bias-variance tradeoff is typically better.

**What to derive formally:**
- The bias term for EWC (gradient non-stationarity within task training)
- The bias term for TCL (exponential weighting of recent gradients)
- Show that for gradient processes with mixing time τ, setting `β = 1 − 1/τ` minimises total bias + variance

**Verification:** A LaTeX proof sketch of at least 1 page exists in `tar_state/papers/fisher_ema_theorem.tex`. The derivation has been verified by computing numerical values for Phase 10 and confirming that `v_i(T)` converges to the empirical Fisher over the training run.

---

### Task 3.3 — Reframe the contribution correctly and retire the thermodynamic language

**Priority:** HIGH — the paper cannot be submitted with a broken theoretical frame

**From the theory audit:** The thermodynamic analogy requires stochastic dynamics (Langevin), which TAR's deterministic SGD does not have. There is no Boltzmann distribution. The sigma metric (`lr × ||grad||²`) is not entropy in any statistical mechanics sense.

**Action plan:**

1. **Paper title:** Remove "thermodynamic" from the title. Candidate titles:
   - "Continuous Gradient Energy Accumulation for Low-Tuning Continual Learning"
   - "Temporal Fisher Smoothing Reduces Catastrophic Forgetting"
   - "Why EWC's Canonical Hyperparameters Underfit Activation Heterogeneity: A Controlled Replication"

2. **Terminology replacement throughout the paper:**
   - "thermodynamic governor" → "adaptive learning rate monitor" (or omit if Path A)
   - "thermal regime" → "training dynamics regime"
   - "ordered/disordered/critical" → "converged/learning/transitioning"
   - "entropy sigma" → "step magnitude"
   - "regime rho" → "relative learning rate ratio"

3. **What can stay:** The motivating intuition — "parameters with high gradient energy are 'hot' and important" — is valid as an analogy. It can appear in the introduction as motivation only, clearly labelled as intuition rather than formal claim.

4. **Update `tar_author.py` prompts:** All section generation prompts that reference "thermodynamic" framing must be updated to use the corrected framing. Check every `_AUTHOR_SYSTEM` and section-specific prompt.

**Verification:** Search `tar_author.py` for the word "thermodynamic"; confirm it appears only in historical comments, not in active prompts.

---

### Task 3.4 — Formally characterise HPC: lambda vs momentum ablation

**Priority:** HIGH (if HPC replication succeeds) — determines whether HPC is a new mechanism or better hyperparameters

HPC differs from TCL baseline in two ways: (1) higher regularization lambda, and (2) SGD momentum disabled. A 4-condition ablation determines which factor drives the forgetting reduction.

**Pre-registered 4-condition design (Split-CIFAR-10, n=5 seeds, Bonferroni threshold α=0.05/3=0.0167):**

| Condition | Lambda | Momentum | Expected |
|---|---|---|---|
| `tcl_baseline` | Standard (0.01) | Enabled (0.9) | Baseline forgetting |
| `tcl_high_lambda` | High (0.1–1.0) | Enabled (0.9) | Tests lambda effect |
| `tcl_no_momentum` | Standard (0.01) | Disabled (0.0) | Tests momentum effect |
| `hpc` | High | Disabled | Combined (best) |

**Pre-registered primary comparison:** `hpc` vs `tcl_high_lambda` — tests whether momentum removal adds value beyond lambda increase.

**Decision rule:**
- If `tcl_high_lambda` ≈ `hpc` (p>0.05): momentum removal adds nothing; HPC is "TCL with better lambda" — a robustness finding.
- If `hpc` significantly beats `tcl_high_lambda`: momentum removal is mechanistically important; HPC is a new contribution.

**Verification:** All 4 conditions complete; pre-registration committed before data collection; result reported in paper ablation section.

---

### Task 3.5 — Build and empirically test the D_PR–forgetting connection

**Priority:** MEDIUM — theoretical standing of D_PR as a forgetting signal

D_PR (participation ratio of activation covariance) is used throughout the system as a monitoring metric but has no formal connection to forgetting established.

**Empirical test:**

For each method (TCL, EWC, SGD, DER++) across all seeds in Phase 10 rerun:
1. Record `D_PR` after each task boundary (at the moment of task switch)
2. Record `forgetting_t` for each task t at the end of training
3. Compute Spearman correlation between `D_PR_drop_t` (D_PR compression during task t+1) and `forgetting_t`

**Expected finding:** Lower D_PR at task boundaries correlates with higher forgetting (compressed representation means task-T features were overwritten). If confirmed, this provides empirical evidence that TCL's penalty resists D_PR compression — making D_PR a valid diagnostic.

**Theoretical connection to write:** TCL's elastic penalty adds a restoring force on each parameter proportional to its importance. High-importance parameters resist update, preserving the covariance structure of task-T activations. This is the mechanism linking the penalty to D_PR stability.

**Verification:** A correlation plot of D_PR_drop vs forgetting_t is produced for each method. Spearman ρ and p-value reported. If |ρ| > 0.5 and p < 0.05, D_PR is a meaningful signal.

---

### Task 3.6 — Assess ASC novelty and plan the TCL-for-LLMs prototype

**Priority:** MEDIUM — establishes whether ASC should be published separately or folded into TCL

**ASC novelty assessment:**

The ASC architecture (EMA target model + online student + consistency loss + LatentWarp MLP) is architecturally identical to Mean Teacher (Tarvainen & Valpola, 2017) with a learned adversarial perturbation head. Mean Teacher is 2017 work. The LatentWarp head does not constitute novelty over standard knowledge distillation.

**Decision:** Do not submit ASC as a standalone contribution. Instead, use it as the foundation for TCL-for-LLMs (Phase 10 long-term), where the combination is genuinely novel.

**TCL-for-LLMs architecture sketch (for Paper 2 planning):**

```
TCL-for-LLMs = Task-aware Continual Fine-tuning for Language Models

During fine-tuning on task T:
  - Accumulate per-parameter EMA of gradient energy (ThermalImportance)
  - EMA target model (from ASC) = frozen anchor for importance computation

During fine-tuning on task T+1:
  - Add TCL elastic penalty: L_reg = λ · Σ_i v_i^T · (θ_i − θ_i^T)²
  - Prevents catastrophic forgetting of task-T fine-tuning knowledge
  - Works with LoRA: importance tracks only the adapter parameters
```

**Verification:** A `tar_state/papers/paper_2_tcl_llms_architecture.md` scoping document exists. The prototype code sketch is committed to a branch `feature/tcl-llms`.

---

### Task 3.7 — Prototype second-order importance (Hessian diagonal upgrade)

**Priority:** MEDIUM — potential +2–4% forgetting reduction; genuinely novel importance estimator

Upgrade `ThermalImportance` from first-order (gradient-squared EMA) to second-order by adding the Hessian diagonal:

```python
def accumulate_second_order(self, model: nn.Module, 
                             loss: torch.Tensor, gamma: float = 0.1) -> None:
    """Call after loss.backward() — before optimizer.step()."""
    beta = self.ema_beta
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and name in self._v:
            g2 = p.grad.detach().float() ** 2
            self._v[name].mul_(beta).add_(g2, alpha=1.0 - beta)
            # Hessian diagonal via finite differences or Pearlmutter trick
            # Simple approximation: h_ii ≈ g_i² / (||g||² + eps) (Fisher-Rao normalization)
            grad_norm_sq = sum((q.grad ** 2).sum() for q in model.parameters() 
                               if q.grad is not None).item()
            h_approx = g2 / (grad_norm_sq + 1e-8)
            self._v[name].add_(h_approx * gamma)
    self._step += 1
```

Run a 3-condition ablation (Split-CIFAR-10, n=5 seeds): first-order only vs second-order only vs combined. Report Δforgetting and ΔCI95.

**Verification:** `ThermalImportance` has `accumulate_second_order` method; ablation result JSON exists.

---

**Phase 3 exit gate — ALL must be true:**
- [ ] 7-condition mechanistic ablation complete; penalty_only vs sgd_baseline confirmed (primary finding)
- [ ] Fisher-EMA theorem proof sketch written in LaTeX; numerically verified
- [ ] Paper title no longer contains "thermodynamic"; `tar_author.py` prompts updated
- [ ] HPC lambda-vs-momentum ablation complete (if HPC replication succeeded)
- [ ] D_PR–forgetting correlation tested; result recorded
- [ ] ASC novelty assessed; TCL-for-LLMs architecture scoped for Paper 2
- [ ] Second-order importance prototype complete with ablation result

**What Phase 3 unlocks:** Paper 1 theory section can be drafted. Paper 2 planning begins.

---

## PHASE 4 — CODE QUALITY & ALGORITHM CORRECTNESS
**Duration:** 3–4 weeks (overlaps with Phases 2–3, begins after Phase 0)
**Track:** B (Code & Engineering)
**Owner:** Algorithm Engineer / Systems Engineer
**Objective:** Bring the algorithm codebase to the standard where it can be shared publicly as supplementary material to a peer-reviewed paper. Fix all known correctness issues.

---

### Task 4.1 — Fix the CI formula in all four locations (mirror of Task 1.1)

This is the engineering implementation of the statistical fix from Task 1.1. Confirm all four files use the correct t-distribution formula and are consistent with each other. Write a shared utility function in `tar_lab/benchmark_stats.py` that all other files import:

```python
# tar_lab/benchmark_stats.py
def compute_ci95(values: list[float]) -> tuple[float, float, str]:
    """Returns (ci_low, ci_high, method) using t-distribution for n<30."""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    df = n - 1
    t_crit = float(t_dist.ppf(0.975, df)) if df > 0 else 1.96
    half = t_crit * (std / np.sqrt(n))
    return (mean - half, mean + half, "t_distribution")
```

Replace all four duplicate implementations with imports of this function.

**Verification:** Unit test: `compute_ci95([0.1, 0.2, 0.3])` returns interval wider than the z-critical version. All four files import from the same source.

---

### Task 4.2 — Fix TCL device placement edge cases

`ThermalMemory.penalty()` returns `torch.zeros(1, requires_grad=False)` without specifying a device. If the model is on CUDA, this creates a CPU tensor; the subsequent addition triggers an implicit CPU→GPU copy or a type error.

**Fix in `tcl.py`:**
```python
def penalty(self, model: nn.Module, device: Optional[torch.device] = None) -> torch.Tensor:
    if not self._tasks:
        # FIX: specify device explicitly
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        return torch.zeros(1, device=device, requires_grad=False)
    # ... rest of method
```

Same fix in `ThermalCheckpoint.effective_importance()`:
```python
def effective_importance(self, name: str) -> torch.Tensor:
    base = self.importance.get(name)
    if base is None:
        return torch.zeros(1)  # FIX: add device=base.device or keep CPU (importance is stored on CPU)
```

**Add unit test in `tests/test_tcl.py`:**
```python
def test_penalty_device_match():
    """Penalty tensor must be on the same device as model parameters."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda:0")
    model = _make_model().to(device)
    memory = ThermalMemory()
    # ... train task 0
    penalty = memory.penalty(model, device=device)
    assert penalty.device.type == "cuda", f"Got {penalty.device}"
```

**Verification:** Test passes on systems with CUDA. On CPU-only systems, test is skipped gracefully.

---

### Task 4.3 — Fix the method synthesizer baseline validation gap

`tar_lab/method_synthesizer.py` validates synthesised code only by running the sandbox without crashing. A method that always returns zero penalty passes sandbox validation but produces invalid scientific results.

**Add mandatory minibench validation after sandbox success:**
```python
def _validate_synthesised_method(method_class_code: str, method_name: str) -> tuple[bool, str]:
    """Run synthesised method on tiny benchmark; verify output sanity."""
    # Inject method into a temporary module
    # Run: split_cifar10, tiny_cnn, 3 tasks, seed=42, 2 epochs
    # Check: 0.0 <= mean_forgetting <= 1.0, accuracy > 0.15 (above absolute floor)
    # Check: no NaN or Inf in any metric
    # Check: method produces a non-zero regularization loss at least once
    try:
        result = _run_minibench(method_class_code)
        if result["mean_forgetting"] < 0 or result["mean_forgetting"] > 1:
            return False, f"Forgetting out of range: {result['mean_forgetting']}"
        if result["mean_accuracy"] < 0.15:
            return False, f"Accuracy below floor (likely collapse): {result['mean_accuracy']}"
        if result.get("max_reg_loss", 0) == 0:
            return False, "Method never produced non-zero regularization loss (likely broken)"
        if any(math.isnan(v) or math.isinf(v) for v in result.values() if isinstance(v, float)):
            return False, "NaN or Inf detected in metrics"
        return True, "Minibench passed"
    except Exception as e:
        return False, f"Minibench crashed: {e}"
```

**Verification:** Inject a known-broken method (always returns `torch.tensor(0.0)` for penalty); confirm it fails with "Method never produced non-zero regularization loss."

---

### Task 4.4 — Fix the generative director's leading prompt

The current prompt frames standard methods as exhausted failures and demands a novel proposal. This biases toward novelty without diagnostic reasoning.

**Replace the proposal prompt with a two-step structure:**

```python
_DIAGNOSIS_PROMPT = """
You are diagnosing why a continual learning experiment has failed.
Given the failure pattern below, classify the most likely root cause:
(a) hyperparameter mis-specification — the algorithm is correct but parameters are wrong
(b) architecture limitation — the backbone is too small or wrong for this task
(c) dataset characteristic — the data distribution is not matched to the method's assumptions
(d) algorithmic limitation — the method itself cannot address this failure mode

Failure pattern: {failure_summary}
Governor metrics: E={energy:.4f}, σ={sigma:.4f}, ρ={rho:.4f}
History of failures: {failure_history}

Respond with a JSON object: {{"root_cause": "<a|b|c|d>", "reasoning": "<1-2 sentences>"}}
"""

_PROPOSAL_PROMPT = """
Root cause identified as: {root_cause} — {reasoning}
Given this root cause, propose ONE experiment design that directly addresses it.
If root cause is (a) or (b): return {{"action": "tune_hyperparameters", "suggestion": "..."}}
If root cause is (c) or (d): propose a specific algorithmic or dataset modification.
"""
```

**Verification:** Inject a failure summary where the root cause is obviously hyperparameter mis-specification; confirm the Director returns `{action: "tune_hyperparameters"}` rather than proposing a new method family.

---

### Task 4.5 — Fix smoke-tier synthetic benchmarks

The smoke-tier vision benchmark generates trivial geometric patterns (vertical stripe, horizontal stripe, diagonal) as "images". These are plumbing tests, not evidence.

**Changes to `tar_lab/multimodal_payloads.py`:**
- Add `is_synthetic_smoke_bench: true` flag to all result JSONs generated by this benchmark.
- Relabel the benchmark in `tar_lab/science_profiles.py` from `"canonical_ready"` to `"smoke_only"`.

**Changes to `tar_lab/validation.py`:**
- Add a check: if `result.is_synthetic_smoke_bench == true`, override trust tier to `smoke_only` regardless of other conditions.
- Exclude from any evidence collection used for publication.

**Verification:** Run smoke benchmark; confirm result JSON has `is_synthetic_smoke_bench: true` and `trust_tier: "smoke_only"`.

---

### Task 4.6 — Fix silent benchmark tier downgrade

When a canonical-tier benchmark is unavailable the system silently runs a lower tier and reports the result without disclosure.

**Changes to `tar_lab/backend_factory.py` and result schemas:**
```python
class ExperimentResult(BaseModel):
    # ... existing fields
    tier_requested: str = "canonical"
    tier_executed: str = "canonical"
    tier_downgrade_reason: Optional[str] = None

    @property
    def tier_downgraded(self) -> bool:
        return self.tier_requested != self.tier_executed
```

In the orchestrator, when a tier downgrade occurs:
```python
if tier_executed != tier_requested:
    result.tier_downgrade_reason = f"Requested {tier_requested}; downgraded to {tier_executed}: {reason}"
    alerts.log(AlertSeverity.WARNING, f"Benchmark tier downgraded: {result.tier_downgrade_reason}")
    # Flag as publication-blocking
    result.publication_allowed = False
```

**Verification:** Request a canonical-tier benchmark that is unavailable; confirm result JSON has `tier_downgraded: true` and `publication_allowed: false`.

---

### Task 4.7 — Replace keyword-based domain classifier

`tar_lab/science_profiles.py` assigns domain labels via keyword scoring ("quantum" → +5.0). This is brittle and was traced as a contributing factor to the financial literature contamination.

**Replacement approach:**

Build a training set of 50 problem statements from existing frontier problems, labeled with their correct domain. Train a logistic regression classifier:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

def build_domain_classifier(training_examples: list[tuple[str, str]]) -> Pipeline:
    """
    training_examples: [(problem_statement, domain_label), ...]
    Returns a calibrated classifier that outputs probabilities per domain.
    """
    texts, labels = zip(*training_examples)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("classifier", CalibratedClassifierCV(LogisticRegression(C=1.0))),
    ])
    clf.fit(texts, labels)
    return clf
```

The classifier outputs a calibrated probability distribution over domains. Report the top domain with its confidence. If max confidence < 0.5, flag as "ambiguous domain" and request human classification.

**Validation:** Hold out 10 labeled examples; confirm classifier achieves >80% accuracy. Save classifier as `tar_state/models/domain_classifier.pkl`.

**Verification:** Re-classify the financial portfolio optimization gap entries; confirm they receive domain "finance_economics" with high confidence and are filtered by the domain allowlist from Task 0.6.

---

### Task 4.8 — Wire ActiveLearner to orchestrator startup

`knowledge_graph.json` has `"entries": []` because `LiteratureBrain.start()` is never called from the main operational loop.

**Fix in `tar_lab/orchestrator.py`** (or the equivalent startup path):
```python
# In TAROrchestrator.__init__ or a startup hook:
from literature import LiteratureBrain

def _start_literature_brain(self) -> None:
    db_path = self.store.state_dir / "literature" / "literature_graph.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    self._literature_brain = LiteratureBrain(db_path=str(db_path))
    self._literature_brain.start()  # Launches background daemon thread
    self._log("Literature brain started; active learner running.")

# Call from startup:
self._start_literature_brain()
```

After wiring: verify that within one active learner cycle (≤4 hours), new papers are ingested and the knowledge graph begins populating.

**Verification:** `knowledge_graph.json` has at least 10 entries after 4 hours of operation. `brain.corpus_summary()` returns a non-zero paper count.

---

### Task 4.9 — Initialize the self-improvement anchor pack

`tar_state/self_improvement/` is empty. `SelfImprovementEngine.initialize_anchor_pack()` has never been called. The improvement gate has no baseline reference.

```python
from tar_lab.self_improvement import SelfImprovementEngine

engine = SelfImprovementEngine(workspace_root=workspace)
# Use current eval results as the baseline
engine.initialize_anchor_pack(
    pack_path="tar_state/eval_packs/baseline_eval_items.jsonl",
    run_manifest_path="tar_state/manifests/baseline_eval_run.json",
    baseline_mean_score=0.82,       # from current best operator eval
    baseline_overclaim_rate=0.0,    # no overclaims in baseline
)
```

Store the anchor manifest in `tar_state/self_improvement/anchor_manifest.json`. Commit to git.

Add health check assertion (Task 6.7): if anchor manifest is absent at startup, log WARNING "Self-improvement gating is inactive — anchor pack not initialized."

**Verification:** `tar_state/self_improvement/anchor_manifest.json` exists; `engine.is_initialized()` returns True.

---

### Task 4.10 — Fix the floating-point equality bug in self-improvement gate

`self_improvement.py` line ~167: `if probe_overclaim_rate != 0.0`

This is a floating-point equality comparison on a rate derived from division. Any non-zero floating-point rounding will cause this to incorrectly flag valid adapters.

**Fix:**
```python
_FLOAT_TOLERANCE = 1e-9

# Before:
if probe_overclaim_rate != 0.0:
    return False, "overclaim_detected"

# After:
if probe_overclaim_rate > _FLOAT_TOLERANCE:
    return False, f"overclaim_detected: rate={probe_overclaim_rate:.6f}"
```

Apply the same tolerance to all floating-point comparisons in `self_improvement.py`.

**Verification:** Pass in a probe_overclaim_rate of 1e-15 (rounding noise); confirm it is accepted. Pass in 0.001; confirm it is rejected.

---

### Task 4.11 — Implement power-analysis-based sample size selection in the pre-registration gate

When an experiment is pre-registered, automatically compute the required sample size:

```python
from statsmodels.stats.power import TTestOneSamplePower
import math

def required_seeds(target_effect_size_d: float, 
                   alpha: float = 0.05, 
                   power: float = 0.80) -> int:
    analysis = TTestOneSamplePower()
    n = analysis.solve_power(
        effect_size=target_effect_size_d, 
        alpha=alpha, 
        power=power, 
        alternative='smaller'
    )
    return max(5, math.ceil(n))

# In pre-registration gate:
if pre_reg.required_seeds > MAX_AFFORDABLE_SEEDS:
    experiment.publication_grade = "exploration_grade"
    experiment.notes.append(
        f"Underpowered: requires {pre_reg.required_seeds} seeds for {power*100:.0f}% power "
        f"at d={pre_reg.min_detectable_effect_d}; budget allows {MAX_AFFORDABLE_SEEDS}. "
        f"Results cannot be cited in publication as evidence."
    )
```

**Verification:** Pre-register a hypothesis with d=0.3 (small effect): `required_seeds` returns 90+. Pre-register with d=2.0 (very large effect): returns 5. Both enter the queue correctly but with different `publication_grade` values.

---

### Task 4.12 — Fix the invariant violations in TCL

`ThermalMemory.penalty()` accumulates positive terms but the assert is missing. While in practice the penalty is always ≥ 0 (sum of non-negative values), an assertion documents the invariant for contributors.

```python
# In ThermalMemory.penalty():
total_penalty = torch.zeros(1, device=device)
# ... accumulation loop ...
assert float(total_penalty.item()) >= 0.0, \
    f"TCL penalty is negative: {float(total_penalty.item())}. " \
    f"This indicates a numerical error in importance or drift computation."
return total_penalty
```

**Add unit test in `tests/test_tcl.py`:**
```python
@pytest.mark.parametrize("importance_val,drift_val", [
    (0.0, 1.0),   # zero importance
    (1.0, 0.0),   # zero drift
    (0.0, 0.0),   # both zero
    (1.0, 1.0),   # both positive
])
def test_penalty_always_nonnegative(importance_val, drift_val):
    model = _make_model()
    memory = ThermalMemory()
    # Manually inject a checkpoint with controlled importance and weights
    # ... setup ...
    penalty = memory.penalty(model)
    assert float(penalty.item()) >= 0.0
```

**Verification:** All four parametrized test cases pass.

---

**Phase 4 exit gate — ALL must be true:**
- [ ] Single shared `compute_ci95()` function; all four files import it
- [ ] TCL device placement fixed; CUDA unit test passes (or skipped on CPU-only)
- [ ] Method synthesizer runs minibench after sandbox; broken methods rejected
- [ ] Generative director has two-step diagnosis/proposal prompt; rejection path working
- [ ] Smoke benchmarks flagged as `trust_tier: "smoke_only"` 
- [ ] Benchmark tier downgrade disclosed in result JSON; publication blocked on downgrade
- [ ] Domain classifier replaces keyword scoring; financial entries correctly classified
- [ ] `LiteratureBrain.start()` called at orchestrator startup; knowledge graph populating
- [ ] Self-improvement anchor pack initialized; `anchor_manifest.json` committed to git
- [ ] Float equality bug fixed with `_FLOAT_TOLERANCE`
- [ ] Power-analysis-based sample size in pre-registration gate; exploration_grade label applied
- [ ] TCL penalty invariant asserted; all parametrized tests passing

**What Phase 4 unlocks:** Phase 8 (research automation enhancements) can begin.

---

*Stage 3 complete — Phases 3 and 4 with 19 tasks (3.1–3.7, 4.1–4.12)*
