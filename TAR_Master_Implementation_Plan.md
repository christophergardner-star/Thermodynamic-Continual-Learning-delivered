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

---

## PHASE 5 — PAPER PIPELINE
**Duration:** 8–12 weeks (begins after Phase 0 + Phase 1; parallel with Phases 2–3)
**Track:** A (Science)
**Owner:** Lead Researcher / Paper Author
**Objective:** Produce one real, defensible, submission-ready manuscript by September 1, 2026 (ContinualAI@NeurIPS 2026 target). Nothing templated. Every claim traceable to `honest_evidence_inventory.json`.

---

### Task 5.1 — Commit to the correct framing before drafting begins

**Priority:** CRITICAL — the wrong frame wastes weeks of writing

**The framing decision (from expert consensus):**

Do NOT frame as: "A new thermodynamic theory of catastrophic forgetting."
DO frame as: "An empirical audit of EWC's failure mode with a controlled mechanistic replication."

**The one-sentence paper identity:**
> "We show that EWC's canonical hyperparameters systematically underfit activation heterogeneity in ResNet-18 architectures; continuous gradient-energy importance accumulation corrects this underfitting and reduces forgetting, with the effect attributable entirely to the importance weighting rather than thermodynamic regime detection."

This framing is:
- **Defensible:** the Phase 11 ablation data supports it
- **Novel:** no paper has isolated EWC's hyperparameter sensitivity under controlled seed management
- **Honest:** it explicitly states the governor does not contribute
- **Citable:** EWC is the most-cited CL paper; an audit of its failure mode has broad relevance

Write this framing decision into `tar_state/papers/paper_1_framing_decision.json` before any section is drafted.

**Verification:** File exists; Lead Researcher has reviewed and approved the framing.

---

### Task 5.2 — Write the complete paper structure with section-by-section evidence mapping

**Priority:** HIGH — do this before any section is drafted

Map every claim in every section to its evidence source in `honest_evidence_inventory.json`:

**Section 1 — Abstract (150 words):** Problem (EWC underperforms), approach (continuous EMA importance), result (Δ forgetting at corrected p-value), implication (importance weighting drives the improvement, not regime detection).

**Section 2 — Introduction:** Catastrophic forgetting as a deployment problem → Current solutions (EWC, SI, replay, prompt-based) → The gap: EWC's snapshot Fisher underestimates gradient energy heterogeneity → Proposal: accumulate continuously → Contributions: (1) controlled EWC replication, (2) continuous importance estimator, (3) mechanism ablation.

**Section 3 — Related Work:** [Requires knowledge graph from Phase 9.2] 15 essential papers mapped to 3 themes: (a) regularization-based CL, (b) Fisher information estimation, (c) evaluation methodology for CL.

**Section 4 — Method:** ThermalImportance algorithm (Algorithm Box). ThermalMemory ring buffer. TCLRegularizer penalty. Fisher-EMA theorem (Task 3.2). Connection to EWC as limiting case.

**Section 5 — Experiments:**
- 5.1: Main comparison table (Phase 10 rerun with all baselines)
- 5.2: Scale generalisation (Phase 16 CIFAR-100, Phase 17 TinyImageNet, 5 seeds each)
- 5.3: Mechanistic ablation table (7 conditions from Task 3.1)
- 5.4: HPC characterisation (Task 3.4) — if HPC replication succeeded

**Section 6 — Analysis:** D_PR–forgetting correlation (Task 3.5). Power analysis table (Task 1.3). BWT/FWT across methods (Task 1.8).

**Section 7 — Limitations:** Governor non-activation. Task-incremental only. Single backbone (ResNet-18). DER++ comparison (honest result). Bonferroni status of headline result.

**Section 8 — Conclusion:** One paragraph. What was shown. What was not shown. What is next.

Write `tar_state/papers/paper_1_section_evidence_map.json` listing the inventory entry IDs supporting each claim in each section.

**Verification:** File exists; no section cites a result that is not in `honest_evidence_inventory.json`.

---

### Task 5.3 — Draft human-written sections (abstract, introduction, limitations, conclusion)

**Priority:** HIGH — `tar_author.py` is a drafting aid, not a final author

These four sections require human voice and strategic framing. `tar_author.py` drafts should be treated as raw material to be substantially edited.

**Process:**
1. Run `tar_author.py` section renderer for each section with the updated non-thermodynamic prompts (Task 3.3).
2. Human edits each draft for: accurate framing, appropriate hedging, correct claim scope, natural academic voice.
3. All numerical claims verified against `honest_evidence_inventory.json` before accepting.
4. All citation keys verified against the bibliography before accepting.

**Time allocation:**
- Abstract: 2 hours (draft + 2 rounds of editing)
- Introduction: 6 hours
- Limitations: 3 hours
- Conclusion: 2 hours

**Verification:** Each section compiles without LaTeX errors; all numerical values match the evidence inventory; no thermodynamic framing language appears in final text.

---

### Task 5.4 — Build the comparison table and related work section from the knowledge graph

**Priority:** HIGH — requires Phase 9 knowledge graph to be populated first

**Standard NeurIPS CL comparison table format:**

| Method | Split-CIFAR-10 Forgetting ↓ | Split-CIFAR-100 Forgetting ↓ | Split-TinyImageNet Forgetting ↓ | BWT ↑ | Source |
|---|---|---|---|---|---|
| SGD baseline | — | — | — | — | Our Phase 10 |
| EWC (λ=1000) | — | — | — | — | Our Phase 10 rerun |
| SI (c=0.01) | — | — | — | — | Our Phase 13 |
| LwF (α=0.5) | — | — | — | — | Our Phase 10 rerun |
| A-GEM (m=200) | — | — | — | — | Our Phase 10 rerun |
| DER++ (m=200) | — | — | — | — | Our Phase 10 rerun |
| TCL (ours) | — | — | — | — | Our Phase 10 rerun |

Fill this table from Phase 2 results. All ±values use the t-distribution CI formula from Task 1.1. All significance values are Bonferroni-corrected (Task 1.2). Bayesian posterior probabilities (Task 1.10) are reported in parentheses.

**Related work section:** Uses the knowledge graph (Phase 9) to identify: (a) the 3 most similar recent papers with explicit comparison, (b) the 5 foundational references (EWC, SI, LwF, DER++, GEM), (c) the 4 theoretical references (Fisher information, natural gradient, K-FAC, NTK).

**Verification:** Table is complete with all methods; all values have CI95; related work section cites all 15 required papers from Phase 9.3.

---

### Task 5.5 — Compile, review, and submit

**Priority:** CRITICAL

**Compilation checklist:**
- [ ] `pdflatex main.tex` succeeds without errors
- [ ] PDF is 8 pages (ContinualAI workshop limit) or 9 pages + references (ICLR limit)
- [ ] All figures have captions; all tables have captions
- [ ] All numerical values in text match comparison JSONs
- [ ] All citations have corresponding BibTeX entries
- [ ] Supplementary material: (a) power analysis table, (b) BWT/FWT for all methods, (c) per-task ECE trajectories, (d) pre-registration documents, (e) code availability statement

**External review:** Before submission, share the draft with at least one external person from the ContinualAI community (Discord/Slack) for feedback. Allow at least 1 week for feedback incorporation.

**arXiv:** Post to arXiv simultaneously with submission, only after Phase 2 HPC replication result is confirmed.

**Submission target:** ContinualAI@NeurIPS 2026 (estimated deadline: September 1, 2026). ICLR 2027 as stretch target (abstract deadline ~October 2026).

**Verification:** Submission confirmation received. arXiv ID exists. Pre-registration documents are publicly linked from the paper.

---

### Task 5.6 — Plan and initiate the multi-paper strategy

**Priority:** MEDIUM — concurrent with Phase 5 main work

| Paper | Target venue | Key result needed | Current status |
|---|---|---|---|
| **Paper 1** | ContinualAI@NeurIPS 2026 | Phase 2 HPC replication or Phase 10 rerun | In preparation |
| **Null-result brief** | NeurIPS brief or workshop | Phase 11 governor ablation | Data complete; write 4 pages |
| **Paper 2: TCL-for-LLMs** | NeurIPS 2027 or ICML 2027 | TCL-LLM prototype (Phase 10.1) | Scoping done (Task 3.6) |
| **Paper 3: TAR system** | AutoML@ICML 2027 | Paper 1 first; TAR audit trail | Requires Paper 1 |

Begin drafting the null-result brief (governor ablation) in parallel with Paper 1. It is 4 pages and the data (Phase 11) is already available. This brief demonstrates rigorous negative result reporting and builds credibility.

**Verification:** `tar_state/papers/paper_1_null_result_brief.tex` exists with at least an introduction and results table.

---

**Phase 5 exit gate — ALL must be true:**
- [ ] Framing decision committed in writing before any drafting
- [ ] Evidence map JSON exists mapping each claim to inventory entry
- [ ] All four human-written sections drafted and reviewed
- [ ] Comparison table complete; all results from Phase 2
- [ ] Knowledge graph used for related work section
- [ ] LaTeX compiles without errors; PDF produced
- [ ] At least one external reviewer has provided feedback
- [ ] arXiv ID exists or submission confirmation received
- [ ] Null-result brief (governor) in draft form

---

## PHASE 6 — GOVERNANCE, SAFETY & LONG-TERM AUTONOMY
**Duration:** 4–6 weeks (overlaps with Phases 3–5)
**Track:** C (Automation & Governance)
**Owner:** Lead Researcher / Systems Engineer
**Objective:** Bring the autonomous research infrastructure to the standard where it can safely run unattended without producing false science, gaming metrics, or modifying its own governance.

---

### Task 6.1 — Cryptographically seal governance state files

**Priority:** HIGH — governance files can currently be modified by any process with filesystem access

**Implementation:**

Generate a persistent Ed25519 keypair at system setup:
```bash
python -c "
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
key = ed25519.Ed25519PrivateKey.generate()
# Write private key to protected location (NOT in repo, NOT in tar_state/)
with open('C:/Users/cgard/.tar_keys/governance_private.pem', 'wb') as f:
    f.write(key.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
# Write public key to repo (safe to commit)
with open('tar_governance_public.pem', 'wb') as f:
    f.write(key.public_key().public_bytes(
            serialization.Encoding.PEM, 
            serialization.PublicFormat.SubjectPublicKeyInfo))
"
```

When any governance file is written, append a signature block:
```python
def sign_governance_file(path: Path, private_key_path: Path) -> None:
    content = path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()
    # Sign the hash
    sig = private_key.sign(content_hash.encode())
    sig_path = path.with_suffix('.sig')
    sig_path.write_bytes(sig)
```

When watchdog or manifest gate reads any governance file, verify signature before trusting:
```python
def verify_governance_file(path: Path, public_key_path: Path) -> bool:
    content = path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()
    sig = path.with_suffix('.sig').read_bytes()
    try:
        public_key.verify(sig, content_hash.encode())
        return True
    except Exception:
        return False  # Fail-closed: treat as tampered
```

Files requiring signing: `stabilisation_mode.json`, `execution_policy.json`, `runtime_policy.json`, manifest files.

**Verification:** Manually modify `stabilisation_mode.json` without re-signing; confirm watchdog refuses to load it and raises CRITICAL alert.

---

### Task 6.2 — Enforce execution_policy.json at code generation time

`tar_state/policies/execution_policy.json` declares `"require_sandbox_for_generated_code": true` but the `ExecutionPolicyViolation` error class exists in `tar_lab/errors.py` with zero enforcement calls.

**Add enforcement in `tar_lab/method_synthesizer.py`:**
```python
def _assert_policy_allows_execution(workspace: Path) -> None:
    policy_path = workspace / "tar_state" / "policies" / "execution_policy.json"
    if not policy_path.exists():
        raise ExecutionPolicyViolation("execution_policy.json not found; cannot execute generated code")
    if not verify_governance_file(policy_path, PUBLIC_KEY_PATH):
        raise ExecutionPolicyViolation("execution_policy.json signature invalid; refusing to execute")
    policy = json.loads(policy_path.read_text())
    if policy.get("require_sandbox_for_generated_code", True):
        # Verify Docker is available
        if not _docker_available():
            raise ExecutionPolicyViolation(
                "execution_policy requires sandbox but Docker is unavailable. "
                "Cannot execute generated code without sandbox.")
```

Call `_assert_policy_allows_execution(workspace)` before any synthesised method code is executed.

**Verification:** Remove Docker; attempt to synthesise a method; confirm `ExecutionPolicyViolation` is raised with a clear message.

---

### Task 6.3 — Make Docker non-bypassable for experiment execution

The `allow_host_fallback` path in `docker_runner.py` allows uncontrolled code to run on the host when Docker is unavailable.

**Remove the fallback:**
```python
# In docker_runner.py, _execute_in_process():
# BEFORE (unsafe):
if not self._docker_available():
    return self._execute_subprocess_fallback(command, env, output_dir)

# AFTER (safe):
if not self._docker_available():
    raise ExecutionPolicyViolation(
        "Docker required for experiment execution. "
        "Subprocess fallback is disabled by execution_policy.json. "
        "Install Docker or run experiments with run_intent='control'.")
```

Allow host execution only for: (a) daemon startup scripts, (b) health checks, (c) experiments explicitly marked `run_intent='control'` (plumbing tests).

**Verification:** Stop Docker service; attempt to launch a research-grade experiment; confirm `ExecutionPolicyViolation` is raised, no subprocess falls back.

---

### Task 6.4 — Build a tamper-proof append-only audit log

The current alert ring buffer (500 entries) loses history and is not tamper-evident.

**Implementation:**

Create `tar_state/audit.jsonl` as a permanent append-only log (never trimmed, gzip-rotated monthly):

```python
def audit_log(event_type: str, details: dict, workspace: Path) -> None:
    """Append a tamper-evident audit event."""
    audit_path = workspace / "tar_state" / "audit.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "event_type": event_type,
        "details": details,
    }
    # Include hash of previous entry for chain integrity
    if audit_path.exists():
        last_line = audit_path.read_text(encoding="utf-8").strip().split("\n")[-1]
        entry["prev_hash"] = hashlib.sha256(last_line.encode()).hexdigest()[:16]
    else:
        entry["prev_hash"] = "genesis"
    line = json.dumps(entry)
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
```

**Events to audit-log:**
- Every manifest authorization
- Every experiment start and stop
- Every governance file write
- Every verdict generation
- Every claim promotion to breakthrough
- Every self-improvement cycle gate decision

**Monthly rotation:** A scheduled task compresses `audit.jsonl` to `audit_YYYY-MM.jsonl.gz` at month start.

**Verification:** Run 10 experiment events; confirm `audit.jsonl` has 10 entries; manually modify an entry; run `verify_audit_chain()` which detects the break.

---

### Task 6.5 — Close the autonomous mode human-review bypass

`human_review.py` has `_UniversalApproval` — a sentinel that returns True for any experiment in autonomous mode. This makes the human review layer non-functional in the mode where it matters most.

**Replace with a genuine 24-hour veto window:**

```python
AUTO_APPROVE_HOURS = 24  # Configurable

def approved_experiment_ids(workspace: Path) -> set[str]:
    vs = load_validation_mode(workspace)
    if vs.get("active"):
        # Stabilisation mode: explicit approval only
        return _load_explicit_approvals(workspace)
    
    # Autonomous mode: 24-hour veto window
    state = load_human_review_state(workspace)
    approved = set()
    now = datetime.now(timezone.utc)
    for proposal in state.get("proposals", []):
        if proposal.get("status") == "approved":
            approved.add(proposal["experiment_id"])
            continue
        # Auto-approve if veto window has elapsed
        submitted_at = datetime.fromisoformat(proposal.get("created_at", ""))
        hours_elapsed = (now - submitted_at).total_seconds() / 3600
        if hours_elapsed >= AUTO_APPROVE_HOURS and proposal.get("status") == "pending_director_review":
            proposal["status"] = "auto_approved"
            proposal["auto_approved_at"] = now.isoformat()
            approved.add(proposal["experiment_id"])
    save_human_review_state(workspace, state)
    return approved
```

**Operator notification:** When an experiment enters the 24-hour window, log: "Experiment X will auto-approve in Y hours. View at dashboard to veto."

**Verification:** Submit an experiment; confirm it is NOT in the approved set immediately. Wait 24 hours (or mock the timestamp); confirm it then auto-approves. Manually veto an experiment; confirm it never auto-approves.

---

### Task 6.6 — Implement per-session API cost budget with hard stop

`tar_author.py` calls Sonnet-4-6 with `max_tokens=4096` for each section, with no cost tracking or budget gate.

**Add cost tracking to `model_router.py`:**
```python
SESSION_BUDGET_USD = 20.0  # Configurable per session
HAIKU_FALLBACK_THRESHOLD = 0.8  # Fall back to Haiku at 80% of budget

class SessionCostTracker:
    def __init__(self, budget_usd: float = SESSION_BUDGET_USD):
        self._budget = budget_usd
        self._spent = 0.0
        self._lock = threading.Lock()
    
    def record_call(self, model: str, tokens_in: int, tokens_out: int) -> float:
        costs = {
            "claude-sonnet-4-6": (0.003, 0.015),   # $/1k tokens in/out
            "claude-haiku-4-5-20251001": (0.0008, 0.004),
        }
        rate = costs.get(model, (0.003, 0.015))
        cost = (tokens_in * rate[0] + tokens_out * rate[1]) / 1000
        with self._lock:
            self._spent += cost
            if self._spent >= self._budget * 2:
                raise BudgetExceededError(f"Session cost ${self._spent:.2f} exceeded 2× budget ${self._budget:.2f}. All LLM calls blocked.")
            if self._spent >= self._budget * HAIKU_FALLBACK_THRESHOLD:
                return cost  # Signal to caller to use Haiku
        return cost
```

**Per-section cost attribution in `tar_author.py`:** Log the cost of each LLM call per section to `tar_state/logs/authoring_costs.jsonl`.

**Verification:** Set `SESSION_BUDGET_USD = 0.01`; make one API call; confirm Haiku fallback triggers. Set to 0.001; confirm `BudgetExceededError` is raised and logged.

---

### Task 6.7 — Establish the recurring integrity check routine

Build `tar_health_check.py` as a comprehensive integrity suite. Run: (a) at every startup before execution, (b) daily at a fixed time, (c) on demand via `tar_cli.py --health-check`.

**Checks to implement:**

```
1. SECURITY
   - api_secrets.json does not exist in the repo
   - publish_config.json does not exist in the repo
   - .gitignore includes all secrets patterns

2. GOVERNANCE
   - All governance files committed to git
   - All governance file signatures valid (Ed25519)
   - execution_enabled.flag state matches active_session.json

3. EVIDENCE INTEGRITY
   - All canonical comparison results pass three-gate registry check
   - honest_evidence_inventory.json exists and matches raw comparison JSONs
   - No result has bonferroni_significant=true at a corrected p-value that fails the threshold

4. SYSTEM STATE
   - active_session.json reflects actual daemon state
   - No orphan experiment PIDs in process_registry.json
   - No stale leases (>24 hours old) in runtime_ledger.json
   - GPU temperature within safe range (<85°C)
   - API key reachable (ping with 10 tokens)

5. STORAGE
   - No JSONL file exceeds 100MB (warn) or 500MB (alert)
   - ChromaDB accessible and non-corrupt
   - Audit log chain integrity check passes

6. DEPENDENCIES
   - requirements.lock exists and is not stale (>30 days)
   - Docker available and responsive
   - ActiveLearner thread alive (knowledge graph populated in last 48h)
```

Output: `tar_state/health_report.json` with pass/fail for each check and recommended actions.

**Verification:** Introduce a deliberate failure (delete `audit.jsonl`); run health check; confirm FAIL is recorded in the report.

---

### Task 6.8 — Implement the Prospective Frontier Seal

**Priority:** HIGH — prevents gradual scope narrowing to easy problems

Before any new research programme begins, create a signed manifest of approved frontier problems. No Director proposal can target a frontier outside the sealed set without a new human-approved manifest.

```python
class FrontierSeal(BaseModel):
    seal_id: str
    created_at: str
    sealed_by: str               # git user.name
    approved_frontier_ids: list[str]
    signature: str               # Ed25519 signature of content
    seal_rationale: str

def create_frontier_seal(frontier_ids: list[str], workspace: Path) -> FrontierSeal:
    seal = FrontierSeal(
        seal_id=f"seal-{utc_stamp()}",
        created_at=utc_now_iso(),
        sealed_by=git_user_name(),
        approved_frontier_ids=frontier_ids,
        seal_rationale="Approved frontier set for current research programme",
        signature="UNSIGNED",
    )
    # Sign and commit
    seal.signature = sign_content(seal.model_dump_json())
    path = workspace / "tar_state" / "frontier_seal.json"
    path.write_text(seal.model_dump_json(indent=2))
    git_add_and_commit(path, f"seal: lock frontier set {seal.seal_id}")
    return seal
```

In the Director proposal generation: if a proposed experiment's `frontier_problem_id` is not in `frontier_seal.approved_frontier_ids`, reject it with a clear message: "Frontier X is not in the current sealed frontier set. Update the frontier seal to include it."

**Verification:** Seal a set of 3 frontiers; attempt to propose an experiment on a fourth frontier; confirm rejection.

---

### Task 6.9 — Implement the Goodhart Canary

**Priority:** HIGH — detects when the Director optimizes the rubric instead of science

Every N weeks (configurable; default 4 weeks), partition linked experiments for each paper into:
- **Publication set** (80%): used for Director training and proposal generation
- **Holdout set** (20%): withheld; used only for periodic out-of-sample validation

```python
def run_goodhart_canary(workspace: Path, paper_id: str) -> dict:
    """Run quarterly canary check: does Director perform on holdout?"""
    pub_results = load_publication_set(workspace, paper_id)
    holdout_results = load_holdout_set(workspace, paper_id)
    
    pub_score = mean([r["evidence_score"] for r in pub_results])
    holdout_score = mean([r["evidence_score"] for r in holdout_results])
    
    gap = pub_score - holdout_score
    if gap > 0.15:  # 15% gap threshold
        audit_log("GOODHART_CANARY_TRIGGERED", {
            "paper_id": paper_id,
            "pub_score": pub_score,
            "holdout_score": holdout_score,
            "gap": gap,
        }, workspace)
        raise GoodhartAlert(
            f"Director may be goodharting the publication rubric. "
            f"Publication set score ({pub_score:.2f}) is {gap:.2f} above holdout "
            f"({holdout_score:.2f}). Human audit required before next training cycle.")
    return {"gap": gap, "status": "ok" if gap < 0.15 else "alert"}
```

**Verification:** Inject synthetic results where the Director consistently produces high-scoring pub-set results but low holdout scores; confirm `GoodhartAlert` is raised and logged.

---

### Task 6.10 — Implement the Immutable Director Snapshot

**Priority:** HIGH — prevents the system from modifying its own governance during live operation

When stabilisation mode is activated, take a signed snapshot of the Director weights and all governance code:

```python
def activate_stabilisation_with_snapshot(workspace: Path) -> SnapshotRecord:
    """Call when entering stabilisation mode."""
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root())
    ).decode().strip()
    
    snapshot = SnapshotRecord(
        snapshot_id=f"snapshot-{utc_stamp()}",
        git_hash=git_hash,
        director_state_hash=sha256_file(
            workspace / "tar_state" / "research_director_state.json"
        ),
        governance_files_hash={
            f: sha256_file(repo_root() / f)
            for f in GOVERNANCE_FILES
        },
        activated_at=utc_now_iso(),
        signature="UNSIGNED",
    )
    snapshot.signature = sign_content(snapshot.model_dump_json())
    
    snap_path = workspace / "tar_state" / "director_snapshot.json"
    snap_path.write_text(snapshot.model_dump_json(indent=2))
    git_add_and_commit(snap_path, f"snapshot: Director locked at {git_hash[:8]}")
    return snapshot
```

During stabilisation: before any self-improvement cycle begins, verify that no governance file has changed from the snapshot. If any hash differs, raise `StabilisationGateMissingOverrideError` and require human review of the diff.

**Verification:** Activate stabilisation; modify `tar_research_director.py`; attempt to start a self-improvement cycle; confirm the snapshot check fails and the cycle is blocked.

---

### Task 6.11 — Fix self-improvement pipeline: add holdout test set for gate evaluation

The self-improvement gate currently evaluates on the anchor pack (same data used to define the baseline). This creates an evaluation/training set overlap.

Add a fixed holdout set (`tar_state/self_improvement/holdout_eval_items.jsonl`) that:
- Is never exposed to the Director during training
- Is used exclusively for gate evaluation
- Contains 20% of the baseline eval items, stratified by task family

```python
def evaluate_adapter_on_holdout(adapter_path: Path, workspace: Path) -> dict:
    """Evaluate fine-tuned adapter on held-out items before deployment."""
    holdout = load_jsonl(workspace / "tar_state/self_improvement/holdout_eval_items.jsonl")
    # Run adapter inference on holdout
    scores = run_eval(adapter_path, holdout)
    return {
        "holdout_mean_score": mean(scores),
        "holdout_overclaim_rate": sum(1 for s in scores if s.get("overclaim")) / len(scores),
        "n_items": len(scores),
    }
```

Gate: adapter is only deployed if holdout_mean_score > anchor_mean_score AND holdout_overclaim_rate ≤ anchor_overclaim_rate.

**Verification:** Train an adapter that overfits the training signal; confirm it fails the holdout gate even if it passes the anchor gate.

---

### Task 6.12 — Fix self-improvement mode collapse risk

Diversity scoring in self-improvement is across signal `kinds`, not signal `content`. A thousand semantically identical signals that happen to have different kind labels pass the diversity filter.

**Fix:** Add content-level deduplication using sentence embeddings:

```python
def _curate_signals_with_dedup(signals: list[SignalRecord], 
                                min_cosine_distance: float = 0.3) -> list[SignalRecord]:
    """Filter out near-duplicate signals regardless of kind."""
    if not signals:
        return []
    embeddings = embed_signals([s.content for s in signals])
    kept = [signals[0]]
    kept_embeddings = [embeddings[0]]
    for i, (sig, emb) in enumerate(zip(signals[1:], embeddings[1:]), 1):
        # Check cosine distance from all kept signals
        distances = [cosine_distance(emb, k_emb) for k_emb in kept_embeddings]
        if min(distances) >= min_cosine_distance:
            kept.append(sig)
            kept_embeddings.append(emb)
    return kept
```

**Verification:** Inject 20 semantically identical signals with different kind labels; confirm dedup reduces to 1–2 unique signals.

---

### Task 6.13 — Add self-improvement rollback mechanism

If a deployed adapter causes regressions in subsequent experiment evaluations (Director proposes worse experiments, more false positives), there is no way to revert.

**Add a monitoring period and rollback trigger:**

```python
class AdapterMonitor:
    MONITORING_CYCLES = 5  # Evaluate for 5 experiment cycles
    REGRESSION_THRESHOLD = 0.05  # 5% score drop triggers rollback
    
    def check_and_rollback_if_needed(self, workspace: Path) -> bool:
        if not self._monitoring_complete():
            return False  # Still in monitoring period
        current_score = self._compute_recent_score()
        baseline_score = self._load_anchor_score()
        if current_score < baseline_score - self.REGRESSION_THRESHOLD:
            self._rollback_to_base_model(workspace)
            audit_log("ADAPTER_ROLLBACK", {
                "reason": f"Score dropped {baseline_score - current_score:.3f} below baseline",
                "cycles_monitored": self.MONITORING_CYCLES,
            }, workspace)
            return True
        return False
```

**Verification:** Deploy an adapter that intentionally produces worse outputs; run 5 experiment cycles; confirm rollback triggers and previous model is restored.

---

**Phase 6 exit gate — ALL must be true:**
- [ ] Ed25519 keypair generated; governance files signed; tampering detected on manual edit
- [ ] `ExecutionPolicyViolation` raised when Docker unavailable for research experiments
- [ ] Docker fallback path removed from research-grade experiment execution
- [ ] `audit.jsonl` populated; chain integrity check passes
- [ ] 24-hour auto-approve veto window replaces `_UniversalApproval`
- [ ] Session API cost budget tracked; Haiku fallback at 80%; hard stop at 200%
- [ ] Health check routine runs at startup and daily; `health_report.json` produced
- [ ] Frontier Seal implemented and tested
- [ ] Goodhart Canary implemented and tested
- [ ] Immutable Director Snapshot created on stabilisation activation
- [ ] Self-improvement holdout set in place
- [ ] Content-level deduplication in self-improvement signal curation
- [ ] Adapter rollback mechanism operational

**What Phase 6 unlocks:** Full autonomous operation can resume with confidence. Phase 8 and 10 can proceed with reduced safety risk.

---

---

## PHASE 7 — ENGINEERING INFRASTRUCTURE
**Duration:** 4–6 weeks (overlaps with Phases 2–5, begins after Phase 0)
**Track:** B (Code & Engineering)
**Owner:** Systems Engineer
**Objective:** Bring operational infrastructure from development-grade to a standard that supports reliable, reproducible, long-term autonomous operation.

---

### Task 7.1 — Add GitHub Actions CI/CD pipeline

**Priority:** CRITICAL — 482 tests provide zero protection without an automated runner

No CI means broken code can be pushed and tests only run when someone remembers to run them manually. The six uncommitted governance files ran for months with no test enforcement.

**Create `.github/workflows/ci.yml`:**
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run detect-secrets
        run: detect-secrets scan --baseline .secrets.baseline
      - name: Run tests
        run: pytest --tb=short -q --timeout=120
      - name: Coverage report
        run: pytest --cov=tar_lab --cov-report=term-missing --cov-fail-under=65
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install flake8 mypy
      - run: flake8 tar_lab/ --max-line-length=120 --ignore=E501,W503
      - run: mypy tar_lab/ --ignore-missing-imports --no-strict-optional
```

**Block merges to `main` unless CI passes.** Add branch protection rules in GitHub repository settings.

**Verification:** Push a commit that breaks a test; confirm CI fails and merge is blocked. Push a commit with a secret pattern; confirm detect-secrets blocks it.

---

### Task 7.2 — Pin all dependency versions and generate a lock file

`requirements.txt` has no version pins. Every install resolves to latest, making the system non-reproducible across machines and times.

**Process:**
1. Create `requirements-core.txt` (daemon operation, no ML):
   ```
   pydantic>=2.5,<3.0
   fastapi>=0.110,<0.120
   uvicorn>=0.27,<0.30
   flask>=3.0,<4.0
   anthropic>=0.25,<0.40
   ```

2. Create `requirements-research.txt` (ML experimentation):
   ```
   torch==2.3.1
   torchvision==0.18.1
   transformers==4.41.2
   scikit-learn==1.5.0
   scipy==1.13.0
   ```

3. Create `requirements-dev.txt` (testing and development):
   ```
   pytest==8.2.0
   pytest-cov==5.0.0
   detect-secrets==1.4.0
   pre-commit==3.7.0
   mypy==1.10.0
   ```

4. Run `pip freeze > requirements.lock` after confirming all tests pass with pinned versions.
5. Commit `requirements.lock`. CI uses `pip install -r requirements.lock` for reproducibility.

**Remove unused packages:** Three PDF parsing libraries (choose primary: PyMuPDF), quantum (PennyLane — move to optional), RL (Gymnasium — move to optional).

**Verification:** On a clean virtualenv, `pip install -r requirements.lock && pytest` succeeds with all tests passing.

---

### Task 7.3 — Pin the Docker base image digest

`Dockerfile.experiment` uses `FROM python:3.13.3-slim` without a digest. Docker registries can re-tag images, making builds non-reproducible.

```dockerfile
# Get the current digest:
# docker inspect python:3.13.3-slim --format='{{index .RepoDigests 0}}'

FROM python:3.13.3-slim@sha256:<actual-digest-here>

# Add build-time health check
RUN python -c "import torch; print('PyTorch:', torch.__version__)" \
    && python -c "import numpy; print('NumPy:', numpy.__version__)"

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch" || exit 1
```

Generate and commit a Software Bill of Materials (SBOM):
```bash
docker sbom python:3.13.3-slim@sha256:<digest> --format spdx-json > docker_sbom.json
```

**Verification:** Build the image twice from the same Dockerfile; confirm both builds produce identical layer hashes.

---

### Task 7.4 — Replace Flask development server with Gunicorn

`tar_dashboard.py` runs Flask's built-in development server — single-threaded, not thread-safe, explicitly documented as "not for production."

**Changes:**
1. Add `gunicorn>=22.0,<23.0` to `requirements-core.txt`.
2. Create `tar_dashboard_wsgi.py`:
   ```python
   from tar_dashboard import app
   # gunicorn tar_dashboard_wsgi:app --workers 2 --bind 127.0.0.1:7860
   ```
3. Update `START_TAR.bat`:
   ```batch
   REM Before:
   start /B pythonw tar_dashboard.py
   REM After:
   start /B pythonw -m gunicorn tar_dashboard_wsgi:app --workers 2 --bind 127.0.0.1:7860 --log-file logs/dashboard.log
   ```
4. Add `SIGTERM` handler in `tar_dashboard.py`:
   ```python
   import signal
   def _handle_shutdown(signum, frame):
       logger.info("Dashboard shutting down gracefully...")
       sys.exit(0)
   signal.signal(signal.SIGTERM, _handle_shutdown)
   ```
5. Add `/readiness` and `/liveness` endpoints:
   ```python
   @app.route("/readiness")
   def readiness():
       return {"status": "ready", "timestamp": utc_now_iso()}, 200

   @app.route("/liveness")
   def liveness():
       return {"status": "alive"}, 200
   ```

**Verification:** Start dashboard with Gunicorn; make 10 concurrent requests to `/status`; confirm all succeed (previously single-threaded Flask would serialize them).

---

### Task 7.5 — Add fsync to all critical JSONL append operations

`state.py`'s `_safe_jsonl_append()` calls `flush()` but not `fsync()`. On Linux and Windows, `flush()` pushes data to the OS buffer but does not guarantee disk write. A power failure between `flush()` and actual disk write silently loses the appended line.

**Fix in `tar_lab/state.py`:**
```python
def _safe_jsonl_append(self, path: Path, data: dict) -> None:
    with self._lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()
            os.fsync(f.fileno())  # ADD THIS — guarantees durability
```

Apply the same pattern to all JSONL write operations in `tar_lab/result_artifacts.py` (already has fsync in canonical_registry path — verify and make consistent).

**Verification:** Use a test that kills the process immediately after `flush()` but before `fsync()`; confirm data loss occurs without fsync and does NOT occur with fsync.

---

### Task 7.6 — Implement JSONL rotation and archival

11 JSONL append-only log files grow without bound. `research_intel.jsonl` is already 7.97MB after 6 months.

**Implementation:**
```python
def rotate_jsonl_if_needed(path: Path, max_size_mb: float = 50.0) -> None:
    """Rotate JSONL file monthly or when it exceeds max_size_mb."""
    if not path.exists():
        return
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return
    # Rotate to dated archive
    month_stamp = datetime.now().strftime("%Y-%m")
    archive_path = path.parent / f"{path.stem}_{month_stamp}.jsonl.gz"
    with open(path, "rb") as f_in:
        with gzip.open(archive_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    path.unlink()  # Remove original; start fresh
    path.touch()   # Create empty file for new appends
```

**Files requiring rotation:** `research_intel.jsonl`, `project_priority_records.jsonl`, `evidence_debt_records.jsonl`, `project_staleness_records.jsonl`, `portfolio_decisions.jsonl`, `research_decisions.jsonl`, `claim_verdicts.jsonl`, and 4 others.

Rotation runs as part of the daily health check (Task 6.7). Alert when any file exceeds 50MB (warn) or 500MB (critical).

**Verification:** Pad `research_intel.jsonl` to 51MB; run health check; confirm rotation creates archive and starts fresh file.

---

### Task 7.7 — Implement schema versioning and migration framework

No JSON state file carries a version field that is checked on read. Renaming or removing a Pydantic field causes silent data loss.

**Add `_schema_version` to all persistent Pydantic models:**
```python
class ExperimentSpec(BaseModel):
    _schema_version: str = "v2"   # Increment on any breaking change
    # ... existing fields
```

**Implement a migration registry:**
```python
MIGRATIONS: dict[str, dict[str, Callable]] = {
    "ExperimentSpec": {
        ("v1", "v2"): _migrate_experiment_spec_v1_to_v2,
    },
    "FrontierProblem": {
        ("v1", "v2"): _migrate_frontier_problem_v1_to_v2,
    },
}

def load_with_migration(path: Path, model_class: type[BaseModel]) -> BaseModel:
    raw = json.loads(path.read_text(encoding="utf-8"))
    file_version = raw.get("_schema_version", "v1")
    current_version = model_class.model_fields["_schema_version"].default
    if file_version != current_version:
        migration_key = (file_version, current_version)
        migration_fn = MIGRATIONS.get(model_class.__name__, {}).get(migration_key)
        if migration_fn:
            raw = migration_fn(raw)
            audit_log("SCHEMA_MIGRATION", {"file": str(path), "from": file_version, "to": current_version})
        else:
            raise SchemaMigrationError(f"No migration from {file_version} to {current_version} for {model_class.__name__}")
    return model_class.model_validate(raw)
```

**Verification:** Create a v1 JSON file with the old schema; confirm `load_with_migration()` applies the migration and returns a valid v2 object. Confirm missing migration raises `SchemaMigrationError`.

---

### Task 7.8 — Back up ChromaDB on a weekly schedule

The vector memory store at `tar_state/memory/` (ChromaDB embedded database) is not backed up anywhere. If the E: drive fails, all literature embeddings from 1,513 papers are permanently lost.

**Weekly backup job (add to `tar_watchdog.py` or a separate scheduled task):**
```python
def backup_chromadb(workspace: Path, backup_root: Path) -> Path:
    memory_dir = workspace / "tar_state" / "memory"
    if not memory_dir.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d")
    backup_path = backup_root / f"chromadb_backup_{timestamp}.tar.gz"
    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(memory_dir, arcname="memory")
    # Keep only last 4 weekly backups
    _prune_old_backups(backup_root, pattern="chromadb_backup_*.tar.gz", keep=4)
    return backup_path
```

**Backup target:** A secondary location — ideally a different drive than E:, or uploaded to S3 (Task 7.9).

Add a health check assertion (Task 6.7): if the most recent ChromaDB backup is >8 days old, raise WARNING.

**Verification:** Run backup; confirm `chromadb_backup_YYYYMMDD.tar.gz` exists. Simulate restore from backup; confirm knowledge graph is accessible.

---

### Task 7.9 — Replace FTP publishing with S3 or SFTP

`sync_research.py` uses cleartext FTP (credentials in plaintext, no retry, blocks main thread, no verification after upload).

**Replacement with S3 (boto3):**
```python
import boto3
from botocore.exceptions import ClientError
import time

def push_to_s3(local_path: Path, bucket: str, key: str,
               max_retries: int = 3) -> bool:
    s3 = boto3.client("s3")  # credentials from environment
    for attempt in range(max_retries):
        try:
            s3.upload_file(str(local_path), bucket, key)
            # Verify remote checksum
            response = s3.head_object(Bucket=bucket, Key=key)
            remote_etag = response["ETag"].strip('"')
            local_md5 = md5_file(local_path)
            if remote_etag != local_md5:
                raise ValueError(f"Checksum mismatch after upload")
            return True
        except ClientError as e:
            wait = 5 * (2 ** attempt)
            time.sleep(wait)
    return False
```

Run uploads in a background thread so the main polling loop is never blocked.

**SFTP alternative (if S3 is not available):** Use `paramiko` with the host key verified, not trusted on first connection.

**Verification:** Upload a test file; confirm it exists in S3 with matching checksum. Kill upload mid-transfer; confirm retry succeeds.

---

### Task 7.10 — Add tar_api.py authentication enforcement and input validation

The API key is optional — if `TAR_API_KEY` is not set, the entire API is unauthenticated. Path parameters are not validated.

**Authentication fix:**
```python
# In tar_api.py startup:
api_key = os.environ.get("TAR_API_KEY")
if not api_key:
    raise RuntimeError(
        "TAR_API_KEY environment variable is required. "
        "Set it before starting the API server.")

def _require_api_key(x_api_key: str = Header(...)):
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
```

**Input validation:**
```python
PROJECT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')

@app.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    _: str = Depends(_require_api_key)
) -> dict:
    if not PROJECT_ID_PATTERN.match(project_id):
        raise HTTPException(status_code=400, detail="Invalid project_id format")
    # ... proceed safely
```

**Sanitize error responses** — no backend command names or internal tracebacks:
```python
@app.exception_handler(Exception)
async def generic_error_handler(request, exc):
    error_id = uuid4().hex[:8]
    logger.error(f"[{error_id}] Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": f"Internal error [{error_id}]"})
```

**Verification:** Make request without API key; confirm 401. Pass `project_id=../../etc/passwd`; confirm 400. Trigger an internal error; confirm response contains only `error_id`, not a stack trace.

---

### Task 7.11 — Refactor tar_author.py into focused modules

`tar_author.py` is 6,437 lines — a god-class combining evidence collection, LaTeX generation, citation handling, bibliography management, originality auditing, and compilation. It cannot be unit-tested, is difficult to understand, and cannot be reused across different research domains.

**Split into four focused modules:**

```
tar_author/
├── __init__.py          (imports from submodules; backward compatible)
├── engine.py            (orchestration, evidence loading, phase gating — ~500 lines)
├── sections.py          (per-section LaTeX generation, domain-agnostic prompts — ~1200 lines)
├── citations.py         (citation validation, bibliography, dedup — ~700 lines)
└── compiler.py          (LaTeX compilation, PDF generation, error handling — ~400 lines)
```

**Parameterize all prompts:** Remove hardcoded TCL-specific numbers and CL-specific references. Every prompt should accept domain, method_name, and dataset as parameters so the authoring pipeline generalises to Paper 2 (TCL-for-LLMs) and Paper 3 (TAR system).

**Minimum test coverage for each module:** At least one unit test per public function. Total new tests: ~20.

**Verification:** Run `pytest tests/test_tar_author*.py`; all tests pass. Import `from tar_author import AuthorEngine` in a new context without errors.

---

### Task 7.12 — Add RunPod executor wrapper

The GTX 1650 (4GB VRAM) is saturated. Phase 2 requires running 5-seed experiments that each need 3.5GB VRAM — cannot run in parallel locally. RunPod A40 (24GB VRAM) enables 6× batch sizes and parallel seed execution.

**Add `RemoteExecutor` to `tar_lab/backend_factory.py`:**
```python
class RunPodExecutor:
    """Translates local experiment launch to RunPod serverless API call."""
    
    def __init__(self, api_key: str, pod_type: str = "AMPERE_16"):
        self.api_key = api_key
        self.pod_type = pod_type
    
    def launch(self, plan: ExperimentLaunchPlan) -> RunPodJobRecord:
        """Submit experiment to RunPod; return job ID for status polling."""
        import runpod
        runpod.api_key = self.api_key
        job = runpod.run(
            endpoint_id=self._get_or_create_endpoint(),
            input={
                "command": plan.command,
                "config": plan.config,
                "output_dir": plan.output_dir,
            }
        )
        return RunPodJobRecord(job_id=job["id"], plan=plan)
    
    def poll_status(self, job_id: str) -> str:
        """Returns: running | completed | failed"""
        import runpod
        status = runpod.status(job_id)
        return status["status"].lower()
    
    def retrieve_results(self, job_record: RunPodJobRecord) -> Path:
        """Download results from RunPod storage to local path."""
        # ... download from S3 or RunPod storage
```

**Configuration in `tar_state/runpod_config.json`** (credentials from environment, not file):
```json
{
  "enabled": true,
  "pod_type": "NVIDIA A40",
  "max_concurrent_jobs": 3,
  "cost_limit_usd_per_day": 25.0
}
```

**Verification:** Submit a 5-minute smoke test experiment to RunPod; confirm it runs, produces results, and downloads correctly.

---

### Task 7.13 — Add Prometheus metrics export and alerting

Three metrics matter operationally: `experiments_running`, `gpu_vram_free_gb`, `daemon_cycle_latency_ms`.

**Add `/metrics` endpoint to `tar_dashboard.py`:**
```python
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

EXPERIMENTS_RUNNING = Gauge("tar_experiments_running", "Number of active experiments")
GPU_VRAM_FREE = Gauge("tar_gpu_vram_free_gb", "Free GPU VRAM in GB")
DAEMON_CYCLE_LATENCY = Gauge("tar_daemon_cycle_latency_ms", "Last daemon cycle duration in ms")

@app.route("/metrics")
def metrics():
    # Update gauges from hardware_state.json
    hw = _load_hardware_state()
    GPU_VRAM_FREE.set(hw.get("gpu", {}).get("vram_free_gb", 0))
    EXPERIMENTS_RUNNING.set(len(hw.get("processes", [])))
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
```

**Alert rules (Prometheus AlertManager or simple threshold polling):**
```yaml
# alerts.yml
- alert: GPUMemoryLow
  expr: tar_gpu_vram_free_gb < 0.3
  for: 5m
  annotations:
    summary: "GPU VRAM critically low — experiment may OOM"

- alert: DaemonCycleStalled
  expr: tar_daemon_cycle_latency_ms > 120000
  for: 3m
  annotations:
    summary: "Daemon cycle taking >2 minutes — possible deadlock"
```

**Verification:** GPU VRAM drops below 0.3GB (simulate by filling VRAM); confirm alert fires within 5 minutes.

---

**Phase 7 exit gate — ALL must be true:**
- [ ] GitHub Actions CI running on every push; merge blocked on failure
- [ ] `requirements.lock` committed with exact pins; clean install + all tests pass
- [ ] Docker image pinned to digest; SBOM generated and committed
- [ ] Gunicorn serving dashboard; `/readiness` and `/liveness` endpoints present
- [ ] `os.fsync()` in all JSONL append paths (confirmed by code search)
- [ ] JSONL rotation running; health check alerts on >50MB files
- [ ] Schema versioning field on all persistent models; migration registry in place
- [ ] ChromaDB weekly backup running; restore tested
- [ ] FTP replaced with S3/SFTP; retry and checksum verification working
- [ ] `TAR_API_KEY` required; input validation on all path parameters; sanitized error responses
- [ ] `tar_author.py` split into 4 focused modules; all new tests passing
- [ ] RunPod executor wrapper functional; smoke test experiment submitted
- [ ] Prometheus `/metrics` endpoint live; at least two alert rules configured

---

## PHASE 8 — RESEARCH AUTOMATION ENHANCEMENTS
**Duration:** 4–6 weeks (begins after Phase 4)
**Track:** C (Automation & Governance)
**Owner:** Lead Researcher / Algorithm Engineer
**Objective:** Upgrade the autonomous research loop from "fast idea generator" to "rigorous scientific discoverer." Prevent false positives, break local optima, ground proposals in literature.

---

### Task 8.1 — Implement the mechanism-isolation ablation engine

**Priority:** HIGH — the most important addition to the research automation architecture

When a hypothesis claims a specific mechanism (e.g., "elastic anchoring via gradient-EMA reduces drift"), the system should automatically generate an orthogonal ablation: disable the claimed mechanism, test if performance degrades.

**Implementation in `tar_lab/generative_director.py`:**
```python
def _generate_mechanism_knockout(self, hypothesis: Hypothesis) -> Optional[ExperimentSpec]:
    """Parse hypothesis rationale for mechanism names; generate a knockout ablation."""
    mechanisms = self._extract_mechanisms_from_rationale(hypothesis.rationale)
    if not mechanisms:
        return None
    
    # For each claimed mechanism, propose disabling it
    knockout_configs = []
    for mechanism in mechanisms:
        if mechanism == "gradient_ema_importance":
            knockout_configs.append({"use_importance_weighting": False})
        elif mechanism == "elastic_penalty":
            knockout_configs.append({"lambda_tcl": 0.0})
        elif mechanism == "per_task_anchor":
            knockout_configs.append({"freeze_sigma_star": True, "anchor_at_init": True})
    
    return ExperimentSpec(
        name=f"knockout_{hypothesis.experiment_id}",
        hypothesis=f"Disabling [{', '.join(mechanisms)}] should eliminate the effect from {hypothesis.experiment_id}",
        config_overrides=knockout_configs[0] if knockout_configs else {},
        pre_registration=PreRegistrationRecord(
            hypothesis=f"If {hypothesis.experiment_id}'s improvement is due to {mechanisms[0]}, "
                       f"disabling it should restore baseline forgetting.",
            min_detectable_effect_d=0.5,
            required_seeds=5,
            stopping_rule="If knockout forgetting matches baseline ± CI95, mechanism confirmed absent.",
        ),
        mechanism_knockout_of=hypothesis.experiment_id,
    )
```

**Gate:** Hypotheses are only marked "mechanism_confirmed" in the evidence inventory after the corresponding knockout experiment is complete and shows statistically significant degradation.

**Verification:** Pass a hypothesis with `rationale` containing "gradient_ema_importance drives the improvement"; confirm a knockout experiment is automatically generated and queued.

---

### Task 8.2 — Implement cross-failure clustering and hypothesis-space escape

**Priority:** MEDIUM — breaks the local failure-mode loop

TAR currently caches each failure diagnosis independently. If 5 experiments fail on "gradient explosion," the system diagnoses it 5 times independently.

**Implementation in `tar_living_research.py`:**
```python
class FailureClusterer:
    CLUSTER_TRIGGER = 3   # failures before escape prompt fires
    SIMILARITY_THRESHOLD = 0.75  # cosine similarity for clustering
    
    def _cluster_recent_failures(self, workspace: Path, window_hours: int = 168) -> dict[str, list]:
        """Group recent failures by semantic category."""
        recent = self._load_recent_failures(workspace, window_hours)
        embeddings = [self._embed(f.diagnosis) for f in recent]
        clusters = {}
        for i, failure in enumerate(recent):
            placed = False
            for cluster_label, cluster_members in clusters.items():
                if cosine_similarity(embeddings[i], cluster_members[0]["embedding"]) > self.SIMILARITY_THRESHOLD:
                    clusters[cluster_label].append({"failure": failure, "embedding": embeddings[i]})
                    placed = True
                    break
            if not placed:
                clusters[failure.diagnosis[:50]] = [{"failure": failure, "embedding": embeddings[i]}]
        return clusters
    
    def check_and_escape(self, workspace: Path) -> Optional[str]:
        """Returns an escape prompt if stuck in a failure mode cluster."""
        clusters = self._cluster_recent_failures(workspace)
        for label, members in clusters.items():
            if len(members) >= self.CLUSTER_TRIGGER:
                return self._build_escape_prompt(label, len(members))
        return None
    
    def _build_escape_prompt(self, failure_mode: str, count: int) -> str:
        return (
            f"We have failed {count} consecutive times on: '{failure_mode}'. "
            f"This failure mode has been diagnosed {count} times — it is systemic, not random. "
            f"Propose a fundamentally different approach that avoids this failure mode entirely. "
            f"Do NOT propose another variant of the same approach."
        )
```

**Verification:** Inject 3 failures with identical diagnoses; confirm escape prompt fires and a structurally different experiment family is proposed.

---

### Task 8.3 — Add pre-synthesis literature grounding

**Priority:** MEDIUM — prevents synthesising code that already exists in the literature

Before generating any method code, the synthesizer should query the knowledge graph: "Does any paper implement mechanism X on dataset Y?"

```python
def _check_literature_before_synthesis(self, idea: str, workspace: Path) -> dict:
    """Query knowledge graph for existing implementations of the proposed idea."""
    kg = KnowledgeGraph.load(workspace)
    if not kg.has_entries():
        return {"grounded": False, "reason": "Knowledge graph empty — run Phase 9"}
    
    # Semantic search for similar methods
    idea_embedding = self._embed(idea)
    similar_papers = kg.search_mechanisms(idea_embedding, top_k=5)
    
    if similar_papers:
        return {
            "grounded": True,
            "similar_papers": [p.title for p in similar_papers[:3]],
            "recommendation": (
                f"Found {len(similar_papers)} similar papers. "
                f"Build on these rather than re-implementing. "
                f"Cite them in the method description."
            ),
        }
    return {"grounded": True, "similar_papers": [], "recommendation": "No prior implementations found — genuinely novel."}
```

**Add a 60-second literature check before `maybe_synthesize_for_spec()` proceeds.**

**Verification:** Ask for synthesis of "elastic weight consolidation with EMA"; confirm the system finds the EWC paper and recommends building on it rather than re-implementing.

---

### Task 8.4 — Fix Director/Strategist/Scout epistemological gaps

**Priority:** MEDIUM

Three specific prompt engineering fixes identified by the research automation expert:

**4a. Director pivot logic:** The current pivot fires after exactly 3 consecutive fail-fast events. Replace with failure-mode-specific pivots (Task 8.2 handles this).

**4b. Strategist memory gap:** The Strategist prompt only receives the Director's policy — not the evidence bundle or contradiction review. Add evidence summary to Strategist context:
```python
_STRATEGIST_PROMPT_ADDITION = """
Current evidence notes:
{evidence_summary}

Contradiction alerts (if any):
{contradiction_alerts}

Do NOT propose hyperparameters that have already been tested if they produced the same failure.
"""
```

**4c. Known-unknown routing:** If a hypothesis requires grounding in prior work that is NOT in the knowledge graph, the system should route to literature synthesis BEFORE the Director proposes experiments:
```python
def _should_route_to_literature_first(self, frontier_problem: FrontierProblem) -> bool:
    kg = KnowledgeGraph.load(self.workspace)
    coverage = kg.domain_coverage(frontier_problem.primary_domain_id)
    return coverage < 0.5  # Route to literature if <50% of domain is covered
```

**Verification (4b):** Inject a contradiction alert into the evidence bundle; confirm the Strategist prompt includes it and the proposed hyperparameters avoid the contradicted configuration.

---

### Task 8.5 — Upgrade Director hierarchy prompt quality

**Priority:** MEDIUM

The Director has three role prompts (RuleDirector, RuleStrategist, RuleScout) that are currently hardcoded with arbitrary thresholds.

**Three targeted improvements:**

**5a. Failure streak threshold:** Change from hardcoded `failure_streak >= 5` to an adaptive threshold based on the failure mode's historical frequency. If a failure mode historically occurs 30% of the time, require more than 5 failures before pivoting (otherwise, you'd pivot on normal statistical variation).

**5b. Memory integration depth:** Currently `_memory_aware_hyperparameters()` extracts alpha/eta candidates from prior runs. Extend to also extract: methods that consistently outperformed on similar datasets, class orderings that produced stable training, batch sizes that avoided OOM.

**5c. Confidence calibration:** Director proposals currently have no uncertainty estimate. Add `confidence: float` (0–1) to all Director outputs. If confidence < 0.5, automatically route to a literature synthesis step before proceeding.

**Verification:** Director proposes an experiment with `confidence: 0.4`; confirm the system routes to literature synthesis before accepting the proposal.

---

**Phase 8 exit gate — ALL must be true:**
- [ ] Mechanism-isolation ablation engine generates knockout experiments for hypothesis with named mechanisms
- [ ] Cross-failure clustering fires escape prompt after 3 same-category failures
- [ ] Literature grounding check runs before all synthesis attempts
- [ ] Strategist prompt includes evidence summary and contradiction alerts
- [ ] Known-unknown routing to literature synthesis when domain coverage < 50%
- [ ] Director outputs include calibrated confidence; low-confidence routed to literature
- [ ] All changes covered by at least one new test each

---

## PHASE 9 — LITERATURE & KNOWLEDGE INFRASTRUCTURE
**Duration:** 2–4 weeks (begins after Phase 4 — ActiveLearner wired in Task 4.8)
**Track:** C (Automation & Governance)
**Owner:** Lead Researcher
**Objective:** Fully activate the literature intelligence layer that has been built but never run. Populate the knowledge graph. Build the 15-paper citation base required for Paper 1.

---

### Task 9.1 — Verify ActiveLearner is running and ingesting papers

Task 4.8 wired `LiteratureBrain.start()` to the orchestrator. This task verifies the result.

**Verification steps:**
1. Check `tar_state/literature/manifests/` contains recent ingest manifests (< 4 hours old)
2. Run `brain.corpus_summary()` — should return non-zero paper count
3. Check `knowledge_graph.json` (or the underlying SQLite db) has entries
4. Run a test query: "Find papers about elastic weight consolidation" — should return ≥ 3 results

If any step fails, diagnose why the background thread stopped and fix the root cause.

**Verification:** `brain.corpus_summary()` returns `{"papers": N, "claims": M, "domains": K}` where all three are > 0.

---

### Task 9.2 — Build the knowledge graph schema and run initial population

The knowledge graph schema from the enhancement report defines 6 node types and 7 edge types (Paper, Mechanism, Claim, Dataset, Benchmark, Domain + typed edges). Implement this schema in the SQLite database that backs ChromaDB, or as a separate `knowledge_graph.db`.

**Population script:** `tar_lab/knowledge_graph_builder.py`

```python
def build_knowledge_graph_from_corpus(workspace: Path) -> KGStats:
    """One-time script to extract KG entries from all ingested papers."""
    kg = KnowledgeGraph.connect(workspace)
    papers = load_all_paper_artifacts(workspace)
    
    for paper in papers:
        # Add paper node
        kg.upsert_paper(paper)
        # Extract and add mechanism mentions
        mechanisms = extract_mechanisms(paper.claims)
        for mechanism in mechanisms:
            kg.upsert_mechanism(mechanism, introduced_by=paper.id)
        # Add claim nodes with contradiction edges
        for claim in paper.claims:
            kg.upsert_claim(claim)
            existing = kg.find_contradicting_claims(claim)
            for contradiction in existing:
                kg.add_edge(claim.id, contradiction.id, "contradicts")
    return kg.stats()
```

Run on all 1,513 ingested papers. Expected runtime: 2–4 hours (LLM extraction calls for each paper).

**Verification:** `kg.stats()` returns `{papers: ~1513, mechanisms: ~500+, claims: ~5000+, domains: 10+}`.

---

### Task 9.3 — Build the 15-paper essential citation base

The enhancement report identified 15 papers that any 2026 CL paper must cite. Download, ingest, and verify all 15 are in the knowledge graph.

| # | Paper | Year | ArXiv / DOI |
|---|---|---|---|
| 1 | Kirkpatrick et al. — EWC | 2017 | arXiv:1612.00796 |
| 2 | Zenke et al. — Synaptic Intelligence | 2017 | PMLR:pmlr-v70-zenke17a |
| 3 | Li & Hoiem — LwF | 2016 | arXiv:1606.09282 |
| 4 | Buzzega et al. — DER++ | 2020 | NeurIPS 2020 |
| 5 | Rebuffi et al. — iCaRL | 2017 | arXiv:1611.07725 |
| 6 | Lopez-Paz & Ranzato — GEM | 2017 | arXiv:1706.08840 |
| 7 | Chaudhry et al. — A-GEM | 2019 | arXiv:1812.00420 |
| 8 | Wang et al. — DualPrompt | 2022 | arXiv:2204.04799 |
| 9 | Wang et al. — L2P | 2022 | arXiv:2112.08654 |
| 10 | Rusu et al. — Progressive NN | 2016 | arXiv:1606.04671 |
| 11 | Javed & White — BWT/FWT | 2019 | arXiv:1905.12588 |
| 12 | Martens & Grosse — K-FAC | 2015 | arXiv:1503.05671 |
| 13 | Amari — Natural Gradient | 1998 | Neural Computation |
| 14 | McCloskey & Cohen — Interference | 1989 | Psychology of Learning |
| 15 | Chaudhuri et al. — CL survey | 2019 | arXiv:1904.07734 |

For each: download PDF (via Semantic Scholar API), ingest via `literature_engine.py`, verify it appears in `knowledge_graph.json`, add to `tar_state/literature/essential_citations.bib` in BibTeX format.

**Verification:** `kg.query("papers_by_title", "Overcoming Catastrophic Forgetting")` returns the EWC paper. All 15 papers have BibTeX entries in `essential_citations.bib`.

---

### Task 9.4 — Implement claim-level novelty detection

Current novelty detection uses keyword matching. Replace with claim-level semantic search:

```python
def is_claim_novel(proposed_claim: str, workspace: Path,
                   similarity_threshold: float = 0.75) -> dict:
    """
    Check if a proposed experimental claim is genuinely novel.
    Uses SPECTER2 embeddings + KG claim index.
    """
    kg = KnowledgeGraph.connect(workspace)
    proposed_embedding = embed_text(proposed_claim)
    
    similar_claims = kg.search_claims(proposed_embedding, top_k=10)
    
    if not similar_claims:
        return {"novel": True, "confidence": 0.9, "similar_papers": []}
    
    max_similarity = max(c.similarity for c in similar_claims)
    if max_similarity > similarity_threshold:
        return {
            "novel": False,
            "confidence": max_similarity,
            "similar_claims": [c.text for c in similar_claims[:3]],
            "similar_papers": [c.paper_title for c in similar_claims[:3]],
            "recommendation": f"Most similar to: '{similar_claims[0].paper_title}' ({max_similarity:.0%} similar). "
                              f"Consider how your work differs from this.",
        }
    
    return {"novel": True, "confidence": 1 - max_similarity,
            "similar_papers": [c.paper_title for c in similar_claims[:3]]}
```

**Integrate into the Director:** Before any experiment is queued with a novelty claim, run `is_claim_novel()`. If `novel: False`, add to the hypothesis: "This claim is similar to [paper]. Explicitly justify how your approach differs."

**Verification:** Pass the claim "EWC reduces catastrophic forgetting in sequential task learning"; confirm `novel: False` with high similarity to the EWC paper. Pass "gradient-EMA temporal smoothing reduces Fisher estimation bias"; confirm `novel: True`.

---

### Task 9.5 — Implement literature-aware Director proposal generation

Replace the current Director (which generates proposals without literature context) with a literature-grounded variant:

```python
_LITERATURE_AWARE_DIRECTOR_PROMPT = """
You are proposing a continual learning experiment.

LITERATURE CONTEXT:
Current SOTA for {dataset} ({metric}):
{sota_table}

Most relevant papers to this frontier:
{relevant_papers}

Papers that CONTRADICT our current hypothesis:
{contradicting_papers}

TASK:
Propose ONE experiment that:
1. Advances beyond the current SOTA in a specific, measurable way
2. Does not replicate an already-published experiment (verified: {novelty_check})
3. Tests a hypothesis that the contradicting papers above cannot explain
4. Is achievable within {gpu_budget_hours} GPU hours

Format: JSON with fields: hypothesis, design, expected_effect_size, novelty_justification
"""
```

**Verification:** Generate a Director proposal; confirm the output JSON includes `novelty_justification` and `expected_effect_size`. Confirm the SOTA table is populated from the knowledge graph, not hardcoded.

---

**Phase 9 exit gate — ALL must be true:**
- [ ] `brain.corpus_summary()` returns non-zero counts for papers, claims, domains
- [ ] Knowledge graph schema implemented; initial population complete on all 1,513 papers
- [ ] All 15 essential papers in KG; BibTeX file exists with all 15 entries
- [ ] Claim-level novelty detection functional; EWC claim correctly identified as non-novel
- [ ] Literature-aware Director proposal includes SOTA table and novelty justification

**What Phase 9 unlocks:** Phase 5 (related work section) can be written from KG. Phase 8 literature grounding becomes functional.

---

*Stage 5 complete — Phases 7, 8, and 9 with 29 tasks (7.1–7.13, 8.1–8.5, 9.1–9.5)*
