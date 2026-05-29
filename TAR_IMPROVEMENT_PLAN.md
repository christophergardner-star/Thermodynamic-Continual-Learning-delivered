# TAR System Improvement Plan

**Status:** Plan only — no code changes made  
**Based on:** Deep codebase survey by 5 parallel research agents (2026-05-28)  
**Scope:** All improvements address real bugs or structural weaknesses found in the live codebase  
**Constraint:** TAR must remain live throughout; no maintenance windows required for any item

---

## Priority Matrix

| # | Track | Improvement | Risk | Impact | Live-safe? |
|---|-------|-------------|------|--------|------------|
| 1 | State Integrity | Fix research_director_state.json double-write | LOW | CRITICAL | Yes |
| 2 | State Integrity | Make process_registry.json writes atomic | LOW | HIGH | Yes |
| 3 | Paper Pipeline | Implement partial-evidence writing (stubs) | MEDIUM | CRITICAL | Yes |
| 4 | Experiment Lifecycle | Wire heartbeat file for liveness decisions | LOW | HIGH | Yes |
| 5 | Experiment Lifecycle | Block pending-duplicate queue submissions | LOW | HIGH | Yes |
| 6 | Experiment Lifecycle | Auto top-up for underpowered experiments | MEDIUM | MEDIUM | Yes |
| 7 | Scheduling | Paper-blocking urgency boost | LOW | MEDIUM | Yes |
| 8 | Scheduling | Cap / fix scheduler rationale truncation | LOW | LOW | Yes |
| 9 | Observability | LLM path transparency per section | LOW | MEDIUM | Yes |
| 10 | Observability | Dual state file consolidation | LOW | LOW | Yes |

---

## Track 1 — State File Integrity

### Background

The agent survey found the following write patterns across all state files:

| File | Pattern | Risk |
|------|---------|------|
| `experiment_queue.json` | Atomic (tmp + os.replace) | Safe |
| `experiment_archive.json` | Atomic (tmp + os.replace) | Safe |
| `tar_lab/state.py` files | threading.RLock + atomic | Safe |
| `runtime_ledger.json` | File lock + atomic | Safe |
| `research_director_state.json` | **Direct write × 2 — DANGER** | CRITICAL |
| `process_registry.json` | Direct write — no atomicity | HIGH |
| `watchdog_state.json` | Direct write | MEDIUM |
| `living_research_daemon.json` | Direct write | MEDIUM |
| `hardware_state.json` | Direct write | MEDIUM |

The two critical paths are addressed below. Medium-risk files (daemon, watchdog, hardware) update frequently and contain mostly diagnostic state — a torn write is recoverable from the next tick. The critical files contain experiment and paper directives; a torn write there corrupts the science pipeline.

---

### Improvement 1 — Fix research_director_state.json Double-Write

**Root cause**  
`tar_research_director.py` lines 347–364 write the same file twice in one function call:

```
Line 349:  self.state_path.write_text(...)           ← first direct write (not atomic)
           ... side-effect: sync_human_review_from_director_state() writes OTHER files ...
Line 363:  self.state_path.write_text(...)           ← second direct write (not atomic)
```

A crash between lines 349 and 363 leaves the file in its first (partial) state. The scheduler reads this file on every 30-second tick. If it reads during a write, it gets garbage frontier priorities and may schedule the wrong experiment or stall indefinitely.

**Constraint note**  
`tar_research_director.py` is on the protected file list. Two approaches are available:

- **Option A (preferred):** Request a one-time exception to add a single helper function and change two `.write_text()` calls to an atomic pattern. The change is ~10 lines and surgical — no logic change, only write pattern.
- **Option B (no-touch workaround):** Add a reader-side retry in the scheduler: if a JSON parse of `research_director_state.json` fails or the file is zero-length, retry once after 500ms before acting on stale state. This does not fix the write but makes the reader resilient.

**Recommended fix (Option A — requires exception to no-touch rule)**

File: `tar_research_director.py`

1. Extract a local helper `_atomic_write(path, payload)` that mirrors the pattern already used in `tar_experiment_orchestrator.py`:
   ```
   tmp = path.with_suffix(".tmp")
   tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
   os.replace(tmp, path)
   ```
2. Build `payload` once before any side-effect calls.
3. Call `_atomic_write(self.state_path, payload)` exactly once, after `sync_human_review_from_director_state()` completes.
4. Remove both existing `self.state_path.write_text(...)` calls.

**Risk:** Near-zero. This is a write-pattern change only — no logic, no data model change. The scheduler continues to read the same JSON structure.

**Can run live:** Yes — the next director cycle after deployment uses the new write path.

---

### Improvement 2 — Make process_registry.json Writes Atomic

**Root cause**  
`tar_experiment_orchestrator.py` `_write_process_registry()` (lines 984–1027) does a direct `path.write_text(...)`. The watchdog reads this file on every 15-second tick. A torn write causes the watchdog to see a corrupted PID map, potentially misidentifying a live process as dead and triggering a false stall.

**Fix**  
File: `tar_experiment_orchestrator.py`

In `_write_process_registry()`, change the final write from:
```python
path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
```
to the atomic tmp-swap pattern already used by `_save()` in the same file:
```python
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
os.replace(tmp, path)
```

**Risk:** Near-zero. Same pattern used by queue/archive writes; os.replace is atomic on NTFS.

**Can run live:** Yes.

---

### Note on Full SQLite Migration

A full migration to SQLite would give transaction semantics across all state files. The agent confirmed no DB layer exists today. This would eliminate the entire class of file-race bugs but is a significant architectural change (weeks of work, requires careful schema design, migration of all readers and writers). It is **not** in this plan's immediate scope — the two targeted fixes above address the only two critical-severity race conditions identified. SQLite migration can be a follow-on project once these are fixed.

---

## Track 2 — Experiment Lifecycle Reliability

### Improvement 3 — Block Pending-Duplicate Queue Submissions

**Root cause**  
`tar_experiment_orchestrator.py` `submit()` (lines 945–974) correctly deduplicates against experiments that are COMPLETE, RUNNING, FAILED, or SKIPPED. It does NOT check for an already-PENDING spec with the same ID.

Additionally, the Director's duplicate-config bug (7 identical experiments queued) stems from experiments receiving DIFFERENT IDs despite identical configs. The SHA1 includes the `name` field — if the Director generates different `name` strings for the same config, they get different IDs and both reach the queue.

Two independent fixes are needed:

**Fix A — Dedup pending IDs (trivial)**  
File: `tar_experiment_orchestrator.py`, `submit()` method

Change line 947 from:
```python
if spec.id in self._specs:
    existing = self._specs[spec.id]
    if existing.status in (EXP_COMPLETE, EXP_RUNNING):
```
to:
```python
if spec.id in self._specs:
    existing = self._specs[spec.id]
    if existing.status in (EXP_COMPLETE, EXP_RUNNING, EXP_PENDING, "queued"):
```
This prevents the same ID from being submitted twice when already waiting in queue.

**Fix B — Config-hash dedup at submit (prevents Director duplicate-config bug)**  
File: `tar_experiment_orchestrator.py`

Add a `_config_fingerprint(spec)` helper that hashes `(dataset, method, sorted config_overrides, sorted seeds)` — WITHOUT the `name` field. In `submit()`, before adding to `self._specs`, check if any existing PENDING spec shares the same fingerprint. If so, log and skip.

```
def _config_fingerprint(spec: ExperimentSpec) -> str:
    key = (
        spec.dataset,
        spec.method,
        json.dumps(spec.config_overrides or {}, sort_keys=True),
        json.dumps(sorted(spec.seeds or [])),
        spec.optimizer_backend,
    )
    return hashlib.sha1(repr(key).encode()).hexdigest()[:12]
```

The in-memory `self._specs` dict can be supplemented with a `_pending_fingerprints: set[str]` set. On submit, check the fingerprint before adding.

**Risk:** Low. The fingerprint check is additive logic; it cannot cause valid experiments to be skipped because legitimate re-runs intentionally change `config_overrides` or `seeds`.

**Can run live:** Yes.

---

### Improvement 4 — Wire Heartbeat File for Liveness Decisions

**Current state (from agent research)**  
The heartbeat system already exists and is already running:
- `tar_experiment_worker.py` writes `run_locks/{experiment_id}.pid` on start
- A background thread updates `last_heartbeat` field every 60 seconds (`_HEARTBEAT_INTERVAL_S = 60`)
- `_check_worker_heartbeats()` in `tar_living_research.py` checks if `last_heartbeat` is older than 10 minutes
- **But:** `_check_worker_heartbeats()` only logs a warning — it does not act

The orchestrator's `reconcile_runtime_state()` still uses `_pid_exists()` (Win32 OpenProcess) which returns True for recently-exited zombie processes on Windows. This causes stalled detection to miss some dead experiments.

**Fix**  
Files: `tar_experiment_orchestrator.py`, `tar_living_research.py`

**Step 1** — In `reconcile_runtime_state()` (lines 1144–1228), add heartbeat-file check as primary liveness signal. Change the liveness check from:
```python
elif spec.pid and not self._pid_exists(spec.pid):
    self._mark_stalled(spec)
```
to a two-stage check:
```python
elif spec.pid:
    heartbeat_fresh = _heartbeat_is_fresh(self.workspace, spec.id, threshold_s=120)
    pid_alive = self._pid_exists(spec.pid)
    if not heartbeat_fresh and not pid_alive:
        self._mark_stalled(spec)
    elif not heartbeat_fresh and pid_alive:
        # PID exists but heartbeat stale — log for investigation; don't mark stalled yet
        self._log(f"[reconcile] {spec.id}: heartbeat stale but PID {spec.pid} alive — monitoring")
```

**Step 2** — Add `_heartbeat_is_fresh(workspace, experiment_id, threshold_s)` helper in `tar_experiment_orchestrator.py` or `tar_runtime_tracking.py`:
```python
def _heartbeat_is_fresh(workspace: Path, experiment_id: str, threshold_s: float = 120.0) -> bool:
    lock_path = workspace / "tar_state" / "run_locks" / f"{experiment_id}.pid"
    if not lock_path.exists():
        return False
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        last_hb = data.get("last_heartbeat") or data.get("started_at_utc") or ""
        if not last_hb:
            return False
        dt = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return age < threshold_s
    except Exception:
        return False
```

**Step 3** — In `_check_worker_heartbeats()` (`tar_living_research.py`), change from logging-only to actually requesting stall marking when heartbeat age > 10 minutes:
```python
# Change from:
logger.warning(f"Heartbeat stale for {exp_id} — {age:.0f}s since last beat")
# Change to:
logger.warning(f"Heartbeat stale for {exp_id} — {age:.0f}s since last beat; requesting stall check")
orch.request_stall_check(exp_id)  # New method that triggers reconcile for specific ID
```

**Risk:** Low. The heartbeat file already exists; we are changing the decision logic, not the data. The two-stage check (heartbeat AND pid) prevents false stalls.

**Can run live:** Yes — the threshold (120s) is generous relative to the 60s heartbeat interval.

---

### Improvement 5 — Auto Top-Up for Underpowered Experiments

**Root cause**  
`tar_lab/validation.py` `classify_trust_tier()` (line 232) requires `seed_count >= MIN_SEEDS_FOR_PUBLICATION` (5) for `publication_allowed=True`. When old experiments completed with 3 seeds (before the 5-seed standard was established), they block papers indefinitely. The current process is entirely manual: discover → supersede in archive → re-queue with 5 seeds → update both state files.

**Key findings from agent research:**
- Director already hardcodes `seeds: [42, 0, 1, 2, 3]` (5 seeds) for all new proposals — so new experiments are fine
- The problem is historical 3-seed experiments
- `seed_count` is NOT stored in the archive entry; it is computed on demand from `result.json`
- No auto-top-up mechanism exists anywhere

**Proposed fix: `_check_and_queue_seed_topups()` in `tar_living_research.py`**  

This is a periodic monitor function called every N daemon cycles (not every tick — validation state loading is moderately expensive). It does the following:

1. Call `build_validation_state(workspace)` to get current validation report
2. For each experiment where `trust_tier` is NOT None but `publication_allowed=False` due to `insufficient_seeds`:
   - Check if a top-up experiment (ID pattern: `{original_id}-topup-5seed`) is already in queue or archive
   - If not, generate a new `ExperimentSpec` copying the original experiment's `dataset`, `method`, `config_overrides`, `backbone`, `hardware_budget`, `frontier_problem_id`, but with:
     - `seeds: [42, 0, 1, 2, 3]` (full 5-seed run; cheaper than detecting which seeds ran)
     - `id`: deterministic from original ID + "-topup-5seed"
     - `archive_reason` on the old entry changed from `complete` to `superseded_by_{new_id}`
     - `priority: 20` (high priority — these unblock papers)
     - `author_paper_id`: set to the paper that is waiting for the original
3. Submit to orchestrator queue
4. Mark original archive entry as `status=skipped, archive_reason=superseded_by_{new_id}`

**Important constraints:**
- Never re-queue if original has `status=skipped` or `archive_reason` already starts with `superseded_by`
- Never re-queue if any top-up variant is already in queue (pending/running) or in archive (complete)
- The function reads from `build_validation_state()` which already knows `insufficient_seeds` — no new logic needed to detect the condition

**Files affected:**
- `tar_living_research.py`: add `_check_and_queue_seed_topups()`, call it every 10 cycles in daemon loop
- `tar_experiment_orchestrator.py`: uses existing `submit()` and `_save_archive_records()` paths — no new functions needed

**Risk:** Medium. The function modifies the archive (marking old entries as skipped) and adds to the queue. These writes use existing atomic patterns. The dedup check (improvement 5) prevents double-submissions. Test thoroughly before enabling.

**Can run live:** Yes, but should be enabled behind a flag initially (`ENABLE_AUTO_TOPUP=true` env var or config file field).

---

## Track 3 — Paper Pipeline Completion

### Improvement 6 — Implement Partial-Evidence Writing (LaTeX Stubs)

**Current state**  
The agent survey confirmed the following is already present in the codebase:
- `PaperSpec.allow_partial_write: bool = False` ✓ (tar_author.py line 334)
- `validate_paper_evidence_partial()` ✓ (tar_lab/validation.py lines 365–408)
- `_generate_stub_experiment_block()` ✓ (tar_author.py lines 1169–1201)
- `_section_has_resolved_stubs()` ✓ (tar_author.py lines 1065–1082)
- `paper_plan.json` fields `has_stubs`, `stub_experiment_ids` ✓

**What is NOT yet implemented** (the parts that complete the end-to-end flow):

**Missing piece 1 — `_paper_gate_context()` does not pass `allow_partial`**  
File: `tar_author.py` (line ~4367)  
The function `_paper_gate_context()` currently only calls `validate_paper_evidence()` (strict mode). It needs an `allow_partial: bool = False` parameter that selects `validate_paper_evidence_partial()` instead.

**Missing piece 2 — `_write_paper_impl()` gate block does not use allow_partial_write**  
File: `tar_author.py` (lines ~5997–6012)  
The gate evaluation currently calls `_paper_gate_context()` without passing `allow_partial`. It needs to pass `allow_partial=spec.allow_partial_write` and then store the returned `partial_mode` and `stub_experiment_ids` for use in `_load_evidence()`.

**Missing piece 3 — `_load_evidence()` does not forward pending_stub_ids**  
File: `tar_author.py` (line ~3611)  
The function signature already accepts `pending_stub_ids: list[str] | None = None` and stores it in the evidence dict (line 3724). The call site in `_write_paper_impl()` just needs to pass the `_stub_exp_ids` list from the gate context.

**Missing piece 4 — `_build_author_spec_from_queue_entry()` does not auto-derive allow_partial_write**  
File: `tar_author.py` (line ~4227)  
When building a `PaperSpec` from a director queue entry, auto-set `allow_partial_write = human_approved AND readiness in {"outline_now", "write_now"}`. This means no per-paper manual flag is needed — any paper the director has declared ready automatically gets partial-evidence mode.

**Missing piece 5 — `_write_paper_validation_bundle()` does not check for stubs**  
File: `tar_author.py`  
After writing sections, the validation bundle needs to set `publication_ready=False` when any `% TAR-STUB:` markers remain in the output `.tex` files. The helper `_paper_has_stubs(out_dir)` (which scans `.tex` files for the marker) needs to be added and called.

**Missing piece 6 — Section loop does not auto-force-regenerate resolved stubs**  
File: `tar_author.py` (lines ~6084–6155)  
When reading an existing section file during an author cycle, `_section_has_resolved_stubs()` should be called to check if any `% TAR-STUB:` markers now refer to trusted experiments. If yes, set `force_gen=True` for that section only.

**Two papers immediately benefit:**
- `tcl-autonomous-mechanism-paper`: 20/24 experiments done, readiness=`outline_now`, `human_approved=True`
- `tcl-hpc-validation-paper`: 4/6 experiments done, readiness=`outline_now`, `human_approved=True`

**What does NOT change:** Gate B is never relaxed. Unvalidated results (missing env snapshot, quarantined, `trust_tier=None`) cannot appear in prose or tables in any mode. `publication_ready` remains False while any stub is present.

**Risk:** Medium. This is the largest set of changes in the plan. However, each piece is additive and isolated. The partial-evidence path is only taken when `allow_partial_write=True`, which requires both `human_approved` and `readiness=outline_now/write_now`. No existing papers can accidentally enter partial mode.

**Can run live:** Yes. After deployment, the next author cycle for the two ready papers will trigger partial-evidence writing. All other papers are unaffected.

---

### Improvement 7 — Dual State File Consolidation (author_state.json)

**Root cause**  
`waiting_for_experiments` is stored in both `author_state.json` (written by author loop) and `research_director_state.json` (written by director). The author loop derives its queue from `research_director_state.json` on every cycle, so `author_state.json` is technically redundant for this field. However, if anything reads `author_state.json` for paper-dependency decisions (e.g. health check Z7), stale data there causes false positives.

**Fix**  
File: `tar_author.py`, `_write_author_state()` (line ~3835)

1. Remove `waiting_for_experiments` from the persisted `author_state.json` payload. It belongs only in `research_director_state.json` (the authoritative source).
2. If dashboard UI reads `waiting_for_experiments` from `author_state.json`, change it to read from `research_director_state.json` directly — or have the author state write a `waiting_count: int` (count only) for display purposes.
3. Update `tar_health_check.py` Z7 check to source `waiting_for_experiments` from `research_director_state.json` only (it may already do this — verify first).

**Risk:** Low. The author state write is dashboard-only; no science pipeline logic reads `waiting_for_experiments` from it. Removing it makes the contract clearer.

**Can run live:** Yes.

---

## Track 4 — Scheduling Intelligence

### Improvement 8 — Paper-Blocking Urgency Boost

**Current state (from agent research)**  
The scheduler priority system has multiple layers:
1. `scheduler_rank` from Director's `experiment_directives` — PRIMARY sort key
2. Director already adds `paper_boost`: +18 for `write_now`, +12 for `outline_now`, +6 for `prepare_now`
3. `ExperimentSpec.priority` (0–100) — fourth sort key (after experiment_rank, stalled_scaleup, frontier_rank)

The paper_boost exists and works — BUT it only takes effect when the Director rebuilds `experiment_directives` (which happens periodically). Experiments queued BEFORE the paper was linked to them don't get the boost until the Director next cycles.

Additionally, the daemon (`tar_living_research.py`) knows which experiments are in `waiting_for_experiments` for each paper at submit time, but does NOT set a higher `priority` on those queue entries when submitting.

**Fix (two-part)**  

**Part A — Daemon sets priority at submit time**  
File: `tar_living_research.py`, in the function that submits experiments to the orchestrator queue

When building an `ExperimentSpec` for submission, check if the experiment ID appears in ANY paper's `waiting_for_experiments` list in `research_director_state.json`. If yes:
- Set `spec.priority = 15` (above default of 50, below Director-boosted at ~10)
- Log: `[Daemon] {exp_id} is paper-blocking — priority elevated to 15`

This ensures the boost is applied immediately at submission, not deferred to the next Director cycle.

**Part B — Rationale includes paper-blocking status**  
File: `tar_scheduler.py`, `_rationale()` method (lines 409–471)

When generating rationale, include a section listing which pending experiments are blocking paper writes. Currently this information is not surfaced in the scheduler rationale.

**Risk:** Low. Setting `priority=15` only moves paper-blocking experiments higher in the queue; it does not override hardware gates or approval gates. A 24-hour scale-up experiment that started before the boost can't be preempted.

**Can run live:** Yes.

---

### Improvement 9 — Scheduler Rationale Truncation Investigation and Fix

**Root cause (uncertain)**  
The `living_research_daemon.json` showed `scheduler_rationale` truncated mid-word: `"...The GTX 1650's tight VRAM budget (only 0.7"`. The agent survey confirmed there is NO string slice in `_rationale()` or the scheduler state save code.

The likely source is the LLM narration call (`narrate_scheduler_decision()` in `tar_lab.llm_bridge`): if the Haiku model returns a response with a mid-sentence cutoff due to `max_tokens`, it would produce exactly this pattern.

**Fix**  
File: `tar_lab/llm_bridge.py`, `call_claude()` or the specific `narrate_scheduler_decision()` call

1. Verify the `max_tokens` value used for scheduler narration. If it is low (e.g. 256 or 512), increase it to 1024 or use `None` (Haiku's max).
2. Add a fallback: if the LLM response ends mid-word (no terminal punctuation), append `"..."` rather than serving a truncated sentence as the rationale.
3. Add a rationale length cap of 4000 characters as a defensive measure in the scheduler state save — not to truncate mid-sentence, but to prevent unbounded growth on long schedules.

**Risk:** Near-zero.

**Can run live:** Yes.

---

## Track 5 — Observability and Transparency

### Improvement 10 — LLM Path Transparency Per Section

**Problem**  
After the OpenAI removal, the author loop uses Anthropic → template fallback. When `_call_anthropic()` fails (no key, rate limit, etc.), `_call_llm()` returns `""` and the section generator falls back to a deterministic template. The generated `.tex` file is identical in either case; there is no record of which path was taken.

This matters for the science: an LLM-enhanced abstract is qualitatively different from a template abstract. The author should know.

**Fix**  
Files: `tar_author.py`, `paper_plan.json` structure

1. Add a `llm_sections: dict[str, str]` field to the paper plan. For each section key, record `"llm"`, `"template"`, or `"llm_partial"` (LLM enhanced but fell back for some blocks).

2. In `_call_anthropic()`, add a structured log on success:
   ```python
   print(f"[TAR-Author] LLM: Anthropic wrote {len(result)} chars for section", flush=True)
   ```
   And on failure (existing log is fine — just make sure it reaches the plan):
   ```python
   print(f"[TAR-Author] LLM: Anthropic unavailable — template fallback", flush=True)
   ```

3. In section generators, after calling `_call_llm()`, record in a local `section_llm_log` dict whether the result was LLM or template (check `bool(llm_result)`). Pass this to `_write_author_state()`.

4. After all sections are written, record `llm_sections` in `paper_plan.json`:
   ```json
   {
     "llm_sections": {
       "abstract": "llm",
       "s1_introduction": "template",
       "s3_method": "llm"
     }
   }
   ```

5. Dashboard can display this as a tag next to each section: "AI-drafted" vs "template".

**Risk:** Near-zero — entirely additive logging and metadata.

**Can run live:** Yes.

---

### Additional Observation — Health Check Z7 Stale Waiting Entries

**Note:** The existing `tar_health_check.py` check Z7 (`author_waiting_for_experiments_all_live`) reads `waiting_for_experiments` from `author_state.json`. After Improvement 7 (dual state file consolidation), this field will be removed from `author_state.json`. The health check must be updated to source from `research_director_state.json` instead.

This is a 3-line change to the health check and should be done simultaneously with Improvement 7.

---

## Implementation Sequence

The improvements are grouped into three delivery phases based on risk and dependency:

### Phase 1 — Zero-Risk Infrastructure (do first, no dependencies)
These changes are additive or pure-fix with no behavioural change to experiments:

1. **Improvement 2** — Atomic process_registry.json writes (3 lines changed)
2. **Improvement 4 (Steps 1+2 only)** — Add `_heartbeat_is_fresh()` helper + two-stage liveness check (additive)
3. **Improvement 5A** — Block pending-duplicate IDs in `submit()` (3 lines changed)
4. **Improvement 9** — Scheduler rationale fix (after investigation, ~10 lines)
5. **Improvement 10** — LLM section transparency (additive logging + metadata)

These can all be deployed in a single commit. TAR continues running during deployment; changes take effect on the next respective cycle.

### Phase 2 — Paper Pipeline (highest science impact)
Depends on Phase 1 being stable first:

6. **Improvement 6** — Partial-evidence paper writing (6 targeted changes in tar_author.py)
7. **Improvement 7** — Dual state file consolidation (after Improvement 6 is tested)
8. **Improvement 10 (health check update)** — Update Z7 simultaneously with Improvement 7

After Phase 2, `tcl-autonomous-mechanism-paper` and `tcl-hpc-validation-paper` will begin writing.

### Phase 3 — Reliability and Intelligence
Can run in parallel with Phase 2, or after:

9. **Improvement 1** — Fix research_director_state.json double-write (requires exception to no-touch rule; discuss first)
10. **Improvement 4 (Step 3)** — Wire stall marking from heartbeat staleness
11. **Improvement 5B** — Config-hash fingerprint dedup at submit
12. **Improvement 8** — Paper-blocking urgency boost at daemon submit time

### Phase 4 — Automation (run after Phase 3 is proven stable)
13. **Improvement 5 (auto top-up)** — Initially behind `ENABLE_AUTO_TOPUP` env flag; enable after one full run-through of the new seed pipeline is manually verified

---

## Constraints This Plan Respects

- `_STRICT_REAL_WORLD_FRONTIER_ONLY` is never touched
- Gate B (`publication_allowed` trust-tier check) remains a permanent hard stop; no result that lacks an env snapshot or is quarantined ever appears in prose or tables in any mode
- No existing result files, manifest files, or experiment archives are overwritten or modified by any of these changes (auto top-up modifies archive STATUS on old 3-seed entries only, never result payloads)
- `tar_experiment_library.py` and `tar_research_director.py` are not touched in Phases 1–3 (the double-write fix in Phase 3 requires a one-time exception discussion)
- `generative_director.py` is never touched
- All changes are live-safe — no maintenance window required for any item
- Seed count minimum of 5 is not changed; it is strengthened by the auto top-up mechanism

---

## What This Plan Does NOT Do

- Does not redesign or replace any loop (daemon, watchdog, director, author, scheduler)
- Does not change the experiment trust-tier system
- Does not modify any existing result files or validation manifests
- Does not add new experiment types or science direction
- Does not change the dashboard API structure (only adds to paper_plan.json metadata)
- Does not require changes to `test_tar_comprehensive.py` (though the health check should be run after Phase 1 to confirm no regressions)
- Does not migrate state to SQLite (documented as future work, not in scope)

---

*Plan generated 2026-05-28 from 5-agent deep survey of the live TAR codebase.*
