# TAR × RunPod Integration Plan
## Cloud GPU Offloading for Long-Running Experiments

**Status:** Planning  
**Goal:** Route expensive experiments (>24h on GTX 1650) to RunPod cloud GPUs automatically.  
**Non-goal:** Replace local execution — cheap/fast experiments stay local.

---

## Why This Matters

The GTX 1650 (4 GB VRAM) is a bottleneck for harder experiments:

| Experiment | Local (GTX 1650) | RTX 4090 @ RunPod | A40 @ RunPod | Cost |
|---|---|---|---|---|
| split_tinyimagenet × 5 seeds | ~100 hours | ~8 hours | ~6 hours | ~$3.50 |
| split_cifar100 × ViT backbone | ~40 hours | ~4 hours | ~3 hours | ~$2.00 |
| cross_backbone_generalization | ~60 hours | ~6 hours | ~4 hours | ~$2.50 |

The queue of 8 pending experiments after the current run = ~400 GPU-hours locally. On RunPod, that's ~40 hours and under $30 total. The local GPU is freed for prototyping and small runs.

---

## Architecture Overview

```
TAR Daemon (local)
  │
  ├── Scheduler.decide()
  │     └── _should_use_runpod(spec) → True if:
  │           • RUNPOD_API_KEY is set
  │           • estimated_runtime_h > RUNPOD_THRESHOLD_H (default 12)
  │           • OR vram_gb > local GPU capacity (3.9 GB)
  │           • OR spec.runtime_context["force_runpod"] = True
  │
  ├── Orchestrator._execute(spec)
  │     └── execution_mode == "runpod"
  │           └── RunPodExecutor.run(spec)
  │                 ├── create_pod()          ← RunPod API
  │                 ├── wait_for_ssh()
  │                 ├── sync_code()           ← rsync repo to pod
  │                 ├── prepare_dataset()     ← download on pod or mount volume
  │                 ├── run_remote_worker()   ← SSH: tar_runpod_worker.py
  │                 │     └── streams stdout  → local log
  │                 ├── poll_progress()       ← rsync progress.json every 30s
  │                 │     └── orch.update_progress()  → dashboard/queue
  │                 ├── sync_result()         ← rsync result.json back
  │                 └── terminate_pod()       ← ALWAYS (finally block)
  │
  └── Result lands in {workspace}/tar_state/experiments/{id}/result.json
        └── Indistinguishable from local run — same schema, same verdict pipeline
```

---

## Files to Create

### 1. `tar_runpod_executor.py` (new — ~400 lines)

The local-side RunPod orchestrator. Handles full pod lifecycle.

```
RunPodExecutor
  ├── __init__(workspace, orch)
  ├── run(spec) → ExperimentResult         # main entry point
  ├── _create_pod(spec) → pod_id
  ├── _wait_for_ssh(pod_id) → ssh_info
  ├── _sync_code(ssh_info)                 # rsync repo → /workspace/repo
  ├── _prepare_dataset(ssh_info, spec)     # download or mount network volume
  ├── _run_worker(ssh_info, spec)          # SSH exec + stdout streaming
  ├── _poll_progress(ssh_info, spec)       # background thread, every 30s
  ├── _sync_result(ssh_info, spec)         # rsync result.json ← pod
  ├── _terminate(pod_id)                   # always in finally block
  └── _cost_watchdog(pod_id, max_h)        # background thread kill timer
```

Key constants (configurable via `tar_state/runpod_config.json`):
```python
RUNPOD_THRESHOLD_H     = 12.0       # experiments longer than this go to RunPod
RUNPOD_GPU_PREFERENCE  = [          # tried in order until one is available
    "NVIDIA A40",
    "NVIDIA RTX 4090",
    "NVIDIA A100-SXM4-80GB",
]
RUNPOD_IMAGE           = "runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04"
RUNPOD_CLOUD_TYPE      = "COMMUNITY"
RUNPOD_WATCHDOG_MULT   = 2.5        # kill pod at estimated_runtime_h × 2.5
RUNPOD_VOLUME_ID       = ""         # populated after one-time setup
RUNPOD_DATACENTER_ID   = ""         # must match volume datacenter
```

**Billing safety — non-negotiable rules baked into the class:**
1. `terminate_pod()` is ALWAYS called in a `finally:` block — no exceptions
2. Watchdog thread kills pod at `estimated_runtime_h × WATCHDOG_MULT` regardless
3. Dry-run mode: set `RUNPOD_DRY_RUN=1` env var — logs all actions, touches nothing

### 2. `tar_runpod_worker.py` (new — ~200 lines)

Runs ON the pod via SSH. Thin wrapper around existing TAR runner code.

```
python tar_runpod_worker.py \
    --experiment-id abc123def456 \
    --dataset split_tinyimagenet \
    --method tcl \
    --seeds 42 0 1 2 3 \
    --epochs 40 \
    --backbone resnet18 \
    --config-overrides '{"key": "val"}' \
    --workspace /workspace \
    --progress-file /workspace/progress.json
```

Responsibilities:
1. Installs dependencies: `pip install -r requirements.txt` (first run only)
2. Sets `HF_HOME`, `TORCH_HOME`, dataset env vars pointing to `/workspace`
3. Downloads dataset if not already present
4. Calls the appropriate runner (`phase17_tinyimagenet`, `phase16_scale_up`, or `generic_cl_runner`)
5. After each seed: writes `progress.json` atomically (os.replace)
6. On completion: writes `result.json` to `/workspace/result_{experiment_id}.json`
7. Exits 0 on success, non-zero on failure

**progress.json schema** (written every seed, polled by local executor):
```json
{
  "experiment_id": "abc123def456",
  "seeds_done": 2,
  "seeds_total": 5,
  "tasks_done": 10,
  "latest_accs": ["0.612", "0.589"],
  "forgetting_so_far": [0.142, 0.167],
  "updated_at": "2026-06-01T12:00:00+00:00"
}
```

### 3. `tar_state/runpod_config.json` (new — config file)

```json
{
  "enabled": true,
  "threshold_runtime_h": 12.0,
  "threshold_vram_gb": 3.9,
  "gpu_preference": ["NVIDIA A40", "NVIDIA RTX 4090", "NVIDIA A100-SXM4-80GB"],
  "cloud_type": "COMMUNITY",
  "image": "runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04",
  "volume_id": "",
  "datacenter_id": "",
  "watchdog_multiplier": 2.5,
  "ssh_key_path": "~/.ssh/id_rsa"
}
```

### 4. `scripts/setup_runpod_volume.py` (new — one-time setup)

Creates a persistent network volume in the target datacenter and pre-downloads
all TAR datasets (CIFAR-10, CIFAR-100, TinyImageNet) onto it. Run once manually.
Writes `volume_id` and `datacenter_id` back into `runpod_config.json`.

Estimated volume size needed: 50 GB  
Estimated setup time: 15 minutes  
Ongoing cost: ~$3.50/month

---

## Files to Modify

### 5. `tar_experiment_preflight.py`

**Change:** Add RunPod routing decision to `prepare()`.

After the existing execution mode detection (around line 186), add:

```python
# Check RunPod routing
_rp_config = _load_runpod_config(workspace)
if _rp_config.get("enabled") and _should_use_runpod(spec, _rp_config):
    report.execution_mode = "runpod"
    report.notes.append(
        f"Routed to RunPod: estimated_runtime_h={spec.estimated_runtime_h:.1f}h "
        f"exceeds threshold={_rp_config['threshold_runtime_h']:.1f}h"
    )
```

New helper functions (add to preflight file):
```python
def _load_runpod_config(workspace: Path) -> dict:
    path = workspace / "tar_state" / "runpod_config.json"
    ...

def _should_use_runpod(spec, config: dict) -> bool:
    if not os.environ.get("RUNPOD_API_KEY"):
        return False  # not configured
    if os.environ.get("RUNPOD_DRY_RUN"):
        return False  # dry-run mode active
    if spec.runtime_context.get("force_runpod"):
        return True   # manual override
    if spec.runtime_context.get("force_local"):
        return False  # manual override
    vram = _spec_vram_budget(spec)
    return (
        spec.estimated_runtime_h > config["threshold_runtime_h"]
        or vram > config["threshold_vram_gb"]
    )
```

### 6. `tar_experiment_orchestrator.py`

**Change 1:** Add RunPod branch in `_execute()` (after line ~1639):
```python
if report.get("execution_mode") == "runpod":
    return self._execute_on_runpod(spec, report)
```

**Change 2:** Add `_execute_on_runpod()` method:
```python
def _execute_on_runpod(self, spec: ExperimentSpec, report: dict) -> ExperimentResult | None:
    from tar_runpod_executor import RunPodExecutor
    executor = RunPodExecutor(workspace=self.workspace, orchestrator=self)
    return executor.run(spec)
```

**Change 3:** `_VRAM_BUDGET` entries — add RunPod-sized estimates for ViT backbone:
```python
"split_tinyimagenet_vit":  8.0,   # ViT-Tiny on TinyImageNet
"split_cifar100_vit":      6.0,   # ViT-Tiny on CIFAR-100
```

### 7. `tar_scheduler.py`

**Change 1:** Add `_should_runpod_route()` check in `decide()`. When RunPod is
configured and a spec exceeds the threshold, the scheduler marks it as
`can_start` even when local VRAM is full (it will run on cloud):

```python
# In decide(), after VRAM check:
if vram_committed + exp_vram > vram_headroom:
    if _runpod_available() and _spec_should_runpod(spec, runpod_config):
        # RunPod takes this — local VRAM doesn't matter
        pass   # do NOT hold; add to can_start
    else:
        hold_reasons.append(HoldReason(...))
        continue
```

**Change 2:** Add RunPod to scheduler rationale narration so the dashboard
shows which experiments are queued for cloud vs local.

### 8. `requirements.txt`

Add:
```
runpod>=1.6.0
paramiko>=3.4.0
```

### 9. `.gitignore`

Add:
```
tar_state/runpod_config.json   # contains volume_id — not secret but env-specific
*.pem
runpod_ssh_key*
```

---

## Implementation Phases

### Phase 1 — Foundation (½ session)

1. Add `runpod` and `paramiko` to `requirements.txt`
2. Create `tar_state/runpod_config.json` schema and loader
3. Add `_load_runpod_config()` and `_should_use_runpod()` to preflight
4. Add `execution_mode = "runpod"` to `PreflightReport` when threshold exceeded
5. Add `_execute_on_runpod()` stub to orchestrator that raises `NotImplementedError`
6. Verify: preflight correctly routes `harder_domain_split_tinyimagenet` to RunPod mode when `RUNPOD_API_KEY` is set

### Phase 2 — Pod Lifecycle (1 session)

Build `tar_runpod_executor.py`:
1. `__init__`, config loading, SSH key validation
2. `_create_pod()` — create with retry on GPU unavailable
3. `_wait_for_ssh()` — poll `get_pod()` until runtime ports appear, timeout 5min
4. `_terminate()` — always in `finally:`
5. `_cost_watchdog()` — background thread, kills at `estimated_h × watchdog_mult`
6. Dry-run mode: log all API calls, no real requests

Verify: end-to-end pod create → wait → terminate with dry-run mode and a real API call.

### Phase 3 — Code & Dataset Sync (½ session)

1. `_sync_code()` — rsync entire repo to `/workspace/repo` on pod
   - Exclude: `tar_runs/`, `tar_state/`, `tool_envs/`, `.git/`, `*.pyc`
   - Include: all `.py` files, `requirements.txt`, `configs/`
2. `_prepare_dataset()`:
   - If `runpod_config.volume_id` is set: pod was created with volume mounted, dataset already there
   - Otherwise: SSH exec `python -c "from datasets import load_dataset; load_dataset(...)"`
3. `scripts/setup_runpod_volume.py` — one-time volume creation + dataset download

### Phase 4 — Remote Worker (½ session)

Build `tar_runpod_worker.py`:
1. Argument parsing
2. `setup_environment()` — pip install, env vars, PYTHONPATH
3. `run_experiment()` — delegate to existing runner functions unchanged:
   - `phase17_tinyimagenet.run_one_seed()` for TinyImageNet
   - `phase16_scale_up.run_one_seed()` for CIFAR-100
   - `generic_cl_runner.run_generic_benchmark()` for others
4. After each seed: atomic write to `/workspace/progress.json`
5. After all seeds: write `result_{experiment_id}.json`
6. Exit codes: 0=success, 1=training error, 2=dataset error, 3=environment error

### Phase 5 — Progress Bridge (½ session)

1. `_run_worker()` in executor: SSH exec + stream stdout to local log
2. `_poll_progress()` — background thread (every 30s):
   ```python
   while running:
       rsync progress.json from pod
       orch.update_progress(spec.id, progress_data)
       time.sleep(30)
   ```
3. Wire into `run(spec)` so dashboard shows live seed progress from RunPod
4. Wire result sync after remote script exits:
   - rsync `result_{experiment_id}.json` → local result path
   - Parse and return as `ExperimentResult`

### Phase 6 — Scheduler Integration (¼ session)

1. Add RunPod routing awareness to `decide()` — bypass local VRAM check for routed specs
2. Update scheduler rationale narration to mention RunPod queue
3. Add `runpod_queued_ids` to daemon state output

### Phase 7 — Testing & Hardening (½ session)

1. Dry-run full flow end-to-end (no actual RunPod API call)
2. Live test with a SMALL experiment first: `permuted_mnist` (1.5 GB VRAM, ~2h)
   - Verify pod creates, code syncs, experiment runs, result returns
   - Verify pod is terminated (check RunPod console)
   - Verify result.json is identical schema to local run
3. Test failure modes:
   - Pod creation fails (GPU unavailable) → retry with next GPU type
   - SSH timeout → terminate and mark spec stalled (retry next cycle)
   - Training error on pod → result written with ERROR verdict
   - Watchdog fires → mark spec stalled, pod terminated
4. Run `tar_health_check.py` — all Y/Z checks should still pass
5. Run `test_tar_comprehensive.py` — all 264 checks pass

---

## Safety Design

### Cost Safeguards (belt-and-braces)

```
Layer 1: finally block          Always terminate_pod() even on exception
Layer 2: Watchdog thread        Kill at estimated_h × 2.5 regardless
Layer 3: RunPod account limit   Set credit limit in RunPod web console
Layer 4: RUNPOD_ENABLED flag    TAR_RUNPOD_ENABLED=0 disables cloud routing
Layer 5: dry-run mode           RUNPOD_DRY_RUN=1 logs without executing
```

### Result Integrity

- RunPod results go through the SAME `_build_result()` → verdict pipeline as local
- Gate B (`publication_allowed`) applies identically — no special treatment for cloud results
- `result.json` has a `"execution_backend": "runpod"` field for provenance tracking
- All cloud results require env snapshot (same trust-tier rules as local)

### RAIL 3 (Manifest Gate)

RunPod execution is subject to the same manifest gate as local execution:
- `requires_manifest=True` inherited by the RunPod executor
- If the active manifest doesn't cover the experiment, RunPod execution is blocked
- No bypass — cloud is not an escape hatch from governance

---

## Configuration: How to Enable

After Phase 7 is complete and tested:

1. **Install deps:** `pip install runpod paramiko`

2. **Set API key:**
   ```powershell
   $env:RUNPOD_API_KEY = "your_key_here"
   # Or add to system environment variables permanently
   ```

3. **Run one-time volume setup:**
   ```powershell
   python scripts/setup_runpod_volume.py
   # Creates 50GB network volume, downloads datasets, writes volume_id to runpod_config.json
   ```

4. **Enable in config:**
   ```json
   // tar_state/runpod_config.json
   { "enabled": true, "threshold_runtime_h": 12.0 }
   ```

5. **Restart TAR** — any experiment with `estimated_runtime_h > 12` will auto-route to RunPod.

To disable: set `"enabled": false` or `$env:RUNPOD_DRY_RUN = "1"`.

---

## What This Does NOT Change

- `_STRICT_REAL_WORLD_FRONTIER_ONLY` — untouched
- Gate B (publication_allowed) — applies to cloud results identically
- Manifest gate (RAIL 3) — cloud execution requires same manifest approval
- `tar_research_director.py` — untouched
- `tar_experiment_library.py` — untouched
- Result schema — identical, cloud adds only `execution_backend` provenance field
- Local execution — unchanged, still default for experiments under threshold

---

## Estimated Effort

| Phase | Work | Time |
|-------|------|------|
| 1 — Foundation | Preflight routing, config schema | ½ session |
| 2 — Pod lifecycle | Executor create/wait/terminate | 1 session |
| 3 — Sync | Code + dataset rsync | ½ session |
| 4 — Remote worker | tar_runpod_worker.py | ½ session |
| 5 — Progress bridge | Live seed updates + result retrieval | ½ session |
| 6 — Scheduler | Routing awareness + narration | ¼ session |
| 7 — Testing | Dry-run, live test, hardening | ½ session |
| **Total** | | **~4 sessions** |

---

## Quick Cost Reference

Pricing as of mid-2025 (community cloud, on-demand):

| GPU | VRAM | $/hr | TinyImageNet × 5 seeds | CIFAR-100 × 5 seeds |
|-----|------|------|------------------------|---------------------|
| RTX 4090 | 24 GB | ~$0.44 | ~$3.50 (8h) | ~$1.76 (4h) |
| A40 | 48 GB | ~$0.64 | ~$3.84 (6h) | ~$1.92 (3h) |
| A100 PCIe | 40 GB | ~$1.19 | ~$5.95 (5h) | ~$2.38 (2h) |
| A100 SXM | 80 GB | ~$1.89 | ~$7.56 (4h) | ~$3.78 (2h) |

**Recommendation:** RTX 4090 community cloud for most TAR experiments. 24 GB VRAM handles everything in the current queue. ~$0.44/hr is comparable to electricity costs of running a PC continuously.
