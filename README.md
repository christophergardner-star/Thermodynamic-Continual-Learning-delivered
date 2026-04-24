# TAR — Thermodynamic Autonomous Researcher

TAR is an autonomous research system that governs its own training runs using thermodynamic principles. Rather than running experiments by hand or relying on fixed schedules, TAR monitors its own activation statistics during training, classifies its thermal state, and adjusts learning behaviour accordingly — then plans the next experiment based on what it finds.

**Author:** Christopher Gardner

---

## What TAR Does

TAR wraps any training payload with three things:

1. **A thermodynamic governor** — monitors the network's activation standard deviation (`sigma`) in real time, classifies the training regime (disordered / critical / ordered), and modulates learning rate based on thermal state.

2. **An autonomous research loop** — a tri-role planning hierarchy (Director, Strategist, Scout) that generates hypotheses, runs experiments, evaluates results against pre-registered criteria, and decides what to investigate next.

3. **Atomic state and reproducibility** — every run is backed by locked manifests, typed schemas, Docker-sandboxed execution, and a persistent knowledge graph so results are traceable and reproducible.

---

## The Thermodynamic Governor

The core insight: a neural network's activation statistics during training behave like a physical system. Early in training, activations are large and variable (disordered — high entropy, exploring). As the network converges, activations settle (ordered — low entropy, consolidating).

TAR quantifies this with the thermal ratio:

```
rho = sigma / sigma_star
```

Where `sigma` is the current activation standard deviation and `sigma_star` is the reference temperature anchored at the start of each task. The regime detector classifies:

| rho | Regime | Action |
|-----|--------|--------|
| > 1.1 | **Disordered** | Boost LR — network is still exploring |
| 0.9–1.1 | **Critical** | Hold LR — near equilibrium |
| < 0.9 | **Ordered** | Reduce LR — consolidating, protect weights |

This mechanism is implemented in [`tar_lab/thermoobserver.py`](./tar_lab/thermoobserver.py).

---

## TCL — Thermodynamic Continual Learning

TCL is the application of the TAR governor to the continual learning problem (sequential tasks, no replay, catastrophic forgetting). It is the primary research payload TAR has been investigating.

**The problem:** Train a network on Task A then Task B — it forgets Task A. Standard fixes (EWC, SI) penalise weight drift. TCL instead uses the thermal signal to detect consolidation and protect weights at the right moment.

**Phase 8 — calibration bug and fix:**

The original `sigma_star` was a rolling median of the current activation window, which meant it co-tracked `sigma` and the ordered threshold was mathematically unreachable. The fix: anchor `sigma_star` from the first 20 batches of each task and freeze it. Now `rho` actually decreases as the network converges.

### Phase 8C Results — Tiny backbone (Split-CIFAR-10, 5 seeds, 15 epochs)

| | Forgetting | Accuracy |
|---|---|---|
| TCL | 0.0918 ± 0.013 | 0.843 |
| SGD | 0.0977 ± 0.022 | 0.837 |

mean delta = −0.006, p = 0.68 — directional but not significant on the 189K-param toy backbone. TCL has **5× lower forgetting variance**.

### Phase 9 Results — ResNet-18, weight penalty (Split-CIFAR-10, 5 seeds, 40 epochs)

Added two components: (1) **warmup guard** — delays anchor collection until after initialisation noise settles; (2) **dimensionality-weighted L2 penalty** — after each task, penalises weight drift scaled by the thermal participation ratio D_PR.

| | Forgetting | Accuracy |
|---|---|---|
| TCL | **0.1119 ± 0.027** | **0.776** |
| SGD | 0.2426 ± 0.065 | 0.686 |

**mean delta = −0.131 (13.1pp), p = 0.0113, Cohen's d = 1.99 — 5/5 seeds TCL better**

**Outcome A: clean win.** The thermodynamic governor produces a statistically significant, large-effect reduction in catastrophic forgetting at realistic scale.

| seed | TCL forg | SGD forg | delta |
|------|----------|----------|-------|
| 42 | 0.130 | 0.192 | −0.063 |
| 0  | 0.090 | 0.209 | −0.118 |
| 1  | 0.115 | 0.190 | −0.075 |
| 2  | 0.145 | 0.332 | −0.187 |
| 3  | 0.080 | 0.290 | −0.210 |

Full investigation log: [`docs/phase8_roadmap.md`](./docs/phase8_roadmap.md)

---

## ASC — Adversarial Self-Consistency

ASC is the language model training method that TAR uses as its reasoning backbone. It adds a consistency loss between a live model and a frozen EMA target, encouraging stable internal representations.

Main class: [`ASCForCausalLM`](./asc_model.py). Supported backbones: GPT-2 (124M–1.5B), DeepSeek Coder, Qwen2.5-Coder.

---

## Quick Start

```bash
# Install
python bootstrap.py

# Run TAR (dry run — no GPU needed)
python tar_cli.py --direct --dry-run --json

# Check current thermal state
python tar_cli.py --direct --check-regime

# Dashboard
streamlit run tar_dashboard.py
```

```bash
# Run the TCL benchmark
python phase8c_validate.py     # single seed, confirms anchor fix
python phase8c_benchmark.py    # 5-seed tiny model
python phase8c_scale.py        # 5-seed ResNet-18
```

```bash
# Run TAR's internal control server
python tar_cli.py --serve --host 127.0.0.1 --port 8765

# In a second terminal, run the HTTP wrapper
uvicorn tar_api:app --host 127.0.0.1 --port 8000
```

Read-only HTTP endpoints exposed by `tar_api.py`:
- `GET /health`
- `GET /status`
- `GET /runtime`
- `GET /projects`
- `GET /queue-health`
- `GET /frontier/status`
- `GET /positioning/reports`
- `GET /comparison/{project_id}`
- `GET /publication-handoff/{project_id}`

Set `TAR_API_KEY` before starting `uvicorn` if you want header-based auth via `X-API-Key`.

---

## Repository Structure

```
tar_lab/
  thermoobserver.py       # Thermodynamic governor — sigma, rho, regime detection
  multimodal_payloads.py  # TCL benchmark (Split-CIFAR-10, TCL vs SGD/EWC/SI)
  schemas.py              # Typed configs, result schemas, governor metrics
  train_template.py       # Docker payload entrypoint
  science_exec.py         # Domain benchmark executors
  science_profiles.py     # Locked domain profiles (ML, NLP, CV, RL, QML)
  scheduler.py            # Persistent study scheduling
  memory/                 # Vector store + research memory

tar_cli.py                # Main TAR command-line interface
TCL_Orchestrator.py       # Orchestration wrapper
dashboard.py / tar_dashboard.py  # Streamlit operator interface

asc_model.py              # ASCForCausalLM
asc_train_full.py         # Canonical ASC training

phase8c_validate.py       # Diagnostic validation script
phase8c_benchmark.py      # 5-seed tiny model benchmark
phase8c_scale.py          # 5-seed ResNet-18 benchmark

docs/
  phase8_roadmap.md       # Phase 8 investigation (calibration bug, results)
  implementation_roadmap.md  # Post-audit remediation roadmap

tar_state/
  comparisons/            # Benchmark result JSONs
  cl_traces/              # Per-epoch diagnostic traces
  manifests/              # Locked run manifests
  recovery.json           # Atomic run state
```

---

## TAR Architecture

**Planning hierarchy:**
- **Director** — sets research goals, evaluates breakthrough claims
- **Strategist** — decomposes goals into experiment plans
- **Scout** — executes runs, reports results back up the chain

**Governor pipeline:**
- Activation telemetry → sigma/rho per parameter group
- Regime classification (disordered / critical / ordered)
- LR modulation and equilibrium gating
- Per-epoch trace written to `tar_state/cl_traces/`

**Reproducibility:**
- Locked dependency manifests under `tar_state/manifests/`
- Docker-only autonomous execution (no host-Python fallback)
- Atomic recovery state — runs can resume after interruption
- Claim-verdict policy (accepted / provisional / rejected / insufficient evidence)

---

## Full TAR CLI

<details>
<summary>All commands</summary>

```bash
# Status and governance
python tar_cli.py --direct --status
python tar_cli.py --direct --check-regime
python tar_cli.py --direct --runtime-status --json
python tar_cli.py --direct --frontier-status --json
python tar_cli.py --direct --sandbox-policy --json
python tar_cli.py --direct --show-manifest --json

# Research loop
python tar_cli.py --direct --chat --message "Analyse current stability" --json
python tar_cli.py --direct --ingest-research --topic "continual learning" --json
python tar_cli.py --direct --verify-last-trial --json
python tar_cli.py --direct --breakthrough-report --json
python tar_cli.py --direct --claim-policy --json
python tar_cli.py --direct --research-decision-log --json

# Problem-driven science
python tar_cli.py --direct --resolve-problem --problem "Investigate barren plateaus in QML" --json
python tar_cli.py --direct --study-problem --problem "Investigate optimisation stability" --json
python tar_cli.py --direct --run-problem-study --json
python tar_cli.py --direct --run-problem-study --use-docker --build-env --json
python tar_cli.py --direct --schedule-problem-study --delay-s 0 --max-runs 1 --json
python tar_cli.py --direct --run-scheduler-once --max-jobs 1 --json

# Inference endpoints
python tar_cli.py --direct --register-checkpoint --checkpoint-name asc-local --model-path /path/to/ckpt --json
python tar_cli.py --direct --build-inference-endpoint --checkpoint-name asc-local --role director --json
python tar_cli.py --direct --start-endpoint --checkpoint-name asc-local --role assistant --wait-for-health --json
python tar_cli.py --direct --list-endpoints --json

# Control
python tar_cli.py --direct --pivot --force --json
python tar_cli.py --direct --panic --json
python tar_cli.py --serve
streamlit run tar_dashboard.py
```

Environment overrides:

```bash
set TAR_DATA_MODE=DOWNLOAD_REAL        # OFFLINE_FALLBACK | CACHED_REAL | DOWNLOAD_REAL
set TAR_LLM_BASE_URL=http://localhost:8000/v1
set TAR_DIRECTOR_MODEL=your-70b-model
set TAR_STRATEGIST_MODEL=your-30b-model
set TAR_SCOUT_MODEL=your-14b-model
set TAR_GPU_INDEX=0
set TAR_CPU_LIMIT=8
set TAR_MEMORY_LIMIT_GB=16
```

</details>

---

## Setup and Tests

```bash
python bootstrap.py           # standard
python bootstrap.py --gpu     # adds GPU stack
python bootstrap.py --dry-run # preview only

python -m pytest tests -q     # 143 tests
```
