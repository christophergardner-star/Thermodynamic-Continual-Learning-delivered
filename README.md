# Thermodynamic Continual Learning Delivered (ASC-TAR)

This repository now contains a coherent research stack built around two core methods:

- `ASC`: Adversarial Self-Consistency for causal language models
- `TCL`: Thermodynamic Continual Learning for catastrophic-forgetting control

Around those core methods, the repo also includes the intended local coding and researcher workflow:

- coding-backbone ASC fine-tuning for DeepSeek Coder and Qwen2.5-Coder
- local corpus preparation and lightweight coding evaluation
- local serving helpers for Continue / OpenAI-compatible clients
- `Cruxy`, a researcher loop that stores traces and classifies claims as fact, measured result, inference, or hypothesis
- `TAR`, a TCL-Autonomous Researcher foundation with typed policy JSON, fail-fast thermodynamic governance, atomic recovery state, and a local operator interface

## Repository Contract

- The research primitives are the root modules: [`asc_model.py`](./asc_model.py) and [`tcl.py`](./tcl.py).
- The coding stack is the root script surface around those primitives.
- The researcher stack is also root-level and intentionally lightweight.
- This is not the `EPTO_-` package repo. No `epto_sdk_v54` package tree is required here.

## Model Families

### ASC standalone family

Named presets exposed by `ASCConfig.for_size(...)`:

| Size | Backbone |
|---|---|
| `124M` | `gpt2` |
| `355M` | `gpt2-medium` |
| `774M` | `gpt2-large` |
| `1558M` | `gpt2-xl` |

For offline smoke tests, ASC also supports a tiny random GPT-2 backbone via
`base_model_name="__tiny_gpt2__"`.

### Coding fine-tune backbones

The coding fine-tune entrypoints are:

- [`coding_asc_finetune.py`](./coding_asc_finetune.py)
- [`deepseek_asc_finetune.py`](./deepseek_asc_finetune.py)

Supported preset registry:

| Preset | HF ID |
|---|---|
| `1.3b` | `deepseek-ai/deepseek-coder-1.3b-base` |
| `6.7b` | `deepseek-ai/deepseek-coder-6.7b-base` |
| `33b` | `deepseek-ai/deepseek-coder-33b-base` |
| `1.3b-instruct` | `deepseek-ai/deepseek-coder-1.3b-instruct` |
| `6.7b-instruct` | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| `33b-instruct` | `deepseek-ai/deepseek-coder-33b-instruct` |
| `qwen-1.5b` | `Qwen/Qwen2.5-Coder-1.5B` |
| `qwen-7b` | `Qwen/Qwen2.5-Coder-7B` |
| `qwen-14b` | `Qwen/Qwen2.5-Coder-14B` |
| `qwen-32b` | `Qwen/Qwen2.5-Coder-32B` |
| `qwen-1.5b-instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| `qwen-7b-instruct` | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `qwen-14b-instruct` | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| `qwen-32b-instruct` | `Qwen/Qwen2.5-Coder-32B-Instruct` |

## Core Methods

### ASC

ASC wraps a causal LM backbone with:

- a small `LatentWarp` MLP
- a frozen EMA target model
- dual training losses: task loss + consistency loss

The main model class is [`ASCForCausalLM`](./asc_model.py).

### TCL

TCL tracks per-parameter gradient-energy EMAs during each task and stores task checkpoints plus normalized importance maps. Later tasks apply an elastic penalty to reduce forgetting.

The main types are:

- `ThermalImportance`
- `ThermalMemory`
- `ThermalCheckpoint`
- `TCLRegularizer`
- `TCLTrainer`

All are defined in [`tcl.py`](./tcl.py).

## Coding Stack

### Prepare a corpus

```bash
python stack_data_prep.py --inputs ./my_code_dir --output ./data/corpus.jsonl
```

### Smoke-train ASC on a tiny offline backbone

```bash
python coding_asc_finetune.py --model tiny --dataset synthetic --max_steps 2
```

### Dry-run a Qwen coding job

```bash
python coding_asc_finetune.py --model qwen-14b --dataset synthetic --dry_run
```

### Summarize coding benchmark predictions

```bash
python eval_coding.py --predictions_jsonl ./predictions.jsonl -k 1
```

### Print Continue config or status payload

```bash
python serve_local.py --print_continue
python serve_local.py --print_status --workspace ./coding_ai_out/qwen2.5-coder-14b
```

## Researcher Stack

### Run the Cruxy loop

```bash
python researcher_agent.py --db research_db.sqlite --session smoke --dry_run
```

### Export strong traces for later training

```bash
python self_train.py --db research_db.sqlite --output research_traces.jsonl --min_score 0.5
```

### Blend researcher traces into a corpus

```bash
python build_researcher_dataset.py --db research_db.sqlite --output ./data/researcher_corpus.jsonl
```

## TAR Lab

The repository now includes a TAR foundation layer for running a safer autonomous lab loop around TCL/ASC experiments.

Current TAR surface:

- tri-role planning hierarchy: Director, Strategist, Scout
- thermodynamic governor with `E`, `sigma`, `rho`, `||grad||`, activation-space `D_PR`, and equilibrium gating
- atomic persistence via `tar_state/recovery.json` and `tar_state/knowledge_graph.json`
- `logs/thermo_metrics.jsonl` and `logs/activity_audit.log`
- dual-stream data preparation into `tar_state/data` with anchor and research manifests
- local Chroma-backed vector memory with self-correction notes and trial retrieval
- live research ingestion from arXiv plus optional RSS feeds
- verification runs with seed sweeps, ablations, and calibration / ECE summaries
- formal breakthrough reports with novelty, stability, calibration, and supporting research memory
- Docker command composition with target caps of `40GB` RAM, `12` CPU cores, `1` GPU, automatically clamped to the Docker engine's actual limits
- NVIDIA power/thermal preparation through `nvidia-smi -pl` and `nvidia-smi -gtt`
- direct CLI, local socket control server, Streamlit sidecar, typed chat mode, and dry-run coverage
- OpenAI-compatible local LLM wiring for Director, Strategist, and Scout with schema-repair retries
- mounted training payload execution through `python -m tar_lab.train_template`
- stateless activation telemetry in [`tar_lab/thermoobserver.py`](./tar_lab/thermoobserver.py)
- optional wake-word voice control through [`tar_lab/voice.py`](./tar_lab/voice.py)

Primary entrypoints:

```bash
python tar_cli.py --direct --dry-run --json
python tar_cli.py --direct --status
python tar_cli.py --direct --check-regime
python tar_cli.py --direct --chat --message "Analyze the current stability" --json
python tar_cli.py --direct --ingest-research --topic "current ai problems" --json
python tar_cli.py --direct --verify-last-trial --json
python tar_cli.py --direct --breakthrough-report --json
python tar_cli.py --direct --live-docker-test --json
python tar_cli.py --serve
streamlit run dashboard.py
streamlit run tar_dashboard.py
```

If Docker is installed but not visible on `PATH`, TAR also honors:

```bash
set TAR_DOCKER_BIN=C:\path\to\docker.exe
```

TAR also honors:

```bash
set TAR_CPU_LIMIT=8
set TAR_MEMORY_LIMIT_GB=6
set TAR_GPU_INDEX=0
```

`tar_cli.py` defaults `--workspace` to the repository directory, so invoking it by full path works from outside the repo.

Local hierarchy configuration:

```bash
set TAR_LLM_BASE_URL=http://localhost:8000/v1
set TAR_DIRECTOR_MODEL=your-70b-model
set TAR_STRATEGIST_MODEL=your-30b-model
set TAR_SCOUT_MODEL=your-14b-model
set TAR_GPU_INDEX=1
```

Per-role overrides are also supported:

```bash
set TAR_DIRECTOR_BASE_URL=http://localhost:11434/v1
set TAR_DIRECTOR_API_KEY=local
```

Control commands:

```bash
python tar_cli.py --direct --pivot --force --json
python tar_cli.py --direct --explain --json
python tar_cli.py --direct --panic --json
python tar_cli.py --direct --chat --listen --wake-word lab --json
```

Dry-run scope:

- validates the policy JSON contracts
- composes the Docker launch command without requiring Docker
- exercises recovery and knowledge-graph persistence
- prepares `/data/anchor` and `/data/research` manifests for the payload
- indexes metric, knowledge-graph, and self-correction history into the vector vault
- drives the fail-fast governor on mock thermodynamic data, including `D_PR` and equilibrium state
- verifies Pivot-Force after repeated fail-fast outcomes

Live Docker smoke path:

- pulls `pytorch/pytorch:latest`
- mounts the repository into `/workspace`
- mounts `tar_state/data` into `/data`
- bootstraps the minimal payload dependency set inside the container and executes `python -m tar_lab.train_template`
- applies the TAR runtime caps and GPU selection through the Docker runner
- probes GPU visibility with `nvidia-smi -L` before the payload launch
- writes thermodynamic metrics to `/workspace/logs/thermo_metrics.jsonl`

## Validation

Current validated surface:

- ASC core model tests
- ASC CPU smoke tests
- TCL unit tests
- coding stack surface tests
- researcher stack tests
- TAR dry-run, live Docker path, memory, research-ingest, verification, and breakthrough-report tests

Run the suite with:

```bash
python -m pytest tests -q
```

Current expected count:

- `93` tests

Important boundary:

- The ASC/TCL core is tested directly.
- The coding and researcher layers are currently lightweight local orchestration surfaces, not large-scale benchmark claims.
- Large-scale DeepSeek/Qwen training results are not claimed by this README unless you run them yourself.

## Files

| File | Purpose |
|---|---|
| [`asc_model.py`](./asc_model.py) | ASC config and model |
| [`asc_train.py`](./asc_train.py) | original ASC training script |
| [`asc_train_cpu.py`](./asc_train_cpu.py) | CPU ASC smoke run |
| [`asc_train_full.py`](./asc_train_full.py) | full ASC training script |
| [`asc_vs_baseline.py`](./asc_vs_baseline.py) | ASC baseline comparison |
| [`tcl.py`](./tcl.py) | thermodynamic continual learning |
| [`coding_asc_finetune.py`](./coding_asc_finetune.py) | canonical coding fine-tune entrypoint |
| [`deepseek_asc_finetune.py`](./deepseek_asc_finetune.py) | coding fine-tune implementation with DeepSeek/Qwen registry |
| [`stack_data_prep.py`](./stack_data_prep.py) | local corpus preparation |
| [`eval_coding.py`](./eval_coding.py) | coding eval helpers |
| [`serve_local.py`](./serve_local.py) | local serving helpers and status payload builder |
| [`research_database.py`](./research_database.py) | persistent research trace store |
| [`researcher_agent.py`](./researcher_agent.py) | Cruxy research loop |
| [`self_train.py`](./self_train.py) | export high-quality research traces |
| [`build_researcher_dataset.py`](./build_researcher_dataset.py) | build researcher corpus |
| [`docs/research_status_panel.html`](./docs/research_status_panel.html) | simple status-panel source |
| [`tar_cli.py`](./tar_cli.py) | TAR command and control CLI |
| [`TCL_Orchestrator.py`](./TCL_Orchestrator.py) | TAR orchestration wrapper |
| [`dashboard.py`](./dashboard.py) | TAR Streamlit sidecar |
| [`tar_dashboard.py`](./tar_dashboard.py) | TAR Streamlit sidecar alias with regime telemetry |
| [`tar_lab/`](./tar_lab) | TAR schemas, governor, state, hierarchy, control, and hardware adapters |
| [`tar_lab/train_template.py`](./tar_lab/train_template.py) | mounted container payload for live TAR runs |
| [`tar_lab/thermoobserver.py`](./tar_lab/thermoobserver.py) | activation telemetry, participation-ratio `D_PR`, and equilibrium tracking |
| [`tar_lab/data_manager.py`](./tar_lab/data_manager.py) | TAR dual-stream dataset preparation and sharding |
| [`tar_lab/memory/`](./tar_lab/memory) | TAR vector memory and background indexing support |

## Author

Christopher Gardner
