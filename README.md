# Thermodynamic Continual Learning Delivered (ASC-TAR + TCL)

This repository contains a coherent research stack built around two core methods:

- `ASC`: Adversarial Self-Consistency for causal language models
- `TCL`: Thermodynamic Continual Learning for catastrophic-forgetting control

Around those core methods, the repo also includes the intended local coding and researcher workflow:

- coding-backbone ASC fine-tuning for DeepSeek Coder and Qwen2.5-Coder
- local corpus preparation and lightweight coding evaluation
- local serving helpers for Continue / OpenAI-compatible clients
- `Cruxy`, a researcher loop that stores traces and classifies claims as fact, measured result, inference, or hypothesis
- `TAR`, a TCL-Autonomous Researcher foundation with typed policy JSON, fail-fast thermodynamic governance, atomic recovery state, and a local operator interface

## Scientific Status

The repository now distinguishes between canonical, experimental, and
quarantined entrypoints.

- Canonical ASC training path: [`asc_train_full.py`](./asc_train_full.py)
- Canonical TAR payload path: [`tar_lab/train_template.py`](./tar_lab/train_template.py)
- Experimental coding ASC path: [`deepseek_asc_finetune.py`](./deepseek_asc_finetune.py)
- Quarantined legacy ASC scripts: [`asc_train.py`](./asc_train.py) and [`asc_train_cpu.py`](./asc_train_cpu.py)

The formal post-audit remediation roadmap is in
[`docs/implementation_roadmap.md`](./docs/implementation_roadmap.md).

Locked reproducibility means every required dependency is pinned to an exact
version. If TAR cannot resolve a required package version locally, it now
refuses the lock and records the unresolved dependency instead of emitting a
best-effort manifest.

## Repository Contract

- The research primitives are the root modules: [`asc_model.py`](./asc_model.py) and [`tcl.py`](./tcl.py).
- The coding stack is the root script surface around those primitives.
- The researcher stack is also root-level and intentionally lightweight.
- This is not the `EPTO_-` package repo. No `epto_sdk_v54` package tree is required here.

## Local Bootstrap

`git pull` should not auto-run installers. Git does not safely distribute active
hooks, and TAR should not mutate a machine implicitly just because the repo was
updated.

Instead, the repo now has a one-command bootstrap path for local setup after
pull or clone:

```powershell
python .\bootstrap.py
```

That creates `.venv` by default, upgrades `pip`, and installs the base local
stack from [`requirements.txt`](./requirements.txt).

Optional bootstrap modes:

```powershell
python .\bootstrap.py --platform-extra
python .\bootstrap.py --gpu
```

- `--platform-extra` adds optional platform-sensitive packages from
  [`requirements_platform_extra.txt`](./requirements_platform_extra.txt)
- `--gpu` installs the GPU-serving stack from
  [`requirements_gpu.txt`](./requirements_gpu.txt)

If you want to inspect the plan without installing anything:

```powershell
python .\bootstrap.py --dry-run
```

Important boundary:

- these requirements files are a convenience install surface for local
  development and validation
- TAR's reproducible runtime truth still comes from its locked manifest builders
  in [`tar_lab/reproducibility.py`](./tar_lab/reproducibility.py)

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

### Canonical ASC training

```bash
python asc_train_full.py --size 124M --max_steps 100
```

### Tiny offline ASC smoke run

```bash
python coding_asc_finetune.py --model tiny --dataset synthetic --max_steps 2
```

### Dry-run a Qwen coding job

```bash
python coding_asc_finetune.py --model qwen-14b --dataset synthetic --dry_run
```

Important boundary:

- [`deepseek_asc_finetune.py`](./deepseek_asc_finetune.py) remains
  experimental until masking, device placement, and scaling are corrected.
- [`asc_train.py`](./asc_train.py) and [`asc_train_cpu.py`](./asc_train_cpu.py)
  are quarantined because they do not implement valid adversarial ASC.

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
- rolling-window thermodynamic estimation with smoothed `D_PR`, `sigma`, regime confidence, and warmup-aware gating
- atomic persistence via `tar_state/recovery.json` and `tar_state/knowledge_graph.json`
- `logs/thermo_metrics.jsonl` and `logs/activity_audit.log`
- dual-stream data preparation into `tar_state/data` with anchor and research manifests
- explicit data modes: `OFFLINE_FALLBACK`, `CACHED_REAL`, and `DOWNLOAD_REAL`
- hard data-starvation gates for research-grade runs: real-dataset failures now raise instead of silently demoting to synthetic corpora or `HashTokenizer`
- manifest-level provenance for every prepared stream, including dataset name/subset/split, tokenizer hash, tokenizer vocab size, sampling strategy, and integrity status
- `--status` now reports current run data purity so fallback/plumbing runs cannot masquerade as research evidence
- local Chroma-backed vector memory with research-gated semantic retrieval, dense-plus-lexical candidate merge, scientific reranking, claim clustering, and contradiction tracking
- live research ingestion from arXiv plus optional RSS feeds
- full-document paper ingestion with page-aware sections, claim spans, citation edges, tables, figures, and OCR fallback for scanned PDFs when the local OCR/render stack is available
- verification runs with seed sweeps, ablations, and calibration / ECE summaries
- formal breakthrough reports with novelty, stability, calibration, and supporting research memory
- problem-domain routing through locked science profiles for quantum ML, RL, CV, NLP, graph ML, deep learning, and general ML
- reproducible science-environment bundle generation with Dockerfile, requirements profile, and study-plan artifacts
- benchmark-backed problem-study execution for generic ML, deep learning, computer vision, graph ML, NLP, reinforcement learning, and quantum ML
- truthful benchmark adapters where available: sklearn breast-cancer and digits suites, NetworkX Karate Club, and PennyLane-backed QML depth/init/noise execution when installed; named canonical suites that are not yet executor-aligned remain registered but are refused rather than overstated
- persistent long-run problem scheduling with queued study runs, repeat intervals, and single-cycle scheduler execution
- locked payload and science-environment manifests with dependency hashes, source-tree fingerprints, and run-manifest hashes under `tar_state/manifests`
- no runtime package mutation in the main payload path: Docker runs now require locked image and run manifests
- runtime daemon semantics with lease acquisition, retry/backoff, stale-lease cleanup, terminal-failure alerts, and heartbeat state in `tar_state/runtime_heartbeat.json`
- Docker-only sandbox execution for autonomous Python tasks, with explicit sandbox policy, artifact capture, and no host-Python fallback
- Docker command composition with target caps of `40GB` RAM, `12` CPU cores, `1` GPU, automatically clamped to the Docker engine's actual limits
- NVIDIA power/thermal preparation through `nvidia-smi -pl` and `nvidia-smi -gtt`
- direct CLI, local socket control server, Streamlit sidecar, typed chat mode, and dry-run coverage
- OpenAI-compatible local LLM wiring for Director, Strategist, and Scout with schema-repair retries
- managed inference endpoints with checkpoint registry, start/stop/restart lifecycle, retained stdout/stderr logs under `tar_state/endpoints/<endpoint_name>/`, health checks, explicit role assignment for `director`, `strategist`, `scout`, and `assistant`, and explicit `trust_remote_code` policy instead of silent defaults
- evidence-grounded research planning with typed evidence bundles, contradiction reviews, and hypothesis records persisted into TAR state
- machine-readable claim-acceptance policy with verdict classes `accepted`, `provisional`, `rejected`, `insufficient_evidence`, and `contradicted`
- research decision logging so operator-facing chat and study flows retain evidence traces, confidence, contradictions, and selected next actions
- mounted training payload execution through `python -m tar_lab.train_template`, now defaulting to a real manifest-backed ASC text backend and reserving `toy_anchor` for explicit control-path tests
- default research-target configuration for the payload now points at `deepseek-ai/deepseek-coder-1.3b-base` with LoRA-style adapter mode, while dry-run control paths fall back to `__tiny_gpt2__`
- multi-modal backend registry entries for `asc_cv`, `asc_rl`, and `asc_qml`, each carrying the mandatory `D_PR`, `sigma`, and `rho` governor contract
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
python tar_cli.py --direct --resolve-problem --problem "Investigate barren plateaus in quantum AI" --json
python tar_cli.py --direct --prepare-science-env --problem "Investigate barren plateaus in quantum AI" --json
python tar_cli.py --direct --study-problem --problem "Investigate barren plateaus in quantum AI" --json
python tar_cli.py --direct --list-benchmarks --profile-id natural_language_processing --benchmark-tier canonical --json
python tar_cli.py --direct --benchmark-status --profile-id quantum_ml --benchmark-tier canonical --json
python tar_cli.py --direct --study-problem --problem "Investigate optimization stability in deep learning" --benchmark-tier validation --json
python tar_cli.py --direct --study-problem --problem "Investigate optimization stability in deep learning" --benchmark-tier canonical --canonical-only --no-proxy-benchmarks --json
python tar_cli.py --direct --run-problem-study --json
python tar_cli.py --direct --run-problem-study --use-docker --build-env --json
python tar_cli.py --direct --schedule-problem-study --delay-s 0 --max-runs 1 --json
python tar_cli.py --direct --scheduler-status --json
python tar_cli.py --direct --run-scheduler-once --max-jobs 1 --json
python tar_cli.py --direct --frontier-status --json
python tar_cli.py --direct --runtime-status --json
python tar_cli.py --direct --prepare-payload-env --json
python tar_cli.py --direct --rebuild-locked-image --json
python tar_cli.py --direct --show-manifest --json
python tar_cli.py --direct --list-experiment-backends --json
python tar_cli.py --direct --run-runtime-cycle --max-jobs 1 --json
python tar_cli.py --direct --list-alerts --json
python tar_cli.py --direct --retry-failed-job --schedule-id <schedule_id> --json
python tar_cli.py --direct --cancel-job --schedule-id <schedule_id> --json
python tar_cli.py --direct --sandbox-policy --json
python tar_cli.py --direct --ingest-papers --paper-path C:\path\to\paper.pdf --json
python tar_cli.py --direct --register-checkpoint --checkpoint-name asc-local --model-path C:\path\to\checkpoint --json
python tar_cli.py --direct --build-inference-endpoint --checkpoint-name asc-local --role director --trust-remote-code --json
python tar_cli.py --direct --list-checkpoints --json
python tar_cli.py --direct --list-endpoints --json
python tar_cli.py --direct --start-endpoint --checkpoint-name asc-local --role assistant --wait-for-health --json
python tar_cli.py --direct --endpoint-health --endpoint-name assistant-asc-local
python tar_cli.py --direct --assign-role --role strategist --checkpoint-name asc-local --json
python tar_cli.py --direct --claim-policy --json
python tar_cli.py --direct --claim-verdict --trial-id <trial_id> --json
python tar_cli.py --direct --research-decision-log --json
python tar_cli.py --direct --live-docker-test --json
python tar_cli.py --serve
streamlit run dashboard.py
streamlit run tar_dashboard.py
```

Workstream 7 operator contract:

- research chat and problem-study flows now carry evidence traces, contradiction warnings, and typed hypotheses instead of heuristic-only summaries
- breakthrough promotion is now coupled to explicit claim-verdict policy rather than a soft narrative summary alone
- endpoint health, role assignment, latest claim verdict, and recent research decisions are surfaced in both CLI status and the Streamlit dashboard
- managed endpoint records now persist trust policy, manifest path, and stdout/stderr log paths so failed starts are diagnosable without rerunning them

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

Workstream 6 runtime contract:

- payload and science runs are expected to execute from locked image and run manifests
- runtime scheduling now uses leases, bounded retries, backoff, stale-run cleanup, and alert records
- autonomous code execution is sandboxed through Docker only; TAR no longer treats host-Python fallback as a valid runtime path
- production runtime mounts are now read-only by default for `/workspace`, with explicit writable paths limited to `/workspace/tar_runs`, `/workspace/logs`, and `/workspace/anchors`
- science-bundle Docker runs now mount the repository read-only and grant write access only to the bundle artifact directory
- `--status`, `--runtime-status`, `--sandbox-policy`, and the dashboard now surface image identity, manifest hash, sandbox profile, read-only mounts, writable mounts, alert count, and lease state

To control TAR's data-grounding policy:

```bash
set TAR_DATA_MODE=OFFLINE_FALLBACK
set TAR_DATA_MODE=CACHED_REAL
set TAR_DATA_MODE=DOWNLOAD_REAL
set TAR_TOKENIZER_ID=deepseek-ai/deepseek-coder-1.3b-base
```

`tar_cli.py` defaults `--workspace` to the repository directory, so invoking it by full path works from outside the repo.

The current frontier architecture contract is documented in [`docs/frontier_stack_plan.md`](./docs/frontier_stack_plan.md). That document explains what is implemented now versus what still requires stronger benchmarks, external data, or production infrastructure.

Literature and retrieval contract:

- `--ingest-papers` now parses full PDF documents with page-aware claims, sections, tables, figures, bibliography extraction, and OCR capability reporting
- research-grade literature retrieval requires a real semantic model; TAR will refuse literature-grounded research queries if the semantic path is unavailable
- retrieval now runs a two-stage pipeline: dense and lexical candidate generation followed by a scientific reranker
- contradiction metadata and evidence traces are attached to paper-claim retrieval hits so planning/reporting can cite paper IDs and page numbers

Science profiles live under [`science_profiles/`](./science_profiles) and define:

- pip packages
- apt packages
- validation imports
- benchmark targets
- named benchmark suites for `smoke`, `validation`, and `canonical` tiers
- metric hooks
- experiment templates
- locked install policy for container-only execution

Benchmark contract:

- `smoke`: laptop-safe proxy or reduced local benchmark paths for plumbing and rapid validation
- `validation`: real named local or cached benchmark slices that preserve benchmark identity without claiming external comparability
- `canonical`: named external benchmark suites that only count as literature-comparable when the executor is benchmark-aligned; otherwise TAR marks them unsupported and refuses the run
- status, study plans, and execution reports now surface benchmark IDs, benchmark names, requested tier, actual executed tiers, benchmark truth status, benchmark alignment, and canonical comparability
- `--canonical-only --no-proxy-benchmarks` enforces strict refusal semantics rather than silent downgrade

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
- mounts the repository into `/workspace` as read-only
- mounts `tar_state/data` into `/data` as read-only
- mounts `/workspace/tar_runs`, `/workspace/logs`, and `/workspace/anchors` as the only writable payload paths
- bootstraps the minimal payload dependency set inside the container and executes `python -m tar_lab.train_template`
- applies the TAR runtime caps and GPU selection through the Docker runner
- probes GPU visibility with `nvidia-smi -L` before the payload launch
- writes thermodynamic metrics to `/workspace/logs/thermo_metrics.jsonl`

Problem-driven science flow:

- resolve a natural-language problem to a domain profile
- generate a locked environment bundle under `tar_state/science_envs/...`
- write `requirements-profile.txt`, `Dockerfile`, and `study_plan.json`
- optionally build the environment image and run `python -m tar_lab.problem_runner`
- persist the resulting study plan and execution report into TAR memory and audit history
- queue the study for later execution with a due time, repeat interval, and run budget

Benchmark-backed domain executors:

- `deep_learning`: tiny supervised optimization and scaling probes with real loss, accuracy, calibration, gradient, and representation-rank metrics
- `natural_language_processing`: grounded retrieval QA plus length-generalization evaluation with real ROUGE, hallucination, perplexity, and calibration signals
- `reinforcement_learning`: policy-gradient exploration and offline-to-online transfer probes with real return, entropy, sample-efficiency, and seed-variance signals
- `quantum_ml`: PennyLane-backed canonical depth, initialization, and noisy-trainability probes, with analytic stand-ins reserved for explicit smoke-only paths

## Validation

Current validated surface:

- ASC core model tests
- ASC CPU smoke tests
- TCL unit tests
- coding stack surface tests
- researcher stack tests
- TAR dry-run, live Docker path, memory, research-ingest, verification, and breakthrough-report tests
- science-profile routing, environment bundle generation, and problem-study planning tests
- problem-study execution, persistence, execution-memory indexing, and scheduler tests

Run the suite with:

```bash
python -m pytest tests -q
```

Current expected count:

- `143` tests

Important boundary:

- The ASC/TCL core is tested directly.
- The coding and researcher layers are currently lightweight local orchestration surfaces, not large-scale benchmark claims.
- Large-scale DeepSeek/Qwen training results are not claimed by this README unless you run them yourself.

## Files

| File | Purpose |
|---|---|
| [`asc_model.py`](./asc_model.py) | ASC config and model |
| [`asc_train.py`](./asc_train.py) | quarantined legacy ASC script; intentionally fails fast |
| [`asc_train_cpu.py`](./asc_train_cpu.py) | quarantined legacy ASC CPU script; intentionally fails fast |
| [`asc_train_full.py`](./asc_train_full.py) | canonical ASC training script |
| [`asc_vs_baseline.py`](./asc_vs_baseline.py) | ASC baseline comparison |
| [`tcl.py`](./tcl.py) | thermodynamic continual learning |
| [`coding_asc_finetune.py`](./coding_asc_finetune.py) | canonical coding fine-tune entrypoint |
| [`deepseek_asc_finetune.py`](./deepseek_asc_finetune.py) | experimental coding fine-tune implementation with DeepSeek/Qwen registry |
| [`stack_data_prep.py`](./stack_data_prep.py) | local corpus preparation |
| [`eval_coding.py`](./eval_coding.py) | coding eval helpers |
| [`serve_local.py`](./serve_local.py) | local serving helpers and status payload builder |
| [`research_database.py`](./research_database.py) | persistent research trace store |
| [`researcher_agent.py`](./researcher_agent.py) | Cruxy research loop |
| [`self_train.py`](./self_train.py) | export high-quality research traces |
| [`build_researcher_dataset.py`](./build_researcher_dataset.py) | build researcher corpus |
| [`docs/implementation_roadmap.md`](./docs/implementation_roadmap.md) | formal post-audit remediation roadmap (`WS8-WS16`) |
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
| [`tar_lab/problem_runner.py`](./tar_lab/problem_runner.py) | science-environment import probe and study runner |
| [`tar_lab/science_exec.py`](./tar_lab/science_exec.py) | domain-aware benchmark executors for generic ML, deep learning, computer vision, graph ML, NLP, RL, and quantum ML |
| [`tar_lab/science_profiles.py`](./tar_lab/science_profiles.py) | domain resolution and reproducible science-environment builder |
| [`tar_lab/scheduler.py`](./tar_lab/scheduler.py) | persistent queued problem-study scheduling and single-cycle execution |
| [`science_profiles/`](./science_profiles) | locked domain profiles for ML, deep learning, NLP, CV, RL, graph ML, and quantum ML |

## Author

Christopher Gardner
