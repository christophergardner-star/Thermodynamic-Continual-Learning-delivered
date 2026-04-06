# Thermodynamic Continual Learning Delivered

This repository now contains a coherent research stack built around two core methods:

- `ASC`: Adversarial Self-Consistency for causal language models
- `TCL`: Thermodynamic Continual Learning for catastrophic-forgetting control

Around those core methods, the repo also includes the intended local coding and researcher workflow:

- coding-backbone ASC fine-tuning for DeepSeek Coder and Qwen2.5-Coder
- local corpus preparation and lightweight coding evaluation
- local serving helpers for Continue / OpenAI-compatible clients
- `Cruxy`, a researcher loop that stores traces and classifies claims as fact, measured result, inference, or hypothesis

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

## Validation

Current validated surface:

- ASC core model tests
- ASC CPU smoke tests
- TCL unit tests
- coding stack surface tests
- researcher stack tests

Run the suite with:

```bash
python -m pytest tests -q
```

Current expected count:

- `64` tests

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

## Author

Christopher Gardner
