# ASC Model Family + Thermodynamic Continual Learning

Two novel ML systems by Christopher Gardner (April 2026).

---

## ASC — Adversarial Self-Consistency Model Family

A new training paradigm that learns invariant causal representations
through latent adversarial consistency. Works with any causal transformer
backbone. The ASC-specific components add <1% parameter overhead.

### Architecture

```
Input Tokens
     |
Base Transformer (GPT-2 / Llama-style blocks)
     |
Last Hidden States H
     |-- Task Head (LM Head) --> next-token loss
     |
     +-- LatentWarp MLP (NEW, ~0.1-0.5% of params)
              |
         Warped Hidden States H'
              |
         Consistency Forward (same base, frozen EMA target)
              |
         Consistency Loss
```

**Total loss = task_loss + lambda * consistency_loss**

### Model Family

| Model    | Params | Backbone     | H100 training estimate |
|----------|--------|--------------|------------------------|
| ASC-124M | 124M   | gpt2         | 1-2 days               |
| ASC-355M | 355M   | gpt2-medium  | 3-5 days               |
| ASC-774M | 774M   | gpt2-large   | 1 week                 |
| ASC-1.5B | 1.5B   | gpt2-xl      | 2 weeks                |
| ASC-7B   | 7B     | Llama/Mistral| 4-8 weeks              |

### Install

```bash
pip install torch transformers datasets accelerate tqdm matplotlib pytest
```

### Quick Start

```python
from asc_model import ASCConfig, ASCForCausalLM

# Build ASC-124M
config = ASCConfig.for_size("124M")
model = ASCForCausalLM(config)
print(model.param_summary())
# {'base_params': 124439808, 'warp_params': 394496, 'warp_pct': 0.317, ...}

# Training step
model.train()
task_loss, consist_loss = model(input_ids=x, labels=y)
total = task_loss + 0.3 * consist_loss
total.backward()
optimizer.step()
model.update_target()   # EMA update after every step

# Inference (consistency path auto-disabled in eval mode)
model.eval()
logits = model(input_ids=x).logits

# Save / load
model.save("my_asc_124m")
model2 = ASCForCausalLM.load("my_asc_124m")
```

### Training Scripts

Full GPU run (any size):
```bash
python asc_train_full.py --size 124M --dataset wikitext-2-raw-v1
python asc_train_full.py --size 355M --dataset wikitext-103-raw-v1 --batch_size 16
```

Multi-GPU (H100 cluster):
```bash
torchrun --nproc_per_node=4 asc_train_full.py --size 1B --batch_size 16
```

CPU smoke run (50 steps, ~10 min, validated):
```bash
python asc_train_cpu.py
```

**Validated CPU result:** distilgpt2, 50 steps, WikiText-2,
avg PPL 134 -> 112, consistency loss stable ~7.1.

### ASC Key Parameters

| Parameter         | Default | Effect |
|-------------------|---------|--------|
| `warp_dim`        | 256     | LatentWarp bottleneck. 64-512 depending on model size. |
| `consistency_lambda` | 0.3  | Consistency loss weight. Start 0.1-0.5. |
| `ema_decay`       | 0.995   | Target model EMA. Higher = slower target updates. |
| `warp_init_scale` | 0.05    | Initial warp magnitude. Keep small for training stability. |

---

## TCL — Thermodynamic Continual Learning

Prevents catastrophic forgetting by tracking which weights were "hot"
(actively learned) during each task, then applying elastic protection on
subsequent tasks. Supports reannealing so frozen knowledge softens over
long task sequences — unlike EWC which freezes importance permanently.

### Quick Start

```python
from tcl import ThermalImportance, ThermalMemory, TCLRegularizer

memory = ThermalMemory(max_tasks=10, anneal_rate=0.9999)

# Task 0
importance = ThermalImportance(model, ema_beta=0.99)
for x, y in task0_loader:
    loss = criterion(model(x), y)
    loss.backward()
    importance.accumulate(model)   # after backward, before step
    optimizer.step()
    optimizer.zero_grad()
memory.commit(model, importance, task_id=0)

# Task 1 (with forgetting protection)
importance = ThermalImportance(model, ema_beta=0.99)
regularizer = TCLRegularizer(memory, lambda_tcl=1.0)
for x, y in task1_loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    importance.accumulate(model)
    regularizer.penalty(model).backward()
    optimizer.step()
    memory.anneal_all()
memory.commit(model, importance, task_id=1)
```

### TCL vs EWC

| | EWC | TCL |
|---|---|---|
| Importance signal | One-shot Fisher diagonal at task end | Continuous EMA of gradient energy throughout task |
| Frozen? | Permanently | No — reannealing decays importance over time |
| Memory | One snapshot per task | Ring buffer (configurable max_tasks) |
| Optimizer | Any | Any |

### TCL Key Parameters

| Parameter     | Default | Effect |
|---------------|---------|--------|
| `lambda_tcl`  | 1.0     | Elastic penalty strength. Start 0.5-5.0. |
| `ema_beta`    | 0.99    | Importance EMA smoothing. |
| `anneal_rate` | 1.0     | Per-step importance decay (0.9999 = ~10% drop/1000 steps). |
| `task_decay`  | 1.0     | Older tasks weighted less. |
| `max_tasks`   | 10      | Ring buffer size. |

---

## Tests

```bash
pytest tests/ -v
```

| Suite | Tests | Time |
|-------|-------|------|
| `test_asc_model.py` | 19/19 | ~5 min CPU |
| `test_asc_smoke.py` | 14/14 | 6.5s CPU |
| `test_tcl.py` | 21/21 | 8.4s CPU |
| **Total** | **54/54** | **~6 min** |

---

## Files

| File | Purpose |
|------|---------|
| `asc_model.py` | ASCConfig + ASCForCausalLM (full model class) |
| `asc_train_full.py` | Full training script, all sizes, H100-ready |
| `asc_train_cpu.py` | CPU smoke run (validated, 50 steps) |
| `asc_train.py` | Original Colab-style training script |
| `tcl.py` | TCL continual learning module |
| `tests/` | 54 tests across both systems |

---

## Author

Christopher Gardner — April 2026
