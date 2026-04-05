# Thermodynamic Continual Learning (TCL)

Continual learning via entropy-production importance weighting.

Prevents catastrophic forgetting by tracking which weights were "hot"
(actively learned) during each task, then applying elastic protection on
subsequent tasks. Supports gradual reannealing so frozen knowledge can
soften over long task sequences — unlike EWC, which freezes importance
permanently.

Also includes a validated implementation of **Adversarial Self-Consistency
(ASC)** training, a consistency-regularized LLM fine-tuning method.

---

## Install

```bash
pip install torch transformers datasets accelerate tqdm matplotlib
```

No other dependencies.

---

## TCL Quick Start

```python
import torch
import torch.nn as nn
from tcl import ThermalImportance, ThermalMemory, TCLRegularizer

model = ...          # any nn.Module
criterion = nn.CrossEntropyLoss()
memory = ThermalMemory(max_tasks=10, anneal_rate=0.9999)

# --- Task 0 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
importance = ThermalImportance(model, ema_beta=0.99)

for x, y in task0_loader:
    loss = criterion(model(x), y)
    loss.backward()
    importance.accumulate(model)   # after backward, before step
    optimizer.step()
    optimizer.zero_grad()

memory.commit(model, importance, task_id=0)

# --- Task 1 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
importance = ThermalImportance(model, ema_beta=0.99)
regularizer = TCLRegularizer(memory, lambda_tcl=1.0)

for x, y in task1_loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    importance.accumulate(model)
    regularizer.penalty(model).backward()   # forgetting protection
    optimizer.step()
    memory.anneal_all()                     # decay importance over time

memory.commit(model, importance, task_id=1)
```

### TCLTrainer (high-level)

```python
from tcl import TCLTrainer

trainer = TCLTrainer(
    model=model,
    optimizer_factory=lambda p: torch.optim.AdamW(p, lr=3e-4),
    lambda_tcl=1.0,
)

for task_id, (train_loader, val_loader) in enumerate(tasks):
    trainer.learn_task(task_id, train_loader, criterion, epochs=5)
    acc, _ = trainer.evaluate(val_loader, criterion)
    trainer.record_peak_acc(task_id, acc)

result = trainer.evaluate_forgetting(all_val_loaders, criterion)
print(f"Avg forgetting: {result['avg_forgetting']:.3f}")
```

---

## ASC Training (Adversarial Self-Consistency)

Full GPU run (GPT-2, WikiText-2, 3 epochs):

```bash
python asc_train.py
```

CPU smoke run (distilgpt2, 50 steps, ~10 min):

```bash
python asc_train_cpu.py
```

**Validated CPU result:** 50 steps on WikiText-2, avg PPL 134 -> 112,
consistency loss stable at ~7.1, both task and warp_net gradients confirmed.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

- `tests/test_tcl.py` — 21 tests (TCL unit + forgetting benchmark, 3 seeds)
- `tests/test_asc_smoke.py` — 14 tests (ASC mechanism, no dataset download)

All 35 tests pass in under 20s on CPU.

---

## How TCL differs from EWC

| | EWC | TCL |
|---|---|---|
| Importance signal | One-shot Fisher diagonal at task end | Continuous EMA of gradient energy throughout task |
| Frozen? | Yes, permanently | No — reannealing decays importance over time |
| Memory | One importance snapshot per task | Ring buffer (configurable max_tasks) |
| Optimizer | Any | Any |

---

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `lambda_tcl` | 1.0 | Elastic penalty strength. Start 0.5–5.0. |
| `ema_beta` | 0.99 | Importance EMA smoothing. Higher = slower decay. |
| `anneal_rate` | 1.0 | Per-step importance decay. 0.9999 = ~10% drop/1000 steps. |
| `task_decay` | 1.0 | Older tasks weighted less. 0.9 = geometric down-weighting. |
| `max_tasks` | 10 | Ring buffer size. Oldest dropped when full. |

---

## Author

Christopher Gardner — April 2026
