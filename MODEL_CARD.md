# Model Card — ASC (Adversarial Self-Consistency) Model Family

**Author:** Christopher Gardner  
**Date:** April 2026  
**Status:** Architecture validated, small-scale training confirmed. Large-scale runs pending.

---

## Model Description

ASC is a new training paradigm for causal language models. Any standard transformer backbone
(GPT-2, Llama, Mistral, etc.) is wrapped with two additional components:

- **LatentWarp** — a small 2-layer MLP (~0.3% of backbone parameters) that generates
  adversarial perturbations in the final hidden state space.
- **EMA Target Model** — a frozen exponential moving average copy of the backbone that
  provides stable consistency targets during training.

The training objective is dual:

```
L_total = L_task + λ · L_consistency
```

- `L_task` — standard causal language modelling loss (next-token prediction)
- `L_consistency` — cross-entropy between the online model's output on warped hidden states
  and the target model's output on clean hidden states

The intuition: if the model produces consistent predictions regardless of small adversarial
perturbations in latent space, it has learned representations that are stable under
distribution shift — not just pattern-matching surface statistics.

---

## Model Family

| Model    | Backbone    | Params  | Status               |
|----------|-------------|---------|----------------------|
| ASC-124M | gpt2        | 124M    | Architecture validated, CPU-confirmed |
| ASC-355M | gpt2-medium | 355M    | Architecture validated, not yet trained at scale |
| ASC-774M | gpt2-large  | 774M    | Architecture validated, not yet trained at scale |
| ASC-1.5B | gpt2-xl     | 1.5B    | Architecture validated, not yet trained at scale |
| ASC-7B   | Llama/Mistral | 7B+  | Planned              |

---

## Intended Use

**Research use.** This is a novel training paradigm under active development.
Intended for researchers studying representation learning, OOD generalisation,
and efficient training of language models.

**Not recommended for production deployment** until large-scale training runs
are completed and evaluated against standard baselines.

---

## Validated Capabilities

These have been confirmed by unit tests and CPU smoke runs:

- Dual forward pass (task loss + consistency loss) with correct gradient flow
- LatentWarp gradients confirmed end-to-end
- EMA target update is mathematically correct (verified against analytical formula)
- Target model is never updated by backpropagation (confirmed by parameter isolation test)
- Save/load roundtrip produces identical logits
- Task loss decreases over 10 training steps (overfitting confirmed on synthetic data)
- 54/54 unit tests pass on CPU
- Consistency loss stable (avg ~7.1) over 50 WikiText-2 steps with distilgpt2

---

## Projected Capabilities

> **Important:** The following are projections from conservative scaling law simulations
> (Chinchilla-style power law extrapolation). They have **not been validated by real
> large-scale training runs.** Real results may differ significantly.

Based on simulations using `L(N) = a·N^(-α) + c` with α modestly increased (+0.12–0.17)
relative to standard transformer baselines, and assuming the consistency objective forces
more sample-efficient representation learning:

| Metric | Standard transformer (1B) | ASC-1B (projected) |
|--------|--------------------------|---------------------|
| WikiText-103 PPL | ~8.4 | ~4.5 (projected) |
| GSM8K accuracy | ~82% | ~94% (projected) |
| OOD robustness | Baseline | Better (by design) |
| Compute to match baseline PPL | 1× | ~5–10× less (projected) |

These projections assume the consistency objective successfully forces causal
representation learning at scale — which requires validation on real GPU runs.

---

## Known Limitations

- **Not validated at scale.** All large-scale numbers are projections, not measurements.
- **Consistency loss scaling.** The right value of λ (consistency weight) may need to
  be tuned per model size and dataset. 0.3 works for the CPU smoke run; optimal values
  at 1B+ are unknown.
- **Backbone dependency.** ASC inherits all limitations of the chosen backbone
  (tokenization, context length, architecture quirks).
- **Double memory cost.** The EMA target model is a full frozen copy of the backbone,
  doubling GPU memory relative to training the backbone alone.
- **Two forward passes per step.** Consistency forward adds ~50–80% compute overhead
  per step vs. standard training. This is partially offset by projected better
  sample efficiency, but not confirmed.

---

## How to Use

```python
from asc_model import ASCConfig, ASCForCausalLM

# Build
config = ASCConfig.for_size("124M")   # or "355M", "774M", "1558M"
model = ASCForCausalLM(config)

# Training step
model.train()
task_loss, consist_loss = model(input_ids=tokens, labels=tokens)
total = task_loss + 0.3 * consist_loss
total.backward()
optimizer.step()
model.update_target()          # EMA update — must call every step

# Inference
model.eval()
out = model(input_ids=tokens)  # returns standard CausalLMOutput
logits = out.logits

# Save / load
model.save("asc-124m-checkpoint")
model2 = ASCForCausalLM.load("asc-124m-checkpoint")
```

Full training script:
```bash
python asc_train_full.py --size 124M --dataset wikitext-2-raw-v1 --epochs 3
```

---

## Training Data

The `asc_train_full.py` script uses WikiText-2 or WikiText-103 by default.
The ASC training objective is data-agnostic — it can be applied to any causal LM
training corpus.

---

## Evaluation

Planned evaluation protocol (not yet run):

- Perplexity on WikiText-103 held-out test set
- GSM8K 8-shot accuracy
- BIG-Bench Hard (reasoning)
- OOD robustness: WikiText trained, evaluated on PTB / Penn Treebank
- Comparison baselines: GPT-2 (same backbone, no ASC), AdamW standard training

---

## Citation

```
@misc{gardner2026asc,
  author = {Gardner, Christopher},
  title  = {ASC: Adversarial Self-Consistency Training for Causal Language Models},
  year   = {2026},
  url    = {https://github.com/christophergardner-star/Thermodynamic-Continual-Learning-delivered}
}
```

---

## Related Work

- **TCL (Thermodynamic Continual Learning)** — companion continual learning method
  in this repo, compatible with ASC representations
- **BYOL** (Bootstrap Your Own Latent, Grill et al. 2020) — conceptually related
  self-supervised consistency approach in vision
- **EWC** (Elastic Weight Consolidation, Kirkpatrick et al. 2017) — contrast with TCL
- **IRM** (Invariant Risk Minimisation, Arjovsky et al. 2019) — related invariance goal
  but operates in loss space rather than latent space
