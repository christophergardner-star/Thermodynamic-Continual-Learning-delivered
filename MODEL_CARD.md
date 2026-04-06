# Model Card: ASC Model Family

**Author:** Christopher Gardner  
**Status:** architecture validated; CPU/synthetic validation present; large-scale coding runs remain user-executed research work

## Model Description

ASC is a training method for causal language models. A standard backbone is wrapped with:

- `LatentWarp`: a small MLP that perturbs the final hidden state
- `EMA target`: a frozen exponential-moving-average copy of the backbone
- dual optimization:
  - `L_task`: standard next-token loss
  - `L_consistency`: consistency loss between clean-target and warped-online predictions

At inference time, the consistency path is disabled and the model behaves like the underlying causal LM.

## Named ASC Sizes

| Model | Backbone | Status |
|---|---|---|
| `ASC-124M` | `gpt2` | tested |
| `ASC-355M` | `gpt2-medium` | architecture supported |
| `ASC-774M` | `gpt2-large` | architecture supported |
| `ASC-1558M` | `gpt2-xl` | architecture supported |

For offline test coverage, the repo also supports `__tiny_gpt2__` as a synthetic backbone.

## Coding Backbones Supported By The Fine-Tune Harness

The coding fine-tune scripts can target:

- DeepSeek Coder: `1.3B`, `6.7B`, `33B`, plus instruct variants
- Qwen2.5-Coder: `1.5B`, `7B`, `14B`, `32B`, plus instruct variants

Those backbones are configured by [`deepseek_asc_finetune.py`](./deepseek_asc_finetune.py). They are not shipped as checkpoints in this repo.

## Intended Use

This repository is for research and local experimentation.

Recommended uses:

- probing ASC loss behavior
- studying continual-learning behavior with TCL
- local code-model fine-tuning experiments
- researcher trace generation and dataset construction

Not claimed here:

- production readiness
- measured large-scale superiority over standard training
- released checkpoints for the external code backbones

## Validated Capabilities

Validated directly by the current test suite:

- ASC forward path returns task and consistency losses in training mode
- ASC save/load roundtrip works
- EMA target update is correct
- warp network receives gradients through the adversarial ascent path
- TCL importance accumulation, memory, regularization, and trainer surfaces work
- coding-stack registry, run-name derivation, and status helpers work
- research trace storage and claim classification work

## Known Limitations

- Large external backbones still require the user to supply compute and dependencies.
- The local coding stack is intentionally lightweight and does not pretend to be a production trainer.
- The researcher loop currently uses a deterministic stub response path for smoke-safe execution unless you replace it with a real generator.

## Evaluation Boundary

`python -m pytest tests -q` validates the repo surface. It does not validate a large-scale DeepSeek or Qwen benchmark run by itself.

## Citation

```bibtex
@misc{gardner2026asc_tcl_delivered,
  author = {Gardner, Christopher},
  title  = {Thermodynamic Continual Learning Delivered},
  year   = {2026},
  url    = {https://github.com/christophergardner-star/Thermodynamic-Continual-Learning-delivered}
}
```
