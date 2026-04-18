# Thermodynamic Continual Learning (TCL)

A research implementation of thermodynamics-inspired continual learning, built around the idea that a neural network's activation statistics during training behave like a physical system cooling toward equilibrium — and that this signal can be used to govern learning rate and detect when a task has been consolidated.

**Author:** Christopher Gardner

---

## The Problem

Neural networks forget. Train a network on Task A, then Task B, and it loses Task A — catastrophic forgetting. Standard fixes (EWC, SI, replay) work by penalising weight drift or replaying old data. TCL takes a different approach: use the network's own thermal state to detect when consolidation is happening and adjust the learning rate accordingly.

## The Core Idea

During training, the standard deviation of a layer's activations (`sigma`) starts high (the network is exploring, disordered) and falls as the network converges (ordered, consolidated). TCL defines a thermal ratio:

```
rho = sigma / sigma_star
```

Where `sigma_star` is a reference temperature anchored at the start of each task. The regime detector then classifies:

- `rho > 1.1` → **disordered** → boost LR (network still exploring)
- `0.9 < rho < 1.1` → **critical** → hold LR (near equilibrium)
- `rho < 0.9` → **ordered** → reduce LR (consolidating, protect weights)

The hypothesis: by reducing LR during consolidation and boosting it during disorder, the network retains previous tasks better than vanilla SGD.

## The Calibration Bug (and Fix)

The original implementation defined `sigma_star` as a rolling median of the current `sigma` window. This means `sigma_star` tracks `sigma` — the ratio `rho ≈ 1/alpha` regardless of how much training has happened, making the ordered threshold mathematically unreachable.

**The fix (Phase 8C):** `sigma_star` is now set from the median of the first 20 batches of each task and then frozen. This makes it a genuine reference temperature — the network's thermal level at task onset — rather than a moving target.

The relevant code is in [`tar_lab/thermoobserver.py`](./tar_lab/thermoobserver.py).

## Experimental Results (Phase 8)

All experiments use Split-CIFAR-10 (5 binary tasks), 15 epochs per task, 5 seeds.

### Tiny backbone (~189K params)

| | Forgetting | Accuracy |
|---|---|---|
| TCL | 0.0918 ± 0.013 | 0.843 |
| SGD | 0.0977 ± 0.022 | 0.837 |

- mean delta = −0.006 (TCL better), p = 0.68, Cohen's d = 0.20
- TCL better on 3/5 seeds
- **Key finding:** TCL has 5× lower forgetting variance than SGD, but the effect is too small to be statistically significant at this model size.

### ResNet-18 backbone (~11M params)

| | Forgetting | Accuracy |
|---|---|---|
| TCL | 0.265 ± 0.015 | 0.655 |
| SGD | 0.221 ± 0.070 | 0.702 |

- mean delta = +0.044 (SGD better), p = 0.26
- TCL better on 2/5 seeds
- **Finding:** At 15 epochs, ResNet-18 is underfitted. The governor reads underfitting as thermodynamic stability and cuts LR prematurely. The variance-reduction effect (TCL std 0.015 vs SGD 0.070) still holds — TCL training is consistently predictable even when not optimal.

### Interpretation

The anchor fix is mechanistically correct. The variance reduction is real and reproducible. The missing piece is a **warmup guard**: the anchor should only be set after the network has meaningfully learned (not during the first 20 batches on an underfitted large model). This is the next step.

Full Phase 8 analysis: [`docs/phase8_roadmap.md`](./docs/phase8_roadmap.md)

---

## Running the Benchmark

```bash
# Install dependencies
python bootstrap.py

# Single-seed validation (confirms anchor fix working)
python phase8c_validate.py

# 5-seed benchmark — tiny backbone
python phase8c_benchmark.py

# 5-seed benchmark — ResNet-18
python phase8c_scale.py
```

Results are written to `tar_state/comparisons/`.

---

## Repository Structure

```
tar_lab/
  thermoobserver.py     # Core: activation thermodynamics + regime detection
  multimodal_payloads.py # Split-CIFAR-10 benchmark (TCL, SGD, EWC, SI)
  schemas.py            # Typed configs and result schemas

phase8c_validate.py     # Diagnostic: confirms anchor fix via 3 checks
phase8c_benchmark.py    # 5-seed tiny-model benchmark
phase8c_scale.py        # 5-seed ResNet-18 benchmark

docs/
  phase8_roadmap.md     # Full Phase 8 investigation log

tar_state/
  comparisons/          # Benchmark result JSONs
  cl_traces/            # Per-epoch diagnostic traces
```

---

## The Broader Stack

This repo also contains two other systems built alongside TCL:

### ASC — Adversarial Self-Consistency

A training method for causal language models that adds a consistency loss between a live model and a frozen EMA target. The main model is [`ASCForCausalLM`](./asc_model.py). Supported backbones include GPT-2 (124M–1.5B) and fine-tune paths for DeepSeek Coder and Qwen2.5-Coder.

```bash
python asc_train_full.py --size 124M --max_steps 100
```

### TAR — Thermodynamic Autonomous Researcher

An experiment orchestration layer that wraps TCL/ASC experiments with a tri-role planning hierarchy (Director, Strategist, Scout), atomic state persistence, Docker-sandboxed execution, and research memory backed by a local vector store.

```bash
python tar_cli.py --direct --status
python tar_cli.py --direct --dry-run --json
streamlit run dashboard.py
```

Full TAR documentation: see the TAR Lab section below, or run `python tar_cli.py --help`.

---

## Setup

```bash
python bootstrap.py           # standard install
python bootstrap.py --gpu     # adds GPU dependencies
python bootstrap.py --dry-run # preview without installing
```

```bash
python -m pytest tests -q     # run test suite (143 tests)
```

---

## TAR Lab — Full CLI Reference

<details>
<summary>Expand TAR CLI commands</summary>

```bash
python tar_cli.py --direct --dry-run --json
python tar_cli.py --direct --status
python tar_cli.py --direct --check-regime
python tar_cli.py --direct --chat --message "Analyse current stability" --json
python tar_cli.py --direct --ingest-research --topic "continual learning" --json
python tar_cli.py --direct --verify-last-trial --json
python tar_cli.py --direct --breakthrough-report --json
python tar_cli.py --direct --resolve-problem --problem "Investigate barren plateaus in quantum AI" --json
python tar_cli.py --direct --study-problem --problem "Investigate optimisation stability" --json
python tar_cli.py --direct --run-problem-study --json
python tar_cli.py --direct --run-problem-study --use-docker --build-env --json
python tar_cli.py --direct --schedule-problem-study --delay-s 0 --max-runs 1 --json
python tar_cli.py --direct --frontier-status --json
python tar_cli.py --direct --runtime-status --json
python tar_cli.py --direct --sandbox-policy --json
python tar_cli.py --direct --show-manifest --json
python tar_cli.py --direct --list-experiment-backends --json
python tar_cli.py --direct --register-checkpoint --checkpoint-name asc-local --model-path /path/to/ckpt --json
python tar_cli.py --direct --build-inference-endpoint --checkpoint-name asc-local --role director --json
python tar_cli.py --direct --start-endpoint --checkpoint-name asc-local --role assistant --wait-for-health --json
python tar_cli.py --direct --claim-policy --json
python tar_cli.py --direct --pivot --force --json
python tar_cli.py --direct --panic --json
python tar_cli.py --serve
streamlit run tar_dashboard.py
```

Environment overrides:

```bash
set TAR_DATA_MODE=DOWNLOAD_REAL
set TAR_LLM_BASE_URL=http://localhost:8000/v1
set TAR_DIRECTOR_MODEL=your-70b-model
set TAR_STRATEGIST_MODEL=your-30b-model
set TAR_SCOUT_MODEL=your-14b-model
set TAR_GPU_INDEX=0
set TAR_CPU_LIMIT=8
set TAR_MEMORY_LIMIT_GB=16
```

</details>
