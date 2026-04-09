# TAR Operator 7B SFT

## Purpose

`train_tar_operator_sft.py` is the secure adapter-training entrypoint for the
first TAR-native 7B operator model.

It is designed to:

- fine-tune a 7B instruct backbone on the TAR master dataset
- preserve assistant-only supervision
- keep the dataset local by default
- refuse `trust_remote_code`
- refuse automatic hub pushes and third-party logging
- record dataset hashes and training posture into a run manifest

## Security Posture

This trainer defaults to the safest reasonable behavior for a core bespoke
dataset:

- `trust_remote_code=false`
- `push_to_hub=false`
- `report_to=[]`
- dataset must live under the repo root unless explicitly overridden
- remote models must be on the approved list unless explicitly overridden
- dataset manifest and split files are hashed into `run_manifest.json`
- outputs go under `training_artifacts/`, not `dataset_artifacts/`

Current approved remote bases:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-Coder-7B-Instruct`

## Required Dataset Shape

The trainer expects a dataset directory containing:

- `manifest.json`
- `tar_master_dataset_train.jsonl`
- `tar_master_dataset_validation.jsonl`
- optionally `tar_master_dataset_test.jsonl`

The canonical builder is:

```bash
python build_tar_master_dataset.py --state-dir tar_state --output-dir dataset_artifacts/tar_master_dataset_v1
```

## Config

Default config:

- [tar_operator_qwen25_7b_lora.json](/c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/configs/tar_operator_qwen25_7b_lora.json)

That config is the first serious TAR operator run shape:

- base: `Qwen/Qwen2.5-7B-Instruct`
- adapter mode: BF16 LoRA
- rank: `32`
- alpha: `64`
- dropout: `0.05`
- sequence length: `2048`
- effective batch size: `16`

## Pod Bring-Up

On the pod:

```bash
cd /workspace/Thermodynamic-Continual-Learning-delivered
source .venv/bin/activate
python build_tar_master_dataset.py --state-dir tar_state --output-dir dataset_artifacts/tar_master_dataset_v1
```

If you have a richer merged dataset already, point the config or command at that
directory instead.

## Smoke Run

This is the low-risk proof that the full 7B TAR operator path works:

```bash
cd /workspace/Thermodynamic-Continual-Learning-delivered
source .venv/bin/activate
python train_tar_operator_sft.py \
  --config configs/tar_operator_qwen25_7b_lora.json \
  --output-dir training_artifacts/tar_operator_qwen25_7b_smoke \
  --max-steps 20 \
  --max-seq-length 1024 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8
```

## Serious Run

Only do this once the dataset is materially larger than the current tiny seed
corpus. A credible first serious run wants at least thousands of examples.

```bash
cd /workspace/Thermodynamic-Continual-Learning-delivered
source .venv/bin/activate
python train_tar_operator_sft.py \
  --config configs/tar_operator_qwen25_7b_lora.json \
  --output-dir training_artifacts/tar_operator_qwen25_7b_lora_run1
```

## Dry Run

Use this to validate security posture, dataset integrity, hashes, and config
resolution without starting training:

```bash
python train_tar_operator_sft.py --config configs/tar_operator_qwen25_7b_lora.json --dry-run
```

## Outputs

Each run writes:

- `run_manifest.json`
- `run_summary.json`
- trainer checkpoints under the output directory
- `final_adapter/` with the final saved adapter and tokenizer

## Professional Boundary

This is the correct secure path for the first TAR operator model. It is not the
same as:

- full-parameter 7B fine-tuning
- ASC large-model canonical training
- hub-published consumer model training

The point here is a disciplined, reproducible, TAR-native operator adapter.
