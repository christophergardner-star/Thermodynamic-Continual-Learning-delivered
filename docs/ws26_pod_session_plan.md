# WS26 Pod Session Plan

## Start Condition

Start the pod only after local WS26 prep is complete.

That condition is now satisfied when all of these are true:

- `dataset_artifacts/tar_master_dataset_ws26_v1` exists
- `dataset_artifacts/tar_master_dataset_ws26_merged_v1` exists
- `eval_artifacts/tar_operator_eval_ws26_v1` exists
- `configs/tar_operator_qwen25_7b_ws26.json` exists
- `configs/tar_operator_eval_ws26_runtime.json` exists
- local tests are green

## Recommended Pod

- GPU: `A100 80GB`
- disk: `250GB+`
- persistent volume enabled

## Pod Purpose

The pod is for one focused WS26 retrain/eval cycle:

1. sync the current repo
2. run the prompt-only WS26 eval baseline
3. run the WS26 LoRA retrain
4. run the adapter-backed WS26 eval
5. compare against the `WS25` baseline
6. preserve artifacts

## Session Steps

From the repo root on the pod:

```bash
cd /workspace/Thermodynamic-Continual-Learning-delivered
source .venv/bin/activate
```

Sanity check:

```bash
python -m pytest tests/test_tar_master_dataset.py tests/test_eval_scorers.py tests/test_eval_harness.py -q
```

Prompt-only baseline on the deeper WS26 eval:

```bash
python eval_tar_operator.py \
  --config configs/tar_operator_eval_ws26_runtime.json \
  --output-dir eval_artifacts/tar_operator_eval_runs/ws26_prompt_only_baseline
```

WS26 retrain:

```bash
python train_tar_operator_sft.py \
  --config configs/tar_operator_qwen25_7b_ws26.json \
  --output-dir training_artifacts/ws26_qwen25_7b_run1
```

Adapter-backed WS26 eval:

```bash
python eval_tar_operator.py \
  --config configs/tar_operator_eval_ws26_runtime.json \
  --adapter-path training_artifacts/ws26_qwen25_7b_run1/final_adapter \
  --output-dir eval_artifacts/tar_operator_eval_runs/ws26_adapter_eval
```

## Artifact Targets

- training:
  - `training_artifacts/ws26_qwen25_7b_run1`
- eval:
  - `eval_artifacts/tar_operator_eval_runs/ws26_prompt_only_baseline`
  - `eval_artifacts/tar_operator_eval_runs/ws26_adapter_eval`

## Success Criteria

- prompt-only baseline completed on the WS26 eval pack
- WS26 adapter training completed cleanly
- adapter eval completed cleanly
- parse errors do not materially regress
- TCL suite improves over the `WS25` adapter baseline
- honesty and governance do not materially regress

## Stop Rules

Stop the pod-backed cycle if:

- the baseline eval cannot run cleanly
- the training run fails repeatedly on environment issues
- the adapter materially regresses honesty or parse reliability
- output format breaks hard enough that the eval is not trustworthy
