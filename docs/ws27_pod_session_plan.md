# WS27 Pod Session Plan

## Start Condition

Open the pod only after all of these are true:

- the public repo is current on `main`
- the private dataset directory
  `dataset_artifacts/tar_master_dataset_ws26_merged_v1` is available locally
  and ready to sync
- the `WS27` branch design is frozen in
  [ws27_branch_design.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27_branch_design.md)
- the `WS27` probe and run configs exist
- the `WS27` eval runtime configs exist
- local tests are green

That condition is now the correct first pod moment for `WS27`.

## Recommended Pod

- GPU: `A100 80GB`
- disk: `300GB+`
- persistent volume enabled

## Required Private Artifact Sync

The repo is public, but the branch dataset is private.

Sync to the pod before training:

- `dataset_artifacts/tar_master_dataset_ws26_merged_v1`

Optional sync to save time:

- `eval_artifacts/tar_operator_eval_ws26_v1`

The eval pack can be rebuilt on the pod from the dataset if needed.

## Pod Purpose

The first `WS27` pod cycle is not an open-ended experiment. It is a disciplined
branch probe:

1. pull the public repo
2. sync the private dataset
3. run the prompt-only probe baseline
4. run the `WS27` probe train config
5. run the adapter-backed probe eval
6. decide whether the first serious branch run is justified

## Session Steps

From the repo root on the pod:

```bash
cd /workspace/Thermodynamic-Continual-Learning-delivered
source .venv/bin/activate
```

Sanity slice:

```bash
python -m pytest tests/test_eval_harness.py tests/test_eval_scorers.py tests/test_tar_operator_sft.py -q
```

Prompt-only probe baseline:

```bash
python eval_tar_operator.py \
  --config configs/tar_operator_eval_ws27_probe_runtime.json \
  --output-dir eval_artifacts/tar_operator_eval_runs/ws27_prompt_only_probe
```

Branch probe train run:

```bash
python train_tar_operator_sft.py \
  --config configs/tar_operator_qwen25_coder_7b_ws27_probe.json \
  --output-dir training_artifacts/ws27_qwen25_coder_7b_probe
```

Adapter-backed probe eval:

```bash
python eval_tar_operator.py \
  --config configs/tar_operator_eval_ws27_probe_runtime.json \
  --adapter-path training_artifacts/ws27_qwen25_coder_7b_probe/final_adapter \
  --output-dir eval_artifacts/tar_operator_eval_runs/ws27_probe_adapter_eval
```

If and only if the probe passes, run the first serious branch cycle:

```bash
python train_tar_operator_sft.py \
  --config configs/tar_operator_qwen25_coder_7b_ws27_run1.json \
  --output-dir training_artifacts/ws27_qwen25_coder_7b_run1
```

```bash
python eval_tar_operator.py \
  --config configs/tar_operator_eval_ws27_runtime.json \
  --adapter-path training_artifacts/ws27_qwen25_coder_7b_run1/final_adapter \
  --output-dir eval_artifacts/tar_operator_eval_runs/ws27_run1_adapter_eval
```

## Success Criteria

Probe-level success:

- probe baseline ran cleanly
- probe training ran cleanly
- probe adapter eval ran cleanly
- no major honesty or parse regression
- enough TCL/ASC signal to justify `run1`

Run-1 success:

- `WS27` branch beats the prompt-only baseline clearly
- branch result is competitive with or better than `WS26` on TCL-heavy suites
- honesty and reproducibility discipline remain intact
- parse reliability remains acceptable

## Stop Rules

Stop the pod-backed cycle if any of these occur:

- the private dataset cannot be synced cleanly
- the prompt-only probe baseline fails structurally
- the probe train run fails on fit or checkpoint integrity
- the branch materially regresses honesty or parse reliability
- the branch gain is too weak to justify `run1`

## Termination Rule

Terminate the pod as soon as:

- the run/eval cycle is complete
- artifacts are copied back locally
- results are recorded
- no immediate GPU command remains
