# WS27 Pod Session Plan

Status:

- this is the current `WS27R1 run1` pod session plan
- the initial coder-backbone probe plan is historical
- the true-continuation justification result is recorded in
  [ws27r1_true_continuation_closeout.md](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r1_true_continuation_closeout.md)

## Start Condition

Open the pod only after all of these are true:

- the public repo is current on `main`
- the private dataset directory
  `dataset_artifacts/tar_master_dataset_ws27_branch_v1` is available locally
  and ready to sync
- the private eval packs are available locally and ready to sync:
  - `eval_artifacts/tar_operator_eval_ws27r1_probe_v1`
  - `eval_artifacts/tar_operator_eval_ws27r1_ws26_regression_v1`
- the true regenerated `WS26` adapter is available locally and ready to sync:
  - `training_artifacts/ws26_qwen25_7b_run1/final_adapter`
- the `WS27R1 run1` config exists
- the `WS27R1` eval runtime configs exist
- the local repo state is pushed
- local tests are green

That condition is the correct `WS27R1 run1` pod moment.

## Recommended Pod

- GPU: `A100 80GB`
- disk: `300GB+`
- persistent volume enabled

## Required Private Artifact Sync

The repo is public, but the branch dataset, eval packs, and continuation
adapter are private.

Sync to the pod before training:

- `dataset_artifacts/tar_master_dataset_ws27_branch_v1`
- `eval_artifacts/tar_operator_eval_ws27r1_probe_v1`
- `eval_artifacts/tar_operator_eval_ws27r1_ws26_regression_v1`
- `training_artifacts/ws26_qwen25_7b_run1/final_adapter`

## Pod Purpose

This pod cycle is for one disciplined `WS27R1 run1` execution:

1. pull the public repo
2. sync the private branch dataset, eval packs, and true `WS26` adapter
3. run the `WS27R1 run1` train config
4. run the branch eval on the frozen `WS27R1` probe pack
5. run the `WS26` non-regression eval
6. preserve artifacts and decide whether a further branch iteration is needed

## Session Steps

From the repo root on the pod:

```bash
cd /workspace/Thermodynamic-Continual-Learning-delivered
source .venv/bin/activate
```

Sanity slice:

```bash
python -m pytest tests/test_eval_harness.py tests/test_eval_scorers.py tests/test_tar_operator_sft.py tests/test_ws27_branch_dataset.py -q
```

Private sync targets inside the repo:

```bash
dataset_artifacts/tar_master_dataset_ws27_branch_v1
eval_artifacts/tar_operator_eval_ws27r1_probe_v1
eval_artifacts/tar_operator_eval_ws27r1_ws26_regression_v1
training_artifacts/ws26_qwen25_7b_run1/final_adapter
```

Branch `run1` train run:

```bash
python train_tar_operator_sft.py \
  --config configs/tar_operator_qwen25_7b_ws27r1_run1.json \
  --output-dir training_artifacts/ws27r1_qwen25_7b_run1
```

Adapter-backed branch eval:

```bash
python eval_tar_operator.py \
  --config configs/tar_operator_eval_ws27r1_probe_runtime.json \
  --adapter-path training_artifacts/ws27r1_qwen25_7b_run1/final_adapter \
  --output-dir eval_artifacts/tar_operator_eval_runs/ws27r1_run1_probe_eval
```

WS26 non-regression gate:

```bash
python eval_tar_operator.py \
  --config configs/tar_operator_eval_ws27r1_ws26_regression_runtime.json \
  --adapter-path training_artifacts/ws27r1_qwen25_7b_run1/final_adapter \
  --output-dir eval_artifacts/tar_operator_eval_runs/ws27r1_run1_ws26_regression_eval
```

## Success Criteria

`run1` success:

- the train run completes cleanly from the true `WS26` continuation base
- branch eval preserves:
  - `overclaim_rate = 0.0`
  - strong TCL reasoning quality
- the branch remains competitive with or better than the true-continuation
  probe reference:
  - `mean_score = 0.8207`
  - `decision_accuracy = 0.8133`
  - `parse_error_rate = 0.16`
- the WS26 non-regression gate remains strong:
  - `mean_score` stays in the strong `0.85` range
  - `decision_accuracy` stays in the strong `0.84+` range
  - `overclaim_rate = 0.0`

## Stop Rules

Stop the pod-backed cycle if any of these occur:

- the private branch artifacts cannot be synced cleanly
- the run fails on fit or checkpoint integrity
- continuation from the true `WS26` adapter fails structurally
- the branch materially regresses honesty or parse reliability
- the WS26 non-regression gate fails materially

## Termination Rule

Terminate the pod as soon as:

- the `run1` train/eval cycle is complete
- artifacts are copied back locally
- results are recorded
- no immediate GPU command remains
