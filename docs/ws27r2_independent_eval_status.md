# WS27R2 Independent Eval Status

## Status

The independent external `WS27R2` eval pass is **prepared but blocked**.

What is already complete:

- sealed external pack:
  `eval_artifacts/tar_operator_eval_external_v1`
- frozen manifest:
  [eval_manifest.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/tar_operator_eval_external_v1/eval_manifest.json)
- manifest SHA256:
  `5f109fff09e87a970a9faf020264fae7e833453cc79de4a1e04300ef3044cecd`
- independent evaluator:
  [eval_tar_operator_external.py](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_tar_operator_external.py)
- runtime config:
  [tar_operator_eval_external_v1.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/configs/tar_operator_eval_external_v1.json)
- regression coverage:
  [test_eval_tar_operator_external.py](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/tests/test_eval_tar_operator_external.py)

## Blocking Condition

The workstation does not currently have a complete local copy of the base model:

- `Qwen/Qwen2.5-7B-Instruct`

The tokenizer/config snapshot is present, but the model-weight shards are not.
The external evaluator now fails fast with a precise preflight error instead of
hanging during endpoint startup.

Missing files:

- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

## Meaning

No external `WS27R2` score has been produced yet.

That means:

- the sealed external eval slice is ready
- the independent scorer is ready
- the comparison doc is not yet valid to write
- `ws27r2_refine_closeout.md` must not be adjusted until the real external run
  exists

## Next Action

As soon as the full base model exists locally, rerun:

```powershell
.\.venv\Scripts\python.exe eval_tar_operator_external.py --run-only
```

At that point TAR should:

1. score `WS27R2` on the sealed external slice
2. write the honest comparison doc
3. adjust published `WS27R2` claims if the external result materially trails the
   internal score
