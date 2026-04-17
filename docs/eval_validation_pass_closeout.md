## Eval Validation Pass Closeout

Status:

- independent eval validation pass: **complete**
- comparison result: **corrected**
- next gate: **WS36 may begin**

Sealed external pack:

- pack directory: [eval_artifacts/tar_operator_eval_external_v1](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/tar_operator_eval_external_v1)
- manifest: [eval_manifest.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/tar_operator_eval_external_v1/eval_manifest.json)
- sealed pack SHA256: `5f109fff09e87a970a9faf020264fae7e833453cc79de4a1e04300ef3044cecd`

Validation outputs:

- results: [results.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/external_validation/results.json)
- family breakdown: [family_breakdown.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/external_validation/family_breakdown.json)
- suite breakdown: [suite_breakdown.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/external_validation/suite_breakdown.json)
- run manifest: [run_manifest.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/external_validation/run_manifest.json)
- run manifest SHA256: `805f81e847380cf1e729969e271eb95fc80e940fe7759d7668f5466f3e0c9b8b`

Headline metrics:

- `mean_score = 0.4625`
- `decision_accuracy = 0.4375`
- `overclaim_rate = 0.0`
- `false_refusal_rate = 0.0`
- `parse_error_rate = 0.4375`

Comparison result:

- external validation did **not** confirm the published `WS27R2` `mean_score = 0.823` claim
- external validation **did** confirm `overclaim_rate = 0.0`
- [ws27r2_refine_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r2_refine_closeout.md) was corrected on `2026-04-17`

Artifact note:

- `suite_breakdown.json` was derived locally from task-family mapping because the sealed external pack does not carry explicit `suite_name` fields
- that derivation note is recorded in [run_manifest.json](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/eval_artifacts/external_validation/run_manifest.json)

Unlock:

- the independent validation obligation is now closed
- `WS36` may begin
