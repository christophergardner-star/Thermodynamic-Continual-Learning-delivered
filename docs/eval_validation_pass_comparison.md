## Independent Eval Validation Pass Comparison

External validation slice:

- pack: `eval_artifacts/external_validation`
- eval version: `tar-operator-eval-external-v1`
- item count: `16`

Headline comparison:

| Metric | Internal WS27R2 Published Baseline | External Validation Slice | Delta |
| --- | ---: | ---: | ---: |
| mean_score | 0.823 | 0.4625 | -0.3605 |
| overclaim_rate | 0.0 | 0.0 | 0.0 |
| decision_accuracy | not published in the WS27R2 claim baseline | 0.4375 | n/a |
| false_refusal_rate | not published in the WS27R2 claim baseline | 0.0 | n/a |
| parse_error_rate | not published in the WS27R2 claim baseline | 0.4375 | n/a |

Material divergence result:

- `mean_score` divergence: **yes**
  - correction threshold: external `< 0.773`
  - observed external: `0.4625`
- `overclaim_rate` hard correction: **no**
  - published: `0.0`
  - external: `0.0`

Conclusion:

- `WS27R2` published metrics are **not confirmed** by the independent external slice.
- The published `mean_score = 0.823` claim is materially overstated relative to the independent validation result.
- The honesty claim on `overclaim_rate = 0.0` is confirmed by the external slice.

Correction required:

- corrected document: [ws27r2_refine_closeout.md](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/docs/ws27r2_refine_closeout.md)
- corrected claim:
  - independently validated external-slice `mean_score = 0.4625`
  - independently validated external-slice `overclaim_rate = 0.0`

Corrected reading of WS27R2:

- the internal probe/regression packs still describe the historical refinement run outcome
- the authoritative independent validation result for the held-out external slice is:
  - `mean_score = 0.4625`
  - `decision_accuracy = 0.4375`
  - `false_refusal_rate = 0.0`
  - `parse_error_rate = 0.4375`
  - `overclaim_rate = 0.0`
