# WS26 Closeout

`WS26` closed successfully.

## Outcome

The `WS26` TCL-deepened adapter materially outperformed both the prompt-only
base model and the `WS25` adapter on the frozen `WS26` eval pack.

Final pod-measured results:

- prompt-only base:
  - `mean_score = 0.0261`
  - `decision_accuracy = 0.0108`
  - `parse_error_rate = 0.6240`
- `WS25` adapter on `WS26` eval:
  - `mean_score = 0.7623`
  - `decision_accuracy = 0.7493`
  - `parse_error_rate = 0.1146`
  - `tcl_reasoning_mismatch = 103`
- `WS26` adapter:
  - `mean_score = 0.8806`
  - `decision_accuracy = 0.8760`
  - `parse_error_rate = 0.1119`
  - `tcl_reasoning_mismatch = 5`
  - `overclaim_rate = 0.0`

## Reading

The important result is not just the aggregate score bump. The decisive signal is
that TCL-specific reasoning errors collapsed while parse robustness held roughly
steady. That is the correct signature for a successful TCL-deepening run.

## Residual Weaknesses

- parse error is still above zero and should continue to be treated as a
  structured-output hygiene issue
- `falsification_or_verification_mismatch` rose slightly in the final run and
  should be inspected in later analysis rather than ignored

## Decision

The correct professional move after `WS26` is:

1. freeze the result
2. capture residual risks explicitly
3. move into `WS22` rather than burning another immediate retraining cycle

That is the posture this repository now takes.
