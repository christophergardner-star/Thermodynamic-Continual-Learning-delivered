# WS30 Closeout

## Outcome

`WS30` is complete.

Locked payload execution and build attestation are now operational truth on the
validated local Docker path, not just metadata plumbing.

## What Was Fixed

The original WS30 local validation exposed multiple real blockers:

1. payload dependency locking was host-derived and could pin versions invalid
   for the target image Python/runtime
2. payload build status and attestation reuse were too sensitive to generated
   workspace churn
3. dataset manifests embedded Windows host shard paths that were not portable
   into the Linux container
4. the local workstation exposed a CUDA device that the target PyTorch build
   could see but not execute on
5. persisted anchors could be reused even when they were incompatible with the
   active model state

WS30 closed those issues by:

- resolving payload dependency locks against the target image when Docker is
  available
- preserving build truth when the locked image inputs are unchanged
- reusing an existing locked image only when the attested build inputs still
  match
- adding portable dataset shard metadata and stale-manifest invalidation
- teaching the payload to resolve container shard paths from mounted `/data`
- falling back to CPU when CUDA is visible but unusable on the current host
- regenerating anchors when the stored anchor state does not match the current
  model keys/shapes

## Validation

Primary local validation bundle:

- `C:\Users\Chris\contLRN\ws30_local_validation_20260415_162923`

Key result:

- `live_docker_test.launched = true`
- `build_status = built`
- `build_attestation_id = build-b41d9eaab0988c7f`
- `image_digest = tar-payload@sha256:a21a7075f2600960c8c7e51fb333661740c9adef907a3e275f9ff1256d1702cb`

Supporting state:

- latest runtime status persisted the attestation and digest correctly
- locked-image execution used the attested payload image
- the local host remained truthful about GPU visibility while the payload ran on
  CPU fallback because the workstation GPU was not compatible with the target
  PyTorch CUDA build

Full regression status after the WS30 fixes:

- `282 passed, 10 warnings`

The warnings are the same pre-existing ASC/PyTorch warnings and are not WS30
regressions.

## Status

`WS30` is closed.

The next active workstream is `WS31: Canonical Benchmark Harnesses And
Statistical Validation`.
