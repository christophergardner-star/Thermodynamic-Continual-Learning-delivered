# WS35 Closeout

## Title

`WS35: Safe Execution Hardening`

## Outcome

`WS35` is complete.

The workstream now hardens both TAR's host-side execution boundary and its
container-runtime boundary:

- unsandboxed generated or external host execution is now policy-gated
- `safe_exec` applies capability drops, seccomp, and read-only workspace
  mounts
- `DockerRunner` applies the same security model to composed runtime commands,
  Docker SDK launches, and science-bundle execution
- runtime results now expose truthful sandbox audit logs

## Delivered

### WS35-A: Universal execution policy boundary

- persisted `TARExecutionPolicy` in `tar_state/policies/execution_policy.json`
- `ExecutionPolicyViolation` for prohibited unsandboxed execution
- orchestrator-owned execution classification:
  - `generated_code`
  - `external_code`
  - `trusted_internal`
- explicit trusted-internal exceptions for:
  - `tar_lab.problem_runner`
  - `tar_lab.orchestrator._active_process_commands`

### WS35-B: SafeExec container hardening

- `SandboxedPythonExecutor` now applies:
  - `--cap-drop=ALL`
  - `--security-opt no-new-privileges`
  - explicit seccomp profile wiring
  - read-only workspace mount at `/workspace`
  - writable sandbox/artifact mount at `/sandbox`
- `SandboxPolicy` now records:
  - `seccomp_profile_path`
  - `capability_drop`
  - `workspace_read_only`
- `SandboxExecutionReport` now records `sandbox_audit_log`
- default seccomp profile added in
  `tar_lab/sandbox_profiles/default_seccomp.json`

### WS35-C: DockerRunner security alignment

- `DockerRunner` now translates sandbox policy into real Docker security flags
- security policy is applied consistently to:
  - dry-run launch command composition
  - Docker SDK launches
  - `run_science_environment()` execution
- read-only workspace enforcement is preserved while explicit artifact mounts
  remain writable
- `LaunchResult` and `CommandResult` now carry `sandbox_audit_log`

## Validation

Local regression coverage passed before closeout:

- `python -m pytest tests\test_safe_exec.py -q`
- `python -m pytest tests\test_docker_runner.py -q`
- `python -m pytest tests\test_tar_foundation.py -k "safe_execution_mode or frontier_status_reports_new_foundations or problem_runner_executes or docker_command_carries_runtime_manifests or compose_command_carries_runtime_manifests or docker_runner or runtime_status_surfaces_sandbox_mount_policy or sandboxed_python_executor_reports_unavailable_without_host_fallback" -q`

Result summary:

- `test_safe_exec.py`: `5 passed`
- `test_docker_runner.py`: `9 passed`
- targeted foundation/runtime/sandbox slice: `14 passed`

Bounded hardening validation then passed on the local Docker host:

- host: Docker Desktop `29.3.1`
- context: `desktop-linux`
- artifact bundle:
  `C:\Users\Chris\contLRN\ws35_local_validation_artifacts.tar`
- SHA256:
  `4B567CF226DEBC3A33025D736C2FA56A045BDB23CFDDA730091991FEA96E2FEF`

Validated behaviors:

- privileged syscall attempt blocked:
  - hardened `docker run` probe returned `res=-1 errno=1` for `ptrace`
- workspace-root write blocked:
  - hardened `docker run` probe failed with `Read-only file system`
- writable artifact path still works:
  - `SandboxedPythonExecutor` wrote successfully under `/sandbox`
  - `DockerRunner.run_science_environment()` wrote successfully to
    `/workspace/tar_runs/docker_runner_ok.txt`
- `DockerRunner` audit logs are truthful:
  - command surfaces contain `--cap-drop=ALL`
  - command surfaces contain `no-new-privileges`
  - command surfaces contain the configured seccomp profile
  - returned `sandbox_audit_log` matches the applied read-only and writable
    mount policy

## Pod Posture

No pod was required for the final proof.

The correct validation host for `WS35` was the local Docker workstation, not a
generic RunPod pod without a reliable host Docker daemon.

## What Is Next

The next active item is the independent `WS27R2` eval validation pass.

That pass should:

- build a sealed external eval slice
- score with an independent scorer
- compare external results honestly against the internal WS27R1 pack
- update published WS27R2 scoring claims if the external slice shows inflation
