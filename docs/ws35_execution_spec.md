# WS35 Execution Spec

## Scope

`WS35` hardens TAR's execution boundary so generated or external code cannot run on the host without an explicit policy decision.

The workstream is being delivered in slices:

- `WS35-A: Universal Execution Policy Boundary`
- `WS35-B: SafeExec Container Hardening`
- `WS35-C: DockerRunner Security Alignment`

## Deliverables

1. Add persisted `TARExecutionPolicy` state under `tar_state/policies/execution_policy.json`.
2. Add `ExecutionPolicyViolation` for prohibited unsandboxed execution.
3. Route orchestrator-owned subprocess execution through a checked boundary.
4. Classify execution surfaces as:
   - `generated_code`
   - `external_code`
   - `trusted_internal`
5. Raise on unsandboxed generated or external execution when policy requires sandboxing.
6. Allow trusted internal execution only when it is explicitly documented or allow-listed.
7. Surface the active execution policy in runtime status.

## WS35-B Deliverables

1. Harden `SandboxedPythonExecutor` with:
   - `--cap-drop=ALL`
   - `--security-opt no-new-privileges`
   - explicit seccomp profile wiring
2. Mount the repo workspace read-only at `/workspace`.
3. Keep `/sandbox` as the only writable execution/artifact area.
4. Add `sandbox_audit_log` so reports record the applied mount/security policy.
5. Ship a bundled default seccomp profile under `tar_lab/sandbox_profiles/default_seccomp.json`.

## WS35-C Deliverables

1. Translate sandbox-policy fields into `docker run` security flags inside `DockerRunner`.
2. Apply the same security model to:
   - composed runtime commands
   - Docker SDK launches
   - science-bundle `run_command` execution
3. Force read-only workspace mounts when policy requires them.
4. Emit `sandbox_audit_log` from Docker runner results so applied security policy is inspectable.

## Boundaries

`WS35-C` aligns runtime execution with the hardened policy model. It does not yet prove that the host Docker runtime actually enforces those flags under a real pod. That is the next validation step.

This slice also does not force every TAR-internal helper through the sandbox. Trusted internal runners remain allowed when they are explicitly marked.

## Initial Trusted Internal Exceptions

- `tar_lab.problem_runner`
  Reason: TAR-owned study runner invoked as a local module.
- `tar_lab.orchestrator._active_process_commands`
  Reason: host process inspection for orphan detection and runtime health.

## Acceptance

`WS35-A` is complete when all are true:

- unsandboxed generated-code subprocess execution from orchestrator context raises `ExecutionPolicyViolation`
- documented trusted-internal subprocess execution remains functional
- execution policy persists in state and is visible through runtime status
- targeted regression coverage passes
