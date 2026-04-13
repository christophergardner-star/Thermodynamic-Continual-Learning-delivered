## WS28: Native Inference Integration And Operator Serving

### Purpose
Make the validated `WS27-R2` operator line a first-class TAR runtime component.

### Problem Statement
TAR already has:
- checkpoint registration
- endpoint planning and lifecycle
- local OpenAI-compatible serving

But it still lacks:
- one authoritative active operator checkpoint
- a persisted distinction between `prompt_only` and `tuned_local`
- a truthful adapter-backed serving path for the validated operator line
- operator-visible status for which checkpoint and endpoint TAR should actually use

### Scope
WS28 integrates the best validated operator checkpoint into the runtime without starting a new training branch.

### Core Deliverables
1. Checkpoint registry support for adapter-backed checkpoints.
2. Persisted operator-serving state in `tar_state`.
3. Explicit operator mode selection:
   - `prompt_only`
   - `tuned_local`
4. Adapter-aware local serving in [serve_local.py](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/serve_local.py).
5. CLI/control/orchestrator support for:
   - selecting the active operator checkpoint
   - inspecting operator-serving status
6. Status/dashboard visibility for the active operator line.
7. Regression coverage for registry, selection, endpoint planning, and adapter manifest behavior.

### Design Rules
- The active operator checkpoint must be explicit and persisted.
- `prompt_only` must point to a base checkpoint.
- `tuned_local` must point to an adapter-backed checkpoint.
- Endpoint plans must tell the truth about whether serving is base-only or base-plus-adapter.
- WS28 is local-first. No pod is required for implementation.

### Data Model
WS28 introduces a persisted operator-serving state containing:
- active checkpoint name
- selected operator mode
- selected role
- selected endpoint name
- timestamp of selection

Checkpoint records must also support adapter-backed serving metadata:
- `checkpoint_kind`
- `base_model_id`
- `adapter_path`

### Command Surface
Add:
- `select_operator_checkpoint`
- `operator_serving_status`

CLI flags:
- `--select-operator-checkpoint`
- `--operator-serving-status`
- `--operator-mode`
- `--base-model-id`
- `--adapter-path`

### Acceptance Criteria
WS28 closes only if all are true:
- TAR can register an adapter-backed operator checkpoint cleanly.
- TAR can persist one active operator selection.
- TAR can report operator-serving status through orchestrator, control, and CLI.
- Endpoint plans for adapter-backed checkpoints include the correct base model and adapter path.
- Local serving manifests expose adapter-backed configuration truthfully.
- Dashboard and status output surface the active operator model.
- Regression tests cover the new behavior.

### Non-Goals
- No new training.
- No pod-only throughput work.
- No distributed serving.
- No production auth layer.
