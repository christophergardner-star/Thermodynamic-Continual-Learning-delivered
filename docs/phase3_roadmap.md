# Phase 3 Roadmap

This is the authoritative forward roadmap after completion of:

- `WS1-WS7` foundation phase
- `WS8-WS16` remediation phase
- `WS17-WS20` first post-remediation autonomy/control phase

The project should no longer be managed as if it were still in the remediation
roadmap. The correct professional framing now is:

- historical phases are closed
- active work begins from `WS21`
- priorities should follow the current bottleneck, not the oldest numbering

## Current State

TAR now has:

- benchmark-honest science execution
- reproducibility hardening
- runtime sandbox control
- endpoint observability
- project continuity and budgeting
- evidence-budgeted prioritization
- falsification planning
- portfolio management
- a TAR master dataset builder
- a secure 7B TAR operator training path
- the first TCL-native dataset expansion

Current repo state at lock-in:

- branch: `main`
- pushed commit: `0686d4c`
- suite status: `218 passed, 8 warnings`

## Guiding Principle

The next phase should be managed the way a serious research engineering group
would manage it:

- preserve completed workstreams as historical record
- prioritize the current bottleneck
- separate product/interface work from model/data work
- only spend pod/GPU budget where local validation is no longer the right tool

## Archived Phases

### Completed Foundation

- `WS1-WS7`

### Completed Remediation

- `WS8`
- `WS9`
- `WS10`
- `WS11`
- `WS12`
- `WS13`
- `WS14`
- `WS15`
- `WS16`

### Completed First Autonomy/Control Phase

- `WS17`
- `WS18`
- `WS19`
- `WS20`

## Active Forward Roadmap

### `WS21`: Operator Interface Upgrade

Purpose:

- make TAR's internal inquiry state legible to an operator

Core deliverables:

- project timeline views
- evidence maps
- contradiction maps
- budget-burn views
- claim lineage panels
- resume-point dashboards

Why it matters:

- TAR now has enough internal state that purely low-level operator views are no
  longer sufficient

Execution posture:

- laptop-first
- pod not required

### `WS23`: TAR Master Dataset Scale-Up

Purpose:

- grow the TAR operator dataset from seed scale to serious training scale

Core deliverables:

- more project continuity traces
- more prioritization and portfolio records
- more falsification plans
- more refusal and downgrade cases
- more TCL runtime traces and recovery cases
- versioned dataset releases

Why it matters:

- the current 7B training path is real, but the dataset is still too small for
  a genuinely strong first run

Execution posture:

- laptop-first for builder work and small merges
- pod optional only when state volume, storage, or batch generation becomes too
  large for the workstation

### `WS24`: TAR/TCL Evaluation Harness

Purpose:

- build a held-out evaluation layer for TAR-native operator behavior

Core deliverables:

- resume fidelity evaluation
- next-action quality evaluation
- benchmark honesty evaluation
- falsification quality evaluation
- contradiction-handling evaluation
- TCL regime diagnosis evaluation
- TCL recovery planning evaluation

Why it matters:

- a serious fine-tune without a serious eval harness is weak science

Execution posture:

- laptop-first
- pod only when evaluating large checkpoints repeatedly

### `WS25`: Serious 7B TAR Operator Training

Purpose:

- run the first scientifically defensible TAR operator fine-tune

Core deliverables:

- baseline prompt-only comparison
- tuned-model evaluation
- held-out results
- saved manifests and run summaries
- ablation-ready run bookkeeping

Why it matters:

- this is the first point where TAR's operator-model path stops being a smoke
  path and becomes a real research result

Execution posture:

- pod or workstation GPU required
- this is the first workstream that clearly justifies renting a pod

### `WS26`: TCL-Native Operator Deepening

Purpose:

- strengthen the operator model's understanding of TCL-specific runtime and
  intervention logic

Core deliverables:

- richer TCL-native dataset families
- stronger TCL eval suites
- better thermodynamic trace interpretation
- better recovery/intervention recommendations

Why it matters:

- TAR should not only manage research well in general; it should become
  especially strong at the TCL domain it is built around

Execution posture:

- laptop-first for dataset and eval design
- pod when retraining or evaluating larger checkpoints

### `WS22`: Publication Handoff Layer

Purpose:

- package structured, evidence-backed findings for a later writing subsystem

Core deliverables:

- accepted-claim bundles
- rejected-alternative bundles
- experiment lineage packages
- benchmark-truth attachments
- limitations and open-questions exports

Why it matters:

- publication support should sit on top of strong evidence packages, not
  compensate for their absence

Execution posture:

- laptop-first
- pod not required

### `WS27`: Large-Model ASC/TCL Research Branch

Purpose:

- explore whether the actual ASC/TCL algorithmic stack should be ported into a
  larger-model branch beyond the TAR operator adapter path

Core deliverables:

- a separate experimental branch
- large-model backbone support decisions
- evaluation against TAR-operator-only baselines
- a clear go/no-go result on deeper large-model TCL adaptation

Why it matters:

- this is the heavier and riskier research branch and should only begin once the
  operator-model path is already strong

Execution posture:

- pod/workstation GPU required
- should not be started before `WS25` and early `WS26` results justify it

## Recommended Execution Order

The recommended professional order is:

1. `WS21`
2. `WS23`
3. `WS24`
4. `WS25`
5. `WS26`
6. `WS22`
7. `WS27`

## Why This Order Is Correct

`WS21` first:

- TAR now has enough internal inquiry state that legibility matters

`WS23` and `WS24` before `WS25`:

- the present bottleneck is dataset scale and evaluation rigor, not core system
  plumbing
- a serious model run without those two pieces is technically possible but
  scientifically weak

`WS26` before `WS22`:

- the domain-deepening work is a stronger research priority than publication
  packaging at this point

`WS27` last:

- the larger-model ASC/TCL branch should only start after the operator-model
  branch has earned it

## Pod Hiring Policy

Pod/GPU time should be rented only when the local workstation is no longer the
right environment for the work.

### Do not hire a pod for

- roadmap writing
- docs work
- schema design
- CLI rendering work
- dashboard-only work
- local builder logic
- regression tests that already pass comfortably on the workstation

### Hire a pod for

- 7B smoke and serious training
- large checkpoint evaluation
- repeated long-running model comparisons
- large TAR state merges that stress local disk or RAM
- runtime campaigns intended to generate lots of TCL traces quickly

### Earliest justified pod point

The first clearly justified pod hire in this roadmap is:

- `WS25`

That assumes:

- `WS23` has produced a materially larger dataset
- `WS24` has produced a real held-out eval harness
- the training config is frozen enough to justify GPU spend

### Pod guidance by workstream

`WS21`

- pod not needed

`WS23`

- pod only if:
  - state roots are large
  - storage pressure is high
  - you want long batch generation sessions

`WS24`

- pod only if evaluation requires repeated large-model inference

`WS25`

- yes, rent a pod
- recommended posture:
  - A100 80GB or equivalent
  - persistent storage
  - enough free space for base model, caches, checkpoints, and run artifacts

`WS26`

- use pod when retraining or running repeated model evaluations

`WS22`

- pod not needed

`WS27`

- yes, but only after a clear research question exists

## Financial Planning Note

For the first serious 7B TAR operator run, a sensible planning budget is:

- around `$50-$60`

That is not because the cleanest successful run must cost that much. It is
because that budget safely covers:

- dataset refresh
- dry run
- one serious training run
- evaluation
- some restart and correction margin

## Current Priority

The current bottleneck is not core stack repair anymore.

The proper immediate priority is:

- make the dataset and evaluation stack strong enough that `WS25` is worth
  paying for

That means the most sensible immediate focus is:

1. `WS21`
2. `WS23`
3. `WS24`

## Success Condition For This Phase

This phase succeeds when TAR can do all of the following credibly:

- manage multiple research threads clearly
- explain why it picked a given next action
- generate falsification pressure against its own claims
- surface portfolio-level state legibly
- train a bespoke operator model on real TAR/TCL behavior
- evaluate that model on held-out TAR-native tasks

Only after that should the project spend major effort on publication handoff or
heavier large-model TCL branches.
