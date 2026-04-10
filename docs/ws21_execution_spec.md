# WS21 Execution Spec

`WS21` is **Operator Interface Upgrade**.

Its purpose is to make TAR's post-`WS20` research state legible to a human
operator. TAR now has project continuity, prioritization, falsification
planning, and portfolio management. The next requirement is not more hidden
state. It is a readable control surface.

## Core purpose

After WS21, an operator should be able to answer quickly:

- what projects exist and which one is active
- what evidence supports or contradicts a project
- what happened recently in the project
- what claim or benchmark lineage exists for the project
- what would need to happen before the project resumes confidently

## Why WS21 matters

Without WS21:

- TAR stores rich inquiry state
- but most of that state is only visible as raw JSON or narrow point views
- operator understanding lags behind system sophistication
- the next dataset, evaluation, and training workstreams become harder to manage

WS21 closes that legibility gap.

## Strategic principle

WS21 is not a cosmetic UI pass.

It is an operator-surface workstream. The goal is to expose the structure of
inquiry clearly enough that later work on dataset generation, evaluation, and
model tuning can be managed responsibly.

## What WS21 must deliver

1. Operator overview surface

- A concise control view for:
  - project counts
  - top action candidates
  - top portfolio projects
  - stale projects
  - resume candidates
  - promotion-blocked projects

2. Project timeline view

- A chronological view of project-relevant events:
  - project creation
  - study generation
  - problem execution
  - research decisions
  - budget allocations
  - falsification plans
  - claim verdicts
  - pause / resume events
  - portfolio decisions affecting the project

3. Evidence map view

- A project-scoped evidence surface showing:
  - supporting evidence IDs
  - contradicting evidence IDs
  - cited research IDs
  - retrieved memory IDs
  - contradiction review summary
  - benchmark truth and comparability context
  - latest evidence debt state

4. Claim lineage view

- A project-scoped provenance view showing:
  - related studies
  - related executions
  - linked claim verdicts
  - linked benchmarks
  - linkage status and comparability source

5. Resume dashboard view

- A project-scoped resume/control view showing:
  - current status
  - latest resume snapshot
  - blockers
  - next action
  - budget remaining
  - latest staleness record
  - latest evidence debt
  - latest falsification plan summary

6. CLI and dashboard upgrades

- The operator should not have to inspect raw JSON for normal use.
- CLI should render the new views readably.
- Dashboard should expose the new views in a structured layout instead of only
  dumping latest raw records.

## File-by-file scope

### `tar_lab/state.py`

- Add audit-log iteration helpers.
- Expose append-only history cleanly enough for project timelines.

### `tar_lab/orchestrator.py`

- Add:
  - `operator_view`
  - `project_timeline`
  - `project_evidence_map`
  - `claim_lineage`
  - `resume_dashboard`
- Keep them project- and operator-oriented rather than generic JSON dumps.

### `tar_lab/control.py`

- Expose the new WS21 commands through the control server.

### `tar_lab/schemas.py`

- Extend `ControlRequest` command literals for the new WS21 operator views.

### `tar_cli.py`

- Add:
  - `--operator-view`
  - `--project-timeline`
  - `--evidence-map`
  - `--claim-lineage`
  - `--resume-dashboard`
- Add readable renderers for each new view.

### `dashboard.py`

- Reorganize into operator-oriented tabs/sections.
- Add a project selector and project-scoped views for:
  - resume state
  - evidence map
  - claim lineage
  - project timeline

## Design decisions

1. Prefer readable summaries over raw blobs.

2. Timeline should come from persisted state and audit history, not ad hoc memory.

3. Claim lineage should remain provenance-first, not narrative-first.

4. Resume view should emphasize what is blocking and what happens next.

5. WS21 should improve legibility without trying to become a polished end-user
   product.

## Acceptance criteria

WS21 closes only if all are true:

- TAR exposes an operator overview surface
- TAR exposes a project timeline surface
- TAR exposes a project evidence map
- TAR exposes a claim lineage view
- TAR exposes a resume dashboard view
- CLI renders those views readably
- dashboard exposes those views clearly
- regression tests prove the new operator surfaces are wired and truthful

## Recommended implementation order

1. add audit-log iteration in state
2. add orchestrator operator views
3. wire control commands
4. wire CLI flags and renderers
5. reorganize dashboard
6. add regression tests
7. run focused slice, then full suite

## What counts as done

WS21 is done when a human operator can look at TAR and answer:

- what is active
- what happened
- what supports or contradicts the current thread
- what claim lineage exists
- what should happen next
- what blocks confident resumption

without having to dig through raw state files by hand.
