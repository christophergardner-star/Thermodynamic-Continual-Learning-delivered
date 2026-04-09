# WS20 Execution Spec

`WS20` is **Research Portfolio Management**.

Its purpose is to move TAR from managing single projects well to managing a portfolio of open research programs with explicit cross-project governance.

## Core purpose

After WS20, TAR should be able to answer:

- which projects are active
- which projects are paused, blocked, stale, parked, completed, or abandoned
- which project deserves the next budget
- which projects should resume, defer, escalate, or retire
- which projects carry too much evidence debt to promote confidently

## Why WS20 matters

Without WS20:

- TAR can manage project continuity and action ranking
- but it still lacks durable portfolio state
- it cannot explicitly govern stale projects
- it cannot justify switching between projects at the lab level
- it cannot expose portfolio health or evidence debt clearly

WS20 closes that gap.

## Strategic principle

WS20 is not about pretending TAR can autonomously discover everything.

It is about making TAR able to say:

- this is the portfolio
- this is the current active program
- this is what is stale
- this is what is blocked
- this is what should resume next
- this is what should not be promoted yet

## What WS20 must deliver

1. First-class portfolio state

- TAR needs a durable portfolio object above project-level continuity.
- It must track active, paused, blocked, stale, parked, completed, and abandoned projects.

2. Explicit cross-project ranking

- TAR should rank projects using:
  - strategic priority
  - expected value of continuation
  - evidence debt
  - contradiction pressure
  - staleness
  - budget pressure
  - benchmark readiness

3. Stale-thread recovery

- TAR should detect projects that have gone cold.
- It should surface resume candidates and closure candidates explicitly.

4. Portfolio-level decisions

- TAR should persist decisions such as:
  - continue
  - defer
  - park
  - resume
  - escalate
  - retire

5. Evidence debt accounting

- Each project should expose:
  - falsification gap
  - replication gap
  - benchmark gap
  - claim-linkage gap
  - calibration gap
  - overall debt
  - whether promotion is blocked

6. Operator-visible portfolio surfaces

- CLI and dashboard should show:
  - top portfolio project
  - stale projects
  - resume candidates
  - evidence debt
  - latest portfolio decision
  - portfolio health snapshot

## Target concepts

- `ResearchPortfolio`
- `PortfolioHealthSnapshot`
- `ProjectPriorityRecord`
- `EvidenceDebtRecord`
- `ProjectStalenessRecord`
- `PortfolioDecision`

## File-by-file scope

### `tar_lab/schemas.py`

- Add portfolio, staleness, evidence-debt, and decision schemas.
- Add new control commands for portfolio review and decision surfaces.

### `tar_lab/state.py`

- Persist portfolio state and append-only portfolio records.
- Expose latest portfolio state in status payload.

### `tar_lab/orchestrator.py`

- Add:
  - `portfolio_review`
  - `portfolio_decide`
  - `stale_projects`
  - `evidence_debt`
  - `resume_candidates`
- Compute portfolio state using WS17 continuity, WS18 prioritization, and WS19 falsification state.

### `tar_lab/control.py`

- Expose WS20 commands through the control server.

### `tar_cli.py`

- Add:
  - `--portfolio-review`
  - `--portfolio-decide`
  - `--stale-projects`
  - `--evidence-debt`
  - `--resume-candidates`
- Render portfolio truth clearly.

### `dashboard.py`

- Show portfolio health, latest portfolio decision, stale projects, and latest evidence debt.

## Design decisions

1. Portfolio state must be explicit, not reconstructed ad hoc.

2. Evidence debt should block promotion, not progress.

3. Stale is distinct from paused, blocked, parked, retired, or completed.

4. WS20 should stay governance-focused, not become fake full autonomy theater.

## Acceptance criteria

WS20 closes only if all are true:

- TAR has explicit portfolio state
- TAR can rank projects across the portfolio
- TAR can identify stale and resume-worthy projects
- TAR can compute evidence debt per project
- TAR can persist portfolio decisions
- operator surfaces expose active, deferred, stale, blocked, and retired truth clearly
- regression tests prove switching, stale recovery, and evidence-debt logic

## Recommended implementation order

1. add schemas
2. add state persistence
3. add orchestrator review/decision logic
4. wire control and CLI
5. add dashboard visibility
6. add regression tests
7. validate locally
8. sync and validate on the pod

## What counts as done

WS20 is done when TAR can truthfully say:

- this is the portfolio
- this is the next project to work on
- these are the stale and resume-worthy projects
- this project has too much evidence debt to promote
- this is why another project was deferred or retired
