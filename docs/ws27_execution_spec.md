# WS27 Execution Spec

`WS27` is `Large-Model ASC/TCL Research Branch`.

## Purpose

`WS25` and `WS26` proved that the TAR operator path works: a 7B instruct
backbone can be adapted into a disciplined TAR/TCL operator with strong held-out
gains. `WS27` does not replace that path. It asks a new and riskier question:

- should the actual ASC/TCL algorithmic stack be ported into a larger-model
  branch beyond the TAR operator adapter path?

This workstream is therefore a controlled research branch with a required
go/no-go decision, not an assumed continuation of the mainline.

## Core Question

Can a larger-model ASC/TCL branch produce meaningful value beyond the current
TAR operator path without unacceptable regression in honesty, reproducibility,
structured-output reliability, or cost discipline?

## Research Posture

This is a branch, not the default stack. `WS27` should be judged against:

- prompt-only base model
- `WS25` TAR operator adapter
- `WS26` TCL-deepened operator adapter

The branch only earns continuation if it shows value that these baselines do
not already provide.

## Hypotheses

- `H1`: a large-model ASC/TCL branch can improve TCL-specific intervention and
  runtime judgement beyond the `WS26` adapter baseline.
- `H2`: a larger branch can improve ASC/TCL-domain reasoning without giving
  back benchmark honesty, reproducibility honesty, or overclaim control.
- `H3`: the engineering and compute overhead of the branch can be bounded
  tightly enough to support a disciplined research cycle.
- `H4`: some candidate implementations will fail the cost/complexity test, and
  `WS27` must be willing to conclude `no-go` if the gains are not real.

## Non-Goals

- no from-scratch large-model training
- no uncontrolled multi-week compute campaign
- no replacement of the proven TAR operator baseline
- no hand-wavy “it ran, therefore it worked” result
- no GPU rental before the branch design and evaluation contract are frozen

## Branch Types To Consider

`WS27` must explicitly choose one primary branch type before any major run.

### Candidate A: Operator-Plus-Auxiliary-TCL

Start from the proven TAR operator path and add:

- TCL-specific auxiliary supervision
- TCL diagnostics and intervention heads at the data/objective level
- stronger run-trace conditioning

This is the safest first branch and the recommended first implementation.

### Candidate B: Partial ASC/TCL Objective Port

Retain a large LM backbone but introduce a bounded subset of ASC/TCL-style
training dynamics, such as:

- stability-aware auxiliary losses
- anchor-aware consistency objectives
- recovery-oriented training traces

This is higher risk and should follow only after Candidate A is understood.

### Candidate C: Broad Large-Model ASC Scaffold

Use the experimental large-model ASC scaffold, but only under explicit
experimental labeling and only after proving:

- fit
- checkpoint integrity
- logging integrity
- evaluation compatibility

This should not be the first serious path.

## Current Repo Constraints

The repo already contains an experimental large-model ASC scaffold in
[deepseek_asc_finetune.py](c:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/deepseek_asc_finetune.py).
That file explicitly warns that it is not yet a scientifically validated
large-model ASC trainer. `WS27` must respect that warning.

Therefore:

- the first `WS27` experiments should be framed as experimental branch probes
- the branch must not be presented as canonical ASC training
- if the branch uses that scaffold, it must be labeled as such in manifests and
  docs

## Deliverables

`WS27` must deliver all of the following:

- a formal branch design and experiment contract
- explicit backbone and precision decisions
- explicit adapter/full-tune/objective choice
- `WS27`-specific train/eval configs
- a fit-probe result
- at least one serious branch run
- evaluation against `WS25`/`WS26` baselines
- a final go/no-go memo

## Required Experimental Outputs

- `ws27_branch_design.md`
- `ws27_execution_spec.md`
- one or more `WS27` train configs
- one or more `WS27` eval runtime configs
- fit-probe manifest
- training run manifest
- held-out comparison report
- final `go` or `no-go` decision note

## Evaluation Standard

The branch must be evaluated on the same class of measures that matter for the
mainline system:

- decision accuracy
- mean score
- parse error rate
- overclaim rate
- benchmark honesty
- reproducibility honesty
- falsification/verification discipline
- TCL reasoning quality
- TCL intervention quality
- TCL recovery planning quality

The branch is not acceptable if it improves one dimension while materially
regressing the truthfulness or control contract that defines TAR.

## Success Criteria

`WS27` is a `go` only if all of the following are true:

- it improves at least one important TCL/ASC-specific capability over `WS26`
- it does not materially regress honesty metrics
- it does not materially regress parse reliability
- the engineering path is understandable and repeatable
- the cost/complexity burden is justified by the gain

`WS27` is a `no-go` if any of the following dominate:

- gains are marginal versus `WS26`
- regressions in honesty or reproducibility appear
- fit or training stability is fragile enough to make iteration impractical
- the branch becomes more expensive than its scientific value justifies

## Execution Order

### Phase A: Local Branch Design

Local only. No pod.

Tasks:

1. choose the primary branch type
2. define exact backbone candidates
3. define objective strategy
4. define training recipe candidates
5. define evaluation comparison plan
6. define artifact/manifest policy

Exit condition:

- the branch design is frozen in docs/config form

### Phase B: Local Scaffolding

Local only. No pod.

Tasks:

1. add `WS27` configs
2. add any branch-specific scripts
3. add any manifest/runtime metadata needed
4. add tests for config parsing or branch-specific packaging where possible

Exit condition:

- the first fit probe can be run without additional planning work

### Phase C: Feasibility Probe

This is the **first justified pod moment**.

Tasks:

1. pull the current repo
2. install/bootstrap
3. download/cache the chosen backbone
4. run a short fit probe
5. verify:
   - memory fit
   - optimizer/precision compatibility
   - checkpoint writing
   - manifest writing
   - eval compatibility

Exit condition:

- either the branch fits and is stable enough to continue, or it is rejected
  early

### Phase D: First Serious Branch Run

Pod required.

Tasks:

1. run one serious branch experiment
2. preserve manifests/checkpoints/results
3. run held-out evaluation
4. compare directly to `WS25` and `WS26`

Exit condition:

- comparative evidence exists

### Phase E: Go/No-Go Decision

Local only after artifacts are secured.

Tasks:

1. inspect results
2. analyze failure buckets
3. assess cost/benefit
4. record `go` or `no-go`

## Pod Policy

### Do Not Open A Pod Yet For WS27

Do not rent a pod while any of these are still true:

- the branch design is still changing
- configs are not frozen
- eval comparisons are not defined
- the next step is still reading/thinking/writing docs

### Open The Pod When

Open a pod only when all of the following are true:

1. the `WS27` branch design is written
2. the first branch config is committed locally
3. the eval comparison contract is written
4. a fit probe is the immediate next step

This is the earliest justified pod point.

### Pod Recommendation

Preferred:

- `A100 80GB`
- persistent volume enabled
- `300GB+` disk

Acceptable for fit probe only:

- `L40S 48GB`, if the branch remains adapter-based and carefully bounded

Why:

- `WS27` is riskier and heavier than `WS25`/`WS26`
- you want memory headroom for branch experiments, not constant firefighting

### What To Use The Pod For

- model download/cache
- fit probe
- first serious branch run
- post-train eval
- artifact bundling and sync-back

### What Not To Use The Pod For

- roadmap writing
- doc-only work
- result interpretation alone
- schema-only or CLI-only work
- commit/push cleanup

### When To Terminate The Pod

Terminate the pod when all are true:

1. the branch run is complete
2. artifacts are copied back locally
3. results/manifests are secured
4. no immediate next GPU-bound command remains

That means:

- do not leave the pod running while doing local analysis
- do not leave it running while writing the go/no-go memo

## Recommended WS27 First Pass

The recommended first pass is:

1. adapter-based large-model branch
2. bounded TCL/ASC auxiliary objective expansion
3. one feasibility probe
4. one serious run
5. direct comparison to `WS26`
6. explicit go/no-go call

This is the shortest path to a defensible result.

## Acceptance Criteria

`WS27` is only ready for pod-backed execution if:

- the branch type is chosen
- the training recipe is defined
- the comparison baselines are defined
- manifests and outputs are specified

`WS27` only closes if:

- the branch was actually evaluated against `WS25`/`WS26`
- a clear go/no-go decision was written
- the decision is based on measured evidence, not enthusiasm
