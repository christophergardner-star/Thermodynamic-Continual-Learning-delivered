# Phase 1: What TAR Is For / What TAR Must Never Do

Author: Christopher Gardner
Drafted: 2026-05-10
Status: Draft - gate document for the TAR rebuild

This is the design gate for all subsequent TAR rebuild work. Phases 2
onward (rails, code changes, restart) must be checkable against this
document. If a code change cannot be justified by something written
here, it should not be made.

---

## 1. Why TAR exists

TAR exists because a single self-taught researcher working evenings
and weekends cannot cover the full surface area of a serious continual
learning research programme alone. Manual scripts run experiments;
they do not plan the next one, ingest literature, track claims, hold
methodology consistent across phases, or notice when a result drifts
from what has been claimed in a paper draft.

TAR was built to compress that work. It is research infrastructure for
one person doing what is normally a small team's worth of project
management, experimental orchestration, literature awareness, and
result tracking - applied specifically to thermodynamic continual
learning, with the broader aim of supporting a research programme that
extends across TCL, EPTO, ASC, and future thermo-ML work.

It is not a research agent in the autonomous-AI sense. It is a tool
that does more of the labour around research than a script can,
without making the scientific decisions itself.

---

## 2. What TAR is for - the four modes

### 2.1 Run authorised experiments

TAR executes experiment plans that the user has explicitly authorised.
This includes long overnight or multi-day campaigns, on the user's
local hardware or on rented compute. The value is throughput: TAR
running through an authorised queue while the user works the day job
or sleeps means more experiments completed per week than manual
launches allow.

Success in this mode is: every executed run can be traced back to a
specific authorisation event. No run executes that the user did not
sanction in advance. Result files are written with full provenance.

#### Chain runner (run_rerun_chain.py)

The chain runner is an authorised execution mode with restricted
scope. It MAY:
  - Poll the canonical index for completion signals.
  - Execute experiments listed in pre-signed manifests committed
    to git, in the order specified.
  - Write status files (active_rerun.json, rerun_chain_state.json)
    to tar_state/ for dashboard observability.

The chain runner MUST NOT:
  - Author or modify manifests.
  - Decide which experiments to run beyond the committed sequence.
  - Persist across reboots without explicit re-authorisation.
  - Trigger experiments not listed in a signed manifest, regardless
    of any internal logic that might suggest they should run.

If the chain runner encounters a state not covered by its signed
manifests, it must halt and write a blocker note rather than
attempting to recover autonomously.

### 2.2 Plan and queue experiments for review

TAR proposes experiments to run next, based on what is currently
unanswered, what reviewers will likely ask, what extension is most
informative, or what the user has flagged as a research priority.
These proposals go into a review queue, not into execution.

The value is reducing the cognitive load of "what should I run next."
Coming home from a shift and finding three pre-formatted experiment
proposals ready to evaluate is much better than coming home and having
to think from scratch.

Success looks like: the proposal queue accelerates user decisions
without ever bypassing them.

### 2.3 Track and document research

TAR maintains the project's state across phases - what experiments
ran, what results came back, what claims are made in what papers,
what is contradicted by what data, and what is still pending. The May
2026 audit happened because TAR could surface a contradiction; that is
mode 2.3 working as intended.

Success looks like: the user can ask "where am I in the project" and
get an accurate, current answer in under a minute. Any contradiction
between a paper claim and a result file is detected and flagged within
the next session.

### 2.4 Generate hypotheses for selection

TAR proposes new directions, mechanisms, or developmental ideas based
on the current corpus of results plus ingested literature. These are
suggestions, not plans. They feed mode 2.2 (planning) only after the
user selects which ones to explore.

The value is breadth: a single researcher cannot read all relevant
literature or hold all the cross-domain analogies in mind. TAR can
surface ideas the user might not generate alone, including ones that
turn out to be wrong but useful.

Success looks like: at least one TAR-suggested hypothesis per month
that the user judges worth exploring further. Most suggestions will
be rejected; that is fine.

---

## 3. What TAR must never do

### 3.1 Never overwrite canonical result files

The May 2 incident was caused by an autonomous TAR campaign
overwriting `phase10_baseline.json` in place. That single failure
nearly cost the paper's headline finding. No code path in TAR may
ever write to a canonical result path that already exists. Result
files are append-only with unique names. This is non-negotiable
because the alternative is silent data corruption that may not be
detected for weeks.

### 3.2 Never trigger experiments without explicit authorisation

The April 29 scheduled task gave TAR open-ended permission to run
"the campaign" with TAR deciding what experiments that included. That
class of permission is what produced the May 2 overwrite. From now
on, any experiment that runs must be traceable to a specific
authorisation event - a manifest file, a CLI command, or a committed
queue. TAR cannot decide on its own to run anything. The user is the
execution gate.

### 3.3 Never persist across reboots without explicit justification

The Windows scheduled task and Startup folder entry meant TAR could
run autonomously even when the user had no awareness it was active.
Going forward, persistence is opt-in per-instance, not on by default.
If TAR needs to run after a reboot, that decision is made
deliberately, with a written justification, not as a side-effect of
installing a watchdog.

### 3.4 Never produce verdict labels that bypass human review

TAR's classifier strings (`PENALTY_DOMINANT`, `BREAKTHROUGH`, and
similar labels) overstated what the underlying statistics supported
in multiple cases during the audit. Verdict labels that drive paper
claims are unsafe. Going forward: TAR can produce advisory labels,
but they live in a separate field clearly marked as opinion, and no
paper section cites a verdict label as evidence. The numerical
statistics are evidence; the labels are commentary.

---

## 4. The authorisation model

The four modes have different authorisation patterns. Stricter
authorisation applies to higher-stakes operations.

**Mode 2.1 (Run):** Authorisation is a manifest file. The manifest
specifies the exact experiments to run, their parameters, their
output paths, and their hard time/run limits. The manifest is
human-written or human-reviewed before TAR reads it. TAR cannot
execute anything not in the manifest. Manifests live in the git repo
on `C:` as the source of truth and are committed to git as part of
the audit trail. Runtime copies may exist in the `E:` workspace, but
the env snapshot must record the manifest path and content hash of the
repo copy.

**Mode 2.2 (Plan):** Authorisation is implicit - TAR can plan
freely, write proposals to a review queue, and update planning state,
but proposals do not auto-promote to manifests. The user reviews the
queue and explicitly moves a proposal into a manifest before it can
run.

**Mode 2.3 (Track):** No authorisation is needed for read-only
tracking and documentation. TAR can monitor state, detect
contradictions, and generate reports. It cannot modify canonical
files, and any reports it generates must be marked as TAR-generated.

**Mode 2.4 (Generate):** No authorisation is needed for hypothesis
generation in a sandboxed proposals area. Generated hypotheses are
suggestions only, surfaced for user review during planning. They do
not auto-promote to anything.

The principle: as the operation moves from reading to writing to
executing, the authorisation requirement strengthens. Reading is
free; writing requires location-bounded permissions; executing
requires per-run manifests.

---

## 5. Write domains

Write permissions in TAR must be scoped by domain, not just by
function.

**Canonical experiment result domains**
- `tar_state/comparisons/`
- `tar_state/experiments/`

These are append-only, unique-name result stores. Every canonical
result written here must have a sibling `_env.json`. No overwrite is
ever allowed.

**Audit domain**
- `tar_state/stat_audit/`

This stores audit notes, reconciliation documents, blocker reports,
and similar artifacts. Append-only is strongly preferred. Historical
anomalies, including the May 2 overwrite artifact, belong here rather
than in canonical result directories.

**Literature / ingest domain**
- `tar_state/literature/`
- isolated ingest caches and manifests

These are non-canonical ingest areas. `tar_evidence_ingest.py` may
write here, but it must never touch canonical experiment result
directories.

**Paper / draft domain**
- `paper/`

This is a draft workspace. It is never itself a source of empirical
truth. Papers may report canonical results, but they do not define
them.

**Planning / proposal domain**
- planning queues
- proposal manifests
- user-authorised run manifests

Planners may write proposals here. Execution workers may read only
user-authorised manifests from this domain.

---

## 6. Component intent - keep, rewrite, or retire

Based on the codebase as it currently exists, this is the initial
disposition for the major TAR components. The user will revise this as
Phase 2 proceeds.

**Keep, with rails added**
- `tar_dashboard.py` - low-risk state display; current heartbeat
  writes must remain non-canonical and isolated
- `tar_runtime_tracking.py` - tracking infrastructure, mode 2.3
- `tar_project_registry.py` - project state, mode 2.3
- `tar_hardware_monitor.py` - read-only runtime monitoring
- `tar_evidence_ingest.py` - literature ingest, but only with write
  boundaries confined to literature / ingest domains
- `tar_research_director.py` - proposal generation, mode 2.4
- `tar_scheduler.py` - queue management, mode 2.2
- `tar_queue_bridge.py` - queue-to-manifest interface, needs
  hardening
- `tar_experiment_preflight.py` - safety check infrastructure
- `tar_experiment_worker.py` - execution under manifest, mode 2.1
- `tar_experiment_library.py` - experiment catalog / specification
  source, not autonomous executor
- `tar_frontier.py` - frontier hypothesis generation, mode 2.4
- `tar_suite_logging.py` - observability and bounded runtime tracing
- `tar_suite_checkpoint.py` - checkpoint / resume support for bounded
  runs
- `tar_hpc_validation.py` - controlled validation runner, but only
  under explicit manifest authority
- `tar_validation_mode.py` - validation policy plumbing, if it is
  reduced to manifest / approval support
- `tar_author.py` - paper drafting / reporting subsystem; keep, but
  split evidence/statistics from narrative/advisory output
- `tar_api.py` - only if API routes cannot trigger execution without
  manifest approval
- `tar_post_queue_eval.py` - keep only in planning / evaluation mode

**Rewrite - current execution authority is unsafe**
- `tar_living_research.py` - currently has execution rights; must be
  reduced to planning only
- `tar_watchdog.py` - currently can restart execution paths; must
  become monitoring-only unless a manifest explicitly authorises an
  execution session
- `tar_stabilise_validation.py` - has execution-adjacent authority
  and must be manifest-gated before restart
- `tar_experiment_orchestrator.py` - orchestration stays, but
  execution authority must come from manifests, not from internal
  decisions

**Retire or quarantine**
- `tar_autonomous_research.py` - the name itself describes the old
  unsafe mode. Parts that survive audit can be moved into safer
  planning modules under new names.
- Scheduled-task and Startup persistence - already disabled; stays
  disabled unless separately justified in a future persistence audit

---

## 7. Success criteria for restart

TAR is "back online correctly" when all of the following are true:

1. No code path in TAR can write to a canonical result file that
   already exists. Verified by test.
2. No daemon can start a training run without referencing a
   user-authored manifest. Verified by test.
3. Every result file in `tar_state/comparisons/` written after
   restart has a sibling `_env.json` with full provenance. Verified
   by inspection.
4. The dashboard, planner, and tracker are running and visibly
   producing useful output, without any execution events not
   traceable to user authorisation. Verified by observation.
5. The user can answer "what is TAR currently doing" in under a
   minute by reading TAR's own state files, without ambiguity or
   contradiction between sources.
6. No scheduled task, Startup entry, or equivalent persistence
   mechanism is active unless explicitly justified and documented.
7. Every execution event records a manifest path and manifest hash in
   its env snapshot.

When these criteria hold, Phase 6 (restart) is complete and TAR is
operational under the new design.

---

## 8. What this design gives up

Honest accounting of what the new TAR cannot do that the old one
could:

- **No more open-ended overnight campaigns.** TAR cannot run "for 48
  hours doing whatever it decides." Every overnight run is a
  pre-defined manifest with a finite, knowable set of experiments.
  This is slower in throughput than the old design but eliminates the
  failure mode that produced May 2.

- **No more reactive research.** The old TAR could observe a result
  and immediately queue a follow-up. The new TAR proposes the
  follow-up; the user authorises it. There is friction between
  observation and next experiment. That friction is the safety.

- **No more silent state correction.** If TAR notices its tracked
  state has drifted from the canonical files, it cannot just fix it.
  It surfaces the contradiction for review. Same friction principle.

- **More user time per week on TAR-related decisions.** The old
  design minimised user involvement; the new design requires the
  user as the execution gate. Honest cost: one to two hours per week
  of reviewing manifests, queues, and proposals.

These costs are accepted because the alternative - open autonomy
with rails that may or may not hold - has already been tried and
produced silent corruption of the project's headline result.

---

## End of Phase 1 design note

Phase 2 (append-only result writes) cannot begin until the user has
read this document, edited where it does not match intent, and
explicitly approved it as the gate. Approval should be a commit to
git, not a verbal or chat-based confirmation.
