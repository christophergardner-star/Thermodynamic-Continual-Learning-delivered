# Post-WS16 Strategy Blueprint

This document locks in the strategic direction for TAR after `WS16`
completion. It is intentionally not part of the remediation critical path.
It defines the next-phase identity, moat, priorities, and non-goals so future
work does not drift into generic agent features or benchmark theater.

## Core Identity

TAR should not position itself as a generic AI researcher, chat assistant, or
paper generator. Its core identity should be:

- a research operating system for disciplined inquiry
- a truthful autonomy system that measures, verifies, refuses, and explains
- a memory-bearing lab substrate for long-horizon research programs
- a compute-to-knowledge efficiency engine, not a scale-for-scale's-sake system

The defining claim should be:

> TAR turns limited compute, structured evidence, and explicit policy into
> credible research progress.

The system should be optimized for:

- credible findings over impressive-looking outputs
- explicit uncertainty over false confidence
- reproducible execution over convenience shortcuts
- project continuity over prompt-by-prompt cleverness
- falsification pressure over confirmation bias

### Output Identity

Every major TAR output should aim to answer:

1. What was asked?
2. What was actually tested?
3. What happened?
4. How strong is the evidence?
5. What contradicts the current interpretation?
6. What would falsify the result next?
7. What should be done next under the available budget?

These are not cosmetic reporting goals. They should become part of TAR's
system contract.

## Strategic Moat

TAR should not try to compete on raw scale with frontier labs. The moat should
come from disciplined system behavior that large compute alone does not buy.

### Primary moat components

- truthful autonomy
- evidence-bound claim formation
- falsification-first research planning
- benchmark honesty
- long-horizon research memory
- compute-efficient experiment prioritization
- structured pause / resume / pivot control

### Why this is defensible

Many systems optimize for looking capable. Very few optimize for being:

- auditable
- reproducible
- self-limiting
- interruption-tolerant
- explicit about what they do not know

If TAR becomes unusually strong at preserving scientific discipline under
autonomy, that is a real differentiator.

### Institutional memory as a moat

TAR should accumulate reusable research knowledge over time:

- prior trials
- failed directions
- unresolved contradictions
- useful ablation templates
- benchmark lessons
- domain-specific playbooks
- evidence chains that changed belief state

That makes the system more valuable with use, not just with bigger models.

## Product Direction

TAR's product direction should be internal-research-first rather than
consumer-facing.

### Phase 1 product

The first serious product is a research control surface for one or a few
operators. It should support:

- defining or ingesting research problems
- seeing what TAR believes and why
- inspecting contradictions and evidence gaps
- approving or rejecting experiments and budgets
- reviewing claim strength and provenance
- resuming interrupted work cleanly

### Phase 2 product

The second product layer is a research portfolio manager:

- multiple concurrent research threads
- priorities and deadlines
- evidence budgets
- stop / continue / pivot rules
- problem aging and stale-thread recovery
- portfolio-level scheduling and triage

### Phase 3 product

The third layer is an evidence publishing and handoff system:

- internal reports
- benchmark truth ledgers
- claim history
- experiment lineage
- contradiction maps
- publication handoff packages

The writing layer is downstream. TAR should not lead with academic drafting
until the evidence package is already strong.

## Research Direction

The research agenda should be about autonomous scientific process, not only
model capability.

### Core research questions

1. How should autonomous systems act under uncertainty without overstating
   authority?
2. How should an autonomous lab decide what is worth testing next?
3. How should claims be governed so they stay bound to exact evidence?
4. How should a research system remember unresolved questions across long time
   horizons?
5. How should a system actively try to break its own beliefs before promoting
   them?

### Priority research themes

#### Autonomy under constraint

TAR should encode explicit policy, provenance, refusal, and escalation rules
into the research loop.

#### Compute-efficient discovery

TAR should learn to spend compute where uncertainty reduction is highest.

#### Claim governance

TAR should distinguish:

- hypothesis
- provisional signal
- replicated result
- benchmark-comparable result
- accepted claim
- rejected claim
- unresolved claim

#### Long-horizon memory

TAR should retain:

- what was tried
- what failed
- what remains unresolved
- why a thread was paused
- why confidence changed
- what should be resumed next

#### Falsification-first automation

TAR should routinely generate:

- ablations
- contradiction checks
- adversarial tests
- replication passes
- minimum-cost disproof attempts

## What To Build Next

After `WS16`, the next roadmap phase should focus on continuity, budgeting,
prioritization, falsification, and output discipline.

### `WS17`: Research Continuity And Budgeting

Purpose:

- introduce project-level state beyond run-level state
- let TAR remember where a hanging project left off
- make pause / resume an explicit system behavior

Capabilities:

- project objects
- hypothesis threads
- unresolved-question queues
- next-action records
- stop reasons
- resume reasons
- wall-clock and compute budgets

Closure standard:

- TAR can say what it is working on, why it paused, and what should happen next

### `WS18`: Evidence-Budgeted Prioritization

Purpose:

- decide which action is worth doing next
- stop wasting compute on low-information work

Capabilities:

- expected information gain scoring
- cost-aware experiment ranking
- strategic importance weighting
- uncertainty-reduction scoring
- replication value scoring
- benchmark value scoring

Closure standard:

- TAR can justify why a given experiment was chosen over alternatives

### `WS19`: Meta-Tests And Falsification Planning

Purpose:

- make truth-proving and truth-breaking first-class system behaviors

Capabilities:

- targeted ablation generation
- contradiction-driven follow-up tests
- adversarial benchmark probes
- minimal falsification suites
- calibration and replication triggers

Closure standard:

- strong claims always face explicit falsification pressure before promotion

### `WS20`: Research Portfolio Management

Purpose:

- move from isolated studies to a portfolio of active research programs

Capabilities:

- active thread queue
- priorities and deadlines
- evidence debt tracking
- stale project recovery
- stop / continue / pivot transitions
- escalation to human review

Closure standard:

- TAR can manage multiple open problems without losing rigor or continuity

### `WS21`: Operator Interface Upgrade

Purpose:

- expose the structure of inquiry clearly once the control logic is mature

Capabilities:

- project timeline views
- evidence maps
- contradiction maps
- budget burn views
- claim lineage panels
- resume-point dashboards

Closure standard:

- the interface makes project state, evidence state, and next actions legible

### `WS22`: Publication Handoff Layer

Purpose:

- hand structured, evidence-backed findings to a writing subsystem

Inputs should include:

- accepted claims
- rejected alternatives
- experiment lineage
- benchmark truth status
- limitations
- contradictions
- open questions

Closure standard:

- the writer receives a structured evidence package, not a vague narrative

## What To Ignore

These are deliberate non-goals unless they clearly strengthen TAR's central
identity.

### Ignore generic agent branding

Do not optimize for sounding like a universal super-agent.

### Ignore flashy interface work too early

Do not spend core effort on polish before the research-control logic is real.

### Ignore paper-writing-first development

Writing is downstream of credibility, not a substitute for it.

### Ignore benchmark theater

Do not allow proxy experiments to masquerade as canonical comparability.

### Ignore feature sprawl

If a feature does not strengthen truthfulness, continuity, prioritization,
falsification, or reproducibility, it should wait.

### Ignore performative autonomy

Do not optimize for the appearance of independence if it weakens discipline.

### Ignore raw scale as the strategy

TAR should not try to outspend big labs on compute. It should out-discipline
them on how evidence is produced and managed.

## Output Standards

These standards should govern future TAR outputs and operator expectations.

### Credible findings

TAR must distinguish between:

- a hypothesis
- a promising signal
- a replicated effect
- a benchmark-comparable result
- an accepted claim

### Explanation and verification

TAR must explain:

- why a finding is being claimed
- what evidence supports it
- what checks were run
- what replication or ablations were performed
- what remains uncertain

### Honesty under incomplete capability

TAR must say when:

- the benchmark is not canonical-ready
- the environment is incomplete
- the model/backend cannot support a stronger claim
- the evidence base is too thin
- the result is only validation-grade

## Phase Success Condition

The next phase succeeds only if TAR becomes better at converting bounded
compute into reliable knowledge, not just better at appearing autonomous.

The strategic test should be:

- does TAR help a serious operator reach stronger conclusions with less wasted
  effort?
- does TAR remember where work left off and resume it coherently?
- does TAR test its own claims rather than merely defend them?
- does TAR remain honest when it lacks the basis for stronger conclusions?

If those answers become yes, TAR will have a real identity that is difficult to
replace with a generic agent stack.
