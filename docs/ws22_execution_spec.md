# WS22 Execution Spec

`WS22` is `Publication Handoff Layer`.

## Purpose

`WS22` packages TAR's structured evidence into a writer-facing handoff package.
It does not generate papers. It produces the bounded, auditable package that a
later writing subsystem or human author can consume.

## Core Question

Can TAR export publication-facing evidence packages that preserve benchmark
truth, falsification pressure, open questions, and claim limits instead of
collapsing them into vague narrative?

## Deliverables

- first-class publication handoff package schema
- persisted publication handoff artifacts in `tar_state/publication_handoffs`
- command/control and CLI access to generate and inspect packages
- accepted-claim bundles
- rejected-alternative bundles
- experiment lineage exports
- benchmark-truth attachments
- limitations, evidence-gap, and open-question exports

## Design Rules

- publication handoff must sit on top of the real project/evidence state
- no package may silently discard active falsification pressure
- no package may hide benchmark truth or canonical comparability state
- accepted and provisional claims must remain distinct
- rejected or contradicted alternatives must remain visible
- the package is evidence-bound support, not an automatic manuscript

## Implementation Order

1. add publication handoff schemas
2. persist package artifacts and logs
3. derive packages from project, verification, claim, falsification, and
   portfolio state
4. expose `publication_handoff` and `publication_log`
5. add CLI rendering
6. add regression tests

## Acceptance Criteria

`WS22` closes only if all are true:

- a project can be exported as a structured publication handoff package
- the package includes accepted claims, rejected alternatives, lineage,
  benchmark-truth attachments, limitations, and open questions
- package artifacts are persisted and auditable
- CLI/control surfaces expose generation and inspection cleanly
- regression tests prove the package preserves claim status and evidence limits

## Non-Goals

- no automated paper drafting
- no citation-style formatting engine
- no rhetorical polishing layer
- no attempt to hide unresolved uncertainty
