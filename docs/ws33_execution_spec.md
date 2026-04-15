# WS33 Execution Spec

## Title

`WS33: Scientific Retrieval And Claim Graph`

## Purpose

Deepen TAR retrieval from document search into contradiction-aware scientific
evidence retrieval on top of the now-closed `WS32` and `WS32.5` loop.

`WS32` gave TAR stronger literature artifacts and cleaner conflict labels.
`WS32.5` ensured literature can influence planning and that evidence debt can
block action. WS33 builds the retrieval layer that turns those stronger
artifacts into first-class evidence objects instead of post-hoc annotations.

## Core Hypothesis

If TAR indexes claim clusters and conflict edges as first-class memory objects,
and if retrieval scoring weights contradiction and source confidence directly,
then literature search will surface real scientific tension instead of only raw
documents with annotations appended after ranking.

## First Local Slice

The first WS33 slice is intentionally bounded.

It must deliver:

- first-class indexing of claim clusters
- first-class indexing of claim conflicts
- source-confidence metadata on literature-memory objects
- contradiction-aware and source-confidence-aware reranking
- regression tests proving claim/conflict objects can be retrieved directly

This slice does **not** yet implement the full evidence-budget feedback loop or
hard refusal of degraded literature studies. Those remain later WS33 slices.

## Scope

### In Scope

- `VectorVault` claim-cluster indexing
- `VectorVault` claim-conflict indexing
- reranker weighting for contradiction-bearing evidence
- reranker weighting for source confidence
- retrieval metadata that exposes contradiction edges directly
- targeted WS33 regression tests

### Out Of Scope

- external retrieval services
- embedding-model changes
- pod-backed large-index experiments
- full claim-graph traversal APIs
- automatic evidence-debt updates from contradiction-heavy retrieval
- verdict lifecycle changes

## Acceptance Criteria

The first WS33 slice is acceptable only if all are true:

- claim clusters are indexed as retrievable memory objects
- claim conflicts are indexed as retrievable memory objects
- reranking uses contradiction/source-confidence signals in scoring
- retrieval can return contradiction-bearing claim-graph evidence directly
- regression tests prove that contradiction-aware retrieval surfaces the
  conflict object instead of only the underlying papers

## Pod Policy

Do not use a pod for this slice.

This is local retrieval-engine work. A pod becomes justified only for later
large-index or embedder-comparison work.

## Execution Order

1. formalize the WS33 spec
2. extend `VectorVault` indexing for claim clusters and conflicts
3. add source-confidence metadata to literature-memory records
4. update `ScientificReranker` scoring
5. add targeted WS33 regression tests
6. validate locally before widening WS33 scope
