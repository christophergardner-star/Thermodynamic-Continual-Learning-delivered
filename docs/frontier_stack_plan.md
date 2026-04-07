# Frontier Stack Plan

This document maps the current TAR/ASC stack to the missing frontier-grade capabilities identified in the audit. It is deliberately written as an execution contract rather than aspirational prose.

## 1. Real Experiment Backend

Question:
Can TAR launch actual research workloads instead of only a control-path payload?

Academic progression:
- Minimum acceptable: a typed backend registry that distinguishes toy control payloads from real ASC training entrypoints.
- MSc-grade: reproducible launch plans for language-model, coding-model, and domain-study runs.
- PhD-grade: checkpoint resume, distributed launch, and standardized artifacts across backends.
- Frontier-grade: heterogeneous multi-node execution with resource-aware placement and auto-resume after interruption.

Implemented now:
- `tar_lab/experiment_backends.py`
- Backend registry for `toy_anchor`, `asc_full`, and `coding_asc`
- Typed launch plans and manifests

Still missing:
- Actual distributed runtime integration for `asc_full`
- Resume semantics for the ASC and coding trainers
- Artifact lineage across pods/nodes

## 2. Paper-Grade Literature Engine

Question:
Can TAR ground its reasoning in parsed papers rather than only abstracts and snippets?

Academic progression:
- Minimum acceptable: local PDF/text ingestion, section extraction, and claim extraction.
- MSc-grade: citation-edge extraction and claim typing (`fact`, `measured_result`, `inference`, `hypothesis`).
- PhD-grade: provenance-preserving section and table parsing with local claim graphs.
- Frontier-grade: cross-paper citation graph, contradiction graph, and claim-to-benchmark lineage.

Implemented now:
- `tar_lab/literature_engine.py`
- Local text/PDF ingestion
- Section extraction
- Claim extraction
- Citation-edge extraction
- Basic contradiction detection

Still missing:
- Table parsing
- Figure extraction
- OCR fallback
- External citation graph integration

## 3. Scientific Retrieval

Question:
Can TAR retrieve evidence semantically and reason over contradictory findings?

Academic progression:
- Minimum acceptable: vector retrieval over paper claims and experiment reports.
- MSc-grade: lexical reranking and typed retrieval over claims versus experiments.
- PhD-grade: semantic embeddings with local reranking and contradiction surfacing.
- Frontier-grade: structured claim memory with source confidence, temporal decay, and uncertainty-aware retrieval.

Implemented now:
- `tar_lab/memory/vault.py`
- Optional semantic embedder hook via `TAR_EMBEDDER_MODEL`
- Lexical reranking over retrieved hits
- Paper-claim indexing

Still missing:
- Strong default local embedding model
- Cross-encoder reranker
- Confidence-aware claim graph retrieval

## 4. Canonical Benchmark Harnesses

Question:
Can TAR validate claims on accepted external suites rather than only internal probes?

Academic progression:
- Minimum acceptable: domain benchmark adapters with explicit success criteria.
- MSc-grade: at least one accepted benchmark family per domain.
- PhD-grade: full benchmark packs with statistical reporting and multi-seed confidence intervals.
- Frontier-grade: benchmark federation across clusters and artifact stores.

Implemented now:
- `tar_lab/science_exec.py`
- Domain adapters for deep learning, NLP, RL, CV, graph ML, quantum ML, and generic ML

Still missing:
- Canonical external datasets wired end to end
- Benchmark-specific artifact schemas
- Statistical testing and multiple-comparison controls

## 5. Reproducible Payload Environment

Question:
Can the main TAR payload run from a locked image rather than mutating the container at launch?

Academic progression:
- Minimum acceptable: a versioned Dockerfile and locked requirements file for the payload.
- MSc-grade: deterministic build manifest stored with every run.
- PhD-grade: artifact registry, image digests, and environment provenance attached to each trial.
- Frontier-grade: full build attestation and reproducibility audits.

Implemented now:
- `tar_lab/reproducibility.py`
- Locked payload package capture
- Dockerfile generation for the main payload path

Still missing:
- Automatic use of the locked payload image in live TAR runs
- Image digest capture and registry publication

## 6. Durable Lab Runtime

Question:
Can TAR behave like a resilient service instead of a one-shot scheduler?

Academic progression:
- Minimum acceptable: heartbeat, stale-run cleanup, and recurring scheduler cycles.
- MSc-grade: retry/backoff and structured daemon status.
- PhD-grade: lease-based workers and orphan-run recovery.
- Frontier-grade: distributed work queue with observability, alerting, and SLA-style health management.

Implemented now:
- `tar_lab/runtime_daemon.py`
- Heartbeat file
- Stale schedule cleanup
- Repeatable runtime cycles

Still missing:
- Worker leases
- Retry policies
- Alerting
- Multi-process or multi-node coordination

## 7. Safe Autonomous Execution

Question:
Can TAR execute generated code without treating the host OS as disposable?

Academic progression:
- Minimum acceptable: no raw host execution by default.
- MSc-grade: container-only execution with no network and capped resources.
- PhD-grade: read-only mounts, filesystem policies, and explicit capability drops.
- Frontier-grade: hardened sandboxing with attested images and audited syscall surfaces.

Implemented now:
- `tar_lab/safe_exec.py`
- Docker-first code execution
- No-network sandbox default
- Host fallback only by explicit opt-in (`CRUXY_ALLOW_HOST_EXEC`)
- `researcher_agent.py` now routes execution through the sandbox layer

Still missing:
- Capability drops
- Read-only workspace policies
- Per-task seccomp/apparmor profiles

## 8. Direct Inference Bridge

Question:
Can a trained ASC checkpoint become a first-class local model for TAR?

Academic progression:
- Minimum acceptable: checkpoint registry and launch spec generation.
- MSc-grade: one local serving backend that exposes an OpenAI-compatible API.
- PhD-grade: direct role mapping for Director/Strategist/Scout and checkpoint-aware routing.
- Frontier-grade: multi-model serving, load balancing, and on-device role specialization.

Implemented now:
- `tar_lab/inference_bridge.py`
- Checkpoint registry
- OpenAI-style endpoint plans
- `serve_local.py` now has a functional `transformers` HTTP server fallback in addition to the `vllm` path

Still missing:
- Role-specialized auto-wiring for Director/Strategist/Scout
- Better batching and throughput in the transformers fallback
- Production auth and rate limiting

## Operating Principle

The stack should move in this order:

1. Ingest papers with provenance.
2. Index claims and experiments into retrievable memory.
3. Route a problem to a domain and benchmark plan.
4. Launch a real backend, not a toy payload, when the question requires training.
5. Run the study under a locked environment.
6. Verify the result with canonical benchmarks and statistical checks.
7. Feed the result back into the claim memory and contradiction graph.

The repo is now closer to that loop, but it is not yet the final frontier system. The purpose of the new modules is to make the next steps architectural extensions rather than another rewrite.
