# Frontier Implementation Roadmap

This roadmap converts the current TAR/ASC stack into a scientifically honest,
reproducible, frontier-grade research system. The milestone order is strict:
later milestones assume the acceptance criteria of earlier milestones have been
met.

## Milestone Order

1. Workstream 1: Scientific Contract Cleanup
2. Workstream 2: Statistical Payload Validity
3. Workstream 3: Real Data and Backend Provenance
4. Workstream 4: Literature and Retrieval Stack
5. Workstream 5: Canonical Benchmark Harnesses
6. Workstream 6: Runtime, Reproducibility, and Safety
7. Workstream 7: Inference Bridge and Research-Agent Discipline

## Workstream 1: Scientific Contract Cleanup

### Goal

Stop the repository from presenting scientifically invalid paths as canonical.

### File-by-file targets

- `asc_train.py`
  Quarantine legacy script and fail fast with a scientific-validity error.
- `asc_train_cpu.py`
  Quarantine legacy script and fail fast with a scientific-validity error.
- `asc_train_full.py`
  Declare as the canonical ASC training path in docs and CLI references.
- `deepseek_asc_finetune.py`
  Mark as experimental and print runtime warnings about masking, device
  placement, and scaling limits.
- `README.md`
  Point only to scientifically valid public entrypoints.
- `docs/implementation_roadmap.md`
  Serve as the authoritative execution roadmap for the remaining workstreams.

### Acceptance criteria

- Invalid ASC entrypoints cannot be run without an explicit scientific-validity
  failure.
- Public documentation names one canonical ASC trainer.
- Experimental paths are explicitly labeled as such in both code and docs.

### Tests

- Subprocess test that `asc_train.py` fails with a scientific-validity message.
- Subprocess test that `asc_train_cpu.py` fails with a scientific-validity
  message.
- Subprocess test that `deepseek_asc_finetune.py --dry_run` prints an
  experimental warning.

## Workstream 2: Statistical Payload Validity

### Goal

Make TAR's default payload scientifically defensible instead of toy-valid.

### File-by-file targets

- `tar_lab/train_template.py`
  Replace one-batch anchor and calibration estimates with multi-batch rolling
  estimators and confidence intervals. Add checkpoint resume and explicit split
  handling.
- `tar_lab/governor.py`
  Consume rolling windows instead of single-step snapshots.
- `tar_lab/verification.py`
  Base verification on sample windows and confidence bounds, not one minibatch.
- `tar_lab/schemas.py`
  Add statistical summary fields for means, variances, windows, and confidence
  intervals.
- `tar_lab/orchestrator.py`
  Promote the default payload from `__tiny_gpt2__` to a small but real research
  baseline and record statistical provenance in reports.
- `tests/test_tar_foundation.py`
  Add tests for rolling metrics, resume behavior, and split-aware payload
  reporting.

### Acceptance criteria

- Default TAR runs report confidence intervals for calibration and `D_PR`.
- The governor makes decisions on rolling windows.
- Payload resumes correctly from checkpoints.
- Reports explicitly state train/val/test split usage.

### Tests

- Unit tests for rolling `D_PR` and calibration estimators.
- Resume test that continues from a saved payload checkpoint.
- Integration test confirming governor action changes only after windowed
  threshold breach.

## Workstream 3: Real Data and Backend Provenance

### Goal

Replace silent fallback science with explicit data regimes and real backend
   routing.

### File-by-file targets

- `tar_lab/data_manager.py`
  Introduce explicit modes: `offline_fallback`, `cached_real`,
  `download_real`. Remove silent tokenizer fallback from research-facing runs.
- `tar_lab/experiment_backends.py`
  Expand real backend adapters for ASC text, coding ASC, CV, RL, graph ML, QML,
  and generic supervised training.
- `tar_lab/orchestrator.py`
  Record backend, corpus, tokenizer, and data mode in every trial report.
- `tar_lab/schemas.py`
  Add provenance fields for dataset, tokenizer, cache mode, and backend.
- `tar_lab/train_template.py`
  Refuse ambiguous fallback mode unless a run is explicitly tagged as plumbing.
- `README.md`
  Document the data-mode contract.

### Acceptance criteria

- Every run report names the exact corpus, tokenizer, backend, and data mode.
- Silent fallback is gone from research-facing runs.
- All named backends have explicit registry entries.

### Tests

- Data-mode tests for each regime.
- Provenance serialization tests.
- Payload failure test when a research run requests silent fallback behavior.

## Workstream 4: Literature and Retrieval Stack

### Goal

Build a paper-grade evidence layer that can support literature-grounded planning.

### File-by-file targets

- `tar_lab/literature_engine.py`
  Add page rendering fallback, OCR over rendered pages, claim spans with page
  references, figure-caption linking, and citation graph extraction.
- `tar_lab/memory/vault.py`
  Make semantic embeddings the default, add hybrid dense/lexical retrieval,
  reranking, structured claim indexing, contradiction tracking, and evidence
  clustering.
- `tar_lab/research_ingest.py`
  Preserve source provenance and ingest richer paper metadata.
- `tar_lab/state.py`
  Persist citation graphs, claim spans, and contradiction clusters.
- `tar_lab/hierarchy.py`
  Force recommendations to cite underlying source spans.
- `tests/test_tar_foundation.py`
  Add retrieval-grounding tests and citation-span persistence tests.

### Acceptance criteria

- Every recommendation can cite source passages or claim spans.
- Scanned PDFs can be ingested through page rendering + OCR.
- Retrieval quality is no longer dependent on hash embeddings by default.

### Tests

- PDF ingestion tests with rendered-page fallback.
- Claim/citation graph roundtrip tests.
- Retrieval ranking tests on known scientific query sets.

## Workstream 5: Canonical Benchmark Harnesses

### Goal

Replace proxies with named, comparable benchmark suites.

### File-by-file targets

- `tar_lab/science_exec.py`
  Wire canonical suites for NLP, CV, RL, graph ML, QML, and generic ML.
- `science_profiles/*.json`
  Map each profile to named benchmark suites and dataset contracts.
- `tar_lab/problem_runner.py`
  Enforce benchmark-specific execution contracts and artifact capture.
- `tar_lab/schemas.py`
  Add benchmark provenance and version fields.
- `README.md`
  Document named benchmark coverage and what is still out of scope.

### Acceptance criteria

- Every domain profile maps to named benchmark suites.
- Results are comparable to published literature, not just internal proxies.
- Reports record benchmark versions and dataset provenance.

### Tests

- Benchmark smoke tests for each domain adapter.
- Artifact-format tests for saved benchmark outputs.
- Provenance tests for suite name, dataset version, and split coverage.

## Workstream 6: Runtime, Reproducibility, and Safety

### Goal

Make the lab durable, reproducible, and bounded under failure.

### File-by-file targets

- `tar_lab/docker_runner.py`
  Remove runtime package mutation and use locked images only.
- `tar_lab/reproducibility.py`
  Produce image manifests, dependency lockfiles, and environment hashes.
- `tar_lab/scheduler.py`
  Add retries, backoff, and lease-aware job execution.
- `tar_lab/runtime_daemon.py`
  Add heartbeat ownership, orphan recovery, and alert hooks.
- `tar_lab/safe_exec.py`
  Restrict generated-code execution to Docker-only sandboxed runs with explicit
  mount and network policy.
- `researcher_agent.py`
  Remove any host-side raw Python execution path.

### Acceptance criteria

- Any run is reproducible from a manifest and locked environment.
- Runtime failures are bounded, logged, and recoverable.
- No autonomous code executes unsandboxed on the host.

### Tests

- Manifest/hash reproducibility tests.
- Scheduler retry/backoff tests.
- Stale-job recovery tests.
- Sandbox policy tests for allowed mounts and blocked host execution.

## Workstream 7: Inference Bridge and Research-Agent Discipline

### Goal

Turn TAR from a smart shell into a disciplined evidence-driven research
operator.

### File-by-file targets

- `tar_lab/inference_bridge.py`
  Add endpoint lifecycle, health checks, and role-aware model assignment.
- `serve_local.py`
  Support managed local serving contracts and health endpoints.
- `tar_lab/hierarchy.py`
  Replace heuristic planning with evidence-grounded hypothesis generation and
  contradiction-aware updates.
- `tar_lab/orchestrator.py`
  Enforce research claim acceptance rules before promotion.
- `tar_lab/schemas.py`
  Add claim-acceptance thresholds and evidence gating structures.
- `README.md`
  Document the research claim contract.

### Acceptance criteria

- Role endpoints are discoverable, health-checked, and lifecycle-managed.
- Director/Strategist/Scout planning is grounded in evidence and contradiction
  checks.
- Breakthrough claims require minimum seed count, ablation gap, calibration
  threshold, and literature support.

### Tests

- Endpoint registry and health-check tests.
- Planning tests that fail when evidence is missing.
- Claim-acceptance tests for seeds, ablations, calibration, and literature
  support.

## Unified Test Plan

### Scientific-math tests

- ASC min-max correctness
- rolling `D_PR`
- rolling calibration / ECE
- equilibrium-gate activation
- quenching detection
- seed variance and ablation thresholds

### Data and provenance tests

- dataset mode selection
- tokenizer provenance capture
- benchmark provenance capture
- report serialization and persistence

### Literature and retrieval tests

- PDF ingestion
- OCR fallback
- claim span extraction
- citation graph extraction
- contradiction clustering
- dense + lexical retrieval ranking

### Runtime and safety tests

- locked-image execution
- scheduler retries/backoff
- stale-job cleanup
- heartbeat ownership
- sandbox policy enforcement

### End-to-end system tests

- ASC canonical run
- TAR payload run
- problem-study run
- benchmark execution run
- literature-grounded recommendation run

## What We Build First This Week

1. Workstream 1 in full
2. Workstream 2 statistical payload validity
3. Workstream 3 explicit data modes and provenance
4. The first half of Workstream 4:
   semantic retrieval defaults
   citation-span persistence
   rendered-page OCR fallback

## What Comes Later

- Full canonical benchmark expansion across all domains
- lease-aware long-run runtime service and alerting
- complete locked-image pipeline for all backend families
- first-class inference lifecycle and claim-acceptance gates
- deeper literature contradiction reasoning and evidence clustering

## Exit Standard

The stack is considered fixed only when:

- default runs are scientifically honest
- outputs are benchmark-comparable
- literature grounding is traceable
- retrieval is semantically strong
- runtime is reproducible
- autonomy is evidence-constrained
