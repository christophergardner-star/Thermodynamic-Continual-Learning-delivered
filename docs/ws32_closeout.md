# WS32 Closeout

## Outcome

`WS32` is complete.

The literature engine is now materially stronger as a provenance-preserving
evidence layer instead of a loose heuristic ingest utility.

## Scope

`WS32` was the literature-fidelity workstream of Phase 4. Its purpose was to
turn existing paper ingestion into a stateful, inspectable, and
regression-covered literature subsystem that later claim review and retrieval
can rely on.

This included:

- stable paper identity by content hash instead of path-only identity
- persisted ingest manifests with repeat-ingest history
- richer structured section, claim, table, and figure metadata
- stronger stored conflict metadata
- retrieval-mode telemetry on literature-grounded studies
- tighter cross-paper conflict gating before `WS33`
- operator-facing inspection surfaces for literature artifacts and conflicts
- a dedicated WS32 literature regression pack

## What WS32 Added

- literature artifacts now persist:
  - content-hash fingerprints
  - canonical source paths
  - ingest-manifest linkage
  - stable stored timestamps
- literature ingest now records first-class manifests under:
  - `tar_state/literature/manifests`
- repeated ingest is now deduplicated truthfully by content hash while
  preserving historical manifest records
- tables now carry:
  - headers
  - row and column counts
  - numeric cell counts
  - metric hints
  - related claim IDs
- figures now carry:
  - figure labels
  - caption hashes
  - OCR-presence flags
  - related claim IDs
- claims and conflicts now expose richer provenance such as:
  - citation counts
  - quality flags
  - conflict kind
  - shared token counts
  - cross-paper conflict identity
- literature-grounded studies now persist:
  - `retrieval_mode = semantic`
  - `retrieval_mode = lexical_fallback`
- cross-paper conflicts now require:
  - at least `3` shared signature tokens
  - opposite polarity
  - a scope / negation compatibility check
- TAR now exposes literature inspection through:
  - orchestrator surfaces
  - control commands
  - CLI renderers
  - dashboard literature tab

## Validation

Dedicated WS32 validation now exists in:

- [test_literature_engine_ws32.py](C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/tests/test_literature_engine_ws32.py)

That regression pack proves:

- content-hash deduplication across duplicate source files
- distinct ingest manifests across repeat ingest
- structured table and figure metadata persistence
- stored conflict persistence and conflict typing
- same-topic but different-scope papers do not get auto-labeled as conflicts
- operator/control/CLI/dashboard literature inspection surfaces

Targeted validation passed:

- `python -m pytest tests/test_literature_engine_ws32.py tests/test_semantic_retrieval.py tests/test_tar_foundation.py -k "literature or frontier_status_reports_new_foundations" -q`
- result: `6 passed, 69 deselected`

Broader regression slice passed:

- `python -m pytest tests/test_literature_engine_ws32.py tests/test_semantic_retrieval.py tests/test_operator_interface.py tests/test_tar_foundation.py -q`
- result: `79 passed`

Compile validation also passed for all touched WS32 files.

## Important Corrections

WS32 surfaced and fixed two real provenance bugs:

- literature manifests originally used raw manifest IDs directly as filenames,
  which broke on Windows because of `:` characters
- repeat-ingest manifests originally relied on incidental state instead of an
  explicit uniqueness input

Those are now corrected in the persisted literature state path.

## Conclusion

`WS32` closes because TAR can now:

- ingest papers with stable content-derived identity
- preserve ingest history in first-class manifests
- expose more structured paper evidence
- record whether literature-grounded studies used semantic retrieval or lexical
  fallback
- hand `WS33` cleaner cross-paper conflict labels instead of permissive
  topic-overlap contradictions
- inspect literature state and conflicts directly through operator surfaces
- validate the literature layer with a dedicated regression pack

No pod was needed for WS32. This was correctly handled as local-first
provenance and evidence engineering.

## What Is Next

The next active workstream is `WS32.5`.

`WS32.5` closes the loop that the Phase 4 audit exposed:

- literature must influence Director policy
- the served operator must route into hierarchy roles without manual env wiring
- evidence debt must hard-block non-remediation scheduling
