# WS36 Closeout

`WS36` is complete. Phase 5 now has a working frontier-gap scanner with stable
state across repeated scans.

## What Was Built

Three slices were completed:

1. Core scanner
- literature-driven gap extraction from persisted `ResearchDocument.problem_statements`
- novelty scoring against existing projects
- domain-alignment gate before proposal
- proposal / promotion / rejection flow for frontier-derived projects

2. Operator surfaces
- control commands for frontier scan, listing, proposal, promotion, rejection,
  and scan-history inspection
- CLI renderers for frontier-gap status, scan results, and scan history
- dashboard frontier-gap tab with filtered review, selected-gap inspection,
  scan-history inspection, and direct actions
- scan provenance and operator review metadata on promote / reject actions

3. Cross-scan dedup
- stable `content_hash` on `FrontierGapRecord`
- cross-scan skip logic during `scan_frontier_gaps()`
- scan reporting via `gaps_skipped_cross_scan`
- stable hash lookups on `TARStateStore`

## Final Validation

- `python -m pytest tests/test_frontier_gap_scanner.py -q` -> `9 passed`
- `python -m pytest tests/test_tar_foundation.py -k "frontier" -q` -> `3 passed`

## Schema Additions

- `FrontierGapRecord`
- `FrontierGapScanReport`
- `FrontierGapRecord.content_hash`
- `FrontierGapScanReport.gaps_skipped_cross_scan`

## State Additions

Added on `TARStateStore`:

- `has_content_hash()`
- `find_by_content_hash()`

## Control Surface

New frontier commands:

- `scan_frontier_gaps`
- `list_frontier_gaps`
- `propose_projects_from_gaps`
- `promote_gap_project`
- `reject_gap_project`
- `list_frontier_gap_scans`

## Unlocked

`WS37` may begin.
