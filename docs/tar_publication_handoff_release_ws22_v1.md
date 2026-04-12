# TAR Publication Handoff Release WS22 v1

This release introduces the first structured publication-handoff layer for TAR.

## Scope

The handoff layer exports:

- accepted claim bundles
- provisional claim bundles
- rejected alternative bundles
- experiment lineage
- benchmark-truth attachments
- limitations
- open questions
- evidence gaps
- writer cautions

## Commands

- `python tar_cli.py --direct --publication-handoff --project-id <PROJECT_ID>`
- `python tar_cli.py --direct --publication-log`

## Storage

Generated package artifacts are persisted under:

- `tar_state/publication_handoffs/`
- `tar_state/publication_handoffs.jsonl`

## Boundary

This is a publication handoff package, not an automatic paper writer. The
package is designed to preserve evidence bounds and unresolved uncertainty for a
later writer subsystem or human author.
