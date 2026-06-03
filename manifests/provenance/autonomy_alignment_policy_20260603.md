# Autonomy Alignment Policy — armed 2026-06-03

Written as part of Phase 0 (Truth & Safety Foundation) of the
`TAR_Phase0_and_StackBridge_Implementation_Plan.md`. Records the deliberate
human-in-the-loop decisions so the rails are *armed*, not merely present.

## Kill switches (filesystem flags on the state tree — `tar_state/`)
- `daemon_paused.flag` — **always wins**: forces the living-research daemon into
  planning-only mode and releases the GPU. (`tar_living_research.py:62-72`.)
- `execution_enabled.flag` — must be **present** for the daemon to execute anything.
  Delete it to halt all autonomous submission.

These remain the primary, reliable controls. The live HPC replication is currently
protected by the daemon being down.

## Autonomy ramp — ARMED 2026-06-03 (was a 0-byte no-op)
`autonomy_ramp.json` initialised via `tar_autonomy_ramp.py init` →
`enabled=true, stage=confirmatory`. Effect:
- Director-**generated** experiments are gated (`is_full_autonomy()` returns False)
  until **all** Phase 2/3 confirmatory runs are terminal, the health + evidence
  safety gates pass, **and a human gives explicit final confirmation**
  (`tar_autonomy_ramp.py confirm` or `tar_state/autonomy_ramp_confirm.flag`).
- The controller **never** self-promotes past `awaiting_confirm`.
- Manual Phase 2/3 runs (identified by runner_key) are unaffected and proceed.

**Policy:** full autonomy is enabled only by an explicit human `confirm` after the
confirmatory science is complete and reviewed. Do not create the confirm flag while
the HPC replication or any Phase 2/3 run is still in flight.

## Human-review veto window (opt-out, 24 h)
`VetoWindowApproval` auto-approves a director proposal if a human does not veto it
within 24 h. This is the documented fallback, **not** a substitute for review.
**Policy:** a human reviews the human-review queue on the dashboard at least daily
while the daemon is active; silence is treated as the considered fallback, not as
absence of oversight. RAIL-4 (the execute-boundary veto re-check) currently fails
OPEN on exception — treat an errored approval lookup as needing manual review.

## Per-domain frontier autonomy (opt-in, default OFF)
`tar_research_director._FRONTIER_AUTONOMY_DOMAINS` is **empty by default**, so the
director cannot autonomously mint new (literature-gap-derived) frontier problems for
any domain. A domain becomes eligible only when a human adds it to that set — the
human-GATED last step of the stack-bridge rollout. The
`FrontierRegistry.register` real-world guard (`well_known_problem` + named external
baselines/datasets/backbones) is **not** relaxed and applies on top.

## Multi-domain intake
`science_profiles._ACCEPTED_DOMAINS` now includes the Stack-B-executable domains
(deep_learning, computer_vision, NLP, RL, graph_ml, quantum_ml, generic_ml) so the
science_exec bridge can run them. **Finance remains hard-blocked** pending a
dedicated FI-2010 adapter and a leakage-controlled (walk-forward / purged-split)
protocol.
