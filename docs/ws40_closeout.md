# WS40 Closeout

## Title
WS40: Frontier Operator Scale

## Outcome
WS40 is complete. TAR now routes operator calls through a cost-aware tier system
rather than a single LocalLLMConfig. Every call is logged with computed token cost,
the routing record is persisted to tar_state/routing/, and a RoutingSummary aggregates
costs and call counts by tier for operator inspection.

## Delivered

### WS40-A: Schema extensions
- LocalLLMConfig extended: model_tier (Literal["efficient","frontier"]), cost_per_token_input,
  cost_per_token_output, context_window, supports_tool_use
- FrontierModelConfig: provider, model_id, api_key_env, cost_per_token_input, cost_per_token_output,
  context_window, supports_tool_use
- ModelRoutingRecord: record_id, timestamp, role, stakes, resolved_tier, model_id,
  token_count_input, token_count_output, cost_usd, policy, budget_cap_triggered
- RoutingSummary: total_calls, total_cost_usd, calls_by_tier, cost_by_tier,
  budget_cap_triggers, window_start, window_end

### WS40-B: ModelRouter
- File: tar_lab/model_router.py
- select_config(): stakes_aware / always_frontier / always_efficient policies;
  budget cap fallback from frontier to efficient when accumulated cost exceeds cap
- log_call(): computes cost from token counts, persists ModelRoutingRecord to
  tar_state/routing/<record_id>.json
- get_summary(): aggregates routing log over a time window
- load_log(): loads all persisted routing records

### WS40-C: TriModelHierarchy integration
- File: tar_lab/hierarchy.py
- Routing hooks at Director and Scout role resolution points
- get_frontier_config_from_workspace(): loads FrontierModelConfig from
  tar_state/frontier_model_config.json when present

### WS40-D: Orchestrator surfaces
- load_frontier_config(), save_frontier_config(), get_routing_summary(), get_routing_log()
  wired through orchestrator.py

### WS40-E: Operator surfaces
- control.py: routing_summary, routing_log, load_frontier_config commands
- tar_cli.py: `tar routing summary`, `tar routing log`, `tar frontier-config show`

## Validation

python -m pytest tests/test_frontier_operator_scale.py -q
-> 5 passed

python -m pytest tests -q --tb=short
-> 358 passed, 1 deselected

py_compile clean on all modified files.

## Pod Posture
No pod required. All routing logic is local. Pod is required only when a real
FrontierModelConfig points to a live API endpoint and run costs are being validated
against real inference budgets.

## What Is Next
WS41: Cross-Domain Literature Corpus. Extend ResearchIngestor._default_sources()
to seven arXiv domains, add CrossDomainBridgeRecord schema, index cross-domain
bridge records in the vault.
