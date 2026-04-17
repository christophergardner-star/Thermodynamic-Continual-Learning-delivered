# Phase 6 Roadmap — Frontier Science

This document is the authoritative plan for making TAR genuinely frontier-level
after Phase 5 autonomy is closed.

Phase 5 closes the autonomous loop: TAR can scan gaps, propose projects, run
experiments, curate training signal, retrain itself, and update its own agenda.

That is necessary but not sufficient for frontier science.

The remaining gaps:

1. The operator model cannot reason at the scale where frontier science lives
2. Literature ingest is narrow — only cs.AI and cs.LG
3. Surprising or anomalous results are not elevated — they are processed the
   same as expected results
4. Every accepted claim is accepted in isolation — no competing theory forces
   explicit comparison against rival explanations
5. TAR produces no external-facing contribution positioning — it cannot say how
   its findings compare to the state of the art

Phase 6 closes all five.

## Dependency

Phase 6 must follow the closed Phase 5 stack:

- `WS39` complete (autonomous agenda with gap scanner, generative director,
  self-improvement loop)
- `WS35` complete (safe execution hardening)
- `WS33` complete (contradiction-aware retrieval, claim-graph indexing)

## Active Forward Roadmap

### `WS40` — Frontier Operator Scale

Purpose:

- let the Director operate at the scale where frontier science actually happens:
  70B+ class inference for the highest-stakes decisions, with cost-aware routing
  that keeps Scout/Strategist work efficient

#### Schema changes

File: `tar_lab/schemas.py`

1. Add to `LocalLLMConfig` (currently at line 89):

```python
model_tier: str = "efficient"           # "efficient" | "frontier"
cost_per_token_input: float = 0.0       # USD per input token
cost_per_token_output: float = 0.0      # USD per output token
context_window: int = 8192
supports_tool_use: bool = True
```

2. New schema `FrontierModelConfig`:

```python
class FrontierModelConfig(BaseModel):
    frontier_role_config: LocalLLMConfig     # tier="frontier"
    efficient_role_config: LocalLLMConfig    # tier="efficient"
    routing_policy: str = "stakes_aware"     # "stakes_aware" | "always_frontier" | "always_efficient"
    max_frontier_budget_usd: float = 0.0     # 0.0 = unlimited
    frontier_decisions: list[str] = [        # which decision types get frontier model
        "director_propose",
        "breakthrough_review",
        "falsification_plan",
        "generative_director_proposal",
    ]
```

3. New schema `ModelRoutingRecord`:

```python
class ModelRoutingRecord(BaseModel):
    decision_type: str
    tier_selected: str
    model_id: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    timestamp_utc: datetime
```

#### Hierarchy changes

File: `tar_lab/hierarchy.py`

1. Add `_select_tier_for_decision(decision_type: str, config: FrontierModelConfig) -> LocalLLMConfig`

   - returns `frontier_role_config` when `decision_type in config.frontier_decisions`
   - returns `efficient_role_config` otherwise
   - respects `routing_policy` override

2. Extend `_role_config_from_serving_state()` (added in WS32.5) to read
   `FrontierModelConfig` from workspace state and apply tier routing

3. Add `TriModelHierarchy` wrapper:

   - wraps existing `DirectorRole`, `StrategistRole`, `ScoutRole`
   - injects `FrontierModelConfig` at construction time
   - logs a `ModelRoutingRecord` for every inference call
   - exposes `.routing_log` for audit surfaces

4. `LocalOpenAIRole.call()` must emit `ModelRoutingRecord` into the workspace
   routing log under `tar_state/routing/`

#### Orchestrator changes

File: `tar_lab/orchestrator.py`

1. `plan_trial()` must pass `decision_type="director_propose"` to hierarchy when
   requesting family selection

2. `build_breakthrough_report()` call path must pass
   `decision_type="breakthrough_review"`

3. Add `get_routing_summary()` → aggregates routing records by tier:
   ```
   {"frontier_calls": N, "efficient_calls": N, "total_cost_usd": X}
   ```

4. Expose routing summary in operator view and dashboard

#### Test requirements

File: `tests/test_frontier_operator_scale.py`

- `test_efficient_tier_selected_for_scout_work`: Scout decisions use efficient
  model, not frontier
- `test_frontier_tier_selected_for_director_propose`: Director proposal path
  selects frontier model
- `test_routing_record_logged`: every inference call writes a routing record
- `test_cost_tracking_accurate`: token counts and cost accumulate correctly
- `test_budget_cap_enforced`: if `max_frontier_budget_usd` is reached, routing
  falls back to efficient model

#### Acceptance criteria

- `FrontierModelConfig` and `ModelRoutingRecord` schemas present and validated
- Director high-stakes decisions route to frontier tier when configured
- Scout/Strategist routine decisions route to efficient tier
- `ModelRoutingRecord` is persisted under `tar_state/routing/`
- Routing summary surfaces in operator view and dashboard
- Budget cap enforcement passes test

Pod policy:

- not required for routing logic; required for real frontier inference throughput
  validation

---

### `WS41` — Cross-Domain Literature Corpus

Purpose:

- break out of the cs.AI / cs.LG tunnel
- ingest physics, mathematics, neuroscience, and adjacent fields so TAR can
  surface cross-domain analogy, import known results, and position contributions
  against real scientific context

#### Schema changes

File: `tar_lab/schemas.py`

1. Add to `PaperArtifact` (existing schema):

```python
primary_domain: str = "cs.AI"           # arXiv primary category
secondary_domains: list[str] = []
cross_domain_links: list[str] = []      # paper_ids this paper shares concept overlap with
imported_from_domain: Optional[str] = None  # non-null when ingest path was cross-domain
```

2. New schema `CrossDomainBridgeRecord`:

```python
class CrossDomainBridgeRecord(BaseModel):
    source_paper_id: str
    target_paper_id: str
    source_domain: str
    target_domain: str
    shared_concept_terms: list[str]
    bridge_strength: float              # 0.0–1.0
    created_at: datetime
```

3. New schema `DomainProfile`:

```python
class DomainProfile(BaseModel):
    arxiv_category: str                 # e.g. "cond-mat.stat-mech"
    display_name: str
    ingest_enabled: bool = True
    max_papers_per_ingest: int = 20
    relevance_terms: list[str]          # used for initial relevance filter
```

#### ResearchIngestor changes

File: `tar_lab/research_ingest.py`

1. Replace `_default_sources()` (currently at line 57, returns only cs.AI +
   cs.LG) with:

```python
def _default_sources(self) -> list[DomainProfile]:
    return [
        DomainProfile(arxiv_category="cs.AI", display_name="AI"),
        DomainProfile(arxiv_category="cs.LG", display_name="Machine Learning"),
        DomainProfile(
            arxiv_category="cond-mat.stat-mech",
            display_name="Statistical Mechanics",
            relevance_terms=["energy", "entropy", "thermodynamic", "phase transition",
                             "free energy", "partition function"],
        ),
        DomainProfile(
            arxiv_category="quant-ph",
            display_name="Quantum Physics",
            relevance_terms=["variational", "quantum circuit", "optimization",
                             "expressibility", "barren plateau"],
        ),
        DomainProfile(
            arxiv_category="math.OC",
            display_name="Optimization and Control",
            relevance_terms=["gradient", "convergence", "loss landscape",
                             "saddle point", "Lyapunov"],
        ),
        DomainProfile(
            arxiv_category="math.ST",
            display_name="Statistics Theory",
            relevance_terms=["estimator", "bias", "variance", "generalization",
                             "concentration", "PAC"],
        ),
        DomainProfile(
            arxiv_category="q-bio.NC",
            display_name="Neurons and Cognition",
            relevance_terms=["continual learning", "catastrophic forgetting",
                             "plasticity", "consolidation", "replay"],
        ),
    ]
```

2. Add `_filter_by_relevance(paper: PaperArtifact, profile: DomainProfile) -> bool`

   - returns True if any `relevance_terms` appear in title + abstract
   - returns True unconditionally for cs.AI and cs.LG (all papers)

3. After ingest, call `_build_cross_domain_bridges()`:

   - for every newly ingested paper, scan existing vault papers from different
     domains
   - compute shared concept term overlap
   - if overlap ≥ 3 terms and domains differ: emit `CrossDomainBridgeRecord`
   - persist to `tar_state/literature/bridges/`

#### VectorVault changes

File: `tar_lab/memory/vault.py`

1. Index `CrossDomainBridgeRecord` as kind `cross_domain_bridge`

2. Add `search_cross_domain_bridges(concept_terms: list[str], top_k: int = 5)`

   - returns bridge records where `shared_concept_terms` overlaps with query

3. Expose bridge count in `get_literature_status()`

#### Operator inspection

- CLI: `tar literature bridges` → lists top cross-domain bridges by strength
- Dashboard: cross-domain bridge count and top bridge pairs visible in literature
  panel

#### Test requirements

File: `tests/test_cross_domain_corpus.py`

- `test_default_sources_includes_cross_domain`: stat-mech, quant-ph, math.OC,
  math.ST, q-bio.NC all present in default profile list
- `test_relevance_filter_accepts_matching`: stat-mech paper with "entropy" in
  abstract passes filter
- `test_relevance_filter_rejects_unrelated`: stat-mech paper with no relevance
  terms is rejected
- `test_cross_domain_bridge_created`: two papers from different domains with
  shared terms produce a bridge record
- `test_bridge_searchable_in_vault`: bridge record is retrievable by concept
  term search

#### Acceptance criteria

- `_default_sources()` returns profiles for all 7 domains
- Relevance filtering prevents non-relevant cross-domain papers from cluttering
  the corpus
- `CrossDomainBridgeRecord` is created and persisted when overlap is sufficient
- Bridge records are indexed and searchable in the vault
- Literature status and dashboard expose bridge count

Pod policy:

- not required; cross-domain is an ingest expansion, not a throughput challenge

---

### `WS42` — Breakthrough Detection And Surprise Elevation

Purpose:

- split "this result is good" from "this result is surprising"
- a result can be below breakthrough threshold but still anomalous relative to
  the prior distribution of results in the vault
- anomalous results deserve a separate elevation pathway: priority replication,
  not standard falsification

The current `BreakthroughReport` (schemas.py:1339) conflates both. It has
`novelty_score` derived only from ablation signal and breakthrough detection
gated only on `control_score ≥ 1.8 AND calibration_score ≥ 0.85`. An
unexpectedly strong result that falls short of the breakthrough threshold
currently disappears into normal evidence flow with no special handling.

#### Schema changes

File: `tar_lab/schemas.py`

1. Add to `BreakthroughReport`:

```python
surprise_score: float = 0.0            # 0.0–1.0; how much this result differs from vault prior
prior_contradiction_score: float = 0.0 # 0.0–1.0; how strongly this contradicts prior accepted claims
anomaly_elevation_triggered: bool = False
anomaly_elevation_reason: Optional[str] = None
```

2. New schema `AnomalyElevationRecord`:

```python
class AnomalyElevationRecord(BaseModel):
    record_id: str
    project_id: str
    trial_id: str
    surprise_score: float
    prior_contradiction_score: float
    elevation_reason: str
    triggered_at: datetime
    status: str = "pending_replication"  # "pending_replication" | "replicated" | "refuted" | "promoted"
    replication_trial_id: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None
```

3. New schema `SurpriseThresholds` (added to `RuntimePolicy`):

```python
surprise_elevation_floor: float = 0.65     # trigger anomaly elevation above this
prior_contradiction_elevation_floor: float = 0.60
```

#### Verification changes

File: `tar_lab/verification.py`

1. Extend `build_breakthrough_report()`:

   After computing existing `novelty_score` and breakthrough check, add:

   ```python
   surprise_score = self._compute_surprise_score(trial_result, project_id)
   prior_contradiction_score = self._compute_prior_contradiction_score(trial_result, project_id)
   ```

2. Add `_compute_surprise_score(trial_result, project_id) -> float`:

   - retrieves up to 30 recent trial outcomes for the same project from vault
   - computes mean and std of the primary metric (e.g. final_score) across those
     trials
   - `surprise_score = sigmoid((result_value - mean) / (std + 1e-6))`
   - clamp to [0.0, 1.0]
   - if fewer than 5 prior results exist: return 0.0 (not enough prior to
     characterise)

3. Add `_compute_prior_contradiction_score(trial_result, project_id) -> float`:

   - retrieve accepted claims for the project from vault
   - for each accepted claim that makes a quantitative prediction about the
     primary metric, check if this trial's result opposes that prediction
   - `prior_contradiction_score = contradicting_claim_count / (total_claims + 1)`
   - clamp to [0.0, 1.0]

4. After computing both scores, check against `SurpriseThresholds`:

   ```python
   if (surprise_score >= policy.surprise_elevation_floor
           or prior_contradiction_score >= policy.prior_contradiction_elevation_floor):
       report.anomaly_elevation_triggered = True
       report.anomaly_elevation_reason = _format_elevation_reason(...)
   ```

#### Orchestrator changes

File: `tar_lab/orchestrator.py`

1. After `build_breakthrough_report()`, check `anomaly_elevation_triggered`:

   ```python
   if report.anomaly_elevation_triggered:
       self._elevate_anomaly(report, trial)
   ```

2. Add `_elevate_anomaly(report: BreakthroughReport, trial: Trial)`:

   - create `AnomalyElevationRecord` and persist to
     `tar_state/anomalies/{record_id}.json`
   - schedule a priority replication trial (same family, same config, new seed)
   - add to operator alert queue rather than standard evidence flow
   - do NOT route to falsification pipeline — anomalies replicate first

3. Add `list_anomaly_elevations(project_id: str) -> list[AnomalyElevationRecord]`

4. Add `resolve_anomaly(record_id: str, resolution: str, note: str)`:

   - transitions to `replicated` / `refuted` / `promoted`
   - if `promoted`: route accepted claim into normal evidence chain
   - if `refuted`: archive record, do not create accepted claim

#### Operator inspection

- CLI: `tar anomalies list [--project PROJECT]` → pending elevation records
- CLI: `tar anomalies resolve RECORD_ID --status replicated|refuted|promoted`
- Dashboard: anomaly alert count in project status panel; pending anomalies
  highlighted

#### Test requirements

File: `tests/test_surprise_elevation.py`

- `test_surprise_score_zero_insufficient_prior`: fewer than 5 prior results
  returns 0.0
- `test_surprise_score_nonzero_with_sufficient_prior`: seeded prior distribution
  + outlier result computes nonzero surprise_score
- `test_prior_contradiction_score`: accepted claim opposing trial result
  increases score
- `test_anomaly_elevation_triggered_above_threshold`: surprise_score ≥
  0.65 sets `anomaly_elevation_triggered = True`
- `test_anomaly_record_persisted`: elevation record written to
  `tar_state/anomalies/`
- `test_replication_trial_scheduled`: elevation schedules replication trial

#### Acceptance criteria

- `surprise_score` and `prior_contradiction_score` computed and stored on
  `BreakthroughReport`
- anomaly elevation triggered when either score exceeds threshold
- `AnomalyElevationRecord` persisted and retrievable
- priority replication trial scheduled automatically on elevation
- operator can list and resolve anomaly records via CLI
- standard falsification pipeline does NOT consume unresolved anomaly records

Pod policy:

- not required

---

### `WS43` — Competing Theory Engine

Purpose:

- every accepted claim is currently accepted in isolation
- a competing theory engine forces TAR to ask: "what is the simplest alternative
  explanation that also fits this data?" and then actively try to invalidate it

This is the difference between confirmation and serious science.

#### Schema changes

File: `tar_lab/schemas.py`

1. New schema `CompetingTheory`:

```python
class CompetingTheory(BaseModel):
    theory_id: str
    parent_claim_id: str
    project_id: str
    theory_text: str                    # plain-language alternative explanation
    supporting_evidence: list[str]      # claim or trial IDs that are consistent
    contradicting_evidence: list[str]   # claim or trial IDs that rule it out
    status: str = "active"              # "active" | "invalidated" | "superseded"
    invalidation_trial_id: Optional[str] = None
    confidence: float = 0.5
    generated_at: datetime
    generated_by: str                   # "operator" | "rule_heuristic"
```

2. New schema `HeadToHeadExperimentPlan`:

```python
class HeadToHeadExperimentPlan(BaseModel):
    plan_id: str
    project_id: str
    primary_claim_id: str
    competing_theory_id: str
    discriminating_condition: str       # what experimental condition distinguishes them
    proposed_family: str
    proposed_config_delta: dict         # modifications to standard family config
    expected_primary_outcome: str
    expected_competing_outcome: str
    created_at: datetime
    trial_id: Optional[str] = None      # filled when trial is run
```

3. New schema `TheoryComparisonReport`:

```python
class TheoryComparisonReport(BaseModel):
    report_id: str
    project_id: str
    claim_id: str
    competing_theories: list[CompetingTheory]
    head_to_head_plans: list[HeadToHeadExperimentPlan]
    surviving_theories: list[str]       # theory_ids not yet invalidated
    eliminated_theories: list[str]
    conclusion: str
    generated_at: datetime
```

#### Verification changes

File: `tar_lab/verification.py`

1. Add `generate_competing_theories(claim: AcceptedClaim, project_id: str, n: int = 3) -> list[CompetingTheory]`:

   - retrieves the claim text, supporting evidence, and related literature from
     vault
   - calls operator (via `LocalOpenAIRole` configured for `frontier` tier) with
     prompt:
     ```
     You are a rigorous scientific critic. Given the following accepted claim and
     its supporting evidence, generate {n} competing theories that could also
     explain the observed results without invoking the same mechanism.
     For each theory, specify: (a) the alternative mechanism, (b) what evidence
     is consistent with it, (c) what experiment would discriminate between this
     theory and the accepted claim.
     ```
   - parses structured response into `CompetingTheory` objects
   - if operator unavailable: fall back to rule heuristics (ablation-based
     alternatives only, marked `generated_by="rule_heuristic"`)

2. Add `build_head_to_head_plan(claim: AcceptedClaim, theory: CompetingTheory) -> HeadToHeadExperimentPlan`:

   - calls operator to propose the minimal experiment that discriminates between
     the accepted claim mechanism and the competing theory
   - returns a `HeadToHeadExperimentPlan`

3. Extend `generate_falsification_plan()` to include competing theory
   invalidation:

   - for every accepted claim with active competing theories, append
     `HeadToHeadExperimentPlan` items to the falsification plan
   - competing theory invalidation experiments are scheduled at higher priority
     than standard incremental falsification

4. Add `build_theory_comparison_report(project_id: str) -> TheoryComparisonReport`:

   - aggregates all competing theories and head-to-head plans for a project
   - computes surviving vs eliminated

#### Orchestrator changes

File: `tar_lab/orchestrator.py`

1. After a claim is accepted (currently via `_record_accepted_claim()`), call:

   ```python
   theories = self._runner.generate_competing_theories(claim, project_id)
   for theory in theories:
       self._persist_competing_theory(theory)
   ```

2. Add `_persist_competing_theory(theory: CompetingTheory)`:

   - write to `tar_state/theories/{theory_id}.json`
   - index into vault as kind `competing_theory`

3. Add `get_theory_comparison_report(project_id: str) -> TheoryComparisonReport`

4. Add `invalidate_theory(theory_id: str, trial_id: str)`:

   - transition theory to `status = "invalidated"`
   - link `invalidation_trial_id`
   - update `TheoryComparisonReport` surviving/eliminated counts

#### Operator inspection

- CLI: `tar theories list PROJECT_ID` → active competing theories
- CLI: `tar theories compare PROJECT_ID` → full theory comparison report
- CLI: `tar theories invalidate THEORY_ID --trial TRIAL_ID`
- Dashboard: theory count and unsatisfied head-to-head count in project panel

#### Test requirements

File: `tests/test_competing_theory_engine.py`

- `test_generate_competing_theories_rule_fallback`: without operator, rule
  heuristic generates ≥1 theory per claim
- `test_competing_theory_persisted`: theory written to
  `tar_state/theories/`
- `test_head_to_head_plan_created`: build_head_to_head_plan() returns valid plan
- `test_falsification_plan_includes_head_to_head`: generate_falsification_plan()
  includes head-to-head experiments when theories exist
- `test_theory_invalidated_after_trial`: invalidate_theory() transitions status
  correctly
- `test_comparison_report_surviving_count`: report counts surviving vs
  eliminated correctly

#### Acceptance criteria

- competing theories generated and persisted for every newly accepted claim
- head-to-head experiment plans created and included in falsification schedule
- head-to-head experiments run at higher priority than standard falsification
- theory status transitions tracked truthfully
- operator can list, compare, and invalidate theories via CLI
- `TheoryComparisonReport` exposes surviving and eliminated counts correctly

Pod policy:

- not required for rule heuristic path; frontier inference for operator-backed
  theory generation requires pod/GPU or API access

---

### `WS44` — Scientific Contribution Positioning

Purpose:

- TAR can produce findings, but currently it cannot say how those findings
  compare to the state of the art
- contribution positioning answers: "what does this project add that is not
  already known?"

This is required before any TAR output can be presented as a scientific
contribution.

#### Schema changes

File: `tar_lab/schemas.py`

1. Add to `PaperArtifact`:

```python
citation_network_depth: int = 0         # how many citation hops this paper is from project claims
sotatracker_score: Optional[float] = None   # metric value if paper is an SoTA result
sotatracker_task: Optional[str] = None
```

2. New schema `SoTAComparison`:

```python
class SoTAComparison(BaseModel):
    comparison_id: str
    claim_id: str
    closest_literature_claim_id: Optional[str]    # vault claim ID most similar to this claim
    closest_paper_id: Optional[str]
    similarity_score: float                         # 0.0–1.0 vault similarity
    our_metric_value: Optional[float]
    literature_metric_value: Optional[float]
    direction: str                                  # "better" | "worse" | "comparable" | "different_task"
    novelty_vs_literature: float                    # 1.0 - similarity_score, clipped to [0, 1]
    comparison_note: str
```

3. New schema `ContributionPositioningReport`:

```python
class ContributionPositioningReport(BaseModel):
    report_id: str
    project_id: str
    generated_at: datetime
    accepted_claims: list[str]           # claim IDs
    sotatracker_comparisons: list[SoTAComparison]
    mean_novelty_vs_literature: float
    contribution_summary: str            # operator or rule-generated plain text
    competing_theories_eliminated: int
    anomalies_resolved: int
    open_questions: list[str]
    limitations: list[str]
    recommended_next_gap: Optional[str]  # from WS36 gap scanner output
```

#### Orchestrator changes

File: `tar_lab/orchestrator.py`

1. Add `position_contribution(project_id: str) -> ContributionPositioningReport`:

   Step A — gather accepted claims for the project

   Step B — for each accepted claim:

   - query vault with claim text → retrieve top-3 similar literature claims
   - for each match: construct `SoTAComparison` with:
     - `similarity_score` from vault scorer
     - `novelty_vs_literature = 1.0 - similarity_score`
     - `direction` inferred from metric comparison when claim text and literature
       claim text contain extractable metric values, otherwise `"different_task"`

   Step C — call operator to generate `contribution_summary`:

   ```
   You are a scientific writer. Given the following accepted claims, their
   novelty scores versus the literature, and any resolved anomalies and
   competing theories, write a 3–5 sentence contribution summary suitable
   for a paper abstract. Be precise and honest about limitations.
   ```

   If operator unavailable: join claim texts with novelty scores into a
   structured plaintext summary (no operator call).

   Step D — attach:

   - `competing_theories_eliminated` from theory comparison report
   - `anomalies_resolved` from anomaly records
   - `open_questions` from project open questions list
   - `limitations` from project limitations list
   - `recommended_next_gap` from most recent `FrontierGapScanReport` for the
     project domain

   Step E — persist to `tar_state/positioning/{project_id}_{report_id}.json`

2. Add `get_contribution_positioning(project_id: str) -> Optional[ContributionPositioningReport]`

   - retrieves most recent report for a project

3. Wire into `publish_handoff_package()` (WS22 path):

   - attach `ContributionPositioningReport` to handoff package if one exists

#### VectorVault changes

File: `tar_lab/memory/vault.py`

1. Add `search_similar_claims(claim_text: str, top_k: int = 5, kind_filter: str = "claim") -> list[Hit]`

   - constrained search against claim-kind objects only
   - returns hits with `similarity_score` from reranker

2. Add `get_sotatracker_score(paper_id: str) -> Optional[float]`

   - returns `sotatracker_score` from paper metadata if set
   - returns None if not set

#### Operator inspection

- CLI: `tar contribution position PROJECT_ID` → generate and display positioning
  report
- CLI: `tar contribution show PROJECT_ID` → display most recent existing report
- Dashboard: novelty-vs-literature score and contribution summary visible in
  project detail panel

#### Test requirements

File: `tests/test_contribution_positioning.py`

- `test_sotatracker_comparison_created`: claim with matching vault literature
  claim produces `SoTAComparison` with correct similarity
- `test_novelty_vs_literature_complement`: `novelty_vs_literature = 1.0 -
  similarity_score` correctly computed
- `test_positioning_report_persisted`: `position_contribution()` writes to
  `tar_state/positioning/`
- `test_no_operator_fallback`: without operator, report generates with
  structured plaintext summary, no crash
- `test_handoff_package_includes_positioning`: `publish_handoff_package()`
  attaches positioning report when available

#### Acceptance criteria

- `SoTAComparison` created for every accepted claim with vault similarity search
- `novelty_vs_literature` truthfully reflects distance from existing literature
  claims (not a self-reported score)
- `ContributionPositioningReport` persisted and retrievable
- `publish_handoff_package()` attaches positioning report
- CLI commands for generating and displaying positioning reports work
- report is available without operator inference (fallback path)

Pod policy:

- not required; operator-backed contribution summary benefits from frontier
  model but falls back correctly without it

---

## Execution Dependency Map

Phase 5 must be fully closed before Phase 6 begins.

Within Phase 6, the recommended order is:

```
WS40 (frontier operator scale)
  -> WS41 (cross-domain corpus)
    -> WS42 (surprise elevation)        <- can start in parallel with WS43
    -> WS43 (competing theory engine)   <- can start in parallel with WS42
      -> WS44 (contribution positioning)
```

`WS42` and `WS43` may run in parallel after `WS41` closes — they share no
code dependencies. Both must complete before `WS44`.

Parallel execution is not required. Sequential is safe and correct.

## New Files Summary

| File | WS | Purpose |
| --- | --- | --- |
| `tests/test_frontier_operator_scale.py` | WS40 | routing and cost tracking tests |
| `tests/test_cross_domain_corpus.py` | WS41 | cross-domain ingest and bridge tests |
| `tests/test_surprise_elevation.py` | WS42 | surprise score and anomaly elevation tests |
| `tests/test_competing_theory_engine.py` | WS43 | theory generation and comparison tests |
| `tests/test_contribution_positioning.py` | WS44 | SoTA comparison and positioning tests |
| `tar_state/anomalies/` | WS42 | persisted anomaly elevation records |
| `tar_state/theories/` | WS43 | persisted competing theory records |
| `tar_state/positioning/` | WS44 | persisted contribution positioning reports |
| `tar_state/routing/` | WS40 | model routing records |
| `tar_state/literature/bridges/` | WS41 | cross-domain bridge records |

## Schema Change Summary

| Schema | File | WS | Change |
| --- | --- | --- | --- |
| `LocalLLMConfig` | schemas.py:89 | WS40 | add `model_tier`, `cost_per_token_input/output`, `context_window`, `supports_tool_use` |
| `FrontierModelConfig` | schemas.py | WS40 | new schema: tier routing config and budget cap |
| `ModelRoutingRecord` | schemas.py | WS40 | new schema: per-inference routing audit record |
| `PaperArtifact` | schemas.py | WS41 | add `primary_domain`, `secondary_domains`, `cross_domain_links`, `imported_from_domain` |
| `CrossDomainBridgeRecord` | schemas.py | WS41 | new schema: cross-domain concept bridge |
| `DomainProfile` | schemas.py | WS41 | new schema: ingest domain configuration |
| `BreakthroughReport` | schemas.py:1339 | WS42 | add `surprise_score`, `prior_contradiction_score`, `anomaly_elevation_triggered`, `anomaly_elevation_reason` |
| `AnomalyElevationRecord` | schemas.py | WS42 | new schema: anomaly elevation lifecycle |
| `SurpriseThresholds` | schemas.py | WS42 | new schema: surprise elevation floor config |
| `CompetingTheory` | schemas.py | WS43 | new schema: alternative explanation record |
| `HeadToHeadExperimentPlan` | schemas.py | WS43 | new schema: discriminating experiment plan |
| `TheoryComparisonReport` | schemas.py | WS43 | new schema: project-level theory comparison |
| `SoTAComparison` | schemas.py | WS44 | new schema: single claim vs literature comparison |
| `ContributionPositioningReport` | schemas.py | WS44 | new schema: project-level contribution positioning |
| `PaperArtifact` | schemas.py | WS44 | add `citation_network_depth`, `sotatracker_score`, `sotatracker_task` |

## Completion Picture — Phase 6

When Phase 6 is complete, TAR will be genuinely frontier-level:

- The Director operates at 70B+ class when it matters, with cost-aware routing
  that does not waste frontier inference on Scout-grade decisions
- Literature ingest spans physics, mathematics, neuroscience, and AI — analogies
  and prior results cross domain boundaries
- Anomalous results are not buried in the standard evidence flow — they are
  flagged, replicated, and either promoted or refuted explicitly
- Every accepted claim faces an explicit competing theory that TAR actively
  tries to invalidate through discriminating experiments
- Every project produces a contribution positioning report that compares its
  findings against the vault of known literature claims — novelty is measured,
  not asserted

That is what frontier science looks like when built into the loop rather than
added at the end.
