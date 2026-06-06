# TAR Keystone H2 — Experiment-Design Module (build-ready spec)

**Status:** spec (not implemented). Branch `phase0-stackbridge`. Style mirrors
`TAR_TruthLock_Implementation_Plan.md` — file:line targets, module boundary, contract, exit tests.

**Why this is the highest-leverage underbuilt horizon.** TAR today *runs* experiments but does not
*design* them. The director's `_frontier_experiment_catalog`
(`tar_research_director.py:1929`) emits **hardcoded** per-frontier dicts — fixed method lists, fixed
`seeds`, fixed `config_overrides`, fixed `estimated_runtime_h` (see the `if frontier_id == …` chain
at ~:1987–2326). Given a hypothesis, a proficient researcher instead derives a **discriminating
protocol**: the baselines that could *kill* the claim, the ablation that isolates the mechanism, the
confound to control, and the seed count *powered* to detect the claimed effect. TAR already owns the
statistics (`tar_lab/stat_utils.py`: `power_analysis`, `holm_bonferroni`, `sprt_boundary`,
`_nct_power`); what's missing is the **design step that consumes them**. This is the hinge between
"automated experimentation" and "autonomous research."

Classification: **[RESEARCH]** for the design heuristics + critic quality; **[ENG]** for the module
boundary, stat wiring, and gate integration. Depends on **H0** (truth-lock enforced + history
reconciled — DONE) and **H1** (loops compound — populated SoTA tables feed baseline selection).

---

## 1. Module boundary

New file: **`tar_lab/experiment_design.py`** (pure, importable without torch; unit-testable offline).
It is a *planner*, not an executor — it never runs training, only emits a protocol object.

```
hypothesis (HypothesisSpec)  ─┐
domain knowledge (catalog)   ─┤→  design_experiment()  →  DiscriminatingProtocol
prior results (recall/SoTA)  ─┤        (+ optional adversarial critic pass)
external baselines (SoTA db) ─┘
```

### 1.1 Contract (dataclasses)

```python
@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    claim: str                  # "tcl_full reduces forgetting vs EWC on Split-CIFAR-10"
    primary_method: str         # method-identity-checked (tar_lab/method_identity.py)
    domain_id: str
    dataset: str
    direction: str              # "less" | "greater"  (forgetting: less)
    min_effect_d: float         # smallest scientifically-meaningful Cohen's d to detect
    null_prediction: str        # what the null world looks like (for falsification)

@dataclass(frozen=True)
class DiscriminatingProtocol:
    hypothesis_id: str
    methods: list[str]          # primary + the baselines that can FALSIFY it
    ablations: list[dict]       # mechanism-isolating arms (e.g. tcl_full vs tcl_penalty_only)
    controlled_confounds: list[str]  # named confounds + how each is held fixed
    seeds: list[int]            # POWERED count from stat_utils.power_analysis (not a constant)
    power_target: float         # e.g. 0.8
    correction: str             # "holm" (family-wise across the arms)
    sequential: dict            # SPRT params (early-stop boundaries) | {}
    config_overrides: dict      # the existing executor contract (matches catalog shape)
    estimated_runtime_h: float
    preregistration: dict       # frozen hypothesis + primary endpoint + stop rule (RAIL-3)
    design_rationale: str       # human-auditable "why these arms / why this n"
    critic_notes: list[str]     # adversarial pass output (uninformative-result risks)
```

**Invariant:** `methods` MUST include ≥1 baseline that, if it wins, refutes the hypothesis; and
`ablations` MUST isolate the named `primary_method` mechanism (else the protocol is a sweep, not a
test). `design_experiment()` raises `NonDiscriminatingProtocolError` if either is empty.

---

## 2. The design algorithm (what `design_experiment()` does)

1. **Baseline selection (falsification-first).** Pull the established baselines for `domain_id` from
   the well-known catalog (`tar_frontier.py:68–256` `_catalog_defaults_for_domain` →
   `external_baselines`) AND the live SoTA bar (`literature_graph.db` `best_result(...,
   exclude_source="tar_internal")` — the Seam-2 internal-baseline bar). Always include the *strongest*
   competitor, not a weak one. **[RESEARCH]**: ranking "most discriminating baseline."
2. **Ablation derivation (mechanism isolation).** From `method_identity(primary_method)`
   (`tar_lab/method_identity.py:29`): if `uses_uniform_l2_proxy` vs `is_canonical_tcl` differ, the
   ablation set must contain both the proxy and the canonical arm so the verdict attributes the
   effect to the *mechanism*, not the family. Generalize: each claimed mechanism component → one
   leave-one-out arm.
3. **Confound control.** Fixed shared config across arms (same backbone/epochs/optimizer); list each
   held-fixed confound in `controlled_confounds`. **[RESEARCH]**: confound enumeration per domain.
4. **Powered seed count.** Invert `stat_utils.power_analysis` /`_nct_power`
   (`tar_lab/stat_utils.py:368`): find the smallest `n` s.t. `achieved_power(min_effect_d, n) ≥
   power_target`. This replaces the hardcoded `seeds=[42,0,1,2,3]`. Cap by a runtime budget; if the
   powered `n` exceeds budget, emit a `calibration_learner`-style **seed-amendment** (B3,
   `calibration/preregistration_amendments.json`) rather than silently under-powering.
5. **Correction + sequential plan.** `correction="holm"` (`holm_bonferroni`) across all arms;
   `sprt_boundary` params for early-stop so a clearly-won/lost arm stops without burning the full n.
6. **Pre-registration.** Freeze {hypothesis, primary endpoint (mean_forgetting, direction),
   stop rule, n, correction} into `preregistration` → written to `tar_state/active_preregistration.json`
   BEFORE any run (RAIL-3 forbids post-hoc n changes; amendments are append-only + human-gated).
7. **Adversarial critic (optional, gated).** See §3.

Output `DiscriminatingProtocol.config_overrides` MUST match the existing executor/`ExperimentSpec`
contract so it drops into `_specs_from_directives` (`tar_living_research.py`) unchanged.

---

## 3. The adversarial critic (the "research taste" layer)

A pre-run and post-run critic, routed through the **WS40 cost-aware model router**
(`tar_lab/llm_bridge.py` — cached per content-hash, frontier-tier model only when the cost model says
the decision warrants it):

- **Pre-run:** *"What would make this result uninformative?"* → returns missing baselines/confounds →
  fed back to amend the protocol (one bounded refinement loop, not unbounded).
- **Post-run (H2.5, after the experiment):** *"What alternative explains this result?"* → attached to
  the verdict as `critic_notes`; a strong alternative explanation downgrades the claim's confidence
  and can open a `negative_result` gap (reuse the K2.2c path, `tar_living_research.py`
  `_write_results_to_knowledge_graph`).

The critic NEVER fabricates data and NEVER overrides the truth-lock gate — it only adds named risks +
calibrated-confidence adjustments. Truth-lock precedence is absolute (H0 before H2).

---

## 4. Integration points (where it plugs in)

| Site | File:line | Change |
|---|---|---|
| Problem → protocol | `tar_research_director.py:1540` `_pick_novel_problem_for_domain` | after a novel problem is picked, call `design_experiment(HypothesisSpec(...))` |
| Replace hardcoded catalog | `tar_research_director.py:1929` `_frontier_experiment_catalog` | for `fp-gap-*` (and progressively the `if id==` arms), return `protocol.config_overrides` + `methods`/`seeds` from the design module instead of the literal dicts |
| Spec construction | `tar_living_research.py` `_specs_from_directives` | consume the protocol's `methods`/`seeds`/`config_overrides` (shape already matches) |
| Prereg | `tar_state/active_preregistration.json` + `calibration/preregistration_amendments.json` (B3) | protocol writes prereg before run; under-power → amendment, not silent change |
| Verify | `tar_lab/canonical_registry.verify_canonical_3gate` (unchanged) | designed experiments hit the SAME gate as any other — no special path |

**Roll-out is gated, not a flip:** behind a flag `tar_state/experiment_design.enabled` (mirrors B2/B4
pattern); off → the legacy catalog path is unchanged. Enable for `continual_learning` only first.

---

## 5. Exit tests (`tests/test_h2_experiment_design.py`) — offline, torch-free

1. `test_protocol_is_discriminating`: a HypothesisSpec for `tcl_full` yields a protocol whose
   `methods` include EWC/SI (refuting baselines) AND `ablations` include the proxy arm; asserting
   `NonDiscriminatingProtocolError` when baselines/ablations would be empty.
2. `test_seed_count_is_powered`: for `min_effect_d=0.5, power_target=0.8`, the returned `seeds` count
   equals the smallest `n` with `power_analysis(0.5, n) ≥ 0.8` (verify against `stat_utils` directly).
3. `test_overpower_emits_amendment`: when powered `n` exceeds the runtime budget, a seed-amendment is
   proposed (append-only, human-gated) rather than the protocol silently under-powering.
4. `test_config_overrides_match_executor_contract`: the protocol's `config_overrides` round-trip into
   an `ExperimentSpec` without error.
5. `test_critic_never_bypasses_gate` (injected fake model): critic output is advisory only — it cannot
   set `publication_allowed` or flip a quarantine.
6. `test_prereg_frozen_before_run`: `active_preregistration.json` is written with the primary endpoint
   + stop rule before any spec is queued.

**DoD (the H2 exit criterion, human-verifiable):** for a posed hypothesis, TAR autonomously produces
a pre-registered protocol with explicit refuting baselines + mechanism-isolating ablations + a
*powered* seed count, runs it through the unchanged truth-lock gate, and reaches a verdict that
distinguishes the hypothesis from the obvious confound — and a domain reviewer agrees *"this is how I'd
have designed it."*

---

## 6. Sequencing + honest scope

- **[ENG], days:** module skeleton + the stat-powered seed count + config_overrides contract + flag +
  tests 2/3/4/6. This alone removes the hardcoded `seeds=` and makes n *earned*.
- **[RESEARCH], weeks:** baseline-discrimination ranking, confound enumeration per domain, and the
  adversarial critic quality — these *are* "research taste" and their quality is itself the outcome;
  budget for iteration, measure against the DoD's "how I'd have designed it" review.
- **Precondition:** H1's SoTA tables must be non-empty (baseline selection reads them); until then the
  design module falls back to the well-known catalog defaults only (still discriminating, just not
  externally-grounded).
- **Truth-lock precedence:** every designed experiment passes the SAME `verify_canonical_3gate` + TL-4
  tier as a routine one. A better designer that bypassed the gate would just produce false claims
  faster — which is exactly why H0 ships before H2.
```
