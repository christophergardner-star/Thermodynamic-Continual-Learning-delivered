"""TAR Keystone H2 — experiment-design planner (ENG skeleton).

Turns a hypothesis into a DISCRIMINATING protocol — the baselines that could
falsify the claim, the ablations that isolate the mechanism, and a seed count
*powered* to detect the claimed effect — instead of the hardcoded method/seed
lists in tar_research_director._frontier_experiment_catalog (:1929).

This is the [ENG] core of TAR_H2_ExperimentDesign_Spec.md:
  - the HypothesisSpec -> DiscriminatingProtocol contract,
  - the powered seed count (inverts tar_lab.stat_utils._solve_n_for_power),
  - the discriminating invariant (a protocol with no falsifying baseline or no
    mechanism-isolating ablation is a sweep, not a test -> raises).

The adversarial critic (research taste) and external-SoTA baseline RANKING are
[RESEARCH] and are intentionally left as hooks (design_critic is a no-op stub;
baselines are supplied by the caller from the catalog + SoTA db). Pure + torch-free.
Truth-lock precedence is absolute: this planner has NO authority over the gate — it
never sets publication_allowed or touches a quarantine; designed experiments hit the
same verify_canonical_3gate as any other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tar_lab.method_identity import method_identity
from tar_lab.stat_utils import _solve_n_for_power, _nct_power

# Mechanism-isolating ablation pool for the CL<->Thermo spine (the home turf where
# ablations are concrete). Each member differs from the others in exactly one
# mechanism flag (proxy vs canonical penalty; regime observer on/off), so an arm that
# wins attributes the effect to a MECHANISM, not to "the TCL family".
_TCL_ABLATION_POOL = ("tcl", "tcl_penalty_only", "tcl_canonical", "tcl_full")


class NonDiscriminatingProtocolError(ValueError):
    """Raised when a protocol would be a sweep, not a test: no baseline that can
    falsify the claim, or no ablation that isolates the named mechanism."""


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    claim: str
    primary_method: str
    domain_id: str
    dataset: str
    direction: str = "less"          # "less" (forgetting) | "greater"
    min_effect_d: float = 0.5        # smallest scientifically-meaningful Cohen's d
    null_prediction: str = ""
    mechanism_components: list[str] = field(default_factory=list)  # explicit ablation arms (non-spine)
    backbone: str = "resnet18"
    epochs: int = 40


@dataclass(frozen=True)
class DiscriminatingProtocol:
    hypothesis_id: str
    dataset: str
    methods: list[str]               # primary + falsifying baselines + ablations
    baselines: list[str]             # the falsifying competitors (subset of methods)
    ablations: list[str]             # mechanism-isolating arms (subset of methods)
    seeds: list[int]                 # POWERED count (not a constant)
    power_target: float
    achieved_power: float
    min_effect_d: float
    correction: str                  # "holm"
    preregistration: dict            # frozen: hypothesis + primary endpoint + stop rule
    config_overrides: dict           # executor/ExperimentSpec contract
    estimated_runtime_h: float
    design_rationale: str
    seed_amendment: Optional[dict] = None  # set when powered n exceeds the runtime budget
    critic_notes: list[str] = field(default_factory=list)


def _falsifying_baselines(primary: str, available: list[str]) -> list[str]:
    """The established baselines that, if they win, refute the claim: any available
    method NOT in the primary's family (so a within-family arm can't masquerade as a
    falsifier)."""
    fam = method_identity(primary)["family"]
    out: list[str] = []
    for m in available:
        if not m or m == primary:
            continue
        if method_identity(m)["family"] != fam and m not in out:
            out.append(m)
    return out


def _mechanism_ablations(hyp: HypothesisSpec) -> list[str]:
    """Arms that isolate the primary method's mechanism. Explicit components win;
    otherwise, for the TCL spine, the other family members (which differ by one
    mechanism flag each)."""
    if hyp.mechanism_components:
        return [m for m in hyp.mechanism_components if m and m != hyp.primary_method]
    if method_identity(hyp.primary_method)["in_tcl_family"]:
        return [m for m in _TCL_ABLATION_POOL
                if m != hyp.primary_method and method_identity(m)["in_tcl_family"]]
    return []


def design_experiment(
    hyp: HypothesisSpec,
    *,
    available_baselines: list[str],
    power_target: float = 0.80,
    alpha: float = 0.05,
    runtime_budget_h: Optional[float] = None,
    per_arm_seed_h: float = 0.25,
    seed_pool: Optional[list[int]] = None,
    critic: Optional[Callable[["DiscriminatingProtocol"], list[str]]] = None,
) -> DiscriminatingProtocol:
    """Produce a discriminating, pre-registered, powered protocol for *hyp*.

    `available_baselines` is supplied by the caller from the well-known catalog
    (tar_frontier._catalog_defaults_for_domain) + the live SoTA bar
    (best_result(exclude_source='tar_internal')). Raises NonDiscriminatingProtocolError
    if no baseline can falsify the claim or no ablation isolates the mechanism.
    """
    baselines = _falsifying_baselines(hyp.primary_method, available_baselines)
    if not baselines:
        raise NonDiscriminatingProtocolError(
            f"No falsifying baseline for '{hyp.primary_method}' in {available_baselines}: "
            f"a protocol with no competitor that could refute the claim is a sweep, not a test."
        )
    ablations = _mechanism_ablations(hyp)
    if not ablations:
        raise NonDiscriminatingProtocolError(
            f"No mechanism-isolating ablation for '{hyp.primary_method}': supply "
            f"mechanism_components, or the protocol cannot attribute an effect to a mechanism."
        )

    # Powered seed count (replaces the hardcoded seeds=).
    n_full = _solve_n_for_power(abs(hyp.min_effect_d), power_target=power_target, alpha=alpha)

    # Runtime-budget reconciliation: NEVER silently under-power. If the powered n
    # exceeds the budget, run the budget-feasible n and emit an append-only,
    # human-gated seed amendment (RAIL-3 — same stance as calibration_learner B3).
    seed_amendment: Optional[dict] = None
    n = n_full
    n_arms = 1 + len(baselines) + len(ablations)
    if runtime_budget_h is not None and per_arm_seed_h > 0:
        max_seeds = int(runtime_budget_h / (per_arm_seed_h * n_arms))
        if 0 < max_seeds < n_full:
            seed_amendment = {
                "reason": "powered_n_exceeds_runtime_budget",
                "powered_n": n_full,
                "budget_feasible_n": max_seeds,
                "status": "proposed",         # human-gated; never auto-applied
            }
            n = max_seeds
    n = max(n, 2)

    pool = seed_pool or list(range(1000))
    seeds = list(pool[:n])
    achieved = round(_nct_power(abs(hyp.min_effect_d), len(seeds), alpha), 4)
    methods = [hyp.primary_method] + [m for m in baselines + ablations if m != hyp.primary_method]
    # dedupe, order-preserving
    seen: set[str] = set()
    methods = [m for m in methods if not (m in seen or seen.add(m))]

    primary_endpoint = "mean_forgetting" if hyp.direction == "less" else "mean_accuracy"
    preregistration = {
        "hypothesis_id": hyp.hypothesis_id,
        "claim": hyp.claim,
        "primary_method": hyp.primary_method,
        "primary_endpoint": primary_endpoint,
        "direction": hyp.direction,
        "min_effect_d": hyp.min_effect_d,
        "power_target": power_target,
        "alpha": alpha,
        "n_seeds": len(seeds),
        "multiple_testing_correction": "holm",
        "stop_rule": "fixed-n with SPRT early-stop; n is the pre-registered ceiling",
        "frozen": True,
    }
    config_overrides = {
        "dataset": hyp.dataset,
        "methods": methods,
        "seeds": seeds,
        "backbone": hyp.backbone,
        "epochs": hyp.epochs,
        "correction": "holm",
    }
    proto = DiscriminatingProtocol(
        hypothesis_id=hyp.hypothesis_id,
        dataset=hyp.dataset,
        methods=methods,
        baselines=baselines,
        ablations=ablations,
        seeds=seeds,
        power_target=power_target,
        achieved_power=achieved,
        min_effect_d=hyp.min_effect_d,
        correction="holm",
        preregistration=preregistration,
        config_overrides=config_overrides,
        estimated_runtime_h=round(per_arm_seed_h * n_arms * len(seeds), 2),
        design_rationale=(
            f"Test of '{hyp.claim}': primary={hyp.primary_method}; "
            f"falsifying baselines={baselines}; mechanism ablations={ablations}; "
            f"n={len(seeds)} powered to {achieved:.0%} for d={hyp.min_effect_d} "
            f"(Holm across {n_arms} arms)."
            + ("" if seed_amendment is None
               else f" NOTE: powered n={n_full} exceeds budget; running {len(seeds)} + proposing a seed amendment.")
        ),
        seed_amendment=seed_amendment,
        critic_notes=[],
    )
    # [RESEARCH] hook: advisory-only; the critic can ADD named risks but has NO
    # authority over the gate (cannot set publication_allowed / clear a quarantine).
    if critic is not None:
        notes = list(critic(proto) or [])
        proto = DiscriminatingProtocol(**{**proto.__dict__, "critic_notes": notes})
    return proto
