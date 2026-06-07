"""H2 experiment-design skeleton — exit tests (offline, torch-free).

Covers the [ENG] core of TAR_H2_ExperimentDesign_Spec.md: the discriminating
invariant, the powered seed count, over-power -> amendment (never silent
under-power), the executor contract shape, critic-is-advisory-only, and
prereg-frozen.
"""
import json

import pytest

from tar_lab.experiment_design import (
    HypothesisSpec,
    DiscriminatingProtocol,
    NonDiscriminatingProtocolError,
    design_experiment,
)
from tar_lab.stat_utils import _solve_n_for_power, _nct_power


def _tcl_hyp(**kw):
    base = dict(
        hypothesis_id="h-tcl-full-cifar10",
        claim="tcl_full reduces forgetting vs EWC on Split-CIFAR-10",
        primary_method="tcl_full",
        domain_id="continual_learning",
        dataset="split_cifar10",
        direction="less",
        min_effect_d=0.5,
    )
    base.update(kw)
    return HypothesisSpec(**base)


def test_protocol_is_discriminating():
    p = design_experiment(_tcl_hyp(), available_baselines=["ewc", "si", "sgd_baseline"])
    # falsifying baselines present (non-TCL family)
    assert "ewc" in p.baselines and "si" in p.baselines
    # mechanism-isolating ablations present (proxy / canonical arms)
    assert "tcl_penalty_only" in p.ablations and "tcl_canonical" in p.ablations
    # methods = primary + baselines + ablations, primary first, deduped
    assert p.methods[0] == "tcl_full"
    assert set(p.baselines).issubset(set(p.methods))
    assert set(p.ablations).issubset(set(p.methods))


def test_non_discriminating_raises():
    # no falsifying baseline (only same-family members) -> raise
    with pytest.raises(NonDiscriminatingProtocolError):
        design_experiment(_tcl_hyp(), available_baselines=["tcl", "tcl_canonical"])
    # non-TCL primary with no mechanism_components -> no ablation -> raise
    with pytest.raises(NonDiscriminatingProtocolError):
        design_experiment(
            _tcl_hyp(primary_method="ewc", claim="ewc beats sgd"),
            available_baselines=["sgd_baseline", "si"],
        )


def test_seed_count_is_powered():
    p = design_experiment(_tcl_hyp(min_effect_d=0.5), available_baselines=["ewc", "si"],
                          power_target=0.80)
    expected_n = _solve_n_for_power(0.5, power_target=0.80)
    assert len(p.seeds) == expected_n
    # the achieved power at that n actually meets the target
    assert _nct_power(0.5, len(p.seeds)) >= 0.80
    assert p.achieved_power >= 0.80


def test_overpower_emits_amendment_not_silent_underpower():
    # tiny effect -> huge powered n; a tight budget forces fewer seeds, but it must
    # PROPOSE an amendment rather than silently under-power.
    p = design_experiment(
        _tcl_hyp(min_effect_d=0.05),
        available_baselines=["ewc", "si"],
        power_target=0.80,
        runtime_budget_h=6.0,
        per_arm_seed_h=0.25,
    )
    assert p.seed_amendment is not None
    assert p.seed_amendment["status"] == "proposed"           # human-gated, not auto-applied
    assert p.seed_amendment["powered_n"] > p.seed_amendment["budget_feasible_n"]
    # run the budget-feasible n, but never below the statistical floor of 2 seeds
    assert len(p.seeds) == max(p.seed_amendment["budget_feasible_n"], 2)
    assert len(p.seeds) < p.seed_amendment["powered_n"]       # genuinely capped, not silent-full


def test_config_overrides_match_executor_contract():
    p = design_experiment(_tcl_hyp(), available_baselines=["ewc", "si", "sgd_baseline"])
    co = p.config_overrides
    for k in ("dataset", "methods", "seeds", "backbone", "epochs"):
        assert k in co
    # must be JSON-serializable (it rides into ExperimentSpec.config_overrides)
    json.dumps(co)
    assert co["seeds"] == p.seeds and co["dataset"] == p.dataset


def test_critic_is_advisory_only_never_touches_gate():
    captured = {}

    def fake_critic(proto: DiscriminatingProtocol):
        captured["seen"] = proto.hypothesis_id
        return ["confound: arms not matched on optimizer", "alt: effect may be seed variance"]

    p = design_experiment(_tcl_hyp(), available_baselines=["ewc", "si"], critic=fake_critic)
    assert captured["seen"] == p.hypothesis_id
    assert len(p.critic_notes) == 2
    # the planner has NO authority over the truth-lock gate
    assert not hasattr(p, "publication_allowed")
    assert not hasattr(p, "canonical_verified")
    assert not hasattr(p, "quarantined")


def test_director_fp_gap_uses_powered_protocol_only_when_enabled(tmp_path):
    """Wiring test: the director's fp-gap probe keeps the hardcoded seeds= when the flag
    is absent, and switches to a powered + discriminating protocol when
    tar_state/experiment_design.enabled exists. Torch-free (reasoning layer)."""
    from tar_research_director import ResearchDirector
    from tar_lab.stat_utils import _solve_n_for_power

    (tmp_path / "tar_state").mkdir()
    d = ResearchDirector(tmp_path)
    frontier = {
        "problem_id": "fp-gap-test-cl-tcl",
        "title": "Gap test",
        "domain": "continual_learning",          # must be in _FRONTIER_AUTONOMY_DOMAINS
        "global_problem_statement": "tcl reduces forgetting vs established baselines",
        "candidate_datasets": ["split_cifar10"],
        "candidate_backbones": ["resnet18"],
        "external_baselines": ["ewc", "si", "sgd_baseline"],
    }
    flag = tmp_path / "tar_state" / "experiment_design.enabled"

    legacy = d._frontier_experiment_catalog(frontier, {}, None)
    assert legacy and legacy[0]["seeds"] == [42, 0, 1, 2, 3]
    assert legacy[0]["config_overrides"].get("experiment_design") is None

    flag.write_text("", encoding="utf-8")
    designed = d._frontier_experiment_catalog(frontier, {}, None)
    assert designed
    spec = designed[0]
    co = spec["config_overrides"]
    assert co.get("experiment_design") is True
    assert spec["seeds"] != [42, 0, 1, 2, 3]               # not a constant
    assert any(m in spec["comparison_methods"]            # mechanism-isolating ablations
               for m in ("tcl_penalty_only", "tcl_canonical", "tcl_full"))
    # the powered n is stat_utils-derived (in the amendment when budget-capped, else powered_seeds)
    powered = (co.get("seed_amendment") or {}).get("powered_n") or co.get("powered_seeds")
    assert powered == _solve_n_for_power(0.5, 0.8)


def test_director_helper_designs_probes_but_skips_suite_and_resume(tmp_path):
    """_apply_experiment_design upgrades fresh director frontier_probe/gap_probe arms,
    but NEVER touches suite/resume reruns (fixed protocols), is idempotent, and no-ops
    when the flag is absent. Torch-free."""
    from tar_research_director import ResearchDirector
    (tmp_path / "tar_state").mkdir()
    d = ResearchDirector(tmp_path)
    flag = tmp_path / "tar_state" / "experiment_design.enabled"
    flag.write_text("", encoding="utf-8")
    frontier = {"domain": "continual_learning"}

    probe = {"experiment_id": "director-x-probe", "proposal_origin": "director",
             "proposal_kind": "frontier_probe", "method": "tcl", "dataset": "split_cifar10",
             "external_baselines": ["ewc", "si", "sgd_baseline"], "seeds": [42, 0, 1, 2, 3],
             "estimated_runtime_h": 8.0, "config_overrides": {}}
    out = d._apply_experiment_design(probe, frontier)
    assert out["config_overrides"]["experiment_design"] is True
    assert out["seeds"] != [42, 0, 1, 2, 3]
    assert any(m in out["comparison_methods"] for m in ("tcl_penalty_only", "tcl_canonical", "tcl_full"))

    suite = {"experiment_id": "phase16", "proposal_origin": "suite", "proposal_kind": "resume_suite",
             "method": "tcl", "dataset": "split_cifar100", "external_baselines": ["ewc", "sgd_baseline"],
             "seeds": [42, 0, 1, 2, 3], "config_overrides": {}}
    out2 = d._apply_experiment_design(suite, frontier)
    assert out2["seeds"] == [42, 0, 1, 2, 3]                      # untouched
    assert out2["config_overrides"].get("experiment_design") is None

    designed = {"proposal_origin": "director", "proposal_kind": "frontier_probe",
                "config_overrides": {"experiment_design": True}, "seeds": [1, 2]}
    assert d._apply_experiment_design(designed, frontier)["seeds"] == [1, 2]   # idempotent

    flag.unlink()
    assert d._apply_experiment_design(dict(probe), frontier)["seeds"] == [42, 0, 1, 2, 3]  # flag off -> no-op


def test_preregistration_frozen_with_primary_endpoint_and_stop_rule():
    p = design_experiment(_tcl_hyp(direction="less"), available_baselines=["ewc", "si"])
    pre = p.preregistration
    assert pre["frozen"] is True
    assert pre["primary_endpoint"] == "mean_forgetting"
    assert pre["direction"] == "less"
    assert pre["n_seeds"] == len(p.seeds)
    assert pre["multiple_testing_correction"] == "holm"
    assert "stop_rule" in pre
