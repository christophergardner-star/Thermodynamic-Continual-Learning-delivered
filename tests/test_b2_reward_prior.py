"""Workstream B2 — bounded, exploration-safe REWARD prior from outcome reliability.

TAR could previously only PUNISH a priority (vetoes, failure penalty). outcome_learner.
reliability_reward() adds the first reinforcement signal, with guarantees:
  * bounded (<= cap, far below the failure penalty + status boosts);
  * keyed on OPERATIONAL reliability (clean completions + reproducible low variance),
    NOT a favourable scientific outcome -> a config that reliably does *badly* earns the
    same reward, so a single winning method is never reinforced (exploration preserved);
  * a novel config (no track record) earns 0, never a penalty.

The director gates this OFF unless tar_state/outcome_reward.enabled exists (inert default);
that flag wiring is verified by inspection — here we pin the pure scoring function.
"""
from tar_lab import outcome_learner as ol


def _priors():
    return {
        "by_experiment": {
            "exp_reliable": {"method": "tcl", "dataset": "split_cifar10"},
            "exp_flaky": {"method": "ewc", "dataset": "split_cifar10"},
        },
        "by_method_dataset": {
            # clean + reproducible -> max reward
            "tcl::split_cifar10": {
                "method": "tcl", "dataset": "split_cifar10", "n": 5, "completed": 5,
                "failed": 0, "forgetting_mean": 0.13, "forgetting_std": 0.01,
                "n_forgetting": 5, "reproducible": True,
            },
            # mostly fails -> not reliable
            "ewc::split_cifar10": {
                "method": "ewc", "dataset": "split_cifar10", "n": 6, "completed": 2,
                "failed": 4, "reproducible": False,
            },
            # clean but not reproducible -> partial reward
            "tcl_partial::split_cifar10": {
                "method": "tcl_partial", "dataset": "split_cifar10", "n": 4, "completed": 3,
                "failed": 1, "reproducible": False,
            },
            # reliably BAD (high forgetting) but clean + reproducible -> still rewarded
            "tcl_bad::split_cifar10": {
                "method": "tcl_bad", "dataset": "split_cifar10", "n": 4, "completed": 4,
                "failed": 0, "forgetting_mean": 0.45, "forgetting_std": 0.005,
                "n_forgetting": 4, "reproducible": True,
            },
        },
    }


def test_invalid_inputs_return_zero():
    assert ol.reliability_reward({}, "x") == (0.0, "")
    assert ol.reliability_reward(_priors(), "") == (0.0, "")
    assert ol.reliability_reward(_priors(), "x", method="", dataset="") == (0.0, "")


def test_reliable_reproducible_config_max_reward():
    reward, why = ol.reliability_reward(_priors(), "exp_reliable")  # method/dataset resolved
    assert reward == ol._RELIABILITY_REWARD_CAP  # 5/5 clean + reproducible -> capped
    assert "reproducible" in why and "tcl/split_cifar10" in why


def test_reward_never_exceeds_cap():
    reward, _ = ol.reliability_reward(_priors(), "x", method="tcl", dataset="split_cifar10")
    assert 0.0 < reward <= ol._RELIABILITY_REWARD_CAP


def test_flaky_config_earns_nothing():
    # 2/6 completed -> below min-completed AND below completion-rate floor
    assert ol.reliability_reward(_priors(), "exp_flaky") == (0.0, "")


def test_clean_but_not_reproducible_is_partial():
    reward, why = ol.reliability_reward(_priors(), "x", method="tcl_partial", dataset="split_cifar10")
    # 3/4 clean, not reproducible -> 0.6 * cap * 0.75, no reproducibility bonus
    assert reward == round(ol._RELIABILITY_REWARD_CAP * 0.6 * 0.75, 1)
    assert 0.0 < reward < ol._RELIABILITY_REWARD_CAP
    assert "reproducible" not in why


def test_exploration_safe_reliable_but_bad_still_rewarded():
    """The key guarantee: reward keys on RELIABILITY, not a good score. A config that
    reliably produces BAD (high-forgetting) results earns the same reward as a good one,
    so the reward cannot reinforce a single winning method and collapse exploration."""
    good, _ = ol.reliability_reward(_priors(), "x", method="tcl", dataset="split_cifar10")
    bad, _ = ol.reliability_reward(_priors(), "x", method="tcl_bad", dataset="split_cifar10")
    assert good == bad == ol._RELIABILITY_REWARD_CAP  # identical despite 0.13 vs 0.45 forgetting


def test_novel_config_is_neutral_not_penalised():
    # no track record -> 0 (never negative; exploration is not pushed down)
    assert ol.reliability_reward(_priors(), "x", method="brand_new", dataset="brand_new") == (0.0, "")


def test_reward_is_bounded_below_failure_penalty():
    # the reinforcement must never outweigh the punishment, by design
    assert ol._RELIABILITY_REWARD_CAP < ol._FAILURE_PENALTY
