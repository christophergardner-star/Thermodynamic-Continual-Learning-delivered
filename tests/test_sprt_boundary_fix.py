"""Regression test for the SPRT stop-rule fix (tar_lab.stat_utils.sprt_boundary).

The bug: sprt_boundary used the BOUNDARY value log((1-beta)/alpha) as the per-seed
increment, so a single favorable seed reached boundary A and a near-even sign split
spuriously accepted H1. That stopped the HPC replication at n=12 (7 favorable vs 5)
with log_LR=8.98 while the effect was honestly inconclusive (d=-0.44, power=0.41).

The fix: each seed contributes the true Bernoulli sign-test log-likelihood ratio,
log(p1/0.5) favorable / log((1-p1)/0.5) unfavorable, with p1 = Phi(h1_effect_size).
Pure stats — torch-free.
"""
import math

from tar_lab.stat_utils import sprt_boundary


def _pp(signs):
    """Favorable seed (delta<0) -> p-proxy below alpha; unfavorable -> above."""
    return [0.025 if s < 0 else 0.975 for s in signs]


def _signs(n_fav, n_unfav):
    return [-1.0] * n_fav + [1.0] * n_unfav


def test_single_favorable_does_not_reach_boundary():
    # THE core bug: one favorable seed must not reach boundary A.
    r = sprt_boundary(n_seeds_run=1, p_values_so_far=_pp(_signs(1, 0)))
    assert r.decision == "continue"
    assert r.log_lr < r.A
    # increment equals the Bernoulli LLR log(Phi(0.5)/0.5) ~ 0.323, NOT the boundary
    phi = 0.5 * (1.0 + math.erf(0.5 / math.sqrt(2.0)))
    assert abs(r.log_lr - math.log(phi / 0.5)) < 1e-9
    assert r.log_lr < r.A / 2  # nowhere near the 2.89 boundary


def test_hpc_near_even_split_continues():
    # The exact regression: 7 favorable / 5 unfavorable must NOT accept H1.
    r = sprt_boundary(n_seeds_run=12, p_values_so_far=_pp(_signs(7, 5)))
    assert r.decision == "continue"
    assert r.B < r.log_lr < r.A
    # the n=8 prefix (5 favorable / 3 unfavorable) also continues
    r8 = sprt_boundary(n_seeds_run=8, p_values_so_far=_pp(_signs(5, 3)))
    assert r8.decision == "continue"


def test_strong_evidence_still_decides_early():
    # genuine strong favorable evidence still accepts H1 (early-stop efficiency kept)
    assert sprt_boundary(n_seeds_run=20, p_values_so_far=_pp(_signs(18, 2))).decision == "accept_H1"
    # strong unfavorable evidence accepts H0
    assert sprt_boundary(n_seeds_run=20, p_values_so_far=_pp(_signs(2, 18))).decision == "accept_H0"


def test_boundaries_unchanged():
    r = sprt_boundary(n_seeds_run=8, p_values_so_far=_pp(_signs(4, 4)), alpha=0.05, beta=0.10)
    assert abs(r.A - math.log(0.90 / 0.05)) < 1e-12
    assert abs(r.B - math.log(0.10 / 0.95)) < 1e-12


def test_h1_effect_size_scales_the_increment():
    # a larger assumed meaningful effect -> larger per-seed favorable increment
    small = sprt_boundary(n_seeds_run=1, p_values_so_far=[0.025], h1_effect_size=0.2)
    large = sprt_boundary(n_seeds_run=1, p_values_so_far=[0.025], h1_effect_size=0.8)
    assert large.log_lr > small.log_lr > 0
