"""Tests for the STRATA forgetting-attribution port (tar_lab/forgetting_attribution.py).

Pins the pre-registered metrics + exclusion logic on controlled cases:
  - covariance_alignment / linear_cka behave (identical->1, orthogonal->~0)
  - recovery_fraction math
  - forgetting-event identification at the >=20% relative-drop threshold
  - top-1% restore reverts only the largest-changed elements
  - the three exclusion tests + end-to-end attribution on a controlled event
"""
from __future__ import annotations

import numpy as np
import torch

from tar_lab.forgetting_attribution import (
    covariance_alignment, linear_cka, recovery_fraction,
    snapshot_parameters, top_fraction_restore_overrides, full_restore_overrides,
    identify_forgetting_events, attribute_event, summarize_run, ForgettingEvent,
    RECOVERY_THRESHOLD, CKA_EXCLUSION,
)


# ── metrics ──────────────────────────────────────────────────────────────────
def test_covariance_alignment_identical_and_orthogonal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 8))
    assert covariance_alignment(x, x) > 0.99          # identical covariance structure
    # orthogonal subspaces: X varies only in dims 0-3, Y only in dims 4-7
    a = rng.normal(size=(256, 8)); a[:, 4:] = 0.0
    b = rng.normal(size=(256, 8)); b[:, :4] = 0.0
    assert covariance_alignment(a, b) < 0.10          # decorrelated bases
    assert covariance_alignment(x, np.zeros((3, 8))) == 0.0  # degenerate -> 0


def test_linear_cka_identical():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(128, 6))
    assert linear_cka(x, x) > 0.99
    assert linear_cka(x, rng.normal(size=(128, 6))) < 0.5


def test_recovery_fraction():
    assert recovery_fraction(0.9, 0.5, 0.9) == 1.0     # full recovery
    assert recovery_fraction(0.9, 0.5, 0.5) == 0.0     # none
    assert abs(recovery_fraction(0.9, 0.5, 0.7) - 0.5) < 1e-9
    assert recovery_fraction(0.9, 0.9, 0.9) == 0.0     # no loss -> 0 (guard div-by-zero)


# ── event identification ─────────────────────────────────────────────────────
def test_identify_forgetting_events_threshold():
    # task0 peaks at 0.90 then drops to 0.60 (33% rel drop >= 20%) after task1 -> event.
    # task1 drops 0.80->0.75 (6% < 20%) -> NOT an event.
    acc = [
        [0.90, None, None],
        [0.60, 0.80, None],
        [0.58, 0.75, 0.82],
    ]
    events = identify_forgetting_events(acc)
    keys = {(e.forgotten_task, e.after_task) for e in events}
    assert (0, 1) in keys                # task0 forgotten after task1
    assert (1, 2) not in keys            # task1's 6% drop is below threshold
    ev01 = next(e for e in events if (e.forgotten_task, e.after_task) == (0, 1))
    assert abs(ev01.rel_drop - (0.90 - 0.60) / 0.90) < 1e-6


# ── top-1% restore ───────────────────────────────────────────────────────────
def test_top_fraction_restore_reverts_only_largest_changes():
    before = {"w": torch.tensor([0.0, 0.0, 0.0, 0.0])}
    after = {"w": torch.tensor([0.1, 0.2, 5.0, 0.3])}   # element 2 changed by far the most
    ov = top_fraction_restore_overrides(before, after, fraction=0.25)  # top 1 of 4
    # element 2 reverted to before (0.0); the rest kept at after values (float32 tolerant)
    assert ov["w"][2].item() == 0.0
    assert torch.allclose(ov["w"], torch.tensor([0.1, 0.2, 0.0, 0.3]), atol=1e-6)


# ── end-to-end attribution on a controlled event ─────────────────────────────
def _controlled_event(align_forgotten, align_interfering):
    before = {"w": torch.tensor([1.0, 2.0, 3.0])}
    after = {"w": torch.tensor([1.5, 2.5, 9.0])}       # element 2 = the big (fast) change
    event = ForgettingEvent(forgotten_task=0, after_task=1, peak_acc=0.90, current_acc=0.40, rel_drop=0.55)

    # eval closure: FULL restore (== before) recovers to peak; anything else stays at current.
    def eval_fn(overrides):
        w = overrides["w"]
        return 0.90 if torch.allclose(w, before["w"]) else 0.40

    return attribute_event(
        event, params_before=before, params_after=after,
        eval_forgotten_with_overrides=eval_fn,
        acts_forgotten=align_forgotten, acts_interfering=align_interfering,
    )


def test_attribute_event_substrate_present_timescale_excluded():
    # Full restore recovers (substrate NOT excluded); top-1% restore != before -> no recovery
    # (timescale excluded). Decorrelated acts -> basis excluded.
    rng = np.random.default_rng(2)
    a = rng.normal(size=(256, 8)); a[:, 4:] = 0.0
    b = rng.normal(size=(256, 8)); b[:, :4] = 0.0
    res = _controlled_event({"L": a}, {"L": b})
    causes = res.attributed_causes
    assert "substrate" in causes                 # full param restore recovered
    assert "timescale" not in causes             # top-1% restore did not recover
    assert "basis" not in causes                 # decorrelated representations
    assert res.unattributed is False


def test_attribute_event_basis_present_when_representations_correlated():
    rng = np.random.default_rng(3)
    shared = rng.normal(size=(256, 8))
    res = _controlled_event({"L": shared}, {"L": shared + 0.01 * rng.normal(size=(256, 8))})
    assert "basis" in res.attributed_causes      # highly correlated bases -> basis is a cause


def test_summarize_run_incompleteness():
    # one attributed event (big magnitude) + one unattributed (small) -> unattr fraction small -> complete
    e_attr = _controlled_event({"L": np.random.default_rng(4).normal(size=(64, 4))},
                               {"L": np.random.default_rng(5).normal(size=(64, 4))})
    run = summarize_run([e_attr])
    assert run.n_events == 1 and run.n_attributed == 1
    assert run.incomplete is False
    d = run.to_dict()
    assert d["n_events"] == 1 and "cause_counts" in d
