"""Workstream B4 — make recall bite.

Recall (prior_trials) was advisory-only: surfaced on directives but it never changed a
priority score. _recall_repeat_guard() turns a CLEAR memory match (same method x dataset as
a recalled prior trial) into an actionable signal so the loop perturbs instead of blindly
re-running an identical experiment. The guard is deterministic (string containment, no vector
store), so it is unit-testable; the director applies a bounded deprioritisation + attaches a
perturbation suggestion, OFF unless tar_state/recall_bite.enabled exists.
"""
from tar_research_director import _recall_repeat_guard, _RECALL_REPEAT_MIN_SCORE


def _trial(score, summary, doc_id="doc-1"):
    return {"document_id": doc_id, "score": score, "summary": summary}


def test_no_match_on_empty_inputs():
    assert _recall_repeat_guard("tcl", "split_cifar10", []) is None
    assert _recall_repeat_guard("", "split_cifar10", [_trial(0.9, "tcl split_cifar10")]) is None
    assert _recall_repeat_guard("tcl", "", [_trial(0.9, "tcl split_cifar10")]) is None


def test_match_when_method_and_dataset_present_above_threshold():
    trials = [_trial(0.82, "Result: tcl on split_cifar10 — mean forgetting 0.13", "doc-A")]
    guard = _recall_repeat_guard("tcl", "split_cifar10", trials)
    assert guard is not None
    assert guard["repeat_detected"] is True
    assert guard["matched_prior"] == "doc-A"
    assert guard["similarity"] == 0.82
    assert "perturb" in guard["suggestion"]


def test_no_match_below_score_threshold():
    low = _RECALL_REPEAT_MIN_SCORE - 0.05
    assert _recall_repeat_guard("tcl", "split_cifar10", [_trial(low, "tcl split_cifar10")]) is None


def test_short_method_token_no_substring_false_positive():
    # "si" must not match inside "using"/"consistent"; summary lacks the token "si"
    trials = [_trial(0.9, "a consistent study using split_cifar10, no method token here")]
    assert _recall_repeat_guard("si", "split_cifar10", trials) is None
    # but a real "si" token DOES match
    trials2 = [_trial(0.9, "si on split_cifar10 best forgetting 0.047")]
    assert _recall_repeat_guard("si", "split_cifar10", trials2) is not None


def test_dataset_normalisation_matches_hyphen_or_underscore():
    # dataset stored as split_cifar10; summary writes it as "Split-CIFAR-10"
    trials = [_trial(0.7, "TCL on Split-CIFAR-10 protocol")]
    guard = _recall_repeat_guard("tcl", "split_cifar10", trials)
    assert guard is not None and guard["matched_prior"] == "doc-1"


def test_picks_highest_scoring_match():
    trials = [
        _trial(0.60, "tcl split_cifar10 run one", "low"),
        _trial(0.91, "tcl split_cifar10 run two", "high"),
        _trial(0.75, "tcl split_cifar10 run three", "mid"),
    ]
    guard = _recall_repeat_guard("tcl", "split_cifar10", trials)
    assert guard["matched_prior"] == "high" and guard["similarity"] == 0.91


def test_no_match_when_dataset_absent_from_summary():
    # method present, but a DIFFERENT dataset -> not a repeat of this config
    trials = [_trial(0.9, "tcl on permuted_mnist, different benchmark")]
    assert _recall_repeat_guard("tcl", "split_cifar10", trials) is None
