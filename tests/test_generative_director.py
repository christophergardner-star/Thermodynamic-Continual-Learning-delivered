import tempfile
from pathlib import Path

from tar_lab.generative_director import GenerativeDirector
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import DirectorPolicy, GovernorMetrics, QuantitativeJustification


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_science_profiles(tmp: str) -> None:
    source_dir = REPO_ROOT / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _policy(*, failure_streak: int, pivot_required: bool) -> DirectorPolicy:
    points = [
        GovernorMetrics(
            trial_id="trial-seed",
            step=1,
            energy_e=0.01,
            entropy_sigma=0.02,
            drift_l2=0.03,
            drift_rho=0.01,
            grad_norm=0.4,
            regime_rho=1.0,
            effective_dimensionality=6.0,
            dimensionality_ratio=0.9,
        ),
        GovernorMetrics(
            trial_id="trial-seed",
            step=2,
            energy_e=0.02,
            entropy_sigma=0.03,
            drift_l2=0.04,
            drift_rho=0.02,
            grad_norm=0.5,
            regime_rho=1.0,
            effective_dimensionality=6.2,
            dimensionality_ratio=0.92,
        ),
        GovernorMetrics(
            trial_id="trial-seed",
            step=3,
            energy_e=0.03,
            entropy_sigma=0.04,
            drift_l2=0.05,
            drift_rho=0.03,
            grad_norm=0.6,
            regime_rho=1.0,
            effective_dimensionality=6.4,
            dimensionality_ratio=0.94,
        ),
    ]
    return DirectorPolicy(
        trial_id="trial-seed",
        objective_slug="thermodynamic-anchor",
        anchor_path="anchors/thermodynamic_anchor.safetensors",
        experiment_family="elastic_anchor",
        pivot_required=pivot_required,
        failure_streak=failure_streak,
        quantitative_justification=QuantitativeJustification(
            energy_e=points[-1].energy_e,
            entropy_sigma=points[-1].entropy_sigma,
            drift_rho=points[-1].drift_rho,
            grad_norm=points[-1].grad_norm,
            regime_rho=points[-1].regime_rho,
            effective_dimensionality=points[-1].effective_dimensionality,
            effective_dimensionality_std_err=points[-1].effective_dimensionality_std_err,
            equilibrium_fraction=points[-1].equilibrium_fraction,
            energy_slope=points[-1].energy_e - points[0].energy_e,
            entropy_slope=points[-1].entropy_sigma - points[0].entropy_sigma,
            drift_slope=points[-1].drift_rho - points[0].drift_rho,
            dimensionality_slope=points[-1].effective_dimensionality - points[0].effective_dimensionality,
        ),
        data_anchor=points,
    )


def test_should_propose_true_above_streak_threshold():
    director = GenerativeDirector(workspace_root=".")
    assert director.should_propose(_policy(failure_streak=5, pivot_required=True)) is True


def test_should_propose_false_below_threshold():
    director = GenerativeDirector(workspace_root=".")
    assert director.should_propose(_policy(failure_streak=3, pivot_required=True)) is False


def test_rule_heuristic_proposal_always_returns_valid_family():
    director = GenerativeDirector(workspace_root=".", operator_role=None)
    proposal = director.propose_family(_policy(failure_streak=6, pivot_required=True), "manual trigger")
    assert proposal.proposed_family.proposed_by == "rule_heuristic"
    assert proposal.proposed_family.name
    assert proposal.proposed_family.status == "pending"


def test_proposal_persisted_to_state():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            proposal = orchestrator.propose_experiment_family("thermodynamic-anchor", "manual")
            proposals = orchestrator.list_family_proposals()
            assert len(proposals) >= 1
            assert proposals[0].proposal_id is not None
            assert any(item.proposal_id == proposal.proposal_id for item in proposals)
        finally:
            orchestrator.shutdown()


def test_approve_family_proposal_adds_to_registered():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            proposal = orchestrator.propose_experiment_family("thermodynamic-anchor", "manual")
            family = orchestrator.approve_family_proposal(proposal.proposal_id)
            registered = orchestrator.list_registered_families()
            assert family.status == "approved"
            assert any(item.family_id == family.family_id for item in registered)
        finally:
            orchestrator.shutdown()


def test_reject_family_proposal_does_not_enter_registered():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            proposal = orchestrator.propose_experiment_family("thermodynamic-anchor", "manual")
            orchestrator.reject_family_proposal(proposal.proposal_id, "operator_rejected")
            registered = orchestrator.list_registered_families()
            proposals = orchestrator.list_family_proposals()
            assert all(item.family_id != proposal.proposed_family.family_id for item in registered)
            updated = next(item for item in proposals if item.proposal_id == proposal.proposal_id)
            assert updated.proposed_family.status == "rejected"
        finally:
            orchestrator.shutdown()
