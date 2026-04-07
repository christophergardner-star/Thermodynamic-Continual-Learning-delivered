from tar_lab.schemas import (
    AblationResult,
    CalibrationBin,
    CalibrationReport,
    ClaimAcceptancePolicy,
    ContradictionReview,
    SeedRunResult,
    SeedVarianceReport,
    VerificationReport,
)
from tar_lab.verification import VerificationRunner


def _verification_report() -> VerificationReport:
    return VerificationReport(
        trial_id="trial-1",
        control_score=2.1,
        seed_variance=SeedVarianceReport(
            num_runs=3,
            loss_mean=0.5,
            loss_std=0.02,
            dimensionality_mean=3.2,
            dimensionality_std=0.1,
            calibration_ece_mean=0.04,
            stable=True,
            runs=[
                SeedRunResult(seed=7, training_loss=0.5, effective_dimensionality=3.1, equilibrium_fraction=0.7, calibration_ece=0.03),
                SeedRunResult(seed=18, training_loss=0.52, effective_dimensionality=3.3, equilibrium_fraction=0.69, calibration_ece=0.04),
                SeedRunResult(seed=29, training_loss=0.48, effective_dimensionality=3.2, equilibrium_fraction=0.71, calibration_ece=0.05),
            ],
        ),
        calibration=CalibrationReport(
            ece=0.05,
            accuracy=0.8,
            mean_confidence=0.78,
            bins=[CalibrationBin(lower=0.0, upper=1.0, count=10, mean_confidence=0.78, accuracy=0.8)],
        ),
        ablations=[
            AblationResult(
                name="no_anchor_penalty",
                training_loss=0.7,
                effective_dimensionality=2.8,
                equilibrium_fraction=0.5,
                calibration_ece=0.09,
                score=1.9,
                delta_vs_control=-0.2,
            )
        ],
        verdict="verified",
        recommendations=[],
    )


def test_claim_acceptance_returns_accepted_for_strong_evidence():
    runner = VerificationRunner(".")
    verdict = runner.assess_claim(
        _verification_report(),
        supporting_research_ids=["research:a", "research:b"],
        policy=ClaimAcceptancePolicy(),
        canonical_comparable=False,
    )
    assert verdict.status == "accepted"


def test_claim_acceptance_returns_contradicted_when_conflicts_exist():
    runner = VerificationRunner(".")
    contradiction = ContradictionReview(
        review_id="review-1",
        query="test",
        conflicting_document_ids=["paper:a", "paper:b"],
        conflicting_claim_ids=["claim:a", "claim:b"],
        contradiction_count=2,
        summary="paper a conflicts with paper b",
        recommended_resolution="run discrimination experiment",
        severity="high",
    )
    verdict = runner.assess_claim(
        _verification_report(),
        supporting_research_ids=["research:a", "research:b"],
        contradiction_review=contradiction,
        policy=ClaimAcceptancePolicy(max_allowed_contradictions=0),
    )
    assert verdict.status == "contradicted"
