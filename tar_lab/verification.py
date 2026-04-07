from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, List, Optional

from tar_lab.schemas import (
    AblationResult,
    BreakthroughReport,
    CalibrationReport,
    ClaimAcceptancePolicy,
    ClaimVerdict,
    ContradictionReview,
    SeedRunResult,
    SeedVarianceReport,
    TrainingPayloadConfig,
    VerificationReport,
)
from tar_lab.train_template import run_payload


def _float_std(values: Iterable[float]) -> float:
    rows = list(values)
    if len(rows) < 2:
        return 0.0
    return float(pstdev(rows))


def _payload_score(summary: dict[str, Any]) -> float:
    metrics = summary.get("last_metrics") or {}
    calibration = summary.get("calibration") or {}
    training_loss = float(metrics.get("training_loss") or 0.0)
    d_ratio = float(metrics.get("dimensionality_ratio") or 0.0)
    eq = float(metrics.get("equilibrium_fraction") or 0.0)
    ece = float(calibration.get("ece") or 0.0)
    loss_term = max(0.0, 1.25 - training_loss)
    calibration_term = max(0.0, 1.0 - ece)
    return round(loss_term + (0.9 * d_ratio) + (0.7 * eq) + (0.5 * calibration_term), 6)


class VerificationRunner:
    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace).resolve()

    def run(
        self,
        config: TrainingPayloadConfig,
        *,
        seed_count: int = 3,
    ) -> VerificationReport:
        control_summary = self._run_variant(config, suffix="control", notes={})
        control_score = _payload_score(control_summary)
        seed_variance = self._run_seed_sweeps(config, seed_count=seed_count)
        calibration = CalibrationReport.model_validate(control_summary.get("calibration") or {})
        ablations = self._run_ablations(config, control_score=control_score)
        verdict = self._verdict(seed_variance, calibration, ablations)
        recommendations = self._recommendations(seed_variance, calibration, ablations, verdict)
        return VerificationReport(
            trial_id=config.trial_id,
            control_score=control_score,
            seed_variance=seed_variance,
            calibration=calibration,
            ablations=ablations,
            verdict=verdict,
            recommendations=recommendations,
        )

    def build_breakthrough_report(
        self,
        verification: VerificationReport,
        supporting_research_ids: Optional[List[str]] = None,
        claim_verdict: Optional[ClaimVerdict] = None,
    ) -> BreakthroughReport:
        control_score = verification.control_score
        stability_score = max(
            0.0,
            1.0 - min(
                1.0,
                (verification.seed_variance.loss_std * 4.0)
                + (verification.seed_variance.dimensionality_std * 0.25),
            ),
        )
        calibration_score = max(0.0, 1.0 - min(1.0, verification.calibration.ece))
        ablation_signal = max(
            [abs(result.delta_vs_control) for result in verification.ablations],
            default=0.0,
        )
        novelty_score = round(min(1.0, (0.25 * control_score) + ablation_signal), 6)

        rationale: list[str] = []
        if verification.verdict == "verified":
            rationale.append("verification suite passed seed variance and calibration gates")
        if verification.seed_variance.stable:
            rationale.append("seed variance remained within the stability budget")
        if verification.calibration.ece <= 0.15:
            rationale.append("calibration remained within the acceptable ECE envelope")
        if any(result.delta_vs_control <= -0.05 for result in verification.ablations):
            rationale.append("ablations degraded the control run, indicating a real mechanism effect")

        if verification.verdict == "verified" and control_score >= 1.8 and calibration_score >= 0.85:
            status = "breakthrough"
            summary = (
                f"Trial {verification.trial_id} is a verified breakthrough: "
                f"control_score={control_score:.3f}, stability={stability_score:.3f}, "
                f"calibration={calibration_score:.3f}."
            )
        elif verification.verdict in {"verified", "inconclusive"} and control_score >= 1.4:
            status = "candidate"
            summary = (
                f"Trial {verification.trial_id} is a candidate signal: "
                f"control_score={control_score:.3f}, but additional evidence is still needed."
            )
        else:
            status = "rejected"
            summary = (
                f"Trial {verification.trial_id} does not yet qualify as a breakthrough: "
                f"control_score={control_score:.3f}, verdict={verification.verdict}."
            )

        return BreakthroughReport(
            trial_id=verification.trial_id,
            status=status,
            summary=summary,
            novelty_score=novelty_score,
            stability_score=round(stability_score, 6),
            calibration_score=round(calibration_score, 6),
            supporting_research_ids=supporting_research_ids or [],
            rationale=rationale,
            verification=verification,
            claim_verdict=claim_verdict,
        )

    def assess_claim(
        self,
        verification: VerificationReport,
        *,
        supporting_research_ids: Optional[List[str]] = None,
        supporting_evidence_ids: Optional[List[str]] = None,
        contradiction_review: Optional[ContradictionReview] = None,
        canonical_comparable: bool = False,
        verification_report_trial_id: Optional[str] = None,
        benchmark_problem_id: Optional[str] = None,
        benchmark_execution_created_at: Optional[str] = None,
        benchmark_execution_mode: Optional[str] = None,
        supporting_benchmark_ids: Optional[List[str]] = None,
        supporting_benchmark_names: Optional[List[str]] = None,
        evidence_bundle_id: Optional[str] = None,
        canonical_comparability_source: str = "none",
        verdict_inputs_complete: bool = True,
        linkage_status: str = "none",
        linkage_note: Optional[str] = None,
        policy: Optional[ClaimAcceptancePolicy] = None,
    ) -> ClaimVerdict:
        policy = policy or ClaimAcceptancePolicy()
        support_ids = supporting_research_ids or []
        evidence_ids = supporting_evidence_ids or []
        rationale: list[str] = []

        enough_seeds = verification.seed_variance.num_runs >= policy.min_seed_runs
        stable_loss = verification.seed_variance.loss_std <= policy.max_seed_loss_std
        stable_dimensionality = (
            verification.seed_variance.dimensionality_std <= policy.max_seed_dimensionality_std
        )
        calibrated = verification.calibration.ece <= policy.max_calibration_ece
        ablation_gap = max((abs(item.delta_vs_control) for item in verification.ablations), default=0.0)
        strong_ablation = ablation_gap >= policy.min_ablation_gap
        enough_support = len(set(support_ids)) >= policy.min_supporting_sources
        contradictions = contradiction_review.contradiction_count if contradiction_review is not None else 0
        contradiction_clear = contradictions <= policy.max_allowed_contradictions
        canonical_ok = (not policy.require_canonical_benchmark) or canonical_comparable

        if linkage_status == "exact":
            rationale.append("claim verdict is bound to explicit trial-local evidence")
        elif linkage_status == "none":
            rationale.append("no benchmark execution was linked to this claim review")
        else:
            rationale.append("claim review linkage is ambiguous")
        if linkage_note:
            rationale.append(linkage_note)

        if enough_seeds:
            rationale.append(f"seed count {verification.seed_variance.num_runs} meets policy")
        else:
            rationale.append(f"seed count {verification.seed_variance.num_runs} is below policy minimum")
        if stable_loss and stable_dimensionality:
            rationale.append("seed variance is within the stability budget")
        else:
            rationale.append("seed variance remains too high for a hard claim")
        if calibrated:
            rationale.append("calibration is within the acceptable envelope")
        else:
            rationale.append("calibration exceeds the claim threshold")
        if strong_ablation:
            rationale.append("ablation gap indicates a real mechanism effect")
        else:
            rationale.append("ablation gap is too small to isolate the claimed mechanism")
        if enough_support:
            rationale.append(f"literature support count {len(set(support_ids))} meets policy")
        else:
            rationale.append("literature support is below the minimum evidence floor")
        if contradiction_clear:
            rationale.append("no unresolved contradictions remain above policy")
        else:
            rationale.append("contradictory evidence remains unresolved")
        if canonical_ok:
            rationale.append("benchmark comparability requirement is satisfied")
        elif policy.require_canonical_benchmark:
            rationale.append("canonical benchmark comparability is required but not satisfied")

        if not contradiction_clear:
            status = "contradicted"
        elif not verdict_inputs_complete or linkage_status == "ambiguous":
            status = "insufficient_evidence"
        elif not enough_seeds or not enough_support:
            status = "insufficient_evidence"
        elif stable_loss and stable_dimensionality and calibrated and strong_ablation and canonical_ok:
            status = "accepted"
        elif verification.verdict in {"verified", "inconclusive"} and calibrated:
            status = "provisional"
        else:
            status = "rejected"

        satisfied = sum(
            int(flag)
            for flag in (enough_seeds, stable_loss and stable_dimensionality, calibrated, strong_ablation, enough_support, contradiction_clear, canonical_ok)
        )
        confidence = round(min(1.0, satisfied / 7.0), 6)
        return ClaimVerdict(
            verdict_id=f"claim-{verification.trial_id}",
            trial_id=verification.trial_id,
            decision_scope="trial_local",
            status=status,  # type: ignore[arg-type]
            rationale=rationale,
            policy=policy,
            supporting_research_ids=list(dict.fromkeys(support_ids)),
            supporting_evidence_ids=list(dict.fromkeys(evidence_ids)),
            verification_report_trial_id=verification_report_trial_id or verification.trial_id,
            benchmark_problem_id=benchmark_problem_id,
            benchmark_execution_created_at=benchmark_execution_created_at,
            benchmark_execution_mode=benchmark_execution_mode,
            supporting_benchmark_ids=list(dict.fromkeys(supporting_benchmark_ids or [])),
            supporting_benchmark_names=list(dict.fromkeys(supporting_benchmark_names or [])),
            evidence_bundle_id=evidence_bundle_id,
            canonical_comparability_source=canonical_comparability_source,  # type: ignore[arg-type]
            verdict_inputs_complete=verdict_inputs_complete,
            linkage_status=linkage_status,  # type: ignore[arg-type]
            linkage_note=linkage_note,
            contradiction_review=contradiction_review,
            canonical_benchmark_required=policy.require_canonical_benchmark,
            canonical_benchmark_satisfied=canonical_comparable,
            confidence=confidence,
        )

    def _run_seed_sweeps(self, config: TrainingPayloadConfig, seed_count: int = 3) -> SeedVarianceReport:
        base_seed = int(config.notes.get("seed", 7))
        seeds: list[int] = []
        candidate = base_seed
        while len(seeds) < max(1, seed_count):
            if candidate not in seeds:
                seeds.append(candidate)
            candidate += 11

        runs: list[SeedRunResult] = []
        for seed in seeds:
            summary = self._run_variant(config, suffix=f"seed_{seed}", notes={"seed": seed})
            metrics = summary.get("last_metrics") or {}
            calibration = summary.get("calibration") or {}
            runs.append(
                SeedRunResult(
                    seed=seed,
                    training_loss=float(metrics.get("training_loss") or 0.0),
                    effective_dimensionality=float(metrics.get("effective_dimensionality") or 0.0),
                    equilibrium_fraction=float(metrics.get("equilibrium_fraction") or 0.0),
                    calibration_ece=float(calibration.get("ece") or 0.0),
                )
            )

        loss_values = [run.training_loss for run in runs]
        d_values = [run.effective_dimensionality for run in runs]
        ece_values = [run.calibration_ece for run in runs]
        stable = _float_std(loss_values) <= 0.08 and _float_std(d_values) <= 0.75
        return SeedVarianceReport(
            num_runs=len(runs),
            loss_mean=mean(loss_values) if loss_values else 0.0,
            loss_std=_float_std(loss_values),
            dimensionality_mean=mean(d_values) if d_values else 0.0,
            dimensionality_std=_float_std(d_values),
            calibration_ece_mean=mean(ece_values) if ece_values else 0.0,
            stable=stable,
            runs=runs,
        )

    def _run_ablations(self, config: TrainingPayloadConfig, control_score: float) -> list[AblationResult]:
        specs = [
            ("no_anchor_penalty", {"disable_anchor_penalty": True}),
            ("no_fim_proxy", {"disable_fim_proxy": True}),
            ("no_ou_jitter", {"disable_ou_jitter": True}),
        ]
        rows: list[AblationResult] = []
        for name, updates in specs:
            summary = self._run_variant(config, suffix=name, notes=updates)
            metrics = summary.get("last_metrics") or {}
            calibration = summary.get("calibration") or {}
            score = _payload_score(summary)
            rows.append(
                AblationResult(
                    name=name,
                    training_loss=float(metrics.get("training_loss") or 0.0),
                    effective_dimensionality=float(metrics.get("effective_dimensionality") or 0.0),
                    equilibrium_fraction=float(metrics.get("equilibrium_fraction") or 0.0),
                    calibration_ece=float(calibration.get("ece") or 0.0),
                    score=score,
                    delta_vs_control=round(score - control_score, 6),
                )
            )
        return rows

    def _run_variant(
        self,
        config: TrainingPayloadConfig,
        *,
        suffix: str,
        notes: dict[str, Any],
    ) -> dict[str, Any]:
        variant_trial_id = f"{config.trial_id}-{suffix}"
        output_dir = self.workspace / "tar_runs" / config.trial_id / "verification" / suffix
        log_path = output_dir / "thermo_metrics.jsonl"
        merged_notes = dict(config.notes)
        merged_notes.update(notes)
        anchor_path = self._host_path(config.anchor_path)
        anchor_manifest_path = self._host_path(config.anchor_manifest_path)
        research_manifest_path = self._host_path(config.research_manifest_path)
        variant = config.model_copy(
            update={
                "trial_id": variant_trial_id,
                "anchor_path": anchor_path,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "anchor_manifest_path": anchor_manifest_path,
                "research_manifest_path": research_manifest_path,
                "notes": merged_notes,
            }
        )
        config_path = output_dir / "config.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(variant.model_dump_json(indent=2), encoding="utf-8")
        return run_payload(variant, dry_run=False)

    def _host_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if path.startswith("/workspace/"):
            return str(self.workspace / Path(path.removeprefix("/workspace/")))
        if path == "/workspace":
            return str(self.workspace)
        if path.startswith("/data/anchor/"):
            return str(self.workspace / "tar_state" / "data" / "anchor" / Path(path).name)
        if path.startswith("/data/research/"):
            return str(self.workspace / "tar_state" / "data" / "research" / Path(path).name)
        return path

    @staticmethod
    def _verdict(
        seed_variance: SeedVarianceReport,
        calibration: CalibrationReport,
        ablations: List[AblationResult],
    ) -> str:
        ablation_effect = any(result.delta_vs_control <= -0.05 for result in ablations)
        if seed_variance.stable and calibration.ece <= 0.15 and ablation_effect:
            return "verified"
        if not seed_variance.stable:
            return "unstable"
        return "inconclusive"

    @staticmethod
    def _recommendations(
        seed_variance: SeedVarianceReport,
        calibration: CalibrationReport,
        ablations: List[AblationResult],
        verdict: str,
    ) -> list[str]:
        notes: list[str] = []
        if not seed_variance.stable:
            notes.append("increase seed count or tighten the hyperparameter neighborhood before claiming a result")
        if calibration.ece > 0.15:
            notes.append("improve calibration before promoting this run to a claimed breakthrough")
        if not any(result.delta_vs_control <= -0.05 for result in ablations):
            notes.append("run stronger ablations because the current control-vs-ablation gap is too small")
        if verdict == "verified":
            notes.append("candidate is ready for promotion into the breakthrough tracker")
        return notes
