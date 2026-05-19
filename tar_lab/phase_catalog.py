"""
Shared phase catalog for canonical TAR phase results.

This module provides one explicit mapping from phase result artifacts to the
frontier problem, paper target, and primary research domain they belong to.
It exists to enforce the post-audit rail that TAR must not mix unrelated
domains or let one paper silently absorb every phase JSON found on disk.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseCatalogEntry:
    logical_name: str
    legacy_filename: str
    phase_number: int
    experiment_id: str
    title: str
    dataset: str
    frontier_problem_id: str
    target_paper_id: str
    target_paper_title: str
    primary_domain_id: str
    research_goal: str


_PHASE_CATALOG: tuple[PhaseCatalogEntry, ...] = (
    PhaseCatalogEntry(
        logical_name="phase10_baseline",
        legacy_filename="phase10_baseline.json",
        phase_number=10,
        experiment_id="phase10_baseline",
        title="Phase 10 - Four-Way Baseline Comparison",
        dataset="split_cifar10",
        frontier_problem_id="fp-catastrophic-forgetting",
        target_paper_id="tcl-autonomous-mechanism-paper",
        target_paper_title="Thermodynamic Continual Learning: Activation Entropy Governs Catastrophic Forgetting",
        primary_domain_id="continual_learning",
        research_goal="Primary catastrophic-forgetting benchmark against EWC, SI, and SGD.",
    ),
    PhaseCatalogEntry(
        logical_name="phase11_ablation",
        legacy_filename="phase11_ablation.json",
        phase_number=11,
        experiment_id="phase11_ablation",
        title="Phase 11 - TCL Component Ablation",
        dataset="split_cifar10",
        frontier_problem_id="fp-regime-detection-accuracy",
        target_paper_id="tcl-autonomous-mechanism-paper",
        target_paper_title="Thermodynamic Continual Learning: Activation Entropy Governs Catastrophic Forgetting",
        primary_domain_id="thermodynamics_ml",
        research_goal="Mechanistic ablation of governor versus anchor penalty.",
    ),
    PhaseCatalogEntry(
        logical_name="phase12_ewc_sweep",
        legacy_filename="phase12_ewc_sweep.json",
        phase_number=12,
        experiment_id="phase12_ewc_sweep",
        title="Phase 12 - EWC Lambda Sweep",
        dataset="split_cifar10",
        frontier_problem_id="fp-hyperparameter-robustness",
        target_paper_id="tcl-autonomous-mechanism-paper",
        target_paper_title="Thermodynamic Continual Learning: Activation Entropy Governs Catastrophic Forgetting",
        primary_domain_id="continual_learning",
        research_goal="Reviewer-defense sweep of EWC lambda sensitivity against the controlled Phase 10 reference.",
    ),
    PhaseCatalogEntry(
        logical_name="phase13_si_sweep",
        legacy_filename="phase13_si_sweep.json",
        phase_number=13,
        experiment_id="phase13_si_sweep",
        title="Phase 13 - SI Robustness Sweep",
        dataset="split_cifar10",
        frontier_problem_id="fp-hyperparameter-robustness",
        target_paper_id="tcl-autonomous-mechanism-paper",
        target_paper_title="Thermodynamic Continual Learning: Activation Entropy Governs Catastrophic Forgetting",
        primary_domain_id="continual_learning",
        research_goal="Scope the SI collapse claim to the default setting versus nearby hyperparameters.",
    ),
    PhaseCatalogEntry(
        logical_name="phase15_class_incremental_search",
        legacy_filename="phase15_class_incremental_search.json",
        phase_number=15,
        experiment_id="phase15_class_incremental_search",
        title="Phase 15 - Class-Incremental Search",
        dataset="split_cifar10",
        frontier_problem_id="fp-class-incremental",
        target_paper_id="frontier-paper-fp-class-incremental",
        target_paper_title="Continuous Class Expansion Without Task Labels",
        primary_domain_id="continual_learning",
        research_goal="Evaluate whether TCL survives the shift from task-ID-assisted to class-incremental learning.",
    ),
    PhaseCatalogEntry(
        logical_name="phase16_scale_up",
        legacy_filename="phase16_scale_up.json",
        phase_number=16,
        experiment_id="phase16_scale_up",
        title="Phase 16 - CIFAR-100 Scale-Up",
        dataset="split_cifar100",
        frontier_problem_id="fp-scale-up",
        target_paper_id="main_tcl_scaleup_paper",
        target_paper_title="Scalable Continual Adaptation on Realistic Visual Streams",
        primary_domain_id="continual_learning",
        research_goal="Scale the TCL comparison to Split-CIFAR-100 under the new rails.",
    ),
    PhaseCatalogEntry(
        logical_name="phase17_tinyimagenet",
        legacy_filename="phase17_tinyimagenet.json",
        phase_number=17,
        experiment_id="phase17_tinyimagenet",
        title="Phase 17 - TinyImageNet Scale-Up",
        dataset="split_tinyimagenet",
        frontier_problem_id="fp-scale-up",
        target_paper_id="main_tcl_scaleup_paper",
        target_paper_title="Scalable Continual Adaptation on Realistic Visual Streams",
        primary_domain_id="continual_learning",
        research_goal="Stress-test TCL on the hardest current visual continual-learning benchmark in the local TAR stack.",
    ),
)


def iter_phase_catalog_entries() -> tuple[PhaseCatalogEntry, ...]:
    return _PHASE_CATALOG


def phase_catalog_by_logical_name() -> dict[str, PhaseCatalogEntry]:
    return {entry.logical_name: entry for entry in _PHASE_CATALOG}


def phase_catalog_by_number() -> dict[int, PhaseCatalogEntry]:
    return {entry.phase_number: entry for entry in _PHASE_CATALOG}
