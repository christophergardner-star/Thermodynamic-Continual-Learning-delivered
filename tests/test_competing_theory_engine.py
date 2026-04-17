from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    CompetingTheory,
    HeadToHeadExperimentPlan,
    TheoryInvalidationRecord,
)


class _VaultStub:
    def __init__(self):
        self.records: list[tuple[str, str, dict[str, object]]] = []

    def _upsert(self, document_id: str, document: str, metadata: dict[str, object]) -> None:
        self.records.append((document_id, document, metadata))


def test_competing_theory_schema_valid():
    theory = CompetingTheory(
        theory_id="theory-1",
        timestamp="2026-04-17T19:00:00",
        trial_id="trial-abc",
        project_id="proj-xyz",
        description="Competing explanation",
        predicted_outcome="Different outcome",
        confidence=0.4,
        source="heuristic",
    )
    assert theory.status == "open"
    assert theory.invalidated_by_trial_id == ""
    assert theory.vault_indexed is False


def test_competing_theory_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CompetingTheory(
            theory_id="theory-1",
            timestamp="2026-04-17T19:00:00",
            trial_id="trial-abc",
            project_id="proj-xyz",
            description="Competing explanation",
            predicted_outcome="Different outcome",
            confidence=0.4,
            source="heuristic",
            unknown="x",
        )


def test_head_to_head_plan_schema_valid():
    plan = HeadToHeadExperimentPlan(
        plan_id="plan-1",
        timestamp="2026-04-17T19:00:00",
        trial_id="trial-abc",
        project_id="proj-xyz",
        primary_theory_description="Primary theory",
        competing_theory_id="theory-1",
        discriminating_variable="random_seed",
        expected_primary_outcome="Primary outcome",
        expected_competing_outcome="Competing outcome",
    )
    assert plan.status == "proposed"


def test_theory_invalidation_record_schema_valid():
    record = TheoryInvalidationRecord(
        invalidation_id="inv-1",
        timestamp="2026-04-17T19:00:00",
        theory_id="theory-1",
        trial_id="trial-abc",
        project_id="proj-xyz",
        evidence_summary="Repeated runs invalidated the effect",
        confidence=0.8,
    )
    assert isinstance(record.confidence, float)


def test_generate_competing_theories_returns_two_heuristics(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub()
    result = orch.generate_competing_theories("trial_abc", "proj_xyz", "anomalous result")
    assert len(result) == 2
    assert all(item.source == "heuristic" for item in result)
    assert all(item.status == "open" for item in result)
    assert all(item.trial_id == "trial_abc" for item in result)


def test_generate_competing_theories_vault_indexed(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub()
    result = orch.generate_competing_theories("trial_abc", "proj_xyz", "anomalous result")
    assert all(item.vault_indexed is True for item in result)


def test_build_head_to_head_plan_initialization_theory(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub()
    theory = CompetingTheory(
        theory_id="theory-1",
        timestamp="2026-04-17T19:00:00",
        trial_id="trial-abc",
        project_id="proj-xyz",
        description="Result is an artifact of initialization variance rather than a genuine effect",
        predicted_outcome="Repeated trials with different seeds will show regression to baseline",
        confidence=0.4,
        source="heuristic",
    )
    plan = orch.build_head_to_head_plan(
        "trial-abc",
        "proj-xyz",
        "Primary theory outcome",
        theory,
    )
    assert plan.discriminating_variable == "random_seed"
    assert plan.status == "proposed"


def test_invalidate_theory_updates_status(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    theories_dir = Path(tmp_path) / "tar_state" / "theories"
    theories_dir.mkdir(parents=True, exist_ok=True)
    theory = CompetingTheory(
        theory_id="theory-1",
        timestamp="2026-04-17T19:00:00",
        trial_id="trial-abc",
        project_id="proj-xyz",
        description="Competing explanation",
        predicted_outcome="Different outcome",
        confidence=0.4,
        source="heuristic",
    )
    (theories_dir / f"{theory.theory_id}.json").write_text(
        theory.model_dump_json(indent=2),
        encoding="utf-8",
    )
    orch.invalidate_theory(
        theory.theory_id,
        "trial-invalidating",
        "proj-xyz",
        "Evidence contradicts the competing theory",
        0.8,
    )
    reloaded = CompetingTheory.model_validate_json(
        (theories_dir / f"{theory.theory_id}.json").read_text(encoding="utf-8")
    )
    assert reloaded.status == "invalidated"
    assert reloaded.invalidated_by_trial_id
