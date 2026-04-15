import tempfile
from pathlib import Path
from types import SimpleNamespace

from tar_lab.hierarchy import TriModelHierarchy, RuleDirector
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    CheckpointRecord,
    EndpointRecord,
    GovernorMetrics,
    LiteraturePolicySignal,
    LocalLLMConfig,
    MemorySearchHit,
    RoleAssignment,
)
from tar_lab.state import TARStateStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_science_profiles(tmp: str) -> None:
    source_dir = REPO_ROOT / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _update_initial_project(
    orchestrator: TAROrchestrator,
    project_id: str,
    *,
    project_updates=None,
    thread_updates=None,
    question_updates=None,
    action_updates=None,
):
    project = orchestrator.store.get_research_project(project_id)
    assert project is not None
    thread = project.hypothesis_threads[0].model_copy(update=thread_updates or {})
    question = project.open_questions[0].model_copy(update=question_updates or {})
    action = project.planned_actions[0].model_copy(update=action_updates or {})
    updated = project.model_copy(
        update={
            **(project_updates or {}),
            "hypothesis_threads": [thread],
            "open_questions": [question],
            "planned_actions": [action],
            "active_thread_id": thread.thread_id,
        }
    )
    orchestrator.store.upsert_research_project(updated)
    return updated


class _FallbackOnlyVault:
    def __init__(self):
        self.study_modes: list[str] = []

    def search(self, query: str, n_results: int = 5, kind=None, require_research_grade: bool = False):
        if require_research_grade:
            raise RuntimeError("semantic retrieval unavailable")
        return [
            MemorySearchHit(
                document_id="paper_claim:1",
                score=0.73,
                document="Bounded drift improves accuracy with stable anchors.",
                metadata={"kind": "paper_claim", "paper_id": "paper-1", "polarity": "positive"},
            )
        ]

    def index_problem_study(self, report) -> None:
        self.study_modes.append(report.retrieval_mode)

    def close(self) -> None:
        return None


def test_ws32_retrieval_mode_records_lexical_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            if orchestrator.vault is not None:
                orchestrator.vault.close()
            fallback_vault = _FallbackOnlyVault()
            orchestrator.vault = fallback_vault  # type: ignore[assignment]
            orchestrator.memory_indexer = None

            report = orchestrator.study_problem(
                "Investigate calibration drift in transformer adapters",
                build_env=False,
                max_results=0,
            )

            assert report.retrieval_mode == "lexical_fallback"
            assert fallback_vault.study_modes == ["lexical_fallback"]
            persisted = orchestrator.store.latest_problem_study(report.problem_id)
            assert persisted is not None
            assert persisted.retrieval_mode == "lexical_fallback"
        finally:
            orchestrator.shutdown()


def test_ws32_5_literature_signal_shifts_rule_director_family():
    with tempfile.TemporaryDirectory() as tmp:
        store = TARStateStore(tmp)
        for idx in range(1, 4):
            store.append_metric(
                GovernorMetrics(
                    trial_id="seed",
                    step=idx,
                    energy_e=0.01 * idx,
                    entropy_sigma=0.02 * idx,
                    drift_l2=0.03 * idx,
                    drift_rho=0.02 * idx,
                    grad_norm=0.1 * idx,
                )
            )
        signal = LiteraturePolicySignal(
            objective_slug="thermodynamic-anchor",
            evidence_count=4,
            recommended_family="ou_drift_jitter",
            dominant_polarity="mixed",
            contradiction_pressure=0.65,
            confidence=0.72,
            topic_terms=["entropy", "drift", "instability"],
            cited_document_ids=["paper-1", "paper-2"],
            rationale=["retrieved evidence emphasizes instability and contradiction pressure"],
        )

        policy = RuleDirector().propose(
            store,
            trial_id="trial-1",
            objective_slug="thermodynamic-anchor",
            literature_signal=signal,
        )

        assert policy.experiment_family == "ou_drift_jitter"
        assert policy.literature_signal is not None
        assert policy.literature_signal.recommended_family == "ou_drift_jitter"


def test_ws32_5_serving_state_activates_live_hierarchy_without_env():
    class FakeClient:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            content = self.outputs.pop(0)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    clients = {
        "director-ckpt": FakeClient(['{"experiment_family":"elastic_anchor","anchor_path":"anchors/live.pt","pivot_required":false,"objective_slug":"anchor"}']),
        "strategist-ckpt": FakeClient(['{"strategy_family":"elastic_anchor","fim_lambda":1.2,"bregman_budget":0.4,"drift_budget":0.05,"protected_layers":["a"],"mutable_layers":["b"],"hyperparameters":{"alpha":0.07,"eta":0.01,"steps":9,"batch_size":4}}']),
        "scout-ckpt": FakeClient(['{"training_entrypoint":"tar_lab/train_template.py","image":"pytorch/pytorch:latest","steps":9,"batch_size":4,"power_limit_w":300,"gpu_target_temp_c":70}']),
    }

    def client_factory(config: LocalLLMConfig):
        return clients[config.model]

    with tempfile.TemporaryDirectory() as tmp:
        store = TARStateStore(tmp)
        for idx, rho in enumerate((0.01, 0.02, 0.03), start=1):
            store.append_metric(
                GovernorMetrics(
                    trial_id="seed",
                    step=idx,
                    energy_e=0.01 * idx,
                    entropy_sigma=0.02 * idx,
                    drift_l2=0.03 * idx,
                    drift_rho=rho,
                    grad_norm=0.1 * idx,
                )
            )
        for role, checkpoint_name in (
            ("director", "director-ckpt"),
            ("strategist", "strategist-ckpt"),
            ("scout", "scout-ckpt"),
        ):
            store.upsert_checkpoint(
                CheckpointRecord(
                    name=checkpoint_name,
                    model_path=f"training_artifacts/{checkpoint_name}",
                    backend="transformers",
                    role=role,
                    checkpoint_kind="adapter",
                    adapter_path=f"training_artifacts/{checkpoint_name}/final_adapter",
                )
            )
            store.upsert_endpoint(
                EndpointRecord(
                    endpoint_name=f"{role}-endpoint",
                    checkpoint_name=checkpoint_name,
                    role=role,
                    host="127.0.0.1",
                    port=8100 + len(role),
                    backend="transformers",
                    base_url=f"http://127.0.0.1:{8100 + len(role)}/v1",
                    status="running",
                )
            )
            store.upsert_role_assignment(
                RoleAssignment(
                    role=role,
                    checkpoint_name=checkpoint_name,
                    endpoint_name=f"{role}-endpoint",
                )
            )

        hierarchy = TriModelHierarchy(
            workspace=tmp,
            client_factory=client_factory,
            allow_rule_fallback=False,
        )

        assert hierarchy.live_enabled is True
        policy, plan, task = hierarchy.produce_bundle(store, "trial-live", tmp, dry_run=True)

        assert policy.experiment_family == "elastic_anchor"
        assert plan.hyperparameters["steps"] == 9
        assert task.runtime.image == "pytorch/pytorch:latest"


def test_ws32_5_evidence_debt_hard_gate_blocks_non_remediation_actions():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            project = orchestrator.create_project("Investigate fragile promoted signal")
            _update_initial_project(
                orchestrator,
                project.project_id,
                thread_updates={"confidence_state": "provisional", "status": "supported"},
                action_updates={
                    "action_kind": "run_problem_study",
                    "description": "Run another broad continuation study before repairing the evidence gap.",
                    "estimated_cost": 0.4,
                    "expected_evidence_gain": 0.7,
                },
            )

            ranked = orchestrator.rank_actions(project_id=project.project_id, include_blocked=True, limit=5)
            candidate = ranked["candidates"][0]
            decision = orchestrator.allocate_budget(
                project_id=project.project_id,
                include_blocked=True,
                schedule_selected=True,
            )
            portfolio = orchestrator.portfolio_decide(limit=5)

            assert candidate["blocked"] is True
            assert any("evidence debt blocks non-remediation scheduling" in item for item in candidate["rationale"])
            assert decision["schedule_created"] is False
            assert decision["selected_candidate"]["blocked"] is True
            assert portfolio["decision"]["selected_project_id"] != project.project_id
            assert project.project_id in portfolio["decision"]["deferred_project_ids"]
        finally:
            orchestrator.shutdown()
