import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

import tar_cli
from tar_lab.control import handle_request
from tar_lab.data_manager import DataManager
from tar_lab.docker_runner import DockerRunner, LaunchResult
from tar_lab.governor import ThermodynamicGovernor
from tar_lab.hierarchy import Director, DirectorDraft, LocalOpenAIRole, TriModelHierarchy
from tar_lab.memory import MemoryIndexer, VectorVault
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    ControlRequest,
    GovernorMetrics,
    GovernorThresholds,
    KnowledgeGraphEntry,
    LocalLLMConfig,
    RecoveryState,
    ResearchDocument,
    ResearchIngestReport,
    RuntimeSpec,
    TrainingPayloadConfig,
)
from tar_lab.state import TARStateStore
from tar_lab.thermoobserver import compute_participation_ratio
from tar_lab.voice import SpeechProcessor


def test_director_requires_three_points():
    with tempfile.TemporaryDirectory() as tmp:
        store = TARStateStore(tmp)
        store.append_metric(
            GovernorMetrics(
                trial_id="t0",
                step=1,
                energy_e=0.1,
                entropy_sigma=0.1,
                drift_l2=0.1,
                drift_rho=0.1,
                grad_norm=0.1,
            )
        )
        with pytest.raises(ValueError, match="three recent log points"):
            Director().propose(store, trial_id="t1", objective_slug="anchor")


def test_state_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = TARStateStore(tmp)
        recovery = RecoveryState(trial_id="trial-1", status="running", consecutive_fail_fast=2)
        store.save_recovery(recovery)
        loaded = store.load_recovery()
        assert loaded.trial_id == "trial-1"
        assert loaded.consecutive_fail_fast == 2


def test_governor_terminates_on_drift_limit():
    governor = ThermodynamicGovernor()
    anchor = {"w": torch.zeros(4)}
    current = {"w": torch.ones(4)}
    metrics = governor.compute_metrics(
        trial_id="t1",
        step=1,
        anchor=anchor,
        current=current,
        gradients={"w": torch.ones(4)},
    )
    decision = governor.evaluate(metrics, GovernorThresholds(max_drift_l2=0.5))
    assert decision.action == "terminate"
    assert "weight_drift_limit" in decision.reasons


def test_compute_participation_ratio_detects_rank_collapse():
    full_rank = torch.eye(4)
    collapsed = torch.ones(4, 4)
    assert compute_participation_ratio(full_rank) > 1.0
    assert compute_participation_ratio(collapsed) == pytest.approx(0.0)


def test_docker_command_carries_resource_caps():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.seed_mock_metrics()
            _, _, task = orchestrator.plan_trial(dry_run=True)
            command = DockerRunner(docker_bin="docker").compose_command(task)
            assert "--memory" in command
            assert "40g" in command
            assert "--cpus" in command
            assert "12" in command
            assert "--gpus" in command
            assert "device=0" in command
        finally:
            orchestrator.shutdown()


def test_docker_runner_uses_env_override(monkeypatch):
    monkeypatch.setenv("TAR_DOCKER_BIN", r"C:\custom\docker.exe")
    runner = DockerRunner()
    assert runner.docker_bin == r"C:\custom\docker.exe"


def test_docker_runner_clamps_runtime_to_engine_limits(monkeypatch):
    runner = DockerRunner(docker_bin="docker")
    monkeypatch.setattr(
        runner,
        "detect_engine_limits",
        lambda: SimpleNamespace(memory_limit_gb=6, cpu_limit=8),
    )
    runtime = runner.normalize_runtime(RuntimeSpec(memory_limit_gb=40, cpu_limit=12, gpu_index=0))
    assert runtime.memory_limit_gb == 6
    assert runtime.cpu_limit == 8


def test_docker_runner_accepts_gpu_zero_override(monkeypatch):
    runner = DockerRunner(docker_bin="docker")
    monkeypatch.setenv("TAR_GPU_INDEX", "0")
    runtime = runner.normalize_runtime(RuntimeSpec(memory_limit_gb=40, cpu_limit=12, gpu_index=3))
    assert runtime.gpu_index == 0


def test_docker_runner_bootstraps_payload_dependency():
    runner = DockerRunner(docker_bin="docker")
    command = runner.prepare_container_command(
        [
            "python",
            "-m",
            "tar_lab.train_template",
            "--config",
            "/workspace/tar_runs/t1/config.json",
            "--dry_run",
        ]
    )
    assert command[:2] == ["sh", "-lc"]
    assert "pip install --quiet pydantic" in command[2]
    assert "python -m tar_lab.train_template" in command[2]


def test_panic_kill_dry_run_uses_placeholder():
    commands = DockerRunner(docker_bin="docker").panic_kill(dry_run=True)
    assert commands[0][:4] == ["docker", "ps", "-q", "--filter"]
    assert commands[1] == ["docker", "kill", "<container_ids>"]


def test_orchestrator_dry_run_writes_state():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            report = orchestrator.run_dry_run()
            assert report.json_schema_ok
            assert report.docker_command_ok
            recovery = orchestrator.store.load_recovery()
            assert recovery.status == "completed"
            assert recovery.max_effective_dimensionality_achieved > 0.0
            assert (Path(tmp) / "tar_state" / "recovery.json").exists()
            assert (Path(tmp) / "logs" / "activity_audit.log").exists()
        finally:
            orchestrator.shutdown()


def test_plan_trial_writes_payload_config():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _, _, task = orchestrator.plan_trial(dry_run=True)
            assert task.payload_config_path is not None
            payload = TrainingPayloadConfig.model_validate_json(Path(task.payload_config_path).read_text(encoding="utf-8"))
            assert payload.trial_id == task.trial_id
            assert payload.log_path.endswith("logs\\thermo_metrics.jsonl") or payload.log_path.endswith("logs/thermo_metrics.jsonl")
            assert payload.anchor_manifest_path == "/data/anchor/manifest.json"
            assert payload.research_manifest_path == "/data/research/manifest.json"
        finally:
            orchestrator.shutdown()


def test_pivot_force_changes_strategy_after_three_failures():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            for _ in range(3):
                orchestrator.run_dry_run(force_fail_fast=True)
            previous_family = orchestrator.store.load_recovery().last_strategy_family
            policy, _, _ = orchestrator.plan_trial(dry_run=True)
            assert policy.pivot_required
            assert policy.experiment_family != previous_family
        finally:
            orchestrator.shutdown()


def test_control_request_dry_run():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            response = handle_request(
                orchestrator,
                ControlRequest(command="dry_run", payload={"force_fail_fast": False}),
            )
            assert response.ok
            assert response.payload["json_schema_ok"] is True
        finally:
            orchestrator.shutdown()


def test_check_regime_reports_quenching():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.run_dry_run(force_fail_fast=True)
            regime = orchestrator.check_regime()
            assert regime["regime"] == "thermodynamic_quenching"
            assert "Degeneracy Warning" in regime["warning"]
        finally:
            orchestrator.shutdown()


def test_research_ingest_indexes_documents():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.research_ingestor.ingest = lambda topic, max_results=6: ResearchIngestReport(  # type: ignore[assignment]
                topic=topic,
                fetched=1,
                indexed=1,
                sources=["arxiv"],
                documents=[
                    ResearchDocument(
                        document_id="arxiv:test-paper",
                        source_kind="arxiv",
                        source_name="arXiv cs.AI",
                        title="Test Research Paper",
                        summary="Calibration drift is a core current AI problem.",
                        url="https://example.com/paper",
                        tags=["calibration", "robustness"],
                        problem_statements=["Calibration drift is a core current AI problem."],
                    )
                ],
            )
            report = orchestrator.ingest_research(topic="current ai problems", max_results=1)
            assert report.indexed == 1
            docs = list(orchestrator.store.iter_research_documents())
            assert len(docs) == 1
            assert orchestrator.vault is not None
            hits = orchestrator.vault.search("current ai problems calibration", n_results=3, kind="research")
            assert hits
            assert hits[0].document_id == "research:arxiv:test-paper"
        finally:
            orchestrator.shutdown()


def test_live_docker_test_report_uses_runner_result():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.hardware.prepare_run = lambda *args, **kwargs: []  # type: ignore[assignment]
            orchestrator.hardware.list_gpus = lambda: ["GPU 0: Test"]  # type: ignore[assignment]
            orchestrator.docker_runner.live_test = lambda task: LaunchResult(  # type: ignore[assignment]
                mode="subprocess",
                command=["docker", "run"],
                container_name="tar-test",
                returncode=0,
                gpu_visible=True,
                probe_output="GPU 0: Test",
            )
            report = orchestrator.live_docker_test()
            assert report.launched
            assert report.payload_config_path is not None
            assert orchestrator.store.load_recovery().status == "completed"
        finally:
            orchestrator.shutdown()


def test_verify_last_trial_and_breakthrough_report():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            _, _, task = orchestrator.plan_trial(dry_run=True)
            assert task.payload_config_path is not None
            config_path = Path(task.payload_config_path)
            payload = TrainingPayloadConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
            payload = payload.model_copy(update={"steps": 3, "batch_size": 4})
            config_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

            report = orchestrator.verify_last_trial(task.trial_id)
            assert report.trial_id == task.trial_id
            assert report.seed_variance.num_runs == 3
            assert report.calibration.ece >= 0.0
            breakthrough = orchestrator.breakthrough_report(task.trial_id)
            assert breakthrough.trial_id == task.trial_id
            assert breakthrough.status in {"breakthrough", "candidate", "rejected"}
            assert orchestrator.store.latest_breakthrough_report(task.trial_id) is not None
        finally:
            orchestrator.shutdown()


def test_local_openai_role_retries_invalid_schema():
    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            self.calls += 1
            content = "not-json" if self.calls == 1 else '{"experiment_family":"elastic_anchor","anchor_path":"anchors/a.pt","pivot_required":false,"objective_slug":"anchor"}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = FakeClient()
    role = LocalOpenAIRole(
        "director",
        LocalLLMConfig(base_url="http://localhost:8000/v1", model="director"),
        DirectorDraft,
        client_factory=lambda cfg: client,
    )
    draft = role.generate("system", "user")
    assert draft.experiment_family == "elastic_anchor"
    assert client.calls == 2


def test_live_hierarchy_builds_bundle_from_llm_outputs():
    class FakeClient:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            content = self.outputs.pop(0)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    clients = {
        "director": FakeClient(['{"experiment_family":"elastic_anchor","anchor_path":"anchors/live.pt","pivot_required":false,"objective_slug":"anchor"}']),
        "strategist": FakeClient(['{"strategy_family":"elastic_anchor","fim_lambda":1.2,"bregman_budget":0.4,"drift_budget":0.05,"protected_layers":["a"],"mutable_layers":["b"],"hyperparameters":{"alpha":0.07,"eta":0.01,"steps":9,"batch_size":4}}']),
        "scout": FakeClient(['{"training_entrypoint":"tar_lab/train_template.py","image":"pytorch/pytorch:latest","steps":9,"batch_size":4,"power_limit_w":300,"gpu_target_temp_c":70}']),
    }

    def client_factory(config):
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
        hierarchy = TriModelHierarchy(
            director_config=LocalLLMConfig(base_url="http://local", model="director"),
            strategist_config=LocalLLMConfig(base_url="http://local", model="strategist"),
            scout_config=LocalLLMConfig(base_url="http://local", model="scout"),
            client_factory=client_factory,
            allow_rule_fallback=False,
        )
        policy, plan, task = hierarchy.produce_bundle(store, "trial-live", tmp, dry_run=True)
        assert policy.experiment_family == "elastic_anchor"
        assert plan.hyperparameters["steps"] == 9
        assert task.command[:4] == ["python", "-m", "tar_lab.train_template", "--config"]
        assert task.runtime.image == "pytorch/pytorch:latest"
        assert task.runtime.volumes[str(store.data_dir)] == "/data"


def test_data_manager_prepares_dual_stream_manifests():
    with tempfile.TemporaryDirectory() as tmp:
        manager = DataManager(tmp)
        bundle = manager.prepare_dual_stream(force=True, shard_size=2)
        assert bundle.anchor_manifest.records > 0
        assert bundle.research_manifest.records > 0
        assert bundle.anchor_manifest_path == "/data/anchor/manifest.json"
        assert (Path(tmp) / "tar_state" / "data" / "anchor" / "manifest.json").exists()
        assert (Path(tmp) / "tar_state" / "data" / "research" / "manifest.json").exists()


def test_vector_vault_indexes_metrics_and_knowledge():
    with tempfile.TemporaryDirectory() as tmp:
        store = TARStateStore(tmp)
        store.append_metric(
            GovernorMetrics(
                trial_id="trial-1",
                step=1,
                energy_e=0.12,
                entropy_sigma=0.04,
                drift_l2=0.05,
                drift_rho=0.02,
                grad_norm=0.3,
            )
        )
        store.append_knowledge_entry(
            KnowledgeGraphEntry(
                trial_id="trial-1",
                strategy_family="elastic_anchor",
                outcome="completed",
                hyperparameters={"alpha": 0.07, "eta": 0.01},
            )
        )
        vault = VectorVault(tmp)
        try:
            indexer = MemoryIndexer(store, vault)
            indexer.sync_once()
            hits = vault.search("similar loss spikes alpha eta", n_results=2)
            assert hits
            assert any(hit.metadata.get("trial_id") == "trial-1" for hit in hits)
        finally:
            vault.close()


def test_orchestrator_recursive_analysis_indexes_self_correction():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            report = orchestrator.run_dry_run(force_fail_fast=True)
            assert report.governor_action == "terminate"
            assert orchestrator.vault is not None
            hits = orchestrator.vault.search("self-correction fail_fast sigma rho", n_results=5)
            assert any("self_correction:" in hit.document_id for hit in hits)
        finally:
            orchestrator.shutdown()


def test_orchestrator_chat_returns_state_of_lab():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.run_dry_run()
            response = orchestrator.chat("Analyze the current stability")
            assert response.mode == "director_chat"
            assert "stability" in response.response_text.lower() or "E=" in response.response_text
            assert response.state_summary
        finally:
            orchestrator.shutdown()


def test_control_request_chat():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.run_dry_run()
            response = handle_request(
                orchestrator,
                ControlRequest(command="chat", payload={"prompt": "Analyze the current stability"}),
            )
            assert response.ok
            assert response.payload["mode"] == "director_chat"
        finally:
            orchestrator.shutdown()


def test_live_docker_test_report_carries_gpu_probe_output():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.hardware.prepare_run = lambda *args, **kwargs: []  # type: ignore[assignment]
            orchestrator.hardware.list_gpus = lambda: ["GPU 0: Test"]  # type: ignore[assignment]
            orchestrator.docker_runner.live_test = lambda task: LaunchResult(  # type: ignore[assignment]
                mode="docker_sdk",
                command=["docker", "run"],
                container_name="tar-test",
                returncode=0,
                gpu_visible=True,
                probe_output="GPU 0: Test",
            )
            report = orchestrator.live_docker_test()
            assert report.launched
            assert report.gpu_visible is True
            assert report.gpu_probe_output == "GPU 0: Test"
        finally:
            orchestrator.shutdown()


def test_cli_defaults_workspace_to_repo(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["tar_cli.py", "--direct", "--status"],
    )
    args = tar_cli.parse_args()
    assert Path(args.workspace) == Path(tar_cli.__file__).resolve().parent


def test_speech_processor_captures_wake_word_and_logs():
    with tempfile.TemporaryDirectory() as tmp:
        processor = SpeechProcessor(
            workspace=tmp,
            transcriber=lambda _: "Hey Lab abort the run",
        )
        processor.start()
        try:
            processor.submit_audio("dummy.wav")
            command = processor.poll_command(timeout=1.0)
        finally:
            processor.stop()
        assert command == "abort the run"
        log_path = Path(tmp) / "logs" / "activity_audit.log"
        contents = log_path.read_text(encoding="utf-8")
        assert "director_command" in contents


def test_speech_processor_listen_once_uses_capture_hook(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        processor = SpeechProcessor(
            workspace=tmp,
            transcriber=lambda _: "Lab analyze the current stability",
        )
        monkeypatch.setattr(processor, "capture_once", lambda duration_s=3.0, sample_rate=16000: "dummy.wav")
        processor.start()
        try:
            command = processor.listen_once(duration_s=0.01, timeout=1.0)
        finally:
            processor.stop()
        assert command == "analyze the current stability"
