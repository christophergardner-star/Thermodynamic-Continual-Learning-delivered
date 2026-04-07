import tempfile
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

import tar_cli
from tar_lab.control import handle_request
from tar_lab.data_manager import DataManager
from tar_lab.docker_runner import DockerRunner, LaunchResult
from tar_lab.errors import ScientificValidityError
from tar_lab.experiment_backends import ExperimentBackendRegistry
from tar_lab.governor import ThermodynamicGovernor
from tar_lab.hierarchy import Director, DirectorDraft, LocalOpenAIRole, TriModelHierarchy
from tar_lab.inference_bridge import InferenceBridge
from tar_lab.literature_engine import LiteratureEngine
from tar_lab.memory import MemoryIndexer, VectorVault
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.reproducibility import PayloadEnvironmentBuilder
from tar_lab.runtime_daemon import LabRuntimeDaemon
from tar_lab.safe_exec import SandboxedPythonExecutor
from tar_lab.science_profiles import ProblemResearchEngine, ScienceProfileRegistry
from tar_lab.science_exec import execute_study_payload
from tar_lab.schemas import (
    ControlRequest,
    GovernorMetrics,
    GovernorThresholds,
    KnowledgeGraphEntry,
    LocalLLMConfig,
    ProblemExecutionReport,
    RecoveryState,
    ResearchDocument,
    ResearchIngestReport,
    RuntimeLease,
    RuntimeSpec,
    TrainingPayloadConfig,
)
from tar_lab.state import TARStateStore
from tar_lab.train_template import run_payload
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


def test_quarantined_asc_train_script_fails_fast():
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo / "asc_train.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "scientifically valid adversarial ASC trainer" in (proc.stderr + proc.stdout)
    assert "asc_train_full.py" in (proc.stderr + proc.stdout)


def test_quarantined_asc_train_cpu_script_fails_fast():
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo / "asc_train_cpu.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "scientifically valid adversarial ASC trainer" in (proc.stderr + proc.stdout)
    assert "asc_train_full.py" in (proc.stderr + proc.stdout)


def test_deepseek_finetune_warns_that_it_is_experimental():
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "deepseek_asc_finetune.py"),
            "--model",
            "tiny",
            "--dataset",
            "synthetic",
            "--dry_run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "EXPERIMENTAL WARNING" in proc.stderr
    assert "asc_train_full.py" in proc.stderr


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
            assert f"{tmp}:/workspace:ro" in command
            assert f"{Path(tmp) / 'logs'}:/workspace/logs" in command
            assert f"{Path(tmp) / 'tar_runs'}:/workspace/tar_runs" in command
            assert f"{Path(tmp) / 'anchors'}:/workspace/anchors" in command
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


def test_docker_runner_requires_locked_runtime():
    runner = DockerRunner(docker_bin="docker")
    task = SimpleNamespace(
        trial_id="t1",
        command=["python", "-m", "tar_lab.train_template", "--dry_run"],
        runtime=RuntimeSpec(),
    )
    with pytest.raises(RuntimeError, match="locked image and run manifests"):
        runner.compose_command(task)  # type: ignore[arg-type]


def test_docker_command_carries_runtime_manifests():
    runtime = RuntimeSpec(
        image="tar-payload:locked",
        image_locked=True,
        image_manifest_path="/workspace/tar_state/manifests/image.json",
        run_manifest_path="/workspace/tar_state/manifests/run.json",
        env={"TAR_TRIAL_ID": "trial-1"},
        volumes={"/host/tar_runs": "/workspace/tar_runs"},
        read_only_volumes={
            "/host/workspace": "/workspace",
            "/host/manifests": "/manifests",
        },
    )
    command = DockerRunner(docker_bin="docker").compose_runtime_command(
        runtime=runtime,
        container_command=["python", "-m", "tar_lab.train_template", "--dry_run"],
        container_name="tar-trial-1",
    )
    assert "--network" in command
    assert "none" in command
    assert "TAR_IMAGE_MANIFEST=/workspace/tar_state/manifests/image.json" in command
    assert "TAR_RUN_MANIFEST=/workspace/tar_state/manifests/run.json" in command
    assert "/host/workspace:/workspace:ro" in command
    assert "/host/manifests:/manifests:ro" in command


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
            assert payload.backend_id == "asc_text"
            assert payload.adapter_mode == "lora"
            assert payload.dry_run_backbone == "__tiny_gpt2__"
            assert payload.notes["base_model_name"] == "deepseek-ai/deepseek-coder-1.3b-base"
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


def test_science_profile_registry_routes_quantum_problem():
    registry = ScienceProfileRegistry("C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered")
    report = registry.resolve_problem("Investigate barren plateaus in quantum AI with PennyLane ansatz sweeps")
    assert report.profile_id == "quantum_ml"
    assert report.domain == "quantum_ml"
    assert report.confidence > 0.5


def test_problem_research_engine_writes_environment_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        profile_dir = Path(tmp) / "science_profiles"
        source_dir = Path("C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/science_profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.json"):
            (profile_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        engine = ProblemResearchEngine(tmp)
        resolution = engine.registry.resolve_problem("Investigate barren plateaus in quantum AI")
        bundle = engine.prepare_environment("Investigate barren plateaus in quantum AI", resolution=resolution)
        assert bundle.domain == "quantum_ml"
        assert Path(bundle.dockerfile_path).exists()
        assert Path(bundle.requirements_path).exists()
        assert "pennylane" in Path(bundle.requirements_path).read_text(encoding="utf-8").lower()
        assert bundle.install_policy == "profile_locked_only"
        assert f"{Path(tmp)}:/workspace:ro" in bundle.run_command
        assert any(str(Path(tmp) / "tar_state" / "science_envs") in item and "/workspace/tar_state/science_envs/" in item for item in bundle.run_command)


def test_orchestrator_study_problem_persists_and_indexes():
    with tempfile.TemporaryDirectory() as tmp:
        profile_dir = Path(tmp) / "science_profiles"
        source_dir = Path("C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/science_profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.json"):
            (profile_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.research_ingestor.ingest = lambda topic, max_results=6: ResearchIngestReport(  # type: ignore[assignment]
                topic=topic,
                fetched=1,
                indexed=1,
                sources=["manual"],
                documents=[
                    ResearchDocument(
                        document_id="manual:quantum-barren",
                        source_kind="manual",
                        source_name="manual",
                        title="Barren Plateaus",
                        summary="Gradient norms collapse with increasing ansatz depth.",
                        url="https://example.com/quantum",
                        tags=["quantum", "barren_plateau"],
                        problem_statements=["Gradient norms collapse with increasing ansatz depth."],
                    )
                ],
            )
            report = orchestrator.study_problem("Investigate barren plateaus in quantum AI", build_env=False, max_results=1)
            assert report.domain == "quantum_ml"
            assert report.environment.profile_id == "quantum_ml"
            assert report.experiments
            latest = orchestrator.store.latest_problem_study()
            assert latest is not None
            assert latest.problem_id == report.problem_id
            assert orchestrator.vault is not None
            hits = orchestrator.vault.search("barren plateaus ansatz depth study", n_results=5, kind="problem_study")
            assert hits
        finally:
            orchestrator.shutdown()


def test_problem_runner_executes_generic_payload(tmp_path: Path):
    payload = {
        "problem_id": "study-1",
        "problem": "Investigate generalization in generic ML",
        "profile_id": "generic_ml",
        "domain": "generic_ml",
        "environment": {"validation_imports": []},
        "experiments": [
            {
                "template_id": "baseline_sweep",
                "name": "Baseline Sweep",
                "benchmark": "cross_validation",
                "metrics": ["accuracy", "seed_variance"],
                "parameter_grid": {"learning_rate": [0.001]},
            }
        ],
    }
    output = tmp_path / "execution_report.json"
    report = execute_study_payload(payload, output)
    assert report.status == "completed"
    assert report.experiments
    assert output.exists()


def test_problem_runner_executes_deep_learning_benchmarks(tmp_path: Path):
    payload = {
        "problem_id": "dl-1",
        "problem": "Investigate optimizer sensitivity in deep learning",
        "profile_id": "deep_learning",
        "domain": "deep_learning",
        "environment": {"validation_imports": []},
        "experiments": [
            {
                "template_id": "optimizer_ablation",
                "name": "Optimizer Ablation",
                "benchmark": "optimizer_comparison",
                "metrics": ["loss", "accuracy", "gradient_norm"],
                "parameter_grid": {"optimizer": ["adamw", "sgd"], "weight_decay": [0.0]},
            },
            {
                "template_id": "depth_width_scale",
                "name": "Depth-Width Scaling",
                "benchmark": "scaling_law_probe",
                "metrics": ["loss", "effective_dimensionality", "calibration_ece"],
                "parameter_grid": {"depth": [4], "width": [128, 256]},
            },
        ],
    }
    output = tmp_path / "deep_learning_execution.json"
    report = execute_study_payload(payload, output)
    assert report.status == "completed"
    assert {item.template_id for item in report.experiments} == {"optimizer_ablation", "depth_width_scale"}
    optimizer_metrics = next(item.metrics for item in report.experiments if item.template_id == "optimizer_ablation")
    assert optimizer_metrics["accuracy"] > 0.5
    assert optimizer_metrics["gradient_norm"] > 0.0


def test_problem_runner_executes_nlp_benchmarks(tmp_path: Path):
    payload = {
        "problem_id": "nlp-1",
        "problem": "Investigate prompt and length failures in NLP",
        "profile_id": "natural_language_processing",
        "domain": "natural_language_processing",
        "environment": {"validation_imports": []},
        "experiments": [
            {
                "template_id": "prompt_retrieval_ablation",
                "name": "Prompt and Retrieval Ablation",
                "benchmark": "retrieval_prompt_ablation",
                "metrics": ["rouge", "hallucination_rate", "calibration_ece"],
                "parameter_grid": {"retrieval": ["off", "bm25", "dense"], "prompt_style": ["short", "chain_of_thought"]},
            },
            {
                "template_id": "length_generalization",
                "name": "Length Generalization",
                "benchmark": "sequence_length_sweep",
                "metrics": ["perplexity", "calibration_ece"],
                "parameter_grid": {"sequence_length": [256, 512]},
            },
        ],
    }
    output = tmp_path / "nlp_execution.json"
    report = execute_study_payload(payload, output)
    assert report.status == "completed"
    assert {item.template_id for item in report.experiments} == {"prompt_retrieval_ablation", "length_generalization"}
    retrieval_metrics = next(item.metrics for item in report.experiments if item.template_id == "prompt_retrieval_ablation")
    assert 0.0 <= retrieval_metrics["hallucination_rate"] <= 1.0
    assert retrieval_metrics["rouge"] > 0.0


def test_problem_runner_executes_rl_benchmarks(tmp_path: Path):
    payload = {
        "problem_id": "rl-1",
        "problem": "Investigate exploration collapse in reinforcement learning",
        "profile_id": "reinforcement_learning",
        "domain": "reinforcement_learning",
        "environment": {"validation_imports": []},
        "experiments": [
            {
                "template_id": "exploration_ablation",
                "name": "Exploration Ablation",
                "benchmark": "exploration_sweep",
                "metrics": ["episodic_return", "policy_entropy", "seed_variance"],
                "parameter_grid": {"entropy_coef": [0.0, 0.05], "algorithm": ["ppo", "a2c"]},
            },
            {
                "template_id": "offline_online_gap",
                "name": "Offline-to-Online Gap",
                "benchmark": "offline_online_transfer",
                "metrics": ["episodic_return", "sample_efficiency"],
                "parameter_grid": {"dataset_quality": ["medium", "expert"], "fine_tune_steps": [0, 10000]},
            },
        ],
    }
    output = tmp_path / "rl_execution.json"
    report = execute_study_payload(payload, output)
    assert report.status == "completed"
    assert {item.template_id for item in report.experiments} == {"exploration_ablation", "offline_online_gap"}
    rl_metrics = next(item.metrics for item in report.experiments if item.template_id == "exploration_ablation")
    assert rl_metrics["episodic_return"] > 0.0
    assert rl_metrics["policy_entropy"] > 0.0


def test_problem_runner_executes_generic_ml_benchmarks(tmp_path: Path):
    payload = {
        "problem_id": "generic-1",
        "problem": "Investigate calibration and baseline variance in generic ML",
        "profile_id": "generic_ml",
        "domain": "generic_ml",
        "environment": {"validation_imports": []},
        "experiments": [
            {
                "template_id": "baseline_sweep",
                "name": "Baseline Sweep",
                "benchmark": "cross_validation",
                "metrics": ["accuracy", "f1", "seed_variance"],
                "parameter_grid": {"learning_rate": [0.001, 0.0003], "batch_size": [32, 64]},
            },
            {
                "template_id": "calibration_check",
                "name": "Calibration Check",
                "benchmark": "holdout_calibration",
                "metrics": ["calibration_ece", "accuracy", "auroc"],
                "parameter_grid": {"temperature_scaling": [False, True]},
            },
        ],
    }
    output = tmp_path / "generic_execution.json"
    report = execute_study_payload(payload, output)
    assert report.status == "completed"
    assert {item.template_id for item in report.experiments} == {"baseline_sweep", "calibration_check"}
    baseline_metrics = next(item.metrics for item in report.experiments if item.template_id == "baseline_sweep")
    assert baseline_metrics["accuracy"] > 0.6
    assert 0.0 <= baseline_metrics["seed_variance"] < 0.2


def test_problem_runner_executes_computer_vision_benchmarks(tmp_path: Path):
    payload = {
        "problem_id": "cv-1",
        "problem": "Investigate robustness and transfer in computer vision",
        "profile_id": "computer_vision",
        "domain": "computer_vision",
        "environment": {"validation_imports": []},
        "experiments": [
            {
                "template_id": "augmentation_robustness",
                "name": "Augmentation Robustness",
                "benchmark": "corruption_robustness",
                "metrics": ["top1_accuracy", "corruption_robustness", "calibration_ece"],
                "parameter_grid": {"augmentation": ["baseline", "randaugment"], "severity": [0, 2]},
            },
            {
                "template_id": "backbone_transfer",
                "name": "Backbone Transfer",
                "benchmark": "transfer_comparison",
                "metrics": ["top1_accuracy", "seed_variance"],
                "parameter_grid": {"backbone": ["resnet18", "vit_tiny"]},
            },
        ],
    }
    output = tmp_path / "cv_execution.json"
    report = execute_study_payload(payload, output)
    assert report.status == "completed"
    assert {item.template_id for item in report.experiments} == {"augmentation_robustness", "backbone_transfer"}
    cv_metrics = next(item.metrics for item in report.experiments if item.template_id == "augmentation_robustness")
    assert cv_metrics["top1_accuracy"] > 0.5
    assert cv_metrics["corruption_robustness"] >= 0.0


def test_problem_runner_executes_graph_ml_benchmarks(tmp_path: Path):
    payload = {
        "problem_id": "graph-1",
        "problem": "Investigate oversmoothing and heterophily in graph ML",
        "profile_id": "graph_ml",
        "domain": "graph_ml",
        "environment": {"validation_imports": []},
        "experiments": [
            {
                "template_id": "depth_oversmoothing",
                "name": "Depth and Oversmoothing",
                "benchmark": "depth_sweep",
                "metrics": ["node_accuracy", "oversmoothing_gap", "representation_dimensionality"],
                "parameter_grid": {"layers": [2, 4, 8]},
            },
            {
                "template_id": "heterophily_ablation",
                "name": "Heterophily Ablation",
                "benchmark": "heterophily_control",
                "metrics": ["node_accuracy", "seed_variance"],
                "parameter_grid": {"rewiring": ["off", "knn"], "normalization": ["batch", "pairnorm"]},
            },
        ],
    }
    output = tmp_path / "graph_execution.json"
    report = execute_study_payload(payload, output)
    assert report.status == "completed"
    assert {item.template_id for item in report.experiments} == {"depth_oversmoothing", "heterophily_ablation"}
    graph_metrics = next(item.metrics for item in report.experiments if item.template_id == "depth_oversmoothing")
    assert graph_metrics["node_accuracy"] > 0.5
    assert graph_metrics["representation_dimensionality"] > 0.0


def test_orchestrator_run_problem_study_reads_execution_report():
    with tempfile.TemporaryDirectory() as tmp:
        profile_dir = Path(tmp) / "science_profiles"
        source_dir = Path("C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/science_profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.json"):
            (profile_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem("Investigate barren plateaus in quantum AI", build_env=False, max_results=0)
            report_path = Path(study.environment.execution_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            fake = ProblemExecutionReport(
                problem_id=study.problem_id,
                problem=study.problem,
                profile_id=study.profile_id,
                domain=study.domain,
                execution_mode="local_python",
                imports_ok=["numpy"],
                experiments=[],
                summary="Synthetic execution complete.",
                recommended_next_step="Promote to the locked Docker bundle.",
                artifact_path=str(report_path),
                status="completed",
            )
            report_path.write_text(fake.model_dump_json(indent=2), encoding="utf-8")
            original_run = subprocess.run
            try:
                subprocess.run = lambda *args, **kwargs: SimpleNamespace(returncode=0)  # type: ignore[assignment]
                report = orchestrator.run_problem_study(problem_id=study.problem_id, use_docker=False)
            finally:
                subprocess.run = original_run  # type: ignore[assignment]
            assert report.status == "completed"
            assert orchestrator.store.latest_problem_execution(study.problem_id) is not None
            assert orchestrator.vault is not None
            hits = orchestrator.vault.search("problem execution barren plateaus", n_results=5, kind="problem_execution")
            assert hits
        finally:
            orchestrator.shutdown()


def test_scheduler_reschedules_then_completes_problem_study():
    with tempfile.TemporaryDirectory() as tmp:
        profile_dir = Path(tmp) / "science_profiles"
        source_dir = Path("C:/Users/Chris/contLRN/Thermodynamic-Continual-Learning-delivered/science_profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.json"):
            (profile_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem("Investigate optimization stability in deep learning", build_env=False, max_results=0)
            fake_report = ProblemExecutionReport(
                problem_id=study.problem_id,
                problem=study.problem,
                profile_id=study.profile_id,
                domain=study.domain,
                execution_mode="local_python",
                imports_ok=["numpy"],
                experiments=[],
                summary="Scheduled execution completed.",
                recommended_next_step="Promote the next run.",
                artifact_path=study.environment.execution_report_path,
                status="completed",
            )
            orchestrator.run_problem_study = lambda *args, **kwargs: fake_report  # type: ignore[assignment]
            entry = orchestrator.schedule_problem_study(
                problem_id=study.problem_id,
                delay_s=0,
                repeat_interval_s=60,
                max_runs=2,
            )
            first = orchestrator.scheduler.run_once(now=datetime.now(timezone.utc), max_jobs=1)
            assert first.executed_count == 1
            updated = orchestrator.store.get_problem_schedule(entry.schedule_id)
            assert updated is not None
            assert updated.status == "scheduled"
            assert updated.run_count == 1

            orchestrator.store.update_problem_schedule(
                entry.schedule_id,
                next_run_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            )
            second = orchestrator.scheduler.run_once(now=datetime.now(timezone.utc), max_jobs=1)
            assert second.executed_count == 1
            completed = orchestrator.store.get_problem_schedule(entry.schedule_id)
            assert completed is not None
            assert completed.status == "completed"
            assert completed.run_count == 2
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
            payload = payload.model_copy(
                update={
                    "steps": 3,
                    "batch_size": 4,
                    "adapter_mode": "full",
                    "notes": {
                        **payload.notes,
                        "base_model_name": "__tiny_gpt2__",
                        "n_embd": 24,
                        "n_layer": 1,
                        "n_head": 2,
                        "max_seq_len": 24,
                    },
                }
            )
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
        assert task.runtime.read_only_volumes[str(store.data_dir)] == "/data"
        assert task.runtime.read_only_volumes[str(store.workspace)] == "/workspace"
        assert task.runtime.volumes[str(store.workspace / "tar_runs")] == "/workspace/tar_runs"


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
            assert isinstance(response.evidence_traces, list)
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


def test_experiment_backend_registry_builds_real_backend_plan():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ExperimentBackendRegistry(tmp)
        backends = {item.backend_id for item in registry.list_backends()}
        assert {"asc_text", "toy_anchor", "asc_full", "coding_asc", "asc_cv", "asc_rl", "asc_qml"} <= backends
        plan = registry.build_plan("asc_full", trial_name="trial-backend", config={"max_steps": 12})
        assert "asc_train_full.py" in plan.command
        assert "--max_steps" in plan.command
        assert Path(plan.manifest_path).exists()
        cv_plan = registry.build_plan("asc_cv", trial_name="trial-cv", config={"run_intent": "control"})
        assert "tar_lab.multimodal_payloads" in cv_plan.command
        assert Path(cv_plan.manifest_path).exists()


def test_literature_engine_extracts_claims_and_conflicts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paper_a = root / "paper_a.txt"
        paper_b = root / "paper_b.txt"
        paper_a.write_text(
            "Paper A\n\nAbstract\nThis method improves robustness.\n\nResults\nThe method is stable and better than baseline [1].\n\n"
            "Table 1 Performance Summary\nModel  Accuracy  ECE\nASC    0.91      0.03\nAdamW  0.84      0.09\n\n"
            "Figure 1 Calibration improves as entropy decreases.",
            encoding="utf-8",
        )
        paper_b.write_text(
            "Paper B\n\nAbstract\nThis method does not improve robustness.\n\nResults\nThe method is not stable and worse than baseline [2].",
            encoding="utf-8",
        )
        engine = LiteratureEngine(tmp)
        report = engine.ingest_paths([str(paper_a), str(paper_b)])
        assert report.ingested == 2
        assert report.artifacts[0].claims
        assert report.artifacts[0].tables
        assert report.artifacts[0].figures
        assert report.conflicts


def test_payload_environment_builder_writes_locked_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        report = builder.prepare(image_tag="tar-payload:test")
        assert Path(report.dockerfile_path).exists()
        assert Path(report.requirements_path).exists()
        assert "pydantic" in "\n".join(report.packages)
        assert "transformers" in "\n".join(report.packages)
        assert "peft" in "\n".join(report.packages)
        assert builder.load() is not None


def test_train_template_runs_real_asc_payload():
    with tempfile.TemporaryDirectory() as tmp:
        manager = DataManager(tmp)
        bundle = manager.prepare_dual_stream(force=True)
        config = TrainingPayloadConfig(
            trial_id="trial-asc-payload",
            backend_id="asc_text",
            strategy_family="elastic_anchor",
            anchor_path=str(Path(tmp) / "anchors" / "anchor.pt"),
            alpha=0.02,
            eta=0.005,
            fim_lambda=0.1,
            bregman_budget=0.2,
            drift_budget=0.05,
            batch_size=1,
            steps=2,
            log_path=str(Path(tmp) / "logs" / "thermo_metrics.jsonl"),
            output_dir=str(Path(tmp) / "tar_runs" / "trial-asc-payload" / "output"),
            anchor_manifest_path=str(manager.store.dataset_manifest_path("anchor")),
            research_manifest_path=str(manager.store.dataset_manifest_path("research")),
            governor_thresholds=GovernorThresholds(),
            protected_layers=["transformer.wte"],
            mutable_layers=["transformer.h.0"],
            notes={"base_model_name": "__tiny_gpt2__", "max_seq_len": 24, "seed": 5, "n_embd": 24, "n_layer": 1, "n_head": 2},
        )
        summary = run_payload(config, dry_run=True)
        assert summary["backend_id"] == "asc_text"
        assert summary["anchor_records"] == bundle.anchor_manifest.records
        assert summary["research_records"] == bundle.research_manifest.records
        assert Path(summary["checkpoint_path"]).exists()
        assert summary["last_metrics"]["effective_dimensionality"] >= 0.0
        assert summary["run_intent"] == "control"
        assert summary["research_grade"] is False


@pytest.mark.parametrize("backend_id", ["asc_cv", "asc_rl", "asc_qml"])
def test_train_template_runs_multimodal_backend_paths(backend_id: str):
    with tempfile.TemporaryDirectory() as tmp:
        registry = ExperimentBackendRegistry(tmp)
        config = TrainingPayloadConfig(
            trial_id=f"trial-{backend_id}",
            backend_id=backend_id,  # type: ignore[arg-type]
            run_intent="control",
            strategy_family="elastic_anchor",
            anchor_path=str(Path(tmp) / "anchors" / "anchor.pt"),
            alpha=0.02,
            eta=0.005,
            fim_lambda=0.1,
            bregman_budget=0.2,
            drift_budget=0.05,
            batch_size=1,
            steps=2,
            log_path=str(Path(tmp) / "logs" / f"{backend_id}.jsonl"),
            output_dir=str(Path(tmp) / "tar_runs" / f"trial-{backend_id}" / "output"),
            governor_thresholds=GovernorThresholds(),
            protected_layers=["layer.a"],
            mutable_layers=["layer.b"],
            backend_provenance=registry.as_provenance(backend_id),
        )
        summary = run_payload(config, dry_run=True)
        assert summary["backend_id"] == backend_id
        assert summary["backend_readiness"] == "executable"
        assert Path(summary["checkpoint_path"]).exists()
        assert summary["data_provenance"] is not None


def test_payload_refuses_research_run_without_complete_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ExperimentBackendRegistry(tmp)
        config = TrainingPayloadConfig(
            trial_id="trial-invalid-research",
            backend_id="asc_text",
            run_intent="research",
            strategy_family="elastic_anchor",
            anchor_path=str(Path(tmp) / "anchors" / "anchor.pt"),
            alpha=0.02,
            eta=0.005,
            fim_lambda=0.1,
            bregman_budget=0.2,
            drift_budget=0.05,
            batch_size=1,
            steps=1,
            log_path=str(Path(tmp) / "logs" / "invalid.jsonl"),
            output_dir=str(Path(tmp) / "tar_runs" / "trial-invalid-research" / "output"),
            governor_thresholds=GovernorThresholds(),
            protected_layers=["layer.a"],
            mutable_layers=["layer.b"],
            backend_provenance=registry.as_provenance("asc_text"),
            provenance_complete=False,
            research_grade=False,
        )
        with pytest.raises(ScientificValidityError, match="provenance is incomplete"):
            run_payload(config, dry_run=True)


def test_status_rendering_makes_control_vs_research_explicit():
    payload = {
        "recovery": {"trial_id": "trial-1", "status": "completed", "consecutive_fail_fast": 0},
        "last_three_metrics": [{"energy_e": 0.1, "entropy_sigma": 0.2, "drift_rho": 0.3, "grad_norm": 0.4, "effective_dimensionality": 2.0, "dimensionality_ratio": 0.8, "equilibrium_fraction": 0.0}],
        "gpu": {"temperature_c": 50.0, "power_w": None},
        "run_intent": "control",
        "backend_id": "toy_anchor",
        "backend_readiness": "executable",
        "data_purity": "fallback",
        "tokenizer_integrity": False,
        "research_grade": False,
        "benchmark_name": "Smoke Grounded QA Probe",
        "benchmark_tier": "smoke",
        "actual_benchmark_tiers": ["smoke"],
        "benchmark_truth_statuses": ["smoke_only"],
        "benchmark_alignment": "aligned",
        "canonical_comparable": False,
    }
    rendered = tar_cli._render_status(payload)
    assert "Run Intent: control" in rendered
    assert "Backend: toy_anchor" in rendered
    assert "Backend Readiness: executable" in rendered
    assert "Research Grade: False" in rendered
    assert "Benchmark: Smoke Grounded QA Probe" in rendered
    assert "Benchmark Tier: smoke" in rendered
    assert "Benchmark Actual Tier(s): smoke" in rendered
    assert "Benchmark Truth: smoke_only" in rendered
    assert "Benchmark Alignment: aligned" in rendered
    assert "Sandbox Profile: n/a" in rendered


def test_status_renderer_surfaces_memory_and_lock_warnings():
    payload = {
        "recovery": {"trial_id": "trial-2", "status": "failed", "consecutive_fail_fast": 1},
        "last_three_metrics": [],
        "gpu": {},
        "memory": {
            "state": "degraded",
            "collection_name": "lab_history__broken",
            "embedder": "lexical-semantic-fallback",
            "embedding_dim": 1369,
        },
        "memory_warning": "memory manifest mismatch",
        "lock_incomplete_reason": "Missing pinned dependency versions: peft",
        "unresolved_dependency_count": 1,
        "safe_execution_mode": "docker_container_only",
        "sandbox_profile": "production",
        "sandbox_read_only_mounts": ["/workspace"],
        "sandbox_writable_mounts": ["/workspace/tar_runs"],
    }
    rendered = tar_cli._render_status(payload)
    assert "Memory Warning: memory manifest mismatch" in rendered
    assert "Lock Warning: Missing pinned dependency versions: peft" in rendered
    assert "Memory State: degraded" in rendered


def test_orchestrator_status_surfaces_benchmark_identity():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem(
                "Investigate optimization stability in deep learning",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            status = orchestrator.status()
            assert study.benchmark_ids
            assert study.benchmark_names
            assert status["benchmark_ids"] == study.benchmark_ids
            assert status["benchmark_names"] == study.benchmark_names
            assert status["benchmark_truth_statuses"] == study.benchmark_truth_statuses
            assert status["benchmark_alignment"] == study.benchmark_alignment
            assert status["actual_benchmark_tiers"] == study.actual_benchmark_tiers
            assert status["benchmark_name"] == study.benchmark_names[0]
        finally:
            orchestrator.shutdown()


def test_benchmark_status_surfaces_refusal_reason_for_unsupported_canonical():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            payload = orchestrator.benchmark_status(profile_id="natural_language_processing", tier="canonical")
            benchmarks = payload["benchmarks"]
            suite = next(item for item in benchmarks if item["spec"]["benchmark_id"] == "beir_fiqa_canonical")
            assert suite["spec"]["truth_status"] == "unsupported"
            assert suite["availability"]["canonical_ready"] is False
            assert "must be refused" in (suite["availability"]["reason"] or "")
        finally:
            orchestrator.shutdown()


def test_runtime_daemon_writes_heartbeat_and_cleans_stale_entry():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem("Investigate calibration in deep learning", build_env=False, max_results=1)
            entry = orchestrator.schedule_problem_study(problem_id=study.problem_id, delay_s=0, max_runs=1)
            orchestrator.store.update_problem_schedule(
                entry.schedule_id,
                status="running",
                last_execution_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                lease=RuntimeLease(
                    owner_id="stale-worker",
                    heartbeat_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                    expires_at=(datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
                    attempt=1,
                ),
            )
            heartbeat = orchestrator.run_runtime_cycle(max_jobs=1, stale_after_s=1)
            assert Path(orchestrator.runtime_daemon.heartbeat_path).exists()
            assert heartbeat.stale_cleanups >= 1
        finally:
            orchestrator.shutdown()


def test_sandboxed_python_executor_reports_unavailable_without_host_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        executor = SandboxedPythonExecutor(workspace=tmp, docker_bin="definitely-not-docker")
        ok, output, mode = executor.run("print('hello')", allow_host_fallback=False)
        assert not ok
        assert mode == "unavailable"
        assert "Sandbox execution unavailable" in output


def test_inference_bridge_registers_checkpoint_and_builds_endpoint():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        model_dir.mkdir()
        bridge = InferenceBridge(tmp)
        record = bridge.register_checkpoint(
            name="asc-local",
            model_path=str(model_dir),
            backend="transformers",
            role="assistant",
        )
        assert record.name == "asc-local"
        endpoint = bridge.build_endpoint("asc-local", port=8100, role="director")
        assert endpoint.base_url.endswith(":8100/v1")
        assert "serve_local.py" in " ".join(endpoint.command)
        assert endpoint.role == "director"
        assert endpoint.manifest_path is not None
        assert Path(endpoint.manifest_path).exists()
        assert endpoint.trust_remote_code is False


def test_frontier_status_reports_new_foundations():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            frontier = orchestrator.frontier_status()
            backend_ids = {item.backend_id for item in frontier.experiment_backends}
            assert "asc_full" in backend_ids
            assert frontier.safe_execution_mode == "docker_container_only"
            assert frontier.reranker == "scientific-hybrid-reranker"
            assert frontier.literature_capabilities is not None
            assert frontier.claim_policy is not None
        finally:
            orchestrator.shutdown()


def test_endpoint_health_renderer_surfaces_logs_and_trust_policy():
    payload = {
        "endpoint": {
            "endpoint_name": "assistant-asc-local",
            "status": "failed",
            "role": "assistant",
            "backend": "transformers",
            "trust_remote_code": True,
            "last_error": "startup failed",
            "stdout_log_path": "C:/tmp/stdout.log",
            "stderr_log_path": "C:/tmp/stderr.log",
        },
        "health": {
            "status": "failed",
            "ok": False,
            "backend": "transformers",
            "model_id": "asc-local",
            "role": "assistant",
            "trust_remote_code": True,
            "detail": "stderr tail: missing dependency",
            "checked_at": "2026-04-07T18:00:00Z",
        },
    }
    rendered = tar_cli._render_endpoint_health(payload)
    assert "Trust Remote Code: True" in rendered
    assert "Stdout Log: C:/tmp/stdout.log" in rendered
    assert "Stderr Log: C:/tmp/stderr.log" in rendered


def test_endpoint_list_renderer_surfaces_record_details():
    payload = {
        "endpoints": [
            {
                "endpoint_name": "assistant-asc-local",
                "status": "running",
                "role": "assistant",
                "checkpoint_name": "asc-local",
                "backend": "transformers",
                "base_url": "http://127.0.0.1:8801/v1",
                "trust_remote_code": False,
                "process_pid": 1234,
                "last_error": None,
                "last_health_at": "2026-04-07T18:00:00Z",
                "stdout_log_path": "C:/tmp/stdout.log",
                "stderr_log_path": "C:/tmp/stderr.log",
                "manifest_path": "C:/tmp/endpoint_manifest.json",
                "health": {"status": "healthy"},
            }
        ]
    }
    rendered = tar_cli._render_endpoint_list(payload)
    assert "Endpoint: assistant-asc-local" in rendered
    assert "Trust Remote Code: False" in rendered
    assert "Manifest: C:/tmp/endpoint_manifest.json" in rendered


def test_runtime_status_renderer_surfaces_lock_warning():
    payload = {
        "safe_execution_mode": "docker_container_only",
        "payload_image": "tar-payload:locked",
        "manifest_hash": "abc123",
        "reproducibility_complete": False,
        "lock_incomplete_reason": "Missing pinned dependency versions: peft",
        "active_leases": [],
        "retry_waiting": [],
        "terminal_failures": [],
        "alerts": [],
        "sandbox_policy": {
            "profile": "production",
            "dev_override_active": False,
            "read_only_mounts": ["/workspace"],
            "writable_mounts": ["/workspace/tar_runs"],
        },
    }
    rendered = tar_cli._render_runtime_status(payload)
    assert "Sandbox Profile: production" in rendered
    assert "Lock Warning: Missing pinned dependency versions: peft" in rendered


def test_runtime_status_surfaces_sandbox_mount_policy():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            runtime = orchestrator.runtime_status()
            sandbox = runtime["sandbox_policy"]
            assert sandbox["profile"] == "production"
            assert "/workspace" in sandbox["read_only_mounts"]
            assert "/workspace/tar_runs" in sandbox["writable_mounts"]
            assert sandbox["dev_override_active"] is False
        finally:
            orchestrator.shutdown()
