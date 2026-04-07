import statistics
import tempfile
from pathlib import Path

import torch

from tar_lab.data_manager import DataManager
from tar_lab.governor import ThermodynamicGovernor
from tar_lab.schemas import GovernorMetrics, GovernorThresholds, TrainingPayloadConfig
from tar_lab.thermoobserver import StatAccumulator, compute_activation_covariance, compute_participation_ratio
from tar_lab.train_template import run_payload


def test_smoothed_dpr_has_lower_variance_than_batchwise_estimate():
    torch.manual_seed(13)
    accumulator = StatAccumulator(window_size=5)
    batchwise: list[float] = []
    smoothed: list[float] = []

    for _ in range(32):
        activations = torch.randn(48, 24)
        covariance = compute_activation_covariance(activations)
        accumulator.push(covariance, sigma=1.0, rho=1.0)
        batchwise.append(compute_participation_ratio(activations))
        stats = accumulator.get_smoothed_metrics()
        if stats.statistically_ready:
            smoothed.append(stats.effective_dimensionality)

    assert len(smoothed) >= 5
    raw_tail = batchwise[accumulator.window_size - 1 :]
    assert statistics.pvariance(smoothed) < statistics.pvariance(raw_tail)


def test_governor_blocks_quenching_until_stat_window_is_ready():
    governor = ThermodynamicGovernor()
    thresholds = GovernorThresholds(min_dimensionality_ratio=0.35, max_quenching_loss=1.2)
    warming = GovernorMetrics(
        trial_id="trial",
        step=3,
        energy_e=0.02,
        entropy_sigma=0.03,
        drift_l2=0.04,
        drift_rho=0.02,
        grad_norm=0.5,
        effective_dimensionality=1.2,
        dimensionality_ratio=0.20,
        training_loss=0.55,
        stat_window_size=5,
        stat_sample_count=3,
        statistically_ready=False,
    )
    assert "thermodynamic_quenching" not in governor.evaluate(warming, thresholds).reasons

    ready = warming.model_copy(update={"stat_sample_count": 5, "statistically_ready": True})
    assert "thermodynamic_quenching" in governor.evaluate(ready, thresholds).reasons


def test_dry_run_uses_tiny_execution_backbone_for_large_requested_model():
    with tempfile.TemporaryDirectory() as tmp:
        manager = DataManager(tmp)
        bundle = manager.prepare_dual_stream(force=True)
        config = TrainingPayloadConfig(
            trial_id="trial-dry-run-large-model",
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
            seed=5,
            log_path=str(Path(tmp) / "logs" / "thermo_metrics.jsonl"),
            output_dir=str(Path(tmp) / "tar_runs" / "trial-dry-run-large-model" / "output"),
            anchor_manifest_path=str(manager.store.dataset_manifest_path("anchor")),
            research_manifest_path=str(manager.store.dataset_manifest_path("research")),
            governor_thresholds=GovernorThresholds(),
            protected_layers=["transformer.wte"],
            mutable_layers=["transformer.h.0"],
            notes={"base_model_name": "deepseek-ai/deepseek-coder-1.3b-base", "max_seq_len": 24, "n_embd": 24, "n_layer": 1, "n_head": 2},
        )
        summary = run_payload(config, dry_run=True)
        assert summary["requested_payload_model"] == "deepseek-ai/deepseek-coder-1.3b-base"
        assert summary["payload_model"] == "__tiny_gpt2__"
