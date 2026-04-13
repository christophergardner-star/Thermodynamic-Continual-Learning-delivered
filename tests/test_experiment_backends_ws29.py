import argparse
import tempfile
from pathlib import Path

import torch

from asc_model import ASCConfig, ASCForCausalLM
from asc_train_full import (
    _load_resume_bundle,
    _normalize_rng_tensor,
    _partial_step_cap_reached,
    _resolve_tokenizer_source,
    _restore_resume_state,
    _save_resume_bundle,
)
from tar_lab.experiment_backends import ExperimentBackendRegistry


def _tiny_model() -> ASCForCausalLM:
    config = ASCConfig(
        base_model_name="__tiny_gpt2__",
        warp_dim=16,
        consistency_lambda=0.3,
        ema_decay=0.995,
        backbone_config_overrides={
            "vocab_size": 64,
            "n_positions": 16,
            "n_ctx": 16,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
        },
    )
    return ASCForCausalLM(config)


def _args(out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        size="124M",
        dataset="wikitext-2-raw-v1",
        max_length=32,
        batch_size=2,
        lr=3e-4,
        epochs=2,
        lambda_c=0.3,
        ema_decay=0.995,
        warp_dim=16,
        save_every=10,
        log_every=5,
        max_steps=20,
        out_dir=str(out_dir),
        seed=7,
        resume_from_checkpoint=None,
        backend_state_path=None,
    )


def test_ws29_build_plan_marks_asc_full_as_resumable_and_persists_runtime_record():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ExperimentBackendRegistry(tmp)
        plan = registry.build_plan("asc_full", trial_name="trial-backend", config={"max_steps": 12, "size": "124M"})

        assert plan.resume.supported is True
        assert plan.resume.mode == "fresh_start"
        assert "--out_dir" in plan.command
        assert "--backend_state_path" in plan.command
        assert Path(plan.manifest_path).exists()
        assert plan.backend_state_path is not None
        record = registry.store.load_experiment_backend_runtime("trial-backend", "asc_full")
        assert record is not None
        assert record.status == "planned"
        assert record.resume.supported is True
        assert record.artifact_lineage.training_log_path is not None
        assert record.artifact_lineage.training_log_path.endswith("training_log.json")


def test_ws29_build_plan_detects_existing_resume_state():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ExperimentBackendRegistry(tmp)
        run_dir = Path(tmp) / "tar_runs" / "trial-backend" / "asc_full" / "ASC-124M"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"latest_checkpoint_path": str(run_dir / "step_12")}, run_dir / "resume_state.pt")

        plan = registry.build_plan("asc_full", trial_name="trial-backend", config={"max_steps": 12, "size": "124M"})

        assert plan.resume.requested is True
        assert plan.resume.checkpoint_exists is True
        assert "--resume_from_checkpoint" in plan.command
        record = registry.store.load_experiment_backend_runtime("trial-backend", "asc_full")
        assert record is not None
        assert record.resume.mode == "checkpoint_resume"


def test_ws29_resume_bundle_round_trip_for_asc_full():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "asc_full" / "ASC-124M"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = _tiny_model()
        base_optimizer = torch.optim.AdamW(model.parameters_trainable(), lr=1e-3)
        warp_optimizer = torch.optim.AdamW(model.warp.parameters(), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=10)
        checkpoint_dir = run_dir / "step_3"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint_dir))

        resume_path = _save_resume_bundle(
            run_dir,
            model=model,
            base_optimizer=base_optimizer,
            warp_optimizer=warp_optimizer,
            scheduler=scheduler,
            step=3,
            epoch_index=1,
            epoch_step=2,
            history={"step": [1.0, 2.0], "task_loss": [1.2, 1.1], "consist_loss": [0.8, 0.7], "ppl": [3.3, 3.0]},
            args=_args(run_dir.parent),
            latest_checkpoint_path=str(checkpoint_dir),
            latest_metrics={"task_loss": 1.1, "consistency_loss": 0.7, "ppl": 3.0},
        )

        payload = _load_resume_bundle(resume_path, torch.device("cpu"))
        assert payload["completed_steps"] == 3
        assert payload["completed_epochs"] == 1
        assert payload["epoch_step"] == 2
        assert payload["latest_checkpoint_path"] == str(checkpoint_dir)
        assert payload["history"]["task_loss"][-1] == 1.1


def test_ws29_resolve_tokenizer_source_falls_back_to_saved_base_model_name():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ASC-124M" / "step_3"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "asc_config.json").write_text(
            '{"base_model_name": "gpt2"}',
            encoding="utf-8",
        )

        tokenizer_source = _resolve_tokenizer_source(
            size="124M",
            resume_bundle={"latest_checkpoint_path": str(checkpoint_dir)},
        )

        assert tokenizer_source == "gpt2"


def test_ws29_restore_resume_state_normalizes_rng_payloads():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "asc_full" / "ASC-124M"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = _tiny_model()
        base_optimizer = torch.optim.AdamW(model.parameters_trainable(), lr=1e-3)
        warp_optimizer = torch.optim.AdamW(model.warp.parameters(), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=10)
        checkpoint_dir = run_dir / "step_3"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint_dir))

        resume_path = _save_resume_bundle(
            run_dir,
            model=model,
            base_optimizer=base_optimizer,
            warp_optimizer=warp_optimizer,
            scheduler=scheduler,
            step=3,
            epoch_index=1,
            epoch_step=2,
            history={"step": [1.0], "task_loss": [1.2], "consist_loss": [0.8], "ppl": [3.3]},
            args=_args(run_dir.parent),
            latest_checkpoint_path=str(checkpoint_dir),
            latest_metrics={"task_loss": 1.2, "consistency_loss": 0.8, "ppl": 3.3},
        )
        payload = _load_resume_bundle(resume_path, torch.device("cpu"))
        payload["torch_rng_state"] = payload["torch_rng_state"].tolist()

        restored = _restore_resume_state(
            payload,
            args=_args(run_dir.parent),
            model=model,
            base_optimizer=base_optimizer,
            warp_optimizer=warp_optimizer,
            scheduler=scheduler,
        )

        assert restored[0] == 3
        assert _normalize_rng_tensor(payload["torch_rng_state"]).dtype == torch.uint8


def test_ws29_partial_step_cap_detects_resumable_partial_runs():
    assert _partial_step_cap_reached(step=3, planned_total_steps=100, max_steps=3) is True
    assert _partial_step_cap_reached(step=6, planned_total_steps=100, max_steps=6) is True
    assert _partial_step_cap_reached(step=100, planned_total_steps=100, max_steps=None) is False
    assert _partial_step_cap_reached(step=100, planned_total_steps=100, max_steps=100) is False
