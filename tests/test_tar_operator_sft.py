import json
from pathlib import Path

import pytest

from train_tar_operator_sft import (
    TrainingRunConfig,
    build_run_manifest,
    resolve_continuation_adapter,
    resolve_dataset_bundle,
    resolve_model_source,
    validate_training_config,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _seed_dataset(root: Path) -> Path:
    dataset_dir = root / "dataset_artifacts" / "seed"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_version": "tar-master-v1",
        "records": 3,
        "splits": {"train": 2, "validation": 1, "test": 0},
        "task_families": {"problem_scoping": 3},
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    sample = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": "{\"ok\": true}"},
        ]
    }
    _write_jsonl(dataset_dir / "tar_master_dataset_train.jsonl", [sample, sample])
    _write_jsonl(dataset_dir / "tar_master_dataset_validation.jsonl", [sample])
    _write_jsonl(dataset_dir / "tar_master_dataset_test.jsonl", [])
    return dataset_dir


def _seed_adapter(root: Path, *, base_model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> Path:
    adapter_dir = root / "training_artifacts" / "seed_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model_name,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"seed")
    (adapter_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return adapter_dir


def test_validate_training_config_refuses_output_inside_dataset_artifacts(tmp_path: Path):
    config = TrainingRunConfig(output_dir="dataset_artifacts/unsafe-output")
    with pytest.raises(ValueError, match="dataset_artifacts"):
        validate_training_config(config, repo_root=tmp_path)


def test_validate_training_config_refuses_continuation_adapter_outside_repo(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside"
    adapter_dir = _seed_adapter(outside)
    config = TrainingRunConfig(
        output_dir="training_artifacts/run1",
        resume_adapter_path=str(adapter_dir),
    )
    with pytest.raises(ValueError, match="outside the repo root"):
        validate_training_config(config, repo_root=repo_root)


def test_resolve_model_source_refuses_unverified_remote_model(tmp_path: Path):
    config = TrainingRunConfig(model_name_or_path="org/private-model")
    with pytest.raises(ValueError, match="approved list"):
        resolve_model_source(config, repo_root=tmp_path)


def test_resolve_dataset_bundle_refuses_external_dataset_without_opt_in(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    dataset_dir = tmp_path / "outside"
    seeded = _seed_dataset(dataset_dir)
    config = TrainingRunConfig(dataset_dir=str(seeded))
    with pytest.raises(ValueError, match="outside the repo root"):
        resolve_dataset_bundle(config, repo_root=repo_root)


def test_build_run_manifest_captures_dataset_hashes(tmp_path: Path):
    repo_root = tmp_path
    dataset_dir = _seed_dataset(repo_root)
    adapter_dir = _seed_adapter(repo_root)
    config = TrainingRunConfig(
        dataset_dir=str(dataset_dir),
        output_dir="training_artifacts/run1",
        resume_adapter_path=str(adapter_dir),
    )
    validate_training_config(config, repo_root=repo_root)
    bundle = resolve_dataset_bundle(config, repo_root=repo_root)
    continuation = resolve_continuation_adapter(
        config,
        repo_root=repo_root,
        resolved_model="Qwen/Qwen2.5-7B-Instruct",
    )
    manifest = build_run_manifest(
        config,
        repo_root=repo_root,
        resolved_model="Qwen/Qwen2.5-7B-Instruct",
        dataset_bundle=bundle,
        continuation_adapter=continuation,
    )
    assert manifest["security_posture"]["trust_remote_code"] is False
    assert manifest["dataset"]["hashes"]["train"]["records"] == 2
    assert len(manifest["dataset"]["hashes"]["manifest"]["sha256"]) == 64
    assert manifest["continuation_adapter"]["path"] == str(adapter_dir)
    assert "adapter_model.safetensors" in manifest["continuation_adapter"]["hashes"]


def test_resolve_continuation_adapter_refuses_mismatched_lora_config(tmp_path: Path):
    repo_root = tmp_path
    adapter_dir = _seed_adapter(repo_root)
    config = TrainingRunConfig(
        output_dir="training_artifacts/run1",
        resume_adapter_path=str(adapter_dir),
    )
    config.lora.rank = 16
    with pytest.raises(ValueError, match="rank does not match"):
        resolve_continuation_adapter(
            config,
            repo_root=repo_root,
            resolved_model="Qwen/Qwen2.5-7B-Instruct",
        )
