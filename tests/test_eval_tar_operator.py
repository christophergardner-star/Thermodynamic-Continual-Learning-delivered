import json
from pathlib import Path

import eval_tar_operator


def test_create_predictor_supports_local_asc_checkpoint(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset_artifacts" / "seed"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = tmp_path / "training_artifacts" / "ws27_asc_probe" / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "asc_config.json").write_text(
        json.dumps({"base_model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"}),
        encoding="utf-8",
    )

    config = eval_tar_operator.EvalConfig(
        dataset_dir=str(dataset_dir),
        eval_pack_dir="eval_artifacts/tar_operator_eval_ws27_probe_v1",
        output_dir="eval_artifacts/tar_operator_eval_runs/ws27_asc_probe_eval",
        runtime=eval_tar_operator.EvalRuntimeSettings(
            predictor_type="asc_causal_lm",
            model_name_or_path=str(checkpoint_dir),
            tokenizer_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        ),
    )
    eval_tar_operator.validate_eval_config(config, repo_root=tmp_path)

    captured: dict[str, object] = {}

    class _FakeASCPredictor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(eval_tar_operator, "ASCCausalLMPredictor", _FakeASCPredictor)

    predictor = eval_tar_operator.create_predictor(config, repo_root=tmp_path)

    assert isinstance(predictor, _FakeASCPredictor)
    assert captured["model_name_or_path"] == str(checkpoint_dir.resolve())
    assert captured["tokenizer_name_or_path"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
