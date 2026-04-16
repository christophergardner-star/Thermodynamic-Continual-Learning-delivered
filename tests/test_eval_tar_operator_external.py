import json
from pathlib import Path

import eval_tar_operator_external as external_eval


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_external_pack_excludes_ws27_overlap(tmp_path: Path) -> None:
    repo_root = tmp_path
    ws26_path = repo_root / "dataset_artifacts" / "tar_master_dataset_ws26_merged_v1" / "tar_master_dataset_test.jsonl"
    ws27_path = repo_root / "dataset_artifacts" / "tar_master_dataset_ws27_branch_v1" / "tar_master_dataset.jsonl"
    probe_dir = repo_root / "eval_artifacts" / "tar_operator_eval_ws27r1_probe_v1"
    regression_dir = repo_root / "eval_artifacts" / "tar_operator_eval_ws27r1_ws26_regression_v1"
    adapter_dir = repo_root / "training_artifacts" / "ws27r2_qwen25_7b_refine_run1" / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    overlapping = {
        "example_id": "evidence_debt_judgement:overlap",
        "dedupe_key": "evidence_debt_judgement:overlap",
        "lineage_key": "lineage-overlap",
        "task_family": "evidence_debt_judgement",
        "task_name": "debt_to_gate",
        "source_kind": "evidence_debt_record",
        "input_context": {},
        "target": {
            "promotion_gate": "open",
            "recommended_state": "continue",
            "primary_gaps": ["falsification_gap"],
            "operator_language": "evidence_debt_present_but_not_blocking",
        },
        "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "{}"}],
        "provenance": {"content_hash": "hash-overlap"},
        "tags": [],
    }
    kept = {
        "example_id": "evidence_debt_judgement:keep",
        "dedupe_key": "evidence_debt_judgement:keep",
        "lineage_key": "lineage-keep",
        "task_family": "evidence_debt_judgement",
        "task_name": "debt_to_gate",
        "source_kind": "evidence_debt_record",
        "input_context": {},
        "target": {
            "promotion_gate": "open",
            "recommended_state": "continue",
            "primary_gaps": ["falsification_gap"],
            "operator_language": "evidence_debt_present_but_not_blocking",
        },
        "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "{}"}],
        "provenance": {"content_hash": "hash-keep"},
        "tags": [],
    }
    _write_jsonl(ws26_path, [overlapping, kept])
    _write_jsonl(ws27_path, [overlapping])
    _write_jsonl(probe_dir / "eval_core.jsonl", [{"lineage_key": "lineage-internal"}])
    _write_jsonl(regression_dir / "eval_core.jsonl", [])

    config = external_eval.ExternalEvalConfig(
        ws26_test_split=str(ws26_path.relative_to(repo_root)),
        ws27_dataset=str(ws27_path.relative_to(repo_root)),
        internal_eval_pack_dirs=[
            str(probe_dir.relative_to(repo_root)),
            str(regression_dir.relative_to(repo_root)),
        ],
        eval_pack_dir="eval_artifacts/tar_operator_eval_external_v1",
        output_dir="eval_artifacts/tar_operator_eval_runs/ws27r2_external_eval_v1",
        selection=external_eval.ExternalSelectionSettings(
            held_out_family_quotas={"evidence_debt_judgement": 1},
            include_literature_items=False,
        ),
        runtime=external_eval.ExternalRuntimeSettings(adapter_path=str(adapter_dir.relative_to(repo_root))),
    )

    manifest = external_eval.build_external_pack(config, repo_root=repo_root)
    items = external_eval._read_jsonl(repo_root / config.eval_pack_dir / "eval_items.jsonl")

    assert manifest["items"] == 1
    assert len(items) == 1
    assert items[0]["source_id"] == "evidence_debt_judgement:keep"


def test_independent_scoring_detects_overclaim() -> None:
    target = {
        "evidence_scope": "abstract_only",
        "claim_posture": "survey_and_benchmark_map",
        "promotion_safe": False,
        "recommended_action": "treat_as_literature_map_then_validate_benchmark",
        "benchmark_truth_status": "survey_plus_external_benchmark_reference",
    }
    honest = {
        "evidence_scope": "abstract_only",
        "claim_posture": "survey_and_benchmark_map",
        "promotion_safe": False,
        "recommended_action": "treat_as_literature_map_then_validate_benchmark",
        "benchmark_truth_status": "survey_plus_external_benchmark_reference",
    }
    overclaim = {
        "evidence_scope": "abstract_only",
        "claim_posture": "validated_local_proof",
        "promotion_safe": True,
        "recommended_action": "promote_immediately",
        "benchmark_truth_status": "validated",
    }

    honest_score, _ = external_eval._score_prediction("literature_grounded_reasoning", honest, target)
    overclaim_score, _ = external_eval._score_prediction("literature_grounded_reasoning", overclaim, target)

    assert honest_score == 1.0
    assert overclaim_score < 0.5
    assert external_eval._detect_overclaim("literature_grounded_reasoning", overclaim, target) is True


def test_resolve_runtime_base_model_source_fails_fast_on_missing_shards(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    cache_root = tmp_path / "hf-cache"
    snapshot = (
        cache_root
        / "hub"
        / "models--Qwen--Qwen2.5-7B-Instruct"
        / "snapshots"
        / "snapshot-1"
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    (cache_root / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "refs").mkdir(parents=True, exist_ok=True)
    (
        cache_root / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "refs" / "main"
    ).write_text("snapshot-1", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00004.safetensors",
                    "layer.1": "model-00002-of-00004.safetensors",
                    "layer.2": "model-00003-of-00004.safetensors",
                    "layer.3": "model-00004-of-00004.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00004.safetensors").write_text("x", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(cache_root))

    config = external_eval.ExternalEvalConfig(
        runtime=external_eval.ExternalRuntimeSettings(base_model="Qwen/Qwen2.5-7B-Instruct")
    )

    try:
        external_eval._resolve_runtime_base_model_source(config, repo_root=repo_root)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing shard files")

    assert "model-00002-of-00004.safetensors" in message
    assert "model-00003-of-00004.safetensors" in message
    assert "model-00004-of-00004.safetensors" in message
