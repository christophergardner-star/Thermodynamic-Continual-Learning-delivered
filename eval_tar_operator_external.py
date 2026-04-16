from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


DEFAULT_LITERATURE_ITEMS: list[dict[str, Any]] = [
    {
        "arxiv_id": "2407.12883",
        "paper_title": "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval",
        "paper_url": "https://arxiv.org/abs/2407.12883",
        "abstract_paraphrase": (
            "The paper introduces BRIGHT, a retrieval benchmark built around queries "
            "that need deeper reasoning than keyword matching, and reports that even "
            "strong retrievers underperform on it while explicit reasoning helps."
        ),
        "target": {
            "evidence_scope": "abstract_only",
            "claim_posture": "benchmark_signal_not_local_proof",
            "promotion_safe": False,
            "recommended_action": "replicate_on_local_retrieval_stack",
            "benchmark_truth_status": "external_benchmark_announced",
        },
    },
    {
        "arxiv_id": "2402.07630",
        "paper_title": "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering",
        "paper_url": "https://arxiv.org/abs/2402.07630",
        "abstract_paraphrase": (
            "The paper presents a graph RAG method for textual graphs, introduces a "
            "GraphQA benchmark, and reports stronger multi-domain graph QA results "
            "with reduced hallucination."
        ),
        "target": {
            "evidence_scope": "abstract_only",
            "claim_posture": "method_paper_requires_local_reproduction",
            "promotion_safe": False,
            "recommended_action": "benchmark_against_graph_tasks_before_adoption",
            "benchmark_truth_status": "paper_reports_external_results",
        },
    },
    {
        "arxiv_id": "2401.17585",
        "paper_title": "Propagation and Pitfalls: Reasoning-based Assessment of Knowledge Editing through Counterfactual Tasks",
        "paper_url": "https://arxiv.org/abs/2401.17585",
        "abstract_paraphrase": (
            "The paper introduces the ReCoE benchmark for reasoning-based "
            "counterfactual editing and reports that current editing methods struggle "
            "to propagate updates across connected facts."
        ),
        "target": {
            "evidence_scope": "abstract_only",
            "claim_posture": "risk_signal_from_external_benchmark",
            "promotion_safe": False,
            "recommended_action": "use_as_falsification_signal_not_as_proof",
            "benchmark_truth_status": "external_benchmark_announced",
        },
    },
    {
        "arxiv_id": "2407.06992",
        "paper_title": "Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective",
        "paper_url": "https://arxiv.org/abs/2407.06992",
        "abstract_paraphrase": (
            "The paper surveys robustness work in neural information retrieval and "
            "introduces BestIR as a heterogeneous robustness benchmark for adversarial "
            "and out-of-distribution evaluation."
        ),
        "target": {
            "evidence_scope": "abstract_only",
            "claim_posture": "survey_and_benchmark_map",
            "promotion_safe": False,
            "recommended_action": "treat_as_literature_map_then_validate_benchmark",
            "benchmark_truth_status": "survey_plus_external_benchmark_reference",
        },
    },
]


@dataclass
class ExternalSelectionSettings:
    held_out_family_quotas: dict[str, int] = field(
        default_factory=lambda: {
            "decision_rationale": 4,
            "endpoint_observability_diagnosis": 3,
            "portfolio_governance": 5,
        }
    )
    include_literature_items: bool = True
    max_items: int | None = None


@dataclass
class ExternalRuntimeSettings:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: str = "training_artifacts/ws27r2_qwen25_7b_refine_run1/final_adapter"
    served_model_name: str = "tar-operator-ws27r2-external"
    host: str = "127.0.0.1"
    port: int = 8827
    base_url: str | None = None
    spawn_local_endpoint: bool = True
    max_new_tokens: int = 192
    temperature: float = 0.0
    startup_timeout_s: int = 900
    request_timeout_s: int = 600


@dataclass
class ExternalEvalConfig:
    ws26_test_split: str = "dataset_artifacts/tar_master_dataset_ws26_merged_v1/tar_master_dataset_test.jsonl"
    ws27_dataset: str = "dataset_artifacts/tar_master_dataset_ws27_branch_v1/tar_master_dataset.jsonl"
    internal_eval_pack_dirs: list[str] = field(
        default_factory=lambda: [
            "eval_artifacts/tar_operator_eval_ws27r1_probe_v1",
            "eval_artifacts/tar_operator_eval_ws27r1_ws26_regression_v1",
        ]
    )
    eval_pack_dir: str = "eval_artifacts/tar_operator_eval_external_v1"
    output_dir: str = "eval_artifacts/tar_operator_eval_runs/ws27r2_external_eval_v1"
    eval_version: str = "tar-operator-eval-external-v1"
    selection: ExternalSelectionSettings = field(default_factory=ExternalSelectionSettings)
    runtime: ExternalRuntimeSettings = field(default_factory=ExternalRuntimeSettings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent external TAR operator evaluation.")
    parser.add_argument("--config", default="configs/tar_operator_eval_external_v1.json")
    parser.add_argument("--eval-pack-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--build-pack-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_dataclass_fields(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    field_names = {field_def.name for field_def in cls.__dataclass_fields__.values()}
    return {key: value for key, value in payload.items() if key in field_names}


def _resolve_path(raw_path: str, *, repo_root: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def load_config(config_path: Path, *, repo_root: Path, args: argparse.Namespace) -> ExternalEvalConfig:
    payload = _read_json(config_path)
    root_payload = _load_dataclass_fields(ExternalEvalConfig, payload)
    root_payload.pop("selection", None)
    root_payload.pop("runtime", None)
    config = ExternalEvalConfig(
        **root_payload,
        selection=ExternalSelectionSettings(
            **_load_dataclass_fields(ExternalSelectionSettings, payload.get("selection", {}))
        ),
        runtime=ExternalRuntimeSettings(
            **_load_dataclass_fields(ExternalRuntimeSettings, payload.get("runtime", {}))
        ),
    )
    if args.eval_pack_dir is not None:
        config.eval_pack_dir = args.eval_pack_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.adapter_path is not None:
        config.runtime.adapter_path = args.adapter_path
    if args.base_url is not None:
        config.runtime.base_url = args.base_url
    if args.port is not None:
        config.runtime.port = args.port
    if args.max_new_tokens is not None:
        config.runtime.max_new_tokens = args.max_new_tokens
    if args.max_items is not None:
        config.selection.max_items = args.max_items
    validate_config(config, repo_root=repo_root)
    return config


def validate_config(config: ExternalEvalConfig, *, repo_root: Path) -> None:
    ws26 = _resolve_path(config.ws26_test_split, repo_root=repo_root)
    ws27 = _resolve_path(config.ws27_dataset, repo_root=repo_root)
    if not ws26.exists():
        raise FileNotFoundError(f"WS26 test split not found: {ws26}")
    if not ws27.exists():
        raise FileNotFoundError(f"WS27 dataset not found: {ws27}")
    for raw in config.internal_eval_pack_dirs:
        pack_dir = _resolve_path(raw, repo_root=repo_root)
        if not pack_dir.exists():
            raise FileNotFoundError(f"Internal eval pack not found: {pack_dir}")
    adapter_path = _resolve_path(config.runtime.adapter_path, repo_root=repo_root)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")


def _huggingface_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface"


def _resolve_cached_model_snapshot(model_id: str) -> Path | None:
    cache_dir = _huggingface_cache_root() / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    ref_path = cache_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_name = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.exists():
            return snapshot_path
    candidates = [item for item in snapshots_dir.iterdir() if item.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.name)[-1]


def _resolve_base_model_dir(base_model: str, *, repo_root: Path) -> Path | None:
    candidate = _resolve_path(base_model, repo_root=repo_root)
    if candidate.exists():
        return candidate
    if "/" in base_model:
        return _resolve_cached_model_snapshot(base_model)
    return None


def _missing_model_shards(model_dir: Path) -> list[str]:
    direct_model = model_dir / "model.safetensors"
    if direct_model.exists():
        return []
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return ["model.safetensors.index.json"]
    payload = _read_json(index_path)
    weight_map = payload.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        return ["model.safetensors.index.json"]
    shard_names = sorted({str(name) for name in weight_map.values()})
    return [name for name in shard_names if not (model_dir / name).exists()]


def _resolve_runtime_base_model_source(config: ExternalEvalConfig, *, repo_root: Path) -> str:
    model_dir = _resolve_base_model_dir(config.runtime.base_model, repo_root=repo_root)
    if model_dir is None:
        raise FileNotFoundError(
            "Base model is not available locally. "
            f"Expected a local path or cached snapshot for {config.runtime.base_model!r}."
        )
    missing = _missing_model_shards(model_dir)
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Base model snapshot is incomplete. Missing files: "
            f"{joined}. Resolve the missing shards before running the external eval."
        )
    return str(model_dir)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _tokenize(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(value))


def _token_overlap_score(predicted: Any, target: Any) -> float:
    target_tokens = set(_tokenize(target))
    predicted_tokens = set(_tokenize(predicted))
    if not target_tokens:
        return 1.0 if not predicted_tokens else 0.0
    return len(target_tokens & predicted_tokens) / float(len(target_tokens))


def _exact_score(predicted: Any, target: Any) -> float:
    return 1.0 if predicted == target else 0.0


def _bool_score(predicted: Any, target: Any) -> float:
    return 1.0 if bool(predicted) is bool(target) else 0.0


def _numeric_closeness(predicted: Any, target: Any, *, tolerance: float = 0.25) -> float:
    try:
        pred = float(predicted)
        gold = float(target)
    except (TypeError, ValueError):
        return 0.0
    delta = abs(pred - gold)
    if delta >= tolerance:
        return 0.0
    return max(0.0, 1.0 - (delta / tolerance))


def _set_recall_score(predicted: Any, target: Any) -> float:
    target_set = {str(item) for item in (target or [])}
    predicted_set = {str(item) for item in (predicted or [])}
    if not target_set:
        return 1.0 if not predicted_set else 0.0
    return len(target_set & predicted_set) / float(len(target_set))


def _lookup_path(payload: dict[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor


def _compare_field(predicted: dict[str, Any], target: dict[str, Any], path: str, comparator: str) -> float:
    pred_value = _lookup_path(predicted, path)
    target_value = _lookup_path(target, path)
    if comparator == "exact":
        return _exact_score(pred_value, target_value)
    if comparator == "bool":
        return _bool_score(pred_value, target_value)
    if comparator == "token_overlap":
        return _token_overlap_score(pred_value, target_value)
    if comparator == "set_recall":
        return _set_recall_score(pred_value, target_value)
    if comparator == "numeric_closeness":
        return _numeric_closeness(pred_value, target_value)
    raise ValueError(f"Unsupported comparator: {comparator}")


RUBRICS: dict[str, list[tuple[str, str, float]]] = {
    "decision_rationale": [
        ("selected_action", "token_overlap", 0.7),
        ("confidence", "numeric_closeness", 0.3),
    ],
    "endpoint_observability_diagnosis": [
        ("diagnosis", "exact", 0.4),
        ("restart_recommended", "bool", 0.2),
        ("trust_policy", "exact", 0.2),
        ("inspect_paths", "set_recall", 0.2),
    ],
    "evidence_debt_judgement": [
        ("promotion_gate", "exact", 0.35),
        ("recommended_state", "exact", 0.25),
        ("primary_gaps", "set_recall", 0.2),
        ("operator_language", "exact", 0.2),
    ],
    "execution_diagnosis": [
        ("diagnosis", "exact", 0.4),
        ("recommended_next_step", "token_overlap", 0.4),
        ("blockers", "set_recall", 0.2),
    ],
    "portfolio_governance": [
        ("selected_project_id", "exact", 0.5),
        ("selected_action_id", "exact", 0.5),
    ],
    "portfolio_staleness_recovery": [
        ("recommended_operator_action", "exact", 0.4),
        ("resume_candidate", "bool", 0.2),
        ("closure_candidate", "bool", 0.2),
        ("staleness_level", "exact", 0.2),
    ],
    "prioritization": [
        ("recommended_state", "exact", 0.6),
        ("rationale", "token_overlap", 0.4),
    ],
    "problem_scoping": [
        ("next_action", "token_overlap", 0.4),
        ("benchmark_assessment.alignment", "exact", 0.2),
        ("benchmark_assessment.canonical_comparable", "bool", 0.2),
        ("reproducibility_risk.complete", "bool", 0.2),
    ],
    "literature_grounded_reasoning": [
        ("evidence_scope", "exact", 0.2),
        ("claim_posture", "exact", 0.2),
        ("promotion_safe", "bool", 0.2),
        ("recommended_action", "exact", 0.2),
        ("benchmark_truth_status", "exact", 0.2),
    ],
}


def _detect_overclaim(task_family: str, predicted: dict[str, Any], target: dict[str, Any]) -> bool:
    if target.get("promotion_safe") is False and predicted.get("promotion_safe") is True:
        return True
    if target.get("promotion_gate") == "blocked" and predicted.get("promotion_gate") == "open":
        return True
    if _lookup_path(target, "benchmark_assessment.canonical_comparable") is False and _lookup_path(
        predicted, "benchmark_assessment.canonical_comparable"
    ) is True:
        return True
    if target.get("benchmark_alignment") == "refused" and predicted.get("benchmark_alignment") in {
        "aligned",
        "canonical_ready",
    }:
        return True
    if task_family == "literature_grounded_reasoning":
        predicted_status = str(predicted.get("benchmark_truth_status") or "")
        if predicted_status in {"validated", "canonical_ready", "local_proof"}:
            return True
    return False


def _score_prediction(task_family: str, predicted: dict[str, Any], target: dict[str, Any]) -> tuple[float, dict[str, float]]:
    rubric = RUBRICS[task_family]
    subtotal = 0.0
    breakdown: dict[str, float] = {}
    for path, comparator, weight in rubric:
        field_score = _compare_field(predicted, target, path, comparator)
        breakdown[path] = field_score
        subtotal += field_score * weight
    return subtotal, breakdown


def _strip_json_fence(payload: str) -> str:
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(payload: str) -> dict[str, Any]:
    text = _strip_json_fence(payload)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("response did not contain a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("response JSON was not an object")
    return parsed


def _trim_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    trimmed = [dict(role=str(item.get("role", "")).strip(), content=str(item.get("content", ""))) for item in messages]
    while trimmed and trimmed[-1]["role"] == "assistant":
        trimmed.pop()
    return trimmed


def _iter_internal_eval_rows(pack_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(pack_dir.glob("eval_*.jsonl")):
        if path.name == "eval_portfolio.jsonl" and path.stat().st_size == 0:
            continue
        yield from _read_jsonl(path)


def _build_seen_sets(config: ExternalEvalConfig, *, repo_root: Path) -> dict[str, set[str]]:
    ws27_rows = _read_jsonl(_resolve_path(config.ws27_dataset, repo_root=repo_root))
    internal_eval_rows: list[dict[str, Any]] = []
    for raw in config.internal_eval_pack_dirs:
        internal_eval_rows.extend(_iter_internal_eval_rows(_resolve_path(raw, repo_root=repo_root)))
    return {
        "ws27_example_ids": {str(row.get("example_id")) for row in ws27_rows},
        "ws27_dedupe_keys": {str(row.get("dedupe_key")) for row in ws27_rows},
        "ws27_content_hashes": {
            str((row.get("provenance") or {}).get("content_hash"))
            for row in ws27_rows
            if (row.get("provenance") or {}).get("content_hash")
        },
        "internal_eval_lineages": {str(row.get("lineage_key")) for row in internal_eval_rows if row.get("lineage_key")},
    }


def _select_held_out_rows(config: ExternalEvalConfig, *, repo_root: Path) -> list[dict[str, Any]]:
    seen = _build_seen_sets(config, repo_root=repo_root)
    ws26_rows = _read_jsonl(_resolve_path(config.ws26_test_split, repo_root=repo_root))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ws26_rows:
        provenance = row.get("provenance") or {}
        if str(row.get("example_id")) in seen["ws27_example_ids"]:
            continue
        if str(row.get("dedupe_key")) in seen["ws27_dedupe_keys"]:
            continue
        content_hash = provenance.get("content_hash")
        if content_hash and str(content_hash) in seen["ws27_content_hashes"]:
            continue
        if str(row.get("lineage_key")) in seen["internal_eval_lineages"]:
            continue
        family = str(row.get("task_family") or "")
        if family not in config.selection.held_out_family_quotas:
            continue
        by_family[family].append(row)
    selected: list[dict[str, Any]] = []
    for family, quota in sorted(config.selection.held_out_family_quotas.items()):
        rows = sorted(by_family.get(family, []), key=lambda item: str(item.get("example_id")))
        if len(rows) < quota:
            raise ValueError(f"Not enough held-out rows for {family}: wanted {quota}, found {len(rows)}")
        selected.extend(rows[:quota])
    return selected


def _build_literature_items(config: ExternalEvalConfig, *, repo_root: Path) -> list[dict[str, Any]]:
    if not config.selection.include_literature_items:
        return []
    ws27_text = _resolve_path(config.ws27_dataset, repo_root=repo_root).read_text(encoding="utf-8").lower()
    items: list[dict[str, Any]] = []
    for item in DEFAULT_LITERATURE_ITEMS:
        overlap = str(item["paper_title"]).lower() in ws27_text
        items.append(
            {
                "item_id": f"{config.eval_version}:literature_grounded_reasoning:{item['arxiv_id']}",
                "task_family": "literature_grounded_reasoning",
                "task_name": "abstract_to_operator_posture",
                "source_kind": "arxiv_abstract",
                "source_id": item["arxiv_id"],
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are TAR, a disciplined research operator. Separate measured "
                            "results, inferences, hypotheses, blockers, and next actions. "
                            "Stay honest when evidence is literature-only. Return only a JSON object."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Task family: literature_grounded_reasoning\n"
                            "Task: abstract_to_operator_posture\n"
                            "Use the TAR operator contract. Return only a concise JSON object aligned to the target.\n\n"
                            "Input state:\n"
                            + json.dumps(
                                {
                                    "paper_title": item["paper_title"],
                                    "arxiv_id": item["arxiv_id"],
                                    "paper_url": item["paper_url"],
                                    "evidence_scope": "abstract_only",
                                    "abstract_paraphrase": item["abstract_paraphrase"],
                                },
                                indent=2,
                            )
                        ),
                    },
                ],
                "input_context": {
                    "paper_title": item["paper_title"],
                    "arxiv_id": item["arxiv_id"],
                    "paper_url": item["paper_url"],
                    "evidence_scope": "abstract_only",
                    "abstract_paraphrase": item["abstract_paraphrase"],
                },
                "target": item["target"],
                "provenance": {
                    "source_type": "arxiv_abstract_paraphrase",
                    "paper_url": item["paper_url"],
                    "paper_title": item["paper_title"],
                    "title_present_in_ws27_branch": overlap,
                    "selection_reason": "new_literature_family_not_present_in_ws27_branch_v1",
                },
                "tags": ["external_eval", "literature_grounded_reasoning", "held_out"],
            }
        )
    return items


def build_external_pack(config: ExternalEvalConfig, *, repo_root: Path) -> dict[str, Any]:
    pack_dir = _resolve_path(config.eval_pack_dir, repo_root=repo_root)
    pack_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = [
        {
            "item_id": f"{config.eval_version}:{row['task_family']}:{row['example_id']}",
            "task_family": row["task_family"],
            "task_name": row["task_name"],
            "source_kind": row["source_kind"],
            "source_id": row["example_id"],
            "messages": _trim_messages(row["messages"]),
            "input_context": row["input_context"],
            "target": row["target"],
            "provenance": {
                **(row.get("provenance") or {}),
                "selection_reason": "ws26_test_holdout_excluding_ws27_branch_and_internal_eval",
            },
            "tags": list(row.get("tags") or []) + ["external_eval", "held_out"],
        }
        for row in _select_held_out_rows(config, repo_root=repo_root)
    ]
    selected_rows.extend(_build_literature_items(config, repo_root=repo_root))
    if config.selection.max_items is not None:
        selected_rows = selected_rows[: config.selection.max_items]
    items_path = pack_dir / "eval_items.jsonl"
    _write_jsonl(items_path, selected_rows)
    manifest_payload = {
        "eval_version": config.eval_version,
        "sealed_at": _utc_now_iso(),
        "items": len(selected_rows),
        "task_families": dict(sorted(Counter(str(item["task_family"]) for item in selected_rows).items())),
        "source_kinds": dict(sorted(Counter(str(item["source_kind"]) for item in selected_rows).items())),
        "selection": {
            "held_out_family_quotas": config.selection.held_out_family_quotas,
            "include_literature_items": config.selection.include_literature_items,
            "max_items": config.selection.max_items,
        },
        "source_dataset": {
            "ws26_test_split": str(_resolve_path(config.ws26_test_split, repo_root=repo_root)),
            "ws26_test_sha256": _sha256_file(_resolve_path(config.ws26_test_split, repo_root=repo_root)),
            "ws27_dataset": str(_resolve_path(config.ws27_dataset, repo_root=repo_root)),
            "ws27_dataset_sha256": _sha256_file(_resolve_path(config.ws27_dataset, repo_root=repo_root)),
        },
        "internal_eval_exclusions": [
            {"path": str(_resolve_path(raw, repo_root=repo_root))}
            for raw in config.internal_eval_pack_dirs
        ],
        "literature_sources": [
            {
                "arxiv_id": item["source_id"],
                "paper_title": item["input_context"]["paper_title"],
                "paper_url": item["input_context"]["paper_url"],
                "title_present_in_ws27_branch": bool((item.get("provenance") or {}).get("title_present_in_ws27_branch")),
            }
            for item in selected_rows
            if item["task_family"] == "literature_grounded_reasoning"
        ],
        "files": {
            "eval_items": {
                "path": "eval_items.jsonl",
                "sha256": _sha256_file(items_path),
                "size_bytes": items_path.stat().st_size,
                "records": len(selected_rows),
            }
        },
    }
    manifest_payload["manifest_sha256"] = _sha256_bytes(_canonical_json(manifest_payload).encode("utf-8"))
    _write_json(pack_dir / "eval_manifest.json", manifest_payload)
    return manifest_payload


class _EndpointClient:
    def __init__(self, *, base_url: str, model_name: str, request_timeout_s: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout_s = request_timeout_s

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.base_url.removesuffix('/v1')}/health", timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(self, messages: list[dict[str, str]], *, max_new_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str((((payload.get("choices") or [{}])[0].get("message") or {}).get("content")) or "")


def _wait_for_health(client: _EndpointClient, *, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        try:
            payload = client.health()
            if payload.get("ok") is True:
                return
            last_error = str(payload)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"Endpoint did not become healthy: {last_error}")


class _LocalEndpointProcess:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)


def _spawn_local_endpoint(config: ExternalEvalConfig, *, repo_root: Path, output_dir: Path) -> tuple[_EndpointClient, _LocalEndpointProcess]:
    python_exe = repo_root / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)
    resolved_model_source = _resolve_runtime_base_model_source(config, repo_root=repo_root)
    stdout_handle = (output_dir / "endpoint_stdout.log").open("w", encoding="utf-8")
    stderr_handle = (output_dir / "endpoint_stderr.log").open("w", encoding="utf-8")
    command = [
        str(python_exe),
        "-u",
        str(repo_root / "serve_local.py"),
        "--backend",
        "transformers",
        "--model",
        resolved_model_source,
        "--adapter-path",
        str(_resolve_path(config.runtime.adapter_path, repo_root=repo_root)),
        "--host",
        config.runtime.host,
        "--port",
        str(config.runtime.port),
        "--role",
        "assistant",
        "--served-model-name",
        config.runtime.served_model_name,
    ]
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        env=env,
    )
    client = _EndpointClient(
        base_url=f"http://{config.runtime.host}:{config.runtime.port}/v1",
        model_name=config.runtime.served_model_name,
        request_timeout_s=config.runtime.request_timeout_s,
    )
    _wait_for_health(client, timeout_s=config.runtime.startup_timeout_s)
    return client, _LocalEndpointProcess(process)


def evaluate_pack(pack_dir: Path, *, predictor: Callable[[list[dict[str, str]]], str], model_name: str, manifest_sha256: str) -> dict[str, Any]:
    rows = _read_jsonl(pack_dir / "eval_items.jsonl")
    results: list[dict[str, Any]] = []
    family_scores: dict[str, list[float]] = defaultdict(list)
    parse_errors = 0
    overclaims = 0
    decision_successes = 0
    for row in rows:
        raw_response = predictor(row["messages"])
        parse_error = False
        parsed_response: dict[str, Any] | None = None
        score = 0.0
        breakdown: dict[str, float] = {}
        error_bucket = "none"
        try:
            parsed_response = _extract_json_object(raw_response)
            score, breakdown = _score_prediction(row["task_family"], parsed_response, row["target"])
            if _detect_overclaim(row["task_family"], parsed_response, row["target"]):
                overclaims += 1
                error_bucket = "overclaim"
            elif score < 0.75:
                error_bucket = "mismatch"
            else:
                decision_successes += 1
        except Exception as exc:  # noqa: BLE001
            parse_error = True
            parse_errors += 1
            error_bucket = f"parse_error:{type(exc).__name__}"
        family_scores[row["task_family"]].append(score)
        results.append(
            {
                "item_id": row["item_id"],
                "task_family": row["task_family"],
                "task_name": row["task_name"],
                "source_kind": row["source_kind"],
                "score": score,
                "parse_error": parse_error,
                "overclaim": error_bucket == "overclaim",
                "error_bucket": error_bucket,
                "field_scores": breakdown,
                "response_text": raw_response,
                "parsed_response": parsed_response,
                "target": row["target"],
            }
        )
    total = len(rows)
    return {
        "overall": {
            "model_name": model_name,
            "item_count": total,
            "mean_score": (sum(item["score"] for item in results) / total) if total else 0.0,
            "decision_accuracy": (decision_successes / total) if total else 0.0,
            "parse_error_rate": (parse_errors / total) if total else 0.0,
            "overclaim_rate": (overclaims / total) if total else 0.0,
            "manifest_sha256": manifest_sha256,
        },
        "family_breakdown": {
            family: {"count": len(scores), "mean_score": sum(scores) / len(scores) if scores else 0.0}
            for family, scores in sorted(family_scores.items())
        },
        "results": results,
    }


def run_external_eval(config: ExternalEvalConfig, *, repo_root: Path) -> Path:
    pack_dir = _resolve_path(config.eval_pack_dir, repo_root=repo_root)
    output_dir = _resolve_path(config.output_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = str(_read_json(pack_dir / "eval_manifest.json").get("manifest_sha256") or "")
    resolved_model_source: str | None = None
    endpoint_process: Optional[_LocalEndpointProcess] = None
    if config.runtime.base_url:
        client = _EndpointClient(
            base_url=config.runtime.base_url,
            model_name=config.runtime.served_model_name,
            request_timeout_s=config.runtime.request_timeout_s,
        )
    else:
        resolved_model_source = _resolve_runtime_base_model_source(config, repo_root=repo_root)
        client, endpoint_process = _spawn_local_endpoint(config, repo_root=repo_root, output_dir=output_dir)
    started_at = _utc_now_iso()
    try:
        evaluation = evaluate_pack(
            pack_dir,
            predictor=lambda messages: client.complete(
                messages,
                max_new_tokens=config.runtime.max_new_tokens,
                temperature=config.runtime.temperature,
            ),
            model_name=config.runtime.served_model_name,
            manifest_sha256=manifest_sha256,
        )
    finally:
        if endpoint_process is not None:
            endpoint_process.stop()
    results_payload = {
        "eval_version": config.eval_version,
        "started_at": started_at,
        "completed_at": _utc_now_iso(),
        "runtime": {
            "base_model": config.runtime.base_model,
            "resolved_base_model_source": resolved_model_source,
            "adapter_path": str(_resolve_path(config.runtime.adapter_path, repo_root=repo_root)),
            "served_model_name": config.runtime.served_model_name,
            "base_url": config.runtime.base_url or f"http://{config.runtime.host}:{config.runtime.port}/v1",
            "spawn_local_endpoint": bool(not config.runtime.base_url),
            "max_new_tokens": config.runtime.max_new_tokens,
            "temperature": config.runtime.temperature,
        },
        **evaluation,
    }
    _write_json(output_dir / "results.json", results_payload)
    _write_jsonl(output_dir / "predictions.jsonl", evaluation["results"])
    return output_dir


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    args = parse_args()
    config = load_config(Path(args.config), repo_root=repo_root, args=args)
    if not args.run_only:
        build_external_pack(config, repo_root=repo_root)
    if args.build_pack_only:
        return 0
    run_external_eval(config, repo_root=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
