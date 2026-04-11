from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from train_tar_operator_sft import APPROVED_REMOTE_MODELS
from tar_lab.eval_harness import (
    GoldPredictor,
    HFCausalLMPredictor,
    HeuristicPredictor,
    build_eval_pack,
    evaluate_eval_pack,
)


@dataclass
class EvalSecuritySettings:
    allow_external_dataset: bool = False
    allow_unverified_model: bool = False
    local_files_only: bool = False
    trust_remote_code: bool = False


@dataclass
class EvalSelectionSettings:
    max_examples_per_family: int | None = None
    family_quotas: dict[str, int] = field(default_factory=dict)


@dataclass
class EvalRuntimeSettings:
    predictor_type: str = "heuristic"
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: str | None = None
    max_new_tokens: int = 384
    temperature: float = 0.0
    top_p: float = 1.0
    use_4bit: bool = False
    bf16: bool = True
    dry_run: bool = False


@dataclass
class EvalConfig:
    dataset_dir: str = "dataset_artifacts/tar_master_dataset_ws23_v1"
    test_split_file: str = "tar_master_dataset_test.jsonl"
    eval_pack_dir: str = "eval_artifacts/tar_operator_eval_ws24_v1"
    output_dir: str = "eval_artifacts/tar_operator_eval_runs/ws24_heuristic"
    eval_version: str = "tar-operator-eval-ws24-v1"
    selection: EvalSelectionSettings = field(default_factory=EvalSelectionSettings)
    security: EvalSecuritySettings = field(default_factory=EvalSecuritySettings)
    runtime: EvalRuntimeSettings = field(default_factory=EvalRuntimeSettings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WS24 TAR/TCL operator evaluation harness.")
    parser.add_argument("--config", default="configs/tar_operator_eval_ws24_v1.json")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--eval-pack-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-version", default=None)
    parser.add_argument("--predictor", default=None, choices=("heuristic", "gold", "hf_causal_lm"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-examples-per-family", type=int, default=None)
    parser.add_argument("--build-pack-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-external-dataset", action="store_true")
    parser.add_argument("--allow-unverified-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_user_path(raw_path: str | None, *, base_dir: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_dataclass_fields(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    field_names = {field_def.name for field_def in cls.__dataclass_fields__.values()}
    return {key: value for key, value in payload.items() if key in field_names}


def load_eval_config(config_path: Path, *, repo_root: Path, args: argparse.Namespace) -> EvalConfig:
    payload = _read_json(config_path)
    root_payload = _load_dataclass_fields(EvalConfig, payload)
    root_payload.pop("selection", None)
    root_payload.pop("security", None)
    root_payload.pop("runtime", None)
    config = EvalConfig(
        **root_payload,
        selection=EvalSelectionSettings(
            **_load_dataclass_fields(EvalSelectionSettings, payload.get("selection", {}))
        ),
        security=EvalSecuritySettings(
            **_load_dataclass_fields(EvalSecuritySettings, payload.get("security", {}))
        ),
        runtime=EvalRuntimeSettings(
            **_load_dataclass_fields(EvalRuntimeSettings, payload.get("runtime", {}))
        ),
    )
    if args.dataset_dir is not None:
        config.dataset_dir = args.dataset_dir
    if args.eval_pack_dir is not None:
        config.eval_pack_dir = args.eval_pack_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.eval_version is not None:
        config.eval_version = args.eval_version
    if args.predictor is not None:
        config.runtime.predictor_type = args.predictor
    if args.model is not None:
        config.runtime.model_name_or_path = args.model
    if args.adapter_path is not None:
        config.runtime.adapter_path = args.adapter_path
    if args.max_new_tokens is not None:
        config.runtime.max_new_tokens = args.max_new_tokens
    if args.max_examples_per_family is not None:
        config.selection.max_examples_per_family = args.max_examples_per_family
    if args.local_files_only:
        config.security.local_files_only = True
    if args.allow_external_dataset:
        config.security.allow_external_dataset = True
    if args.allow_unverified_model:
        config.security.allow_unverified_model = True
    if args.dry_run:
        config.runtime.dry_run = True
    validate_eval_config(config, repo_root=repo_root)
    return config


def validate_eval_config(config: EvalConfig, *, repo_root: Path) -> None:
    if config.security.trust_remote_code:
        raise ValueError("trust_remote_code must remain false for TAR operator evaluation.")
    if config.runtime.predictor_type not in {"heuristic", "gold", "hf_causal_lm"}:
        raise ValueError("Unsupported predictor_type.")
    dataset_dir = _resolve_user_path(config.dataset_dir, base_dir=repo_root)
    if dataset_dir is None or not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {config.dataset_dir}")
    if (
        not config.security.allow_external_dataset
        and not _is_relative_to(dataset_dir, repo_root)
    ):
        raise ValueError("Refusing dataset outside the repo root without opt-in.")
    eval_pack_dir = _resolve_user_path(config.eval_pack_dir, base_dir=repo_root)
    output_dir = _resolve_user_path(config.output_dir, base_dir=repo_root)
    if eval_pack_dir is None or output_dir is None:
        raise ValueError("eval_pack_dir and output_dir must be set.")
    if _is_relative_to(eval_pack_dir, repo_root / "dataset_artifacts"):
        raise ValueError("Refusing to write eval pack inside dataset_artifacts.")
    if _is_relative_to(output_dir, repo_root / "dataset_artifacts"):
        raise ValueError("Refusing to write eval results inside dataset_artifacts.")


def resolve_model_source(config: EvalConfig, *, repo_root: Path) -> str | None:
    if config.runtime.predictor_type != "hf_causal_lm":
        return None
    maybe_path = _resolve_user_path(config.runtime.model_name_or_path, base_dir=repo_root)
    if maybe_path is not None and maybe_path.exists():
        return str(maybe_path)
    if (
        not config.security.allow_unverified_model
        and config.runtime.model_name_or_path not in APPROVED_REMOTE_MODELS
    ):
        raise ValueError("Remote base model is not on the approved list.")
    return config.runtime.model_name_or_path


def create_predictor(config: EvalConfig, *, repo_root: Path):
    predictor_type = config.runtime.predictor_type
    if predictor_type == "gold":
        return GoldPredictor()
    if predictor_type == "heuristic":
        return HeuristicPredictor()
    resolved_model = resolve_model_source(config, repo_root=repo_root)
    adapter_path = None
    if config.runtime.adapter_path:
        adapter_path = str(
            _resolve_user_path(config.runtime.adapter_path, base_dir=repo_root)
        )
    return HFCausalLMPredictor(
        model_name_or_path=str(resolved_model),
        adapter_path=adapter_path,
        local_files_only=config.security.local_files_only,
        max_new_tokens=config.runtime.max_new_tokens,
        temperature=config.runtime.temperature,
        top_p=config.runtime.top_p,
        use_4bit=config.runtime.use_4bit,
        bf16=config.runtime.bf16,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    args = parse_args()
    config_path = _resolve_user_path(args.config, base_dir=repo_root)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Eval config not found: {args.config}")
    config = load_eval_config(config_path, repo_root=repo_root, args=args)

    dataset_dir = _resolve_user_path(config.dataset_dir, base_dir=repo_root)
    eval_pack_dir = _resolve_user_path(config.eval_pack_dir, base_dir=repo_root)
    output_dir = _resolve_user_path(config.output_dir, base_dir=repo_root)
    assert dataset_dir is not None and eval_pack_dir is not None and output_dir is not None

    summary: dict[str, Any] = {
        "eval_version": config.eval_version,
        "dataset_dir": str(dataset_dir),
        "eval_pack_dir": str(eval_pack_dir),
        "output_dir": str(output_dir),
        "predictor_type": config.runtime.predictor_type,
    }

    if not args.run_only:
        pack_manifest = build_eval_pack(
            dataset_dir=dataset_dir,
            eval_pack_dir=eval_pack_dir,
            eval_version=config.eval_version,
            test_split_file=config.test_split_file,
            family_quotas=config.selection.family_quotas,
            max_examples_per_family=config.selection.max_examples_per_family,
        )
        summary["eval_pack"] = {
            "items": pack_manifest["items"],
            "task_families": pack_manifest["task_families"],
            "lineage_count": pack_manifest["lineage_count"],
        }

    if args.build_pack_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if config.runtime.dry_run:
        summary["dry_run"] = True
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    predictor = create_predictor(config, repo_root=repo_root)
    run_summary = evaluate_eval_pack(
        eval_pack_dir=eval_pack_dir,
        output_dir=output_dir,
        predictor=predictor,
        max_items=args.max_items,
    )
    summary["run"] = run_summary["overall"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
