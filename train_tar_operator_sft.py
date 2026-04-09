from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


APPROVED_REMOTE_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
}

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class SecuritySettings:
    allow_external_dataset: bool = False
    allow_unverified_model: bool = False
    local_files_only: bool = False
    trust_remote_code: bool = False
    push_to_hub: bool = False
    report_to: list[str] = field(default_factory=list)


@dataclass
class LoraSettings:
    enabled: bool = True
    use_qlora: bool = False
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES))


@dataclass
class TrainingRunConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_dir: str = "dataset_artifacts/tar_master_dataset_merged_v1"
    train_split_file: str = "tar_master_dataset_train.jsonl"
    validation_split_file: str = "tar_master_dataset_validation.jsonl"
    test_split_file: str = "tar_master_dataset_test.jsonl"
    output_dir: str = "training_artifacts/tar_operator_qwen25_7b_lora"
    seed: int = 42
    max_seq_length: int = 2048
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    assistant_only_loss: bool = True
    pad_to_multiple_of: int = 8
    dry_run: bool = False
    lora: LoraSettings = field(default_factory=LoraSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)


@dataclass
class DatasetFileFingerprint:
    path: str
    sha256: str
    records: int


@dataclass
class DatasetBundle:
    dataset_dir: Path
    manifest: Path
    train_file: Path
    validation_file: Path | None
    test_file: Path | None
    manifest_payload: dict[str, Any]
    files: dict[str, DatasetFileFingerprint]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure TAR operator SFT trainer.")
    parser.add_argument(
        "--config",
        default="configs/tar_operator_qwen25_7b_lora.json",
        help="Path to the training config JSON.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-external-dataset", action="store_true")
    parser.add_argument("--allow-unverified-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_user_path(raw_path: str | None, *, base_dir: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_jsonl_records(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _require_no_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked path for secure dataset handling: {path}")


def _load_dataclass_fields(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    field_names = {field_def.name for field_def in cls.__dataclass_fields__.values()}
    return {key: value for key, value in payload.items() if key in field_names}


def load_training_config(
    config_path: Path,
    *,
    repo_root: Path,
    args: argparse.Namespace | None = None,
) -> TrainingRunConfig:
    payload = _read_json(config_path)
    root_payload = _load_dataclass_fields(TrainingRunConfig, payload)
    root_payload.pop("lora", None)
    root_payload.pop("security", None)
    config = TrainingRunConfig(
        **root_payload,
        lora=LoraSettings(**_load_dataclass_fields(LoraSettings, payload.get("lora", {}))),
        security=SecuritySettings(
            **_load_dataclass_fields(SecuritySettings, payload.get("security", {}))
        ),
    )

    if args is not None:
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        if args.dataset_dir is not None:
            config.dataset_dir = args.dataset_dir
        if args.model is not None:
            config.model_name_or_path = args.model
        if args.max_steps is not None:
            config.max_steps = args.max_steps
        if args.num_train_epochs is not None:
            config.num_train_epochs = args.num_train_epochs
        if args.max_seq_length is not None:
            config.max_seq_length = args.max_seq_length
        if args.per_device_train_batch_size is not None:
            config.per_device_train_batch_size = args.per_device_train_batch_size
        if args.per_device_eval_batch_size is not None:
            config.per_device_eval_batch_size = args.per_device_eval_batch_size
        if args.gradient_accumulation_steps is not None:
            config.gradient_accumulation_steps = args.gradient_accumulation_steps
        if args.learning_rate is not None:
            config.learning_rate = args.learning_rate
        if args.use_qlora:
            config.lora.use_qlora = True
        if args.local_files_only:
            config.security.local_files_only = True
        if args.allow_external_dataset:
            config.security.allow_external_dataset = True
        if args.allow_unverified_model:
            config.security.allow_unverified_model = True
        if args.dry_run:
            config.dry_run = True

    validate_training_config(config, repo_root=repo_root)
    return config


def validate_training_config(config: TrainingRunConfig, *, repo_root: Path) -> None:
    if config.security.trust_remote_code:
        raise ValueError("trust_remote_code must remain false for TAR operator training.")
    if config.security.push_to_hub:
        raise ValueError("push_to_hub must remain false for TAR operator training.")
    if config.security.report_to:
        raise ValueError("report_to must remain empty for TAR operator training.")
    if config.max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive.")
    if config.per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive.")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not config.lora.enabled:
        raise ValueError("This trainer requires LoRA or QLoRA; full fine-tuning is refused.")
    if config.lora.rank <= 0 or config.lora.alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if not config.lora.target_modules:
        raise ValueError("LoRA target_modules cannot be empty.")

    output_dir = _resolve_user_path(config.output_dir, base_dir=repo_root)
    if output_dir is None:
        raise ValueError("output_dir must be set.")
    if _is_relative_to(output_dir, repo_root / "dataset_artifacts"):
        raise ValueError("Refusing to write training output inside dataset_artifacts.")


def resolve_model_source(config: TrainingRunConfig, *, repo_root: Path) -> str:
    maybe_path = _resolve_user_path(config.model_name_or_path, base_dir=repo_root)
    if maybe_path is not None and maybe_path.exists():
        _require_no_symlink(maybe_path)
        return str(maybe_path)
    if (
        not config.security.allow_unverified_model
        and config.model_name_or_path not in APPROVED_REMOTE_MODELS
    ):
        raise ValueError(
            "Remote base model is not on the approved list. "
            "Use --allow-unverified-model only if you intentionally reviewed it."
        )
    return config.model_name_or_path


def resolve_dataset_bundle(config: TrainingRunConfig, *, repo_root: Path) -> DatasetBundle:
    dataset_dir = _resolve_user_path(config.dataset_dir, base_dir=repo_root)
    if dataset_dir is None or not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {config.dataset_dir}")
    _require_no_symlink(dataset_dir)
    if not config.security.allow_external_dataset and not _is_relative_to(dataset_dir, repo_root):
        raise ValueError(
            "Refusing dataset outside the repo root without --allow-external-dataset."
        )

    manifest_path = dataset_dir / "manifest.json"
    train_file = dataset_dir / config.train_split_file
    validation_file = dataset_dir / config.validation_split_file
    test_file = dataset_dir / config.test_split_file
    for path in (manifest_path, train_file):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset artifact missing: {path}")
        _require_no_symlink(path)
    if validation_file.exists():
        _require_no_symlink(validation_file)
    else:
        validation_file = None
    if test_file.exists():
        _require_no_symlink(test_file)
    else:
        test_file = None

    manifest_payload = _read_json(manifest_path)
    files = {
        "manifest": DatasetFileFingerprint(
            path=str(manifest_path),
            sha256=_sha256_file(manifest_path),
            records=1,
        ),
        "train": DatasetFileFingerprint(
            path=str(train_file),
            sha256=_sha256_file(train_file),
            records=_count_jsonl_records(train_file),
        ),
    }
    if validation_file is not None:
        files["validation"] = DatasetFileFingerprint(
            path=str(validation_file),
            sha256=_sha256_file(validation_file),
            records=_count_jsonl_records(validation_file),
        )
    if test_file is not None:
        files["test"] = DatasetFileFingerprint(
            path=str(test_file),
            sha256=_sha256_file(test_file),
            records=_count_jsonl_records(test_file),
        )
    return DatasetBundle(
        dataset_dir=dataset_dir,
        manifest=manifest_path,
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        manifest_payload=manifest_payload,
        files=files,
    )


def build_run_manifest(
    config: TrainingRunConfig,
    *,
    repo_root: Path,
    resolved_model: str,
    dataset_bundle: DatasetBundle,
) -> dict[str, Any]:
    config_payload = asdict(config)
    config_payload["resolved_model"] = resolved_model
    return {
        "timestamp": int(time.time()),
        "repo_root": str(repo_root),
        "security_posture": {
            "trust_remote_code": False,
            "push_to_hub": False,
            "report_to": [],
            "allow_external_dataset": config.security.allow_external_dataset,
            "allow_unverified_model": config.security.allow_unverified_model,
            "local_files_only": config.security.local_files_only,
        },
        "config": config_payload,
        "dataset": {
            "dataset_version": dataset_bundle.manifest_payload.get("dataset_version"),
            "records": dataset_bundle.manifest_payload.get("records"),
            "splits": dataset_bundle.manifest_payload.get("splits"),
            "task_families": dataset_bundle.manifest_payload.get("task_families"),
            "hashes": {
                key: asdict(fingerprint) for key, fingerprint in dataset_bundle.files.items()
            },
        },
        "config_sha256": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _set_secure_runtime_env() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")


def _render_messages_with_masks(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    assistant_only_loss: bool,
    max_seq_length: int,
) -> dict[str, list[int]]:
    conversation: list[dict[str, str]] = []
    input_ids: list[int] = []
    labels: list[int] = []

    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        conversation.append({"role": role, "content": content})
        current_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
        )
        if current_ids[: len(input_ids)] != input_ids:
            raise ValueError("Chat template is not prefix-stable for assistant mask generation.")
        delta = current_ids[len(input_ids) :]
        input_ids = current_ids
        if assistant_only_loss and role != "assistant":
            labels.extend([-100] * len(delta))
        else:
            labels.extend(delta)

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_training(config: TrainingRunConfig, *, repo_root: Path) -> dict[str, Any]:
    _set_secure_runtime_env()
    resolved_model = resolve_model_source(config, repo_root=repo_root)
    dataset_bundle = resolve_dataset_bundle(config, repo_root=repo_root)
    output_dir = _resolve_user_path(config.output_dir, base_dir=repo_root)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = build_run_manifest(
        config,
        repo_root=repo_root,
        resolved_model=resolved_model,
        dataset_bundle=dataset_bundle,
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if config.dry_run:
        summary = {
            "dry_run": True,
            "resolved_model": resolved_model,
            "dataset_dir": str(dataset_bundle.dataset_dir),
            "train_records": dataset_bundle.files["train"].records,
            "validation_records": dataset_bundle.files.get("validation").records
            if dataset_bundle.files.get("validation")
            else 0,
            "output_dir": str(output_dir),
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        trust_remote_code=False,
        local_files_only=config.security.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.chat_template is None:
        raise ValueError(
            "Approved TAR operator training expects a chat-template tokenizer."
        )

    data_files: dict[str, str] = {"train": str(dataset_bundle.train_file)}
    has_validation = (
        dataset_bundle.validation_file is not None
        and dataset_bundle.files["validation"].records > 0
    )
    if has_validation and dataset_bundle.validation_file is not None:
        data_files["validation"] = str(dataset_bundle.validation_file)
    dataset = load_dataset("json", data_files=data_files)

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Dataset example is missing a messages list.")
        encoded = _render_messages_with_masks(
            messages,
            tokenizer,
            assistant_only_loss=config.assistant_only_loss,
            max_seq_length=config.max_seq_length,
        )
        encoded["supervised_tokens"] = sum(
            1 for label in encoded["labels"] if label != -100
        )
        return encoded

    column_names = dataset["train"].column_names
    train_dataset = dataset["train"].map(preprocess, remove_columns=column_names)
    train_dataset = train_dataset.filter(lambda row: row["supervised_tokens"] > 0)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset has no supervised assistant tokens after preprocessing.")
    train_dataset = train_dataset.remove_columns("supervised_tokens")
    eval_dataset = None
    if has_validation:
        eval_dataset = dataset["validation"].map(preprocess, remove_columns=column_names)
        eval_dataset = eval_dataset.filter(lambda row: row["supervised_tokens"] > 0)
        if len(eval_dataset) == 0:
            eval_dataset = None
        else:
            eval_dataset = eval_dataset.remove_columns("supervised_tokens")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": config.security.local_files_only,
    }
    if config.lora.use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if config.bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(resolved_model, **model_kwargs)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if config.lora.use_qlora:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "seed": config.seed,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "logging_steps": config.logging_steps,
        "logging_first_step": True,
        "eval_steps": config.eval_steps if eval_dataset is not None else None,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "save_strategy": "steps",
        "load_best_model_at_end": bool(config.load_best_model_at_end and eval_dataset is not None),
        "metric_for_best_model": "eval_loss" if eval_dataset is not None else None,
        "greater_is_better": False if eval_dataset is not None else None,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "report_to": [],
        "push_to_hub": False,
        "remove_unused_columns": False,
        "logging_dir": str(output_dir / "logs"),
        "save_safetensors": True,
        "label_names": ["labels"],
        "optim": "paged_adamw_8bit" if config.lora.use_qlora else "adamw_torch",
    }
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = (
            "steps" if eval_dataset is not None else "no"
        )
    elif "eval_strategy" in training_args_signature.parameters:
        training_args_kwargs["eval_strategy"] = "steps" if eval_dataset is not None else "no"
    else:
        raise ValueError("TrainingArguments has no supported evaluation strategy parameter.")

    training_args = TrainingArguments(**training_args_kwargs)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        label_pad_token_id=-100,
        pad_to_multiple_of=config.pad_to_multiple_of,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_state()
    final_adapter_dir = output_dir / "final_adapter"
    model.save_pretrained(final_adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter_dir)

    summary = {
        "dry_run": False,
        "resolved_model": resolved_model,
        "dataset_dir": str(dataset_bundle.dataset_dir),
        "train_records": len(train_dataset),
        "validation_records": len(eval_dataset) if eval_dataset is not None else 0,
        "output_dir": str(output_dir),
        "final_adapter_dir": str(final_adapter_dir),
        "train_metrics": train_result.metrics,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    args = parse_args()
    config_path = _resolve_user_path(args.config, base_dir=repo_root)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {args.config}")
    config = load_training_config(config_path, repo_root=repo_root, args=args)
    summary = run_training(config, repo_root=repo_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
