"""
Generic coding-backbone ASC fine-tuning entrypoint.

Despite the historical filename, this script now supports both DeepSeek Coder
and Qwen2.5-Coder backbones.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import torch

from asc_model import ASCConfig, ASCForCausalLM

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


MODEL_PRESETS = {
    "tiny": "__tiny_gpt2__",
    "1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "6.7b": "deepseek-ai/deepseek-coder-6.7b-base",
    "33b": "deepseek-ai/deepseek-coder-33b-base",
    "1.3b-instruct": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "33b-instruct": "deepseek-ai/deepseek-coder-33b-instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-Coder-1.5B",
    "qwen-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen-14b": "Qwen/Qwen2.5-Coder-14B",
    "qwen-32b": "Qwen/Qwen2.5-Coder-32B",
    "qwen-1.5b-instruct": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen-14b-instruct": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "qwen-32b-instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
}


EXPERIMENTAL_WARNING = (
    "EXPERIMENTAL WARNING: deepseek_asc_finetune.py is not yet a "
    "scientifically validated large-model ASC trainer. Masking, device "
    "placement, distributed scaling, and memory strategy are still incomplete. "
    "Use asc_train_full.py for the canonical ASC path and treat this script as "
    "experimental scaffolding."
)


def resolve_model_id(model: Optional[str]) -> str:
    if not model:
        return MODEL_PRESETS["qwen-7b"]
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)
    return MODEL_PRESETS.get(model, model)


def derive_run_name(model_id: str) -> str:
    slug = model_id.rstrip("/").split("/")[-1]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-_.").lower()
    return slug or "coding-asc"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def iter_local_texts(dataset_path: Path) -> Iterable[str]:
    if dataset_path.is_file():
        if dataset_path.suffix.lower() == ".jsonl":
            for line in dataset_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text") or item.get("prompt") or item.get("code")
                if text:
                    yield text
            return
        yield dataset_path.read_text(encoding="utf-8")
        return
    for path in sorted(dataset_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt", ".jsonl"}:
            yield from iter_local_texts(path)


def load_text_samples(dataset: str, split: str, max_samples: int) -> List[str]:
    dataset_path = Path(dataset)
    if dataset_path.exists():
        texts = list(iter_local_texts(dataset_path))
        return texts[:max_samples]
    if dataset == "synthetic":
        return [
            "def add(a: int, b: int) -> int:\n    return a + b",
            "def fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)",
            "class Counter:\n    def __init__(self):\n        self.value = 0\n    def inc(self):\n        self.value += 1\n        return self.value",
        ][:max_samples]
    if load_dataset is None:
        raise RuntimeError("datasets is required for non-local dataset loading")
    ds = load_dataset(dataset, split=split)
    texts: List[str] = []
    for row in ds:
        text = row.get("text") or row.get("content") or row.get("code")
        if text:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    return texts


def encode_texts(model_id: str, texts: List[str], seq_len: int) -> List[torch.Tensor]:
    if model_id == "__tiny_gpt2__":
        encoded = []
        for text in texts:
            token_ids = [(ord(ch) % 255) + 1 for ch in text[:seq_len]]
            if not token_ids:
                token_ids = [1]
            token_ids = token_ids[:seq_len]
            if len(token_ids) < seq_len:
                token_ids.extend([0] * (seq_len - len(token_ids)))
            encoded.append(torch.tensor(token_ids, dtype=torch.long))
        return encoded
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required for tokenizer-backed encoding")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    return [row.clone().detach() for row in batch["input_ids"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune coding backbones with ASC")
    parser.add_argument("--model", default="qwen-7b")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warp_dim", type=int, default=128)
    parser.add_argument("--consistency_lambda", type=float, default=0.3)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> dict:
    model_id = resolve_model_id(args.model)
    run_name = args.run_name or derive_run_name(model_id)
    output_dir = Path(args.output_dir or Path("coding_ai_out") / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": args.model,
        "resolved_model_id": model_id,
        "run_name": run_name,
        "dataset": args.dataset,
        "split": args.split,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "max_samples": args.max_samples,
        "dry_run": args.dry_run,
        "timestamp": int(time.time()),
    }
    write_json(output_dir / "training_manifest.json", manifest)

    tiny_overrides = {}
    if model_id == "__tiny_gpt2__":
        tiny_overrides = {
            "vocab_size": 256,
            "n_positions": max(args.seq_len, 32),
            "n_ctx": max(args.seq_len, 32),
            "n_embd": 64,
            "n_layer": 2,
            "n_head": 4,
        }

    status = {
        "phase": "initializing",
        "step": 0,
        "max_steps": args.max_steps,
        "run_name": run_name,
        "model_id": model_id,
        "dry_run": args.dry_run,
    }
    write_json(output_dir / "status.json", status)

    if args.dry_run:
        status.update({"phase": "dry_run_complete", "step": 0})
        write_json(output_dir / "status.json", status)
        return {"output_dir": str(output_dir), "status": status, "manifest": manifest}

    texts = load_text_samples(args.dataset, args.split, args.max_samples)
    encoded = encode_texts(model_id, texts, args.seq_len)
    if not encoded:
        raise RuntimeError("No usable training samples were found")

    config = ASCConfig(
        base_model_name=model_id,
        warp_dim=args.warp_dim,
        consistency_lambda=args.consistency_lambda,
        ema_decay=args.ema_decay,
        backbone_config_overrides=tiny_overrides,
    )
    model = ASCForCausalLM(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters_trainable(), lr=args.lr)
    warp_optimizer = torch.optim.Adam(model.warp.parameters(), lr=args.lr)

    losses = []
    for step in range(1, args.max_steps + 1):
        batch_tensors = encoded[(step - 1) % len(encoded):(step - 1) % len(encoded) + args.batch_size]
        if len(batch_tensors) < args.batch_size:
            batch_tensors.extend(encoded[: args.batch_size - len(batch_tensors)])
        batch = torch.stack(batch_tensors)

        optimizer.zero_grad()
        warp_optimizer.zero_grad()
        task_loss, consist_loss = model(input_ids=batch, labels=batch)
        total_loss = task_loss + args.consistency_lambda * consist_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters_trainable(), 1.0)
        optimizer.step()

        warp_loss = model.warp_ascent_loss(input_ids=batch, labels=batch)
        (-warp_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.warp.parameters(), 1.0)
        warp_optimizer.step()
        model.update_target()

        losses.append(float(total_loss.detach().cpu()))
        status.update(
            {
                "phase": "training",
                "step": step,
                "task_loss": float(task_loss.detach().cpu()),
                "consistency_loss": float(consist_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
            }
        )
        write_json(output_dir / "status.json", status)

    model.save(str(output_dir / "final"))
    status.update({"phase": "complete", "mean_total_loss": sum(losses) / len(losses)})
    write_json(output_dir / "status.json", status)
    return {"output_dir": str(output_dir), "status": status, "manifest": manifest}


def main() -> int:
    args = parse_args()
    print(EXPERIMENTAL_WARNING, file=sys.stderr)
    result = run_training(args)
    print(json.dumps(result["status"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
