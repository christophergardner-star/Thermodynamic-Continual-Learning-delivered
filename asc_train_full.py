"""
asc_train_full.py -- ASC family training script
===============================================

Trains ASC model sizes on WikiText-2 or WikiText-103.

WS29 note
---------
This script now supports checkpoint-aware relaunch through
`--resume_from_checkpoint`. Resume state is persisted separately from the
model weights so TAR can reason about launch state, artifacts, and completed
steps without guessing from directory names alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from asc_model import ASCConfig, ASCForCausalLM

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ASC model")
    parser.add_argument(
        "--size",
        default="124M",
        choices=["124M", "355M", "774M", "1558M"],
        help="Model size preset",
    )
    parser.add_argument(
        "--dataset",
        default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lambda_c", type=float, default=0.3, help="Consistency loss weight")
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--warp_dim", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=None, help="Cap total steps")
    parser.add_argument("--out_dir", default="asc_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--backend_state_path", default=None)
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / f"step_{step}"


def _resume_state_path(run_dir: Path) -> Path:
    return run_dir / "resume_state.pt"


def _training_log_path(run_dir: Path) -> Path:
    return run_dir / "training_log.json"


def _plot_path(run_dir: Path) -> Path:
    return run_dir / "training_curves.png"


def _normalize_resume_source(path_str: str | None, run_dir: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_dir():
        candidate = path / "resume_state.pt"
        if candidate.exists():
            return candidate
        return path
    if path.exists():
        return path
    default_candidate = run_dir / "resume_state.pt"
    return default_candidate if default_candidate.exists() else path


def _load_resume_bundle(path: Path, device: torch.device) -> dict[str, Any]:
    if path.is_dir():
        payload = {
            "latest_checkpoint_path": str(path),
            "completed_steps": 0,
            "completed_epochs": 0,
            "epoch_step": 0,
            "history": {"step": [], "task_loss": [], "consist_loss": [], "ppl": []},
            "resume_source_kind": "model_dir_only",
        }
        return payload
    return torch.load(path, map_location=device, weights_only=False)


def _save_training_log(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    status: str,
    step: int,
    epoch_index: int,
    epoch_step: int,
    resumed_from_checkpoint: bool,
    latest_checkpoint_path: str | None,
    final_checkpoint_path: str | None,
    history: dict[str, list[float]],
    latest_metrics: dict[str, float] | None,
) -> Path:
    payload = {
        "status": status,
        "size": args.size,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_c": args.lambda_c,
        "ema_decay": args.ema_decay,
        "warp_dim": args.warp_dim,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "completed_steps": step,
        "completed_epochs": epoch_index,
        "epoch_step": epoch_step,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resume_source": args.resume_from_checkpoint,
        "latest_checkpoint_path": latest_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
        "history": history,
        "latest_metrics": latest_metrics,
        "updated_at": utc_now_iso(),
    }
    path = _training_log_path(run_dir)
    _atomic_write_json(path, payload)
    return path


def _update_backend_state(
    state_path: str | None,
    *,
    status: str,
    output_dir: Path,
    completed_steps: int,
    current_epoch: int,
    epoch_step: int,
    latest_checkpoint_path: str | None,
    final_checkpoint_path: str | None = None,
    resumed_from_checkpoint: bool = False,
    resume_source: str | None = None,
    last_error: str | None = None,
) -> None:
    if not state_path:
        return
    path = Path(state_path)
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    launch_count = int(payload.get("launch_count", 0))
    if status == "running":
        launch_count += 1
        payload["last_started_at"] = utc_now_iso()
    if status in {"completed", "failed", "interrupted"}:
        payload["last_completed_at"] = utc_now_iso()
    payload["status"] = status
    payload["launch_count"] = launch_count
    payload["output_dir"] = str(output_dir)
    payload["completed_steps"] = completed_steps
    payload["current_epoch"] = current_epoch
    payload["epoch_step"] = epoch_step
    payload["last_heartbeat_at"] = utc_now_iso()
    payload["last_error"] = last_error
    payload.setdefault("resume", {})
    payload["resume"].update(
        {
            "requested": bool(resume_source),
            "mode": "checkpoint_resume" if resumed_from_checkpoint else "fresh_start",
            "resume_from_checkpoint": resume_source,
            "latest_checkpoint_path": latest_checkpoint_path,
            "checkpoint_exists": bool(latest_checkpoint_path and Path(latest_checkpoint_path).exists()),
        }
    )
    payload.setdefault("artifact_lineage", {})
    payload["artifact_lineage"].update(
        {
            "training_log_path": str(_training_log_path(output_dir)),
            "latest_checkpoint_path": latest_checkpoint_path,
            "final_checkpoint_path": final_checkpoint_path,
        }
    )
    payload["updated_at"] = utc_now_iso()
    _atomic_write_json(path, payload)


def _save_resume_bundle(
    run_dir: Path,
    *,
    model: ASCForCausalLM,
    base_optimizer: torch.optim.Optimizer,
    warp_optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    epoch_index: int,
    epoch_step: int,
    history: dict[str, list[float]],
    args: argparse.Namespace,
    latest_checkpoint_path: str,
    latest_metrics: dict[str, float] | None,
) -> Path:
    bundle = {
        "size": args.size,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_c": args.lambda_c,
        "ema_decay": args.ema_decay,
        "warp_dim": args.warp_dim,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "completed_steps": step,
        "completed_epochs": epoch_index,
        "epoch_step": epoch_step,
        "latest_checkpoint_path": latest_checkpoint_path,
        "history": history,
        "latest_metrics": latest_metrics,
        "base_optimizer_state": base_optimizer.state_dict(),
        "warp_optimizer_state": warp_optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "updated_at": utc_now_iso(),
        "asc_config": model.config.__dict__,
    }
    path = _resume_state_path(run_dir)
    torch.save(bundle, path)
    return path


def _restore_resume_state(
    bundle: dict[str, Any],
    *,
    args: argparse.Namespace,
    model: ASCForCausalLM,
    base_optimizer: torch.optim.Optimizer,
    warp_optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, int, int, dict[str, list[float]], dict[str, float] | None]:
    if str(bundle.get("size")) != str(args.size):
        raise ValueError("Resume checkpoint size mismatch.")
    if str(bundle.get("dataset")) != str(args.dataset):
        raise ValueError("Resume checkpoint dataset mismatch.")
    base_state = bundle.get("base_optimizer_state")
    warp_state = bundle.get("warp_optimizer_state")
    scheduler_state = bundle.get("scheduler_state")
    if base_state:
        base_optimizer.load_state_dict(base_state)
    if warp_state:
        warp_optimizer.load_state_dict(warp_state)
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)
    if bundle.get("python_random_state") is not None:
        random.setstate(bundle["python_random_state"])
    if bundle.get("numpy_random_state") is not None:
        np.random.set_state(bundle["numpy_random_state"])
    if bundle.get("torch_rng_state") is not None:
        torch.set_rng_state(bundle["torch_rng_state"])
    cuda_state = bundle.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    history = bundle.get("history") or {"step": [], "task_loss": [], "consist_loss": [], "ppl": []}
    return (
        int(bundle.get("completed_steps", 0)),
        int(bundle.get("completed_epochs", 0)),
        int(bundle.get("epoch_step", 0)),
        history,
        bundle.get("latest_metrics"),
    )


def _load_or_build_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    resume_bundle: dict[str, Any] | None,
) -> ASCForCausalLM:
    if resume_bundle and resume_bundle.get("latest_checkpoint_path"):
        checkpoint_path = Path(str(resume_bundle["latest_checkpoint_path"]))
        if checkpoint_path.exists():
            return ASCForCausalLM.load(str(checkpoint_path)).to(device)
    config = ASCConfig.for_size(
        args.size,
        warp_dim=args.warp_dim,
        consistency_lambda=args.lambda_c,
        ema_decay=args.ema_decay,
    )
    return ASCForCausalLM(config).to(device)


def _build_loader(tokenized: Any, collator: Any, *, batch_size: int, seed: int) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )


def _save_plot(run_dir: Path, history: dict[str, list[float]]) -> None:
    if not history["step"]:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["step"], history["ppl"], color="#E74C3C")
    ax1.set_title("ASC PPL")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("PPL")
    ax1.set_yscale("log")
    ax1.grid(True)

    ax2.plot(history["step"], history["task_loss"], label="task", color="#3498DB")
    ax2.plot(history["step"], history["consist_loss"], label="consist", color="#E67E22", alpha=0.7)
    ax2.set_title("ASC Losses")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(_plot_path(run_dir), dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)
    run_dir = out_root / f"ASC-{args.size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_source = _normalize_resume_source(args.resume_from_checkpoint, run_dir)
    resume_bundle = _load_resume_bundle(resume_source, device) if resume_source and resume_source.exists() else None
    resumed_from_checkpoint = resume_bundle is not None

    print(f"\n{'=' * 60}")
    print(f"  ASC-{args.size} Training")
    print(f"  Device:   {device}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LR:       {args.lr}")
    print(f"  lambda_c: {args.lambda_c}")
    print(f"  Out:      {run_dir}")
    if resume_source:
        print(f"  Resume:   {resume_source}")
    print(f"{'=' * 60}\n")

    print("Loading dataset...")
    dataset = load_dataset("wikitext", args.dataset, split="train")
    tokenizer_source = (
        str(resume_bundle["latest_checkpoint_path"])
        if resume_bundle and resume_bundle.get("latest_checkpoint_path")
        else ASCConfig.for_size(args.size).base_model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 0)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"Building ASC-{args.size}...")
    model = _load_or_build_model(args, device=device, resume_bundle=resume_bundle)
    summary = model.param_summary()
    print(f"  Base params:      {summary['base_params']:,}")
    print(f"  Warp params:      {summary['warp_params']:,} ({summary['warp_pct']}% of base)")
    print(f"  Total trainable:  {summary['total_trainable']:,}\n")

    steps_per_epoch = max(1, math.ceil(len(tokenized) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    base_optimizer = torch.optim.AdamW(
        model.parameters_trainable(),
        lr=args.lr,
        weight_decay=0.1,
    )
    warp_optimizer = torch.optim.AdamW(
        model.warp.parameters(),
        lr=args.lr * 3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=max(total_steps, 1))

    history: dict[str, list[float]] = {"step": [], "task_loss": [], "consist_loss": [], "ppl": []}
    latest_metrics: dict[str, float] | None = None
    step = 0
    start_epoch = 0
    start_epoch_step = 0

    if resume_bundle is not None:
        step, start_epoch, start_epoch_step, history, latest_metrics = _restore_resume_state(
            resume_bundle,
            args=args,
            model=model,
            base_optimizer=base_optimizer,
            warp_optimizer=warp_optimizer,
            scheduler=scheduler,
        )

    _update_backend_state(
        args.backend_state_path,
        status="running",
        output_dir=run_dir,
        completed_steps=step,
        current_epoch=start_epoch,
        epoch_step=start_epoch_step,
        latest_checkpoint_path=str(resume_bundle.get("latest_checkpoint_path")) if resume_bundle else None,
        resumed_from_checkpoint=resumed_from_checkpoint,
        resume_source=str(resume_source) if resume_source else None,
    )
    _save_training_log(
        run_dir,
        args=args,
        status="running",
        step=step,
        epoch_index=start_epoch,
        epoch_step=start_epoch_step,
        resumed_from_checkpoint=resumed_from_checkpoint,
        latest_checkpoint_path=str(resume_bundle.get("latest_checkpoint_path")) if resume_bundle else None,
        final_checkpoint_path=None,
        history=history,
        latest_metrics=latest_metrics,
    )

    global_start = time.time()

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            loader = _build_loader(tokenized, collator, batch_size=args.batch_size, seed=args.seed + epoch)
            epoch_task = 0.0
            epoch_n = 0
            skip_batches = start_epoch_step if epoch == start_epoch else 0

            bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch_index, batch in enumerate(bar):
                if batch_index < skip_batches:
                    continue
                if args.max_steps and step >= args.max_steps:
                    break

                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                base_optimizer.zero_grad(set_to_none=True)
                task_loss, consist_loss = model(**batch)
                total_loss = task_loss + args.lambda_c * consist_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters_trainable(), 1.0)
                base_optimizer.step()
                scheduler.step()

                warp_optimizer.zero_grad(set_to_none=True)
                warp_loss = model.warp_ascent_loss(**batch)
                (-warp_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.warp.parameters(), 1.0)
                warp_optimizer.step()

                model.update_target()

                step += 1
                epoch_step = batch_index + 1
                task_value = float(task_loss.item())
                consist_value = float(consist_loss.item())
                epoch_task += task_value
                epoch_n += 1
                latest_metrics = {
                    "task_loss": task_value,
                    "consistency_loss": consist_value,
                    "ppl": math.exp(min(task_value, 20.0)),
                }

                if step % args.log_every == 0:
                    elapsed = (time.time() - global_start) / 60.0
                    bar.set_postfix(
                        {
                            "task": f"{task_value:.4f}",
                            "consist": f"{consist_value:.4f}",
                            "PPL": f"{latest_metrics['ppl']:.2f}",
                            "min": f"{elapsed:.1f}",
                        }
                    )
                    history["step"].append(float(step))
                    history["task_loss"].append(task_value)
                    history["consist_loss"].append(consist_value)
                    history["ppl"].append(latest_metrics["ppl"])
                    _save_training_log(
                        run_dir,
                        args=args,
                        status="running",
                        step=step,
                        epoch_index=epoch,
                        epoch_step=epoch_step,
                        resumed_from_checkpoint=resumed_from_checkpoint,
                        latest_checkpoint_path=None,
                        final_checkpoint_path=None,
                        history=history,
                        latest_metrics=latest_metrics,
                    )

                if step % args.save_every == 0:
                    checkpoint_dir = _checkpoint_dir(run_dir, step)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save(str(checkpoint_dir))
                    resume_state = _save_resume_bundle(
                        run_dir,
                        model=model,
                        base_optimizer=base_optimizer,
                        warp_optimizer=warp_optimizer,
                        scheduler=scheduler,
                        step=step,
                        epoch_index=epoch,
                        epoch_step=epoch_step,
                        history=history,
                        args=args,
                        latest_checkpoint_path=str(checkpoint_dir),
                        latest_metrics=latest_metrics,
                    )
                    print(f"\n  [ckpt] Saved -> {checkpoint_dir}")
                    _save_training_log(
                        run_dir,
                        args=args,
                        status="running",
                        step=step,
                        epoch_index=epoch,
                        epoch_step=epoch_step,
                        resumed_from_checkpoint=resumed_from_checkpoint,
                        latest_checkpoint_path=str(resume_state),
                        final_checkpoint_path=None,
                        history=history,
                        latest_metrics=latest_metrics,
                    )
                    _update_backend_state(
                        args.backend_state_path,
                        status="running",
                        output_dir=run_dir,
                        completed_steps=step,
                        current_epoch=epoch,
                        epoch_step=epoch_step,
                        latest_checkpoint_path=str(resume_state),
                        resumed_from_checkpoint=resumed_from_checkpoint,
                        resume_source=str(resume_source) if resume_source else None,
                    )

            avg_ppl = math.exp(min(epoch_task / max(epoch_n, 1), 20.0))
            print(f"\nEpoch {epoch + 1} done -- avg PPL: {avg_ppl:.2f}\n")
            start_epoch_step = 0
            if args.max_steps and step >= args.max_steps:
                break

        final_path = run_dir / "final"
        model.save(str(final_path))
        resume_state = _save_resume_bundle(
            run_dir,
            model=model,
            base_optimizer=base_optimizer,
            warp_optimizer=warp_optimizer,
            scheduler=scheduler,
            step=step,
            epoch_index=args.epochs,
            epoch_step=0,
            history=history,
            args=args,
            latest_checkpoint_path=str(final_path),
            latest_metrics=latest_metrics,
        )
        _save_plot(run_dir, history)
        _save_training_log(
            run_dir,
            args=args,
            status="completed",
            step=step,
            epoch_index=args.epochs,
            epoch_step=0,
            resumed_from_checkpoint=resumed_from_checkpoint,
            latest_checkpoint_path=str(resume_state),
            final_checkpoint_path=str(final_path),
            history=history,
            latest_metrics=latest_metrics,
        )
        _update_backend_state(
            args.backend_state_path,
            status="completed",
            output_dir=run_dir,
            completed_steps=step,
            current_epoch=args.epochs,
            epoch_step=0,
            latest_checkpoint_path=str(resume_state),
            final_checkpoint_path=str(final_path),
            resumed_from_checkpoint=resumed_from_checkpoint,
            resume_source=str(resume_source) if resume_source else None,
        )

        print(f"Final model saved -> {final_path}")
        total_min = (time.time() - global_start) / 60.0
        print(f"\nTotal training time: {total_min:.1f} min  |  Steps: {step}")
        return 0
    except KeyboardInterrupt:
        _update_backend_state(
            args.backend_state_path,
            status="interrupted",
            output_dir=run_dir,
            completed_steps=step,
            current_epoch=start_epoch,
            epoch_step=start_epoch_step,
            latest_checkpoint_path=str(_resume_state_path(run_dir)) if _resume_state_path(run_dir).exists() else None,
            resumed_from_checkpoint=resumed_from_checkpoint,
            resume_source=str(resume_source) if resume_source else None,
            last_error="interrupted",
        )
        raise
    except Exception as exc:
        _update_backend_state(
            args.backend_state_path,
            status="failed",
            output_dir=run_dir,
            completed_steps=step,
            current_epoch=start_epoch,
            epoch_step=start_epoch_step,
            latest_checkpoint_path=str(_resume_state_path(run_dir)) if _resume_state_path(run_dir).exists() else None,
            resumed_from_checkpoint=resumed_from_checkpoint,
            resume_source=str(resume_source) if resume_source else None,
            last_error=str(exc),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
