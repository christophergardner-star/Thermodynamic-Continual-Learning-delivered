"""
asc_train_full.py -- ASC family training script
================================================

Trains any ASC model size on WikiText-2 or WikiText-103.

Usage
-----
  # ASC-124M on WikiText-2 (default)
  python asc_train_full.py

  # ASC-355M on WikiText-103
  python asc_train_full.py --size 355M --dataset wikitext-103-raw-v1

  # Custom config
  python asc_train_full.py --size 124M --epochs 3 --lr 3e-4 --lambda_c 0.3

Cluster (H100)
--------------
  torchrun --nproc_per_node=4 asc_train_full.py --size 1B --batch_size 16

The script checkpoints every --save_every steps and logs PPL, task loss,
and consistency loss. Final model saved to asc_out/<size>/.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from asc_model import ASCConfig, ASCForCausalLM


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train an ASC model")
    p.add_argument("--size", default="124M",
                   choices=["124M", "355M", "774M", "1558M"],
                   help="Model size preset")
    p.add_argument("--dataset", default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lambda_c", type=float, default=0.3,
                   help="Consistency loss weight")
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--warp_dim", type=int, default=256)
    p.add_argument("--save_every", type=int, default=500,
                   help="Checkpoint every N steps")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Cap total steps (useful for smoke tests)")
    p.add_argument("--out_dir", default="asc_out")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_dir, f"ASC-{args.size}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ASC-{args.size} Training")
    print(f"  Device:   {device}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LR:       {args.lr}")
    print(f"  lambda_c: {args.lambda_c}")
    print(f"  Out:      {out_dir}")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_dataset("wikitext", args.dataset, split="train")
    config = ASCConfig.for_size(
        args.size,
        warp_dim=args.warp_dim,
        consistency_lambda=args.lambda_c,
        ema_decay=args.ema_decay,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True,
                         max_length=args.max_length)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(tokenized, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collator)

    # ── Model ─────────────────────────────────────────────────────────
    print(f"Building ASC-{args.size}...")
    model = ASCForCausalLM(config).to(device)
    summary = model.param_summary()
    print(f"  Base params:      {summary['base_params']:,}")
    print(f"  Warp params:      {summary['warp_params']:,} ({summary['warp_pct']}% of base)")
    print(f"  Total trainable:  {summary['total_trainable']:,}\n")

    # ── Optimizers (min-max: base minimises, warp maximises consist loss) ──
    total_steps = len(loader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    base_optimizer = torch.optim.AdamW(
        model.parameters_trainable(),
        lr=args.lr,
        weight_decay=0.1,
    )
    warp_optimizer = torch.optim.AdamW(
        model.warp.parameters(),
        lr=args.lr * 3,   # warp LR slightly higher to keep adversary active
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer, T_max=total_steps
    )

    # ── Training ──────────────────────────────────────────────────────
    history = {"step": [], "task_loss": [], "consist_loss": [], "ppl": []}
    step = 0
    global_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_task = 0.0
        epoch_n = 0

        bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in bar:
            if args.max_steps and step >= args.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}

            # ── Base model step (minimise task + consistency) ──────────
            base_optimizer.zero_grad()
            task_loss, consist_loss = model(**batch)
            total_loss = task_loss + args.lambda_c * consist_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters_trainable(), 1.0)
            base_optimizer.step()
            scheduler.step()

            # ── Warp step (maximise consistency — gradient ascent) ─────
            warp_optimizer.zero_grad()
            warp_loss = model.warp_ascent_loss(**batch)
            (-warp_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.warp.parameters(), 1.0)
            warp_optimizer.step()

            model.update_target()

            step += 1
            tl = float(task_loss.item())
            cl = float(consist_loss.item())
            epoch_task += tl
            epoch_n += 1

            if step % args.log_every == 0:
                ppl = math.exp(min(tl, 20))
                elapsed = (time.time() - global_start) / 60
                bar.set_postfix({
                    "task": f"{tl:.4f}",
                    "consist": f"{cl:.4f}",
                    "PPL": f"{ppl:.2f}",
                    "min": f"{elapsed:.1f}",
                })
                history["step"].append(step)
                history["task_loss"].append(tl)
                history["consist_loss"].append(cl)
                history["ppl"].append(ppl)

            if step % args.save_every == 0:
                ckpt = os.path.join(out_dir, f"step_{step}")
                model.save(ckpt)
                print(f"\n  [ckpt] Saved -> {ckpt}")

        avg_ppl = math.exp(min(epoch_task / max(epoch_n, 1), 20))
        print(f"\nEpoch {epoch+1} done -- avg PPL: {avg_ppl:.2f}\n")

    # ── Save final ────────────────────────────────────────────────────
    final_path = os.path.join(out_dir, "final")
    model.save(final_path)
    print(f"Final model saved -> {final_path}")

    # ── Plot ─────────────────────────────────────────────────────────
    if history["step"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history["step"], history["ppl"], color="#E74C3C")
        ax1.set_title(f"ASC-{args.size} PPL")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("PPL")
        ax1.set_yscale("log")
        ax1.grid(True)

        ax2.plot(history["step"], history["task_loss"],
                 label="task", color="#3498DB")
        ax2.plot(history["step"], history["consist_loss"],
                 label="consist", color="#E67E22", alpha=0.7)
        ax2.set_title(f"ASC-{args.size} Losses")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved -> {plot_path}")

    total_min = (time.time() - global_start) / 60
    print(f"\nTotal training time: {total_min:.1f} min  |  Steps: {step}")


if __name__ == "__main__":
    main()
