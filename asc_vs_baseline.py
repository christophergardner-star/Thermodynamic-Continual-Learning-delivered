"""
asc_vs_baseline.py — ASC-124M vs GPT-2 baseline, same batches, same steps.

The fairest possible CPU comparison:
  - Same backbone (distilgpt2 — faster on CPU, same hidden_dim=768)
  - Same optimizer (AdamW, same lr/wd)
  - Same batches in the same order (identical DataLoader seed)
  - Same number of steps (MAX_STEPS each)
  - ASC gets consistency loss; baseline gets nothing extra

If ASC doesn't beat baseline here, the mechanism needs more work.

Runtime: ~100 steps × ~5s/step × 2 models ≈ 15-20 min on CPU.
"""

import math
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
BACKBONE      = "distilgpt2"   # 82M, hidden=768, fast on CPU
MAX_STEPS     = 100            # steps per model (~15-20 min total)
BATCH_SIZE    = 2
MAX_LENGTH    = 64
LR            = 3e-4
WEIGHT_DECAY  = 0.1
LAMBDA_C      = 0.3            # consistency loss weight for ASC
EMA_DECAY     = 0.995
WARP_DIM      = 256
SEED          = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Backbone: {BACKBONE}  |  Steps: {MAX_STEPS}  |  Batch: {BATCH_SIZE}  |  SeqLen: {MAX_LENGTH}")
print(f"ASC lambda_c: {LAMBDA_C}  |  EMA decay: {EMA_DECAY}\n")

# ── Dataset (built once, replayed identically for both models) ────────────────
print("Loading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def make_loader(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        tokenized, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, generator=g,
    )

# ── LatentWarp ────────────────────────────────────────────────────────────────
class LatentWarp(nn.Module):
    def __init__(self, hidden_dim=768, warp_dim=256, init_scale=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, warp_dim), nn.ReLU(), nn.Linear(warp_dim, hidden_dim)
        )
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
    def forward(self, h):
        return h + self.net(h) * self.scale

# ── Run one model for MAX_STEPS, return per-step losses ──────────────────────
def run_model(label, use_asc=False):
    print(f"\n{'='*55}")
    print(f"  Running: {label}  (ASC={use_asc})")
    print(f"{'='*55}")

    torch.manual_seed(SEED)
    loader = make_loader(SEED)

    # Online model
    model = GPT2LMHeadModel.from_pretrained(BACKBONE).to(device)
    params = list(model.parameters())

    # ASC extras
    warp = None
    target = None
    if use_asc:
        hidden_dim = model.config.hidden_size
        warp = LatentWarp(hidden_dim=hidden_dim, warp_dim=WARP_DIM).to(device)
        target = copy.deepcopy(model)
        target.eval()
        for p in target.parameters():
            p.requires_grad = False
        params = params + list(warp.parameters())

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    task_losses = []
    consist_losses = []
    step = 0
    t0 = time.time()

    for batch in loader:
        if step >= MAX_STEPS:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()

        # Task loss
        out = model(**batch)
        task_loss = out.loss

        if use_asc:
            # Consistency forward
            with torch.no_grad():
                tgt_out = target(**{k: v for k, v in batch.items() if k != "labels"},
                                 output_hidden_states=True)
                clean_h = tgt_out.hidden_states[-1]
            warped_h = warp(clean_h)
            consist_out = model(inputs_embeds=warped_h, labels=batch["labels"])
            consist_loss = consist_out.loss
            total = task_loss + LAMBDA_C * consist_loss
            # EMA update
            with torch.no_grad():
                for po, pt in zip(model.parameters(), target.parameters()):
                    pt.data.mul_(EMA_DECAY).add_(po.data, alpha=1 - EMA_DECAY)
        else:
            total = task_loss
            consist_loss = torch.tensor(0.0)

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        tl = float(task_loss.item())
        cl = float(consist_loss.item())
        task_losses.append(tl)
        consist_losses.append(cl)
        step += 1

        if step % 10 == 0:
            elapsed = (time.time() - t0) / 60
            avg_ppl = math.exp(min(sum(task_losses[-10:]) / 10, 20))
            suffix = f" | consist={cl:.4f}" if use_asc else ""
            print(f"  step {step:3d} | task={tl:.4f}{suffix} | PPL={avg_ppl:.2f} | {elapsed:.1f}min")

    total_time = (time.time() - t0) / 60
    final_ppl = math.exp(min(sum(task_losses[-10:]) / 10, 20))
    first_ppl = math.exp(min(sum(task_losses[:10]) / 10, 20))
    print(f"\n  {label} done: {step} steps in {total_time:.1f} min")
    print(f"  PPL: {first_ppl:.2f} (step 10) -> {final_ppl:.2f} (step {step})")

    return {
        "label": label,
        "task_losses": task_losses,
        "consist_losses": consist_losses,
        "final_ppl": final_ppl,
        "first_ppl": first_ppl,
        "time_min": total_time,
    }


# ── Run both ──────────────────────────────────────────────────────────────────
baseline = run_model("GPT-2 Baseline", use_asc=False)
asc      = run_model(f"ASC-{BACKBONE}", use_asc=True)


# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  COMPARISON RESULTS")
print("=" * 55)

b_ppl  = baseline["final_ppl"]
a_ppl  = asc["final_ppl"]
delta  = b_ppl - a_ppl
pct    = 100.0 * delta / b_ppl

print(f"  {'Model':<22} {'Start PPL':>10} {'Final PPL':>10} {'Time':>8}")
print(f"  {'-'*54}")
print(f"  {'GPT-2 Baseline':<22} {baseline['first_ppl']:>10.2f} {b_ppl:>10.2f} {baseline['time_min']:>7.1f}m")
print(f"  {f'ASC ({BACKBONE})':<22} {asc['first_ppl']:>10.2f} {a_ppl:>10.2f} {asc['time_min']:>7.1f}m")
print(f"  {'-'*54}")

if delta > 0:
    print(f"\n  ASC wins: PPL {b_ppl:.2f} -> {a_ppl:.2f}  ({pct:+.1f}% improvement)")
    verdict = "ASC WINS"
elif delta < -0.5:
    print(f"\n  Baseline wins: PPL {a_ppl:.2f} vs {b_ppl:.2f}  ({pct:+.1f}%)")
    verdict = "BASELINE WINS"
else:
    print(f"\n  Inconclusive: delta={delta:.2f} PPL (within noise for {MAX_STEPS} steps)")
    verdict = "INCONCLUSIVE"

print(f"  Verdict: {verdict}")
print("=" * 55)
print(f"\nNote: {MAX_STEPS} CPU steps is a short run. A full GPU run (3 epochs)")
print("is needed for a statistically meaningful result.")


# ── Plot ──────────────────────────────────────────────────────────────────────
steps = list(range(1, MAX_STEPS + 1))
b_ppl_curve = [math.exp(min(l, 20)) for l in baseline["task_losses"]]
a_ppl_curve = [math.exp(min(l, 20)) for l in asc["task_losses"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(steps, b_ppl_curve, label="GPT-2 Baseline", color="#3498DB", linewidth=1.2)
ax1.plot(steps, a_ppl_curve, label=f"ASC ({BACKBONE})", color="#E74C3C", linewidth=1.2)
ax1.set_title(f"PPL per step ({MAX_STEPS} steps, CPU)")
ax1.set_xlabel("Step")
ax1.set_ylabel("PPL")
ax1.set_yscale("log")
ax1.legend()
ax1.grid(True, alpha=0.4)

# Rolling average (window=10)
def rolling(lst, w=10):
    return [sum(lst[max(0,i-w):i+1]) / min(i+1, w) for i in range(len(lst))]

ax2.plot(steps, rolling(baseline["task_losses"]), label="GPT-2 Baseline",
         color="#3498DB", linewidth=1.5)
ax2.plot(steps, rolling(asc["task_losses"]), label=f"ASC ({BACKBONE})",
         color="#E74C3C", linewidth=1.5)
ax2.set_title("Task loss (10-step rolling avg)")
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.suptitle(f"ASC vs Baseline -- {BACKBONE} -- {MAX_STEPS} steps -- CPU", fontsize=11)
plt.tight_layout()
plt.savefig("asc_vs_baseline.png", dpi=150)
print("\nPlot saved -> asc_vs_baseline.png")
