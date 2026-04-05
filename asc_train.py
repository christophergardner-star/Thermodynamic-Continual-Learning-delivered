"""
ASC - Adversarial Self-Consistency Training
FIXED VERSION (April 2026) — adapted from Colab script for local/CLI run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves figure to file
import matplotlib.pyplot as plt

# ----------------- DEVICE -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- CONFIG -----------------
config = {
    "model_name": "gpt2",
    "max_length": 512,
    "batch_size": 8,
    "lr": 3e-4,
    "epochs": 3,
    "consistency_lambda": 0.3,
    "ema_decay": 0.995,
    "warp_dim": 256,
    "seed": 42,
}

torch.manual_seed(config["seed"])

# ----------------- DATASET -----------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=config["max_length"])

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_loader = DataLoader(
    tokenized, batch_size=config["batch_size"], shuffle=True, collate_fn=data_collator
)

# ----------------- MODELS -----------------
base_model = GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device)

online_model = GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device)
online_model.load_state_dict(base_model.state_dict())

target_model = GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device)
target_model.load_state_dict(base_model.state_dict())
target_model.eval()

# Latent Warp Network
class LatentWarp(nn.Module):
    def __init__(self, hidden_dim=768, warp_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, warp_dim),
            nn.ReLU(),
            nn.Linear(warp_dim, hidden_dim),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.05)

    def forward(self, hidden_states):
        return hidden_states + self.net(hidden_states) * self.scale

warp_net = LatentWarp().to(device)

# ----------------- OPTIMIZER & SCHEDULER -----------------
optimizer = torch.optim.AdamW(
    list(online_model.parameters()) + list(warp_net.parameters()),
    lr=config["lr"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(train_loader) * config["epochs"]
)

def update_target(online, target, decay=0.995):
    with torch.no_grad():
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1 - decay)

# ----------------- TRAINING LOOP -----------------
ppl_history = []
step = 0

print("Starting ASC Training...\n")

for epoch in range(config["epochs"]):
    online_model.train()
    total_loss = 0.0
    total_consistency = 0.0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        optimizer.zero_grad()

        # Main task loss
        outputs = online_model(**batch)
        task_loss = outputs.loss

        # Latent adversarial warp (target provides stable hidden states)
        with torch.no_grad():
            target_outputs = target_model(**batch, output_hidden_states=True)
            clean_hidden = target_outputs.hidden_states[-1]

        warped_hidden = warp_net(clean_hidden)

        # Consistency loss: online model must handle warped latents
        warped_outputs = online_model(inputs_embeds=warped_hidden, labels=batch["labels"])
        consistency_loss = warped_outputs.loss

        total_loss_val = task_loss + config["consistency_lambda"] * consistency_loss
        total_loss_val.backward()
        optimizer.step()
        scheduler.step()

        update_target(online_model, target_model, config["ema_decay"])

        step += 1
        total_loss += task_loss.item()
        total_consistency += consistency_loss.item()

        progress.set_postfix({
            "task_loss": f"{task_loss.item():.4f}",
            "consist": f"{consistency_loss.item():.4f}",
            "PPL": f"{np.exp(min(task_loss.item(), 20)):.2f}",
        })

        if step % 200 == 0:
            window_ppl = np.exp(min(total_loss / step, 20))
            ppl_history.append(window_ppl)
            print(f"  [step {step}] running PPL={window_ppl:.2f}")

    epoch_ppl = np.exp(min(total_loss / len(train_loader), 20))
    print(f"\nEpoch {epoch+1} done — Avg PPL: {epoch_ppl:.2f}\n")

# ----------------- RESULTS -----------------
print("\n" + "=" * 70)
print("ASC TRAINING COMPLETE")
print(f"Final training PPL: {epoch_ppl:.2f}")
print("=" * 70)

if ppl_history:
    plt.figure(figsize=(8, 4))
    plt.plot(ppl_history, marker="o")
    plt.title("ASC Training Perplexity (×200 steps)")
    plt.xlabel("Checkpoint (×200 steps)")
    plt.ylabel("PPL")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("asc_ppl_curve.png", dpi=150)
    print("PPL curve saved → asc_ppl_curve.png")

# Save model
online_model.save_pretrained("asc_model_fixed")
tokenizer.save_pretrained("asc_model_fixed")
print("Model saved → asc_model_fixed/")
