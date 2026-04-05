"""
ASC - Adversarial Self-Consistency Training
CPU smoke run — 50 steps, 1 epoch, small batch size.
Full GPU run: set MAX_STEPS=None, epochs=3, batch_size=8.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_STEPS = 50       # set None for full run
BATCH_SIZE = 2       # small for CPU
EPOCHS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  MAX_STEPS={MAX_STEPS}  |  BATCH_SIZE={BATCH_SIZE}")

config = {
    "model_name": "distilgpt2",  # 82M params, same hidden_dim=768, ~30% faster than gpt2
    "max_length": 64,            # shorter seq for CPU speed
    "lr": 3e-4,
    "consistency_lambda": 0.3,
    "ema_decay": 0.995,
    "warp_dim": 256,
    "seed": 42,
}
torch.manual_seed(config["seed"])

# Dataset
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=config["max_length"])

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
# Drop rows with 0 tokens (empty WikiText-2 lines crash the collator)
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

# Models
print(f"Loading {config['model_name']}...")
online_model = GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device)
target_model = GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device)
target_model.load_state_dict(online_model.state_dict())
target_model.eval()

class LatentWarp(nn.Module):
    def __init__(self, hidden_dim=768, warp_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, warp_dim), nn.ReLU(), nn.Linear(warp_dim, hidden_dim)
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.05)
    def forward(self, h):
        return h + self.net(h) * self.scale

warp_net = LatentWarp().to(device)

optimizer = torch.optim.AdamW(
    list(online_model.parameters()) + list(warp_net.parameters()), lr=config["lr"]
)

def update_target(online, target, decay=0.995):
    with torch.no_grad():
        for po, pt in zip(online.parameters(), target.parameters()):
            pt.data.mul_(decay).add_(po.data, alpha=1 - decay)

# Training
print(f"\nStarting ASC training ({MAX_STEPS} steps on {device})...\n")
ppl_history = []
task_losses, consist_losses = [], []
step = 0

for epoch in range(EPOCHS):
    online_model.train()
    total_task = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        if MAX_STEPS and step >= MAX_STEPS:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()

        # Task loss
        task_out = online_model(**batch)
        task_loss = task_out.loss

        # Consistency loss via latent warp
        with torch.no_grad():
            tgt_out = target_model(**batch, output_hidden_states=True)
            clean_h = tgt_out.hidden_states[-1]
        warped_h = warp_net(clean_h)
        consist_out = online_model(inputs_embeds=warped_h, labels=batch["labels"])
        consist_loss = consist_out.loss

        total = task_loss + config["consistency_lambda"] * consist_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(online_model.parameters()) + list(warp_net.parameters()), 1.0
        )
        optimizer.step()
        update_target(online_model, target_model, config["ema_decay"])

        step += 1
        tl = float(task_loss.item())
        cl = float(consist_loss.item())
        total_task += tl
        task_losses.append(tl)
        consist_losses.append(cl)
        ppl_history.append(float(np.exp(min(tl, 20))))

        if step % 10 == 0:
            avg_ppl = np.exp(min(total_task / step, 20))
            print(f"  step {step:3d} | task={tl:.4f} | consist={cl:.4f} | PPL={avg_ppl:.2f}")

    epoch_ppl = np.exp(min(total_task / max(step, 1), 20))
    print(f"\nEpoch {epoch+1} done — steps={step}, avg PPL={epoch_ppl:.2f}\n")

# Results
print("=" * 60)
print("ASC CPU RUN COMPLETE")
print(f"Steps run:       {step}")
print(f"Final task loss: {task_losses[-1]:.4f}")
print(f"Final PPL:       {np.exp(min(task_losses[-1], 20)):.2f}")
print(f"First PPL:       {np.exp(min(task_losses[0], 20)):.2f}")
print(f"PPL change:      {np.exp(min(task_losses[0],20)) - np.exp(min(task_losses[-1],20)):+.2f}")
print(f"Avg consist loss:{np.mean(consist_losses):.4f}")
print("=" * 60)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(ppl_history, color="#E74C3C")
ax1.set_title("ASC — Task PPL per step")
ax1.set_xlabel("Step")
ax1.set_ylabel("PPL")
ax1.set_yscale("log")
ax1.grid(True)

ax2.plot(task_losses, label="task loss", color="#3498DB")
ax2.plot(consist_losses, label="consistency loss", color="#E67E22", alpha=0.7)
ax2.set_title("ASC — Loss curves")
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("asc_cpu_curves.png", dpi=150)
print("Plot saved -> asc_cpu_curves.png")

# Save
online_model.save_pretrained("asc_model_fixed")
tokenizer.save_pretrained("asc_model_fixed")
print("Model saved → asc_model_fixed/")
