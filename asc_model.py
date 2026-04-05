"""
asc_model.py -- ASC (Adversarial Self-Consistency) Model Family
===============================================================

A training paradigm that learns invariant representations through
latent adversarial consistency. Any causal transformer backbone +
a small LatentWarp head + an EMA target model.

Model family
------------
  ASC-124M  -- gpt2           (124M params)
  ASC-355M  -- gpt2-medium    (355M params)
  ASC-1B    -- gpt2-large + custom config (~1B)
  ASC-7B    -- use Llama/Mistral 7B backbone

Key new components (all else is standard transformer):
  LatentWarp  -- 2-layer MLP, ~0.1-0.5% of total params
  Target      -- frozen EMA copy of the base model
  Dual loss   -- task (causal LM) + consistency (warped latents)

Usage
-----
    from asc_model import ASCConfig, ASCForCausalLM

    config = ASCConfig.for_size("124M")
    model = ASCForCausalLM(config)

    # training step
    task_loss, consist_loss = model(input_ids=x, labels=y)
    total = task_loss + 0.3 * consist_loss
    total.backward()
    optimizer.step()
    model.update_target()

    # inference (consistency path disabled)
    model.eval()
    logits = model(input_ids=x)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------

_SIZE_PRESETS = {
    "124M":  "gpt2",
    "355M":  "gpt2-medium",
    "774M":  "gpt2-large",
    "1558M": "gpt2-xl",
}


# ---------------------------------------------------------------------------
# ASCConfig
# ---------------------------------------------------------------------------

class ASCConfig(PretrainedConfig):
    """Configuration for the ASC model family.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model ID or local path for the backbone.
    warp_dim : int
        Hidden dimension of the LatentWarp MLP.
    warp_init_scale : float
        Initial scale factor for the warp perturbation (keeps early
        training stable; typically 0.01–0.1).
    consistency_lambda : float
        Default weighting of the consistency loss relative to task loss.
        Can be overridden in the training loop.
    ema_decay : float
        EMA decay for the target model update.
    """

    model_type = "asc"

    def __init__(
        self,
        base_model_name: str = "gpt2",
        warp_dim: int = 256,
        warp_init_scale: float = 0.05,
        consistency_lambda: float = 0.3,
        ema_decay: float = 0.995,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.warp_dim = warp_dim
        self.warp_init_scale = warp_init_scale
        self.consistency_lambda = consistency_lambda
        self.ema_decay = ema_decay

    @classmethod
    def for_size(cls, size: str, **kwargs) -> "ASCConfig":
        """Return a config for a named model size.

        Parameters
        ----------
        size : str
            One of '124M', '355M', '774M', '1558M'.
        """
        if size not in _SIZE_PRESETS:
            raise ValueError(
                f"Unknown size '{size}'. Choose from: {list(_SIZE_PRESETS)}"
            )
        return cls(base_model_name=_SIZE_PRESETS[size], **kwargs)


# ---------------------------------------------------------------------------
# LatentWarp
# ---------------------------------------------------------------------------

class LatentWarp(nn.Module):
    """Small 2-layer MLP that produces a latent perturbation.

    Designed to be <<1% of backbone parameter count. The perturbation
    magnitude is controlled by a learnable scalar `scale`, initialized
    small so early training is not destabilised.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the backbone hidden states.
    warp_dim : int
        Internal bottleneck dimension.
    init_scale : float
        Initial value of the learnable scale parameter.
    """

    def __init__(self, hidden_dim: int, warp_dim: int = 256, init_scale: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, warp_dim),
            nn.ReLU(),
            nn.Linear(warp_dim, hidden_dim),
        )
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply additive warp perturbation.

        Parameters
        ----------
        hidden_states : Tensor of shape (B, T, D)

        Returns
        -------
        Tensor of shape (B, T, D)  -- perturbed hidden states
        """
        return hidden_states + self.net(hidden_states) * self.scale

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# ASCForCausalLM
# ---------------------------------------------------------------------------

class ASCForCausalLM(nn.Module):
    """ASC causal language model.

    Wraps any HuggingFace causal LM backbone with:
      - A LatentWarp head
      - An EMA target model (frozen, no gradients)
      - A dual forward pass for training

    During eval / inference, the consistency path is disabled and the
    model behaves identically to the bare backbone.

    Parameters
    ----------
    config : ASCConfig
    """

    def __init__(self, config: ASCConfig):
        super().__init__()
        self.config = config

        # Backbone (online model — trained)
        self.base = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        hidden_dim = self.base.config.hidden_size

        # LatentWarp (ASC-specific; tiny relative to backbone)
        self.warp = LatentWarp(
            hidden_dim=hidden_dim,
            warp_dim=config.warp_dim,
            init_scale=config.warp_init_scale,
        )

        # Target model (EMA; frozen)
        self.target = copy.deepcopy(self.base)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Training mode
        -------------
        Returns (task_loss, consistency_loss) when `labels` is provided.
        Total loss = task_loss + lambda * consistency_loss.

        Eval / inference mode
        ---------------------
        Returns the backbone CausalLMOutput directly (logits, etc.).
        Consistency path is skipped entirely.

        Parameters
        ----------
        input_ids : LongTensor (B, T), optional
        attention_mask : BoolTensor (B, T), optional
        labels : LongTensor (B, T), optional
        inputs_embeds : FloatTensor (B, T, D), optional

        Returns
        -------
        Training: Tuple[Tensor, Tensor]  -- (task_loss, consistency_loss)
        Eval:     CausalLMOutput
        """
        base_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # Remove None values to avoid HF signature complaints
        base_kwargs = {k: v for k, v in base_kwargs.items() if v is not None}

        # ── Standard task forward ──────────────────────────────────────
        task_out = self.base(**base_kwargs)

        if not self.training or labels is None:
            return task_out

        # ── Consistency forward (training only) ────────────────────────
        with torch.no_grad():
            tgt_kwargs = {k: v for k, v in base_kwargs.items()
                          if k not in ("labels",)}
            tgt_out = self.target(**tgt_kwargs, output_hidden_states=True)
            clean_h = tgt_out.hidden_states[-1]  # final layer hidden states

        warped_h = self.warp(clean_h)

        consist_out = self.base(
            inputs_embeds=warped_h,
            labels=labels,
            attention_mask=attention_mask,
        )

        return task_out.loss, consist_out.loss

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target(self, decay: Optional[float] = None) -> None:
        """Update the target model via EMA.

        Call once per optimizer step after `optimizer.step()`.

        Parameters
        ----------
        decay : float, optional
            Override the config EMA decay for this step.
        """
        d = decay if decay is not None else self.config.ema_decay
        for p_base, p_tgt in zip(self.base.parameters(), self.target.parameters()):
            p_tgt.data.mul_(d).add_(p_base.data, alpha=1.0 - d)

    # ------------------------------------------------------------------
    # Generation helpers (delegate to backbone)
    # ------------------------------------------------------------------

    def generate(self, *args, **kwargs):
        """Delegate generate() to the backbone."""
        self.eval()
        return self.base.generate(*args, **kwargs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def parameters_trainable(self):
        """Return only trainable parameters (base + warp; not target)."""
        return list(self.base.parameters()) + list(self.warp.parameters())

    def param_summary(self) -> dict:
        """Return a summary of parameter counts."""
        base_n = sum(p.numel() for p in self.base.parameters())
        warp_n = self.warp.param_count
        return {
            "base_params": base_n,
            "warp_params": warp_n,
            "warp_pct": round(100.0 * warp_n / base_n, 3),
            "total_trainable": base_n + warp_n,
            "target_params": sum(p.numel() for p in self.target.parameters()),
        }

    def save(self, path: str) -> None:
        """Save base model + warp weights to `path`."""
        import os, json
        os.makedirs(path, exist_ok=True)
        self.base.save_pretrained(path)
        torch.save(self.warp.state_dict(), os.path.join(path, "warp.pt"))
        with open(os.path.join(path, "asc_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ASCForCausalLM":
        """Load a saved ASC model from `path`."""
        import json, os
        with open(os.path.join(path, "asc_config.json")) as f:
            cfg_dict = json.load(f)
        config = ASCConfig(**cfg_dict)
        # Load base from saved weights, not the original HF checkpoint
        config.base_model_name = path
        model = cls(config)
        warp_path = os.path.join(path, "warp.pt")
        if os.path.exists(warp_path):
            model.warp.load_state_dict(torch.load(warp_path, map_location="cpu"))
        return model
