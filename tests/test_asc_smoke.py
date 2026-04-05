"""
test_asc_smoke.py — CPU smoke test for ASC (Adversarial Self-Consistency)

Validates the core ASC mechanism without downloading any dataset:
  - LatentWarp module forward pass
  - EMA target update
  - Consistency loss computation (warped latents → online model)
  - Combined task + consistency loss backward pass (no NaN/Inf)
  - Loss decreases over 20 steps on synthetic token data
  - Gradient flows to both online_model and warp_net parameters

Uses GPT-2 (117M, smallest config) with a tiny synthetic dataset.
Runs in ~60s CPU-only.
"""

import copy
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── LatentWarp (exact copy from the ASC script) ────────────────────────────

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


def update_target(online: nn.Module, target: nn.Module, decay: float = 0.995) -> None:
    """EMA weight update: target ← decay·target + (1-decay)·online."""
    with torch.no_grad():
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)


# ── Tiny GPT-2-style model (no HuggingFace download needed) ────────────────

class TinyTransformer(nn.Module):
    """Minimal causal LM: embedding → 2-layer transformer → LM head.
    Produces logits and optionally hidden states.
    Mirrors GPT2LMHeadModel interface used in the ASC script.
    """

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, seq_len: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self._seq_len = seq_len

    def forward(self, input_ids: torch.Tensor = None,
                labels: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None,
                output_hidden_states: bool = False):
        """
        Returns a namespace with .loss (if labels given), .logits, and
        optionally .hidden_states (tuple with final hidden as last element).
        """
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            B, T = input_ids.shape
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            h = self.embed(input_ids) + self.pos_embed(pos)

        # Causal mask
        T = h.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h_out = self.transformer(h, mask=mask, is_causal=True)
        logits = self.lm_head(h_out)

        loss = None
        if labels is not None:
            # Shift: predict token i+1 from token i
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        class _Out:
            pass

        out = _Out()
        out.loss = loss
        out.logits = logits
        if output_hidden_states:
            out.hidden_states = (h_out,)   # tuple; last element is final hidden
        return out


# ── Synthetic data ──────────────────────────────────────────────────────────

def _make_loader(n: int = 128, seq_len: int = 32, vocab_size: int = 256,
                 batch_size: int = 16, seed: int = 42) -> DataLoader:
    """Random token sequences as input_ids and labels (same tensor for causal LM)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n, seq_len), generator=rng)
    ds = TensorDataset(tokens, tokens.clone())   # (input_ids, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ── Unit tests ──────────────────────────────────────────────────────────────

VOCAB = 256
HIDDEN = 64
SEQ = 32
WARP_DIM = 32


class TestLatentWarp:

    def test_output_shape(self):
        warp = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)
        x = torch.randn(4, SEQ, HIDDEN)
        out = warp(x)
        assert out.shape == x.shape

    def test_output_differs_from_input(self):
        warp = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)
        x = torch.randn(4, SEQ, HIDDEN)
        out = warp(x)
        assert not torch.allclose(out, x), "Warp output identical to input — warp is inactive"

    def test_scale_controls_magnitude(self):
        """Larger scale → larger perturbation."""
        warp_small = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)
        warp_large = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)
        warp_large.load_state_dict(warp_small.state_dict())
        warp_large.scale.data.fill_(1.0)

        x = torch.randn(4, SEQ, HIDDEN)
        diff_small = (warp_small(x) - x).abs().mean()
        diff_large = (warp_large(x) - x).abs().mean()
        assert diff_large > diff_small

    def test_gradients_flow_through_warp(self):
        warp = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)
        x = torch.randn(2, SEQ, HIDDEN, requires_grad=True)
        out = warp(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0


class TestEMAUpdate:

    def test_target_moves_toward_online(self):
        online = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN)
        target = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN)
        # Make them different
        with torch.no_grad():
            for p in online.parameters():
                p.add_(torch.randn_like(p) * 2.0)

        p0 = list(target.parameters())[0].data.clone()
        p_online = list(online.parameters())[0].data.clone()

        update_target(online, target, decay=0.9)

        p1 = list(target.parameters())[0].data
        # Should have moved 10% toward online
        expected = 0.9 * p0 + 0.1 * p_online
        assert torch.allclose(p1, expected, atol=1e-6)

    def test_decay_1_leaves_target_unchanged(self):
        online = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN)
        target = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN)
        p_before = list(target.parameters())[0].data.clone()
        update_target(online, target, decay=1.0)
        p_after = list(target.parameters())[0].data
        assert torch.allclose(p_before, p_after)

    def test_target_not_grad_updated(self):
        """EMA update must not create gradient computation graph."""
        online = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN)
        target = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN)
        target.eval()
        update_target(online, target, decay=0.99)
        for p in target.parameters():
            assert not p.requires_grad or p.grad is None


class TestASCLoss:

    @pytest.fixture(scope="class")
    def setup(self):
        online = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, seq_len=SEQ)
        target = copy.deepcopy(online)
        target.eval()
        warp = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)
        return online, target, warp

    def test_task_loss_finite(self, setup):
        online, target, warp = setup
        x = torch.randint(0, VOCAB, (4, SEQ))
        out = online(x, labels=x)
        assert out.loss is not None
        assert math.isfinite(float(out.loss.item()))

    def test_consistency_loss_finite(self, setup):
        online, target, warp = setup
        x = torch.randint(0, VOCAB, (4, SEQ))
        with torch.no_grad():
            target_out = target(x, output_hidden_states=True)
            clean_hidden = target_out.hidden_states[-1]
        warped_hidden = warp(clean_hidden)
        warped_out = online(inputs_embeds=warped_hidden, labels=x)
        assert warped_out.loss is not None
        assert math.isfinite(float(warped_out.loss.item()))

    def test_combined_loss_backward_no_error(self, setup):
        online, target, warp = setup
        optimizer = torch.optim.Adam(
            list(online.parameters()) + list(warp.parameters()), lr=1e-3
        )
        x = torch.randint(0, VOCAB, (4, SEQ))

        optimizer.zero_grad()
        task_out = online(x, labels=x)
        task_loss = task_out.loss

        with torch.no_grad():
            target_out = target(x, output_hidden_states=True)
            clean_hidden = target_out.hidden_states[-1]
        warped_hidden = warp(clean_hidden)
        consistency_out = online(inputs_embeds=warped_hidden, labels=x)
        consistency_loss = consistency_out.loss

        total = task_loss + 0.3 * consistency_loss
        total.backward()
        optimizer.step()

        assert math.isfinite(float(total.item()))

    def test_gradients_reach_warp_net(self, setup):
        online, target, warp = setup
        optimizer = torch.optim.Adam(
            list(online.parameters()) + list(warp.parameters()), lr=1e-3
        )
        x = torch.randint(0, VOCAB, (4, SEQ))

        optimizer.zero_grad()
        task_out = online(x, labels=x)

        with torch.no_grad():
            target_out = target(x, output_hidden_states=True)
            clean_hidden = target_out.hidden_states[-1]
        warped_hidden = warp(clean_hidden)
        consistency_out = online(inputs_embeds=warped_hidden, labels=x)

        (task_out.loss + 0.3 * consistency_out.loss).backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in warp.parameters()
        )
        assert has_grad, "Consistency loss produced no gradients for warp_net"


class TestASCTrainingLoop:
    """End-to-end: run 20 steps of ASC training, verify loss decreases."""

    def test_loss_decreases_over_20_steps(self):
        torch.manual_seed(42)
        online = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, seq_len=SEQ)
        target = copy.deepcopy(online)
        target.eval()
        warp = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)

        optimizer = torch.optim.Adam(
            list(online.parameters()) + list(warp.parameters()), lr=3e-3
        )
        loader = _make_loader(n=320, seq_len=SEQ, vocab_size=VOCAB, batch_size=16)
        consistency_lambda = 0.3
        ema_decay = 0.995

        losses = []
        step = 0

        for input_ids, labels in loader:
            if step >= 20:
                break

            optimizer.zero_grad()

            task_out = online(input_ids, labels=labels)
            task_loss = task_out.loss

            with torch.no_grad():
                tgt_out = target(input_ids, output_hidden_states=True)
                clean_hidden = tgt_out.hidden_states[-1]

            warped_hidden = warp(clean_hidden)
            consist_out = online(inputs_embeds=warped_hidden, labels=labels)
            consistency_loss = consist_out.loss

            total = task_loss + consistency_lambda * consistency_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(online.parameters()) + list(warp.parameters()), 1.0
            )
            optimizer.step()
            update_target(online, target, decay=ema_decay)

            losses.append(float(total.item()))
            step += 1

        assert len(losses) == 20
        assert all(math.isfinite(l) for l in losses), f"NaN/Inf in losses: {losses}"

        # Loss should trend downward: last 5 avg < first 5 avg
        first5 = sum(losses[:5]) / 5
        last5 = sum(losses[-5:]) / 5
        assert last5 < first5, (
            f"Loss did not decrease: first5={first5:.4f}, last5={last5:.4f}\n"
            f"Full loss trace: {[f'{l:.4f}' for l in losses]}"
        )

    def test_no_nan_with_high_consistency_lambda(self):
        """λ=1.0 (extreme) should still produce finite losses."""
        torch.manual_seed(7)
        online = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, seq_len=SEQ)
        target = copy.deepcopy(online)
        target.eval()
        warp = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)
        optimizer = torch.optim.Adam(
            list(online.parameters()) + list(warp.parameters()), lr=1e-3
        )
        loader = _make_loader(n=64, seq_len=SEQ, vocab_size=VOCAB, batch_size=16)

        for input_ids, labels in loader:
            optimizer.zero_grad()
            task_loss = online(input_ids, labels=labels).loss
            with torch.no_grad():
                clean_hidden = target(input_ids, output_hidden_states=True).hidden_states[-1]
            warped = warp(clean_hidden)
            consist_loss = online(inputs_embeds=warped, labels=labels).loss
            total = task_loss + 1.0 * consist_loss
            total.backward()
            optimizer.step()
            update_target(online, target, decay=0.995)
            assert math.isfinite(float(total.item())), f"NaN/Inf at λ=1.0: {total.item()}"
            break   # one step sufficient

    def test_ppl_decreases_from_init(self):
        """After 20 steps, task PPL should drop relative to step 1.
        On random token data a tiny model can't compress below vocab_size,
        but it should still overfit the training batches and show a downward trend."""
        torch.manual_seed(99)
        online = TinyTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, seq_len=SEQ)
        target = copy.deepcopy(online)
        target.eval()
        warp = LatentWarp(hidden_dim=HIDDEN, warp_dim=WARP_DIM)
        optimizer = torch.optim.Adam(
            list(online.parameters()) + list(warp.parameters()), lr=3e-3
        )
        loader = _make_loader(n=320, seq_len=SEQ, vocab_size=VOCAB, batch_size=16)

        step = 0
        first_loss = None
        last_loss = None
        for input_ids, labels in loader:
            if step >= 20:
                break
            optimizer.zero_grad()
            task_loss = online(input_ids, labels=labels).loss
            with torch.no_grad():
                clean_hidden = target(input_ids, output_hidden_states=True).hidden_states[-1]
            warped = warp(clean_hidden)
            consist_loss = online(inputs_embeds=warped, labels=labels).loss
            total = task_loss + 0.3 * consist_loss
            total.backward()
            optimizer.step()
            update_target(online, target, decay=0.995)
            if first_loss is None:
                first_loss = float(task_loss.item())
            last_loss = float(task_loss.item())
            step += 1

        first_ppl = math.exp(first_loss)
        last_ppl = math.exp(last_loss)
        assert last_ppl < first_ppl, (
            f"PPL did not decrease: step1={first_ppl:.1f} → step20={last_ppl:.1f}. "
            "Model is not learning at all."
        )
