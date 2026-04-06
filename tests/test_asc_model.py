"""
tests/test_asc_model.py -- ASCForCausalLM unit tests
CPU-only, no dataset download.
"""

import copy
import math
import tempfile
import pytest
import torch

from asc_model import ASCConfig, ASCForCausalLM, LatentWarp


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_config():
    """Build one tiny offline ASC model shared across tests."""
    config = ASCConfig(
        base_model_name="__tiny_gpt2__",
        warp_dim=32,
        backbone_config_overrides={
            "vocab_size": 256,
            "n_positions": 32,
            "n_ctx": 32,
            "n_embd": 64,
            "n_layer": 2,
            "n_head": 4,
        },
    )
    model = ASCForCausalLM(config)
    return model, config


@pytest.fixture(scope="module")
def sample_batch():
    torch.manual_seed(42)
    x = torch.randint(0, 256, (2, 16))
    return {"input_ids": x, "labels": x.clone()}


# ── ASCConfig tests ───────────────────────────────────────────────────────────

class TestASCConfig:

    def test_for_size_124M(self):
        c = ASCConfig.for_size("124M")
        assert c.base_model_name == "gpt2"

    def test_for_size_355M(self):
        c = ASCConfig.for_size("355M")
        assert c.base_model_name == "gpt2-medium"

    def test_for_size_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown size"):
            ASCConfig.for_size("999B")

    def test_defaults(self):
        c = ASCConfig()
        assert c.consistency_lambda == 0.3
        assert c.ema_decay == 0.995
        assert c.warp_dim == 256


# ── LatentWarp tests ──────────────────────────────────────────────────────────

class TestLatentWarp:

    def test_output_shape(self):
        warp = LatentWarp(hidden_dim=768, warp_dim=64)
        x = torch.randn(2, 16, 768)
        assert warp(x).shape == x.shape

    def test_perturbation_nonzero(self):
        warp = LatentWarp(hidden_dim=768, warp_dim=64)
        x = torch.randn(2, 16, 768)
        assert not torch.allclose(warp(x), x)

    def test_grad_flows(self):
        warp = LatentWarp(hidden_dim=768, warp_dim=64)
        x = torch.randn(2, 8, 768, requires_grad=True)
        warp(x).sum().backward()
        assert x.grad is not None


# ── ASCForCausalLM tests ──────────────────────────────────────────────────────

class TestASCForCausalLM:

    def test_param_summary(self, model_and_config):
        model, _ = model_and_config
        s = model.param_summary()
        assert s["base_params"] > 0
        assert s["warp_params"] > 0
        assert s["warp_pct"] < 100.0
        assert s["total_trainable"] == s["base_params"] + s["warp_params"]

    def test_target_has_no_grad(self, model_and_config):
        model, _ = model_and_config
        for p in model.target.parameters():
            assert not p.requires_grad

    def test_train_forward_returns_two_losses(self, model_and_config, sample_batch):
        model, _ = model_and_config
        model.train()
        result = model(**sample_batch)
        assert isinstance(result, tuple) and len(result) == 2
        task_loss, consist_loss = result
        assert math.isfinite(float(task_loss))
        assert math.isfinite(float(consist_loss))

    def test_eval_forward_returns_output(self, model_and_config, sample_batch):
        model, _ = model_and_config
        model.eval()
        with torch.no_grad():
            out = model(input_ids=sample_batch["input_ids"])
        assert hasattr(out, "logits")
        assert out.logits.shape == (2, 16, 256)

    def test_backward_no_error(self, model_and_config, sample_batch):
        model, _ = model_and_config
        model.train()
        task_loss, consist_loss = model(**sample_batch)
        (task_loss + 0.3 * consist_loss).backward()

    def test_warp_receives_gradients(self, model_and_config, sample_batch):
        model, _ = model_and_config
        model.train()
        for p in model.warp.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss = model.warp_ascent_loss(**sample_batch)
        (-loss).backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.warp.parameters()
        )
        assert has_grad, "Warp network received no gradients from warp_ascent_loss"

    def test_target_not_updated_by_backward(self, model_and_config, sample_batch):
        """Backward should never touch target params."""
        model, _ = model_and_config
        model.train()
        # snapshot target
        tgt_snapshot = {n: p.data.clone()
                        for n, p in model.target.named_parameters()}
        task_loss, consist_loss = model(**sample_batch)
        (task_loss + 0.3 * consist_loss).backward()
        for n, p in model.target.named_parameters():
            assert torch.allclose(p.data, tgt_snapshot[n]), \
                f"Target param {n} changed during backward"

    def test_ema_update_moves_target(self, model_and_config):
        model, _ = model_and_config
        # Perturb base
        with torch.no_grad():
            for p in model.base.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        tgt_before = list(model.target.parameters())[0].data.clone()
        base_now = list(model.base.parameters())[0].data.clone()
        model.update_target(decay=0.9)
        tgt_after = list(model.target.parameters())[0].data
        expected = 0.9 * tgt_before + 0.1 * base_now
        assert torch.allclose(tgt_after, expected, atol=1e-6)

    def test_parameters_trainable_excludes_target(self, model_and_config):
        model, _ = model_and_config
        trainable_ids = {id(p) for p in model.parameters_trainable()}
        for p in model.target.parameters():
            assert id(p) not in trainable_ids

    def test_save_load_roundtrip(self, model_and_config, sample_batch):
        model, _ = model_and_config
        model.eval()
        with torch.no_grad():
            logits_before = model(input_ids=sample_batch["input_ids"]).logits

        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            model2 = ASCForCausalLM.load(tmp)
            model2.eval()
            with torch.no_grad():
                logits_after = model2(input_ids=sample_batch["input_ids"]).logits

        assert torch.allclose(logits_before, logits_after, atol=1e-4), \
            "Logits changed after save/load"

    def test_no_nan_in_losses(self, model_and_config, sample_batch):
        model, _ = model_and_config
        model.train()
        for _ in range(5):
            task_loss, consist_loss = model(**sample_batch)
            assert not torch.isnan(task_loss), "NaN in task_loss"
            assert not torch.isnan(consist_loss), "NaN in consist_loss"
            assert not torch.isinf(task_loss), "Inf in task_loss"
            assert not torch.isinf(consist_loss), "Inf in consist_loss"


# ── Training loop smoke ────────────────────────────────────────────────────────

class TestASCTrainingLoop:

    def test_10_steps_loss_decreases(self):
        """10-step loop: last-5 avg task loss < first-5 avg."""
        torch.manual_seed(0)
        config = ASCConfig(
            base_model_name="__tiny_gpt2__",
            warp_dim=32,
            backbone_config_overrides={
                "vocab_size": 256,
                "n_positions": 32,
                "n_ctx": 32,
                "n_embd": 64,
                "n_layer": 2,
                "n_head": 4,
            },
        )
        model = ASCForCausalLM(config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters_trainable(), lr=1e-3)
        losses = []

        for _ in range(10):
            x = torch.randint(0, 256, (2, 16))
            optimizer.zero_grad()
            task_loss, consist_loss = model(input_ids=x, labels=x)
            (task_loss + 0.3 * consist_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters_trainable(), 1.0)
            optimizer.step()
            model.update_target()
            losses.append(float(task_loss.item()))

        first5 = sum(losses[:5]) / 5
        last5 = sum(losses[5:]) / 5
        assert last5 < first5, (
            f"Task loss did not decrease: first5={first5:.4f} last5={last5:.4f}"
        )
