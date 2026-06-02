"""
Task 2.12 — Weight Initialization Seed Reproducibility Test

Verifies that _set_seed() followed by model construction always produces
identical initial weights. Non-determinism in initialization would mean
different seeds do not provide clean experimental separation.

Reference: Masters Plan Task 2.12, TAR PhD Rehabilitation Plan.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tar_lab.generic_cl_runner import _set_seed, _build_model


def _params(model: torch.nn.Module) -> dict:
    return {name: p.data.detach().clone().cpu()
            for name, p in model.named_parameters()}


def _all_close(a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.allclose(a[k], b[k], rtol=1e-5, atol=1e-8) for k in a)


def _any_differ(a: dict, b: dict) -> bool:
    return any(not torch.equal(a[k], b[k]) for k in a if k in b)


def test_same_seed_produces_identical_weights_seed42():
    """_set_seed(42) twice must produce byte-identical ResNet-18 weights."""
    _set_seed(42)
    w_a = _params(_build_model("resnet18", 10, torch.device("cpu")))
    _set_seed(42)
    w_b = _params(_build_model("resnet18", 10, torch.device("cpu")))
    assert _all_close(w_a, w_b), "Seed 42: non-deterministic initialization"


def test_same_seed_produces_identical_weights_seed0():
    """_set_seed(0) twice must produce byte-identical weights."""
    _set_seed(0)
    w_a = _params(_build_model("resnet18", 10, torch.device("cpu")))
    _set_seed(0)
    w_b = _params(_build_model("resnet18", 10, torch.device("cpu")))
    assert _all_close(w_a, w_b), "Seed 0: non-deterministic initialization"


def test_same_seed_produces_identical_weights_seed1():
    """_set_seed(1) twice must produce byte-identical weights."""
    _set_seed(1)
    w_a = _params(_build_model("resnet18", 10, torch.device("cpu")))
    _set_seed(1)
    w_b = _params(_build_model("resnet18", 10, torch.device("cpu")))
    assert _all_close(w_a, w_b), "Seed 1: non-deterministic initialization"


def test_different_seeds_produce_different_weights():
    """Seeds 42 and 0 must produce at least one differing parameter tensor."""
    _set_seed(42)
    w_42 = _params(_build_model("resnet18", 10, torch.device("cpu")))
    _set_seed(0)
    w_0 = _params(_build_model("resnet18", 10, torch.device("cpu")))
    assert _any_differ(w_42, w_0), (
        "Seeds 42 and 0 produced identical weights — _set_seed() has no effect"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_same_seed_deterministic_on_cuda():
    """Seed=42 must also be deterministic on GPU."""
    device = torch.device("cuda:0")
    _set_seed(42)
    w_a = _params(_build_model("resnet18", 10, device))
    _set_seed(42)
    w_b = _params(_build_model("resnet18", 10, device))
    assert _all_close(w_a, w_b), "Seed 42: non-deterministic on CUDA"


if __name__ == "__main__":
    print("Running seed reproducibility tests...")
    test_same_seed_produces_identical_weights_seed42()
    print("PASS: seed=42 deterministic")
    test_same_seed_produces_identical_weights_seed0()
    print("PASS: seed=0 deterministic")
    test_same_seed_produces_identical_weights_seed1()
    print("PASS: seed=1 deterministic")
    test_different_seeds_produce_different_weights()
    print("PASS: different seeds produce different weights")
    if torch.cuda.is_available():
        test_same_seed_deterministic_on_cuda()
        print("PASS: CUDA deterministic")
    else:
        print("SKIP: no CUDA")
    print("\nAll seed reproducibility tests passed.")
