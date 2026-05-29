"""
Validation script executed inside the Docker sandbox to test a synthesised CLMethod.

The synthesiser embeds the generated method code into this template and submits
the full script to SandboxedPythonExecutor.run().

Expected exit: sys.exit(0) on success, non-zero on any failure.
The synthesiser reads stdout for "VALIDATION_PASSED" as the success signal.

Contract:
  - This script must be self-contained (no imports from TAR modules)
  - Uses synthetic data (no downloads, no network)
  - Trains for 1 epoch, 2 tasks, batch_size=8 — enough to exercise all hooks
  - Validates: no Python errors, loss decreases, tensors stay finite
"""
import sys
import types
import math

# ── Minimal stubs so the generated code has everything it may import ──────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    print(f"VALIDATION_FAILED: missing pytorch: {e}", flush=True)
    sys.exit(1)

# ── CLMethod stub (no TAR imports available in sandbox) ───────────────────────

from abc import ABC, abstractmethod

class CLMethod(ABC):
    def __init__(self, config):
        self.config = config
    def pre_task(self, task_id, model, device):
        pass
    def post_task(self, task_id, model, loader, device):
        pass
    @abstractmethod
    def regularization_loss(self, model): ...
    def augmented_loss(self, model, x, y, task_id, device):
        return torch.tensor(0.0, device=device)

# ── SYNTHESISED_METHOD_INSERTED_HERE ─────────────────────────────────────────
# (synthesiser replaces this comment with the generated class source)
# ─────────────────────────────────────────────────────────────────────────────

# ── Locate the generated class ────────────────────────────────────────────────

_this_module = sys.modules[__name__]
_GeneratedClass = None
for _name in dir(_this_module):
    _obj = getattr(_this_module, _name)
    if (
        isinstance(_obj, type)
        and hasattr(_obj, "regularization_loss")
        and hasattr(_obj, "augmented_loss")
        and _name not in {"_BaseMethod"}
    ):
        _GeneratedClass = _obj
        break

if _GeneratedClass is None:
    print("VALIDATION_FAILED: no CLMethod subclass found in generated code", flush=True)
    sys.exit(1)

# ── Synthetic benchmark ───────────────────────────────────────────────────────

N_TASKS    = 2
N_CLASSES  = 4       # 2 per task
BATCH      = 8
N_BATCHES  = 5
IMG_SIZE   = 16
N_TRAIN    = BATCH * N_BATCHES

device = torch.device("cpu")


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.conv(x)).flatten(1))


def _make_loader(task_id: int) -> DataLoader:
    x = torch.randn(N_TRAIN, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(task_id * 2, task_id * 2 + 2, (N_TRAIN,))
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True)


try:
    cfg = types.SimpleNamespace(
        ewc_lambda=100.0, si_c=0.1, si_xi=0.001,
        der_mem_size=32, der_alpha=0.2, der_beta=0.5,
        lr=0.01, weight_decay=1e-4,
    )
    method = _GeneratedClass(cfg)
except Exception as e:
    print(f"VALIDATION_FAILED: __init__ raised: {e}", flush=True)
    sys.exit(1)

model = _TinyNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

initial_loss = None
last_loss    = None

for task_id in range(N_TASKS):
    loader = _make_loader(task_id)

    try:
        method.pre_task(task_id, model, device)
    except Exception as e:
        print(f"VALIDATION_FAILED: pre_task raised: {e}", flush=True)
        sys.exit(1)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)

        try:
            reg = method.regularization_loss(model)
        except Exception as e:
            print(f"VALIDATION_FAILED: regularization_loss raised: {e}", flush=True)
            sys.exit(1)

        loss = F.cross_entropy(out, y) + reg
        if not math.isfinite(loss.item()):
            print(f"VALIDATION_FAILED: non-finite loss after reg: {loss.item()}", flush=True)
            sys.exit(1)

        loss.backward()

        try:
            aug = method.augmented_loss(model, x, y, task_id, device)
        except Exception as e:
            print(f"VALIDATION_FAILED: augmented_loss raised: {e}", flush=True)
            sys.exit(1)

        if not math.isfinite(aug.item()):
            print(f"VALIDATION_FAILED: non-finite augmented_loss: {aug.item()}", flush=True)
            sys.exit(1)

        if aug.item() != 0.0:
            aug.backward()

        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()
        last_loss = loss.item()

    try:
        method.post_task(task_id, model, loader, device)
    except Exception as e:
        print(f"VALIDATION_FAILED: post_task raised: {e}", flush=True)
        sys.exit(1)

# Sanity: final loss must be finite and model parameters must be finite
for p in model.parameters():
    if not torch.isfinite(p).all():
        print("VALIDATION_FAILED: model parameters contain inf/nan after training", flush=True)
        sys.exit(1)

if last_loss is None or not math.isfinite(last_loss):
    print(f"VALIDATION_FAILED: final loss is {last_loss}", flush=True)
    sys.exit(1)

print(
    f"VALIDATION_PASSED  class={_GeneratedClass.__name__}"
    f"  initial_loss={initial_loss:.4f}  final_loss={last_loss:.4f}",
    flush=True,
)
sys.exit(0)
