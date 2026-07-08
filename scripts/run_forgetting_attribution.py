"""
Run STRATA's forgetting-attribution methodology on a TAR continual-learning run.

This is the executable demonstration of tar_lab/forgetting_attribution.py: it trains a real
torch model (MLP + BatchNorm) sequentially on a deterministic, hermetic 5-task benchmark that
INDUCES real catastrophic forgetting, then applies STRATA's pre-registered trichotomy exclusion
tests (substrate / timescale / basis) to attribute EACH forgetting event to a cause, and writes
an honest attribution report to tar_state/attribution/.

Self-contained (CPU-only, no downloads, seeded) so it is verifiable and cannot contend with the
live GPU daemon. The tasks share an input distribution but have task-specific decision
boundaries (maximal interference), so forgetting is guaranteed and the probes have signal.

The attribution module is HARNESS-AGNOSTIC: the same three functions can be driven by
tar_lab/generic_cl_runner outputs (real Split-CIFAR runs) — that live wiring is the follow-on
step; this script proves the methodology end-to-end on a TAR-run model first.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from tar_lab.forgetting_attribution import (  # noqa: E402
    snapshot_parameters, identify_forgetting_events, attribute_event, summarize_run,
)

_D = 24            # input dim
_HIDDEN = 64
_N_TASKS = 5
_N_TRAIN = 512
_N_TEST = 512      # >=512 held-out samples for the basis test (STRATA spec)
_EPOCHS = 20
_SEED = 0


class _MLP(nn.Module):
    """Small classifier WITH BatchNorm so the substrate test (params-restored, buffers-kept)
    genuinely discriminates weight-overwrite from buffer/representation drift."""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_D, _HIDDEN)
        self.bn1 = nn.BatchNorm1d(_HIDDEN)
        self.fc2 = nn.Linear(_HIDDEN, _HIDDEN)
        self.bn2 = nn.BatchNorm1d(_HIDDEN)
        self.head = nn.Linear(_HIDDEN, 2)

    def features(self, x):
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h2 = torch.relu(self.bn2(self.fc2(h1)))
        return h1, h2

    def forward(self, x):
        _, h2 = self.features(x)
        return self.head(h2)


def _make_tasks(rng: np.random.Generator):
    """Shared input distribution, task-specific linear decision boundary => interference."""
    tasks = []
    for _t in range(_N_TASKS):
        w = rng.normal(size=(_D,)); w /= np.linalg.norm(w)
        b = rng.normal() * 0.1

        def _gen(n, w=w, b=b):
            x = rng.normal(size=(n, _D)).astype(np.float32)
            y = ((x @ w + b) > 0).astype(np.int64)
            return torch.from_numpy(x), torch.from_numpy(y)

        tasks.append({"train": _gen(_N_TRAIN), "test": _gen(_N_TEST)})
    return tasks


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        return float((pred == y).float().mean().item())


def _train_task(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(_EPOCHS):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()


def _layer_activations(model: nn.Module, x: torch.Tensor) -> "dict[str, np.ndarray]":
    model.eval()
    with torch.no_grad():
        h1, h2 = model.features(x)
    return {"h1": h1.cpu().numpy(), "h2": h2.cpu().numpy()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run STRATA forgetting-attribution on a TAR CL run.")
    ap.add_argument("--workspace", default=r"E:\TAR\Thermodynamic-Continual-Learning-delivered")
    ap.add_argument("--seed", type=int, default=_SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tasks = _make_tasks(rng)
    model = _MLP()

    acc_matrix: "list[list[float | None]]" = [[None] * _N_TASKS for _ in range(_N_TASKS)]
    snap_before: "list[dict]" = [None] * _N_TASKS
    snap_after: "list[dict]" = [None] * _N_TASKS
    state_after: "list[dict]" = [None] * _N_TASKS

    for t in range(_N_TASKS):
        snap_before[t] = snapshot_parameters(model)
        _train_task(model, *tasks[t]["train"])
        snap_after[t] = snapshot_parameters(model)
        state_after[t] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for j in range(t + 1):
            acc_matrix[t][j] = _accuracy(model, *tasks[j]["test"])
        print(f"[task {t}] accuracies: " +
              "  ".join(f"t{j}={acc_matrix[t][j]:.3f}" for j in range(t + 1)), flush=True)

    events = identify_forgetting_events(acc_matrix)
    print(f"\n[attribution] {len(events)} forgetting event(s) at >=20% relative drop", flush=True)

    attributions = []
    for ev in events:
        k = ev.after_task
        after_state = state_after[k]

        def _eval_forgotten(overrides, _after_state=after_state, _j=ev.forgotten_task):
            model.load_state_dict(_after_state)          # restore full AFTER state (buffers incl.)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in overrides:
                        p.copy_(overrides[name])          # override params only; buffers kept at AFTER
            return _accuracy(model, *tasks[_j]["test"])

        model.load_state_dict(after_state)
        acts_forgotten = _layer_activations(model, tasks[ev.forgotten_task]["test"][0])
        acts_interfering = _layer_activations(model, tasks[k]["test"][0])

        att = attribute_event(
            ev, params_before=snap_before[k], params_after=snap_after[k],
            eval_forgotten_with_overrides=_eval_forgotten,
            acts_forgotten=acts_forgotten, acts_interfering=acts_interfering,
        )
        attributions.append(att)
        print(f"  event: task{ev.forgotten_task} forgotten after task{k} "
              f"(peak {ev.peak_acc:.3f} -> {ev.current_acc:.3f}) => "
              f"{att.attributed_causes or 'UNATTRIBUTED'}", flush=True)
        for tst in att.tests:
            print(f"      - {tst.cause:9s} excluded={tst.excluded}  {tst.detail}", flush=True)

    run = summarize_run(attributions)
    report = {
        "methodology": "STRATA trichotomy of forgetting causes (substrate/timescale/basis) — "
                       "faithful port; github.com/christophergardner-star/CF",
        "harness": "tar_lab.forgetting_attribution on a TAR-run MLP+BatchNorm; hermetic 5-task "
                   "shared-input/task-specific-boundary benchmark (CPU, deterministic)",
        "seed": args.seed,
        "acc_matrix": acc_matrix,
        **run.to_dict(),
        "honest_note": ("STRATA methodology applied to a TAR model to diagnose WHY it forgets; "
                        "this is a validation of STRATA's method, not a TAR performance result. "
                        "Unattributed events are legitimate trichotomy holes."),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = Path(args.workspace) / "tar_state" / "attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"forgetting_attribution_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[attribution] {run.n_attributed}/{run.n_events} events attributed; "
          f"unattributed magnitude fraction={run.unattributed_fraction:.3f}; "
          f"trichotomy_incomplete={run.incomplete}", flush=True)
    print(f"[attribution] report -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
