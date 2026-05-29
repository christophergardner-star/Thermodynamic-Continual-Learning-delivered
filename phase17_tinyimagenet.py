"""
Phase 17 - Scale-up: Split-TinyImageNet
=======================================

Import-safe native runner for the TinyImageNet scale-up suite.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_repo = str(Path(__file__).resolve().parent)
sys.path.insert(0, _repo)
from tar_storage import ensure_workspace_layout, resolve_workspace
workspace = str(ensure_workspace_layout(resolve_workspace(Path(_repo)), repo_root=Path(_repo)))

from tar_lab.thermoobserver import ActivationThermoObserver
from tar_optimizer_backend import build_optimizer, maybe_apply_optimizer_safety
from tar_runtime_tracking import tracked_process, update_stage
from tar_suite_checkpoint import (
    append_completed_seed,
    build_suite_state,
    checkpoint_path,
    clear_suite_state,
    load_suite_state,
    recover_suite_state_from_log,
    save_suite_state,
)
from tar_suite_logging import tee_console
from tar_lab.manifest import load_and_verify_manifest, ManifestGateError, write_refuse_note
from tar_lab.result_artifacts import collect_environment_snapshot, write_canonical_comparison_result

SEEDS = [42, 0, 1]
BACKBONE = "resnet18"
EPOCHS = 40
METHODS = ["tcl", "ewc", "sgd_baseline"]
N_TASKS = 10
CLASSES_PER_TASK = 20
EWC_LAMBDA = 100.0
TCL_ALPHA = 0.5
TCL_PENALTY_LAMBDA = 0.01
TCL_ORDERED_LR = 0.5
TCL_DISORDERED_LR = 1.2
BATCH_SIZE = 64
BASE_LR = 0.01

HF_DATASET_NAME = "Maysee/tiny-imagenet"
OUT_PATH = Path(workspace) / "tar_state" / "comparisons" / "phase17_tinyimagenet.json"
DATASET_ID = "split_tinyimagenet"
EXPERIMENT_ID = "phase17_tinyimagenet"

TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET_STD = (0.2302, 0.2265, 0.2262)


def _require_manifest() -> None:
    manifest_path_str = str(os.environ.get("TAR_MANIFEST_PATH", "") or "").strip()
    if not manifest_path_str:
        print("REFUSED: TAR_MANIFEST_PATH not set. Set it to the path of the signed manifest and re-run.", flush=True)
        raise SystemExit(1)
    manifest_path = Path(manifest_path_str)
    if not manifest_path.is_absolute():
        manifest_path = Path(_repo) / manifest_path
    try:
        manifest = load_and_verify_manifest(manifest_path, Path(_repo))
        for experiment_id in (EXPERIMENT_ID, "phase17-tinyimagenet-rerun"):
            try:
                manifest.assert_experiment_authorised(experiment_id)
                print(f"[RAIL 3] Manifest gate: OK ({manifest.manifest_id})", flush=True)
                return
            except ManifestGateError:
                continue
        raise ManifestGateError(
            "Manifest does not authorise any accepted Phase 17 execution id "
            "('phase17_tinyimagenet', 'phase17-tinyimagenet-rerun')."
        )
    except ManifestGateError as exc:
        write_refuse_note(
            Path(workspace),
            component="phase17_tinyimagenet",
            reason=str(exc),
            experiment_id=EXPERIMENT_ID,
            manifest_path=str(manifest_path),
        )
        print(f"REFUSED: {exc}", flush=True)
        raise SystemExit(1)


def _suite_config_payload(
    *,
    optimizer_backend: str,
    optimizer_backend_config: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "suite": "phase17_tinyimagenet",
        "dataset": DATASET_ID,
        "hf_dataset_name": HF_DATASET_NAME,
        "backbone": BACKBONE,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "methods": METHODS,
        "n_tasks": N_TASKS,
        "classes_per_task": CLASSES_PER_TASK,
        "batch_size": BATCH_SIZE,
        "base_lr": BASE_LR,
        "ewc_lambda": EWC_LAMBDA,
        "tcl_alpha": TCL_ALPHA,
        "tcl_penalty_lambda": TCL_PENALTY_LAMBDA,
        "tcl_ordered_lr": TCL_ORDERED_LR,
        "tcl_disordered_lr": TCL_DISORDERED_LR,
        "optimizer_backend": optimizer_backend,
        "optimizer_backend_config": optimizer_backend_config or {},
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / max(len(values) - 1, 1))


def _notify(title: str, body: str) -> None:
    safe_t = title.replace('"', "'")
    safe_b = body.replace('"', "'")
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Application; "
        f'$n.BalloonTipTitle = "{safe_t}"; '
        f'$n.BalloonTipText  = "{safe_b}"; '
        "$n.Visible = $True; $n.ShowBalloonTip(20000); Start-Sleep 21; $n.Dispose()"
    )
    try:
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        pass


def _resume_checkpoint_path(workspace: str) -> Path:
    return checkpoint_path(workspace, EXPERIMENT_ID)


def _log_path(workspace: str) -> Path:
    return Path(workspace) / "tar_state" / "logs" / "phase17.log"


def _load_resume_state(workspace: str, restart: bool = False) -> dict[str, Any]:
    ckpt_path = _resume_checkpoint_path(workspace)
    if restart:
        clear_suite_state(ckpt_path)
        return build_suite_state(EXPERIMENT_ID, SEEDS, METHODS, status="running", source="restart")

    state = load_suite_state(ckpt_path)
    if state:
        return state

    recovered = recover_suite_state_from_log(EXPERIMENT_ID, SEEDS, METHODS, _log_path(workspace))
    if recovered:
        save_suite_state(ckpt_path, recovered)
        return recovered

    return build_suite_state(EXPERIMENT_ID, SEEDS, METHODS, status="running", source="fresh")


def _tcl_lr(observer: ActivationThermoObserver, base_lr: float) -> float:
    regime = observer.current_regime
    if regime == "ordered":
        return base_lr * TCL_ORDERED_LR
    if regime == "disordered":
        return base_lr * TCL_DISORDERED_LR
    return base_lr


class _HFSubset(Dataset):
    def __init__(self, items: list, label_map: dict[int, int], transform):
        self._items = items
        self._label_map = label_map
        self._transform = transform

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        image, label = self._items[idx]
        return self._transform(image.convert("RGB")), self._label_map[int(label)]


def _load_hf_tinyimagenet(workspace: str):
    import datasets as hf_datasets

    cache_dir = str(Path(workspace) / "dataset_artifacts" / "tinyimagenet")
    print("  Loading TinyImageNet from HuggingFace (downloads on first run)...", flush=True)
    ds = hf_datasets.load_dataset(HF_DATASET_NAME, cache_dir=cache_dir, trust_remote_code=True)
    train_items = [(row["image"], int(row["label"])) for row in ds["train"]]
    val_items = [(row["image"], int(row["label"])) for row in ds["valid"]]
    print(f"  Loaded {len(train_items)} train / {len(val_items)} val samples", flush=True)
    return train_items, val_items


def _build_tinyimagenet_tasks(seed: int, train_items: list, val_items: list, backbone: str = "resnet18"):
    import torchvision.transforms as T

    if backbone == "vit_tiny":
        # ViT-Tiny/16 requires 224×224 inputs; upscale from 64×64 via bicubic
        train_tf = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ])
        test_tf = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ])
    else:
        train_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(64, padding=8),
            T.ToTensor(),
            T.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ])
        test_tf = T.Compose([
            T.ToTensor(),
            T.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ])

    rng = random.Random(seed)
    all_classes = list(range(200))
    rng.shuffle(all_classes)
    class_order = [
        all_classes[i * CLASSES_PER_TASK:(i + 1) * CLASSES_PER_TASK]
        for i in range(N_TASKS)
    ]

    train_subsets = []
    test_subsets = []
    for task_classes in class_order:
        label_map = {orig: local for local, orig in enumerate(task_classes)}
        task_set = set(task_classes)
        tr = [(img, label) for img, label in train_items if int(label) in task_set]
        te = [(img, label) for img, label in val_items if int(label) in task_set]
        train_subsets.append(_HFSubset(tr, label_map, train_tf))
        test_subsets.append(_HFSubset(te, label_map, test_tf))
    return train_subsets, test_subsets


class _ResNet18Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models

        rn = models.resnet18(weights=None)
        rn.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        rn.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(rn.children())[:-1])
        self.feat_dim = 512

    def forward(self, x):
        return self.features(x).flatten(1)


class _ViTTinyTrunk(nn.Module):
    """ViT-Tiny/16 backbone via timm. Input must be 224×224."""

    def __init__(self):
        super().__init__()
        import timm
        self._vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        self.feat_dim = self._vit.num_features  # 192

    def forward(self, x):
        return self._vit(x)


def run_one_seed(
    seed: int,
    method: str,
    train_subsets: list,
    test_subsets: list,
    progress_callback: Callable[[dict], None] | None = None,
    optimizer_backend: str = "sgd",
    optimizer_backend_config: dict[str, Any] | None = None,
    backbone: str = "resnet18",
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trunk = (_ViTTinyTrunk() if backbone == "vit_tiny" else _ResNet18Trunk()).to(device)
    heads = nn.ModuleList([nn.Linear(trunk.feat_dim, CLASSES_PER_TASK) for _ in range(N_TASKS)]).to(device)
    all_params = list(trunk.parameters()) + list(heads.parameters())

    ewc_fisher: dict[str, torch.Tensor] = {}
    ewc_params: dict[str, torch.Tensor] = {}
    observer: Optional[ActivationThermoObserver] = None
    tcl_anchor_params: dict[str, torch.Tensor] = {}
    tcl_anchor_dpr = 0.0

    if method == "tcl":
        observer = ActivationThermoObserver(
            trunk,
            stat_window_size=5,
            alpha=TCL_ALPHA,
            warmup_batches=60,
            compute_dpr=False,
        )

    accuracy_matrix: dict[int, dict[int, float]] = {}

    for train_t in range(N_TASKS):
        if method == "tcl" and observer is not None and train_t > 0:
            observer.reset_for_new_task()

        optimizer = build_optimizer(
            all_params,
            backend=optimizer_backend,
            lr=BASE_LR,
            weight_decay=1e-4,
            momentum=0.9,
            workspace=workspace,
            run_label=f"phase17-tinyimagenet-{method}-seed{seed}-task{train_t}",
            config=optimizer_backend_config,
        )
        loader = DataLoader(train_subsets[train_t], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        for _epoch in range(EPOCHS):
            trunk.train()
            heads[train_t].train()
            _epoch_loss = 0.0
            _epoch_n_batches = 0
            _epoch_correct = 0
            _epoch_total = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                reps = trunk(bx)
                logits = heads[train_t](reps)
                loss = F.cross_entropy(logits, by)
                _epoch_n_batches += 1
                _epoch_loss += loss.item()
                _epoch_correct += int((logits.detach().argmax(1) == by).sum())
                _epoch_total += int(by.size(0))

                if method == "ewc" and ewc_fisher:
                    ewc_pen = torch.zeros((), device=device)
                    for name, param in trunk.named_parameters():
                        fisher = ewc_fisher.get(name)
                        ref = ewc_params.get(name)
                        if fisher is not None and ref is not None:
                            ewc_pen = ewc_pen + (fisher.to(device) * (param - ref.to(device)).pow(2)).sum()
                    loss = loss + (EWC_LAMBDA / 2.0) * ewc_pen

                if method == "tcl" and tcl_anchor_params and tcl_anchor_dpr > 0.0:
                    tcl_pen = torch.zeros((), device=device)
                    for name, param in trunk.named_parameters():
                        ref = tcl_anchor_params.get(name)
                        if ref is not None:
                            tcl_pen = tcl_pen + (param.float() - ref.to(device).float()).pow(2).sum()
                    loss = loss + TCL_PENALTY_LAMBDA * tcl_anchor_dpr * tcl_pen

                optimizer.zero_grad()
                loss.backward()
                if method == "tcl" and observer is not None:
                    observer.step(optimizer)
                    adj_lr = _tcl_lr(observer, BASE_LR)
                    for group in optimizer.param_groups:
                        group["lr"] = adj_lr
                maybe_apply_optimizer_safety(optimizer, all_params)
                optimizer.step()

            print(
                f"\n  seed={seed}  task {train_t + 1}/{N_TASKS}"
                f"  epoch {_epoch + 1}/{EPOCHS}"
                f"  loss {_epoch_loss / max(_epoch_n_batches, 1):.4f}"
                f"  acc {100.0 * _epoch_correct / max(_epoch_total, 1):.1f}%",
                flush=True,
            )

        if method == "ewc":
            trunk.eval()
            fisher_new = {name: torch.zeros_like(param, device="cpu") for name, param in trunk.named_parameters()}
            sample_loader = DataLoader(train_subsets[train_t], batch_size=32, shuffle=True, num_workers=0)
            n_seen = 0
            for bx, by in sample_loader:
                if n_seen >= 100:
                    break
                bx, by = bx.to(device), by.to(device)
                log_probs = F.log_softmax(heads[train_t](trunk(bx)), dim=1)
                for i in range(bx.size(0)):
                    log_probs[i, by[i]].backward(retain_graph=(i < bx.size(0) - 1))
                    for name, param in trunk.named_parameters():
                        if param.grad is not None:
                            fisher_new[name] = fisher_new[name] + param.grad.detach().cpu().pow(2)
                    trunk.zero_grad()
                    heads[train_t].zero_grad()
                n_seen += bx.size(0)
            n_seen = max(n_seen, 1)
            for name, fisher in fisher_new.items():
                normed = fisher / n_seen
                ewc_fisher[name] = ewc_fisher[name] + normed if name in ewc_fisher else normed
            ewc_params = {name: param.detach().cpu().clone() for name, param in trunk.named_parameters()}

        if method == "tcl" and observer is not None:
            if train_t == 0:
                observer.anchor_snapshot()
            tcl_anchor_params = {name: param.detach().cpu().clone() for name, param in trunk.named_parameters()}
            tcl_anchor_dpr = observer.anchor_effective_dimensionality

        trunk.eval()
        row: dict[int, float] = {}
        for eval_t in range(N_TASKS):
            loader_eval = DataLoader(test_subsets[eval_t], batch_size=256, shuffle=False, num_workers=0)
            correct = 0
            total = 0
            with torch.no_grad():
                for bx, by in loader_eval:
                    bx, by = bx.to(device), by.to(device)
                    logits = heads[eval_t](trunk(bx))
                    correct += int((logits.argmax(1) == by).sum())
                    total += by.size(0)
            row[eval_t] = correct / max(total, 1)
        accuracy_matrix[train_t] = row
        latest_accs = [f"{accuracy_matrix[train_t][t]:.3f}" for t in range(train_t + 1)]
        print(f"    after task {train_t}: acc on seen tasks = {latest_accs}", flush=True)
        if progress_callback is not None:
            progress_callback({
                "seed": seed,
                "method": method,
                "tasks_done": train_t + 1,
                "latest_accs": latest_accs,
            })

    forgetting_per_task = []
    for task_idx in range(N_TASKS - 1):
        peak = max(accuracy_matrix[step][task_idx] for step in range(task_idx, N_TASKS))
        final = accuracy_matrix[N_TASKS - 1][task_idx]
        forgetting_per_task.append(peak - final)
    final_accs = [accuracy_matrix[N_TASKS - 1][task_idx] for task_idx in range(N_TASKS)]
    result = {
        "mean_forgetting": _mean(forgetting_per_task),
        "mean_accuracy": _mean(final_accs),
        "forgetting_per_task": forgetting_per_task,
        "final_accs_per_task": final_accs,
        "accuracy_matrix": {str(k): v for k, v in accuracy_matrix.items()},
    }
    # Explicitly close observer hooks and release GPU references before returning
    # so the caller's torch.cuda.empty_cache() can fully reclaim VRAM.
    if observer is not None:
        observer.close()
        observer = None
    del trunk, heads, ewc_fisher, ewc_params, tcl_anchor_params
    return result


def _run_phase17_suite_impl(
    workspace: str,
    progress_callback: Callable[[dict], None] | None = None,
    restart: bool = False,
    optimizer_backend: str = "sgd",
    optimizer_backend_config: dict[str, Any] | None = None,
) -> dict:
    run_started_at = datetime.now(timezone.utc).isoformat()
    resume_state = _load_resume_state(workspace, restart=restart)
    completed_seeds = list(resume_state.get("completed_seeds", []))
    per_seed = list(resume_state.get("per_seed", []))
    forgetting = {
        method: list((resume_state.get("forgetting", {}) or {}).get(method, []))
        for method in METHODS
    }
    accuracy = {
        method: list((resume_state.get("accuracy", {}) or {}).get(method, []))
        for method in METHODS
    }
    recovered_count = len(completed_seeds)

    print(f"\n{'='*70}")
    print("Phase 17 - Scale-up: Split-TinyImageNet (native runner)")
    print(f"backbone={BACKBONE}  epochs={EPOCHS}  seeds={SEEDS}")
    print(f"n_tasks={N_TASKS}  classes_per_task={CLASSES_PER_TASK}")
    print(f"methods={METHODS}")
    if restart:
        print("restart=True  — ignoring prior checkpoint and restarting suite from seed 1", flush=True)
    elif completed_seeds:
        print(f"resume=True   — recovered completed seeds: {completed_seeds}", flush=True)
    print(f"{run_started_at}")
    print(f"{'='*70}", flush=True)

    if progress_callback is not None:
        progress_callback({
            "seeds_done": recovered_count,
            "seeds_total": len(SEEDS),
            "tasks_done": N_TASKS if recovered_count else 0,
            "latest_accs": [],
            "tcl_forgetting_so_far": forgetting["tcl"][:],
        })

    try:
        with tracked_process(
            Path(workspace),
            experiment_id=EXPERIMENT_ID,
            stage="running",
            owner="legacy-script",
            name="Phase 17 - TinyImageNet Scale-up",
            dataset=DATASET_ID,
            project_id=EXPERIMENT_ID,
        ):
            train_items, val_items = _load_hf_tinyimagenet(workspace)
            for seed in SEEDS:
                if seed in completed_seeds:
                    print(f"\n--- seed={seed} ---", flush=True)
                    print("  resume checkpoint found — skipping completed seed", flush=True)
                    continue
                print(f"\n--- seed={seed} ---", flush=True)
                train_subsets, test_subsets = _build_tinyimagenet_tasks(seed, train_items, val_items)
                row = {"seed": seed}
                for method in METHODS:
                    print(f"  running {method}...", flush=True)
                    done_before = len(per_seed)
                    result = run_one_seed(
                        seed,
                        method,
                        train_subsets,
                        test_subsets,
                        progress_callback=lambda payload, seed_done=done_before: (
                            progress_callback({
                                "seed": payload.get("seed"),
                                "method": payload.get("method"),
                                "seeds_done": seed_done,
                                "seeds_total": len(SEEDS),
                                "tasks_done": payload.get("tasks_done", 0),
                                "latest_accs": payload.get("latest_accs", []),
                                "tcl_forgetting_so_far": forgetting["tcl"][:],
                            }) if progress_callback is not None else None
                        ),
                        optimizer_backend=optimizer_backend,
                        optimizer_backend_config=optimizer_backend_config,
                    )
                    row[f"{method}_forgetting"] = result["mean_forgetting"]
                    row[f"{method}_acc"] = result["mean_accuracy"]
                    forgetting[method].append(result["mean_forgetting"])
                    accuracy[method].append(result["mean_accuracy"])
                    print(f"  {method:12s}  forgetting={result['mean_forgetting']:.4f}"
                          f"  acc={result['mean_accuracy']:.4f}", flush=True)
                per_seed.append(row)
                resume_state = append_completed_seed(resume_state, row)
                resume_state["status"] = "running"
                save_suite_state(_resume_checkpoint_path(workspace), resume_state)
                if progress_callback is not None:
                    progress_callback({
                        "seeds_done": len(per_seed),
                        "seeds_total": len(SEEDS),
                        "tasks_done": N_TASKS,
                        "latest_accs": [],
                        "tcl_forgetting_so_far": forgetting["tcl"][:],
                    })
            update_stage(Path(workspace), EXPERIMENT_ID, "analyzing")
    except Exception as exc:
        err_msg = f"Exception during benchmark run: {exc}"
        print(f"\nERROR: {err_msg}", flush=True)
        import traceback
        traceback.print_exc()
        resume_state["status"] = "stalled"
        save_suite_state(_resume_checkpoint_path(workspace), resume_state)
        completed_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "status": "ERROR",
            "dataset": DATASET_ID,
            "n_tasks": N_TASKS,
            "verdict": f"ERROR - {err_msg}",
            "error": err_msg,
            "exception": str(exc),
            "completed_at": completed_at,
        }
        env_payload = collect_environment_snapshot(
            repo_root=Path(_repo),
            workspace=Path(workspace),
            config=_suite_config_payload(
                optimizer_backend=optimizer_backend,
                optimizer_backend_config=optimizer_backend_config,
            ),
            trigger="manual_script",
            source_script=Path(__file__).name,
            run_started_at=run_started_at,
            run_ended_at=completed_at,
            extra={"logical_name": "phase17_tinyimagenet", "status": "ERROR"},
        )
        write_canonical_comparison_result(
            workspace=Path(workspace),
            logical_name="phase17_tinyimagenet",
            payload=payload,
            env_payload=env_payload,
            phase_number=17,
            source_script=Path(__file__).name,
        )
        _notify("TAR Phase 17 FAILED", str(exc)[:120])
        raise

    print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
    aggregate = {}
    for method in METHODS:
        aggregate[method] = {
            "forgetting_mean": _mean(forgetting[method]),
            "forgetting_std": _std(forgetting[method]),
            "acc_mean": _mean(accuracy[method]),
            "acc_std": _std(accuracy[method]),
        }
        print(f"  {method:12s}  forgetting={aggregate[method]['forgetting_mean']:.4f}"
              f"+-{aggregate[method]['forgetting_std']:.4f}"
              f"  acc={aggregate[method]['acc_mean']:.4f}+-{aggregate[method]['acc_std']:.4f}")

    from scipy import stats as scipy_stats

    tcl_forg = forgetting["tcl"]
    pairwise = {}
    for baseline in ["ewc", "sgd_baseline"]:
        deltas = [tcl - base for tcl, base in zip(tcl_forg, forgetting[baseline])]
        t_stat, p_val = scipy_stats.ttest_1samp(deltas, 0)
        d_stat = abs(_mean(deltas)) / max(_std(deltas), 1e-12)
        n_better = sum(1 for delta in deltas if delta < 0)
        pairwise[baseline] = {
            "mean_delta": _mean(deltas),
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "cohens_d": float(d_stat),
            "n_tcl_better": n_better,
        }
        print(f"  TCL vs {baseline:12s}: delta={_mean(deltas):+.4f}"
              f"  p={p_val:.4f}  d={d_stat:.3f}  {n_better}/{len(SEEDS)}")

    vs_ewc = pairwise["ewc"]
    vs_sgd = pairwise["sgd_baseline"]
    if vs_ewc["mean_delta"] < -0.01 and vs_ewc["p_val"] < 0.05 and vs_ewc["cohens_d"] > 0.5:
        verdict_key = "OUTCOME_A"
        verdict = (
            f"OUTCOME A - TCL beats EWC on TinyImageNet (delta={vs_ewc['mean_delta']:+.4f}, "
            f"p={vs_ewc['p_val']:.4f}, d={vs_ewc['cohens_d']:.2f}). Strong multi-dataset result."
        )
    elif vs_sgd["mean_delta"] < -0.01 and vs_sgd["p_val"] < 0.05 and vs_sgd["cohens_d"] > 0.5:
        verdict_key = "OUTCOME_B"
        verdict = (
            f"OUTCOME B - TCL beats SGD on TinyImageNet (delta={vs_sgd['mean_delta']:+.4f}, "
            f"p={vs_sgd['p_val']:.4f}, d={vs_sgd['cohens_d']:.2f}). Generalises beyond CIFAR-10."
        )
    else:
        verdict_key = "OUTCOME_C"
        verdict = "OUTCOME C - No significant improvement on TinyImageNet at current hyperparameters."

    completed_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "backbone": BACKBONE,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "dataset": DATASET_ID,
        "n_tasks": N_TASKS,
        "classes_per_task": CLASSES_PER_TASK,
        "methods": METHODS,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "pairwise": pairwise,
        "verdict_key": verdict_key,
        "verdict": verdict,
        "completed_at": completed_at,
    }
    env_payload = collect_environment_snapshot(
        repo_root=Path(_repo),
        workspace=Path(workspace),
        config=_suite_config_payload(
            optimizer_backend=optimizer_backend,
            optimizer_backend_config=optimizer_backend_config,
        ),
        trigger="manual_script",
        source_script=Path(__file__).name,
        run_started_at=run_started_at,
        run_ended_at=completed_at,
        extra={"logical_name": "phase17_tinyimagenet", "status": "COMPLETE"},
    )
    artifacts = write_canonical_comparison_result(
        workspace=Path(workspace),
        logical_name="phase17_tinyimagenet",
        payload=payload,
        env_payload=env_payload,
        phase_number=17,
        source_script=Path(__file__).name,
    )
    resume_state = build_suite_state(
        EXPERIMENT_ID,
        SEEDS,
        METHODS,
        per_seed=per_seed,
        status="complete",
        source="completed-run",
    )
    save_suite_state(_resume_checkpoint_path(workspace), resume_state)
    _notify("TAR Phase 17 COMPLETE", f"{verdict_key} - {verdict[:80]}")
    print(f"\nResult written: {artifacts['result_path']}")
    print(f"Env snapshot: {artifacts['env_path']}")
    print(f"Index updated: {artifacts['index_path']}")
    print(f"[{completed_at}] Phase 17 complete")
    return payload


def run_phase17_suite(
    workspace: str,
    progress_callback: Callable[[dict], None] | None = None,
    restart: bool = False,
    optimizer_backend: str = "sgd",
    optimizer_backend_config: dict[str, Any] | None = None,
) -> dict:
    with tee_console(_log_path(workspace)):
        return _run_phase17_suite_impl(
            workspace,
            progress_callback=progress_callback,
            restart=restart,
            optimizer_backend=optimizer_backend,
            optimizer_backend_config=optimizer_backend_config,
        )


def main() -> None:
    try:
        _require_manifest()
        restart = "--restart" in set(sys.argv[1:])
        run_phase17_suite(workspace, restart=restart)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
