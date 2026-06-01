"""
TAR RunPod Worker — runs inside a RunPod pod via SSH.

Called by tar_runpod_executor.py. Delegates to the existing TAR runner
functions unchanged. Writes atomic progress.json after every seed so the
local executor can poll live progress. Writes result_{id}.json on completion.

Usage (invoked by executor over SSH):
  python tar_runpod_worker.py \
    --experiment-id abc123def456 \
    --dataset split_tinyimagenet \
    --method tcl \
    --seeds 42 0 1 2 3 \
    --epochs 40 \
    --backbone resnet18 \
    --config-overrides '{}' \
    --workspace /workspace \
    --progress-file /workspace/progress_abc123def456.json

Exit codes: 0=success, 1=training error, 2=dataset error, 3=env error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: str, data: dict[str, Any]) -> None:
    """Atomic JSON write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _write_progress(path: str, data: dict[str, Any], volume_path: str = "") -> None:
    """Write progress JSON atomically — to pod disk AND to network volume if mounted."""
    data["updated_at"] = _ts()
    _atomic_write(path, data)
    # Mirror to network volume for durability (survives pod death)
    if volume_path:
        try:
            _atomic_write(f"{volume_path}/progress.json", data)
        except Exception:
            pass  # volume write failure is non-fatal


def _write_result(workspace: str, experiment_id: str, result: dict[str, Any], volume_path: str = "") -> None:
    """Write result JSON to pod disk AND to network volume."""
    result["execution_backend"] = "runpod"
    result["completed_at"] = _ts()

    # Pod disk (for immediate SFTP pull)
    pod_path = str(Path(workspace) / f"result_{experiment_id}.json")
    _atomic_write(pod_path, result)
    print(f"[worker] Result written to pod disk: {pod_path}", flush=True)

    # Network volume (durable — survives pod death)
    if volume_path:
        try:
            vol_path = f"{volume_path}/result.json"
            _atomic_write(vol_path, result)
            print(f"[worker] Result written to volume: {vol_path}", flush=True)
        except Exception as exc:
            print(f"[worker] Volume write warning: {exc}", flush=True)


def _setup_env(workspace: str) -> None:
    """Set environment variables so TAR's dataset/cache logic points to /workspace."""
    os.environ.setdefault("HF_HOME",          f"{workspace}/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", f"{workspace}/hf_cache/datasets")
    os.environ.setdefault("TORCH_HOME",        f"{workspace}/torch_cache")
    os.environ.setdefault("TMPDIR",            f"{workspace}/tmp")
    os.environ.setdefault("TAR_WORKSPACE",     workspace)
    for d in ("hf_cache", "torch_cache", "tmp", "data"):
        Path(workspace, d).mkdir(parents=True, exist_ok=True)


def _build_workspace_layout(workspace: str) -> None:
    """Create minimal workspace dirs expected by TAR."""
    ws = Path(workspace)
    for sub in ("tar_state/experiments", "tar_state/logs", "training_artifacts", "dataset_artifacts"):
        (ws / sub).mkdir(parents=True, exist_ok=True)


def run_tinyimagenet(spec_args: argparse.Namespace, progress_file: str, volume_path: str = "") -> dict[str, Any]:
    """Run split_tinyimagenet via phase17_tinyimagenet runner."""
    print("[worker] Loading TinyImageNet dataset...", flush=True)
    from phase17_tinyimagenet import _load_hf_tinyimagenet, run_one_seed as _run17
    workspace = spec_args.workspace

    try:
        train_loaders, test_loaders, n_tasks = _load_hf_tinyimagenet(workspace)
    except Exception as exc:
        print(f"[worker] Dataset load error: {exc}", flush=True)
        sys.exit(2)

    seeds = spec_args.seeds
    forgetting_list, accuracy_list, seed_results = [], [], []

    for i, seed in enumerate(seeds):
        print(f"\n[worker] === Seed {seed} ({i+1}/{len(seeds)}) ===", flush=True)
        try:
            res = _run17(
                seed=seed,
                workspace=workspace,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                n_tasks=n_tasks,
                epochs=spec_args.epochs,
                method=spec_args.method,
                backbone=spec_args.backbone,
                config_overrides=spec_args.config_overrides,
            )
        except Exception as exc:
            print(f"[worker] Seed {seed} error: {exc}", flush=True)
            sys.exit(1)

        forgetting_list.append(res["mean_forgetting"])
        accuracy_list.append(res["mean_accuracy"])
        seed_results.append({
            "seed": seed,
            "forgetting": res["mean_forgetting"],
            "accuracy":   res["mean_accuracy"],
        })
        _write_progress(progress_file, {
            "experiment_id": spec_args.experiment_id,
            "seeds_done":    i + 1,
            "seeds_total":   len(seeds),
            "tasks_done":    n_tasks,
            "latest_accs":   [f"{acc:.4f}" for acc in accuracy_list],
            "forgetting_so_far": forgetting_list[:],
        }, volume_path=volume_path)
        print(f"[worker] Seed {seed}: forgetting={res['mean_forgetting']:.4f} acc={res['mean_accuracy']:.4f}", flush=True)

    return {"seed_results": seed_results, "forgetting_list": forgetting_list, "accuracy_list": accuracy_list}


def run_cifar100(spec_args: argparse.Namespace, progress_file: str, volume_path: str = "") -> dict[str, Any]:
    """Run split_cifar100 via phase16_scale_up runner."""
    print("[worker] Loading CIFAR-100 dataset...", flush=True)
    from phase16_scale_up import _load_hf_cifar100, run_one_seed as _run16
    workspace = spec_args.workspace

    try:
        dataset = _load_hf_cifar100(workspace)
    except Exception as exc:
        print(f"[worker] Dataset load error: {exc}", flush=True)
        sys.exit(2)

    seeds = spec_args.seeds
    forgetting_list, accuracy_list, seed_results = [], [], []
    n_tasks = 10

    for i, seed in enumerate(seeds):
        print(f"\n[worker] === Seed {seed} ({i+1}/{len(seeds)}) ===", flush=True)
        try:
            res = _run16(
                seed=seed,
                workspace=workspace,
                dataset=dataset,
                epochs=spec_args.epochs,
                method=spec_args.method,
                backbone=spec_args.backbone,
                config_overrides=spec_args.config_overrides,
            )
        except Exception as exc:
            print(f"[worker] Seed {seed} error: {exc}", flush=True)
            sys.exit(1)

        forgetting_list.append(res["mean_forgetting"])
        accuracy_list.append(res["mean_accuracy"])
        seed_results.append({
            "seed": seed,
            "forgetting": res["mean_forgetting"],
            "accuracy":   res["mean_accuracy"],
        })
        _write_progress(progress_file, {
            "experiment_id": spec_args.experiment_id,
            "seeds_done":    i + 1,
            "seeds_total":   len(seeds),
            "tasks_done":    n_tasks,
            "latest_accs":   [f"{acc:.4f}" for acc in accuracy_list],
            "forgetting_so_far": forgetting_list[:],
        }, volume_path=volume_path)
        print(f"[worker] Seed {seed}: forgetting={res['mean_forgetting']:.4f} acc={res['mean_accuracy']:.4f}", flush=True)

    return {"seed_results": seed_results, "forgetting_list": forgetting_list, "accuracy_list": accuracy_list}


def run_generic(spec_args: argparse.Namespace, progress_file: str, volume_path: str = "") -> dict[str, Any]:
    """Run any other dataset via generic_cl_runner."""
    from tar_lab.generic_cl_runner import run_generic_benchmark
    workspace = spec_args.workspace
    seeds = spec_args.seeds
    n_tasks = 10

    def _progress_cb(progress_dict: dict[str, Any]) -> None:
        _write_progress(progress_file, {
            "experiment_id": spec_args.experiment_id,
            "seeds_done":    progress_dict.get("seeds_done", 0),
            "seeds_total":   len(seeds),
            "tasks_done":    progress_dict.get("tasks_done", 0),
            "latest_accs":   progress_dict.get("latest_accs", []),
            "forgetting_so_far": progress_dict.get("forgetting_so_far", []),
        })

    try:
        seed_results, forgetting_list, accuracy_list = run_generic_benchmark(
            dataset_name=spec_args.dataset,
            backbone_name=spec_args.backbone,
            method_name=spec_args.method,
            seeds=seeds,
            epochs=spec_args.epochs,
            config_overrides=spec_args.config_overrides,
            data_root=str(Path(workspace) / "data"),
            log_fn=lambda msg: print(f"[runner] {msg}", flush=True),
            progress_callback=_progress_cb,
        )
    except Exception as exc:
        print(f"[worker] Generic runner error: {exc}", flush=True)
        sys.exit(1)

    return {
        "seed_results":   seed_results,
        "forgetting_list": forgetting_list,
        "accuracy_list":   accuracy_list,
    }


def build_result(spec_args: argparse.Namespace, run_output: dict[str, Any]) -> dict[str, Any]:
    """Build a result dict matching TAR's ExperimentResult schema."""
    seed_results   = run_output["seed_results"]
    forgetting_list = run_output["forgetting_list"]
    accuracy_list   = run_output["accuracy_list"]

    n = len(forgetting_list)
    if n == 0:
        return {"verdict": "ERROR", "error": "no seeds completed", "execution_backend": "runpod"}

    mean_forgetting = sum(forgetting_list) / n
    mean_accuracy   = sum(accuracy_list) / n

    try:
        import statistics as _st
        std_forgetting = _st.stdev(forgetting_list) if n > 1 else 0.0
        std_accuracy   = _st.stdev(accuracy_list)   if n > 1 else 0.0
    except Exception:
        std_forgetting = std_accuracy = 0.0

    # Simple verdict without baseline (orchestrator will re-evaluate with baseline)
    verdict = "NULL"
    if mean_forgetting < 0.10:
        verdict = "DIRECTIONAL"

    return {
        "experiment_id":   spec_args.experiment_id,
        "dataset":         spec_args.dataset,
        "method":          spec_args.method,
        "seeds":           spec_args.seeds,
        "config_overrides": spec_args.config_overrides,
        "seed_results":    seed_results,
        "mean_forgetting": mean_forgetting,
        "std_forgetting":  std_forgetting,
        "mean_accuracy":   mean_accuracy,
        "std_accuracy":    std_accuracy,
        "verdict":         verdict,
        "execution_backend": "runpod",
        "backbone":        spec_args.backbone,
        "epochs":          spec_args.epochs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TAR RunPod experiment worker")
    parser.add_argument("--experiment-id",   required=True)
    parser.add_argument("--dataset",          required=True)
    parser.add_argument("--method",           required=True)
    parser.add_argument("--seeds",            nargs="+", type=int, required=True)
    parser.add_argument("--epochs",           type=int, default=40)
    parser.add_argument("--backbone",         default="resnet18")
    parser.add_argument("--config-overrides", default="{}")
    parser.add_argument("--workspace",        default="/workspace")
    parser.add_argument("--progress-file",    required=True)
    parser.add_argument("--volume-path",      default="",
                        help="Network volume path for durable result storage, e.g. /runpod-volume/experiments/abc123")
    args = parser.parse_args()

    # Parse config overrides
    try:
        args.config_overrides = json.loads(args.config_overrides)
    except Exception:
        args.config_overrides = {}

    print(f"[worker] TAR RunPod Worker starting", flush=True)
    print(f"[worker] experiment={args.experiment_id} dataset={args.dataset} method={args.method}", flush=True)
    print(f"[worker] seeds={args.seeds} epochs={args.epochs} backbone={args.backbone}", flush=True)

    _setup_env(args.workspace)
    _build_workspace_layout(args.workspace)

    # Ensure repo root is on PYTHONPATH
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Initialise progress file
    _write_progress(args.progress_file, {
        "experiment_id": args.experiment_id,
        "seeds_done":    0,
        "seeds_total":   len(args.seeds),
        "tasks_done":    0,
        "latest_accs":   [],
        "forgetting_so_far": [],
    })

    volume_path = str(getattr(args, "volume_path", "") or "").strip()
    if volume_path:
        # Ensure volume exp dir exists
        Path(volume_path).mkdir(parents=True, exist_ok=True)
        print(f"[worker] Results will be mirrored to volume: {volume_path}", flush=True)
    else:
        print("[worker] No volume path — results on pod disk only", flush=True)

    # Route to correct runner
    dataset = args.dataset
    print(f"[worker] Routing to runner for: {dataset}", flush=True)

    if dataset == "split_tinyimagenet":
        run_output = run_tinyimagenet(args, args.progress_file, volume_path=volume_path)
    elif dataset == "split_cifar100":
        run_output = run_cifar100(args, args.progress_file, volume_path=volume_path)
    else:
        run_output = run_generic(args, args.progress_file, volume_path=volume_path)

    # Build and write result to pod disk + volume
    result = build_result(args, run_output)
    _write_result(args.workspace, args.experiment_id, result, volume_path=volume_path)

    print(f"\n[worker] DONE: verdict={result.get('verdict')} mean_forgetting={result.get('mean_forgetting', '?'):.4f}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
