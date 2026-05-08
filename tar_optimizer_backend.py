from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import torch

from tar_storage import ensure_workspace_layout, resolve_workspace

_REPO = Path(__file__).resolve().parent

_NON_BENCHMARK_OVERRIDE_KEYS = {
    "comparison_methods",
    "external_baselines",
    "candidate_datasets",
    "candidate_backbones",
    "research_strategy",
    "internal_method_role",
}


def normalize_optimizer_backend(value: str | None) -> str:
    raw = str(value or "sgd").strip().lower().replace("-", "_")
    if raw in {"", "default"}:
        return "sgd"
    if raw in {"adam", "adamw"}:
        return "adamw"
    if raw in {"cruxy", "cruxy_optimizer"}:
        return "cruxy"
    if raw == "sgd":
        return "sgd"
    return raw


def split_optimizer_config(
    config_overrides: dict[str, Any] | None,
    *,
    explicit_backend: str = "",
    explicit_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    config = dict(config_overrides or {})
    backend = explicit_backend or str(config.pop("optimizer_backend", "") or "")
    backend_config = explicit_config
    if backend_config is None:
        raw_backend_config = config.pop("optimizer_backend_config", {})
        backend_config = raw_backend_config if isinstance(raw_backend_config, dict) else {}
    for key in list(config.keys()):
        if key in _NON_BENCHMARK_OVERRIDE_KEYS:
            config.pop(key, None)
    return normalize_optimizer_backend(backend), dict(backend_config or {}), config


def _metrics_path(workspace: str | Path | None, run_label: str) -> str:
    if workspace:
        base = ensure_workspace_layout(Path(workspace), repo_root=_REPO)
    else:
        base = ensure_workspace_layout(resolve_workspace(_REPO), repo_root=_REPO)
    metrics_dir = base / "tar_state" / "optimizer_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_label).strip("-") or "optimizer"
    return str(metrics_dir / f"{slug}.jsonl")


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    backend: str = "sgd",
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    workspace: str | Path | None = None,
    run_label: str = "",
    config: dict[str, Any] | None = None,
):
    backend = normalize_optimizer_backend(backend)
    cfg = dict(config or {})
    param_list = list(params)

    if backend == "sgd":
        return torch.optim.SGD(
            param_list,
            lr=lr,
            momentum=float(cfg.get("momentum", momentum)),
            weight_decay=float(cfg.get("weight_decay", weight_decay)),
            nesterov=bool(cfg.get("nesterov", False)),
        )

    if backend == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(
            param_list,
            lr=lr,
            betas=(float(betas[0]), float(betas[1])),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=float(cfg.get("weight_decay", weight_decay)),
        )

    if backend == "cruxy":
        from cruxy import CruxyOptimizer

        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        return CruxyOptimizer(
            param_list,
            mode=str(cfg.get("mode", "meta3") or "meta3"),
            lr=float(cfg.get("lr", lr)),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=float(cfg.get("weight_decay", weight_decay)),
            decoupled_weight_decay=bool(cfg.get("decoupled_weight_decay", True)),
            use_nesterov=bool(cfg.get("use_nesterov", True)),
            use_lion=bool(cfg.get("use_lion", False)),
            use_gc=bool(cfg.get("use_gc", False)),
            meta_lr_eta=float(cfg.get("meta_lr_eta", 0.05)),
            meta_lr_beta2=float(cfg.get("meta_lr_beta2", 0.03)),
            meta_interval=int(cfg.get("meta_interval", 10)),
            gamma_norm=float(cfg.get("gamma_norm", 1.5)),
            log_metrics=bool(cfg.get("log_metrics", True)),
            metrics_path=str(cfg.get("metrics_path") or _metrics_path(workspace, run_label)),
            compile=bool(cfg.get("compile", False)),
        )

    raise ValueError(f"Unsupported optimizer backend: {backend}")


def maybe_apply_optimizer_safety(
    optimizer: Any,
    params: Iterable[torch.nn.Parameter],
) -> None:
    if optimizer.__class__.__name__ != "CruxyOptimizer":
        return
    from cruxy import SafetyGuard

    param_list = list(params)
    SafetyGuard.check_numerical_safety(
        [p.data for p in param_list],
        [p.grad for p in param_list],
    )
