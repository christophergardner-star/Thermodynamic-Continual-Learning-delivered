from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_storage import ensure_workspace_layout

try:
    import psutil as _psutil
except Exception:
    _psutil = None


PRIMARY_CLAIM = (
    "high_penalty_conservative reduces catastrophic forgetting vs TCL baseline "
    "and EWC on Split-CIFAR-10 without accuracy collapse."
)
VALIDATION_MODE_ID = "stabilisation_hpc_cifar10_validation"
VALIDATION_FRONTIER_ID = "fp-hyperparameter-robustness"
VALIDATION_PAPER_ID = "tcl-hpc-validation-paper"
VALIDATION_PROJECT_ID = "tcl-hpc-validation-cifar10-v1"
VALIDATION_EXPERIMENT_ID = "claim_validation_hpc_suite"
VALIDATION_EXPERIMENT_TITLE = "Claim Validation - HPC vs TCL/EWC on Split-CIFAR-10"
VALIDATION_TITLE = "Conservative TCL reduces forgetting on Split-CIFAR-10"
VALIDATION_SHORT_TITLE = "HPC validation"
VALIDATION_DATASET = "split_cifar10"
VALIDATION_BACKBONE = "resnet18"
VALIDATION_EPOCHS = 40
VALIDATION_BATCH_SIZE = 64
VALIDATION_SETTING = "task_incremental"
VALIDATION_METHOD_ORDER = [
    "sgd_baseline",
    "ewc_lambda_100",
    "ewc_lambda_1000",
    "si_c_0_01",
    "tcl_baseline",
    "high_penalty_conservative",
]
DEFAULT_MIN_SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_TARGET_SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "stabilisation_mode.json"


def validation_root(workspace: Path) -> Path:
    return workspace / "tar_state" / "validation"


def validation_bundle_root(workspace: Path, bundle_id: str) -> Path:
    return validation_root(workspace) / bundle_id


def load_state(workspace: Path) -> dict[str, Any]:
    path = state_path(workspace)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_state(workspace: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def is_active(workspace: Path) -> bool:
    return bool(load_state(workspace).get("active"))


def method_matrix() -> list[dict[str, Any]]:
    return [
        {
            "key": "sgd_baseline",
            "label": "SGD baseline",
            "method": "sgd_baseline",
            "config_overrides": {},
        },
        {
            "key": "ewc_lambda_100",
            "label": "EWC (lambda=100)",
            "method": "ewc",
            "config_overrides": {"ewc_lambda": 100.0},
        },
        {
            "key": "ewc_lambda_1000",
            "label": "EWC (lambda=1000)",
            "method": "ewc",
            "config_overrides": {"ewc_lambda": 1000.0},
        },
        {
            "key": "si_c_0_01",
            "label": "SI (c=0.01)",
            "method": "si",
            "config_overrides": {"si_c": 0.01, "si_xi": 0.001},
        },
        {
            "key": "tcl_baseline",
            "label": "TCL baseline",
            "method": "tcl",
            "config_overrides": {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_ordered_lr_scale": 0.5,
                "tcl_disordered_lr_scale": 1.2,
                "tcl_reset_on_task_boundary": True,
            },
        },
        {
            "key": "high_penalty_conservative",
            "label": "High-Penalty Conservative",
            "method": "tcl",
            "config_overrides": {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.05,
                "tcl_ordered_lr_scale": 0.3,
                "tcl_disordered_lr_scale": 1.2,
                "tcl_alpha": 0.45,
                "tcl_reset_on_task_boundary": True,
            },
        },
    ]


def build_config_snapshots(
    *,
    min_seeds: list[int] | None = None,
    target_seeds: list[int] | None = None,
) -> dict[str, Any]:
    from tar_lab.schemas import ContinualLearningBenchmarkConfig

    min_seed_list = list(min_seeds or DEFAULT_MIN_SEEDS)
    target_seed_list = list(target_seeds or DEFAULT_TARGET_SEEDS)
    base = ContinualLearningBenchmarkConfig(
        dataset=VALIDATION_DATASET,
        setting=VALIDATION_SETTING,
        train_epochs_per_task=VALIDATION_EPOCHS,
        batch_size=VALIDATION_BATCH_SIZE,
        seed=min_seed_list[0],
    ).model_dump()
    payload: dict[str, Any] = {
        "captured_at": _now_iso(),
        "dataset": VALIDATION_DATASET,
        "setting": VALIDATION_SETTING,
        "backbone": VALIDATION_BACKBONE,
        "min_seed_list": min_seed_list,
        "target_seed_list": target_seed_list,
        "methods": {},
    }
    for rec in method_matrix():
        cfg = dict(base)
        cfg.update(rec["config_overrides"])
        payload["methods"][rec["key"]] = {
            "label": rec["label"],
            "method": rec["method"],
            "config": cfg,
        }
    return payload


def _stable_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _validation_suite_record_from_spec(spec: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(spec, "id", "") or ""),
        "name": str(getattr(spec, "name", "") or ""),
        "project_id": str(getattr(spec, "project_id", "") or ""),
        "hypothesis_name": str(getattr(spec, "hypothesis_name", "") or ""),
        "dataset": str(getattr(spec, "dataset", "") or ""),
        "method": str(getattr(spec, "method", "") or ""),
        "seeds": [int(seed) for seed in list(getattr(spec, "seeds", []) or [])],
        "priority": int(getattr(spec, "priority", 0) or 0),
        "estimated_runtime_h": float(getattr(spec, "estimated_runtime_h", 0.0) or 0.0),
        "backbone": str(getattr(spec, "backbone", "") or ""),
        "epochs": int(getattr(spec, "epochs", 0) or 0),
        "description": str(getattr(spec, "description", "") or ""),
        "tags": [str(tag) for tag in list(getattr(spec, "tags", []) or [])],
        "hardware_budget": _stable_clone(getattr(spec, "hardware_budget", {}) or {}),
        "frontier_problem_id": str(getattr(spec, "frontier_problem_id", "") or ""),
        "context": _stable_clone(getattr(spec, "context", {}) or {}),
        "author_paper_id": str(getattr(spec, "author_paper_id", "") or ""),
        "observer_class_name": str(getattr(spec, "observer_class_name", "") or ""),
        "depends_on": [str(dep) for dep in list(getattr(spec, "depends_on", []) or [])],
        "runner_key": str(getattr(spec, "runner_key", "") or ""),
        "optimizer_backend": str(getattr(spec, "optimizer_backend", "sgd") or "sgd"),
        "optimizer_backend_config": _stable_clone(
            getattr(spec, "optimizer_backend_config", {}) or {}
        ),
        "config_overrides": _stable_clone(getattr(spec, "config_overrides", {}) or {}),
    }


def ensure_validation_method_order_exact(method_order: list[str]) -> list[str]:
    observed = [str(item) for item in list(method_order or [])]
    expected = list(VALIDATION_METHOD_ORDER)
    if observed != expected:
        raise ValueError(
            "Validation suite method order drift detected. "
            f"Expected {expected}, got {observed}."
        )
    return observed


def validation_suite_lock_payload(
    workspace: Path,
    *,
    min_seeds: list[int] | None = None,
    target_seeds: list[int] | None = None,
) -> dict[str, Any]:
    spec = build_validation_suite_spec(
        workspace,
        min_seeds=min_seeds,
        target_seeds=target_seeds,
    )
    ensure_validation_method_order_exact(
        list((spec.config_overrides or {}).get("method_order", []) or [])
    )
    payload = {
        "lock_version": 1,
        "primary_claim": PRIMARY_CLAIM,
        "dataset": VALIDATION_DATASET,
        "setting": VALIDATION_SETTING,
        "backbone": VALIDATION_BACKBONE,
        "epochs": VALIDATION_EPOCHS,
        "batch_size": VALIDATION_BATCH_SIZE,
        "method_order": list(VALIDATION_METHOD_ORDER),
        "method_matrix": [
            {
                "key": str(rec["key"]),
                "label": str(rec["label"]),
                "method": str(rec["method"]),
                "config_overrides": _stable_clone(rec["config_overrides"]),
            }
            for rec in method_matrix()
        ],
        "spec": _validation_suite_record_from_spec(spec),
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def validation_suite_drift(spec: Any, lock_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = dict(lock_payload.get("spec", {}) or {})
    current = _validation_suite_record_from_spec(spec)
    drift: dict[str, dict[str, Any]] = {}
    for field, expected_value in expected.items():
        current_value = current.get(field)
        if current_value != expected_value:
            drift[field] = {
                "expected": expected_value,
                "current": current_value,
            }
    return drift


def apply_validation_suite_lock(spec: Any, lock_payload: dict[str, Any]) -> bool:
    expected = dict(lock_payload.get("spec", {}) or {})
    changed = False
    for field, expected_value in expected.items():
        current_value = _stable_clone(getattr(spec, field, None))
        if current_value == expected_value:
            continue
        setattr(spec, field, _stable_clone(expected_value))
        changed = True
    return changed


def assert_validation_suite_spec_locked(spec: Any, workspace: Path) -> dict[str, Any]:
    state = load_state(workspace)
    lock_payload = state.get("validation_suite_lock", {})
    if not isinstance(lock_payload, dict) or not lock_payload:
        lock_payload = validation_suite_lock_payload(
            workspace,
            min_seeds=list(state.get("min_seed_list", []) or []),
            target_seeds=list(state.get("target_seed_list", []) or []),
        )
    expected_lock = validation_suite_lock_payload(
        workspace,
        min_seeds=list(state.get("min_seed_list", []) or []),
        target_seeds=list(state.get("target_seed_list", []) or []),
    )
    if lock_payload.get("fingerprint") != expected_lock.get("fingerprint"):
        raise ValueError(
            "Validation suite lock fingerprint mismatch. "
            "Frozen HPC suite configuration was changed after stabilisation."
        )
    ensure_validation_method_order_exact(
        list((_validation_suite_record_from_spec(spec)["config_overrides"] or {}).get("method_order", []) or [])
    )
    drift = validation_suite_drift(spec, lock_payload)
    if drift:
        fields = ", ".join(sorted(drift.keys()))
        raise ValueError(
            "Validation suite drift detected before execution. "
            f"Locked fields changed: {fields}."
        )
    return lock_payload


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_dirty(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _gpu_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": "",
        "gpu_name": "",
        "vram_gb": 0.0,
    }
    try:
        import torch

        payload["torch_version"] = str(torch.__version__)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
        if torch.cuda.is_available():
            payload["gpu_name"] = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            payload["vram_gb"] = round(float(props.total_memory) / (1024 ** 3), 3)
    except Exception:
        payload["torch_version"] = ""
    return payload


def capture_environment_snapshot(repo_root: Path, workspace: Path) -> dict[str, Any]:
    gpu = _gpu_snapshot()
    cpu_count = os.cpu_count() or 0
    ram_total_gb = 0.0
    ram_available_gb = 0.0
    cpu_name = platform.processor()
    if _psutil is not None:
        try:
            vm = _psutil.virtual_memory()
            ram_total_gb = round(vm.total / (1024 ** 3), 3)
            ram_available_gb = round(vm.available / (1024 ** 3), 3)
        except Exception:
            pass
    return {
        "captured_at": _now_iso(),
        "repo_root": str(repo_root),
        "workspace": str(workspace),
        "git_commit": _git_commit(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu_name": cpu_name,
        "cpu_logical_cores": cpu_count,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        **gpu,
    }


def _copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _archive_targets(workspace: Path, repo_root: Path) -> list[tuple[Path, str]]:
    return [
        (workspace / "tar_state" / "experiments", "archive/experiments"),
        (workspace / "tar_state" / "comparisons", "archive/comparisons"),
        (workspace / "tar_state" / "logs", "archive/logs"),
        (workspace / "tar_state" / "experiment_queue.json", "archive/state/experiment_queue.json"),
        (workspace / "tar_state" / "experiment_archive.json", "archive/state/experiment_archive.json"),
        (workspace / "tar_state" / "author_state.json", "archive/state/author_state.json"),
        (workspace / "tar_state" / "research_director_state.json", "archive/state/research_director_state.json"),
        (workspace / "tar_state" / "scheduler_state.json", "archive/state/scheduler_state.json"),
        (workspace / "tar_state" / "living_research_daemon.json", "archive/state/living_research_daemon.json"),
        (workspace / "tar_state" / "queue_maintainer_state.json", "archive/state/queue_maintainer_state.json"),
        (workspace / "tar_state" / "research_coordination_state.json", "archive/state/research_coordination_state.json"),
        (workspace / "tar_state" / "watchdog_state.json", "archive/state/watchdog_state.json"),
        (workspace / "paper", "archive/paper"),
        (repo_root / "configs", "archive/repo/configs"),
        (repo_root / "requirements.txt", "archive/repo/requirements.txt"),
        (repo_root / "requirements_gpu.txt", "archive/repo/requirements_gpu.txt"),
        (repo_root / "requirements_platform_extra.txt", "archive/repo/requirements_platform_extra.txt"),
        (repo_root / "phase10_baseline.py", "archive/repo/phase10_baseline.py"),
        (repo_root / "phase11_ablation.py", "archive/repo/phase11_ablation.py"),
        (repo_root / "phase12_ewc_sweep.py", "archive/repo/phase12_ewc_sweep.py"),
        (repo_root / "phase13_si_sweep.py", "archive/repo/phase13_si_sweep.py"),
        (repo_root / "tar_lab" / "multimodal_payloads.py", "archive/repo/tar_lab/multimodal_payloads.py"),
        (repo_root / "tar_lab" / "schemas.py", "archive/repo/tar_lab/schemas.py"),
        (repo_root / "tar_lab" / "thermoobserver.py", "archive/repo/tar_lab/thermoobserver.py"),
    ]


def validation_paper_scaffold() -> str:
    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{amsmath}\n\n"
        f"\\title{{{VALIDATION_TITLE}}}\n"
        "\\author{Christopher Gardner \\\\ TAR validation mode}\n"
        "\\date{\\today}\n\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        "\\begin{abstract}\n"
        "This draft is reserved for the single stabilisation-mode claim. "
        "Only Split-CIFAR-10 task-incremental replication evidence belongs here. "
        "CIFAR-100, TinyImageNet, class-incremental, quantum, and unfinished results are excluded.\n"
        "\\end{abstract}\n\n"
        "\\section{Claim}\n"
        f"{PRIMARY_CLAIM}\n\n"
        "\\section{Protocol}\n"
        "Methods: SGD, EWC $\\lambda=100$, EWC $\\lambda=1000$, SI $c=0.01$, TCL baseline, and high-penalty conservative. "
        "Backbone: ResNet-18. Dataset: Split-CIFAR-10 task-incremental. "
        "Metrics: forgetting, accuracy, JAF $= \\text{accuracy} - \\text{forgetting}$, variance, per-task accuracy, confusion matrices, and collapse diagnostics.\n\n"
        "\\section{Status}\n"
        "Replication pending or in progress. Replace this section with validated results only.\n\n"
        "\\section{Limitations}\n"
        "All prior breakthrough labels are provisional until the stabilisation-mode replication completes.\n\n"
        "\\end{document}\n"
    )


def scaffold_validation_outputs(workspace: Path, bundle_root: Path, mode_state: dict[str, Any]) -> dict[str, str]:
    outputs_dir = bundle_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    paper_dir = workspace / "paper" / VALIDATION_PAPER_ID
    paper_dir.mkdir(parents=True, exist_ok=True)
    tex_path = paper_dir / "main.tex"
    if not tex_path.exists():
        tex_path.write_text(validation_paper_scaffold(), encoding="utf-8")
    plan_path = paper_dir / "paper_plan.json"
    if not plan_path.exists():
        plan_path.write_text(
            json.dumps(
                {
                    "project_id": VALIDATION_PAPER_ID,
                    "title": VALIDATION_TITLE,
                    "status": "planned",
                    "readiness": "blocked",
                    "truth_status": "provisional",
                    "waiting_for_experiments": [VALIDATION_EXPERIMENT_ID],
                    "sections_planned": [
                        "abstract",
                        "introduction",
                        "related_work",
                        "method",
                        "experiments",
                        "results",
                        "limitations",
                        "conclusion",
                    ],
                    "tex_path": str(tex_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    report_md = outputs_dir / "replication_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# HPC Claim Replication Report",
                "",
                f"- Mode: `{VALIDATION_MODE_ID}`",
                f"- Claim: {PRIMARY_CLAIM}",
                f"- Bundle: `{mode_state.get('bundle_id', '')}`",
                "",
                "## Status",
                "",
                "- Environment frozen.",
                "- Validation suite prepared.",
                "- Replication results pending.",
                "",
                "## Protocol",
                "",
                "- Dataset: Split-CIFAR-10 task-incremental",
                "- Backbone: ResNet-18",
                f"- Seeds (minimum): `{mode_state.get('min_seed_list', DEFAULT_MIN_SEEDS)}`",
                f"- Seeds (target): `{mode_state.get('target_seed_list', DEFAULT_TARGET_SEEDS)}`",
                "",
                "## Required outputs",
                "",
                "- forgetting",
                "- accuracy",
                "- JAF = accuracy - forgetting",
                "- per-task accuracy",
                "- confusion matrices",
                "- collapse diagnostics",
                "- strict classification",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rejected_md = outputs_dir / "rejected_or_unsupported_claims.md"
    rejected_md.write_text(
        "\n".join(
            [
                "# Rejected or Unsupported Claims",
                "",
                "The following claims are outside stabilisation scope or unsupported pending replication:",
                "",
                "- Any CIFAR-100 claim beyond historical context.",
                "- Any TinyImageNet claim before the run is complete and independently reviewed.",
                "- Any class-incremental claim in the main validation paper.",
                "- Any quantum or non-continual-learning claim in the main validation paper.",
                "- Any blanket claim that TCL dominates all strong baselines.",
                "- Any unqualified 'breakthrough' label carried over from exploratory mode.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_md = outputs_dir / "validated_claim_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Validated Claim Summary",
                "",
                f"Primary target: {PRIMARY_CLAIM}",
                "",
                "Status: pending replication.",
                "",
                "This file should be updated only after the dedicated validation suite completes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "paper_dir": str(paper_dir),
        "tex_path": str(tex_path),
        "plan_path": str(plan_path),
        "report_markdown": str(report_md),
        "rejected_claims_markdown": str(rejected_md),
        "claim_summary_markdown": str(summary_md),
    }


def create_validation_bundle(
    workspace: Path,
    repo_root: Path,
    *,
    min_seeds: list[int] | None = None,
    target_seeds: list[int] | None = None,
) -> dict[str, Any]:
    bundle_id = f"hpc_claim_validation_{_slug_timestamp()}"
    bundle_root = validation_bundle_root(workspace, bundle_id)
    bundle_root.mkdir(parents=True, exist_ok=True)

    env_snapshot = capture_environment_snapshot(repo_root, workspace)
    config_snapshot = build_config_snapshots(min_seeds=min_seeds, target_seeds=target_seeds)
    suite_lock = validation_suite_lock_payload(
        workspace,
        min_seeds=min_seeds,
        target_seeds=target_seeds,
    )

    (bundle_root / "freeze").mkdir(parents=True, exist_ok=True)
    (bundle_root / "freeze" / "environment_snapshot.json").write_text(
        json.dumps(env_snapshot, indent=2),
        encoding="utf-8",
    )
    (bundle_root / "freeze" / "method_configs.json").write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )
    (bundle_root / "freeze" / "validation_suite_lock.json").write_text(
        json.dumps(suite_lock, indent=2),
        encoding="utf-8",
    )

    for src, rel_dst in _archive_targets(workspace, repo_root):
        _copy_path(src, bundle_root / rel_dst)

    return {
        "bundle_id": bundle_id,
        "bundle_root": str(bundle_root),
        "environment_snapshot_path": str(bundle_root / "freeze" / "environment_snapshot.json"),
        "method_configs_path": str(bundle_root / "freeze" / "method_configs.json"),
        "validation_suite_lock_path": str(bundle_root / "freeze" / "validation_suite_lock.json"),
        "validation_suite_lock": suite_lock,
    }


def build_validation_suite_spec(
    workspace: Path,
    *,
    min_seeds: list[int] | None = None,
    target_seeds: list[int] | None = None,
) -> Any:
    from tar_experiment_orchestrator import ExperimentSpec

    seed_list = list(min_seeds or DEFAULT_MIN_SEEDS)
    target_seed_list = list(target_seeds or DEFAULT_TARGET_SEEDS)
    context = {
        "why": (
            "Exploration is paused. TAR is running a single controlled replication study "
            "to verify whether high-penalty conservative TCL reduces forgetting against TCL baseline "
            "and EWC on Split-CIFAR-10 without accuracy collapse."
        ),
        "hypothesis": PRIMARY_CLAIM,
        "projected_outcome": (
            f"Validation suite prepared for {len(seed_list)} minimum seeds "
            f"(target {len(target_seed_list)} if feasible)."
        ),
        "frontier_problem": VALIDATION_FRONTIER_ID,
        "feeds_paper": VALIDATION_TITLE,
        "methodology_note": (
            "Single-claim replication suite: SGD, EWC(100), EWC(1000), SI(0.01), TCL baseline, "
            "and HPC on Split-CIFAR-10 task-incremental with ResNet-18 and matched hyperparameters."
        ),
    }
    return ExperimentSpec(
        id=VALIDATION_EXPERIMENT_ID,
        name=VALIDATION_EXPERIMENT_TITLE,
        project_id=VALIDATION_PROJECT_ID,
        hypothesis_name="hpc_claim_validation",
        dataset=VALIDATION_DATASET,
        method="validation_suite",
        seeds=seed_list,
        priority=1,
        estimated_runtime_h=max(48.0, float(len(seed_list)) * 8.0),
        backbone=VALIDATION_BACKBONE,
        epochs=VALIDATION_EPOCHS,
        description=(
            "Single-claim stabilisation suite for the high_penalty_conservative result. "
            "This run supersedes exploratory Director probes until the claim is verified or rejected."
        ),
        tags=["stabilisation_mode", "claim_validation", "hpc"],
        hardware_budget={"vram_gb": 2.5, "cpu_cores": 4},
        frontier_problem_id=VALIDATION_FRONTIER_ID,
        author_paper_id=VALIDATION_PAPER_ID,
        runner_key="hpc_claim_validation_suite",
        context=context,
        config_overrides={
            "min_seed_list": seed_list,
            "target_seed_list": target_seed_list,
            "method_order": list(VALIDATION_METHOD_ORDER),
            "target_claim": PRIMARY_CLAIM,
        },
    )


def build_validation_paper_entry(workspace: Path, orch: Any | None = None) -> dict[str, Any]:
    suite_spec = build_validation_suite_spec(workspace)
    experiment_ids = [VALIDATION_EXPERIMENT_ID]
    waiting = [VALIDATION_EXPERIMENT_ID]
    running = 0
    complete = 0
    pending = 1
    if orch is not None and hasattr(orch, "_specs"):
        spec = orch._specs.get(VALIDATION_EXPERIMENT_ID)
        if spec is not None:
            experiment_ids = [spec.id]
            if str(spec.status) == "running":
                waiting = [spec.id]
                running = 1
                pending = 0
            elif str(spec.status) == "complete":
                waiting = []
                complete = 1
                pending = 0
            else:
                waiting = [spec.id]
                pending = 1
    return {
        "project_id": VALIDATION_PAPER_ID,
        "title": VALIDATION_TITLE,
        "experiment_ids": experiment_ids,
        "frontier_problem_ids": [VALIDATION_FRONTIER_ID],
        "waiting_for_experiments": waiting,
        "complete_count": complete,
        "running_count": running,
        "pending_count": pending,
        "status": "blocked" if waiting else "ready",
        "priority": 1,
        "director_priority_score": 100.0,
        "director_recommendation": "Write exactly one defensible paper around the validated HPC claim.",
        "truth_status": "provisional",
        "readiness": "blocked" if waiting else "write_now",
        "active_domain_id": "continual_learning",
        "active_path_id": "validation-hpc-claim",
        "path_kind": "replication_check",
        "scope_status": "active",
        "director_focus": PRIMARY_CLAIM,
        "writing_policy": "single_claim_only",
        "paper_dir": str(workspace / "paper" / VALIDATION_PAPER_ID),
        "tex_path": str(workspace / "paper" / VALIDATION_PAPER_ID / "main.tex"),
        "plan_path": str(workspace / "paper" / VALIDATION_PAPER_ID / "paper_plan.json"),
        "has_pdf": bool((workspace / "paper" / VALIDATION_PAPER_ID / "main.pdf").exists()),
        "progress": {
            "complete": complete,
            "running": running,
            "pending": pending,
            "total": len(experiment_ids),
        },
        "revision_reason": "",
        "revision_requests": [],
    }


def activate_validation_mode(
    workspace: Path,
    repo_root: Path,
    *,
    min_seeds: list[int] | None = None,
    target_seeds: list[int] | None = None,
    allow_current_run_to_finish: bool = True,
) -> dict[str, Any]:
    workspace = ensure_workspace_layout(workspace, repo_root=repo_root)
    bundle = create_validation_bundle(
        workspace,
        repo_root,
        min_seeds=min_seeds,
        target_seeds=target_seeds,
    )
    state = {
        "active": True,
        "mode_id": VALIDATION_MODE_ID,
        "activated_at": _now_iso(),
        "primary_claim": PRIMARY_CLAIM,
        "primary_frontier_problem_id": VALIDATION_FRONTIER_ID,
        "primary_paper_id": VALIDATION_PAPER_ID,
        "primary_project_id": VALIDATION_PROJECT_ID,
        "primary_validation_experiment_id": VALIDATION_EXPERIMENT_ID,
        "allow_current_nonprimary_run_to_finish": bool(allow_current_run_to_finish),
        "min_seed_list": list(min_seeds or DEFAULT_MIN_SEEDS),
        "target_seed_list": list(target_seeds or DEFAULT_TARGET_SEEDS),
        "validation_suite_lock": bundle.get("validation_suite_lock", {}),
        "phase17_archive_policy": "scale_up_exploratory_incomplete_until_reviewed",
        **bundle,
    }
    outputs = scaffold_validation_outputs(workspace, Path(bundle["bundle_root"]), state)
    state.update(outputs)
    return save_state(workspace, state)
