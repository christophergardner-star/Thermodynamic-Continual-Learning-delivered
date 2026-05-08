"""
TAR experiment preflight.

Guarantees that each experiment starts from a clean runtime sandbox, that prior
results are archived before the next run plan is chosen, and that missing tools
are installed into a workspace-local environment on a non-C drive when
possible.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_storage import ensure_workspace_layout, storage_env


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(text: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "-" for ch in text]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "experiment"


def _json_load(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@dataclass
class DependencyAction:
    module: str
    package: str
    status: str
    detail: str = ""


@dataclass
class ArchiveAction:
    source: str
    destination: str
    action: str


@dataclass
class PreflightReport:
    experiment_id: str
    prepared_at: str
    workspace: str
    runtime_root: str
    current_runtime_dir: str
    python_executable: str
    bootstrap_venv: str
    execution_mode: str
    clean_state_ready: bool
    resume_preserved: bool
    archive_actions: list[ArchiveAction] = field(default_factory=list)
    dependency_actions: list[DependencyAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    manifest_path: str = ""


class ExperimentPreflightManager:
    def __init__(self, workspace: Path, repo_root: Path):
        self.workspace = ensure_workspace_layout(workspace, repo_root=repo_root)
        self.repo_root = repo_root
        self.archive_root = self.workspace / "tar_state" / "experiment_archives"
        self.runtime_root = self.workspace / "tar_runs" / "experiment_runtime"
        self.tool_env_root = self.workspace / "tool_envs"

    def prepare(self, spec: Any) -> PreflightReport:
        experiment_runtime_root = self.runtime_root / spec.id
        archive_actions = self._archive_completed_results()
        current_runtime_dir, rotation_actions = self._prepare_runtime_dir(experiment_runtime_root)
        archive_actions.extend(rotation_actions)

        profile = self._dependency_profile(spec)
        execution_mode = "in_process"
        python_executable = Path(sys.executable)
        bootstrap_venv = self.tool_env_root / profile["profile_name"]

        required_modules = list(profile.get("required_modules", []))
        optional_modules = list(profile.get("optional_modules", []))
        current_missing_required = self._missing_modules(python_executable, required_modules)
        current_missing_optional = self._missing_modules(python_executable, optional_modules)
        dependency_actions = [
            DependencyAction(
                module=mod,
                package=profile["packages"].get(mod, mod),
                status=(
                    "available"
                    if mod not in current_missing_required and mod not in current_missing_optional
                    else "optional_missing" if mod in current_missing_optional else "missing"
                ),
            )
            for mod in profile["modules"]
        ]

        if current_missing_required:
            install_actions = self._ensure_workspace_environment(
                venv_dir=bootstrap_venv,
                profile=profile,
                missing_modules=current_missing_required,
                log_path=current_runtime_dir / "bootstrap.log",
            )
            dependency_actions.extend(install_actions)
            venv_python = self._venv_python(bootstrap_venv)
            venv_missing = self._missing_modules(venv_python, required_modules) if venv_python.exists() else current_missing_required
            if venv_python.exists() and not venv_missing:
                execution_mode = "workspace_venv"
                python_executable = venv_python
                dependency_actions.append(DependencyAction(
                    module="__environment__",
                    package=str(bootstrap_venv),
                    status="ready",
                    detail="Workspace-local experiment environment prepared.",
                ))
            elif current_missing_required:
                unresolved = ", ".join(sorted(current_missing_required))
                dependency_actions.append(DependencyAction(
                    module="__environment__",
                    package=str(bootstrap_venv),
                    status="failed",
                    detail=f"Missing modules remain unresolved: {unresolved}",
                ))
                raise RuntimeError(
                    f"Experiment preflight could not resolve required tools in workspace env {bootstrap_venv}: {unresolved}"
                )

        for subdir in ("tmp", "logs", "cache", "manifests", "artifacts"):
            (current_runtime_dir / subdir).mkdir(parents=True, exist_ok=True)

        runtime_env = storage_env(self.workspace)
        pythonpath_parts = [str(self.repo_root)]
        existing_pythonpath = runtime_env.get("PYTHONPATH", "") or os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        runtime_env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
        runtime_env["TAR_EXPERIMENT_RUNTIME_DIR"] = str(current_runtime_dir)
        runtime_env["TAR_EXPERIMENT_ID"] = str(spec.id)
        runtime_env["TAR_EXPERIMENT_PROJECT_ID"] = str(spec.project_id)

        resume_preserved = bool(
            getattr(spec, "runner_key", "") in {"phase16_scale_up_suite", "phase17_tinyimagenet_suite"}
            and (self.workspace / "tar_state" / "suite_checkpoints").exists()
        )
        notes = [
            "Preflight archived prior result state before scheduling execution.",
            "Runtime temp/cache/env paths are pinned to the workspace rather than C:.",
        ]
        if resume_preserved:
            notes.append("Suite checkpoint state was preserved so resume-safe runs do not lose progress.")

        report = PreflightReport(
            experiment_id=str(spec.id),
            prepared_at=_now_iso(),
            workspace=str(self.workspace),
            runtime_root=str(experiment_runtime_root),
            current_runtime_dir=str(current_runtime_dir),
            python_executable=str(python_executable),
            bootstrap_venv=str(bootstrap_venv),
            execution_mode=execution_mode,
            clean_state_ready=True,
            resume_preserved=resume_preserved,
            archive_actions=archive_actions,
            dependency_actions=dependency_actions,
            notes=notes,
        )
        manifest_path = current_runtime_dir / "manifests" / "preflight.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        report.manifest_path = str(manifest_path)
        manifest_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        return report

    def environment_for_subprocess(self, report: PreflightReport | dict[str, Any]) -> dict[str, str]:
        current_runtime_dir = report.current_runtime_dir if hasattr(report, "current_runtime_dir") else str(report.get("current_runtime_dir", ""))
        manifest_path = report.manifest_path if hasattr(report, "manifest_path") else str(report.get("manifest_path", ""))
        env = storage_env(self.workspace)
        pythonpath_parts = [str(self.repo_root)]
        existing_pythonpath = env.get("PYTHONPATH", "") or os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
        env["TAR_EXPERIMENT_RUNTIME_DIR"] = current_runtime_dir
        env["TAR_EXPERIMENT_PREFLIGHT_MANIFEST"] = manifest_path
        return env

    def _archive_completed_results(self) -> list[ArchiveAction]:
        actions: list[ArchiveAction] = []
        queue_path = self.workspace / "tar_state" / "experiment_queue.json"
        if not queue_path.exists():
            return actions
        raw = _json_load(queue_path)
        experiments = raw.get("experiments", []) if isinstance(raw, dict) else []
        for rec in experiments:
            result_path = Path(str(rec.get("result_path", "") or ""))
            exp_id = str(rec.get("id", "") or "")
            if not exp_id or not result_path.exists():
                continue
            completed_at = str(rec.get("completed_at", "") or "")[:19].replace(":", "-")
            stamp = completed_at or datetime.fromtimestamp(result_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            dest_dir = self.archive_root / exp_id / stamp
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_result = dest_dir / result_path.name
            if not dest_result.exists():
                shutil.copy2(result_path, dest_result)
                actions.append(ArchiveAction(str(result_path), str(dest_result), "archived_result"))
            spec_path = result_path.parent / "spec.json"
            if spec_path.exists():
                dest_spec = dest_dir / "spec.json"
                if not dest_spec.exists():
                    shutil.copy2(spec_path, dest_spec)
                    actions.append(ArchiveAction(str(spec_path), str(dest_spec), "archived_spec"))
        return actions

    def _prepare_runtime_dir(self, experiment_runtime_root: Path) -> tuple[Path, list[ArchiveAction]]:
        actions: list[ArchiveAction] = []
        current_runtime_dir = experiment_runtime_root / "current"
        if current_runtime_dir.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            rotated = current_runtime_dir.parent / f"previous-{stamp}"
            rotated.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(current_runtime_dir), str(rotated))
                actions.append(ArchiveAction(str(current_runtime_dir), str(rotated), "rotated_runtime"))
            except (OSError, PermissionError):
                current_runtime_dir = experiment_runtime_root / f"run-{stamp}"
                actions.append(ArchiveAction(
                    str(experiment_runtime_root / "current"),
                    str(current_runtime_dir),
                    "runtime_locked_new_session",
                ))
        current_runtime_dir.mkdir(parents=True, exist_ok=True)
        return current_runtime_dir, actions

    def _dependency_profile(self, spec: Any) -> dict[str, Any]:
        modules = ["numpy", "scipy", "torch", "PIL", "torchvision"]
        packages = {
            "numpy": "numpy",
            "scipy": "scipy",
            "torch": "torch",
            "PIL": "Pillow",
            "torchvision": "torchvision",
            "datasets": "datasets",
            "psutil": "psutil",
            "tar_lab": "local_repo",
            "tar_runtime_tracking": "local_repo",
            "tar_suite_checkpoint": "local_repo",
        }
        optional_modules = ["psutil"]
        if str(getattr(spec, "dataset", "")) in {"split_cifar100", "split_tinyimagenet"}:
            modules.extend(["datasets", "tar_runtime_tracking", "tar_suite_checkpoint"])
        else:
            modules.extend(["tar_lab"])

        required_modules = list(dict.fromkeys(modules))
        modules = list(dict.fromkeys(required_modules + optional_modules))
        dataset_label = str(getattr(spec, "dataset", "generic") or "generic").replace("_", "-")
        return {
            "profile_name": f"experiment-{_safe_slug(dataset_label)}",
            "modules": modules,
            "required_modules": required_modules,
            "optional_modules": optional_modules,
            "packages": packages,
            "requirements_gpu": False,
            "requirements_platform_extra": False,
        }

    def _venv_python(self, venv_dir: Path) -> Path:
        if os.name == "nt":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    def _missing_modules(self, python_executable: Path, modules: list[str]) -> set[str]:
        if not python_executable.exists():
            return set(modules)
        env = storage_env(self.workspace)
        env["PYTHONPATH"] = os.pathsep.join([
            str(self.repo_root),
            env.get("PYTHONPATH", "") or os.environ.get("PYTHONPATH", ""),
        ]).strip(os.pathsep)
        probe = (
            "import importlib.util, json; "
            f"mods={json.dumps(modules)}; "
            "print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}))"
        )
        try:
            proc = subprocess.run(
                [str(python_executable), "-c", probe],
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=str(self.repo_root),
                timeout=60,
            )
            result = json.loads(proc.stdout.strip() or "{}")
            return {mod for mod, ok in result.items() if not ok}
        except Exception:
            return set(modules)

    def _ensure_workspace_environment(
        self,
        *,
        venv_dir: Path,
        profile: dict[str, Any],
        missing_modules: set[str],
        log_path: Path,
    ) -> list[DependencyAction]:
        actions: list[DependencyAction] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)
        venv_python = self._venv_python(venv_dir)
        if not venv_python.exists():
            actions.append(DependencyAction(
                module="__venv__",
                package=str(venv_dir),
                status="creating",
                detail="Creating workspace-local environment for experiment tools.",
            ))
        env = storage_env(self.workspace)
        env["PYTHONPATH"] = os.pathsep.join([
            str(self.repo_root),
            env.get("PYTHONPATH", "") or os.environ.get("PYTHONPATH", ""),
        ]).strip(os.pathsep)
        command = [
            str(sys.executable),
            str(self.repo_root / "bootstrap.py"),
            "--venv",
            str(venv_dir),
        ]
        if profile.get("requirements_gpu"):
            command.append("--gpu")
        if profile.get("requirements_platform_extra"):
            command.append("--platform-extra")
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[{_now_iso()}] bootstrap start missing={sorted(missing_modules)}\n")
                subprocess.run(
                    command,
                    check=True,
                    env=env,
                    cwd=str(self.repo_root),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            actions.append(DependencyAction(
                module="__bootstrap__",
                package=str(venv_dir),
                status="installed",
                detail=f"Installed/verified requirements into {venv_dir}",
            ))
        except Exception as exc:
            actions.append(DependencyAction(
                module="__bootstrap__",
                package=str(venv_dir),
                status="failed",
                detail=f"Workspace bootstrap failed: {exc}",
            ))
        return actions
