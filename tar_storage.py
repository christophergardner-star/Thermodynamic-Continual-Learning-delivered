"""
TAR storage helpers.

Centralizes workspace selection so TAR prefers E:/D:/F: storage on this machine
and keeps large artifacts, caches, temp files, and runtime state off C: when
possible.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Mapping

_REPO = Path(__file__).resolve().parent
_WORKSPACE_NAME = _REPO.name
_PREFERRED_DRIVES = ("E", "D", "F")

_WORKSPACE_DIRS = (
    "tar_state",
    "tar_runs",
    "logs",
    "tool_envs",
    "dataset_artifacts",
    "training_artifacts",
    "eval_artifacts",
    "paper",
    "hf",
    "torch",
    "tmp",
    "pip",
    "xdg",
)


def _resolve_candidate(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _drive_workspace(drive_letter: str) -> Path:
    return Path(f"{drive_letter}:/{_REPO.parent.name}/{_WORKSPACE_NAME}")


def _workspace_score(path: Path) -> tuple[int, int]:
    populated_markers = (
        path / "tar_state",
        path / "training_artifacts",
        path / "dataset_artifacts",
        path / "paper",
    )
    populated = int(any(marker.exists() for marker in populated_markers))
    free_bytes = 0
    try:
        free_bytes = shutil.disk_usage(path.anchor or str(path.drive)).free
    except OSError:
        pass
    return populated, free_bytes


def default_workspace_candidates(repo_root: Path | None = None) -> list[Path]:
    repo_root = (repo_root or _REPO).resolve()
    candidates: list[Path] = []
    for env_name in ("TAR_WORKSPACE", "TAR_STORAGE_ROOT"):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            candidates.append(_resolve_candidate(env_value))
    for drive_letter in _PREFERRED_DRIVES:
        drive_root = Path(f"{drive_letter}:/")
        if drive_root.exists():
            candidates.append(_drive_workspace(drive_letter))
    candidates.append(repo_root)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_workspace(repo_root: Path | None = None) -> Path:
    repo_root = (repo_root or _REPO).resolve()
    for env_name in ("TAR_WORKSPACE", "TAR_STORAGE_ROOT"):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return _resolve_candidate(env_value)

    candidates = default_workspace_candidates(repo_root)

    drive_candidates = [path for path in candidates if path != repo_root]
    populated = [path for path in drive_candidates if _workspace_score(path)[0] > 0]
    if populated:
        populated.sort(key=_workspace_score, reverse=True)
        return populated[0]

    available = [path for path in drive_candidates if path.parent.exists() or Path(path.anchor).exists()]
    if available:
        available.sort(key=_workspace_score, reverse=True)
        return available[0]

    return repo_root


def storage_env(workspace: Path | None = None, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    workspace = (workspace or resolve_workspace()).resolve()
    env = dict(base_env or os.environ)
    hf_root = workspace / "hf"
    tmp_root = workspace / "tmp"
    torch_root = workspace / "torch"
    pip_root = workspace / "pip"
    xdg_root = workspace / "xdg"
    env.update({
        "TAR_WORKSPACE": str(workspace),
        "TAR_STORAGE_ROOT": str(workspace),
        "HF_HOME": str(hf_root),
        "HUGGINGFACE_HUB_CACHE": str(hf_root / "hub"),
        "TRANSFORMERS_CACHE": str(hf_root / "transformers"),
        "HF_DATASETS_CACHE": str(hf_root / "datasets"),
        "TORCH_HOME": str(torch_root),
        "PIP_CACHE_DIR": str(pip_root),
        "XDG_CACHE_HOME": str(xdg_root),
        "MPLCONFIGDIR": str(xdg_root / "matplotlib"),
        "NUMBA_CACHE_DIR": str(xdg_root / "numba"),
        "PYTHONPYCACHEPREFIX": str(tmp_root / "pycache"),
        "TEMP": str(tmp_root),
        "TMP": str(tmp_root),
        "TMPDIR": str(tmp_root),
    })
    # Merge API keys from secrets file — survives watchdog restarts and new logins.
    # Keys in the file override nothing that is already set in the environment.
    secrets_path = workspace / "tar_state" / "api_secrets.json"
    if secrets_path.exists():
        try:
            import json as _json
            secrets = _json.loads(secrets_path.read_text(encoding="utf-8"))
            if isinstance(secrets, dict):
                for k, v in secrets.items():
                    if k and v and k not in env:
                        env[str(k)] = str(v)
        except Exception:
            pass
    return env


def ensure_workspace_layout(workspace: Path | None = None, repo_root: Path | None = None) -> Path:
    workspace = (workspace or resolve_workspace(repo_root)).resolve()
    def _safe_mkdir(path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    for name in _WORKSPACE_DIRS:
        _safe_mkdir(workspace / name)
    for name in ("hub", "transformers", "datasets"):
        _safe_mkdir(workspace / "hf" / name)
    for name in ("matplotlib", "numba"):
        _safe_mkdir(workspace / "xdg" / name)
    os.environ.update(storage_env(workspace))
    return workspace


def preferred_python(repo_root: Path | None = None) -> Path:
    repo_root = (repo_root or _REPO).resolve()
    candidates = [
        repo_root.parent / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("python")


def dump_cmd_env(workspace: Path | None = None) -> str:
    env = storage_env(workspace)
    keys = (
        "TAR_WORKSPACE",
        "TAR_STORAGE_ROOT",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "NUMBA_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
        "TEMP",
        "TMP",
        "TMPDIR",
    )
    return "\n".join(f'set "{key}={env[key]}"' for key in keys)


def main() -> None:
    workspace = ensure_workspace_layout()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "workspace"
    if cmd == "workspace":
        print(workspace)
    elif cmd == "json-env":
        print(json.dumps(storage_env(workspace), indent=2))
    elif cmd == "cmd-env":
        print(dump_cmd_env(workspace))
    elif cmd == "prepare":
        print(workspace)
    else:
        raise SystemExit(f"Unknown tar_storage command: {cmd}")


if __name__ == "__main__":
    main()
