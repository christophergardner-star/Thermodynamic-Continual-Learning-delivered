from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a local TAR development environment.")
    parser.add_argument("--venv", default=".venv", help="Virtual environment directory to create/use.")
    parser.add_argument("--no-venv", action="store_true", help="Install into the current interpreter instead of a venv.")
    parser.add_argument("--gpu", action="store_true", help="Install GPU-oriented extras from requirements_gpu.txt.")
    parser.add_argument(
        "--platform-extra",
        action="store_true",
        dest="platform_extra",
        help="Install platform-sensitive extras such as qiskit, jax, and optax.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the bootstrap plan without executing installs.")
    return parser.parse_args()


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _activation_hint(venv_dir: Path) -> str:
    if os.name == "nt":
        return rf"{venv_dir}\Scripts\Activate.ps1"
    return f"source {venv_dir}/bin/activate"


def _format_command(command: Iterable[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def _run(command: list[str], *, dry_run: bool) -> None:
    print(f"> {_format_command(command)}")
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    if args.no_venv:
        python_executable = Path(sys.executable)
        activation_message = "Using current Python interpreter."
    else:
        venv_dir = (repo_root / args.venv).resolve()
        python_executable = _venv_python(venv_dir)
        activation_message = f"Activate with: {_activation_hint(venv_dir)}"
        if not python_executable.exists():
            _run([sys.executable, "-m", "venv", str(venv_dir)], dry_run=args.dry_run)

    requirement_files = ["requirements_gpu.txt" if args.gpu else "requirements.txt"]
    if args.platform_extra:
        requirement_files.append("requirements_platform_extra.txt")

    for requirement_name in requirement_files:
        requirement_path = repo_root / requirement_name
        if not requirement_path.exists():
            raise FileNotFoundError(f"Missing requirement file: {requirement_path}")

    commands = [
        [str(python_executable), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        *[
            [str(python_executable), "-m", "pip", "install", "-r", str(repo_root / requirement_name)]
            for requirement_name in requirement_files
        ],
    ]

    print("Bootstrapping TAR local environment")
    print(f"Repo root: {repo_root}")
    print(f"Python: {python_executable}")
    print(f"Requirement sets: {', '.join(requirement_files)}")
    for command in commands:
        _run(command, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete.")
    else:
        print("Bootstrap complete.")
    print(activation_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
