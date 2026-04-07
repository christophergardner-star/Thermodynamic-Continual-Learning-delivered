from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from tar_lab.schemas import ExecutionArtifact, SandboxExecutionReport, SandboxPolicy


class SandboxedPythonExecutor:
    def __init__(self, workspace: str = ".", docker_bin: str = "docker", image: str = "python:3.11-slim"):
        self.workspace = Path(workspace).resolve()
        self.docker_bin = docker_bin
        self.image = image

    def execute(
        self,
        code: str,
        *,
        timeout_s: int = 10,
        cpu_limit: int = 1,
        memory_limit_gb: int = 1,
        network_policy: str = "off",
    ) -> SandboxExecutionReport:
        if network_policy == "profile_required":
            policy = SandboxPolicy(
                mode="docker_only",
                profile="production",
                network_policy="profile_required",
                cpu_limit=cpu_limit,
                memory_limit_gb=memory_limit_gb,
                timeout_s=timeout_s,
                workspace_root="/sandbox",
            )
            return SandboxExecutionReport(
                ok=False,
                output="",
                error="Sandbox execution requires an explicit network profile.",
                image=self.image,
                command=[],
                sandbox_policy=policy,
            )
        with TemporaryDirectory(dir=self.workspace) as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "task.py"
            output_path = tmp_path / "stdout.txt"
            script_path.write_text(code, encoding="utf-8")
            policy = SandboxPolicy(
                mode="docker_only",
                profile="production",
                network_policy=network_policy,  # type: ignore[arg-type]
                allowed_mounts=["/sandbox"],
                read_only_mounts=[],
                writable_mounts=["/sandbox"],
                cpu_limit=cpu_limit,
                memory_limit_gb=memory_limit_gb,
                timeout_s=timeout_s,
                artifact_dir="/sandbox",
                workspace_root="/sandbox",
            )
            command = [
                self.docker_bin,
                "run",
                "--rm",
                "--read-only",
                "--tmpfs",
                "/tmp",
                "--network",
                "none" if network_policy in {"off", "restricted"} else "bridge",
                "--cpus",
                str(cpu_limit),
                "--memory",
                f"{memory_limit_gb}g",
                "-v",
                f"{tmp_path}:/sandbox",
                "-w",
                "/sandbox",
                self.image,
                "sh",
                "-lc",
                "python task.py > stdout.txt 2>&1",
            ]
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
            except Exception as exc:
                return SandboxExecutionReport(
                    ok=False,
                    output="",
                    error=f"Sandbox execution unavailable: {exc}",
                    image=self.image,
                    command=command,
                    sandbox_policy=policy,
                )

            output = ""
            artifacts: list[ExecutionArtifact] = []
            if output_path.exists():
                output = output_path.read_text(encoding="utf-8", errors="replace").strip()
                artifacts.append(self._artifact(output_path, "sandbox_stdout"))
            if script_path.exists():
                artifacts.append(self._artifact(script_path, "sandbox_script"))
            if not output:
                output = ((proc.stdout or "") + (proc.stderr or "")).strip()
            return SandboxExecutionReport(
                ok=proc.returncode == 0,
                output=output,
                error=None if proc.returncode == 0 else f"Sandbox exited with code {proc.returncode}",
                image=self.image,
                command=command,
                sandbox_policy=policy,
                artifacts=artifacts,
            )

    def run(self, code: str, timeout_s: int = 10, allow_host_fallback: bool = False) -> tuple[bool, str, str]:
        report = self.execute(code, timeout_s=timeout_s)
        message = report.output if report.ok else (report.error or report.output)
        mode = "unavailable" if (not report.ok and message.startswith("Sandbox execution unavailable")) else report.mode
        return report.ok, message, mode

    @staticmethod
    def _artifact(path: Path, kind: str) -> ExecutionArtifact:
        data = path.read_bytes()
        return ExecutionArtifact(
            path=str(path),
            kind=kind,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )
