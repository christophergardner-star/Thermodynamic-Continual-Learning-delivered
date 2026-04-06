from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import List, Optional

from tar_lab.schemas import RuntimeSpec, ScoutTask

try:
    import docker  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    docker = None


@dataclass
class LaunchResult:
    mode: str
    command: List[str]
    container_name: str
    returncode: Optional[int] = None
    gpu_visible: Optional[bool] = None
    probe_output: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class EngineLimits:
    memory_limit_gb: Optional[int] = None
    cpu_limit: Optional[int] = None


class DockerRunner:
    def __init__(self, docker_bin: Optional[str] = None):
        self.docker_bin = docker_bin or self.resolve_docker_bin()

    @staticmethod
    def resolve_docker_bin() -> str:
        for env_name in ("TAR_DOCKER_BIN", "DOCKER_BIN"):
            value = os.environ.get(env_name)
            if value:
                return value

        candidates = []

        resolved = which("docker")
        if resolved:
            candidates.append(resolved)

        local_appdata = os.environ.get("LOCALAPPDATA", "")
        common_paths = [
            r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
            r"C:\Program Files\Docker\Docker\resources\docker.exe",
            r"C:\Program Files\Docker\cli-plugins\docker.exe",
        ]
        if local_appdata:
            common_paths.append(
                str(Path(local_appdata) / "Programs" / "Docker" / "Docker" / "resources" / "bin" / "docker.exe")
            )
        candidates.extend(common_paths)

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return str(Path(candidate))
        return "docker"

    def detect_engine_limits(self) -> EngineLimits:
        env_memory = os.environ.get("TAR_MEMORY_LIMIT_GB")
        env_cpu = os.environ.get("TAR_CPU_LIMIT")
        if env_memory or env_cpu:
            return EngineLimits(
                memory_limit_gb=self._parse_positive_int(env_memory),
                cpu_limit=self._parse_positive_int(env_cpu),
            )

        if docker is not None:
            try:
                client = docker.from_env()
                info = client.info()
                return EngineLimits(
                    memory_limit_gb=self._bytes_to_safe_gb(info.get("MemTotal")),
                    cpu_limit=self._parse_positive_int(info.get("NCPU")),
                )
            except Exception:
                pass

        try:
            proc = subprocess.run(
                [self.docker_bin, "info", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                info = json.loads(proc.stdout)
                return EngineLimits(
                    memory_limit_gb=self._bytes_to_safe_gb(info.get("MemTotal")),
                    cpu_limit=self._parse_positive_int(info.get("NCPU")),
                )
        except Exception:
            pass

        return EngineLimits()

    def normalize_runtime(self, runtime: RuntimeSpec) -> RuntimeSpec:
        limits = self.detect_engine_limits()
        updated = {}

        if limits.memory_limit_gb is not None:
            updated["memory_limit_gb"] = max(1, min(runtime.memory_limit_gb, limits.memory_limit_gb))
        if limits.cpu_limit is not None:
            updated["cpu_limit"] = max(1, min(runtime.cpu_limit, limits.cpu_limit))

        gpu_override = self._parse_nonnegative_int(os.environ.get("TAR_GPU_INDEX"))
        if gpu_override is not None:
            updated["gpu_index"] = gpu_override

        if not updated:
            return runtime
        return runtime.model_copy(update=updated)

    @staticmethod
    def _parse_positive_int(value: object) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _parse_nonnegative_int(value: object) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    def _bytes_to_safe_gb(value: object) -> Optional[int]:
        try:
            total_bytes = int(value)
        except (TypeError, ValueError):
            return None
        if total_bytes <= 0:
            return None
        gib = total_bytes / float(1024**3)
        safe_gib = math.floor(gib * 0.85)
        return max(1, safe_gib)

    def compose_command(self, task: ScoutTask) -> List[str]:
        runtime = self.normalize_runtime(task.runtime)
        container_name = self.container_name(task.trial_id)
        return self.compose_runtime_command(
            runtime=runtime,
            container_command=self.prepare_container_command(task.command),
            container_name=container_name,
        )

    @staticmethod
    def prepare_container_command(container_command: List[str]) -> List[str]:
        if container_command[:3] == ["python", "-m", "tar_lab.train_template"]:
            install_cmd = "python -m pip install --quiet pydantic"
            run_cmd = shlex.join(container_command)
            return ["sh", "-lc", f"{install_cmd} && {run_cmd}"]
        return container_command

    def compose_runtime_command(
        self,
        runtime: RuntimeSpec,
        container_command: List[str],
        container_name: str,
    ) -> List[str]:
        full_command = [
            self.docker_bin,
            "run",
            "--rm",
            "--name",
            container_name,
            "--memory",
            f"{runtime.memory_limit_gb}g",
            "--cpus",
            str(runtime.cpu_limit),
            "--gpus",
            f"device={runtime.gpu_index}",
            "-w",
            runtime.working_dir,
        ]
        for host_path, container_path in sorted(runtime.volumes.items()):
            full_command.extend(["-v", f"{host_path}:{container_path}"])
        for key, value in sorted(runtime.env.items()):
            full_command.extend(["-e", f"{key}={value}"])
        full_command.append(runtime.image)
        full_command.extend(container_command)
        return full_command

    def container_name(self, trial_id: str) -> str:
        return f"tar-{trial_id}"

    def launch(self, task: ScoutTask, dry_run: bool = False) -> LaunchResult:
        runtime = self.normalize_runtime(task.runtime)
        task = task.model_copy(update={"runtime": runtime})
        command = self.compose_command(task)
        name = self.container_name(task.trial_id)
        if dry_run:
            return LaunchResult(mode="dry_run", command=command, container_name=name)

        if docker is not None:
            client = docker.from_env()
            volumes = {
                host: {"bind": container, "mode": "rw"}
                for host, container in runtime.volumes.items()
            }
            environment = runtime.env
            cpu_quota = int(runtime.cpu_limit * 100000)
            container = client.containers.run(
                runtime.image,
                self.prepare_container_command(task.command),
                detach=True,
                remove=True,
                name=name,
                working_dir=runtime.working_dir,
                environment=environment,
                volumes=volumes,
                mem_limit=f"{runtime.memory_limit_gb}g",
                cpu_quota=cpu_quota,
                cpu_period=100000,
                device_requests=[
                    docker.types.DeviceRequest(device_ids=[str(runtime.gpu_index)], capabilities=[["gpu"]])
                ],
            )
            return LaunchResult(mode="docker_sdk", command=command, container_name=container.name)

        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        return LaunchResult(
            mode="subprocess",
            command=command,
            container_name=name,
            returncode=proc.returncode,
        )

    def pull_image(self, image: str) -> Optional[int]:
        if docker is not None:
            client = docker.from_env()
            client.images.pull(image)
            return 0
        proc = subprocess.run([self.docker_bin, "pull", image], capture_output=True, text=True, check=False)
        return proc.returncode

    def verify_gpu_access(self, runtime: RuntimeSpec) -> tuple[bool, str]:
        runtime = self.normalize_runtime(runtime)
        probe_command = ["nvidia-smi", "-L"]
        if docker is not None:
            client = docker.from_env()
            volumes = {
                host: {"bind": container, "mode": "rw"}
                for host, container in runtime.volumes.items()
            }
            result = client.containers.run(
                runtime.image,
                probe_command,
                detach=False,
                remove=True,
                working_dir=runtime.working_dir,
                environment=runtime.env,
                volumes=volumes,
                mem_limit=f"{runtime.memory_limit_gb}g",
                nano_cpus=int(runtime.cpu_limit * 1_000_000_000),
                device_requests=[
                    docker.types.DeviceRequest(device_ids=[str(runtime.gpu_index)], capabilities=[["gpu"]])
                ],
            )
            output = result.decode("utf-8", errors="replace") if isinstance(result, bytes) else str(result)
            return ("GPU " in output, output)

        probe = self.compose_runtime_command(
            runtime=runtime,
            container_command=probe_command,
            container_name=f"tar-gpu-probe-{runtime.gpu_index}",
        )
        proc = subprocess.run(probe, capture_output=True, text=True, check=False)
        output = (proc.stdout or "") + (proc.stderr or "")
        return (proc.returncode == 0 and "GPU " in output, output)

    def live_test(self, task: ScoutTask) -> LaunchResult:
        runtime = self.normalize_runtime(task.runtime)
        task = task.model_copy(update={"runtime": runtime})
        name = self.container_name(task.trial_id)
        pull_code = self.pull_image(runtime.image)
        if pull_code not in (None, 0):
            return LaunchResult(mode="pull_failed", command=[], container_name=name, returncode=pull_code)

        gpu_visible, probe_output = self.verify_gpu_access(runtime)
        if not gpu_visible:
            return LaunchResult(
                mode="gpu_probe_failed",
                command=self.compose_command(task),
                container_name=name,
                returncode=1,
                gpu_visible=False,
                probe_output=probe_output,
            )

        command = self.compose_command(task)
        if docker is not None:
            client = docker.from_env()
            volumes = {
                host: {"bind": container, "mode": "rw"}
                for host, container in runtime.volumes.items()
            }
            environment = runtime.env
            nano_cpus = int(runtime.cpu_limit * 1_000_000_000)
            result = client.containers.run(
                runtime.image,
                self.prepare_container_command(task.command),
                detach=False,
                remove=True,
                name=name,
                working_dir=runtime.working_dir,
                environment=environment,
                volumes=volumes,
                mem_limit=f"{runtime.memory_limit_gb}g",
                nano_cpus=nano_cpus,
                device_requests=[
                    docker.types.DeviceRequest(device_ids=[str(runtime.gpu_index)], capabilities=[["gpu"]])
                ],
            )
            stdout = result.decode("utf-8", errors="replace") if isinstance(result, bytes) else str(result)
            return LaunchResult(
                mode="docker_sdk",
                command=command,
                container_name=name,
                returncode=0,
                gpu_visible=True,
                probe_output=probe_output,
                stdout=stdout,
            )

        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        return LaunchResult(
            mode="subprocess",
            command=command,
            container_name=name,
            returncode=proc.returncode,
            gpu_visible=True,
            probe_output=probe_output,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    def panic_kill(self, dry_run: bool = False) -> List[List[str]]:
        commands = [[self.docker_bin, "ps", "-q", "--filter", "name=tar-"]]
        if dry_run:
            commands.append([self.docker_bin, "kill", "<container_ids>"])
            return commands

        proc = subprocess.run(commands[0], capture_output=True, text=True, check=False)
        ids = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if ids:
            kill_cmd = [self.docker_bin, "kill", *ids]
            subprocess.run(kill_cmd, capture_output=True, text=True, check=False)
            commands.append(kill_cmd)
        return commands
