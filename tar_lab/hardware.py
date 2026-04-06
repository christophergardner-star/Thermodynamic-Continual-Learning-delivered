from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from shutil import which
from typing import Callable, Optional


Runner = Callable[..., subprocess.CompletedProcess]


@dataclass
class GpuTelemetry:
    temperature_c: Optional[float] = None
    memory_temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    target_temperature_c: Optional[float] = None


class NvidiaSMI:
    """Thin wrapper around nvidia-smi.

    Local NVSMI v581.83 exposes `-gtt/--gpu-target-temp`; there is no `-tt`
    write flag. The lab uses `-pl` for power capping and `-q -d TEMPERATURE,POWER`
    for telemetry reads.
    """

    def __init__(self, runner: Runner = subprocess.run):
        self.runner = runner

    def available(self) -> bool:
        return which("nvidia-smi") is not None

    def _run(self, args: list[str], apply: bool) -> subprocess.CompletedProcess | None:
        if not apply:
            return None
        return self.runner(args, capture_output=True, text=True, check=False)

    def set_power_limit(self, gpu_index: int, watts: int, apply: bool = False) -> list[str]:
        cmd = ["nvidia-smi", "-i", str(gpu_index), "-pl", str(watts)]
        self._run(cmd, apply=apply)
        return cmd

    def set_gpu_target_temp(self, gpu_index: int, temp_c: int, apply: bool = False) -> list[str]:
        cmd = ["nvidia-smi", "-i", str(gpu_index), "-gtt", str(temp_c)]
        self._run(cmd, apply=apply)
        return cmd

    def prepare_run(
        self,
        gpu_index: int,
        power_limit_w: int,
        gpu_target_temp_c: int,
        apply: bool = False,
    ) -> list[list[str]]:
        return [
            self.set_power_limit(gpu_index, power_limit_w, apply=apply),
            self.set_gpu_target_temp(gpu_index, gpu_target_temp_c, apply=apply),
        ]

    def query_telemetry(self, gpu_index: int = 0) -> GpuTelemetry:
        if not self.available():
            return GpuTelemetry()
        proc = self.runner(
            ["nvidia-smi", "-i", str(gpu_index), "-q", "-d", "TEMPERATURE,POWER"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = proc.stdout or ""
        return GpuTelemetry(
            temperature_c=_extract_float(text, r"GPU Current Temp\s+:\s+([0-9.]+) C"),
            memory_temperature_c=_extract_float(text, r"Memory Current Temp\s+:\s+([0-9.]+) C"),
            power_w=_extract_float(text, r"Instantaneous Power Draw\s+:\s+([0-9.]+) W"),
            power_limit_w=_extract_float(text, r"Current Power Limit\s+:\s+([0-9.]+) W"),
            target_temperature_c=_extract_float(text, r"GPU Target Temperature\s+:\s+([0-9.]+) C"),
        )

    def list_gpus(self) -> list[str]:
        if not self.available():
            return []
        proc = self.runner(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


def _extract_float(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))
