"""
TAR Hardware Monitor
====================
Polls system resources every N seconds and writes:
  tar_state/hardware_state.json   — live GPU/CPU/RAM/temp metrics + per-process breakdown
  tar_state/process_registry.json — PID → experiment_id mapping (written by orchestrator)

Run as background thread:
    from tar_hardware_monitor import HardwareMonitor
    mon = HardwareMonitor(workspace); mon.start_background()

Run as standalone daemon:
    python tar_hardware_monitor.py
"""
from __future__ import annotations

import json
import os
import ctypes
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_storage import ensure_workspace_layout, resolve_workspace

_REPO = Path(__file__).resolve().parent

# ── GPU backend detection ──────────────────────────────────────────────────────
_NVML_OK = False
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    pass

_TORCH_OK = False
try:
    import torch
    _TORCH_OK = torch.cuda.is_available()
except Exception:
    pass

try:
    import psutil as _psutil
    _PSUTIL_OK = True
except Exception:
    _psutil = None   # type: ignore
    _PSUTIL_OK = False


# ── helpers ───────────────────────────────────────────────────────────────────
def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _run_capture(args: list[str], timeout_s: float = 5.0) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except Exception:
        return ""


def _parse_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text.strip())
    except Exception:
        return default


def _typeperf_value(counter: str) -> float:
    out = _run_capture(["typeperf", counter, "-sc", "1"], timeout_s=4.0)
    for line in out.splitlines():
        if line.startswith('"') and '","' in line and "PDH-CSV" not in line:
            try:
                return float(line.rsplit('","', 1)[1].rstrip('"'))
            except Exception:
                return 0.0
    return 0.0


def _gpu_snapshot() -> dict:
    """Return GPU metrics dict. Tries pynvml first, torch.cuda as fallback."""
    snap: dict[str, Any] = {
        "available": False,
        "name": "",
        "utilization_pct": 0,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "temperature_c": 0,
        "power_w": 0,
        "fan_pct": 0,
        "clock_mhz": 0,
    }

    if _NVML_OK:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
            snap.update({
                "available":       True,
                "name":            pynvml.nvmlDeviceGetName(handle),
                "utilization_pct": util.gpu,
                "vram_used_gb":    round(info.used  / 1e9, 2),
                "vram_total_gb":   round(info.total / 1e9, 2),
                "vram_free_gb":    round(info.free  / 1e9, 2),
                "temperature_c":   _safe(lambda: pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU), 0),
                "power_w":         _safe(lambda: round(
                    pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1), 0),
                "fan_pct":         _safe(lambda: pynvml.nvmlDeviceGetFanSpeed(handle), 0),
                "clock_mhz":       _safe(lambda: pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_SM), 0),
            })
            # Decode bytes name if needed
            if isinstance(snap["name"], bytes):
                snap["name"] = snap["name"].decode()
            return snap
        except Exception:
            pass

    if _TORCH_OK:
        try:
            p = torch.cuda.get_device_properties(0)
            used  = torch.cuda.memory_allocated(0)
            total = p.total_memory
            snap.update({
                "available":     True,
                "name":          p.name,
                "vram_used_gb":  round(used  / 1e9, 2),
                "vram_total_gb": round(total / 1e9, 2),
                "vram_free_gb":  round((total - used) / 1e9, 2),
                "utilization_pct": 0,  # torch.cuda doesn't expose util %
            })
        except Exception:
            pass

    out = _run_capture([
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed,clocks.sm",
        "--format=csv,noheader,nounits",
    ])
    if out:
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        if len(parts) >= 8:
            total_gb = round(_parse_float(parts[3]) / 1024.0, 2)
            used_gb = round(_parse_float(parts[2]) / 1024.0, 2)
            return {
                "available": True,
                "name": parts[0],
                "utilization_pct": int(_parse_float(parts[1])),
                "vram_used_gb": used_gb,
                "vram_total_gb": total_gb,
                "vram_free_gb": round(max(total_gb - used_gb, 0.0), 2),
                "temperature_c": int(_parse_float(parts[4])),
                "power_w": round(_parse_float(parts[5]), 1),
                "fan_pct": int(_parse_float(parts[6])),
                "clock_mhz": int(_parse_float(parts[7])),
            }

    return snap


def _gpu_process_vram_map() -> dict[int, float]:
    """
    Best-effort per-process GPU memory map in GB.
    On Windows/WDDM this may be unavailable; in that case an empty map is returned.
    """
    queries = [
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-accounted-apps=pid,gpu_memory_usage",
            "--format=csv,noheader,nounits",
        ],
    ]
    for cmd in queries:
        out = _run_capture(cmd, timeout_s=4.0)
        if not out:
            continue
        mapping: dict[int, float] = {}
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            mem_txt = parts[1].replace("MiB", "").replace("[N/A]", "").strip()
            if not mem_txt:
                continue
            mem_gb = round(_parse_float(mem_txt) / 1024.0, 2)
            mapping[pid] = mem_gb
        if mapping:
            return mapping
    return {}


def _cpu_snapshot() -> dict:
    if not _PSUTIL_OK:
        logical = os.cpu_count() or 0
        return {
            "utilization_pct": round(_typeperf_value(r"\Processor(_Total)\% Processor Time"), 1),
            "core_count": logical,
            "logical_count": logical,
            "frequency_mhz": 0,
            "temperature_c": 0,
            "load_1m": 0.0,
        }
    try:
        freq  = _safe(lambda: _psutil.cpu_freq(), None)
        temps = _safe(lambda: _psutil.sensors_temperatures(), {}) or {}
        temp_c = 0
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps:
                entries = temps[key]
                if entries:
                    temp_c = int(sum(e.current for e in entries) / len(entries))
                break
        load = _safe(lambda: _psutil.getloadavg(), (0, 0, 0))
        return {
            "utilization_pct": _safe(lambda: _psutil.cpu_percent(interval=None), 0),
            "core_count":      _psutil.cpu_count(logical=False) or 0,
            "logical_count":   _psutil.cpu_count(logical=True)  or 0,
            "frequency_mhz":   int(freq.current) if freq else 0,
            "temperature_c":   temp_c,
            "load_1m":         round(load[0], 2),
        }
    except Exception:
        return {"utilization_pct": 0, "core_count": 0, "frequency_mhz": 0,
                "temperature_c": 0, "load_1m": 0.0}


def _ram_snapshot() -> dict:
    if not _PSUTIL_OK:
        try:
            vm = _MEMORYSTATUSEX()
            vm.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(vm))
            total_gb = round(vm.ullTotalPhys / 1e9, 2)
            avail_gb = round(vm.ullAvailPhys / 1e9, 2)
            used_gb = round(max(total_gb - avail_gb, 0.0), 2)
            return {
                "used_gb": used_gb,
                "total_gb": total_gb,
                "available_gb": avail_gb,
                "percent": int(vm.dwMemoryLoad),
            }
        except Exception:
            return {"used_gb": 0, "total_gb": 0, "available_gb": 0, "percent": 0}
    try:
        vm = _psutil.virtual_memory()
        return {
            "used_gb":      round(vm.used      / 1e9, 2),
            "total_gb":     round(vm.total     / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
            "percent":      vm.percent,
        }
    except Exception:
        return {"used_gb": 0, "total_gb": 0, "available_gb": 0, "percent": 0}


def _registry_info(proc_registry: dict[str, Any], pid: int) -> tuple[str, str]:
    raw = proc_registry.get(str(pid), "")
    if isinstance(raw, dict):
        return raw.get("experiment_id", ""), raw.get("stage", "")
    return str(raw or ""), ""


def _process_snapshot(proc_registry: dict[str, Any]) -> list[dict]:
    """
    Return per-process metrics for all Python processes + any PIDs in proc_registry.
    proc_registry: {str(pid): experiment_id | {experiment_id, stage}}
    """
    gpu_vram = _gpu_process_vram_map()
    if not _PSUTIL_OK:
        raw = _run_capture([
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process python -ErrorAction SilentlyContinue | "
            "Select-Object Id,ProcessName,WS | ConvertTo-Json -Compress",
        ], timeout_s=4.0)
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            entries = parsed if isinstance(parsed, list) else [parsed]
            procs = []
            for item in entries:
                pid = int(item.get("Id", 0))
                ram_gb = round(float(item.get("WS", 0)) / 1e9, 2)
                if pid <= 0 or ram_gb < 0.05:
                    continue
                exp_id, stage = _registry_info(proc_registry, pid)
                procs.append({
                    "pid": pid,
                    "name": item.get("ProcessName", "python"),
                    "cmd_short": item.get("ProcessName", "python"),
                    "cpu_pct": 0.0,
                    "ram_gb": ram_gb,
                    "vram_gb": gpu_vram.get(pid, 0.0),
                    "status": "running",
                    "experiment_id": exp_id,
                    "stage": stage,
                })
            return procs
        except Exception:
            return []
    procs = []
    seen_pids: set[int] = set()
    # Tracked experiment processes first
    for pid_str, exp_info in proc_registry.items():
        try:
            pid = int(pid_str)
            p   = _psutil.Process(pid)
            with p.oneshot():
                cmd = " ".join(p.cmdline()[-3:])  # last 3 args only
                cpu = _safe(lambda: p.cpu_percent(interval=None), 0)
                ram = _safe(lambda: round(p.memory_info().rss / 1e9, 2), 0)
            exp_id, stage = _registry_info(proc_registry, pid)
            procs.append({
                "pid":           pid,
                "name":          p.name(),
                "cmd_short":     cmd[-60:],
                "cpu_pct":       cpu,
                "ram_gb":        ram,
                "vram_gb":       gpu_vram.get(pid, 0.0),
                "status":        p.status(),
                "experiment_id": exp_id,
                "stage":         stage,
            })
            seen_pids.add(pid)
        except Exception:
            pass
    # All other python processes
    try:
        for p in _psutil.process_iter(["pid", "name", "status"]):
            if p.info["pid"] in seen_pids:
                continue
            if "python" not in (p.info.get("name") or "").lower():
                continue
            try:
                with p.oneshot():
                    cmd = " ".join(p.cmdline()[-3:])
                    cpu = _safe(lambda: p.cpu_percent(interval=None), 0)
                    ram = _safe(lambda: round(p.memory_info().rss / 1e9, 2), 0)
                if cpu < 0.1 and ram < 0.05:
                    continue  # skip idle tiny procs
                exp_id, stage = _registry_info(proc_registry, p.info["pid"])
                procs.append({
                    "pid":           p.info["pid"],
                    "name":          p.info["name"],
                    "cmd_short":     cmd[-60:],
                    "cpu_pct":       cpu,
                    "ram_gb":        ram,
                    "vram_gb":       gpu_vram.get(p.info["pid"], 0.0),
                    "status":        p.info["status"],
                    "experiment_id": exp_id,
                    "stage":         stage,
                })
            except Exception:
                pass
    except Exception:
        pass
    return procs


# ── snapshot builder ──────────────────────────────────────────────────────────
def take_snapshot(workspace: Path) -> dict:
    proc_reg_path = workspace / "tar_state" / "process_registry.json"
    proc_registry: dict[str, str] = {}
    if proc_reg_path.exists():
        try:
            proc_registry = json.loads(proc_reg_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu":       _gpu_snapshot(),
        "cpu":       _cpu_snapshot(),
        "ram":       _ram_snapshot(),
        "processes": _process_snapshot(proc_registry),
    }


def write_snapshot(workspace: Path, snap: dict) -> None:
    path = workspace / "tar_state" / "hardware_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2), encoding="utf-8")


# ── background thread ─────────────────────────────────────────────────────────
class HardwareMonitor:
    """
    Polls hardware every `interval_s` seconds in a daemon thread.
    Call start_background() once; it runs until the process exits.
    """

    def __init__(self, workspace: Path, interval_s: float = 5.0):
        self.workspace  = workspace
        self.interval_s = interval_s
        self._thread: threading.Thread | None = None
        self._stop_ev = threading.Event()

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_ev.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hw-monitor")
        self._thread.start()
        print(f"[HardwareMonitor] Started (interval={self.interval_s}s)", flush=True)

    def stop(self) -> None:
        self._stop_ev.set()

    def _loop(self) -> None:
        # Prime psutil CPU% (first call always returns 0)
        if _PSUTIL_OK:
            _safe(lambda: _psutil.cpu_percent(interval=None))
            for p in _safe(lambda: list(_psutil.process_iter()), []):
                _safe(lambda: p.cpu_percent(interval=None))

        while not self._stop_ev.is_set():
            try:
                snap = take_snapshot(self.workspace)
                write_snapshot(self.workspace, snap)
            except Exception as exc:
                print(f"[HardwareMonitor] Error: {exc}", flush=True)
            self._stop_ev.wait(self.interval_s)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ws = ensure_workspace_layout(resolve_workspace(_REPO), repo_root=_REPO)
    print(f"[HardwareMonitor] workspace={ws}")
    print(f"[HardwareMonitor] pynvml={'OK' if _NVML_OK else 'unavailable'}  "
          f"torch={'OK' if _TORCH_OK else 'unavailable'}  "
          f"psutil={'OK' if _PSUTIL_OK else 'unavailable'}")

    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    mon = HardwareMonitor(ws, interval_s=interval)
    mon.start_background()
    print("Monitoring — Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
            snap = json.loads((ws / "tar_state" / "hardware_state.json").read_text())
            gpu  = snap["gpu"]
            cpu  = snap["cpu"]
            ram  = snap["ram"]
            print(f"  GPU {gpu.get('utilization_pct',0):3d}%  "
                  f"VRAM {gpu.get('vram_used_gb',0):.1f}/{gpu.get('vram_total_gb',0):.1f}GB  "
                  f"Temp {gpu.get('temperature_c',0)}C  |  "
                  f"CPU {cpu.get('utilization_pct',0):3d}%  "
                  f"RAM {ram.get('used_gb',0):.1f}/{ram.get('total_gb',0):.1f}GB")
    except KeyboardInterrupt:
        print("Stopped.")
