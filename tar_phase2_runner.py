"""
tar_phase2_runner.py
====================
Subprocess-based runners for the TAR PhD Rehabilitation Plan Phase 2/3
pre-registered experiment scripts.

Called by tar_experiment_orchestrator.py dispatch methods.
Each function:
  - Invokes the corresponding Phase 2 script as a clean subprocess
  - Monitors stdout for progress (forwarded to progress_callback if provided)
  - Returns a result dict for the orchestrator to build an ExperimentResult
  - Does NOT call sys.exit() — raises on unrecoverable failure

Runner key → function mapping:
  "hp_selection"              → run_hp_selection()
  "hpc_replication_phase2"    → run_hpc_replication()
  "mechanistic_ablation_7c"   → run_mechanistic_ablation()
  "phase16_cifar100_rerun"    → run_phase16_cifar100_rerun()
  "phase17_tinyimagenet_rerun"→ run_phase17_tinyimagenet_rerun()
  "hpc_lambda_momentum_abl"   → run_hpc_lambda_momentum_ablation()
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parent
_VENV_PYTHON = _REPO.parent / ".venv" / "Scripts" / "python.exe"
if not _VENV_PYTHON.exists():
    _VENV_PYTHON = Path(sys.executable)

_TAR_STATE = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state")


def _run_script(
    script_name: str,
    extra_args: list[str] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    timeout_s: int = 86400,
) -> dict[str, Any]:
    """Invoke a Phase 2 script as a subprocess. Stream stdout to progress_callback."""
    script = _REPO / script_name
    if not script.exists():
        raise FileNotFoundError(f"Phase 2 script not found: {script}")

    cmd = [str(_VENV_PYTHON), str(script)] + (extra_args or [])
    proc = subprocess.Popen(
        cmd,
        cwd=str(_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines: list[str] = []
    start = time.time()
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            stripped = line.rstrip()
            lines.append(stripped)
            if progress_callback and stripped:
                progress_callback({"log_line": stripped, "elapsed_s": int(time.time() - start)})
            if time.time() - start > timeout_s:
                proc.kill()
                raise TimeoutError(f"{script_name} exceeded {timeout_s}s timeout")
    finally:
        proc.wait()

    return {
        "returncode": proc.returncode,
        "stdout_lines": lines,
        "elapsed_s": int(time.time() - start),
        "script": script_name,
    }


def _latest_comparison_json(pattern: str) -> dict[str, Any] | None:
    comp_dir = _TAR_STATE / "comparisons"
    matches = sorted(comp_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return None
    return json.loads(matches[0].read_text(encoding="utf-8"))


def run_hp_selection(
    workspace: str = "",
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run run_hyperparameter_selection.py and return selected configs."""
    result = _run_script("run_hyperparameter_selection.py",
                         progress_callback=progress_callback, timeout_s=10800)
    hp_path = _TAR_STATE / "hyperparameter_selection.json"
    selected = {}
    if hp_path.exists():
        d = json.loads(hp_path.read_text(encoding="utf-8"))
        selected = d.get("selected", {})
    return {
        **result,
        "output_file": str(hp_path) if hp_path.exists() else None,
        "selected": selected,
        "verdict": "COMPLETE" if result["returncode"] == 0 else "FAILED",
    }


def run_hpc_replication(
    workspace: str = "",
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run run_hpc_replication.py (n=20 seeds SPRT)."""
    result = _run_script("run_hpc_replication.py",
                         progress_callback=progress_callback, timeout_s=43200)
    raw = _latest_comparison_json("hpc_replication_*.json")
    verdict = "FAILED"
    p_val = None
    cohens_d = None
    if raw and result["returncode"] == 0:
        verdict = raw.get("sprt_decision", raw.get("verdict", "INCONCLUSIVE"))
        p_val = raw.get("p_wilcoxon") or raw.get("p_val")
        cohens_d = raw.get("cohens_d") or raw.get("d")
    return {
        **result,
        "result_json": raw,
        "verdict": verdict,
        "p_val": p_val,
        "cohens_d": cohens_d,
    }


def run_mechanistic_ablation(
    workspace: str = "",
    conditions: list[str] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run run_mechanistic_ablation.py (3 new conditions)."""
    args = []
    if conditions:
        args = ["--conditions", ",".join(conditions)]
    else:
        args = ["--conditions", "anchor_frozen_init,warmup_batches_60,ewc_best_lambda"]
    result = _run_script("run_mechanistic_ablation.py", extra_args=args,
                         progress_callback=progress_callback, timeout_s=21600)
    raw = _latest_comparison_json("mechanistic_ablation_*.json")
    return {
        **result,
        "result_json": raw,
        "verdict": raw.get("mechanistic_verdict", "UNKNOWN") if raw else "FAILED",
    }


def run_phase16_cifar100_rerun(
    workspace: str = "",
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run phase16_cifar100_rerun.py (5 seeds, 7 methods)."""
    result = _run_script("phase16_cifar100_rerun.py",
                         progress_callback=progress_callback, timeout_s=57600)
    raw = _latest_comparison_json("phase16_cifar100_rerun_*.json")
    tcl_forg = None
    if raw:
        methods = raw.get("method_results", {})
        tcl = methods.get("tcl", {})
        tcl_forg = tcl.get("mean_forgetting")
    return {
        **result,
        "result_json": raw,
        "verdict": "COMPLETE" if result["returncode"] == 0 and raw else "FAILED",
        "tcl_mean_forgetting": tcl_forg,
    }


def run_phase17_tinyimagenet_rerun(
    workspace: str = "",
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run phase17_tinyimagenet_rerun.py (5 seeds, 7 methods)."""
    result = _run_script("phase17_tinyimagenet_rerun.py",
                         progress_callback=progress_callback, timeout_s=86400)
    raw = _latest_comparison_json("phase17_tinyimagenet_rerun_*.json")
    tcl_forg = None
    if raw:
        methods = raw.get("method_results", {})
        tcl = methods.get("tcl", {})
        tcl_forg = tcl.get("mean_forgetting")
    return {
        **result,
        "result_json": raw,
        "verdict": "COMPLETE" if result["returncode"] == 0 and raw else "FAILED",
        "tcl_mean_forgetting": tcl_forg,
    }


def run_hpc_lambda_momentum_ablation(
    workspace: str = "",
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run run_hpc_lambda_momentum_ablation.py (4 conditions, requires HPC result)."""
    result = _run_script("run_hpc_lambda_momentum_ablation.py",
                         progress_callback=progress_callback, timeout_s=28800)
    raw = _latest_comparison_json("hpc_lambda_momentum_*.json")
    return {
        **result,
        "result_json": raw,
        "verdict": "COMPLETE" if result["returncode"] == 0 and raw else "FAILED",
    }
