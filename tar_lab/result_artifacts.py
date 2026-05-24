from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


CANONICAL_RESULTS_INDEX = "canonical_results_index.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _normalize_payload(value.model_dump(mode="json"))
    return value


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Lockfile-protected append — prevents interleaved bytes from concurrent writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    deadline = time.monotonic() + 10.0
    lock_h = None
    while True:
        try:
            lock_h = open(lock_path, "x", encoding="utf-8")
            lock_h.write(json.dumps({"pid": os.getpid()}))
            lock_h.flush()
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise RuntimeError(f"JSONL append lock timeout: {lock_path}")
            time.sleep(0.05)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_json_default) + "\n")
    finally:
        if lock_h is not None:
            lock_h.close()
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _run_command(command: list[str], cwd: Optional[Path] = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": type(exc).__name__,
        }
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _torch_runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "torch_version": None,
        "cuda_version": None,
        "cuda_available": False,
        "gpu_name": None,
    }
    try:
        import torch
    except Exception as exc:
        info["import_error"] = str(exc)
        return info

    info.update(
        {
            "available": True,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    if info["cuda_available"]:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info["gpu_name_error"] = str(exc)
    return info


def collect_environment_snapshot(
    *,
    repo_root: Path,
    workspace: Path,
    config: Any,
    trigger: str,
    source_script: str,
    run_started_at: Optional[str] = None,
    run_ended_at: Optional[str] = None,
    manifest_path: Optional[str] = None,
    manifest_hash: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    git_head = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_status = _run_command(["git", "status", "--porcelain"], cwd=repo_root)
    pip_freeze = _run_command([sys.executable, "-m", "pip", "freeze"], cwd=repo_root)
    nvidia_smi = _run_command(["nvidia-smi"], cwd=repo_root)
    torch_info = _torch_runtime_info()

    return {
        "artifact_schema": "tar_env_snapshot_v1",
        "captured_at": utc_now_iso(),
        "run_started_at": run_started_at,
        "run_ended_at": run_ended_at,
        "trigger": trigger,
        "source_script": source_script,
        "workspace": str(workspace),
        "repo_root": str(repo_root),
        "git": {
            "head": (git_head.get("stdout") or "").strip() if git_head.get("ok") else None,
            "head_command": git_head["command"],
            "status_porcelain": git_status.get("stdout", ""),
            "status_clean": not bool((git_status.get("stdout") or "").strip()) if git_status.get("ok") else None,
            "head_error": None if git_head.get("ok") else git_head.get("stderr"),
            "status_error": None if git_status.get("ok") else git_status.get("stderr"),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
        },
        "platform": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "packages": {
            "pip_freeze": (pip_freeze.get("stdout") or "").splitlines() if pip_freeze.get("ok") else [],
            "pip_freeze_error": None if pip_freeze.get("ok") else pip_freeze.get("stderr"),
        },
        "torch": torch_info,
        "nvidia_smi": {
            "ok": nvidia_smi.get("ok"),
            "output": nvidia_smi.get("stdout", ""),
            "error": None if nvidia_smi.get("ok") else nvidia_smi.get("stderr"),
        },
        "authorization": {
            "manifest_path": manifest_path,
            "manifest_hash": manifest_hash,
        },
        "config": _normalize_payload(config),
        "extra": _normalize_payload(extra or {}),
    }


def write_append_only_result_pair(
    *,
    result_path: Path,
    payload: dict[str, Any],
    env_payload: dict[str, Any],
) -> dict[str, Path]:
    env_path = result_path.with_name(f"{result_path.stem}_env{result_path.suffix}")
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result artifact: {result_path}")
    if env_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing env artifact: {env_path}")

    try:
        _write_new_json(result_path, payload)
        _write_new_json(env_path, env_payload)
    except Exception:
        _safe_unlink(result_path)
        _safe_unlink(env_path)
        raise

    return {"result_path": result_path, "env_path": env_path}


def write_canonical_comparison_result(
    *,
    workspace: Path,
    logical_name: str,
    payload: dict[str, Any],
    env_payload: dict[str, Any],
    phase_number: Optional[int] = None,
    source_script: str,
    trust_tier: Optional[str] = None,
    provenance_status: Optional[str] = None,
    basis: Optional[str] = None,
    publication_allowed: Optional[bool] = None,
    supersedes: Optional[list[str]] = None,
    superseded_by: Optional[str] = None,
    extra_record: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    comparisons_dir = workspace / "tar_state" / "comparisons"
    stamp = utc_stamp()
    result_path = comparisons_dir / f"{logical_name}__{stamp}.json"
    env_path = comparisons_dir / f"{logical_name}__{stamp}_env.json"
    index_path = comparisons_dir / CANONICAL_RESULTS_INDEX

    if result_path.exists() or env_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite canonical comparison artifact(s): {result_path.name}, {env_path.name}"
        )

    record = {
        "logical_name": logical_name,
        "phase_number": phase_number,
        "created_at": utc_now_iso(),
        "source_script": source_script,
        "result_path": str(result_path),
        "env_path": str(env_path),
        "artifact_schema": "tar_canonical_comparison_record_v1",
        "trust_tier": trust_tier or "trusted_rerun_with_env",
        "provenance_status": provenance_status or "env_snapshot_present",
        "basis": basis or "append_only_canonical_write_with_env",
        "publication_allowed": True if publication_allowed is None else bool(publication_allowed),
        "supersedes": supersedes or [],
        "superseded_by": superseded_by or "",
    }
    if extra_record:
        record.update(_normalize_payload(extra_record))

    try:
        _write_new_json(result_path, payload)
        _write_new_json(env_path, env_payload)
        _append_jsonl(index_path, record)
    except Exception:
        _safe_unlink(result_path)
        _safe_unlink(env_path)
        raise

    return {
        "result_path": result_path,
        "env_path": env_path,
        "index_path": index_path,
    }


def iter_canonical_comparison_records(workspace: Path) -> list[dict[str, Any]]:
    index_path = workspace / "tar_state" / "comparisons" / CANONICAL_RESULTS_INDEX
    if not index_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _iter_queue_experiment_records(workspace: Path) -> list[dict[str, Any]]:
    """Return pseudo-canonical records for completed queue experiments.

    Queue experiments write to tar_state/experiments/<id>/result.json with
    spec.json as provenance, but are never written to canonical_results_index.jsonl.
    This surfaces them so build_validation_state can include them as trusted results.

    env_path prefers result_env.json (full env snapshot) and falls back to
    spec.json (runtime_context, timestamps, dependency manifest).
    """
    experiments_dir = workspace / "tar_state" / "experiments"
    if not experiments_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        result_path = exp_dir / "result.json"
        spec_path = exp_dir / "spec.json"
        if not result_path.exists() or not spec_path.exists():
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stage = str(spec.get("stage") or spec.get("status") or "")
        if stage != "complete":
            continue
        experiment_id = str(spec.get("id") or exp_dir.name)
        completed_at = str(spec.get("completed_at") or "")
        # Prefer result_env.json (full env snapshot); fall back to spec.json
        env_path = result_path.with_name("result_env.json")
        env_is_real = env_path.exists()
        if not env_is_real:
            env_path = spec_path
        records.append({
            "logical_name": experiment_id,
            "phase_number": None,
            "created_at": completed_at,
            "source_script": "queue_runner",
            "result_path": str(result_path),
            "env_path": str(env_path),
            "artifact_schema": "tar_queue_experiment_record_v1",
            "trust_tier": "trusted_rerun_with_env" if env_is_real else "corrected_recomputation_no_env",
            "provenance_status": "env_snapshot_present" if env_is_real else "missing_env_snapshot",
            "basis": "queue_experiment_with_env" if env_is_real else "queue_experiment_spec_only",
            "publication_allowed": env_is_real,
            "supersedes": [],
            "superseded_by": "",
        })
    return records


def latest_canonical_record(workspace: Path, logical_name: str) -> Optional[dict[str, Any]]:
    matches = [
        record for record in iter_canonical_comparison_records(workspace)
        if str(record.get("logical_name", "")) == logical_name
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: str(item.get("created_at", "")))
    return matches[-1]


def resolve_canonical_comparison_path(
    workspace: Path,
    logical_name: str,
    legacy_filename: Optional[str] = None,
) -> Optional[Path]:
    record = latest_canonical_record(workspace, logical_name)
    if record is not None:
        path = Path(str(record["result_path"]))
        if path.exists():
            return path
    if legacy_filename:
        legacy_path = workspace / "tar_state" / "comparisons" / legacy_filename
        if legacy_path.exists():
            return legacy_path
    return None


def load_canonical_comparison(
    workspace: Path,
    logical_name: str,
    legacy_filename: Optional[str] = None,
) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    path = resolve_canonical_comparison_path(workspace, logical_name, legacy_filename=legacy_filename)
    if path is None:
        # Fallback: queue experiments directory for completed results not in the JSONL index
        queue_path = workspace / "tar_state" / "experiments" / logical_name / "result.json"
        if queue_path.exists():
            path = queue_path
    if path is None:
        return None, None
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path, None


def _phase_number_from_name(name: str) -> Optional[int]:
    stem = Path(name).stem
    digits = []
    for ch in stem:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    try:
        return int("".join(digits))
    except ValueError:
        return None


def load_latest_phase_comparisons(workspace: Path) -> dict[int, dict[str, Any]]:
    comparisons_dir = workspace / "tar_state" / "comparisons"
    results: dict[int, dict[str, Any]] = {}

    records = iter_canonical_comparison_records(workspace)
    latest_by_phase: dict[int, dict[str, Any]] = {}
    for record in records:
        phase_number = record.get("phase_number")
        if phase_number is None:
            phase_number = _phase_number_from_name(str(record.get("logical_name", "")))
        if phase_number is None:
            continue
        current = latest_by_phase.get(int(phase_number))
        if current is None or str(record.get("created_at", "")) > str(current.get("created_at", "")):
            latest_by_phase[int(phase_number)] = record

    for phase_number, record in latest_by_phase.items():
        path = Path(str(record.get("result_path", "")))
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        results[phase_number] = data

    if not comparisons_dir.is_dir():
        return results

    for path in comparisons_dir.glob("phase*.json"):
        name = path.name
        if name.endswith("_env.json") or "__" in name:
            continue
        phase_number = _phase_number_from_name(name)
        if phase_number is None or phase_number in results:
            continue
        try:
            results[phase_number] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return results


# ── RAIL 5: verdict / statistics separation ───────────────────────────────────

_STATISTICS_FIELDS = frozenset({
    "mean_forgetting", "std_forgetting",
    "mean_accuracy",   "std_accuracy",
    "mean_delta",      "t_stat",
    "p_val",           "cohens_d",
    "n_better",
    "seed_results",    "baseline_forgetting",
    "per_seed",        "aggregate",     "pairwise",
})

_VERDICT_FIELDS = frozenset({"verdict", "verdict_key", "notes"})


def wrap_verdict_separation(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Re-structure a result payload to separate numerical statistics from
    advisory verdict labels (RAIL 5).

    Numerical fields move into payload["statistics"].
    Verdict label moves into payload["advisory_verdict"] with an explicit
    label_is_advisory_only flag.

    Fields not in either set remain at the top level (id, name, dataset, …).
    The original flat fields are kept alongside for backwards compatibility
    with readers that haven't yet been updated.
    """
    out = {k: v for k, v in payload.items()}

    stats: dict[str, Any] = {}
    for field in _STATISTICS_FIELDS:
        if field in out:
            stats[field] = out[field]

    verdict_label = str(out.get("verdict") or out.get("verdict_key") or "")
    notes = str(out.get("notes") or "")

    if stats:
        out["statistics"] = stats
    out["advisory_verdict"] = {
        "label": verdict_label,
        "notes": notes,
        "label_is_advisory_only": True,
        "interpretation": (
            "This label is generated heuristically from the numerical statistics "
            "above. It is advisory commentary, not empirical evidence. "
            "Cite the statistics, not this label."
        ),
    }

    return out


def read_statistics(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return the statistics block from a result dict, whether it uses the new
    schema (with a 'statistics' sub-dict) or the old flat schema.
    Falls back gracefully so callers work on both old and new files.
    """
    if "statistics" in result:
        return dict(result["statistics"])
    # Legacy flat schema: extract known statistical fields directly.
    stats: dict[str, Any] = {}
    for field in _STATISTICS_FIELDS:
        if field in result:
            stats[field] = result[field]
    return stats


def read_advisory_verdict(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return the advisory_verdict block.  For old schema files that only have
    a flat 'verdict' field, synthesise the advisory block on the fly.
    """
    if "advisory_verdict" in result:
        return dict(result["advisory_verdict"])
    return {
        "label": str(result.get("verdict") or result.get("verdict_key") or ""),
        "notes": str(result.get("notes") or ""),
        "label_is_advisory_only": True,
        "interpretation": (
            "Legacy result file — advisory flag applied retrospectively by reader."
        ),
    }
