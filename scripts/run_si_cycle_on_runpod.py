"""
scripts/run_si_cycle_on_runpod.py
TAR Self-Improvement RunPod Bridge — Cycle 2

Dispatches one operator-LoRA self-improvement cycle (run_self_improvement_cycle2.py)
to a RunPod cloud GPU pod with >=16GB VRAM, because the local GTX 1650 (4GB) cannot
run Qwen2.5-7B-Instruct LoRA fine-tuning.

SAFETY GUARANTEES
  - Pod is ALWAYS terminated in a finally block — no runaway billing
  - Cost watchdog thread kills the pod at max_cost_usd (hard ceiling)
  - Warning fires at max_cost_usd / 3 (1/3 of configured ceiling)
  - Time watchdog kills at estimated_h * watchdog_multiplier
  - Gate can HONESTLY reject — this script never hardcodes a passing score
  - --dry-run flag performs ALL pre-flight and logs the exact plan but creates
    NO pod and makes ZERO billable API calls

HARDWARE TARGET
  GPU with >=16GB VRAM (A40=48GB, 3090=24GB, 4090=24GB all qualify).
  The bridge overrides min_vram_gb to 16 when calling _create_pod.
  Estimated runtime: 1.5 h.  Hard time ceiling: 1.5 × watchdog_multiplier h.

USAGE
  # Validate-only — zero cloud cost:
  python scripts/run_si_cycle_on_runpod.py --dry-run

  # Real run (lead only — spends real money):
  $env:RUNPOD_API_KEY = "rp_…"
  python scripts/run_si_cycle_on_runpod.py

ENVIRONMENT VARIABLES (all optional — secrets file is fallback)
  RUNPOD_API_KEY        RunPod API key (required for real run; loaded from
                        tar_state/api_secrets.json if not set)
  TAR_WORKSPACE         Override workspace root (default: auto-resolved)
  TAR_WS38_BASE_MODEL   Override base model path on pod (default: auto-resolved
                        by self_improvement.py: tries env, then /workspace/models,
                        then Qwen/Qwen2.5-7B-Instruct from HF Hub)

DO NOT EDIT THE FOLLOWING FILES (read-only per task rules):
  tar_runpod_executor.py, tar_lab/self_improvement.py,
  run_self_improvement_cycle2.py, tar_dashboard.py, tar_living_research.py,
  tar_evidence_ingest.py, tar_research_director.py, tar_lab/*, vault.py

DO NOT COMMIT, PUSH, OR PROVISION REAL PODS without the lead's explicit go-ahead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Force UTF-8 stdout/stderr so the bridge's Unicode status glyphs (->, checks, etc.)
# and any pod-streamed output never crash on a Windows cp1252 console mid-run
# (a print crash after pod creation must not be a failure mode — terminate-in-finally
# still fires, but we avoid losing the result report entirely).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# ── Repo root on sys.path ──────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── Constants ──────────────────────────────────────────────────────────────────
CYCLE_ID = "cycle-202e9f28"
DELTA_ID = "delta-24de4a79"
_ESTIMATED_RUNTIME_H = 1.5          # conservative wall-clock for Qwen2.5-7B LoRA on 3090
_MIN_VRAM_GB_SI = 16                # minimum needed; prefer 24 GB (3090/4090/A40)
_REMOTE_REPO = "/workspace/repo"
_REMOTE_SI_STATE = "/workspace/si_state"
_DEP_INSTALL_TIMEOUT = 480          # seconds; GPU deps can be slow
_WORKER_LINE_BUF = 4096


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Workspace + API key bootstrap ──────────────────────────────────────────────

def _resolve_workspace() -> Path:
    """
    Resolve workspace using tar_storage.resolve_workspace() — same logic as the
    daemon. Respects TAR_WORKSPACE / TAR_STORAGE_ROOT env vars first, then prefers
    a populated drive (E: → D: → F:), then falls back to repo root.
    """
    env_ws = os.environ.get("TAR_WORKSPACE", "").strip() or os.environ.get("TAR_STORAGE_ROOT", "").strip()
    if env_ws:
        return Path(env_ws).resolve()
    try:
        from tar_storage import resolve_workspace
        return resolve_workspace(_REPO)
    except Exception:
        # Hard fallback identical to tar_runpod_control.py's _get_workspace()
        for candidate in (
            Path("E:/TAR/Thermodynamic-Continual-Learning-delivered"),
            _REPO,
        ):
            if (candidate / "tar_state").exists():
                return candidate
        return _REPO


def _load_api_key(workspace: Path) -> str:
    """
    Load RUNPOD_API_KEY from env or tar_state/api_secrets.json (mirrors
    tar_storage.storage_env secret-merge logic).  Returns empty string if missing.
    """
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    secrets_path = workspace / "tar_state" / "api_secrets.json"
    if secrets_path.exists():
        try:
            secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
            key = str(secrets.get("RUNPOD_API_KEY", "") or "").strip()
        except Exception:
            pass
    if key:
        os.environ["RUNPOD_API_KEY"] = key  # so RunPodExecutor can read os.environ directly
    return key


# ── Pre-flight checks ──────────────────────────────────────────────────────────

def _preflight(workspace: Path, api_key: str, *, dry_run: bool) -> None:
    """
    Confirm that all required state exists before touching the cloud.
    Raises SystemExit(1) with a clear message on any missing prerequisite.
    Called in BOTH dry-run and real-run paths.
    """
    errors: list[str] = []

    # 1. API key
    if not api_key:
        errors.append(
            "RUNPOD_API_KEY is not set and not found in tar_state/api_secrets.json.\n"
            "  Set it with:  $env:RUNPOD_API_KEY = 'rp_…'"
        )

    # 2. Cycle record
    cycle_path = workspace / "tar_state" / "self_improvement" / "cycles" / f"{CYCLE_ID}.json"
    if not cycle_path.exists():
        errors.append(f"Cycle record not found: {cycle_path}")
    else:
        try:
            cycle_data = json.loads(cycle_path.read_text(encoding="utf-8"))
            status = cycle_data.get("status", "")
            if status not in {"curating", "idle", "gate_failed"}:
                errors.append(
                    f"Cycle {CYCLE_ID} has status='{status}' — expected 'curating', 'idle', or "
                    f"'gate_failed'.  If status is 'training' a prior run may still be in progress."
                )
        except Exception as exc:
            errors.append(f"Could not parse cycle record: {exc}")

    # 3. Delta record (ready=True)
    delta_path = workspace / "tar_state" / "self_improvement" / "deltas" / f"{DELTA_ID}.json"
    if not delta_path.exists():
        errors.append(f"Delta record not found: {delta_path}")
    else:
        try:
            delta_data = json.loads(delta_path.read_text(encoding="utf-8"))
            if not delta_data.get("ready", False):
                errors.append(
                    f"Delta {DELTA_ID} has ready=False "
                    f"(signal_count={delta_data.get('signal_count', '?')}, "
                    f"diversity={delta_data.get('diversity_score', '?'):.3f})"
                )
            else:
                signal_ids = delta_data.get("signal_ids", [])
                print(
                    f"[preflight] Delta OK: {DELTA_ID} — {len(signal_ids)} signals, "
                    f"ready=True, diversity={delta_data.get('diversity_score', 0):.3f}"
                )
        except Exception as exc:
            errors.append(f"Could not parse delta record: {exc}")

    # 4. Signals resolve
    signals_dir = workspace / "tar_state" / "self_improvement" / "signals"
    if not signals_dir.exists():
        errors.append(f"Signals directory not found: {signals_dir}")
    else:
        try:
            delta_data  # noqa: F821 — only accessed if delta_path existed above
            signal_ids = delta_data.get("signal_ids", [])  # type: ignore[possibly-undefined]
            missing_signals = [
                sid for sid in signal_ids
                if not (signals_dir / f"{sid}.json").exists()
            ]
            if missing_signals:
                errors.append(
                    f"{len(missing_signals)} signal(s) listed in delta not found in signals dir: "
                    + ", ".join(missing_signals[:5])
                    + ("…" if len(missing_signals) > 5 else "")
                )
            else:
                print(f"[preflight] Signals OK: all {len(signal_ids)} signal files present")
        except Exception:
            pass  # delta_data not defined if delta_path was missing — already errored above

    # 5. Anchor pack + eval pack
    anchor_path = workspace / "tar_state" / "self_improvement" / "anchor_manifest.json"
    if not anchor_path.exists():
        errors.append(f"Anchor manifest not found: {anchor_path}")
    else:
        try:
            anchor_data = json.loads(anchor_path.read_text(encoding="utf-8"))
            pack_path_rel = anchor_data.get("pack_path", "")
            pack_dir = (workspace / pack_path_rel) if not Path(pack_path_rel).is_absolute() else Path(pack_path_rel)
            run_manifest = pack_dir / "run_manifest.json"
            eval_manifest = pack_dir / "eval_manifest.json"
            items_file = next(
                (pack_dir / n for n in ("eval_core.jsonl", "eval_items.jsonl") if (pack_dir / n).exists()),
                None,
            )
            if not run_manifest.exists():
                errors.append(f"Anchor run_manifest.json not found: {run_manifest}")
            elif not eval_manifest.exists():
                errors.append(f"Anchor eval_manifest.json not found (evaluate_eval_pack requires it): {eval_manifest}")
            elif items_file is None:
                errors.append(f"Anchor items file (eval_core.jsonl / eval_items.jsonl) not found in {pack_dir}")
            else:
                # Verify anchor integrity (sha256 hash check — same logic as engine.verify_anchor_integrity)
                import hashlib
                digest = hashlib.sha256()
                with run_manifest.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        digest.update(chunk)
                actual_hash = digest.hexdigest()
                expected_hash = anchor_data.get("run_manifest_hash_sha256", "")
                if actual_hash != expected_hash:
                    errors.append(
                        f"Anchor integrity FAILED: run_manifest.json hash mismatch.\n"
                        f"  expected: {expected_hash}\n"
                        f"  actual  : {actual_hash}\n"
                        f"  The anchor pack has been modified since sealing — ABORT."
                    )
                else:
                    print(
                        f"[preflight] Anchor OK: {anchor_data.get('manifest_id', '?')} "
                        f"(pack_path={pack_path_rel}, hash verified, "
                        f"baseline_mean_score={anchor_data.get('baseline_mean_score', '?')})"
                    )
        except Exception as exc:
            errors.append(f"Could not verify anchor: {exc}")

    if errors:
        print("\n[PREFLIGHT FAILED]")
        for i, err in enumerate(errors, 1):
            print(f"  [{i}] {err}")
        sys.exit(1)

    print(f"[preflight] All checks passed. {'DRY RUN — no pod will be created.' if dry_run else 'Ready to launch pod.'}")


# ── Minimal spec shim for RunPodExecutor._create_pod ──────────────────────────

def _make_si_spec(workspace: Path) -> SimpleNamespace:
    """
    Construct a minimal spec object with the attributes _create_pod reads:
      - id           : used as pod name prefix and for partial/progress filenames
      - estimated_runtime_h : used by should_use_runpod + watchdog
      - hardware_budget.vram_gb : used by should_use_runpod
    The spec does NOT need seeds/dataset/method/epochs/backbone — those are only
    read by tar_runpod_worker.py, which we do not use (we run run_self_improvement_cycle2.py).
    """
    return SimpleNamespace(
        id=f"si-{CYCLE_ID}-{DELTA_ID}",
        name=f"tar-si-cycle2",
        estimated_runtime_h=_ESTIMATED_RUNTIME_H,
        hardware_budget=SimpleNamespace(vram_gb=_MIN_VRAM_GB_SI),
        runtime_context={},
        config_overrides={},
        dataset="",
        method="",
        epochs=0,
        backbone="",
        seeds=[],
    )


# ── SI state sync to pod ───────────────────────────────────────────────────────

def _sync_si_state(client: Any, workspace: Path, delta_data: dict) -> None:
    """
    Upload the minimal SI state needed by run_self_improvement_cycle2.py.
    Layout synced to pod at /workspace/si_state/:
      self_improvement/cycles/cycle-202e9f28.json
      self_improvement/deltas/delta-24de4a79.json
      self_improvement/signals/sig-*.json   (all 29 signals in the delta)
      self_improvement/anchor_manifest.json
      eval_packs/baseline_eval_v1/run_manifest.json
      eval_packs/baseline_eval_v1/eval_items.jsonl
      human_review_state.json               (for harvest_human_review_signals)
    """
    sftp = client.open_sftp()
    created_dirs: set[str] = set()

    def _mkdir(remote_dir: str) -> None:
        if remote_dir not in created_dirs:
            _ssh_exec_simple(client, f"mkdir -p {remote_dir}", timeout=10)
            created_dirs.add(remote_dir)

    def _put(local: Path, remote: str) -> None:
        remote_dir = remote.rsplit("/", 1)[0]
        _mkdir(remote_dir)
        sftp.put(str(local), remote)

    si_local = workspace / "tar_state" / "self_improvement"
    si_remote = f"{_REMOTE_SI_STATE}/tar_state/self_improvement"

    # Cycle record
    _put(si_local / "cycles" / f"{CYCLE_ID}.json",
         f"{si_remote}/cycles/{CYCLE_ID}.json")

    # Delta record
    _put(si_local / "deltas" / f"{DELTA_ID}.json",
         f"{si_remote}/deltas/{DELTA_ID}.json")

    # All signal files referenced by the delta
    for sid in delta_data.get("signal_ids", []):
        sig_file = si_local / "signals" / f"{sid}.json"
        if sig_file.exists():
            _put(sig_file, f"{si_remote}/signals/{sid}.json")

    # Anchor manifest + eval pack
    anchor_path = workspace / "tar_state" / "self_improvement" / "anchor_manifest.json"
    _put(anchor_path, f"{si_remote}/anchor_manifest.json")

    anchor_data = json.loads(anchor_path.read_text(encoding="utf-8"))
    pack_path_rel = anchor_data.get("pack_path", "tar_state/eval_packs/baseline_eval_v1")
    pack_dir = workspace / pack_path_rel
    pack_remote = f"{_REMOTE_SI_STATE}/{pack_path_rel}"
    # Sync the ENTIRE eval pack dir, not a hardcoded 2 files: evaluate_eval_pack needs
    # eval_manifest.json and load_eval_items reads eval_core.jsonl (+ suite files +
    # scoring_rubrics.json + run_manifest.json). A 2-file hardcode caused the earlier
    # FileNotFoundError(eval_manifest.json) on the pod.
    if pack_dir.is_dir():
        for local_f in sorted(pack_dir.glob("*")):
            if local_f.is_file():
                _put(local_f, f"{pack_remote}/{local_f.name}")

    # human_review_state.json — needed for harvest_human_review_signals()
    # Non-fatal if missing (harvest() treats unreadable state as non-fatal)
    hr_path = workspace / "tar_state" / "human_review_state.json"
    if hr_path.exists():
        _put(hr_path, f"{_REMOTE_SI_STATE}/tar_state/human_review_state.json")

    sftp.close()
    print(f"[bridge] SI state synced to pod at {_REMOTE_SI_STATE}", flush=True)


def _ssh_exec_simple(client: Any, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    """Thin wrapper around paramiko exec_command — same as RunPodExecutor._ssh_exec."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


# ── Dep install ────────────────────────────────────────────────────────────────

def _install_si_deps(client: Any) -> None:
    """
    Install Python packages required by tar_lab/self_improvement.py + eval_harness.py
    beyond what the runpod/pytorch image supplies.
    The base image already has torch; we add the LoRA / HF stack + pydantic.
    """
    print("[bridge] Installing SI dependencies on pod…", flush=True)
    # pydantic v2 is required by tar_lab/schemas.py (model_validate_json, model_dump_json)
    deps = (
        "transformers>=4.40 peft>=0.10 datasets>=2.18 accelerate>=0.29 "
        "sentencepiece tiktoken pydantic>=2.0"
    )
    cmd = (
        f"pip install -q --no-warn-script-location {deps} 2>&1 | tail -12"
    )
    code, out, err = _ssh_exec_simple(client, cmd, timeout=_DEP_INSTALL_TIMEOUT)
    if code != 0:
        raise RuntimeError(f"Dep install failed (code={code}): {(out + err)[-600:]}")
    print("[bridge] SI dependencies installed", flush=True)


# ── Remote SI worker ───────────────────────────────────────────────────────────

# run_self_improvement_cycle2.py hardcodes `WORKSPACE = Path(__file__).resolve().parent`
# (the "DELIVERED" path — on the pod: /workspace/repo).  SelfImprovementEngine.__init__
# receives that as workspace_root, so tar_state/ must live inside /workspace/repo OR
# we must monkeypatch before import.  Rather than editing the launcher (forbidden), the
# bridge uploads a one-line shim that patches the WORKSPACE constant before delegating to
# the launcher's main().  No safety logic is bypassed — engine, gates, and deploy all run
# from the unmodified launcher code.
_SHIM_SCRIPT = """\
#!/usr/bin/env python3
# on-pod shim — generated by run_si_cycle_on_runpod.py; do not edit
import sys, os, pathlib

# Point the launcher at /workspace/si_state so SelfImprovementEngine finds
# the synced tar_state/self_improvement/ tree.
sys.path.insert(0, "/workspace/repo")
import run_self_improvement_cycle2 as _launcher
_launcher.WORKSPACE = pathlib.Path("/workspace/si_state")

# Also patch DELIVERED so the engine's _workspace resolves correctly
# (the engine uses Path(workspace_root).resolve() — setting WORKSPACE is enough).

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
_launcher.main(dry_run=args.dry_run)
"""

_SHIM_REMOTE = f"{_REMOTE_REPO}/si_runpod_shim.py"


def _upload_shim(client: Any) -> None:
    """Write the workspace-patch shim onto the pod via SFTP."""
    sftp = client.open_sftp()
    with sftp.open(_SHIM_REMOTE, "w") as fh:
        fh.write(_SHIM_SCRIPT.encode("utf-8"))
    sftp.close()
    print(f"[bridge] Shim uploaded to {_SHIM_REMOTE}", flush=True)


def _run_si_worker(client: Any, workspace_path: str) -> int:
    """
    Run the SI cycle on the pod via the workspace-patch shim, streaming stdout.
    Returns exit code (0 = gate passed or completed; 2 = gate failed; 1 = error).

    Why a shim?  run_self_improvement_cycle2.py hardcodes WORKSPACE = DELIVERED
    (= Path(__file__).parent on the pod, i.e. /workspace/repo).  We cannot edit
    that file (task rules), so the shim patches the WORKSPACE attribute before
    calling main() — a clean, read-only workaround.

    TAR_WS38_BASE_MODEL falls back through self_improvement._resolve_base_model_id():
      1. env TAR_WS38_BASE_MODEL (not set → skip)
      2. /workspace/models/Qwen2.5-7B-Instruct (not present → skip)
      3. Qwen/Qwen2.5-7B-Instruct from HF Hub (TAR_ALLOW_MODEL_DOWNLOAD=1 permits this)
    """
    _upload_shim(client)

    cmd = (
        f"cd {_REMOTE_REPO} && "
        "PYTHONPATH=/workspace/repo "
        "TAR_ALLOW_MODEL_DOWNLOAD=1 "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "HF_HOME=/workspace/hf_cache "
        "TORCH_HOME=/workspace/torch_cache "
        "python si_runpod_shim.py "
        "2>&1"
    )

    print(f"[bridge] Launching SI worker on pod:\n  {cmd}", flush=True)

    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(cmd)

    while not chan.exit_status_ready():
        if chan.recv_ready():
            data = chan.recv(_WORKER_LINE_BUF).decode("utf-8", errors="replace")
            for line in data.splitlines():
                if line.strip():
                    print(f"  [pod] {line}", flush=True)
        time.sleep(0.3)

    # Drain remainder
    while chan.recv_ready():
        data = chan.recv(_WORKER_LINE_BUF).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                print(f"  [pod] {line}", flush=True)

    code = chan.recv_exit_status()
    chan.close()
    print(f"[bridge] SI worker finished with exit code {code}", flush=True)
    return code


# ── Result retrieval ───────────────────────────────────────────────────────────

def _retrieve_si_results(client: Any, workspace: Path) -> dict[str, Any]:
    """
    Pull produced SI artifacts from the pod back into the local workspace.
    Never fabricates results — copies what the pod produced.

    Retrieves:
      tar_state/self_improvement/retrains/*.json  (RetainRecord with probe scores + gate verdict)
      tar_state/self_improvement/cycles/*.json    (updated CycleRecord)
      tar_state/adapters/*                        (trained LoRA adapter weights)
      tar_state/serving/active_adapter.json       (only if gate passed + deploy ran)
    """
    sftp = client.open_sftp()
    retrieved: dict[str, Any] = {"retrains": [], "adapters": [], "cycle_updated": False}

    def _safe_get(remote: str, local: Path) -> bool:
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, str(local))
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            print(f"[bridge] Retrieve warning: {remote}: {exc}", flush=True)
            return False

    def _listdir(remote: str) -> list[str]:
        try:
            return sftp.listdir(remote)
        except Exception:
            return []

    # Updated cycle record
    cycle_remote = f"{_REMOTE_SI_STATE}/tar_state/self_improvement/cycles/{CYCLE_ID}.json"
    cycle_local = workspace / "tar_state" / "self_improvement" / "cycles" / f"{CYCLE_ID}.json"
    if _safe_get(cycle_remote, cycle_local):
        retrieved["cycle_updated"] = True
        print(f"[bridge] Retrieved updated cycle record → {cycle_local}", flush=True)

    # Retrain records (there should be exactly one new one per cycle)
    retrains_remote = f"{_REMOTE_SI_STATE}/tar_state/self_improvement/retrains"
    for fname in _listdir(retrains_remote):
        if not fname.endswith(".json"):
            continue
        local_f = workspace / "tar_state" / "self_improvement" / "retrains" / fname
        if _safe_get(f"{retrains_remote}/{fname}", local_f):
            retrieved["retrains"].append(str(local_f))
            try:
                rec = json.loads(local_f.read_text(encoding="utf-8"))
                print(
                    f"[bridge] RetrainRecord {fname}: "
                    f"gate_passed={rec.get('gate_passed', '?')}, "
                    f"probe_mean_score={rec.get('probe_mean_score', '?')}, "
                    f"probe_overclaim_rate={rec.get('probe_overclaim_rate', '?')}, "
                    f"gate_failure_reason={rec.get('gate_failure_reason', None)}",
                    flush=True,
                )
            except Exception:
                pass

    # Adapter weights (potentially large — always retrieve to preserve the artifact)
    adapters_remote = f"{_REMOTE_SI_STATE}/tar_state/adapters"
    for adapter_name in _listdir(adapters_remote):
        adapter_dir_remote = f"{adapters_remote}/{adapter_name}"
        adapter_dir_local = workspace / "tar_state" / "adapters" / adapter_name
        adapter_dir_local.mkdir(parents=True, exist_ok=True)
        files_retrieved = 0
        for fname in _listdir(adapter_dir_remote):
            if _safe_get(f"{adapter_dir_remote}/{fname}", adapter_dir_local / fname):
                files_retrieved += 1
        if files_retrieved > 0:
            retrieved["adapters"].append(str(adapter_dir_local))
            print(
                f"[bridge] Adapter '{adapter_name}' retrieved: {files_retrieved} file(s) → {adapter_dir_local}",
                flush=True,
            )

    # active_adapter.json only if gate passed and deploy ran
    active_remote = f"{_REMOTE_SI_STATE}/tar_state/serving/active_adapter.json"
    active_local = workspace / "tar_state" / "serving" / "active_adapter.json"
    if _safe_get(active_remote, active_local):
        retrieved["active_adapter_updated"] = True
        print(f"[bridge] active_adapter.json retrieved → {active_local}", flush=True)

    sftp.close()
    return retrieved


# ── Dry-run plan printer ───────────────────────────────────────────────────────

def _print_dry_run_plan(workspace: Path, delta_data: dict) -> None:
    from tar_runpod_executor import load_runpod_config, _WATCHDOG_MULT

    config = load_runpod_config(workspace)
    min_vram = max(_MIN_VRAM_GB_SI, float(config.get("min_vram_gb", 24)))
    max_cost_h = float(config.get("max_cost_per_hour", 2.0))
    max_exp_cost = float(config.get("max_experiment_cost_usd", 10.0))
    watchdog_mult = float(config.get("watchdog_multiplier", _WATCHDOG_MULT))

    est_cost_low = _ESTIMATED_RUNTIME_H * 0.40   # cheapest community 3090 ~$0.40/hr
    est_cost_high = _ESTIMATED_RUNTIME_H * max_cost_h
    warn_at = max_exp_cost / 3.0
    hard_kill_at = max_exp_cost
    time_hard_kill_h = _ESTIMATED_RUNTIME_H * watchdog_mult

    signal_ids = delta_data.get("signal_ids", [])

    print()
    print("=" * 70)
    print("DRY-RUN PLAN — no pod created, zero cost")
    print("=" * 70)
    print(f"  cycle_id         : {CYCLE_ID}")
    print(f"  delta_id         : {DELTA_ID} (ready=True, {len(signal_ids)} signals)")
    print(f"  GPU target       : >={min_vram:.0f}GB VRAM, max ${max_cost_h:.2f}/hr")
    print(f"  Preferred GPUs   : {config.get('gpu_preference', [])[:4]}")
    print(f"  Estimated time   : {_ESTIMATED_RUNTIME_H:.1f} h")
    print(f"  Est. cost range  : ${est_cost_low:.2f} – ${est_cost_high:.2f}")
    print(f"  Cost WARNING at  : ${warn_at:.2f} (1/3 of max_experiment_cost_usd)")
    print(f"  Cost HARD KILL   : ${hard_kill_at:.2f} (max_experiment_cost_usd)")
    print(f"  Time HARD KILL   : {time_hard_kill_h:.1f} h ({_ESTIMATED_RUNTIME_H:.1f}h × {watchdog_mult}× watchdog_mult)")
    print()
    print("  Files to sync to pod:")
    print(f"    Code (repo)     → {_REMOTE_REPO}   (via _sync_code, ~.py + requirements)")
    print(f"    SI cycle record → {_REMOTE_SI_STATE}/tar_state/self_improvement/cycles/")
    print(f"    SI delta record → {_REMOTE_SI_STATE}/tar_state/self_improvement/deltas/")
    print(f"    SI signals ({len(signal_ids)}) → {_REMOTE_SI_STATE}/tar_state/self_improvement/signals/")
    print(f"    Anchor manifest → {_REMOTE_SI_STATE}/tar_state/self_improvement/anchor_manifest.json")
    print(f"    Eval pack       → {_REMOTE_SI_STATE}/tar_state/eval_packs/baseline_eval_v1/")
    print(f"    human_review_state.json (if present)")
    print()
    print("  Remote command (via workspace-patch shim):")
    cmd = (
        f"cd {_REMOTE_REPO} && "
        "PYTHONPATH=/workspace/repo "
        "TAR_ALLOW_MODEL_DOWNLOAD=1 "
        "HF_HOME=/workspace/hf_cache TORCH_HOME=/workspace/torch_cache "
        "python si_runpod_shim.py"
    )
    print(f"    {cmd}")
    print(f"    (shim patches run_self_improvement_cycle2.WORKSPACE → {_REMOTE_SI_STATE})")
    print()
    print("  Dep install:")
    print("    transformers>=4.40 peft>=0.10 datasets>=2.18 accelerate>=0.29")
    print("    sentencepiece tiktoken pydantic>=2.0")
    print()
    print("  On completion, bridge retrieves:")
    print("    tar_state/self_improvement/retrains/*.json")
    print("    tar_state/self_improvement/cycles/*.json")
    print("    tar_state/adapters/ws38-r1-<hash>/")
    print("    tar_state/serving/active_adapter.json  (only if gate PASSED)")
    print()
    print("  Gate behaviour:")
    print("    Gate can HONESTLY reject — bridge never hardcodes a pass score.")
    print("    Exit 2 from run_self_improvement_cycle2.py = gate FAILED.")
    print("    Exit 0 = gate PASSED and adapter deployed on the pod-side workspace.")
    print("    Either way: retrain record + probe scores are retrieved and reported.")
    print("=" * 70)
    print()


# ── Main entry point ───────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    workspace = _resolve_workspace()
    print(f"[bridge] workspace : {workspace}")
    print(f"[bridge] cycle_id  : {CYCLE_ID}")
    print(f"[bridge] delta_id  : {DELTA_ID}")
    print(f"[bridge] dry_run   : {dry_run}")
    print()

    # Load API key early so preflight can report on it
    api_key = _load_api_key(workspace)

    # Load delta data for use in preflight + plan printing + sync
    delta_path = workspace / "tar_state" / "self_improvement" / "deltas" / f"{DELTA_ID}.json"
    delta_data: dict = {}
    if delta_path.exists():
        try:
            delta_data = json.loads(delta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Always run pre-flight, even in dry-run mode
    _preflight(workspace, api_key, dry_run=dry_run)

    if dry_run:
        _print_dry_run_plan(workspace, delta_data)
        return

    # ── Real run ──────────────────────────────────────────────────────────────
    # Import RunPodExecutor read-only — we subclass nothing, just use its primitives
    from tar_runpod_executor import RunPodExecutor, load_runpod_config, _WATCHDOG_MULT

    # Override min_vram_gb for SI without editing runpod_config.json
    class _SIExecutor(RunPodExecutor):
        """
        Thin subclass that overrides min_vram_gb so _create_pod targets >=16GB for SI.
        We do NOT override any safety logic — terminate-in-finally and watchdog are inherited.
        """
        def __init__(self, workspace: Path, orchestrator: Any) -> None:
            super().__init__(workspace, orchestrator)
            # Force min VRAM for SI to 16 GB (Qwen2.5-7B needs 14GB; 16 gives headroom)
            current_min = float(self.config.get("min_vram_gb", 24))
            self.config["min_vram_gb"] = max(_MIN_VRAM_GB_SI, current_min)
            print(
                f"[bridge] RunPod config: min_vram_gb={self.config['min_vram_gb']:.0f}GB "
                f"max_cost_per_hour=${self.config.get('max_cost_per_hour', 2.0):.2f} "
                f"max_experiment_cost_usd=${self.config.get('max_experiment_cost_usd', 10.0):.2f}",
                flush=True,
            )

    # Minimal no-op orchestrator shim (RunPodExecutor.__init__ stores it as self.orch
    # and only calls self.orch.update_progress() from the progress-polling thread,
    # which we do not use in the SI bridge)
    class _NullOrch:
        def update_progress(self, *a: Any, **kw: Any) -> None:
            pass

    exec_ = _SIExecutor(workspace, _NullOrch())
    spec = _make_si_spec(workspace)

    config = exec_.config
    estimated_h = _ESTIMATED_RUNTIME_H
    watchdog_mult = float(config.get("watchdog_multiplier", _WATCHDOG_MULT))
    max_h = estimated_h * watchdog_mult
    max_cost_usd = float(config.get("max_experiment_cost_usd", 10.0))

    pod_id = ""
    price_per_hour = 0.0
    client: Any = None

    try:
        # Step 1: Create pod
        print(f"[bridge] Creating pod (target: >={_MIN_VRAM_GB_SI}GB VRAM, est {estimated_h:.1f}h)…", flush=True)
        pod_id, gpu_type, price_per_hour = exec_._create_pod(spec)
        exec_._pod_id = pod_id

        if price_per_hour > 0:
            est_cost = estimated_h * price_per_hour
            print(
                f"[bridge] Cost estimate: ~${est_cost:.2f} "
                f"({estimated_h:.1f}h × ${price_per_hour:.2f}/hr on {gpu_type}). "
                f"Warning at ${max_cost_usd/3:.2f}. Hard kill at ${max_cost_usd:.2f}.",
                flush=True,
            )

        # Step 2: Start cost watchdog (inherits RunPodExecutor logic exactly)
        exec_._cost_watchdog(pod_id, max_h, price_per_hour, max_cost_usd)

        # Step 3: Wait for SSH
        ssh_info = exec_._wait_for_ssh(pod_id)

        # Step 4: Get SSH client
        client = exec_._get_ssh_client(ssh_info)

        # Step 5: Sync code (reuse executor's _sync_code — uploads all .py, .txt, etc.)
        exec_._sync_code(client, spec)

        # Step 6: Sync SI state (cycles, delta, signals, anchor, eval pack)
        _sync_si_state(client, workspace, delta_data)

        # Step 7: Install SI-specific Python deps
        _install_si_deps(client)

        # Step 8: Run SI worker, stream stdout
        exit_code = _run_si_worker(client, str(workspace))

        # Step 9: Retrieve results regardless of exit code (gate reject = exit 2)
        print("[bridge] Retrieving SI results from pod…", flush=True)
        results = _retrieve_si_results(client, workspace)

        # Report outcome
        print()
        print("=" * 60)
        if exit_code == 0:
            print("[bridge] CYCLE COMPLETE — gate PASSED, adapter deployed")
        elif exit_code == 2:
            print("[bridge] CYCLE COMPLETE — gate FAILED (adapter NOT deployed)")
            print("  Check the retrain record for probe_mean_score + gate_failure_reason.")
            print("  No fabrication: the gate verdict is exactly what the probe produced.")
        else:
            print(f"[bridge] Worker exited with code {exit_code} (unexpected — check pod log)")
        print(f"  Retrains retrieved : {results.get('retrains', [])}")
        print(f"  Adapters retrieved : {results.get('adapters', [])}")
        print(f"  Cycle updated      : {results.get('cycle_updated', False)}")
        print(f"  Active adapter upd : {results.get('active_adapter_updated', False)}")
        print("=" * 60)

    finally:
        # Always terminate — inherits the executor's _terminate() so billing stops
        if client:
            try:
                client.close()
            except Exception:
                pass
        exec_._terminate(pod_id)
        exec_._clear_pod_state()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TAR SI RunPod Bridge — run_si_cycle_on_runpod.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pre-flight and print the plan; create NO pod, spend NOTHING",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
