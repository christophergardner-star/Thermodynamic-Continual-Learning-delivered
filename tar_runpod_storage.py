"""
TAR RunPod Storage — persistent results volume management.

Every experiment run gets its own folder on a network volume that persists
independently of the pod. Results survive pod termination, credit exhaustion,
and spot preemption.

Storage layout on the network volume (/runpod-volume/ inside pod):
  experiments/
    {experiment_id}/
      manifest.json      — created before pod starts (run metadata)
      progress.json      — updated after every completed seed
      result.json        — written when all seeds complete
      run.log            — captured stdout (optional)

Local workspace mirrors:
  {workspace}/tar_state/runpod_runs/{experiment_id}/
    manifest.json        — copy of what was sent to pod
    result.json          — pulled after pod completes

After the training pod terminates, retrieve_results() spins up a tiny
retrieval pod with the same volume mounted, pulls the result files via SFTP,
and immediately terminates the retrieval pod.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_RETRIEVAL_IMAGE   = "runpod/base:0.4.2-cuda11.8.0"   # minimal, cheap
_RETRIEVAL_WAIT    = 120                                 # seconds to wait for retrieval pod SSH
_VOLUME_MOUNT_PATH = "/runpod-volume"
_EXP_PREFIX        = "experiments"


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Volume management ──────────────────────────────────────────────────────────

def ensure_results_volume(
    workspace: Path,
    config: dict[str, Any],
    api_key: str,
    *,
    log_fn: Any = print,
) -> str:
    """
    Return the network volume ID to use for results storage.
    Creates one if none is configured. Updates config file with the ID.
    """
    import runpod as rp
    rp.api_key = api_key

    volume_id = str(config.get("volume_id", "") or "").strip()
    if volume_id:
        # Verify it still exists
        try:
            user = rp.get_user()
            vols = user.get("networkVolumes") or []
            if any(str(v.get("id", "")) == volume_id for v in vols):
                log_fn(f"[storage] Using existing volume: {volume_id}")
                return volume_id
            log_fn(f"[storage] Configured volume {volume_id} not found — creating new one")
        except Exception as exc:
            log_fn(f"[storage] Volume check error: {exc} — proceeding with configured ID")
            return volume_id

    # Create a new volume
    datacenter_id = str(config.get("datacenter_id", "") or "US-TX-3").strip() or "US-TX-3"
    log_fn(f"[storage] Creating 20GB results volume in {datacenter_id}...")
    try:
        vol = rp.create_network_volume(
            name="tar-results",
            size=20,
            datacenter_id=datacenter_id,
        )
        volume_id = str(vol["id"])
        log_fn(f"[storage] Volume created: {volume_id}")
    except Exception as exc:
        log_fn(f"[storage] Could not create volume: {exc}")
        log_fn(f"[storage] Results will only be stored on pod disk (SFTP pull on completion)")
        return ""

    # Persist volume_id to config
    _update_config(workspace, {"volume_id": volume_id, "datacenter_id": datacenter_id})
    return volume_id


def _update_config(workspace: Path, updates: dict[str, Any]) -> None:
    path = workspace / "tar_state" / "runpod_config.json"
    try:
        data: dict[str, Any] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        data.update(updates)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


# ── Per-run folder setup ───────────────────────────────────────────────────────

def setup_run_folder(
    workspace: Path,
    spec: Any,
    volume_id: str,
    gpu_type: str,
    seeds: list[int],
    *,
    log_fn: Any = print,
) -> dict[str, Any]:
    """
    Write a manifest for this run to the local workspace.
    The pod will write an identical copy to the network volume on startup.
    Returns the manifest dict.
    """
    manifest = {
        "experiment_id":    str(getattr(spec, "id", "")),
        "experiment_name":  str(getattr(spec, "name", "")),
        "dataset":          str(getattr(spec, "dataset", "")),
        "method":           str(getattr(spec, "method", "")),
        "seeds":            list(seeds),
        "epochs":           int(getattr(spec, "epochs", 40)),
        "backbone":         str(getattr(spec, "backbone", "resnet18")),
        "config_overrides": dict(getattr(spec, "config_overrides", {}) or {}),
        "volume_id":        volume_id,
        "gpu_type":         gpu_type,
        "started_at":       _ts(),
        "status":           "running",
    }

    # Save locally
    local_dir = workspace / "tar_state" / "runpod_runs" / spec.id
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_fn(f"[storage] Run folder: tar_state/runpod_runs/{spec.id}/")
    return manifest


def mark_run_complete(workspace: Path, experiment_id: str, verdict: str) -> None:
    local_dir = workspace / "tar_state" / "runpod_runs" / experiment_id
    manifest_path = local_dir / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["status"] = "complete"
            data["verdict"] = verdict
            data["completed_at"] = _ts()
            manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


# ── Result retrieval ──────────────────────────────────────────────────────────

def retrieve_results(
    workspace: Path,
    spec: Any,
    volume_id: str,
    config: dict[str, Any],
    api_key: str,
    pub_key: str,
    priv_key_path: Path,
    *,
    log_fn: Any = print,
) -> dict[str, Any] | None:
    """
    Spin up a minimal retrieval pod with the results volume mounted,
    pull result.json and progress.json via SFTP, terminate retrieval pod.
    Returns parsed result dict or None.
    """
    if not volume_id:
        log_fn("[storage] No volume configured — cannot retrieve via volume")
        return None

    import runpod as rp
    rp.api_key = api_key

    cloud_type = str(config.get("cloud_type", "COMMUNITY"))
    dc_id      = str(config.get("datacenter_id", "US-TX-3") or "US-TX-3")

    # Try to find a cheap CPU-capable GPU in the same datacenter
    retrieval_gpu = "NVIDIA GeForce RTX 3090"  # cheapest with community cloud usually

    log_fn(f"[storage] Spinning retrieval pod to pull results from volume {volume_id}...")
    retrieval_pod_id = ""
    try:
        pod = rp.create_pod(
            name=f"tar-retrieve-{spec.id[:8]}",
            image_name=_RETRIEVAL_IMAGE,
            gpu_type_id=retrieval_gpu,
            cloud_type=cloud_type,
            gpu_count=1,
            container_disk_in_gb=5,
            network_volume_id=volume_id,
            volume_mount_path=_VOLUME_MOUNT_PATH,
            ports="22/tcp",
            support_public_ip=True,
            start_ssh=True,
            env={"PUBLIC_KEY": pub_key},
        )
        retrieval_pod_id = str(pod["id"])
        log_fn(f"[storage] Retrieval pod: {retrieval_pod_id}")

        # Wait for SSH
        ssh_info = _wait_for_ssh(rp, retrieval_pod_id, timeout=_RETRIEVAL_WAIT)
        result = _pull_results_via_sftp(
            workspace, spec, ssh_info, priv_key_path, log_fn=log_fn
        )
        return result

    except Exception as exc:
        log_fn(f"[storage] Retrieval pod error: {exc}")
        return None
    finally:
        if retrieval_pod_id:
            try:
                rp.terminate_pod(retrieval_pod_id)
                log_fn(f"[storage] Retrieval pod {retrieval_pod_id} terminated")
            except Exception:
                pass


def _wait_for_ssh(rp: Any, pod_id: str, timeout: int = 120) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = rp.get_pod(pod_id)
        runtime = info.get("runtime")
        if info.get("desiredStatus") == "RUNNING" and runtime:
            ports = runtime.get("ports") or []
            p = next((x for x in ports if x.get("privatePort") == 22), None)
            if p and p.get("publicPort"):
                return {"host": p["ip"], "port": int(p["publicPort"])}
        time.sleep(6)
    raise TimeoutError(f"Retrieval pod {pod_id} not SSH-ready in {timeout}s")


def _pull_results_via_sftp(
    workspace: Path,
    spec: Any,
    ssh_info: dict[str, Any],
    priv_key_path: Path,
    *,
    log_fn: Any = print,
) -> dict[str, Any] | None:
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=ssh_info["host"],
            port=ssh_info["port"],
            username="root",
            key_filename=str(priv_key_path),
            timeout=20,
            banner_timeout=40,
        )
        sftp = client.open_sftp()

        exp_id   = getattr(spec, "id", "")
        vol_base = f"{_VOLUME_MOUNT_PATH}/{_EXP_PREFIX}/{exp_id}"
        local_dir = workspace / "tar_state" / "runpod_runs" / exp_id
        local_dir.mkdir(parents=True, exist_ok=True)

        result_data: dict[str, Any] | None = None
        for filename in ("result.json", "progress.json", "manifest.json"):
            remote = f"{vol_base}/{filename}"
            local  = local_dir / filename
            try:
                sftp.get(remote, str(local))
                log_fn(f"[storage] Pulled: {filename}")
                if filename == "result.json":
                    result_data = json.loads(local.read_text(encoding="utf-8"))
            except FileNotFoundError:
                log_fn(f"[storage] Not found on volume: {filename}")
            except Exception as exc:
                log_fn(f"[storage] Pull error {filename}: {exc}")

        # Copy result to canonical location
        if result_data:
            canonical = workspace / "tar_state" / "experiments" / exp_id / "result.json"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
            log_fn(f"[storage] Result saved to tar_state/experiments/{exp_id}/result.json")

        sftp.close()
        return result_data

    finally:
        try:
            client.close()
        except Exception:
            pass


# ── Volume remote-path helpers (used by worker) ───────────────────────────────

def volume_exp_path(experiment_id: str) -> str:
    return f"{_VOLUME_MOUNT_PATH}/{_EXP_PREFIX}/{experiment_id}"


def volume_result_path(experiment_id: str) -> str:
    return f"{volume_exp_path(experiment_id)}/result.json"


def volume_progress_path(experiment_id: str) -> str:
    return f"{volume_exp_path(experiment_id)}/progress.json"
