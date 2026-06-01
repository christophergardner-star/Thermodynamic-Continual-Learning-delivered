"""
TAR RunPod Volume Setup — one-time setup of a persistent network volume.

Creates a 50GB network volume in your preferred datacenter, pre-downloads
all TAR datasets onto it, and writes the volume_id back into
tar_state/runpod_config.json so every future pod mounts it automatically.

Run once before first RunPod experiment:
  python scripts/setup_runpod_volume.py

After this, CIFAR-100 / TinyImageNet / etc are cached on the volume.
Subsequent pods skip dataset download and start training immediately.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_workspace() -> Path:
    try:
        from tar_storage import preferred_workspace
        return Path(preferred_workspace())
    except Exception:
        for c in (
            Path("E:/TAR/Thermodynamic-Continual-Learning-delivered"),
            Path(__file__).resolve().parent.parent,
        ):
            if (c / "tar_state").exists():
                return c
        return Path(__file__).resolve().parent.parent


def _load_config(ws: Path) -> dict:
    from tar_runpod_executor import load_runpod_config
    return load_runpod_config(ws)


def _save_config(ws: Path, config: dict) -> None:
    path = ws / "tar_state" / "runpod_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Config updated: {path}")


def _wait_for_pod_ssh(rp: object, pod_id: str, timeout: int = 300) -> dict:
    print(f"Waiting for setup pod {pod_id}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = rp.get_pod(pod_id)  # type: ignore[attr-defined]
        runtime = info.get("runtime")
        if info.get("desiredStatus") == "RUNNING" and runtime:
            ports = runtime.get("ports") or []
            p = next((x for x in ports if x.get("privatePort") == 22), None)
            if p and p.get("publicPort"):
                return {"host": p["ip"], "port": int(p["publicPort"])}
        time.sleep(8)
    raise TimeoutError(f"Pod {pod_id} not ready in {timeout}s")


def main() -> None:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set.")
        sys.exit(1)

    import runpod as rp
    rp.api_key = api_key

    ws = _get_workspace()
    config = _load_config(ws)

    print("\n── TAR RunPod Volume Setup ───────────────────────────────────")

    # Check existing volume
    if config.get("volume_id"):
        print(f"Volume already configured: {config['volume_id']}")
        print("Delete it from runpod_config.json to recreate.")
        sys.exit(0)

    # List datacenters
    print("\nFetching available datacenters...")
    try:
        gpus = rp.get_gpus()
        # Extract unique datacenters — RTX 4090 community preferred
        rtx_gpus = [g for g in gpus if "4090" in g.get("displayName", "") and g.get("communityCloud")]
        if rtx_gpus:
            dc_id = "US-TX-3"  # common RunPod community datacenter
            print(f"Using datacenter: {dc_id} (community cloud, RTX 4090 available)")
        else:
            dc_id = "EU-RO-1"
            print(f"Falling back to datacenter: {dc_id}")
    except Exception as exc:
        print(f"Could not fetch GPUs: {exc}. Using default datacenter US-TX-3")
        dc_id = "US-TX-3"

    # Create network volume
    print(f"\nCreating 50GB network volume in {dc_id}...")
    try:
        volume = rp.create_network_volume(
            name="tar-datasets",
            size=50,
            datacenter_id=dc_id,
        )
        volume_id = volume["id"]
        print(f"Volume created: {volume_id}")
    except Exception as exc:
        print(f"ERROR creating volume: {exc}")
        print("You may need to create it manually in the RunPod console.")
        sys.exit(1)

    # Save to config immediately
    config["volume_id"]     = volume_id
    config["datacenter_id"] = dc_id
    _save_config(ws, config)

    # Spin up a temporary pod to pre-download datasets
    print("\nSpinning up temp pod to pre-download datasets...")
    image = str(config.get("image", "runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04"))

    # Use CPU-only for dataset download (cheaper)
    setup_pod = None
    pod_id    = ""
    try:
        setup_pod = rp.create_pod(
            name="tar-dataset-setup",
            image_name=image,
            gpu_type_id="NVIDIA RTX 4090",
            cloud_type="COMMUNITY",
            gpu_count=1,
            container_disk_in_gb=20,
            network_volume_id=volume_id,
            volume_mount_path="/runpod-volume",
            ports="22/tcp",
            support_public_ip=True,
            start_ssh=True,
            env={"PUBLIC_KEY": _get_pub_key(ws)},
        )
        pod_id = setup_pod["id"]

        ssh_info = _wait_for_pod_ssh(rp, pod_id)
        client = _get_ssh_client(ws, ssh_info)

        print("\nDownloading datasets to network volume...")
        datasets = [
            ("CIFAR-100",     "pip install datasets -q && python -c \"from datasets import load_dataset; load_dataset('uoft-cs/cifar100', cache_dir='/runpod-volume/datasets/split_cifar100')\""),
            ("TinyImageNet",  "python -c \"from datasets import load_dataset; load_dataset('zh-plus/tiny-imagenet', cache_dir='/runpod-volume/datasets/split_tinyimagenet', trust_remote_code=True)\""),
            ("CIFAR-10",      "python -c \"import torchvision; torchvision.datasets.CIFAR10('/runpod-volume/datasets/split_cifar10', download=True)\""),
        ]
        for name, cmd in datasets:
            print(f"  Downloading {name}...")
            _, stdout, stderr = client.exec_command(cmd)
            stdout.channel.recv_exit_status()
            print(f"  {name} done")

        client.close()
        print("\nAll datasets cached on network volume.")

    except Exception as exc:
        print(f"\nSetup pod error: {exc}")
        print("Volume was created — datasets can be downloaded on first pod use.")
    finally:
        if pod_id:
            try:
                rp.terminate_pod(pod_id)
                print(f"Setup pod {pod_id} terminated.")
            except Exception:
                pass

    print(f"\n── Setup Complete ────────────────────────────────────────────")
    print(f"  Volume ID:    {volume_id}")
    print(f"  Datacenter:   {dc_id}")
    print(f"  Config saved: tar_state/runpod_config.json")
    print(f"\nNow enable RunPod routing:")
    print(f"  python tar_runpod_control.py enable")
    print(f"─────────────────────────────────────────────────────────────\n")


def _get_pub_key(ws: Path) -> str:
    pub = ws / "tar_state" / "runpod_ssh" / "id_ed25519.pub"
    if pub.exists():
        return pub.read_text(encoding="utf-8").strip()
    # Auto-generate
    priv = pub.with_name("id_ed25519")
    priv.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(priv), "-N", "", "-C", "tar-runpod"],
        check=True, capture_output=True,
    )
    return pub.read_text(encoding="utf-8").strip()


def _get_ssh_client(ws: Path, ssh_info: dict) -> object:
    import paramiko
    priv = ws / "tar_state" / "runpod_ssh" / "id_ed25519"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=ssh_info["host"],
        port=ssh_info["port"],
        username="root",
        key_filename=str(priv),
        timeout=30,
        banner_timeout=60,
    )
    return client


if __name__ == "__main__":
    main()
