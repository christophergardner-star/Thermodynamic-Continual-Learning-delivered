"""
TAR RunPod Executor — cloud GPU offloading for long-running experiments.

Lifecycle: create pod → wait for SSH → sync code → install deps →
prepare dataset → run worker → poll progress → sync result → terminate pod.

Safety guarantees:
  - terminate_pod() is always called in a finally block — no runaway billing
  - Watchdog thread kills pod at estimated_h × watchdog_multiplier regardless
  - Seed-level resume: interrupted runs continue from the last completed seed
  - Credit exhaustion auto-suspends RunPod routing and leaves TAR healthy
  - Dry-run mode: RUNPOD_DRY_RUN=1 logs all actions without touching the API
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_GPU_PREFERENCE = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX 5090",
    "NVIDIA A40",
]
_DEFAULT_MIN_VRAM_GB    = 24
_DEFAULT_MAX_COST_PER_H = 2.0
_DEFAULT_IMAGE = "runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04"
_DEFAULT_THRESHOLD_H = 12.0
_DEFAULT_THRESHOLD_VRAM = 3.9
_WATCHDOG_MULT = 2.5
_SSH_READY_TIMEOUT = 360
_PROGRESS_POLL_S = 30
_POD_CREATE_RETRY_WAIT_S = 45
_DATASET_DOWNLOAD_TIMEOUT = 600
_DEP_INSTALL_TIMEOUT = 360
_WORKER_LINE_BUF = 4096

_CREDIT_ERROR_KEYWORDS = (
    "insufficient funds", "insufficient credit", "billing",
    "credit", "payment", "out of credits", "account balance",
)

# ── Exceptions ────────────────────────────────────────────────────────────────

class RunPodNoGPUError(RuntimeError):
    pass

class RunPodInterruptedError(RuntimeError):
    pass

class RunPodCreditError(RuntimeError):
    pass

# ── Config helpers ─────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_runpod_config(workspace: Path) -> dict[str, Any]:
    path = workspace / "tar_state" / "runpod_config.json"
    defaults: dict[str, Any] = {
        "enabled": False,
        "threshold_runtime_h": _DEFAULT_THRESHOLD_H,
        "threshold_vram_gb": _DEFAULT_THRESHOLD_VRAM,
        "gpu_preference": _DEFAULT_GPU_PREFERENCE,
        "cloud_type": "COMMUNITY",
        "image": _DEFAULT_IMAGE,
        "volume_id": "",
        "datacenter_id": "",
        "watchdog_multiplier": _WATCHDOG_MULT,
        "ssh_key_path": "",
        "min_vcpu_count": 4,
        "min_memory_in_gb": 16,
        "container_disk_in_gb": 30,
        "min_vram_gb": _DEFAULT_MIN_VRAM_GB,
        "max_cost_per_hour": _DEFAULT_MAX_COST_PER_H,
        "max_experiment_cost_usd": 10.0,   # hard dollar ceiling per experiment
    }
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                defaults.update(data)
        except Exception:
            pass
    return defaults


def is_runpod_enabled(workspace: Path) -> bool:
    env = str(os.environ.get("RUNPOD_ENABLED", "") or "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    # Suspended flag blocks routing even if env says enabled
    if (workspace / "tar_state" / "runpod_suspended.flag").exists():
        return False
    if env in {"1", "true", "yes", "on"}:
        return True
    return (workspace / "tar_state" / "runpod_enabled.flag").exists()


def should_use_runpod(spec: Any, workspace: Path) -> bool:
    """Return True when this spec should be offloaded to RunPod."""
    if not os.environ.get("RUNPOD_API_KEY"):
        return False
    if str(os.environ.get("RUNPOD_DRY_RUN", "") or "").strip() in {"1", "true"}:
        return False
    if not is_runpod_enabled(workspace):
        return False

    config = load_runpod_config(workspace)
    rt = getattr(spec, "runtime_context", {}) or {}
    if rt.get("force_runpod"):
        return True
    if rt.get("force_local"):
        return False

    estimated_h = float(getattr(spec, "estimated_runtime_h", 0) or 0)
    hb = getattr(spec, "hardware_budget", {}) or {}
    vram_needed = float(hb.get("vram_gb", 0) or 0)

    return (
        estimated_h > float(config["threshold_runtime_h"])
        or vram_needed > float(config["threshold_vram_gb"])
    )


# ── Executor ──────────────────────────────────────────────────────────────────

class RunPodExecutor:
    """
    Manages the full lifecycle of a single experiment running on a RunPod pod.
    Created fresh per experiment; not shared across runs.
    """

    def __init__(self, workspace: Path, orchestrator: Any):
        self.workspace = workspace
        self.orch = orchestrator
        self.config = load_runpod_config(workspace)
        self._pod_id: str = ""
        self._last_progress: dict[str, Any] = {}
        self._progress_stop = threading.Event()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        line = f"[{_ts()}] [RunPod] {msg}"
        print(line, flush=True)
        log_path = self.workspace / "tar_state" / "logs" / "runpod.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    # ── State persistence ─────────────────────────────────────────────────────

    def _save_pod_state(self, state: dict[str, Any]) -> None:
        path = self.workspace / "tar_state" / "runpod_state.json"
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass

    def _clear_pod_state(self) -> None:
        (self.workspace / "tar_state" / "runpod_state.json").unlink(missing_ok=True)

    def _save_partial(self, spec: Any, reason: str) -> None:
        if not self._last_progress:
            return
        partial_dir = self.workspace / "tar_state" / "runpod_partial"
        partial_dir.mkdir(parents=True, exist_ok=True)
        path = partial_dir / f"{spec.id}.json"
        tmp = path.with_suffix(".tmp")
        payload = {**self._last_progress, "interrupted_at": _ts(), "pod_id": self._pod_id, "reason": reason}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        seeds_done = self._last_progress.get("seeds_done", 0)
        self._log(f"Partial progress saved: {seeds_done} seed(s) complete")

    def _get_resume_seeds(self, spec: Any) -> list[int]:
        path = self.workspace / "tar_state" / "runpod_partial" / f"{spec.id}.json"
        if not path.exists():
            return list(spec.seeds)
        try:
            partial = json.loads(path.read_text(encoding="utf-8"))
            seeds_done = int(partial.get("seeds_done", 0))
            if seeds_done <= 0:
                return list(spec.seeds)
            remaining = list(spec.seeds)[seeds_done:]
            if not remaining:
                return list(spec.seeds)
            self._log(f"Resume: {seeds_done}/{len(spec.seeds)} seeds done — running {remaining}")
            return remaining
        except Exception:
            return list(spec.seeds)

    # ── SSH key management ────────────────────────────────────────────────────

    def _setup_ssh_key(self) -> tuple[Path, str]:
        """Return (private_key_path, public_key_str), generating if needed."""
        key_dir = self.workspace / "tar_state" / "runpod_ssh"
        key_dir.mkdir(parents=True, exist_ok=True)
        priv = key_dir / "id_ed25519"
        pub  = key_dir / "id_ed25519.pub"

        if not priv.exists():
            self._log("Generating TAR-dedicated SSH key pair for RunPod...")
            import subprocess as _sp
            _sp.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(priv), "-N", "", "-C", "tar-runpod"],
                check=True, capture_output=True,
            )
            self._log("SSH key pair generated and stored in tar_state/runpod_ssh/")

        return priv, pub.read_text(encoding="utf-8").strip()

    # ── Pod lifecycle ─────────────────────────────────────────────────────────

    def _select_gpu_candidates(self, rp: Any) -> list[tuple[str, float]]:
        """
        Query RunPod for available GPUs, filter by budget constraints, return
        sorted list of (gpu_type_id, price_per_hour) cheapest first.

        Hard constraints (from config):
          min_vram_gb      — must have at least this much VRAM (default 24 GB)
          max_cost_per_hour — must cost no more than this per hour (default $2.00)

        Soft preference: gpu_preference list in config is tried first, then any
        other qualifying GPU is appended as further fallback.
        """
        min_vram  = float(self.config.get("min_vram_gb",    _DEFAULT_MIN_VRAM_GB))
        max_price = float(self.config.get("max_cost_per_hour", _DEFAULT_MAX_COST_PER_H))
        cloud_type = str(self.config.get("cloud_type", "COMMUNITY")).upper()
        prefs      = [str(g).strip() for g in self.config.get("gpu_preference", _DEFAULT_GPU_PREFERENCE) if g]

        self._log(f"Budget: min {min_vram:.0f}GB VRAM, max ${max_price:.2f}/hr")

        try:
            gpus = rp.get_gpus()
        except Exception as exc:
            self._log(f"Could not fetch GPU list: {exc} — falling back to preference list")
            return [(g, 0.0) for g in prefs]

        qualifying: list[tuple[str, float]] = []
        for g in gpus:
            gpu_id   = str(g.get("id", g.get("displayName", "")))
            vram     = float(g.get("memoryInGb", 0) or 0)
            # Community or secure cloud availability
            avail    = g.get("communityCloud") if "COMMUNITY" in cloud_type else g.get("secureCloud")
            if not avail:
                continue
            if vram < min_vram:
                continue

            # Extract price — RunPod returns lowestPrice.uninterruptablePrice for on-demand
            prices   = g.get("lowestPrice") or {}
            price_od = float(prices.get("uninterruptablePrice") or prices.get("minimumBidPrice") or 999.0)
            if price_od > max_price:
                self._log(f"  Skip {gpu_id} ({vram:.0f}GB) — ${price_od:.2f}/hr exceeds ${max_price:.2f} budget")
                continue

            qualifying.append((gpu_id, price_od))
            self._log(f"  Candidate: {gpu_id} ({vram:.0f}GB VRAM, ${price_od:.2f}/hr)")

        if not qualifying:
            self._log("No GPUs found matching budget — will try preference list directly")
            return [(g, 0.0) for g in prefs]

        # Sort: preference list order first, then by price ascending
        pref_index = {g: i for i, g in enumerate(prefs)}
        qualifying.sort(key=lambda x: (pref_index.get(x[0], len(prefs)), x[1]))

        self._log(f"GPU selection order: {[f'{g}(${p:.2f})' for g,p in qualifying[:5]]}")
        return qualifying

    def _create_pod(self, spec: Any) -> tuple[str, str, float]:
        """
        Select cheapest eligible GPU (min 24GB VRAM, max $2/hr), try in order.
        Returns (pod_id, gpu_type, price_per_hour). Raises RunPodNoGPUError if all fail.
        """
        import runpod as rp
        rp.api_key = os.environ["RUNPOD_API_KEY"]

        _, pub_key = self._setup_ssh_key()
        cloud_type = str(self.config.get("cloud_type", "COMMUNITY"))
        image      = str(self.config.get("image", _DEFAULT_IMAGE))
        volume_id  = str(self.config.get("volume_id", "") or "")

        candidates = self._select_gpu_candidates(rp)
        last_error: Exception | None = None

        for gpu_type, price in candidates:
            for attempt in range(2):
                try:
                    price_str = f"${price:.2f}/hr" if price > 0 else "price unknown"
                    self._log(f"Trying: {gpu_type} ({price_str}) attempt {attempt + 1}")
                    kwargs: dict[str, Any] = dict(
                        name=f"tar-{spec.id[:8]}",
                        image_name=image,
                        gpu_type_id=gpu_type,
                        cloud_type=cloud_type,
                        gpu_count=1,
                        container_disk_in_gb=int(self.config.get("container_disk_in_gb", 30)),
                        min_vcpu_count=int(self.config.get("min_vcpu_count", 4)),
                        min_memory_in_gb=int(self.config.get("min_memory_in_gb", 16)),
                        ports="22/tcp",
                        support_public_ip=True,
                        start_ssh=True,
                        env={"PUBLIC_KEY": pub_key},
                    )
                    if volume_id:
                        kwargs["network_volume_id"] = volume_id
                        kwargs["volume_mount_path"] = "/runpod-volume"
                    pod    = rp.create_pod(**kwargs)
                    pod_id = str(pod["id"])
                    self._log(f"Pod {pod_id} created: {gpu_type} @ {price_str}")
                    return pod_id, gpu_type, price
                except Exception as exc:
                    err_lo = str(exc).lower()
                    if any(k in err_lo for k in _CREDIT_ERROR_KEYWORDS):
                        raise RunPodCreditError(str(exc)) from exc
                    if any(k in err_lo for k in ("no longer available", "out of capacity", "no instances", "unavailable", "not available")):
                        self._log(f"  {gpu_type} unavailable: {exc}")
                        last_error = exc
                        if attempt == 0:
                            time.sleep(_POD_CREATE_RETRY_WAIT_S)
                        break
                    raise

        raise RunPodNoGPUError(
            f"No GPU available within budget (min {self.config.get('min_vram_gb',24)}GB VRAM, "
            f"max ${self.config.get('max_cost_per_hour',2.0):.2f}/hr). Last: {last_error}"
        )


    def _wait_for_ssh(self, pod_id: str) -> dict[str, Any]:
        """Poll until pod SSH port is ready. Return ssh_info dict."""
        import runpod as rp
        rp.api_key = os.environ["RUNPOD_API_KEY"]
        self._log(f"Waiting for pod {pod_id} SSH readiness (timeout={_SSH_READY_TIMEOUT}s)...")
        deadline = time.time() + _SSH_READY_TIMEOUT
        while time.time() < deadline:
            try:
                info = rp.get_pod(pod_id)
                desired = info.get("desiredStatus", "")
                # Detect billing termination
                if desired == "TERMINATED":
                    raise RunPodCreditError(f"Pod {pod_id} was terminated (possibly credit exhaustion)")
                runtime = info.get("runtime")
                if desired == "RUNNING" and runtime:
                    ports = runtime.get("ports") or []
                    ssh_p = next((p for p in ports if p.get("privatePort") == 22), None)
                    if ssh_p and ssh_p.get("publicPort"):
                        host = str(ssh_p["ip"])
                        port = int(ssh_p["publicPort"])
                        self._log(f"SSH ready: {host}:{port}")
                        return {"host": host, "port": port, "pod_id": pod_id}
            except (RunPodCreditError, RunPodNoGPUError):
                raise
            except Exception as exc:
                self._log(f"Poll error: {exc}")
            time.sleep(8)
        raise TimeoutError(f"Pod {pod_id} not SSH-ready after {_SSH_READY_TIMEOUT}s")

    def _get_ssh_client(self, ssh_info: dict[str, Any]):
        """Return connected paramiko SSH client."""
        import paramiko
        priv_path, _ = self._setup_ssh_key()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ssh_info["host"],
            port=ssh_info["port"],
            username="root",
            key_filename=str(priv_path),
            timeout=30,
            banner_timeout=60,
            auth_timeout=30,
        )
        return client

    def _ssh_exec(self, client: Any, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
        """Run command via SSH. Returns (exit_code, stdout, stderr)."""
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        return code, out, err

    def _terminate(self, pod_id: str) -> None:
        if not pod_id:
            return
        try:
            import runpod as rp
            rp.api_key = os.environ.get("RUNPOD_API_KEY", "")
            rp.terminate_pod(pod_id)
            self._log(f"Pod {pod_id} terminated")
        except Exception as exc:
            self._log(f"Terminate error for {pod_id}: {exc}")

    def _cost_watchdog(
        self,
        pod_id: str,
        max_h: float,
        price_per_hour: float = 0.0,
        max_cost_usd: float = 0.0,
    ) -> None:
        """
        Daemon thread with two tiers:

        WARNING tier  — logs and writes runpod_cost_warning.flag when
                         spent >= max_cost_usd / 3  (the configured warning threshold).
                         Dashboard turns orange. User decides whether to kill.
                         Does NOT terminate the pod — legitimate slow runs continue.

        HARD KILL tier — terminates only when spent >= max_cost_usd (which is set
                         to 3× the user's warning threshold in run(), so this fires
                         at $30 when the user set $10 as their warning level).
                         Also fires if elapsed time exceeds max_h (genuine stuck process).

        Checks every 60s. Maximum billing overshoot per check: 1 minute.
        """
        warn_threshold = max_cost_usd / 3.0 if max_cost_usd > 0 else 0.0
        warned = False

        def _watch() -> None:
            nonlocal warned
            start = time.time()
            while True:
                time.sleep(60.0)
                elapsed_h = (time.time() - start) / 3600.0
                spent     = elapsed_h * price_per_hour if price_per_hour > 0 else 0.0

                # Warning tier — dashboard signal, no kill
                if not warned and warn_threshold > 0 and spent >= warn_threshold:
                    warned = True
                    self._log(
                        f"COST WARNING: ${spent:.2f} spent "
                        f"(${price_per_hour:.2f}/hr × {elapsed_h:.1f}h). "
                        f"Threshold ${warn_threshold:.2f} reached. "
                        f"Use dashboard Kill Pod Now if you want to stop early."
                    )
                    # Write warning flag for dashboard to display
                    warn_path = self.workspace / "tar_state" / "runpod_cost_warning.json"
                    try:
                        warn_path.write_text(json.dumps({
                            "pod_id":       pod_id,
                            "spent_usd":    round(spent, 2),
                            "price_per_h":  price_per_hour,
                            "elapsed_h":    round(elapsed_h, 2),
                            "written_at":   _ts(),
                        }, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                # Hard kill tier — only genuine runaways
                if max_cost_usd > 0 and spent >= max_cost_usd:
                    self._log(
                        f"WATCHDOG HARD KILL: ${spent:.2f} >= hard ceiling ${max_cost_usd:.2f} "
                        f"— terminating {pod_id}"
                    )
                    self._terminate(pod_id)
                    return

                if elapsed_h >= max_h:
                    self._log(
                        f"WATCHDOG HARD KILL: {elapsed_h:.1f}h >= time ceiling {max_h:.1f}h "
                        f"— terminating {pod_id}"
                    )
                    self._terminate(pod_id)
                    return

        t = threading.Thread(target=_watch, daemon=True, name=f"runpod-watchdog-{pod_id}")
        t.start()

    # ── Code & dataset sync ───────────────────────────────────────────────────

    def _sync_code(self, client: Any, spec: Any) -> None:
        """Upload TAR codebase to pod /workspace/repo via paramiko SFTP."""
        import paramiko
        self._log("Syncing TAR codebase to pod...")
        sftp = client.open_sftp()

        repo_root = Path(__file__).resolve().parent
        remote_root = "/workspace/repo"

        _EXCLUDE_DIRS = {
            "tar_runs", "tar_state", "tool_envs", ".git", "__pycache__",
            ".pytest_cache", ".venv", "dataset_artifacts", "training_artifacts",
            "hf", "torch", "tmp", "paper", "coding_ai_out", "anchors",
            "eval_artifacts", "legacy_quarantine", "docs", "literature",
            ".tmp_author_llm_test", "hf", "manifests",
        }
        _ALLOWED_EXT = {".py", ".txt", ".yaml", ".yml", ".cfg", ".toml"}
        _ALWAYS_INCLUDE = {"requirements.txt", "bootstrap.py", "requirements_gpu.txt"}

        # Ensure base dirs exist
        for d in (remote_root, f"{remote_root}/tar_lab", f"{remote_root}/configs",
                  f"{remote_root}/tar_state/experiments", f"{remote_root}/scripts"):
            self._ssh_exec(client, f"mkdir -p {d}", timeout=15)

        created_dirs: set[str] = set()
        uploaded = 0
        for src in repo_root.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(repo_root)
            parts = set(rel.parts[:-1])  # directories only
            if parts & _EXCLUDE_DIRS or rel.parts[0] in _EXCLUDE_DIRS:
                continue
            if src.suffix not in _ALLOWED_EXT and src.name not in _ALWAYS_INCLUDE:
                continue

            remote_path = f"{remote_root}/{rel.as_posix()}"
            remote_dir  = remote_path.rsplit("/", 1)[0]
            if remote_dir not in created_dirs:
                self._ssh_exec(client, f"mkdir -p {remote_dir}", timeout=10)
                created_dirs.add(remote_dir)
            try:
                sftp.put(str(src), remote_path)
                uploaded += 1
            except Exception as exc:
                self._log(f"  Warning: skipped {rel}: {exc}")

        sftp.close()
        self._log(f"Code sync: {uploaded} files uploaded to {remote_root}")

    def _install_deps(self, client: Any) -> None:
        """pip install requirements on pod."""
        self._log("Installing dependencies on pod...")
        cmd = (
            "cd /workspace/repo && "
            "pip install -r requirements.txt -q --no-warn-script-location 2>&1 | tail -8"
        )
        code, out, err = self._ssh_exec(client, cmd, timeout=_DEP_INSTALL_TIMEOUT)
        if code != 0:
            raise RuntimeError(f"pip install failed (code={code}): {(out + err)[-500:]}")
        self._log("Dependencies installed")

    def _prepare_dataset(self, client: Any, spec: Any) -> None:
        """Download dataset on pod if not already on the mounted network volume."""
        dataset = str(getattr(spec, "dataset", "") or "")
        self._log(f"Preparing dataset: {dataset}")

        # Check volume mount
        code, out, _ = self._ssh_exec(client, "ls /runpod-volume/datasets/ 2>/dev/null && echo VOL_OK || echo NO_VOL", timeout=10)
        if "VOL_OK" in out:
            code2, out2, _ = self._ssh_exec(client, f"ls /runpod-volume/datasets/{dataset}/ 2>/dev/null && echo FOUND || echo NOTFOUND", timeout=10)
            if "FOUND" in out2:
                self._log(f"Dataset {dataset} found on network volume — skipping download")
                return

        _dl: dict[str, str] = {
            "split_tinyimagenet":  "python -c \"from datasets import load_dataset; load_dataset('zh-plus/tiny-imagenet', cache_dir='/workspace/hf_cache', trust_remote_code=True)\"",
            "split_cifar100":      "python -c \"from datasets import load_dataset; load_dataset('uoft-cs/cifar100', cache_dir='/workspace/hf_cache')\"",
            "split_cifar10":       "python -c \"import torchvision; torchvision.datasets.CIFAR10('/workspace/data', download=True)\"",
            "permuted_mnist":      "python -c \"import torchvision; torchvision.datasets.MNIST('/workspace/data', download=True)\"",
        }
        dl_cmd = _dl.get(dataset, f"echo 'No pre-download needed for {dataset}'")
        full_cmd = (
            "cd /workspace/repo && "
            "pip install datasets -q 2>/dev/null; "
            f"HF_HOME=/workspace/hf_cache {dl_cmd}"
        )
        self._log(f"Downloading {dataset} on pod...")
        code, out, err = self._ssh_exec(client, full_cmd, timeout=_DATASET_DOWNLOAD_TIMEOUT)
        if code != 0:
            self._log(f"Dataset download warning (code={code}): {(err or out)[:300]}")
        else:
            self._log(f"Dataset {dataset} ready")

    # ── Worker execution ──────────────────────────────────────────────────────

    def _run_worker(self, client: Any, spec: Any, seeds: list[int], *, volume_path: str = "") -> int:
        """SSH-exec tar_runpod_worker.py on pod, stream stdout. Returns exit code."""
        seeds_arg      = " ".join(str(s) for s in seeds)
        config_json    = json.dumps(getattr(spec, "config_overrides", {}) or {})
        config_escaped = config_json.replace("'", "'\\''")

        volume_arg = f"--volume-path '{volume_path}'" if volume_path else ""

        cmd = (
            "cd /workspace/repo && "
            "PYTHONPATH=/workspace/repo "
            "HF_HOME=/workspace/hf_cache "
            "TORCH_HOME=/workspace/torch_cache "
            f"python tar_runpod_worker.py "
            f"--experiment-id {spec.id} "
            f"--dataset {getattr(spec, 'dataset', '')} "
            f"--method {getattr(spec, 'method', 'tcl')} "
            f"--seeds {seeds_arg} "
            f"--epochs {getattr(spec, 'epochs', 40)} "
            f"--backbone {getattr(spec, 'backbone', 'resnet18')} "
            f"--config-overrides '{config_escaped}' "
            f"--workspace /workspace "
            f"--progress-file /workspace/progress_{spec.id}.json "
            f"{volume_arg} "
            f"2>&1"
        )
        self._log(f"Launching worker on pod (seeds={seeds})")

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
        return code

    # ── Progress polling ──────────────────────────────────────────────────────

    def _start_progress_polling(self, client: Any, spec: Any) -> threading.Thread:
        """Background thread: rsync progress JSON from pod and call update_progress()."""
        self._progress_stop.clear()

        def _poll() -> None:
            import paramiko
            sftp: Any = None
            try:
                sftp = client.open_sftp()
                while not self._progress_stop.wait(timeout=_PROGRESS_POLL_S):
                    try:
                        with sftp.open(f"/workspace/progress_{spec.id}.json", "r") as fh:
                            data = json.loads(fh.read().decode("utf-8"))
                        if data != self._last_progress:
                            self._last_progress = data
                            try:
                                self.orch.update_progress(spec.id, data)
                            except Exception:
                                pass
                            self._log(
                                f"Progress: {data.get('seeds_done', 0)}/{data.get('seeds_total', '?')} seeds, "
                                f"{data.get('tasks_done', 0)} tasks"
                            )
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        self._log(f"Progress poll warn: {exc}")
            except Exception as exc:
                self._log(f"Progress thread error: {exc}")
            finally:
                if sftp:
                    try:
                        sftp.close()
                    except Exception:
                        pass

        t = threading.Thread(target=_poll, daemon=True, name=f"runpod-progress-{spec.id}")
        t.start()
        return t

    # ── Result sync ───────────────────────────────────────────────────────────

    def _sync_result(self, client: Any, spec: Any) -> dict[str, Any] | None:
        """Download result JSON from pod to local workspace. Return parsed dict."""
        remote_path = f"/workspace/result_{spec.id}.json"
        local_path  = self.workspace / "tar_state" / "experiments" / spec.id / "result.json"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            sftp = client.open_sftp()
            sftp.get(remote_path, str(local_path))
            sftp.close()
            data = json.loads(local_path.read_text(encoding="utf-8"))
            self._log(f"Result synced: verdict={data.get('verdict', '?')} mean_forgetting={data.get('mean_forgetting', '?')}")
            return data
        except FileNotFoundError:
            self._log("Result file not found on pod after worker completion")
            return None
        except Exception as exc:
            self._log(f"Result sync error: {exc}")
            return None

    # ── Auto-suspend on credit exhaustion ─────────────────────────────────────

    def _handle_credit_exhaustion(self, spec: Any) -> None:
        self._log("CREDIT EXHAUSTED — auto-suspending RunPod routing")
        suspended = self.workspace / "tar_state" / "runpod_suspended.flag"
        suspended.parent.mkdir(parents=True, exist_ok=True)
        suspended.write_text(json.dumps({
            "suspended_at": _ts(),
            "reason": "credit_exhaustion",
            "interrupted_experiment": getattr(spec, "id", ""),
            "seeds_done": self._last_progress.get("seeds_done", 0),
        }, indent=2), encoding="utf-8")
        (self.workspace / "tar_state" / "runpod_enabled.flag").unlink(missing_ok=True)
        self._log("RunPod suspended. Top up credit and run: python tar_runpod_control.py enable")

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, spec: Any) -> dict[str, Any] | None:
        """
        Execute spec on RunPod. Returns raw result dict on success, None to
        trigger local fallback. Raises RunPodInterruptedError on retriable failure.

        Storage flow:
          1. Ensure a persistent network volume exists for results
          2. Mount volume on pod at /runpod-volume/
          3. Worker writes result to both /workspace/ (fast) and /runpod-volume/ (durable)
          4. Executor pulls result via SFTP while pod is alive
          5. If pod dies before SFTP pull: spin retrieval pod, mount same volume, pull result
        """
        from tar_runpod_storage import (
            ensure_results_volume,
            setup_run_folder,
            mark_run_complete,
            retrieve_results,
            volume_exp_path,
        )

        dry_run = str(os.environ.get("RUNPOD_DRY_RUN", "") or "").strip() in {"1", "true"}
        if dry_run:
            self._log(
                f"DRY RUN: would create pod for {spec.id} "
                f"({getattr(spec, 'dataset', '?')}, {getattr(spec, 'estimated_runtime_h', 0):.1f}h)"
            )
            return None

        api_key    = os.environ["RUNPOD_API_KEY"]
        seeds          = self._get_resume_seeds(spec)
        estimated_h    = float(getattr(spec, "estimated_runtime_h", 8.0) or 8.0)
        max_h          = estimated_h * float(self.config.get("watchdog_multiplier", _WATCHDOG_MULT))
        max_cost_usd   = float(self.config.get("max_experiment_cost_usd", 10.0))
        pod_id         = ""
        price_per_hour = 0.0
        client: Any    = None

        # ── Step 1: Ensure persistent results volume ────────────────────────
        volume_id = ensure_results_volume(
            self.workspace, self.config, api_key, log_fn=self._log
        )
        # Refresh config so _create_pod picks up the volume_id
        from tar_runpod_executor import load_runpod_config
        self.config = load_runpod_config(self.workspace)

        try:
            # ── Step 2: Create training pod ─────────────────────────────────
            try:
                pod_id, gpu_type, price_per_hour = self._create_pod(spec)
            except RunPodCreditError as exc:
                self._handle_credit_exhaustion(spec)
                raise RunPodInterruptedError(f"Credit exhausted: {exc}") from exc
            except RunPodNoGPUError as exc:
                self._log(f"No GPU available — falling back to local: {exc}")
                rt = dict(getattr(spec, "runtime_context", {}) or {})
                rt["runpod_fallback"] = "no_gpu_available"
                spec.runtime_context = rt
                return None

            self._pod_id = pod_id

            # ── Pre-flight cost estimate ─────────────────────────────────────
            if price_per_hour > 0:
                est_cost = estimated_h * price_per_hour
                self._log(
                    f"Cost estimate: ~${est_cost:.2f} "
                    f"({estimated_h:.1f}h × ${price_per_hour:.2f}/hr on {gpu_type}). "
                    f"Dollar warning threshold: ${max_cost_usd:.2f}. "
                    f"Hard kill at: ${max_cost_usd * 3:.2f} (3× warning threshold)."
                )

            # ── Step 3: Set up per-run storage folder ───────────────────────
            manifest = setup_run_folder(
                self.workspace, spec, volume_id, gpu_type, seeds, log_fn=self._log
            )
            vol_exp_path = volume_exp_path(spec.id)

            self._save_pod_state({
                "active_pod_id":  pod_id,
                "gpu_type":       gpu_type,
                "experiment_id":  getattr(spec, "id", ""),
                "pod_created_at": _ts(),
                "seeds_todo":     seeds,
                "seeds_total":    len(getattr(spec, "seeds", [])),
                "estimated_h":    getattr(spec, "estimated_runtime_h", 0),
                "watchdog_kills_at_h": max_h,
                "volume_id":      volume_id,
                "vol_exp_path":   vol_exp_path,
            })

            # Start watchdog — HARD kill only at 3× the dollar warning threshold
            # (a genuine runaway, not just a slow legitimate run).
            # Normal overspend shows a dashboard warning; user kills manually.
            self._cost_watchdog(
                pod_id,
                max_h=max_h,
                price_per_hour=price_per_hour,
                max_cost_usd=max_cost_usd * 3.0,  # hard kill at 3× warning threshold
            )

            # ── Step 4: SSH + environment setup ─────────────────────────────
            ssh_info = self._wait_for_ssh(pod_id)
            client   = self._get_ssh_client(ssh_info)

            # Write manifest to volume on pod
            if volume_id:
                import json as _json
                self._ssh_exec(client, f"mkdir -p {vol_exp_path}", timeout=15)
                manifest_json = _json.dumps(manifest, indent=2).replace("'", "'\\''")
                self._ssh_exec(
                    client,
                    f"echo '{manifest_json}' > {vol_exp_path}/manifest.json",
                    timeout=10,
                )

            self._sync_code(client, spec)
            self._install_deps(client)
            self._prepare_dataset(client, spec)

            # ── Step 5: Run experiment ───────────────────────────────────────
            progress_thread = self._start_progress_polling(client, spec)
            exit_code = self._run_worker(client, spec, seeds, volume_path=vol_exp_path if volume_id else "")
            self._progress_stop.set()
            progress_thread.join(timeout=5)

            if exit_code != 0:
                raise RunPodInterruptedError(f"Worker exited with code {exit_code}")

            # ── Step 6: Pull result (SFTP primary, volume fallback) ──────────
            result_data = self._sync_result(client, spec)

            if result_data is None and volume_id:
                self._log("SFTP pull failed — trying volume retrieval pod...")
                priv_path, pub_key = self._setup_ssh_key()
                result_data = retrieve_results(
                    self.workspace, spec, volume_id, self.config, api_key,
                    pub_key, priv_path, log_fn=self._log,
                )

            if result_data is None:
                raise RunPodInterruptedError("No result.json found after worker completed")

            spec.result_path = str(
                self.workspace / "tar_state" / "experiments" / spec.id / "result.json"
            )
            mark_run_complete(self.workspace, spec.id, result_data.get("verdict", "?"))
            (self.workspace / "tar_state" / "runpod_partial" / f"{spec.id}.json").unlink(missing_ok=True)
            return result_data

        except RunPodCreditError as exc:
            self._handle_credit_exhaustion(spec)
            self._save_partial(spec, str(exc))
            # Try volume retrieval even after credit exhaustion — results may already be written
            if volume_id:
                self._log("Attempting volume retrieval after credit exhaustion...")
                try:
                    priv_path, pub_key = self._setup_ssh_key()
                    result_data = retrieve_results(
                        self.workspace, spec, volume_id, self.config, api_key,
                        pub_key, priv_path, log_fn=self._log,
                    )
                    if result_data:
                        self._log("Volume retrieval succeeded despite credit exhaustion")
                        return result_data
                except Exception as re:
                    self._log(f"Volume retrieval also failed: {re}")
            raise RunPodInterruptedError(str(exc)) from exc

        except (BrokenPipeError, ConnectionResetError, EOFError, TimeoutError) as exc:
            self._save_partial(spec, str(exc))
            if volume_id and pod_id:
                self._log("Connection lost — trying volume retrieval...")
                try:
                    priv_path, pub_key = self._setup_ssh_key()
                    result_data = retrieve_results(
                        self.workspace, spec, volume_id, self.config, api_key,
                        pub_key, priv_path, log_fn=self._log,
                    )
                    if result_data:
                        return result_data
                except Exception:
                    pass
            raise RunPodInterruptedError(f"Connection lost: {exc}") from exc

        except RunPodInterruptedError:
            self._save_partial(spec, "worker_error")
            raise

        except Exception as exc:
            err_lo = str(exc).lower()
            if any(k in err_lo for k in _CREDIT_ERROR_KEYWORDS):
                self._handle_credit_exhaustion(spec)
            self._save_partial(spec, str(exc))
            raise

        finally:
            self._progress_stop.set()
            if client:
                try:
                    client.close()
                except Exception:
                    pass
            if pod_id:
                self._terminate(pod_id)
            self._clear_pod_state()
