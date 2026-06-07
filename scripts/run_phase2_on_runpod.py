"""scripts/run_phase2_on_runpod.py <runner_key> [--dry-run]

Dispatch a Phase-2 confirmatory run (the standalone scripts the ramp gate waits on)
to a RunPod GPU pod, so the runs can go in PARALLEL instead of serially on the local
4GB card. Generalises the SI bridge (run_si_cycle_on_runpod.py).

Per-runner config (RUNNERS) names the script, the hardcoded module paths to patch on
the Linux pod (the scripts hardcode E:\\... paths), the local inputs to sync, the pod
flags to create, and the output comparison-JSON glob to retrieve.

SAFETY: pod ALWAYS terminated in finally; cost watchdog inherited from RunPodExecutor;
datacenter-first GPU fall-through; --dry-run does pre-flight + plan only (zero cost).
Registration of the retrieved result as a terminal queue entry (so the ramp counts it)
is a SEPARATE, lock-aware step (register_phase2_result), not run in the parallel path.
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
from typing import Any, Optional

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_REMOTE_REPO = "/workspace/repo"
_POD_STATE = "/workspace/state"          # patched tar_state root on the pod
_POD_OUT = "/workspace/out/comparisons"  # patched OUTPUT_DIR on the pod
_MIN_VRAM_GB = 16
_EST_H = 4.0                              # conservative; watchdog ceiling = est * multiplier

# Per-runner config. patch_paths: module attr -> pod path. sync_inputs: local tar_state-relative
# files to upload (placed under _POD_STATE). create_flags: pod paths to `touch`.
RUNNERS: dict[str, dict[str, Any]] = {
    "phase16_cifar100_rerun": {
        "script": "phase16_cifar100_rerun.py",
        "module": "phase16_cifar100_rerun",
        "patch_paths": {
            "OUTPUT_DIR": _POD_OUT,
            "PREREG_FILE": f"{_POD_STATE}/preregistrations/phase16_rerun.json",
            "EXEC_FLAG": f"{_POD_STATE}/execution_enabled.flag",
        },
        "sync_inputs": ["preregistrations/phase16_rerun.json"],
        "create_flags": [f"{_POD_STATE}/execution_enabled.flag"],
        "output_dir": _POD_OUT,
        "output_glob": "phase16_cifar100_rerun_*.json",
        "est_h": 5.0,
    },
    "mechanistic_ablation_7c": {
        "script": "run_mechanistic_ablation.py",
        "module": "run_mechanistic_ablation",
        "patch_paths": {
            "_TAR_STATE": _POD_STATE,
            "PREREG_FILE": f"{_POD_STATE}/preregistrations/mechanistic_ablation_7condition.json",
            "EXEC_FLAG": f"{_POD_STATE}/execution_enabled.flag",
            "CHECKPOINT_FILE": f"{_POD_STATE}/comparisons/mechanistic_ablation_checkpoint.json",
            "PHASE11_FILE": f"{_POD_STATE}/comparisons/phase11_ablation__20260511T113318Z.json",
        },
        "sync_inputs": [
            "preregistrations/mechanistic_ablation_7condition.json",
            "comparisons/phase11_ablation__20260511T113318Z.json",
        ],
        "create_flags": [f"{_POD_STATE}/execution_enabled.flag"],
        "args": ["--conditions", "anchor_frozen_init,warmup_batches_60,ewc_best_lambda"],
        "output_dir": f"{_POD_STATE}/comparisons",
        "output_glob": "mechanistic_ablation_*.json",
        "output_exclude": "checkpoint",
        "est_h": 4.0,
    },
    "hpc_replication_phase2": {
        "script": "run_hpc_replication.py",
        "module": "run_hpc_replication",
        "patch_paths": {
            "_TAR_STATE": _POD_STATE,
            "PREREG_FILE": f"{_POD_STATE}/preregistrations/hpc_replication.json",
            "EXEC_FLAG": f"{_POD_STATE}/execution_enabled.flag",
            "CHECKPOINT_FILE": f"{_POD_STATE}/comparisons/hpc_replication_checkpoint.json",
        },
        # FRESH run (no checkpoint synced) -> all 20 seeds on one GPU with the FIXED
        # SPRT (f58eed3): one code version, one hardware, fully reproducible. The
        # bridge syncs the working copy, so the pod gets the fixed stat_utils.py.
        "sync_inputs": ["preregistrations/hpc_replication.json"],
        "create_flags": [f"{_POD_STATE}/execution_enabled.flag"],
        "output_dir": f"{_POD_STATE}/comparisons",
        "output_glob": "hpc_replication_*.json",
        "output_exclude": "checkpoint",
        "est_h": 6.0,
    },
    # phase17 added after these prove (needs TinyImageNet data staging).
}

_PIP = ("pip install -q --no-warn-script-location "
        "torchvision numpy scipy scikit-learn pandas 2>&1 | tail -8")


def _resolve_workspace() -> Path:
    env = os.environ.get("TAR_WORKSPACE", "").strip()
    if env:
        return Path(env).resolve()
    try:
        from tar_storage import resolve_workspace
        return resolve_workspace(_REPO)
    except Exception:
        c = Path("E:/TAR/Thermodynamic-Continual-Learning-delivered")
        return c if (c / "tar_state").exists() else _REPO


def _load_api_key(ws: Path) -> str:
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if k:
        return k
    p = ws / "tar_state" / "api_secrets.json"
    if p.exists():
        try:
            k = str(json.loads(p.read_text(encoding="utf-8")).get("RUNPOD_API_KEY", "") or "").strip()
        except Exception:
            k = ""
    if k:
        os.environ["RUNPOD_API_KEY"] = k
    return k


def _preflight(ws: Path, cfg: dict, api_key: str, *, dry_run: bool) -> None:
    errs: list[str] = []
    if not api_key:
        errs.append("RUNPOD_API_KEY not set / not in tar_state/api_secrets.json")
    if not (_REPO / cfg["script"]).exists():
        errs.append(f"script not found: {cfg['script']}")
    for rel in cfg.get("sync_inputs", []):
        if not (ws / "tar_state" / rel).exists():
            errs.append(f"required input missing: tar_state/{rel}")
    if errs:
        print("[PREFLIGHT FAILED]")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"[preflight] OK ({cfg['script']}). {'DRY RUN.' if dry_run else 'Ready to launch pod.'}")


def _ssh(client: Any, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    _i, o, e = client.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return o.channel.recv_exit_status(), out, err


def _make_shim(cfg: dict) -> str:
    patch = "\n".join(
        f'_m.{attr} = pathlib.Path({path!r}); '
        f'(_m.{attr}.parent if _m.{attr}.suffix else _m.{attr}).mkdir(parents=True, exist_ok=True)'
        for attr, path in cfg["patch_paths"].items()
    )
    flags = "\n".join(
        f'pathlib.Path({f!r}).parent.mkdir(parents=True, exist_ok=True); pathlib.Path({f!r}).write_text("1")'
        for f in cfg.get("create_flags", [])
    )
    args = cfg.get("args", [])
    argv = f"sys.argv = [{cfg['script']!r}] + {list(args)!r}\n" if args else ""
    return (
        "#!/usr/bin/env python3\n"
        "import sys, pathlib, importlib\n"
        f"sys.path.insert(0, {_REMOTE_REPO!r})\n"
        f"{flags}\n"
        f"_m = importlib.import_module({cfg['module']!r})\n"
        f"{patch}\n"
        f"{argv}"
        "print('[shim] patched paths + flags; launching main()', flush=True)\n"
        "_m.main()\n"
    )


def _sync_inputs(client: Any, ws: Path, cfg: dict) -> None:
    sftp = client.open_sftp()
    made: set[str] = set()
    for rel in cfg.get("sync_inputs", []):
        local = ws / "tar_state" / rel
        remote = f"{_POD_STATE}/{rel}"
        rd = remote.rsplit("/", 1)[0]
        if rd not in made:
            _ssh(client, f"mkdir -p {rd}", 15)
            made.add(rd)
        sftp.put(str(local), remote)
    sftp.close()
    print(f"[bridge] synced {len(cfg.get('sync_inputs', []))} input(s) to {_POD_STATE}", flush=True)


def _run_remote(client: Any, cfg: dict) -> int:
    shim = _make_shim(cfg)
    sftp = client.open_sftp()
    with sftp.open(f"{_REMOTE_REPO}/_phase2_shim.py", "w") as fh:
        fh.write(shim.encode("utf-8"))
    sftp.close()
    cmd = (f"cd {_REMOTE_REPO} && PYTHONPATH={_REMOTE_REPO} "
           "HF_HOME=/workspace/hf TORCH_HOME=/workspace/torch "
           "python _phase2_shim.py 2>&1")
    print(f"[bridge] launching: {cmd}", flush=True)
    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(cmd)
    while not chan.exit_status_ready():
        if chan.recv_ready():
            for ln in chan.recv(8192).decode("utf-8", "replace").splitlines():
                if ln.strip():
                    print(f"  [pod] {ln}", flush=True)
        time.sleep(0.5)
    while chan.recv_ready():
        for ln in chan.recv(8192).decode("utf-8", "replace").splitlines():
            if ln.strip():
                print(f"  [pod] {ln}", flush=True)
    code = chan.recv_exit_status()
    chan.close()
    print(f"[bridge] remote run exit code {code}", flush=True)
    return code


def _retrieve(client: Any, ws: Path, cfg: dict) -> Optional[Path]:
    out_dir = cfg.get("output_dir", _POD_OUT)
    exclude = cfg.get("output_exclude")
    sftp = client.open_sftp()
    try:
        names = [n for n in sftp.listdir(out_dir)
                 if n.startswith(cfg["output_glob"].split("*")[0]) and n.endswith(".json")
                 and not (exclude and exclude in n)]
    except Exception:
        names = []
    if not names:
        print("[bridge] no comparison JSON produced on the pod", flush=True)
        sftp.close()
        return None
    # newest by pod mtime (alphabetical can misorder timestamp vs literal-name siblings)
    names.sort(key=lambda n: getattr(sftp.stat(f"{out_dir}/{n}"), "st_mtime", 0))
    latest = names[-1]
    local_dir = ws / "tar_state" / "comparisons"
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / latest
    sftp.get(f"{out_dir}/{latest}", str(local))
    # env sibling, if the script wrote one
    for sib in (latest.replace(".json", "_env.json"),):
        try:
            sftp.get(f"{out_dir}/{sib}", str(local_dir / sib))
        except Exception:
            pass
    sftp.close()
    print(f"[bridge] RETRIEVED -> {local}", flush=True)
    return local


def main(runner_key: str, dry_run: bool = False) -> None:
    cfg = RUNNERS.get(runner_key)
    if not cfg:
        print(f"unknown runner_key '{runner_key}'. known: {list(RUNNERS)}")
        sys.exit(1)
    ws = _resolve_workspace()
    api_key = _load_api_key(ws)
    print(f"[bridge] runner={runner_key} script={cfg['script']} workspace={ws} dry_run={dry_run}")
    _preflight(ws, cfg, api_key, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] would: pod(>= {_MIN_VRAM_GB}GB, datacenter-first) -> sync code + "
              f"{cfg.get('sync_inputs')} -> pip torchvision+stats -> shim(patch {list(cfg['patch_paths'])}) "
              f"-> run {cfg['script']} -> retrieve {cfg['output_glob']} -> terminate.")
        return

    from tar_runpod_executor import RunPodExecutor, _WATCHDOG_MULT

    class _Exec(RunPodExecutor):
        def __init__(self, w: Path, orch: Any) -> None:
            super().__init__(w, orch)
            self.config["min_vram_gb"] = max(_MIN_VRAM_GB, float(self.config.get("min_vram_gb", 24)))

    class _NullOrch:
        def update_progress(self, *a: Any, **k: Any) -> None: ...

    exr = _Exec(ws, _NullOrch())
    spec = SimpleNamespace(id=f"phase2-{runner_key}", name=f"tar-{runner_key}",
                           estimated_runtime_h=cfg.get("est_h", _EST_H),
                           hardware_budget=SimpleNamespace(vram_gb=_MIN_VRAM_GB),
                           runtime_context={}, config_overrides={}, dataset="", method="",
                           epochs=0, backbone="", seeds=[])
    est_h = float(cfg.get("est_h", _EST_H))
    max_h = est_h * float(exr.config.get("watchdog_multiplier", _WATCHDOG_MULT))
    max_cost = float(exr.config.get("max_experiment_cost_usd", 10.0))
    pod_id = ""
    client: Any = None
    try:
        gpus = list(exr.config.get("gpu_preference", []))
        errs = []
        for g in gpus:
            exr.config["gpu_preference"] = [g]
            try:
                pod_id, gtype, price = exr._create_pod(spec)
                print(f"[bridge] pod {pod_id} on {gtype}", flush=True)
                break
            except Exception as ce:
                errs.append(f"{g}: {str(ce)[:80]}")
                print(f"[bridge] {g} unavailable -> next", flush=True)
        exr.config["gpu_preference"] = gpus
        if not pod_id:
            raise RuntimeError("no GPU available. " + " | ".join(errs))
        exr._pod_id = pod_id
        exr._cost_watchdog(pod_id, max_h, price if "price" in dir() else 0.0, max_cost)
        ssh_info = exr._wait_for_ssh(pod_id)
        client = exr._get_ssh_client(ssh_info)
        exr._sync_code(client, spec)
        _sync_inputs(client, ws, cfg)
        print("[bridge] installing torchvision + stats deps ...", flush=True)
        code, out, err = _ssh(client, _PIP, timeout=600)
        if code != 0:
            print(f"[bridge] WARNING dep install rc={code}: {(out + err)[-400:]}", flush=True)
        rc = _run_remote(client, cfg)
        result = _retrieve(client, ws, cfg)
        print("=" * 60)
        print(f"[bridge] {runner_key}: exit={rc} result={'retrieved' if result else 'NONE'}")
        if result:
            print(f"  -> {result}")
            print(f"  REGISTER (lock-aware, separate step) to flip the ramp gate for this runner_key.")
        print("=" * 60)
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass
        if pod_id:
            exr._terminate(pod_id)
            exr._clear_pod_state()
            print(f"[bridge] pod {pod_id} terminated", flush=True)


def _verify_phase2_success(runner_key: str, result_path: Path) -> tuple[bool, str]:
    """Self-contained genuine-completion guard (mirrors tar_dashboard._phase2_run_succeeded).
    Reads the retrieved result JSON; hpc requires seeds_run>=20 OR an SPRT decision, so a
    crashed/truncated run is never registered 'complete' (the 2026-06-03 6/20 false-advance)."""
    if not result_path or not result_path.exists():
        return False, f"result file missing: {result_path}"
    try:
        res = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"result unreadable: {e}"
    if runner_key == "hpc_replication_phase2":
        seeds = int(res.get("seeds_run", 0) or 0)
        decided = str(res.get("sprt_final_decision", "")) in {"accept_H1", "accept_H0"}
        if not (seeds >= 20 or decided):
            return False, (f"hpc not genuinely complete (seeds_run={seeds}, "
                           f"decision={res.get('sprt_final_decision')!r})")
    if not (res.get("verdict") or res.get("result_id")
            or res.get("honest_verdict") or res.get("honest_verdict_detail")):
        return False, "result JSON lacks verdict/result_id (not a finished comparison)"
    return True, "ok"


def register_phase2_result(runner_key: str, result_path: str, *,
                           dry_run: bool = False, ws: Optional[Path] = None) -> bool:
    """Lock-aware registration of a retrieved Phase-2 result as the TERMINAL queue entry the
    autonomy ramp counts (tar_autonomy_ramp._phase2_status + evidence gate, status=='complete').
    REFUSES unless the run genuinely produced its output. Atomic .tmp+os.replace write.

    NB: the live daemon also manages experiment_queue.json; run this in a quiescent window
    (or it may clobber a concurrent daemon write — last-writer-wins)."""
    ws = ws or _resolve_workspace()
    rp = Path(result_path).resolve()
    ok, why = _verify_phase2_success(runner_key, rp)
    if not ok:
        print(f"[register] REFUSED for {runner_key}: {why}", flush=True)
        return False
    qpath = ws / "tar_state" / "experiment_queue.json"
    try:
        data = json.loads(qpath.read_text(encoding="utf-8")) if qpath.exists() else {"experiments": []}
        if not isinstance(data, dict):
            data = {"experiments": []}
    except Exception:
        data = {"experiments": []}
    exps = data.setdefault("experiments", [])
    now = datetime.now(timezone.utc).isoformat()
    action = "appended new"
    for e in exps:
        if isinstance(e, dict) and str(e.get("runner_key", "")) == runner_key:
            e.update(status="complete", stage="complete", result_path=str(rp),
                     completed_at=now, pid=0)
            action = "updated existing"
            break
    else:
        exps.append({
            "id": f"phase2-{runner_key}",
            "runner_key": runner_key,
            "status": "complete",
            "stage": "complete",
            "result_path": str(rp),
            "name": f"Phase-2 {runner_key} (RunPod bridge)",
            "completed_at": now,
            "started_at": now,
            "pid": 0,
        })
    if dry_run:
        print(f"[register] DRY RUN — would {action} TERMINAL entry: "
              f"runner_key={runner_key} status=complete result={rp.name}", flush=True)
        return True
    tmp = qpath.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, qpath)
    print(f"[register] {runner_key} -> complete ({action}; result {rp.name})", flush=True)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runner_key")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--register", metavar="RESULT_PATH",
                    help="Register a retrieved result as the terminal queue entry (skips dispatch).")
    a = ap.parse_args()
    if a.register:
        register_phase2_result(a.runner_key, a.register, dry_run=a.dry_run)
    else:
        main(a.runner_key, dry_run=a.dry_run)
