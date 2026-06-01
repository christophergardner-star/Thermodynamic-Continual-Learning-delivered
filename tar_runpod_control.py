"""
TAR RunPod Control — manual enable/disable/status CLI.

Usage:
  python tar_runpod_control.py status       # show full current state
  python tar_runpod_control.py enable       # turn on RunPod routing
  python tar_runpod_control.py disable      # turn off (experiments run locally)
  python tar_runpod_control.py pause        # disable; let any active pod finish
  python tar_runpod_control.py kill         # disable + terminate active pod NOW
  python tar_runpod_control.py check-gpus   # list available GPUs and prices
  python tar_runpod_control.py dry-run-on   # enable dry-run mode (logs, no API calls)
  python tar_runpod_control.py dry-run-off  # disable dry-run mode
  python tar_runpod_control.py setup        # first-time setup wizard
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_workspace() -> Path:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tar_storage import preferred_workspace
        return Path(preferred_workspace())
    except Exception:
        # Fallback: look for E:/TAR or C:/Users/.../TAR workspace
        for candidate in (
            Path("E:/TAR/Thermodynamic-Continual-Learning-delivered"),
            Path(__file__).resolve().parent.parent / "Thermodynamic-Continual-Learning-delivered",
        ):
            if (candidate / "tar_state").exists():
                return candidate
        return Path(__file__).resolve().parent


def _load_config(ws: Path) -> dict:
    from tar_runpod_executor import load_runpod_config
    return load_runpod_config(ws)


def _load_state(ws: Path) -> dict:
    p = ws / "tar_state" / "runpod_state.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_suspended(ws: Path) -> dict:
    p = ws / "tar_state" / "runpod_suspended.flag"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"suspended_at": "unknown"}
    return {}


def cmd_status(ws: Path) -> None:
    from tar_runpod_executor import is_runpod_enabled, load_runpod_config

    config    = load_runpod_config(ws)
    enabled   = is_runpod_enabled(ws)
    suspended = _load_suspended(ws)
    state     = _load_state(ws)
    dry_run   = str(os.environ.get("RUNPOD_DRY_RUN", "") or "").strip() in {"1", "true"}
    api_key   = os.environ.get("RUNPOD_API_KEY", "")

    print("\n── TAR RunPod Status ────────────────────────────────────────")

    # Overall state
    if dry_run:
        print("  Mode:        DRY RUN (no real API calls)")
    elif suspended:
        print(f"  Mode:        SUSPENDED — {suspended.get('reason','unknown')}")
        print(f"               Suspended at: {suspended.get('suspended_at','?')}")
        print(f"               Top up credit then: python tar_runpod_control.py enable")
    elif enabled:
        print("  Mode:        ENABLED  (will route eligible experiments to RunPod)")
    else:
        print("  Mode:        DISABLED (all experiments run locally)")

    # API key
    if api_key:
        print(f"  API Key:     set ({len(api_key)} chars, starts {api_key[:6]}...)")
    else:
        print("  API Key:     NOT SET — set RUNPOD_API_KEY environment variable")

    # Thresholds and budget
    print(f"  Threshold:   >{config['threshold_runtime_h']:.0f}h estimated runtime OR >{config['threshold_vram_gb']:.1f}GB VRAM → routes to cloud")
    print(f"  GPU budget:  min {config.get('min_vram_gb', 24):.0f}GB VRAM  |  max ${config.get('max_cost_per_hour', 2.0):.2f}/hr")
    print(f"  GPU prefs:   {', '.join(config['gpu_preference'][:4])}")

    # Active pod
    if state.get("active_pod_id"):
        pod_id   = state["active_pod_id"]
        exp_id   = state.get("experiment_id", "?")
        gpu      = state.get("gpu_type", "?")
        created  = state.get("pod_created_at", "?")
        seeds_done  = state.get("seeds_done_snapshot", "?")
        seeds_total = state.get("seeds_total", "?")
        print(f"\n  Active pod:  {pod_id}")
        print(f"  GPU:         {gpu}")
        print(f"  Experiment:  {exp_id}")
        print(f"  Started:     {created}")
        if seeds_done != "?":
            print(f"  Progress:    {seeds_done}/{seeds_total} seeds")

        # Try to get cost estimate
        try:
            import runpod as rp
            rp.api_key = api_key
            info = rp.get_pod(pod_id)
            uptime_s = (info.get("runtime") or {}).get("uptimeInSeconds", 0)
            if uptime_s:
                uptime_h = uptime_s / 3600
                # Rough estimate at $0.44/hr
                cost_est = uptime_h * 0.44
                print(f"  Uptime:      {uptime_h:.1f}h (~${cost_est:.2f} est.)")
        except Exception:
            pass
    else:
        print("\n  Active pod:  none")

    # Account balance
    if api_key:
        try:
            import runpod as rp
            rp.api_key = api_key
            user = rp.get_user()
            balance = user.get("currentSpend") or user.get("balance") or "unknown"
            print(f"  Account:     {balance}")
        except Exception:
            pass

    print("─────────────────────────────────────────────────────────────\n")


def cmd_enable(ws: Path) -> None:
    from tar_runpod_executor import load_runpod_config

    # Check API key
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY is not set. Set it first:")
        print("  $env:RUNPOD_API_KEY = 'your_key_here'")
        sys.exit(1)

    # Check account balance if possible
    try:
        import runpod as rp
        rp.api_key = os.environ["RUNPOD_API_KEY"]
        user = rp.get_user()
        balance = float(user.get("currentSpend") or user.get("balance") or 0)
        if balance <= 0:
            print(f"WARNING: RunPod account balance appears to be ${balance:.2f}")
            print("         Top up credit before enabling to avoid immediate suspension.")
    except Exception:
        pass

    # Remove suspended flag
    suspended = ws / "tar_state" / "runpod_suspended.flag"
    if suspended.exists():
        suspended.unlink()
        print("Cleared suspension flag.")

    # Create enabled flag
    flag = ws / "tar_state" / "runpod_enabled.flag"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(json.dumps({"enabled_at": _ts()}), encoding="utf-8")

    config = load_runpod_config(ws)
    print(f"RunPod routing ENABLED.")
    print(f"  Threshold: >{config['threshold_runtime_h']:.0f}h or >{config['threshold_vram_gb']:.1f}GB VRAM")
    print(f"  GPU prefs: {', '.join(config['gpu_preference'][:3])}")
    print("  TAR will route eligible experiments on the next daemon cycle (30s).")


def cmd_disable(ws: Path) -> None:
    flag = ws / "tar_state" / "runpod_enabled.flag"
    flag.unlink(missing_ok=True)
    print("RunPod routing DISABLED. Experiments will run locally.")
    print("(Any currently running pod will finish and terminate naturally.)")


def cmd_pause(ws: Path) -> None:
    """Disable routing but let the active pod finish."""
    flag = ws / "tar_state" / "runpod_enabled.flag"
    flag.unlink(missing_ok=True)
    state = _load_state(ws)
    if state.get("active_pod_id"):
        print(f"RunPod routing paused. Active pod {state['active_pod_id']} will run to completion.")
    else:
        print("RunPod routing paused. No active pod.")


def cmd_kill(ws: Path) -> None:
    """Disable routing AND immediately terminate the active pod."""
    flag = ws / "tar_state" / "runpod_enabled.flag"
    flag.unlink(missing_ok=True)

    state = _load_state(ws)
    pod_id = state.get("active_pod_id", "")
    if not pod_id:
        print("RunPod routing disabled. No active pod to kill.")
        return

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print(f"ERROR: RUNPOD_API_KEY not set. Cannot terminate pod {pod_id}.")
        print(f"       Kill it manually at: https://www.runpod.io/console/pods/{pod_id}")
        sys.exit(1)

    try:
        import runpod as rp
        rp.api_key = api_key
        rp.terminate_pod(pod_id)
        print(f"Pod {pod_id} TERMINATED.")
        (ws / "tar_state" / "runpod_state.json").unlink(missing_ok=True)
    except Exception as exc:
        print(f"ERROR terminating {pod_id}: {exc}")
        print(f"Kill manually at: https://www.runpod.io/console/pods/{pod_id}")
        sys.exit(1)


def cmd_check_gpus(ws: Path) -> None:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set.")
        sys.exit(1)

    try:
        import runpod as rp
        rp.api_key = api_key
        gpus = rp.get_gpus()
    except Exception as exc:
        print(f"ERROR fetching GPUs: {exc}")
        sys.exit(1)

    config     = _load_config(ws)
    min_vram   = float(config.get("min_vram_gb", 24))
    max_price  = float(config.get("max_cost_per_hour", 2.0))

    print(f"\n── Available RunPod GPUs (budget: min {min_vram:.0f}GB VRAM, max ${max_price:.2f}/hr) ──")
    print(f"  {'GPU':<32} {'VRAM':>6}  {'On-Demand':>10}  {'In Budget':>10}  {'Avail':>7}")
    print(f"  {'-'*32} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*7}")
    qualifying = []
    for g in sorted(gpus, key=lambda x: x.get("memoryInGb", 0)):
        name   = str(g.get("displayName", g.get("id", "?")))
        mem    = float(g.get("memoryInGb", 0) or 0)
        comm   = g.get("communityCloud", False)
        prices = g.get("lowestPrice") or {}
        price  = float(prices.get("uninterruptablePrice") or prices.get("minimumBidPrice") or 0)
        avail  = "yes" if comm else "no"
        price_str = f"${price:.2f}/hr" if price > 0 else "unknown"
        in_budget = (mem >= min_vram and price <= max_price and price > 0 and comm)
        budget_str = "✓ YES" if in_budget else "no"
        if in_budget:
            qualifying.append((name, mem, price))
        print(f"  {name:<32} {mem:>5.0f}GB  {price_str:>10}  {budget_str:>10}  {avail:>7}")
    print()
    if qualifying:
        qualifying.sort(key=lambda x: x[2])
        print(f"  Cheapest qualifying GPU: {qualifying[0][0]} ({qualifying[0][1]:.0f}GB @ ${qualifying[0][2]:.2f}/hr)")
    else:
        print(f"  WARNING: No GPUs found matching budget constraints.")
    print("─────────────────────────────────────────────────────────────\n")


def cmd_dry_run_on(ws: Path) -> None:
    flag = ws / "tar_state" / "runpod_dryrun.flag"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(json.dumps({"enabled_at": _ts()}), encoding="utf-8")
    print("Dry-run mode ON. RunPod actions will be logged but no pods created.")
    print("To activate: also set $env:RUNPOD_DRY_RUN=1 in your shell.")


def cmd_dry_run_off(ws: Path) -> None:
    (ws / "tar_state" / "runpod_dryrun.flag").unlink(missing_ok=True)
    print("Dry-run mode OFF.")


def cmd_setup(ws: Path) -> None:
    print("\n── TAR RunPod First-Time Setup ──────────────────────────────")

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("Step 1: Set your RunPod API key as an environment variable:")
        print("  $env:RUNPOD_API_KEY = 'rp_...'")
        print("  [System.Environment]::SetEnvironmentVariable('RUNPOD_API_KEY', 'rp_...', 'User')")
        print()
    else:
        print(f"Step 1: API key set ✓ ({len(api_key)} chars)")

    # SSH key
    key_dir = ws / "tar_state" / "runpod_ssh"
    priv    = key_dir / "id_ed25519"
    if priv.exists():
        print("Step 2: SSH key exists ✓")
    else:
        print("Step 2: SSH key will be auto-generated on first pod creation.")

    # Config
    config_path = ws / "tar_state" / "runpod_config.json"
    if config_path.exists():
        print("Step 3: runpod_config.json exists ✓")
    else:
        default_config = {
            "enabled": False,
            "threshold_runtime_h": 12.0,
            "threshold_vram_gb": 3.9,
            "gpu_preference": ["NVIDIA RTX 4090", "NVIDIA A40", "NVIDIA A100-SXM4-80GB"],
            "cloud_type": "COMMUNITY",
            "image": "runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04",
            "volume_id": "",
            "datacenter_id": "",
            "watchdog_multiplier": 2.5,
            "container_disk_in_gb": 30,
            "min_vcpu_count": 4,
            "min_memory_in_gb": 16,
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(default_config, indent=2), encoding="utf-8")
        print("Step 3: runpod_config.json created ✓")

    print()
    print("When ready:")
    print("  python tar_runpod_control.py enable          # turn on routing")
    print("  python tar_runpod_control.py check-gpus      # see available GPUs")
    print("  python tar_runpod_control.py status          # current state")
    print("─────────────────────────────────────────────────────────────\n")


COMMANDS = {
    "status":      cmd_status,
    "enable":      cmd_enable,
    "disable":     cmd_disable,
    "pause":       cmd_pause,
    "kill":        cmd_kill,
    "check-gpus":  cmd_check_gpus,
    "dry-run-on":  cmd_dry_run_on,
    "dry-run-off": cmd_dry_run_off,
    "setup":       cmd_setup,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python tar_runpod_control.py <command>")
        print("Commands:", ", ".join(COMMANDS))
        sys.exit(1)

    ws = _get_workspace()
    COMMANDS[sys.argv[1]](ws)


if __name__ == "__main__":
    main()
