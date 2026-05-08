"""
TAR Phase Queue Manager
=======================
Waits for the running 24h campaigns to complete, then runs every phase
in order, unattended. Sends a Windows toast notification after each step.

Run this once and leave the PC - it manages everything from here.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tar_queue_bridge import queue_step_env, update_queue_state
from tar_storage import ensure_workspace_layout, preferred_python, storage_env


REPO = Path(__file__).resolve().parent
WORKSPACE = ensure_workspace_layout(repo_root=REPO)
PYTHON = preferred_python(REPO)
QUEUE_NAME = "queue1"

CAMPAIGN_LOGS = [
    WORKSPACE / "training_artifacts" / "campaigns" / "tcl24-0430" / "campaign.log",
    WORKSPACE / "training_artifacts" / "campaigns" / "tcl24-0430b" / "campaign.log",
]
CAMPAIGN_IDLE_SECS = 600
POLL_INTERVAL_SECS = 120
QUEUE_LOG = WORKSPACE / "tar_state" / "queue_run.log"

ENV = storage_env(WORKSPACE)
ENV["PYTHONUNBUFFERED"] = "1"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    try:
        QUEUE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with QUEUE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _notify(title: str, body: str) -> None:
    safe_title = title.replace('"', "'")
    safe_body = body.replace('"', "'")
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Application; "
        f'$n.BalloonTipTitle = "{safe_title}"; '
        f'$n.BalloonTipText  = "{safe_body}"; '
        "$n.Visible = $True; "
        "$n.ShowBalloonTip(20000); "
        "Start-Sleep 21; "
        "$n.Dispose()"
    )
    try:
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        pass


def _campaign_idle_secs(log_path: Path) -> float:
    if not log_path.exists():
        return float("inf")
    return time.time() - log_path.stat().st_mtime


def _all_campaigns_done() -> bool:
    return all(_campaign_idle_secs(log) >= CAMPAIGN_IDLE_SECS for log in CAMPAIGN_LOGS)


def _wait_for_campaigns() -> None:
    _log("=" * 70)
    _log("TAR Phase Queue - waiting for campaigns to finish")
    _log(f"Watching: {[str(p) for p in CAMPAIGN_LOGS]}")
    _log(f"A campaign is considered done when its log has been idle {CAMPAIGN_IDLE_SECS}s")
    _log("=" * 70)
    update_queue_state(
        WORKSPACE,
        queue_name=QUEUE_NAME,
        status="waiting",
        current_step="campaign_wait",
        message="Waiting for precursor campaigns",
    )

    while not _all_campaigns_done():
        statuses = []
        for log in CAMPAIGN_LOGS:
            idle = _campaign_idle_secs(log)
            name = log.parent.name
            if idle == float("inf"):
                statuses.append(f"{name}: log not found")
            elif idle < CAMPAIGN_IDLE_SECS:
                statuses.append(f"{name}: running (idle {idle:.0f}s)")
            else:
                statuses.append(f"{name}: DONE (idle {idle:.0f}s)")
        _log("Campaign status: " + " | ".join(statuses))
        update_queue_state(
            WORKSPACE,
            queue_name=QUEUE_NAME,
            status="waiting",
            current_step="campaign_wait",
            message=" | ".join(statuses),
        )
        time.sleep(POLL_INTERVAL_SECS)

    _log("All campaigns finished - starting phase queue")
    _notify("TAR Campaigns Done", "Starting phase queue now")


def _run_step(
    name: str,
    script: Path,
    *,
    step_index: int,
    step_total: int,
    extra_args: list[str] | None = None,
) -> bool:
    if not script.exists():
        _log(f"SKIP {name} - script not found: {script}")
        return False

    _log("")
    _log("=" * 70)
    _log(f"START  {name}")
    _log(f"Script {script}")
    _log("=" * 70)
    _notify(f"TAR Starting: {name}", str(script.name))

    cmd = [str(PYTHON), str(script)] + (extra_args or [])
    step_env = queue_step_env(
        ENV,
        queue_name=QUEUE_NAME,
        current_step=name,
        step_index=step_index,
        step_total=step_total,
        active_script=script.name,
    )
    update_queue_state(
        WORKSPACE,
        queue_name=QUEUE_NAME,
        status="running",
        current_step=name,
        step_index=step_index,
        step_total=step_total,
        active_script=script.name,
        message=f"Launching {script.name}",
    )

    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=step_env)
        elapsed = time.time() - t0
        hrs = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        if result.returncode == 0:
            _log(f"DONE   {name} - elapsed {hrs}h {mins}m")
            _notify(f"TAR Done: {name}", f"Completed in {hrs}h {mins}m - returncode 0")
            update_queue_state(
                WORKSPACE,
                queue_name=QUEUE_NAME,
                status="running",
                current_step=name,
                step_index=step_index,
                step_total=step_total,
                active_script=script.name,
                message=f"Completed {script.name}",
                last_returncode=0,
            )
            return True

        _log(f"FAIL   {name} - returncode {result.returncode} - elapsed {hrs}h {mins}m")
        _notify(f"TAR FAILED: {name}", f"Exit code {result.returncode} after {hrs}h {mins}m")
        update_queue_state(
            WORKSPACE,
            queue_name=QUEUE_NAME,
            status="running",
            current_step=name,
            step_index=step_index,
            step_total=step_total,
            active_script=script.name,
            message=f"Failed {script.name}",
            last_returncode=result.returncode,
        )
        return False
    except Exception as exc:
        elapsed = time.time() - t0
        _log(f"ERROR  {name} - {exc} - elapsed {elapsed:.0f}s")
        _notify(f"TAR ERROR: {name}", str(exc)[:120])
        update_queue_state(
            WORKSPACE,
            queue_name=QUEUE_NAME,
            status="failed",
            current_step=name,
            step_index=step_index,
            step_total=step_total,
            active_script=script.name,
            message=str(exc),
        )
        return False


def _build_queue() -> list[tuple[str, Path, list[str]]]:
    steps = [
        ("Phase 10 - 4-way baseline (TCL/EWC/SI/SGD)", REPO / "phase10_baseline.py", []),
        ("Phase 11 - TCL ablation (4 conditions)", REPO / "phase11_ablation.py", []),
        ("Phase 12 - EWC lambda sweep", REPO / "phase12_ewc_sweep.py", []),
        ("Phase 13 - SI robustness sweep", REPO / "phase13_si_sweep.py", []),
        ("Phase 14 - Quantum publishability", REPO / "phase14_quantum_publishability.py", []),
        ("Phase 15 - Class-incremental search", REPO / "phase15_class_incremental_search.py", []),
    ]

    build_script = REPO / "build_tar_master_dataset.py"
    train_script = REPO / "train_tar_operator_sft.py"
    if build_script.exists():
        steps.append(("Build master dataset", build_script, ["--workspace", str(WORKSPACE)]))
    if train_script.exists():
        steps.append((
            "ASC fine-tune (DeepSeek-Coder 1.3B QLoRA)",
            train_script,
            [
                "--workspace", str(WORKSPACE),
                "--model", "deepseek-ai/deepseek-coder-1.3b-instruct",
                "--use-qlora", "--bits", "4",
                "--output-dir", str(WORKSPACE / "training_artifacts" / "asc_finetune"),
            ],
        ))

    steps.append((
        "TAR-Author - write paper + compile PDF",
        REPO / "tar_author.py",
        ["--workspace", str(WORKSPACE), "--output-dir", str(WORKSPACE / "paper")],
    ))
    steps.append(("TAR Autonomous Research - self-directed breakthrough science", REPO / "tar_autonomous_research.py", []))
    return steps


def main() -> None:
    ensure_workspace_layout(WORKSPACE, repo_root=REPO)
    _log("TAR Phase Queue Manager starting")
    _log(f"Python:    {PYTHON}")
    _log(f"Repo:      {REPO}")
    _log(f"Workspace: {WORKSPACE}")
    _log(f"Queue log: {QUEUE_LOG}")
    update_queue_state(
        WORKSPACE,
        queue_name=QUEUE_NAME,
        status="starting",
        current_step="boot",
        message="Queue 1 boot",
    )

    skip_wait = "--no-wait" in sys.argv
    if skip_wait:
        _log("--no-wait flag set - skipping campaign wait, running queue immediately")
    else:
        _wait_for_campaigns()

    queue = _build_queue()
    total = len(queue)
    _log(f"\nQueue has {total} steps:")
    for i, (name, script, _) in enumerate(queue, 1):
        exists = "OK" if script.exists() else "MISSING"
        _log(f"  {i:2d}. [{exists}] {name}")

    passed = 0
    failed = 0
    skipped = 0
    t_queue_start = time.time()

    for i, (name, script, args) in enumerate(queue, 1):
        _log(f"\nQueue progress: {i}/{total}")
        if not script.exists():
            _log(f"SKIP {name} - script missing")
            update_queue_state(
                WORKSPACE,
                queue_name=QUEUE_NAME,
                status="running",
                current_step=name,
                step_index=i,
                step_total=total,
                active_script=script.name,
                message="Skipped missing script",
            )
            skipped += 1
            continue
        ok = _run_step(name, script, step_index=i, step_total=total, extra_args=args)
        if ok:
            passed += 1
        else:
            failed += 1

    elapsed = time.time() - t_queue_start
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)

    _log("")
    _log("=" * 70)
    _log(f"QUEUE COMPLETE - {passed} passed / {failed} failed / {skipped} skipped")
    _log(f"Total elapsed: {hrs}h {mins}m")
    _log("=" * 70)
    _notify("TAR Queue Complete", f"{passed}/{total} steps passed | {hrs}h {mins}m total")
    update_queue_state(
        WORKSPACE,
        queue_name=QUEUE_NAME,
        status="complete" if failed == 0 else "failed",
        current_step="done",
        step_index=total,
        step_total=total,
        message=f"passed={passed} failed={failed} skipped={skipped}",
        last_returncode=0 if failed == 0 else 1,
    )


if __name__ == "__main__":
    main()
