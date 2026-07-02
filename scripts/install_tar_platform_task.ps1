<#
Install (or remove) the TAR platform supervisor as a Windows Scheduled Task.

Fixes the watchdog single-point-of-failure: the supervisor runs at logon and
every N minutes, resurrecting the platform if the watchdog has died or the
machine rebooted. Runs in the CURRENT USER context (no admin required) via
schtasks.exe with an ONLOGON trigger + minute repetition.

SAFETY: this only keeps the SUPERVISOR alive. Execution stays human-gated:
the autonomy ramp is fail-closed, RAIL-3 needs a committed manifest, and the
supervisor only acts while tar_state\watchdog_autostart.enabled exists. This
is NOT the retired auto-execution-on-boot behaviour — nothing runs an
experiment without the existing human confirmations.

Usage:
  powershell -ExecutionPolicy Bypass -File scripts\install_tar_platform_task.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install_tar_platform_task.ps1 -IntervalMinutes 10
  powershell -ExecutionPolicy Bypass -File scripts\install_tar_platform_task.ps1 -Uninstall
#>
param(
    [int]$IntervalMinutes = 10,
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
$TaskName = "TAR Platform Supervisor"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$state = Join-Path $repo "tar_state"
$supervisor = Join-Path $repo "scripts\tar_platform_supervisor.py"

# Resolve venv python
$python = Join-Path (Split-Path $repo -Parent) ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = Join-Path $repo ".venv\Scripts\python.exe" }
if (-not (Test-Path $python)) { throw "venv python not found near $repo" }

if ($Uninstall) {
    schtasks /Delete /TN $TaskName /F 2>$null
    $flag = Join-Path $state "watchdog_autostart.enabled"
    if (Test-Path $flag) { Remove-Item $flag -Force }
    Write-Output "Uninstalled '$TaskName' and cleared the autostart opt-in flag."
    return
}

# Durable opt-in flag (installing the task IS the opt-in).
New-Item -ItemType Directory -Force $state | Out-Null
$flag = Join-Path $state "watchdog_autostart.enabled"
if (-not (Test-Path $flag)) {
    Set-Content -Path $flag -Value ((Get-Date).ToUniversalTime().ToString("o") + " enabled by install_tar_platform_task.ps1") -Encoding utf8
}

# schtasks: MINUTE schedule in the current-user context (no admin needed).
# Runs every N minutes while the user is logged on; after a reboot it resumes
# once the user logs in, so it covers both a watchdog crash and a reboot
# (relaunch within N minutes). The supervisor itself is a cheap no-op when the
# watchdog is already alive.
$action = "`"$python`" `"$supervisor`""
schtasks /Create /TN $TaskName /TR $action /SC MINUTE /MO $IntervalMinutes /F | Out-Null

Write-Output "Installed '$TaskName':"
Write-Output "  trigger    : every $IntervalMinutes min (current-user context)"
Write-Output "  action     : $action"
Write-Output "  opt-in flag: $flag"
schtasks /Query /TN $TaskName /FO LIST 2>$null | Select-String "TaskName|Status|Schedule|Next Run"
Write-Output "Run now to verify: schtasks /Run /TN `"$TaskName`""
