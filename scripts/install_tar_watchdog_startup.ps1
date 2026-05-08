$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $repo "run_tar_watchdog.bat"
$taskName = "TAR Watchdog"

if (-not (Test-Path $launcher)) {
    throw "Launcher not found: $launcher"
}

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$launcher`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Keeps TAR dashboard and research daemons running after crashes or reboots." `
    -User $env:USERNAME `
    -Force | Out-Null

Write-Host "Installed scheduled task: $taskName"
