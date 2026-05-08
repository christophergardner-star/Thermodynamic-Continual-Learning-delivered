[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [double]$DurationHours = 48,
    [double]$PollIntervalSeconds = 180,
    [string]$CampaignRoot = "",
    [string]$RunId = "",
    [string]$PythonExe = "python"
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Split-Path -Parent $scriptDir
}

$resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $CampaignRoot = Join-Path $resolvedRepoRoot "training_artifacts\campaigns"
}
$resolvedCampaignRoot = [System.IO.Path]::GetFullPath($CampaignRoot)

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "tar-tcl-48hr-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

if (-not (Test-Path -LiteralPath (Join-Path $resolvedRepoRoot "README.md"))) {
    throw "Repo root does not look like the TAR repository: $resolvedRepoRoot"
}

. (Join-Path $PSScriptRoot "use_e_drive_env.ps1")
$env:TAR_TARGET_IMAGE_LOCKING = "host"
$env:PYTHONUNBUFFERED = "1"

$campaignDir = Join-Path $resolvedCampaignRoot $RunId
New-Item -ItemType Directory -Force -Path $campaignDir | Out-Null

$stdoutLog = Join-Path $campaignDir "host_stdout.log"
$stderrLog = Join-Path $campaignDir "host_stderr.log"
$statusPath = Join-Path $campaignDir "status.json"
$launchPath = Join-Path $campaignDir "launch.json"
$activeCampaignDir = Join-Path $resolvedRepoRoot "tar_state\campaigns"
$activeCampaignPath = Join-Path $activeCampaignDir "active_tar_tcl_campaign.json"
New-Item -ItemType Directory -Force -Path $activeCampaignDir | Out-Null

$argumentList = @(
    "scripts/run_tar_tcl_campaign.py",
    "--workspace", $resolvedRepoRoot,
    "--campaign-root", $resolvedCampaignRoot,
    "--run-id", $RunId,
    "--duration-hours", $DurationHours.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--poll-interval-s", $PollIntervalSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)

$process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $argumentList `
    -WorkingDirectory $resolvedRepoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$launchInfo = [ordered]@{
    run_id = $RunId
    pid = $process.Id
    repo_root = $resolvedRepoRoot
    campaign_root = $resolvedCampaignRoot
    campaign_dir = $campaignDir
    status_path = $statusPath
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    launch_path = $launchPath
    started_at = [DateTime]::UtcNow.ToString("o")
    duration_hours = $DurationHours
    poll_interval_s = $PollIntervalSeconds
    python_exe = $PythonExe
    tar_target_image_locking = $env:TAR_TARGET_IMAGE_LOCKING
    tar_storage_root = $env:TAR_STORAGE_ROOT
}

$launchInfo | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding UTF8
$launchInfo | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $activeCampaignPath -Encoding UTF8
$launchInfo | ConvertTo-Json -Depth 5
