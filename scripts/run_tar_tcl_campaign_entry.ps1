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
    throw "RunId is required."
}

. (Join-Path $PSScriptRoot "use_e_drive_env.ps1")
$env:TAR_TARGET_IMAGE_LOCKING = "host"
$env:PYTHONUNBUFFERED = "1"

Set-Location -LiteralPath $resolvedRepoRoot

& $PythonExe `
    "scripts/run_tar_tcl_campaign.py" `
    "--workspace" $resolvedRepoRoot `
    "--campaign-root" $resolvedCampaignRoot `
    "--run-id" $RunId `
    "--duration-hours" $DurationHours.ToString([System.Globalization.CultureInfo]::InvariantCulture) `
    "--poll-interval-s" $PollIntervalSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)

exit $LASTEXITCODE
