[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$StorageRoot = ""
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Split-Path -Parent $scriptDir
}

$resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$resolvedStorageRoot = [System.IO.Path]::GetFullPath($StorageRoot)

if (-not (Test-Path -LiteralPath $resolvedRepoRoot)) {
    throw "Repo root does not exist: $resolvedRepoRoot"
}

if (-not (Test-Path -LiteralPath (Join-Path $resolvedRepoRoot "README.md"))) {
    throw "Repo root does not look like the TAR repository: $resolvedRepoRoot"
}

. (Join-Path $PSScriptRoot "use_e_drive_env.ps1") -StorageRoot $resolvedStorageRoot

$repoDirs = @(
    "tar_state",
    "tar_runs",
    "logs",
    "dataset_artifacts",
    "training_artifacts",
    "eval_artifacts"
)

function Test-ReparsePoint {
    param([System.IO.FileSystemInfo]$Item)

    return ($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
}

function Move-DirectoryContents {
    param(
        [string]$Source,
        [string]$Target
    )

    foreach ($child in Get-ChildItem -LiteralPath $Source -Force) {
        $destination = Join-Path $Target $child.Name
        if (Test-Path -LiteralPath $destination) {
            throw "Refusing to overwrite existing path: $destination"
        }
        Move-Item -LiteralPath $child.FullName -Destination $Target
    }
}

foreach ($name in $repoDirs) {
    $sourcePath = Join-Path $resolvedRepoRoot $name
    $targetPath = Join-Path $resolvedStorageRoot $name

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetPath) | Out-Null

    if (Test-Path -LiteralPath $sourcePath) {
        $sourceItem = Get-Item -LiteralPath $sourcePath -Force
        if (Test-ReparsePoint -Item $sourceItem) {
            Write-Host "Leaving existing junction/reparse point in place: $sourcePath"
            continue
        }

        if (-not (Test-Path -LiteralPath $targetPath)) {
            Move-Item -LiteralPath $sourcePath -Destination $targetPath
        }
        else {
            Move-DirectoryContents -Source $sourcePath -Target $targetPath
            Remove-Item -LiteralPath $sourcePath -Force
        }
    }
    else {
        New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
    }

    if (-not (Test-Path -LiteralPath $sourcePath)) {
        New-Item -ItemType Junction -Path $sourcePath -Target $targetPath | Out-Null
    }

    Write-Host ("{0} -> {1}" -f $sourcePath, $targetPath)
}

Write-Host ""
Write-Host "Storage layout is ready."
Write-Host "Dot-source scripts\use_e_drive_env.ps1 in future shells before long runs."
