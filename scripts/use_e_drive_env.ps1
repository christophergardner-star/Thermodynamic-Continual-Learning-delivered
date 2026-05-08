param(
    [string]$StorageRoot = ""
)

function Resolve-TarStorageRoot {
    param([string]$RequestedRoot)

    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        return [System.IO.Path]::GetFullPath($RequestedRoot)
    }

    $repoName = Split-Path -Leaf (Split-Path -Parent $PSScriptRoot)
    $candidateRoots = @(
        "E:\TAR\$repoName",
        "D:\TAR\$repoName",
        "F:\TAR\$repoName"
    )

    foreach ($candidate in $candidateRoots) {
        if (Test-Path -LiteralPath (Split-Path -Parent $candidate)) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    return [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
}

$resolvedRoot = Resolve-TarStorageRoot -RequestedRoot $StorageRoot
$hfRoot = Join-Path $resolvedRoot "hf"
$tempRoot = Join-Path $resolvedRoot "tmp"
$torchRoot = Join-Path $resolvedRoot "torch"
$pipRoot = Join-Path $resolvedRoot "pip"
$xdgRoot = Join-Path $resolvedRoot "xdg"
$mplRoot = Join-Path $xdgRoot "matplotlib"
$numbaRoot = Join-Path $xdgRoot "numba"
$pycacheRoot = Join-Path $tempRoot "pycache"

$requiredDirs = @(
    $resolvedRoot,
    $hfRoot,
    (Join-Path $hfRoot "hub"),
    (Join-Path $hfRoot "transformers"),
    (Join-Path $hfRoot "datasets"),
    $tempRoot,
    $torchRoot,
    $pipRoot,
    $xdgRoot,
    $mplRoot,
    $numbaRoot,
    $pycacheRoot,
    (Join-Path $resolvedRoot "tar_state"),
    (Join-Path $resolvedRoot "tar_runs"),
    (Join-Path $resolvedRoot "logs"),
    (Join-Path $resolvedRoot "dataset_artifacts"),
    (Join-Path $resolvedRoot "training_artifacts"),
    (Join-Path $resolvedRoot "eval_artifacts"),
    (Join-Path $resolvedRoot "paper")
)

foreach ($dir in $requiredDirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$env:TAR_STORAGE_ROOT = $resolvedRoot
$env:TAR_WORKSPACE = $resolvedRoot
$env:HF_HOME = $hfRoot
$env:HUGGINGFACE_HUB_CACHE = Join-Path $hfRoot "hub"
$env:TRANSFORMERS_CACHE = Join-Path $hfRoot "transformers"
$env:HF_DATASETS_CACHE = Join-Path $hfRoot "datasets"
$env:TORCH_HOME = $torchRoot
$env:PIP_CACHE_DIR = $pipRoot
$env:XDG_CACHE_HOME = $xdgRoot
$env:MPLCONFIGDIR = $mplRoot
$env:NUMBA_CACHE_DIR = $numbaRoot
$env:PYTHONPYCACHEPREFIX = $pycacheRoot
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:TMPDIR = $tempRoot

Write-Host "Configured TAR cache and temp environment on $resolvedRoot"
