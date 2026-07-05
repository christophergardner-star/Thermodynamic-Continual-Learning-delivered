<#
    WS5.1 — nightly backup of TAR's non-git state.

    tar_state lives on E: (PhysicalDisk ST3000DM001). git-tracked code is already
    safe on GitHub (origin). This backs up the state that is NOT in git — the
    560 MB literature/knowledge DB (incl. chroma embeddings that are expensive to
    regenerate), experiment results, the integrity anchors, and the governance
    JSONs — to the dedicated G: "Backup" drive, a SEPARATE physical disk.

    - Big dirs: robocopy /MIR (incremental mirror; cheap after the first run).
    - Governance JSONs: also snapshotted into a dated folder so point-in-time
      history survives even if the live file is corrupted (mirror alone would
      propagate the corruption). Snapshots older than 30 days are pruned.

    Safe to run any time; read-only w.r.t. the source. Registered as a nightly
    Scheduled Task "TAR State Backup".
#>
param(
    [string]$Src    = "E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state",
    [string]$Root   = "G:\TAR_state_backup"
)

$Dst    = Join-Path $Root "tar_state"
$LogDir = Join-Path $Root "logs"
$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path $Dst, $LogDir | Out-Null
$log = Join-Path $LogDir "backup_$stamp.log"

function Invoke-Robo {
    param([string[]]$RoboArgs)
    robocopy @RoboArgs /XJ /R:1 /W:1 /NP /NFL /NDL "/LOG+:$log" | Out-Null
    # robocopy: 0-7 = success (bit-coded), 8+ = real failure.
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $($RoboArgs -join ' ')" }
    $global:LASTEXITCODE = 0
}

# --- 1. incremental mirror of the ENTIRE tar_state tree ----------------------
# Mirror the whole tree (not a hand-picked subset) so nothing is silently
# missed: anchors live in several places (self_improvement/anchor_manifest.json,
# autonomous_research/deep_anchor.json), and dirs like validation/ and adapters/
# hold non-regenerable results. /XJ skips junctions (tar_state is reached via one).
Invoke-Robo @($Src, $Dst, "/MIR", "/MT:8")

# --- 2. the CANONICAL ANCHOR CHAIN lives at the WORKSPACE ROOT, not tar_state -
$WsRoot = Split-Path $Src -Parent
$rootAnchors = Join-Path $WsRoot "anchors"
if (Test-Path $rootAnchors) {
    Invoke-Robo @($rootAnchors, (Join-Path $Root "workspace_anchors"), "/MIR", "/MT:4")
}

# --- 3. dated integrity snapshot of the tiny governance + anchor files -------
$snap = Join-Path (Join-Path $Root "snapshots") $stamp
New-Item -ItemType Directory -Force -Path $snap | Out-Null
foreach ($f in @("autonomy_ramp.json", "autonomy_ramp_configured.flag",
                 "honest_evidence_inventory.json", "active_preregistration.json",
                 "frontier_problems.json", "evidence_ingest_state.json")) {
    $p = Join-Path $Src $f
    if (Test-Path $p) { Copy-Item $p (Join-Path $snap $f) -Force }
}
# scattered anchors + the loop kill-ledger
foreach ($rel in @("solution_loop", "self_improvement\anchor_manifest.json",
                   "autonomous_research\deep_anchor.json")) {
    $p = Join-Path $Src $rel
    if (Test-Path $p) { Copy-Item $p (Join-Path $snap (Split-Path $rel -Leaf)) -Recurse -Force }
}
if (Test-Path $rootAnchors) { Copy-Item $rootAnchors (Join-Path $snap "workspace_anchors") -Recurse -Force }

# --- 4. prune snapshots older than 30 days -----------------------------------
Get-ChildItem (Join-Path $Root "snapshots") -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.CreationTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Recurse -Force -Confirm:$false -ErrorAction SilentlyContinue

"OK  $stamp  ->  $Dst" | Out-File -FilePath (Join-Path $Root "last_backup.txt") -Encoding utf8
Write-Output "backup complete: $stamp"
