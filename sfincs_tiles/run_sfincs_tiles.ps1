<#
.SYNOPSIS
    Build and run SFINCS for a list of GFM tiles, end to end.

.DESCRIPTION
    Runs the full sfincs_tiles/ pipeline (build_elevation.py ->
    build_roughness.py -> build_boundary_forcing.py -> build_sfincs_tile.py
    -> run_sfincs_tile.py) for each tile ID given, across the two required
    conda environments (gfm_python_preprocessing for the prep stage,
    hydromt-sfincs-dev for the SFINCS build/run stage - hydromt_sfincs needs
    a much newer hydromt than the rest of this pipeline uses, see
    build_sfincs_tile.py's own module docstring).

    Continues to the next tile if one fails (e.g. "no COAST-HG station
    within range" is a real, expected drop for some tiles per the plan doc,
    not necessarily a bug) - prints a final per-tile PASS/FAIL summary
    rather than stopping the whole batch on the first failure.

.PARAMETER TileIds
    One or more tile IDs to process, e.g. -TileIds 1573,929,851

.PARAMETER SfincsExe
    Path to sfincs.exe. Defaults to the one already located this session.

.PARAMETER Config
    Path to the GFM config.yml. Defaults to the repo's own production config.

.EXAMPLE
    .\run_sfincs_tiles.ps1 -TileIds 1573,929,851
#>

param(
    [Parameter(Mandatory = $true)]
    [int[]] $TileIds,

    [string] $MainPy = "C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\python.exe",
    [string] $SfincsPy = "C:\Users\schlumbe\AppData\Local\miniforge3\envs\hydromt-sfincs-dev\python.exe",
    [string] $SfincsExe = "C:\Users\schlumbe\hydromt_sfincs\delta_model\software\SFINCS_v2.3.0_mt_Faber_release_exe\sfincs.exe",
    [string] $Config = "$PSScriptRoot\..\snakemake_workflow\config\config.yml",
    [double] $TimeoutS = 1800.0
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot

# One (label, exe, script, extraArgs) step per pipeline stage - run in this
# exact order for every tile. $tileId is substituted into ExtraArgs at call
# time, not here.
$steps = @(
    @{ Label = "prep: elevation";        Exe = $MainPy;   Script = "build_elevation.py" },
    @{ Label = "prep: roughness";        Exe = $MainPy;   Script = "build_roughness.py" },
    @{ Label = "prep: boundary forcing"; Exe = $MainPy;   Script = "build_boundary_forcing.py" },
    @{ Label = "build SFINCS model";     Exe = $SfincsPy; Script = "build_sfincs_tile.py" },
    @{ Label = "run SFINCS + postprocess"; Exe = $SfincsPy; Script = "run_sfincs_tile.py"; ExtraArgs = @("--sfincs-exe", $SfincsExe, "--timeout-s", $TimeoutS) }
)

$results = @{}

foreach ($tileId in $TileIds) {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "=== Tile $tileId ==="
    Write-Host "=================================================="

    $tileFailed = $false
    $failedStep = $null

    foreach ($step in $steps) {
        if ($tileFailed) { break }

        $args = @("--tile-id", $tileId, "--config", $Config)
        if ($step.ExtraArgs) { $args += $step.ExtraArgs }

        Write-Host ""
        Write-Host "--- [$($step.Label)] tile $tileId ---"
        $t0 = Get-Date
        & $step.Exe "$scriptDir\$($step.Script)" @args
        $exitCode = $LASTEXITCODE
        $elapsed = (Get-Date) - $t0

        if ($exitCode -ne 0) {
            Write-Host "FAILED: [$($step.Label)] tile $tileId exited with code $exitCode after $($elapsed.TotalSeconds.ToString('F1'))s" -ForegroundColor Red
            $tileFailed = $true
            $failedStep = $step.Label
        }
        else {
            Write-Host "OK: [$($step.Label)] tile $tileId ($($elapsed.TotalSeconds.ToString('F1'))s)" -ForegroundColor Green
        }
    }

    $results[$tileId] = if ($tileFailed) { "FAILED at '$failedStep'" } else { "PASSED" }
}

Write-Host ""
Write-Host "=================================================="
Write-Host "=== Summary ==="
Write-Host "=================================================="
foreach ($tileId in $TileIds) {
    $status = $results[$tileId]
    $color = if ($status -eq "PASSED") { "Green" } else { "Red" }
    Write-Host ("tile {0,-8} {1}" -f $tileId, $status) -ForegroundColor $color
}
