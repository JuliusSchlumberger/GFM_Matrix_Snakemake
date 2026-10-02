<#
Thailand / Bangkok bay side project - full local sequential run.
preprocess -> simulate -> postprocess -> exposure/EAI analysis, each stage
only starting once the previous one has fully finished. Tiles 31 and 499
are both hop_distance=0 (real ocean edge, manually resized by Marjolijn for
guaranteed overlap - see snakemake_workflow/config/thailand_bangkok.yml's
own comment), so there is no wave-ordering dependency between them and
`simulate` can run as a single stage.

Usage (from the repo root, in PowerShell):
    .\run_thailand_bangkok.ps1

Side-project scoped - not part of the core maintained pipeline. Safe to
delete once this project is done.
#>

$ErrorActionPreference = "Stop"

$Python    = "C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\python.exe"
$Snakemake = "C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\Scripts\snakemake.exe"
$Cfg       = "C:\Users\schlumbe\GFM_Matrix_Snakemake\snakemake_workflow\config\thailand_bangkok_materialized.yml"
$Cores     = 4
$MemMb     = 24000

$env:GFM_CONFIG_PATH = $Cfg

function Invoke-Stage($Name, $ScriptBlock) {
    Write-Host "################################################################"
    Write-Host "# $Name"
    Write-Host "################################################################"
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) {
        Write-Host "STAGE FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Stage "STAGE 1/4: preprocess (DEM/mask/friction/boundaries, 2 tiles)" {
    & $Snakemake preprocess --cores $Cores --resources "mem_mb=$MemMb" --rerun-triggers mtime -p
}

Invoke-Stage "STAGE 2/4: simulate (31, 499 - single wave, both hop_distance=0)" {
    & $Snakemake simulate --cores $Cores --resources "mem_mb=$MemMb" --rerun-triggers mtime -p
}

Invoke-Stage "STAGE 3/4: postprocess (merge chunks, flood fraction, mosaics)" {
    & $Snakemake postprocess --cores $Cores --resources "mem_mb=$MemMb" --rerun-triggers mtime -p
}

Invoke-Stage "STAGE 4/4: exposure / EAI analysis (skip world maps - 2-tile run)" {
    & $Python analysis/run_analysis.py --config $Cfg --skip-world-maps
}

Write-Host "################################################################"
Write-Host "# ALL STAGES COMPLETE"
Write-Host "################################################################"
