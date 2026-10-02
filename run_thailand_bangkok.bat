@echo off
REM Thailand / Bangkok bay side project - full local sequential run.
REM preprocess -> simulate -> postprocess -> exposure/EAI analysis, each
REM stage only starting once the previous one has fully finished. Tiles 31
REM and 499 are both hop_distance=0 (real ocean edge, manually resized for
REM guaranteed overlap - see snakemake_workflow\config\thailand_bangkok.yml's
REM own comment), so there is no wave-ordering dependency between them and
REM `simulate` can run as a single stage.
REM
REM Usage (from the repo root, in cmd.exe / Miniforge Prompt or PowerShell):
REM     run_thailand_bangkok.bat
REM
REM Side-project scoped - not part of the core maintained pipeline. Safe to
REM delete once this project is done.

set "PYTHON=C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\python.exe"
set "SNAKEMAKE=C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\Scripts\snakemake.exe"
set "CFG=C:\Users\schlumbe\GFM_Matrix_Snakemake\snakemake_workflow\config\thailand_bangkok_materialized.yml"
set "CORES=4"
set "MEM_MB=24000"
set "GFM_CONFIG_PATH=%CFG%"

echo ################################################################
echo # STAGE 1/4: preprocess (DEM/mask/friction/boundaries, 2 tiles)
echo ################################################################
"%SNAKEMAKE%" preprocess --cores %CORES% --resources mem_mb=%MEM_MB% --rerun-triggers mtime -p
if errorlevel 1 (
    echo STAGE FAILED: preprocess
    exit /b 1
)

echo ################################################################
echo # STAGE 2/4: simulate - tiles 31 and 499, single wave, both hop_distance=0
echo ################################################################
"%SNAKEMAKE%" simulate --cores %CORES% --resources mem_mb=%MEM_MB% --rerun-triggers mtime -p
if errorlevel 1 (
    echo STAGE FAILED: simulate
    exit /b 1
)

echo ################################################################
echo # STAGE 3/4: postprocess (merge chunks, flood fraction, mosaics)
echo ################################################################
"%SNAKEMAKE%" postprocess --cores %CORES% --resources mem_mb=%MEM_MB% --rerun-triggers mtime -p
if errorlevel 1 (
    echo STAGE FAILED: postprocess
    exit /b 1
)

echo ################################################################
echo # STAGE 4/4: exposure / EAI analysis (skip world maps - 2-tile run)
echo ################################################################
"%PYTHON%" analysis\run_analysis.py --config "%CFG%" --skip-world-maps
if errorlevel 1 (
    echo STAGE FAILED: exposure analysis
    exit /b 1
)

echo ################################################################
echo # ALL STAGES COMPLETE
echo ################################################################
