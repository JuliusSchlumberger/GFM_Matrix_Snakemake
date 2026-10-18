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

REM Conda env with the pipeline (`gfm`; named gfm_python_preprocessing on this machine).
REM Override by setting GFM_ENV before running this file.
if not defined GFM_ENV set "GFM_ENV=%LOCALAPPDATA%\miniforge3\envs\gfm_python_preprocessing"
set "PYTHON=%GFM_ENV%\python.exe"
set "SNAKEMAKE=%GFM_ENV%\Scripts\snakemake.exe"
set "CFG=%~dp0snakemake_workflow\config\thailand_bangkok_materialized.yml"
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
