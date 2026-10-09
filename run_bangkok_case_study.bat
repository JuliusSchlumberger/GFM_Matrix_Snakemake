@echo off
REM Bangkok / Chao Phraya case study - full pipeline, run locally from the
REM terminal (Marjolijn, 2026-10-08). Steps 1 (tile selection) and 2 (input
REM copy + config materialize/resolve) have already been done once; this
REM script runs steps 3-6: simulate (two hop-ordered waves, 3 cores) ->
REM merge+exposure (Snakemake postprocess) -> case-polygon exposure CSV ->
REM plots. Each step must finish before the next starts - stops immediately
REM on any failure (non-zero exit code) rather than continuing on broken
REM input, same "fail loud, don't cascade" approach run_simulation_batch.py
REM itself already uses for one triple's own failure.
REM
REM Usage: just double-click this file, or run it from a terminal:
REM     run_bangkok_case_study.bat

setlocal

set REPO=%~dp0
set PYEXE=C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\python.exe
set SNAKEMAKE=C:\Users\schlumbe\AppData\Local\miniforge3\envs\gfm_python_preprocessing\Scripts\snakemake.exe
set STUDY_ROOT=P:\11212688-004-global-floodmaps\modelling\bangkok_chao_phraya
set CASE_GPKG=%STUDY_ROOT%\bangkok_chao_phraya_domain.gpkg
set MATERIALIZED=%REPO%snakemake_workflow\config\bangkok_chao_phraya_materialized.yml
set RESOLVED=%STUDY_ROOT%\resolved_config.yml

echo.
echo === Step 3a: simulate wave 0 (hop_distance=0 tiles, 270 triples, 3 workers) ===
"%PYEXE%" "%REPO%snakemake_workflow\scripts\run_simulation_batch.py" ^
  --triples-file "%STUDY_ROOT%\dispatch\wave0_triples.csv" ^
  --config "%RESOLVED%" --workers 3 ^
  --fail-log "%STUDY_ROOT%\dispatch\wave0_fail.txt"
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo FAILED at wave 0 - see %STUDY_ROOT%\dispatch\wave0_fail.txt for the failed tile/rp/slr triples.
  exit /b 1
)

echo.
echo === Step 3b: simulate wave 1 (hop_distance=1 tiles, 135 triples, 3 workers) ===
"%PYEXE%" "%REPO%snakemake_workflow\scripts\run_simulation_batch.py" ^
  --triples-file "%STUDY_ROOT%\dispatch\wave1_triples.csv" ^
  --config "%RESOLVED%" --workers 3 ^
  --fail-log "%STUDY_ROOT%\dispatch\wave1_fail.txt"
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo FAILED at wave 1 - see %STUDY_ROOT%\dispatch\wave1_fail.txt for the failed tile/rp/slr triples.
  exit /b 1
)

echo.
echo === Step 4+5: merge + exposure (Snakemake postprocess target) ===
set GFM_CONFIG_PATH=%MATERIALIZED%
"%SNAKEMAKE%" postprocess --cores 3 --rerun-triggers mtime --snakefile "%REPO%Snakefile"
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo FAILED at Snakemake postprocess.
  exit /b 1
)

echo.
echo === Step 5b: case-polygon exposure CSV ===
"%PYEXE%" "%REPO%analysis\compute_bangkok_case_exposure.py" ^
  --config "%MATERIALIZED%" ^
  --case-polygon-gpkg "%CASE_GPKG%" --case-polygon-layer bangkok_tile
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo FAILED at compute_bangkok_case_exposure.py.
  exit /b 1
)

set EXPOSURE_CSV=%STUDY_ROOT%\merged_results\exposure\bangkok_case_exposure.csv

echo.
echo === Step 6a: exposure vs RP plot ===
"%PYEXE%" "%REPO%analysis\plot_bangkok_exposure_vs_rp.py" ^
  --exposure-csv "%EXPOSURE_CSV%" --outdir "%STUDY_ROOT%\figures"
if %ERRORLEVEL% NEQ 0 (echo FAILED at plot_bangkok_exposure_vs_rp.py. & exit /b 1)

echo.
echo === Step 6b: case flood map (RP100) ===
"%PYEXE%" "%REPO%analysis\plot_bangkok_case_flood_map.py" ^
  --config "%MATERIALIZED%" ^
  --case-polygon-gpkg "%CASE_GPKG%" --case-polygon-layer bangkok_tile ^
  --return-period RP100
if %ERRORLEVEL% NEQ 0 (echo FAILED at plot_bangkok_case_flood_map.py. & exit /b 1)

echo.
echo === Step 6c: EAI time series (SSP1/SSP2/SSP5) ===
"%PYEXE%" "%REPO%analysis\plot_bangkok_eai_timeseries.py" ^
  --config "%MATERIALIZED%" --exposure-csv "%EXPOSURE_CSV%"
if %ERRORLEVEL% NEQ 0 (echo FAILED at plot_bangkok_eai_timeseries.py. & exit /b 1)

echo.
echo === Done. Figures written to %STUDY_ROOT%\figures ===
endlocal
