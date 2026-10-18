@echo off
REM Bangkok / Chao Phraya case study, run locally from the terminal: steps 3-6,
REM simulate (two hop-ordered waves, 3 cores) -> merge+exposure (Snakemake
REM postprocess) -> case-polygon exposure CSV -> plots.
REM Prerequisites (steps 1-2): tile selection, input copy, and the materialized +
REM resolved configs, i.e. %STUDY_ROOT%\resolved_config.yml,
REM %STUDY_ROOT%\dispatch\wave{0,1}_triples.csv and
REM snakemake_workflow\config\bangkok_chao_phraya_materialized.yml.
REM Each step must finish before the next starts: the script stops on the first
REM non-zero exit code rather than continuing on broken input.
REM
REM Usage: just double-click this file, or run it from a terminal:
REM     run_bangkok_case_study.bat

setlocal

set REPO=%~dp0
REM Conda env with the pipeline (`gfm`; named gfm_python_preprocessing on this machine).
REM Override by setting GFM_ENV before running this file.
if not defined GFM_ENV set "GFM_ENV=%LOCALAPPDATA%\miniforge3\envs\gfm_python_preprocessing"
set PYEXE=%GFM_ENV%\python.exe
set SNAKEMAKE=%GFM_ENV%\Scripts\snakemake.exe
set MATERIALIZED=%REPO%snakemake_workflow\config\bangkok_chao_phraya_materialized.yml
REM Study data lives at {paths.root}/bangkok_chao_phraya, paths.root read from the
REM materialized config. Override by setting STUDY_ROOT before running this file.
if not defined STUDY_ROOT (
  for /f "usebackq delims=" %%R in (`call "%PYEXE%" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['paths']['root'])" "%MATERIALIZED%"`) do set "STUDY_ROOT=%%R/bangkok_chao_phraya"
)
if not defined STUDY_ROOT (
  echo FAILED to read paths.root from %MATERIALIZED% - materialize it or set STUDY_ROOT.
  exit /b 1
)
set "STUDY_ROOT=%STUDY_ROOT:/=\%"
set CASE_GPKG=%STUDY_ROOT%\bangkok_chao_phraya_domain.gpkg
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
  --exposure-csv "%EXPOSURE_CSV%" --out-dir "%STUDY_ROOT%\figures"
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
