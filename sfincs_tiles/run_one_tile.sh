#!/bin/bash
# Per-tile pipeline for a validation batch: copy eikonal inputs -> regenerate
# dem.tif/mask.tif (gfm env) -> build SFINCS inputs (hydromt-sfincs-dev env)
# -> bathtub+eikonal (gfm env) -> SFINCS run (direct apptainer, staged to
# local scratch) -> postprocess + per-model summary (hydromt-sfincs-dev env).
# Called per-tile from each batch sbatch script's loop - see
# generate_validation_batch_jobs.py --base-dir-name <name>
# --runner-script-name run_one_tile.sh.
#
# Idempotent: checks for the expected output file before redoing work, so
# re-running after a partial batch failure only redoes what's missing.
#
# Completeness/failure tracking: this script no longer keeps its own
# run_one_tile_failures.txt log (removed 2026-10-07) - that file only ever
# recorded that a tile failed AT SOME POINT, not whether it's still failing
# now (idempotent re-runs regularly fix a tile on a later attempt, but
# nothing ever removed its earlier entry - found to be ~75% stale in
# practice). Every failure still prints to this job's own stderr (so it's
# visible in hpc_jobs/logs/*.err as before); the authoritative CURRENT
# completeness check is report_calibration_tile_status.py, which reads
# real file presence on disk instead of a historical event log, and is
# also what generates missing_eikonal_pairs.csv for a targeted resubmission.
#
# Usage: run_one_tile.sh <tile_id> [--models bathtub,eikonal,sfincs] [--max-rounds N]
#
# --models: comma-separated subset of bathtub,eikonal,sfincs to (re-)run
# this pass. Default: all three. postprocess_tile_summary.py always runs
# regardless of --models, writing one summary_{model}.json per model
# (bathtub, eikonal, sfincs) from whatever that model's own raster on disk
# says right now - null fields for a model with no output yet, real values
# once it has one.
# --max-rounds: forwarded to run_eikonal_on_sfincs_subgrid.py's own
# --max-rounds (only meaningful when eikonal is in --models).
set -uo pipefail

# Clear conda env vars inherited from the parent shell so each python
# binary below falls back to its own bundled PROJ/GDAL data.
unset PROJ_LIB PROJ_DATA GDAL_DATA

TILE_ID="$1"
shift
MODELS="bathtub,eikonal,sfincs"
MAX_ROUNDS=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
    *) echo "run_one_tile.sh: unknown argument '$1'" >&2; exit 1 ;;
  esac
done

RUN_BATHTUB=false
RUN_EIKONAL=false
RUN_SFINCS=false
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
for m in "${MODEL_ARR[@]}"; do
  case "$m" in
    bathtub) RUN_BATHTUB=true ;;
    eikonal) RUN_EIKONAL=true ;;
    sfincs) RUN_SFINCS=true ;;
    *) echo "run_one_tile.sh: unknown model '$m' in --models (expected bathtub,eikonal,sfincs)" >&2; exit 1 ;;
  esac
done

CODE_ROOT="/u/schlumbe/gfm_code"
DATA_ROOT="/p/11212688-004-global-floodmaps/modelling"
# Required env var, set via a leading BASE_DIR_NAME=... prefix on the invocation.
: "${BASE_DIR_NAME:?BASE_DIR_NAME must be set, e.g. BASE_DIR_NAME=validation_sfincs_v5 bash run_one_tile.sh <tile_id>}"
CONFIG="$DATA_ROOT/$BASE_DIR_NAME/resolved_config.yml"
SFINCS_IMAGE="docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release"
SFINCS_TIMEOUT_S=14400

# Full python binary paths, not `conda activate` - a non-interactive
# subshell doesn't source ~/.bashrc, so `conda activate` isn't reliable.
HYDROMT_SFINCS_DEV_PY="/u/schlumbe/.conda/envs/hydromt-sfincs-dev/bin/python"
GFM_PY="/u/schlumbe/.conda/envs/gfm/bin/python"

TILE_DIR="$DATA_ROOT/$BASE_DIR_NAME/$TILE_ID"
MODEL_OUTPUTS_DIR="$DATA_ROOT/model_outputs/$TILE_ID/inputs"
INPUTS_DIR="$TILE_DIR/inputs"
SFINCS_MODEL_DIR="$TILE_DIR/sfincs_model"
TILE_STATUS_PY="$CODE_ROOT/sfincs_tiles/tile_status.py"

echo "=== tile $TILE_ID: starting ==="

# -- 0. fast short-circuit for a tile already known PERMANENTLY unsolvable
# (no_station / no_boundary_cells / antimeridian - see tile_status.py's own
# module docstring) - skip immediately, no retry, so neither this
# invocation nor a later --models "" postprocess-only rerun (the sweep
# batch's own final pass, which otherwise re-attempts every tile
# unconditionally) wastes time re-attempting guaranteed-to-fail work.
STATUS_CHECK=$("$GFM_PY" "$TILE_STATUS_PY" check --root "$DATA_ROOT" --base-dir-name "$BASE_DIR_NAME" --tile-id "$TILE_ID")
if [ "$?" -eq 0 ]; then
  echo "tile $TILE_ID: skipping - already known permanently unsolvable ($STATUS_CHECK)"
  echo "=== tile $TILE_ID: done (skipped) ==="
  exit 0
fi

# -- 1. copy eikonal inputs (no env needed - plain files, already built by production preprocessing) --
mkdir -p "$INPUTS_DIR"
for f in tile_geometry.gpkg model_bbox.json dem.tif mask.tif friction.tif boundaries_RP100_SLR_0.gpkg; do
  if [ ! -f "$INPUTS_DIR/$f" ]; then
    if [ ! -f "$MODEL_OUTPUTS_DIR/$f" ]; then
      echo "tile $TILE_ID: missing $MODEL_OUTPUTS_DIR/$f - skipping tile" >&2
      exit 0
    fi
    cp -f "$MODEL_OUTPUTS_DIR/$f" "$INPUTS_DIR/$f"
  fi
done

# -- 1.5. fast no-station check (first-time detection, before any status.json
# exists yet) - build_boundary_forcing.py does this same check internally and
# self-reports via tile_status.py, but it only runs AFTER
# regenerate_dem_mask.py/build_elevation.py/build_roughness.py (step 2-3
# below) - checking here too means a no-station tile skips BEFORE wasting
# time on any of that, on every attempt until tile_status.json exists to
# catch it via the fast path above instead. --
if ! "$GFM_PY" "$TILE_STATUS_PY" check-station --inputs-dir "$INPUTS_DIR" 2>/dev/null; then
  "$GFM_PY" "$TILE_STATUS_PY" write --root "$DATA_ROOT" --base-dir-name "$BASE_DIR_NAME" --tile-id "$TILE_ID" \
    --status no_station --stage run_one_tile.sh \
    --message "boundaries_RP100_SLR_0.gpkg is empty - no COAST-RP station for this tile"
  echo "tile $TILE_ID: no COAST-RP station (boundaries file empty) - skipping tile" >&2
  echo "=== tile $TILE_ID: done (skipped) ==="
  exit 0
fi

cd "$CODE_ROOT/sfincs_tiles"

# -- 2. regenerate dem.tif/mask.tif via extract_dem/extract_dem_mask (gfm env
# - needs src/config_utils.py's hydromt.DataCatalog). Gated on the same
# elevation_combined.tif check as step 3: dem.tif/mask.tif must not change
# once the SFINCS-input build has started. --
if [ ! -f "$SFINCS_MODEL_DIR/elevation_combined.tif" ]; then
  "$GFM_PY" regenerate_dem_mask.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || echo "tile $TILE_ID: regenerate_dem_mask.py failed (non-fatal - falling back to the copied model_outputs/ dem.tif/mask.tif)" >&2
fi

# -- 3. build SFINCS inputs (hydromt-sfincs-dev env). build_boundary_forcing.py/
# build_sfincs_tile.py self-report a classified status via tile_status.py
# (no_station/no_boundary_cells/antimeridian/other_error - see their own
# modules); build_elevation.py/build_roughness.py don't (never seen fail in
# practice), so this writes a generic other_error here as a fallback -
# "any other failure is properly logged" without requiring every build_*.py
# script to import tile_status.py for a category that's never been hit. --
if [ ! -f "$SFINCS_MODEL_DIR/elevation_combined.tif" ]; then
  "$HYDROMT_SFINCS_DEV_PY" build_elevation.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || {
      echo "tile $TILE_ID: build_elevation.py failed" >&2
      "$GFM_PY" "$TILE_STATUS_PY" write --root "$DATA_ROOT" --base-dir-name "$BASE_DIR_NAME" --tile-id "$TILE_ID" \
        --status other_error --stage build_elevation.py --message "see this batch's own .err log for the traceback"
      exit 0
    }
fi
if [ ! -f "$SFINCS_MODEL_DIR/manning_n.tif" ]; then
  "$HYDROMT_SFINCS_DEV_PY" build_roughness.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || {
      echo "tile $TILE_ID: build_roughness.py failed" >&2
      "$GFM_PY" "$TILE_STATUS_PY" write --root "$DATA_ROOT" --base-dir-name "$BASE_DIR_NAME" --tile-id "$TILE_ID" \
        --status other_error --stage build_roughness.py --message "see this batch's own .err log for the traceback"
      exit 0
    }
fi
if [ ! -f "$SFINCS_MODEL_DIR/matched_boundary_points.gpkg" ]; then
  "$HYDROMT_SFINCS_DEV_PY" build_boundary_forcing.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || { echo "tile $TILE_ID: build_boundary_forcing.py failed" >&2; exit 0; }
fi
if [ ! -f "$SFINCS_MODEL_DIR/sfincs.inp" ]; then
  "$HYDROMT_SFINCS_DEV_PY" build_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || { echo "tile $TILE_ID: build_sfincs_tile.py failed" >&2; exit 0; }
fi

# -- 4. bathtub + eikonal (gfm env - needs src/flood_model.py's older-hydromt import chain) --
PY_MODELS=()
$RUN_BATHTUB && PY_MODELS+=(bathtub)
$RUN_EIKONAL && PY_MODELS+=(eikonal)
if [ "${#PY_MODELS[@]}" -gt 0 ]; then
  MAX_ROUNDS_ARGS=()
  [ -n "$MAX_ROUNDS" ] && MAX_ROUNDS_ARGS=(--max-rounds "$MAX_ROUNDS")
  "$GFM_PY" run_eikonal_on_sfincs_subgrid.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    --models "${PY_MODELS[@]}" "${MAX_ROUNDS_ARGS[@]}" \
    || echo "tile $TILE_ID: run_eikonal_on_sfincs_subgrid.py failed (non-fatal - postprocess_tile_summary.py degrades gracefully on a missing method)" >&2
else
  echo "tile $TILE_ID: bathtub/eikonal not requested (--models=$MODELS), skipping"
fi

# -- 5. SFINCS run (direct apptainer, staged to local scratch) --
if ! $RUN_SFINCS; then
  echo "tile $TILE_ID: sfincs not requested (--models=$MODELS), skipping"
elif [ ! -f "$SFINCS_MODEL_DIR/sfincs_map.nc" ]; then
  LOCAL_DIR="${TMPDIR:-/tmp}/sfincs_${TILE_ID}_${SLURM_JOB_ID:-$$}"
  rm -rf "$LOCAL_DIR"; mkdir -p "$LOCAL_DIR"
  attempt=1
  while [ "$attempt" -le 5 ]; do
    cp -r "$SFINCS_MODEL_DIR"/. "$LOCAL_DIR"/ 2>/dev/null
    [ -f "$LOCAL_DIR/sfincs.inp" ] && break
    echo "  [stage retry $attempt/4] tile $TILE_ID input copy failed - retrying in 5s..." >&2
    sleep 5
    attempt=$((attempt + 1))
  done
  if [ ! -f "$LOCAL_DIR/sfincs.inp" ]; then
    echo "tile $TILE_ID: failed to stage input to node-local scratch after 5 attempts" >&2
    rm -rf "$LOCAL_DIR"
    exit 0
  fi

  export OMP_NUM_THREADS=4
  echo "=== tile $TILE_ID: starting sfincs (timeout ${SFINCS_TIMEOUT_S}s) ==="
  ( cd "$LOCAL_DIR" && timeout "$SFINCS_TIMEOUT_S" apptainer exec -B "$LOCAL_DIR":/mnt/data "$SFINCS_IMAGE" sfincs ) 2>&1 | tee "$LOCAL_DIR/sfincs_hpc_run.log"
  run_rc=${PIPESTATUS[0]}

  if [ "$run_rc" -ne 0 ] || [ ! -f "$LOCAL_DIR/sfincs_map.nc" ]; then
    echo "tile $TILE_ID: sfincs run failed (exit $run_rc) or produced no sfincs_map.nc" >&2
    cp -f "$LOCAL_DIR/sfincs_hpc_run.log" "$SFINCS_MODEL_DIR/" 2>/dev/null
    rm -rf "$LOCAL_DIR"
    exit 0
  fi

  cp -f "$LOCAL_DIR/sfincs_map.nc" "$SFINCS_MODEL_DIR/"
  cp -f "$LOCAL_DIR/sfincs.log" "$SFINCS_MODEL_DIR/" 2>/dev/null
  cp -f "$LOCAL_DIR/sfincs_hpc_run.log" "$SFINCS_MODEL_DIR/"
  rm -rf "$LOCAL_DIR"
  echo "tile $TILE_ID: sfincs run done"
fi

# -- 6. postprocess (hmax.tif/flood_extent.tif) + per-tile summary.json (hydromt-sfincs-dev env) --
# run_sfincs_tile.py --skip-run requires a real sfincs_map.nc; only called when one exists.
# postprocess_tile_summary.py always runs regardless of --models, writing null for any
# missing model's stats.
if [ -f "$SFINCS_MODEL_DIR/sfincs_map.nc" ]; then
  "$HYDROMT_SFINCS_DEV_PY" run_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --skip-run --base-dir-name "$BASE_DIR_NAME" \
    || echo "tile $TILE_ID: run_sfincs_tile.py postprocessing failed" >&2
else
  echo "tile $TILE_ID: no sfincs_map.nc yet - skipping run_sfincs_tile.py postprocessing (hmax.tif/flood_extent.tif)"
fi
"$HYDROMT_SFINCS_DEV_PY" postprocess_tile_summary.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || echo "tile $TILE_ID: postprocess_tile_summary.py failed" >&2

# Mark success LAST, only if sfincs_map.nc actually exists - overwrites any
# earlier failure status from a previous attempt now that this one made it
# all the way through (status is current-state, not history - see
# tile_status.py's own module docstring).
if [ -f "$SFINCS_MODEL_DIR/sfincs_map.nc" ]; then
  "$GFM_PY" "$TILE_STATUS_PY" write --root "$DATA_ROOT" --base-dir-name "$BASE_DIR_NAME" --tile-id "$TILE_ID" \
    --status ok --stage run_one_tile.sh --message "sfincs_map.nc present"
fi

echo "=== tile $TILE_ID: done ==="
