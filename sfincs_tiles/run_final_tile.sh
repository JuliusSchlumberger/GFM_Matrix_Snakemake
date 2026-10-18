#!/bin/bash
# Final SFINCS calibration (calibration_roughness_SFINCS), stage 2, ONE tile:
#
#   1. copy the production inputs from model_outputs/<tile>/inputs (copies, no
#      symlinks - the analysis reads them from Windows too) and the tile's
#      boundary line (boundary_lines/<tile>.gpkg, stage 1)
#   2. regenerate dem.tif/mask.tif (gfm), elevation_combined.tif + manning_n.tif
#      (hydromt-sfincs-dev)
#   3. forcing on the line's station nodes (build_line_forcing.py) and the SFINCS
#      model with the line's mask (build_sfincs_tile.py --boundary-lines-gpkg)
#   4. bathtub + eikonal friction sweep fsf 3..30 on the SFINCS subgrid
#      (run_calibration_models.py, solver settings from the config)
#   5. SFINCS on node-local scratch (apptainer)
#   6. hmax_subgrid.tif/hmax.tif (run_sfincs_tile.py --skip-run) and the merged
#      outputs/calibration_result.json (compute_calibration_result.py)
#
# Idempotent: every step is skipped when its output exists; a tile with
# outputs/calibration_run.done is skipped entirely. A failure appends to
# outputs/calibration_run.failed (removed on a later success) and moves on.
# outputs/calibration_run.step holds the step currently running
# (check_calibration_progress.py).
#
# Usage: BASE_DIR_NAME=calibration_roughness_SFINCS SFINCS_THREADS=4 bash run_final_tile.sh <tile_id>
set -uo pipefail
unset PROJ_LIB PROJ_DATA GDAL_DATA

TILE_ID="$1"
BASE_DIR_NAME="${BASE_DIR_NAME:-calibration_roughness_SFINCS}"
SFINCS_THREADS="${SFINCS_THREADS:-4}"

# Hydrax paths: single definition in hydrax_env.sh (env-overridable).
source "$(dirname "${BASH_SOURCE[0]}")/hydrax_env.sh"
CODE_ROOT="$GFM_CODE_ROOT"
DATA_ROOT="$GFM_DATA_ROOT"
CONFIG="$DATA_ROOT/$BASE_DIR_NAME/resolved_config.yml"
SFINCS_IMAGE="docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release"
SFINCS_TIMEOUT_S=14400
HYDROMT_SFINCS_DEV_PY="$GFM_SFINCS_PYTHON"
GFM_PY="$GFM_PYTHON"

SRC="$DATA_ROOT/model_outputs/$TILE_ID/inputs"
TILE_DIR="$DATA_ROOT/$BASE_DIR_NAME/$TILE_ID"
IN="$TILE_DIR/inputs"
SM="$TILE_DIR/sfincs_model"
OUT="$TILE_DIR/outputs"
LINES_SRC="$DATA_ROOT/$BASE_DIR_NAME/boundary_lines/$TILE_ID.gpkg"
LINES="$TILE_DIR/boundary_line_$TILE_ID.gpkg"
DONE="$OUT/calibration_run.done"
FAILED="$OUT/calibration_run.failed"
STEP="$OUT/calibration_run.step"

fail() {
  echo "tile $TILE_ID: $1" >&2
  mkdir -p "$OUT"
  echo "$(date '+%Y-%m-%d %H:%M:%S') job ${SLURM_JOB_ID:-local}: $1" >> "$FAILED"
  rm -f "$STEP"
  exit 0  # never abort the job's tile loop
}
step() { echo "$1" > "$STEP"; echo "--- tile $TILE_ID: $1 $(date '+%H:%M:%S')"; }

echo "=== tile $TILE_ID: final calibration run starting $(date '+%H:%M:%S') ==="
if [ -f "$DONE" ]; then
  echo "tile $TILE_ID: already done ($(cat "$DONE")) - skipping"
  exit 0
fi
[ -f "$LINES_SRC" ] || fail "no boundary line ($LINES_SRC)"
mkdir -p "$IN" "$SM" "$OUT" || fail "cannot create $TILE_DIR"

# -- 1. inputs + line --
step "copy_inputs"
for f in tile_geometry.gpkg model_bbox.json dem.tif mask.tif friction.tif boundaries_RP100_SLR_0.gpkg; do
  if [ ! -f "$IN/$f" ]; then
    [ -f "$SRC/$f" ] || fail "missing $SRC/$f"
    /bin/cp -f --remove-destination "$SRC/$f" "$IN/$f" </dev/null || fail "copying $f failed"
  fi
done
/bin/cp -f "$LINES_SRC" "$LINES" </dev/null || fail "copying the boundary line failed"

cd "$CODE_ROOT/sfincs_tiles" || fail "cannot cd to $CODE_ROOT/sfincs_tiles"

# -- 2. elevation / roughness (dem.tif/mask.tif are regenerated only before the first build) --
if [ ! -f "$SM/elevation_combined.tif" ]; then
  step "regenerate_dem_mask"
  "$GFM_PY" regenerate_dem_mask.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || echo "tile $TILE_ID: regenerate_dem_mask.py failed (non-fatal - keeping the model_outputs dem.tif/mask.tif)" >&2
  step "build_elevation"
  "$HYDROMT_SFINCS_DEV_PY" build_elevation.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || fail "build_elevation.py failed"
fi
if [ ! -f "$SM/manning_n.tif" ]; then
  step "build_roughness"
  "$HYDROMT_SFINCS_DEV_PY" build_roughness.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || fail "build_roughness.py failed"
fi

# -- 3. line forcing + SFINCS model --
if [ ! -f "$SM/sfincs.inp" ] || [ ! -f "$SM/subgrid/dep_subgrid.tif" ]; then
  step "build_line_forcing"
  "$HYDROMT_SFINCS_DEV_PY" build_line_forcing.py --tile-id "$TILE_ID" --base-dir-name "$BASE_DIR_NAME" \
    --lines-gpkg "$LINES" --config "$CONFIG" || fail "build_line_forcing.py failed"
  step "build_sfincs_tile"
  rm -f "$SM/sfincs_map.nc" "$SM/hmax_subgrid.tif"
  "$HYDROMT_SFINCS_DEV_PY" build_sfincs_tile.py --tile-id "$TILE_ID" --base-dir-name "$BASE_DIR_NAME" \
    --boundary-lines-gpkg "$LINES" --config "$CONFIG" || fail "build_sfincs_tile.py failed"
fi

# -- 4. bathtub + eikonal sweep (each fsf skipped when its raster exists) --
step "eikonal_sweep"
"$GFM_PY" run_calibration_models.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || fail "run_calibration_models.py failed"

# -- 5. SFINCS on node-local scratch --
if [ ! -f "$SM/sfincs_map.nc" ]; then
  step "sfincs_run"
  LOCAL_DIR="${TMPDIR:-/tmp}/sfincs_final_${TILE_ID}_${SLURM_JOB_ID:-$$}"
  rm -rf "$LOCAL_DIR"; mkdir -p "$LOCAL_DIR"
  staged=false
  for attempt in 1 2 3 4 5; do
    cp -rL "$SM"/. "$LOCAL_DIR"/ 2>/dev/null </dev/null && [ -f "$LOCAL_DIR/sfincs.inp" ] && { staged=true; break; }
    echo "  [stage retry $attempt/5] tile $TILE_ID input copy failed - retrying in 5s..." >&2
    sleep 5
  done
  $staged || { rm -rf "$LOCAL_DIR"; fail "failed to stage model to node-local scratch"; }
  rm -f "$LOCAL_DIR"/elevation_combined*.tif "$LOCAL_DIR"/manning_n*.tif  # prep rasters, not SFINCS inputs

  export OMP_NUM_THREADS="$SFINCS_THREADS"
  echo "tile $TILE_ID: starting sfincs with $OMP_NUM_THREADS threads (timeout ${SFINCS_TIMEOUT_S}s)"
  ( cd "$LOCAL_DIR" && timeout "$SFINCS_TIMEOUT_S" apptainer exec -B "$LOCAL_DIR":/mnt/data "$SFINCS_IMAGE" sfincs ) 2>&1 \
    | tee "$LOCAL_DIR/sfincs_hpc_run.log" | grep -v "% complete"
  run_rc=${PIPESTATUS[0]}
  cp -f "$LOCAL_DIR/sfincs_hpc_run.log" "$SM/" 2>/dev/null
  if [ "$run_rc" -ne 0 ] || [ ! -f "$LOCAL_DIR/sfincs_map.nc" ]; then
    rm -rf "$LOCAL_DIR"
    fail "sfincs run failed (exit $run_rc) or produced no sfincs_map.nc"
  fi
  cp -f "$LOCAL_DIR/sfincs_map.nc" "$SM/" || { rm -rf "$LOCAL_DIR"; fail "copying sfincs_map.nc back failed"; }
  cp -f "$LOCAL_DIR/sfincs.log" "$SM/" 2>/dev/null
  rm -rf "$LOCAL_DIR"
fi

# -- 6. postprocess + merged result --
if [ ! -f "$SM/hmax_subgrid.tif" ]; then
  step "sfincs_postprocess"
  "$HYDROMT_SFINCS_DEV_PY" run_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --skip-run --base-dir-name "$BASE_DIR_NAME" \
    || fail "run_sfincs_tile.py postprocessing failed"
fi
step "calibration_result"
"$HYDROMT_SFINCS_DEV_PY" compute_calibration_result.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || fail "compute_calibration_result.py failed"

rm -f "$FAILED" "$STEP"
date '+%Y-%m-%d %H:%M:%S' > "$DONE"
echo "=== tile $TILE_ID: done $(date '+%H:%M:%S') ==="
