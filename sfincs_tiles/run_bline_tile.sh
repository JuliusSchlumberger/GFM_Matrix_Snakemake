#!/bin/bash
# Builds and runs ONE calibration tile's SFINCS model on its station boundary
# line (build_station_boundary_lines.py) into a separate base dir, then
# produces the same downstream products as the smoothed-forcing rerun
# (rerun_smoothed_forcing_tile.sh): hmax_subgrid.tif/hmax.tif/flood_extent.tif
# (run_sfincs_tile.py --skip-run), summary_{sfincs,eikonal,bathtub}.json
# (postprocess_tile_summary.py) and sweep_comparison_cache.json
# (tile_sweep_cache.py) - so the whole calibration analysis
# (run_calibration_sweep_analysis.py --base-dir-name $BASE_DIR_NAME) runs on it.
#
#   - model: active cells = tile minus the line's inactive (ocean-side) faces,
#     waterlevel boundary along the line (build_sfincs_tile.py --boundary-lines-gpkg);
#   - forcing: one point per station node on the line, its own COAST-HG
#     hydrograph scaled to its COAST-RP RP100 level (build_line_forcing.py).
#
# Nothing in $SRC_BASE_DIR_NAME is modified: the tile's inputs/, the prepared
# elevation_combined.tif/manning_n.tif and the eikonal/bathtub rasters
# (computed on the same SFINCS subgrid - the grid itself does not change with
# the mask) are symlinked; the boundary line gpkg is copied (provenance - a
# later hand edit of the source does not silently change a finished run).
#
# Idempotent: a tile with outputs/bline_run.done is skipped. A failure writes
# outputs/bline_run.failed (cleared on a later success).
#
# Usage: BASE_DIR_NAME=sfincs_calibration_bline SRC_BASE_DIR_NAME=sfincs_calibration \
#        SFINCS_THREADS=4 bash run_bline_tile.sh <tile_id>
set -uo pipefail
unset PROJ_LIB PROJ_DATA GDAL_DATA

TILE_ID="$1"
BASE_DIR_NAME="${BASE_DIR_NAME:-sfincs_calibration_bline}"
SRC_BASE_DIR_NAME="${SRC_BASE_DIR_NAME:-sfincs_calibration}"
SFINCS_THREADS="${SFINCS_THREADS:-4}"
MAX_OUTER_ITERATIONS=5
FRICTION_SCALE_FACTORS="3 6 9 12 15 18 21 24 27 30"

CODE_ROOT="/u/schlumbe/gfm_code"
DATA_ROOT="/p/11212688-004-global-floodmaps/modelling"
CONFIG="$DATA_ROOT/$SRC_BASE_DIR_NAME/resolved_config.yml"
SFINCS_IMAGE="docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release"
SFINCS_TIMEOUT_S=14400
HYDROMT_SFINCS_DEV_PY="/u/schlumbe/.conda/envs/hydromt-sfincs-dev/bin/python"
GFM_PY="/u/schlumbe/.conda/envs/gfm/bin/python"

SRC="$DATA_ROOT/$SRC_BASE_DIR_NAME/$TILE_ID"
TILE_DIR="$DATA_ROOT/$BASE_DIR_NAME/$TILE_ID"
SM="$TILE_DIR/sfincs_model"
OUT="$TILE_DIR/outputs"
LINES_SRC="$DATA_ROOT/$SRC_BASE_DIR_NAME/boundary_lines/$TILE_ID.gpkg"
LINES="$TILE_DIR/boundary_line_$TILE_ID.gpkg"
DONE="$OUT/bline_run.done"
FAILED="$OUT/bline_run.failed"

fail() {
  echo "tile $TILE_ID: $1" >&2
  mkdir -p "$OUT"
  echo "$(date '+%Y-%m-%d %H:%M:%S') job ${SLURM_JOB_ID:-local}: $1" >> "$FAILED"
  exit 0  # never abort the batch loop
}

echo "=== tile $TILE_ID: boundary-line run starting $(date '+%H:%M:%S') ==="
if [ -f "$DONE" ]; then
  echo "tile $TILE_ID: already done ($(cat "$DONE")) - skipping"
  exit 0
fi
[ -f "$LINES_SRC" ] || fail "no boundary line ($LINES_SRC)"
[ -f "$SRC/sfincs_model/elevation_combined.tif" ] || fail "no prepared elevation_combined.tif in $SRC/sfincs_model"

# -- 1. tile dir: symlinked inputs / prepared rasters / eikonal + bathtub rasters, copied line --
mkdir -p "$TILE_DIR/inputs" "$SM" "$OUT" || fail "cannot create $TILE_DIR"
for f in "$SRC"/inputs/*; do ln -sfn "$f" "$TILE_DIR/inputs/$(basename "$f")"; done
for f in elevation_combined.tif manning_n.tif; do ln -sfn "$SRC/sfincs_model/$f" "$SM/$f"; done
for f in "$SRC"/outputs/eikonal_on_subgrid_waterdepth_*.tif "$SRC"/outputs/bathtub_waterdepth_*.tif; do
  [ -f "$f" ] && ln -sfn "$f" "$OUT/$(basename "$f")"
done
cp -f "$LINES_SRC" "$LINES" || fail "copying the boundary line failed"

cd "$CODE_ROOT/sfincs_tiles" || fail "cannot cd to $CODE_ROOT/sfincs_tiles"

# -- 2. forcing on the line's station nodes (matched_boundary_points.gpkg + corrected_hydrographs.csv) --
"$HYDROMT_SFINCS_DEV_PY" build_line_forcing.py --tile-id "$TILE_ID" --base-dir-name "$BASE_DIR_NAME" \
  --lines-gpkg "$LINES" --config "$CONFIG" || fail "build_line_forcing.py failed"

# -- 3. SFINCS model: boundary-line mask + forcing --
"$HYDROMT_SFINCS_DEV_PY" build_sfincs_tile.py --tile-id "$TILE_ID" --base-dir-name "$BASE_DIR_NAME" \
  --boundary-lines-gpkg "$LINES" --config "$CONFIG" || fail "build_sfincs_tile.py failed"

# -- 4. SFINCS run on node-local scratch --
rm -f "$SM/sfincs_map.nc"
LOCAL_DIR="${TMPDIR:-/tmp}/sfincs_bline_${TILE_ID}_${SLURM_JOB_ID:-$$}"
rm -rf "$LOCAL_DIR"; mkdir -p "$LOCAL_DIR"
staged=false
for attempt in 1 2 3 4 5; do
  cp -rL "$SM"/. "$LOCAL_DIR"/ 2>/dev/null && [ -f "$LOCAL_DIR/sfincs.inp" ] && { staged=true; break; }
  echo "  [stage retry $attempt/5] tile $TILE_ID input copy failed - retrying in 5s..." >&2
  sleep 5
done
$staged || { rm -rf "$LOCAL_DIR"; fail "failed to stage model to node-local scratch"; }
# prep rasters are not SFINCS inputs
rm -f "$LOCAL_DIR"/elevation_combined*.tif "$LOCAL_DIR"/manning_n*.tif

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

# -- 5. postprocess: hmax_subgrid.tif/hmax.tif/flood_extent.tif + summary_{sfincs,eikonal,bathtub}.json --
"$HYDROMT_SFINCS_DEV_PY" run_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --skip-run --base-dir-name "$BASE_DIR_NAME" \
  || fail "run_sfincs_tile.py postprocessing failed"
"$HYDROMT_SFINCS_DEV_PY" postprocess_tile_summary.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || fail "postprocess_tile_summary.py failed"

# -- 6. eikonal-sweep/bathtub-vs-SFINCS comparison cache from the new hmax_subgrid.tif --
"$GFM_PY" tile_sweep_cache.py --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" --tile-id "$TILE_ID" \
  --friction-scale-factors $FRICTION_SCALE_FACTORS --max-outer-iterations "$MAX_OUTER_ITERATIONS" \
  || fail "tile_sweep_cache.py failed"

rm -f "$FAILED"
date '+%Y-%m-%d %H:%M:%S' > "$DONE"
echo "=== tile $TILE_ID: done $(date '+%H:%M:%S') ==="
