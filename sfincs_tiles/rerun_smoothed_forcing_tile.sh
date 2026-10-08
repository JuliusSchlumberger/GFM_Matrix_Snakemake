#!/bin/bash
# Re-runs ONE already-built calibration tile's SFINCS model with the smoothed
# nearest-boundary forcing (boundary_forcing_smoothed.py) instead of hydromt's
# own water_level.create(buffer=...) station selection, then refreshes every
# downstream product in place: hmax_subgrid.tif/hmax.tif/flood_extent.tif
# (run_sfincs_tile.py --skip-run), summary_*.json (postprocess_tile_summary.py)
# and sweep_comparison_cache.json (tile_sweep_cache.py - rebuilt, not reused).
# Called per tile from each sbatch script written by
# generate_smoothed_forcing_rerun_jobs.py.
#
# Before anything is overwritten, the small per-tile files are copied once to
# *_origforcing.* next to the originals (never overwritten on a re-attempt);
# the old sfincs_map.nc is replaced.
#
# Idempotent: a tile with outputs/rerun_smoothed_forcing.done is skipped, so a
# resubmitted job only redoes unfinished tiles. A failure writes
# outputs/rerun_smoothed_forcing.failed (cleared on a later success) - the
# tile's hmax/summary/cache files then still hold the ORIGINAL-forcing results.
#
# Usage: BASE_DIR_NAME=sfincs_calibration SFINCS_THREADS=16 bash rerun_smoothed_forcing_tile.sh <tile_id>
set -uo pipefail
unset PROJ_LIB PROJ_DATA GDAL_DATA

TILE_ID="$1"
BASE_DIR_NAME="${BASE_DIR_NAME:-sfincs_calibration}"
SFINCS_THREADS="${SFINCS_THREADS:-16}"
MAX_OUTER_ITERATIONS=5
FRICTION_SCALE_FACTORS="3 6 9 12 15 18 21 24 27 30"

CODE_ROOT="/u/schlumbe/gfm_code"
DATA_ROOT="/p/11212688-004-global-floodmaps/modelling"
CONFIG="$DATA_ROOT/$BASE_DIR_NAME/resolved_config.yml"
SFINCS_IMAGE="docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release"
SFINCS_TIMEOUT_S=14400
HYDROMT_SFINCS_DEV_PY="/u/schlumbe/.conda/envs/hydromt-sfincs-dev/bin/python"
GFM_PY="/u/schlumbe/.conda/envs/gfm/bin/python"

TILE_DIR="$DATA_ROOT/$BASE_DIR_NAME/$TILE_ID"
SM="$TILE_DIR/sfincs_model"
OUT="$TILE_DIR/outputs"
DONE="$OUT/rerun_smoothed_forcing.done"
FAILED="$OUT/rerun_smoothed_forcing.failed"

fail() {
  echo "tile $TILE_ID: $1" >&2
  mkdir -p "$OUT"
  echo "$(date '+%Y-%m-%d %H:%M:%S') job ${SLURM_JOB_ID:-local}: $1" >> "$FAILED"
  exit 0  # never abort the batch loop
}

echo "=== tile $TILE_ID: smoothed-forcing rerun starting $(date '+%H:%M:%S') ==="
if [ -f "$DONE" ]; then
  echo "tile $TILE_ID: already done ($(cat "$DONE")) - skipping"
  exit 0
fi
if "$GFM_PY" "$CODE_ROOT/sfincs_tiles/tile_status.py" check --root "$DATA_ROOT" --base-dir-name "$BASE_DIR_NAME" --tile-id "$TILE_ID" >/dev/null; then
  echo "tile $TILE_ID: permanently unsolvable per tile_status.json - skipping"
  exit 0
fi
[ -f "$SM/sfincs.inp" ] || fail "no built SFINCS model ($SM/sfincs.inp missing) - skipping"
[ -f "$SM/sfincs.bzs" ] || [ -f "$SM/sfincs_origforcing.bzs" ] || fail "no sfincs.bzs to take the run window from"

cd "$CODE_ROOT/sfincs_tiles" || fail "cannot cd to $CODE_ROOT/sfincs_tiles"

# -- 1. back up the small original-forcing files once (never overwrite a backup) --
backup() {  # <src> -> <stem>_origforcing.<ext> in the same directory
  local src="$1" dir base stem ext dst
  [ -f "$src" ] || return 0
  dir=$(dirname "$src"); base=$(basename "$src"); stem="${base%.*}"; ext="${base##*.}"
  dst="$dir/${stem}_origforcing.$ext"
  [ -f "$dst" ] || cp -p "$src" "$dst"
}
for f in sfincs.bnd sfincs.bzs sfincs.log sfincs_hpc_run.log hmax_subgrid.tif gis/bnd.geojson; do backup "$SM/$f"; done
for f in hmax.tif flood_extent.tif summary_sfincs.json summary_eikonal.json summary_bathtub.json sweep_comparison_cache.json; do
  backup "$OUT/$f"
done

# -- 2. smoothed nearest-boundary forcing (rewrites sfincs.bnd/sfincs.bzs/gis/bnd.geojson) --
"$GFM_PY" boundary_forcing_smoothed.py --tile-dir "$TILE_DIR" --reference-bzs "$SM/sfincs_origforcing.bzs" \
  || fail "boundary_forcing_smoothed.py failed"

# -- 3. SFINCS run (same container/staging as run_one_tile.sh step 5, $SFINCS_THREADS threads) --
rm -f "$SM/sfincs_map.nc"
LOCAL_DIR="${TMPDIR:-/tmp}/sfincs_rerun_${TILE_ID}_${SLURM_JOB_ID:-$$}"
rm -rf "$LOCAL_DIR"; mkdir -p "$LOCAL_DIR"
staged=false
for attempt in 1 2 3 4 5; do
  cp -r "$SM"/. "$LOCAL_DIR"/ 2>/dev/null && [ -f "$LOCAL_DIR/sfincs.inp" ] && { staged=true; break; }
  echo "  [stage retry $attempt/5] tile $TILE_ID input copy failed - retrying in 5s..." >&2
  sleep 5
done
$staged || { rm -rf "$LOCAL_DIR"; fail "failed to stage model to node-local scratch"; }
# backups and old outputs are not SFINCS inputs - drop them from the staged copy
rm -f "$LOCAL_DIR"/*_origforcing.* "$LOCAL_DIR"/gis/*_origforcing.* "$LOCAL_DIR"/hmax_subgrid.tif

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

# -- 4. postprocess: hmax_subgrid.tif/hmax.tif/flood_extent.tif + summary_*.json --
"$HYDROMT_SFINCS_DEV_PY" run_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --skip-run --base-dir-name "$BASE_DIR_NAME" \
  || fail "run_sfincs_tile.py postprocessing failed"
"$HYDROMT_SFINCS_DEV_PY" postprocess_tile_summary.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || fail "postprocess_tile_summary.py failed"

# -- 5. rebuild the eikonal-sweep/bathtub-vs-SFINCS comparison cache from the new hmax_subgrid.tif --
"$GFM_PY" tile_sweep_cache.py --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" --tile-id "$TILE_ID" \
  --friction-scale-factors $FRICTION_SCALE_FACTORS --max-outer-iterations "$MAX_OUTER_ITERATIONS" \
  || fail "tile_sweep_cache.py failed"

rm -f "$FAILED"
echo "$(date '+%Y-%m-%d %H:%M:%S') job ${SLURM_JOB_ID:-local}, ${SFINCS_THREADS} threads" > "$DONE"
echo "=== tile $TILE_ID: smoothed-forcing rerun done $(date '+%H:%M:%S') ==="
