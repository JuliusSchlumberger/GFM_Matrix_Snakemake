#!/bin/bash
# Per-tile pipeline for the v3 validation batch - identical to
# run_one_tile_v2.sh except BASE_DIR_NAME points at a fresh output tree
# (validation_sfincs_v3), so every tile rebuilds from scratch under today's
# code instead of hitting any of v2's per-stage "output already exists"
# skip-checks with pre-fix (2026-09-23) outputs. In particular this picks up:
#   - src/rasters.py's GEBCO-based deltadtm_mask nodata (255) resolution
#     (resolve_offshore_mask_gaps_via_gebco) - fixes far-offshore cells
#     wrongly classified as land, which starved hydromt_sfincs's
#     create_boundary() of most of the real coastline on elongated tiles
#     (confirmed live on tile 1273: boundary cells went from 332 at a
#     single row to 1347 spanning 85% of the grid height).
#   - the NaN-vs-literal-nodata DeltaDTM fix and model_outputs/ -> per-tile
#     base-dir-name read-path fixes (build_elevation.py/build_roughness.py/
#     build_boundary_forcing.py/build_sfincs_tile.py/run_sfincs_tile.py).
#   - build_elevation.py's -20m subgrid-clamp land floor and lake/river
#     elevation re-derivation.
# See run_one_tile_v2.sh for the full per-stage pipeline commentary this
# mirrors: copy eikonal inputs -> regenerate dem.tif/mask.tif (gfm env) ->
# build SFINCS inputs (hydromt-sfincs-dev env) -> bathtub+eikonal (gfm env)
# -> SFINCS run (direct apptainer, staged to local scratch) -> postprocess +
# summary (hydromt-sfincs-dev env). Called per-tile from each batch sbatch
# script's own loop - see generate_v2_batch_jobs.py --base-dir-name
# validation_sfincs_v3 --runner-script-name run_one_tile_v3.sh.
#
# Idempotent at every stage (checks for the expected output file before
# redoing work), so re-running after a partial batch failure only redoes
# what's actually missing - safe because validation_sfincs_v3/ starts empty,
# so every one of those checks starts out false for every tile.
#
# Usage: run_one_tile_v3.sh <tile_id> <A|B>
set -uo pipefail

# Clear any PROJ_LIB/PROJ_DATA/GDAL_DATA inherited from whichever conda env
# happened to be active in the parent interactive shell (confirmed live
# 2026-09-23: a leaked PROJ_LIB from an earlier `conda activate gfm` pointed
# hydromt-sfincs-dev's own python at the gfm env's older/incompatible
# proj.db - "DATABASE.LAYOUT.VERSION.MINOR = 2 whereas a number >= 6 is
# expected. It comes from another PROJ installation."). Calling each env's
# python binary directly (see HYDROMT_SFINCS_DEV_PY/GFM_PY below) means we
# don't get `conda activate`'s own automatic env-var reset, so this has to
# be done explicitly. Unset (not hardcoded to some path) so each package
# falls back to its own bundled default, relative to whichever python
# binary actually ran it.
unset PROJ_LIB PROJ_DATA GDAL_DATA

TILE_ID="$1"
TILE_SET="$2"

CODE_ROOT="/u/schlumbe/gfm_code"
DATA_ROOT="/p/11212688-004-global-floodmaps/modelling"
BASE_DIR_NAME="validation_sfincs_v3"
CONFIG="$DATA_ROOT/$BASE_DIR_NAME/resolved_config.yml"
SFINCS_IMAGE="docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release"
SFINCS_TIMEOUT_S=14400

# Full python binary paths, not `conda activate` - confirmed live 2026-09-23:
# `module load miniconda && eval "$(conda shell.bash hook)" && conda activate
# hydromt-sfincs-dev` silently stayed in (base) when run inside this script's
# own non-interactive `bash run_one_tile_v3.sh` subshell (plain `bash
# script.sh` doesn't source ~/.bashrc, so the `module` function - normally
# defined there - didn't exist in this child shell), producing
# `ModuleNotFoundError: No module named 'hydromt_sfincs'` despite the env
# genuinely existing (`conda env list` confirmed it). Calling each env's own
# python binary directly sidesteps all of that - also the safer pattern for
# an unattended sbatch batch job, where the same activation fragility would
# otherwise bite identically.
HYDROMT_SFINCS_DEV_PY="/u/schlumbe/.conda/envs/hydromt-sfincs-dev/bin/python"
GFM_PY="/u/schlumbe/.conda/envs/gfm/bin/python"

TILE_DIR="$DATA_ROOT/$BASE_DIR_NAME/$TILE_ID"
MODEL_OUTPUTS_DIR="$DATA_ROOT/model_outputs/$TILE_ID/inputs"
INPUTS_DIR="$TILE_DIR/inputs"
SFINCS_MODEL_DIR="$TILE_DIR/sfincs_model"
FAIL_LOG="$DATA_ROOT/$BASE_DIR_NAME/hpc_jobs/logs/run_one_tile_failures.txt"

log_fail() { echo "$TILE_ID  $1" >> "$FAIL_LOG"; }

echo "=== tile $TILE_ID (set $TILE_SET): starting ==="

# -- 1. copy eikonal inputs (no env needed - plain files, already built by production preprocessing) --
mkdir -p "$INPUTS_DIR"
for f in tile_geometry.gpkg model_bbox.json dem.tif mask.tif friction.tif boundaries_RP100_SLR_0.gpkg; do
  if [ ! -f "$INPUTS_DIR/$f" ]; then
    if [ ! -f "$MODEL_OUTPUTS_DIR/$f" ]; then
      log_fail "missing $MODEL_OUTPUTS_DIR/$f - skipping tile"
      exit 0
    fi
    cp -f "$MODEL_OUTPUTS_DIR/$f" "$INPUTS_DIR/$f"
  fi
done

cd "$CODE_ROOT/sfincs_tiles"

# -- 2. regenerate dem.tif/mask.tif with current extract_dem/extract_dem_mask
# logic (gfm env - needs src/config_utils.py's hydromt.DataCatalog, same
# constraint as run_eikonal_on_sfincs_subgrid.py below). Gated on the same
# elevation_combined.tif check as step 3 - only needs doing once, and once
# the SFINCS-input build has started, dem.tif/mask.tif must not change under it. --
if [ ! -f "$SFINCS_MODEL_DIR/elevation_combined.tif" ]; then
  "$GFM_PY" regenerate_dem_mask.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || log_fail "regenerate_dem_mask.py failed (non-fatal - falling back to the copied model_outputs/ dem.tif/mask.tif)"
fi

# -- 3. build SFINCS inputs (hydromt-sfincs-dev env) --
if [ ! -f "$SFINCS_MODEL_DIR/elevation_combined.tif" ]; then
  "$HYDROMT_SFINCS_DEV_PY" build_elevation.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || { log_fail "build_elevation.py failed"; exit 0; }
fi
if [ ! -f "$SFINCS_MODEL_DIR/manning_n.tif" ]; then
  "$HYDROMT_SFINCS_DEV_PY" build_roughness.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || { log_fail "build_roughness.py failed"; exit 0; }
fi
if [ ! -f "$SFINCS_MODEL_DIR/matched_boundary_points.gpkg" ]; then
  "$HYDROMT_SFINCS_DEV_PY" build_boundary_forcing.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || { log_fail "build_boundary_forcing.py failed"; exit 0; }
fi
if [ ! -f "$SFINCS_MODEL_DIR/sfincs.inp" ]; then
  "$HYDROMT_SFINCS_DEV_PY" build_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    || { log_fail "build_sfincs_tile.py failed"; exit 0; }
fi

# -- 4. bathtub + eikonal (gfm env - needs src/flood_model.py's older-hydromt import chain) --
"$GFM_PY" run_eikonal_on_sfincs_subgrid.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || log_fail "run_eikonal_on_sfincs_subgrid.py failed (non-fatal - postprocess_tile_summary.py degrades gracefully on a missing method)"

# -- 5. SFINCS run (direct apptainer, staged to local scratch) --
if [ ! -f "$SFINCS_MODEL_DIR/sfincs_map.nc" ]; then
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
    log_fail "failed to stage input to node-local scratch after 5 attempts"
    rm -rf "$LOCAL_DIR"
    exit 0
  fi

  export OMP_NUM_THREADS=4
  echo "=== tile $TILE_ID: starting sfincs (timeout ${SFINCS_TIMEOUT_S}s) ==="
  ( cd "$LOCAL_DIR" && timeout "$SFINCS_TIMEOUT_S" apptainer exec -B "$LOCAL_DIR":/mnt/data "$SFINCS_IMAGE" sfincs ) 2>&1 | tee "$LOCAL_DIR/sfincs_hpc_run.log"
  run_rc=${PIPESTATUS[0]}

  if [ "$run_rc" -ne 0 ] || [ ! -f "$LOCAL_DIR/sfincs_map.nc" ]; then
    log_fail "sfincs run failed (exit $run_rc) or produced no sfincs_map.nc"
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
"$HYDROMT_SFINCS_DEV_PY" run_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --skip-run --base-dir-name "$BASE_DIR_NAME" \
  || log_fail "run_sfincs_tile.py postprocessing failed"
"$HYDROMT_SFINCS_DEV_PY" postprocess_tile_summary.py --tile-id "$TILE_ID" --set "$TILE_SET" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || log_fail "postprocess_tile_summary.py failed"

echo "=== tile $TILE_ID (set $TILE_SET): done ==="
