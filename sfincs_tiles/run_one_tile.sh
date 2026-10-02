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
FAIL_LOG="$DATA_ROOT/$BASE_DIR_NAME/hpc_jobs/logs/run_one_tile_failures.txt"

log_fail() { echo "$TILE_ID  $1" >> "$FAIL_LOG"; }

echo "=== tile $TILE_ID: starting ==="

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

# -- 2. regenerate dem.tif/mask.tif via extract_dem/extract_dem_mask (gfm env
# - needs src/config_utils.py's hydromt.DataCatalog). Gated on the same
# elevation_combined.tif check as step 3: dem.tif/mask.tif must not change
# once the SFINCS-input build has started. --
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
PY_MODELS=()
$RUN_BATHTUB && PY_MODELS+=(bathtub)
$RUN_EIKONAL && PY_MODELS+=(eikonal)
if [ "${#PY_MODELS[@]}" -gt 0 ]; then
  MAX_ROUNDS_ARGS=()
  [ -n "$MAX_ROUNDS" ] && MAX_ROUNDS_ARGS=(--max-rounds "$MAX_ROUNDS")
  "$GFM_PY" run_eikonal_on_sfincs_subgrid.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
    --models "${PY_MODELS[@]}" "${MAX_ROUNDS_ARGS[@]}" \
    || log_fail "run_eikonal_on_sfincs_subgrid.py failed (non-fatal - postprocess_tile_summary.py degrades gracefully on a missing method)"
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
# run_sfincs_tile.py --skip-run requires a real sfincs_map.nc; only called when one exists.
# postprocess_tile_summary.py always runs regardless of --models, writing null for any
# missing model's stats.
if [ -f "$SFINCS_MODEL_DIR/sfincs_map.nc" ]; then
  "$HYDROMT_SFINCS_DEV_PY" run_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --skip-run --base-dir-name "$BASE_DIR_NAME" \
    || log_fail "run_sfincs_tile.py postprocessing failed"
else
  echo "tile $TILE_ID: no sfincs_map.nc yet - skipping run_sfincs_tile.py postprocessing (hmax.tif/flood_extent.tif)"
fi
"$HYDROMT_SFINCS_DEV_PY" postprocess_tile_summary.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || log_fail "postprocess_tile_summary.py failed"

echo "=== tile $TILE_ID: done ==="
