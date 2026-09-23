#!/bin/bash
# Per-tile pipeline for the v2 validation batch: copy eikonal inputs -> build
# SFINCS inputs (hydromt-sfincs-dev env) -> bathtub+eikonal (gfm env) -> SFINCS
# run (direct apptainer, staged to local scratch - same pattern this
# session's own sfincs_batch_*.sbatch already used and proved) -> postprocess
# + summary (hydromt-sfincs-dev env). Called per-tile from each batch
# sbatch script's own loop - see generate_v2_batch_jobs.py.
#
# Idempotent at every stage (checks for the expected output file before
# redoing work), so re-running after a partial batch failure only redoes
# what's actually missing.
#
# Usage: run_one_tile_v2.sh <tile_id> <A|B>
set -uo pipefail

TILE_ID="$1"
TILE_SET="$2"

CODE_ROOT="/u/schlumbe/gfm_code"
DATA_ROOT="/p/11212688-004-global-floodmaps/modelling"
BASE_DIR_NAME="validation_sfincs_v2"
CONFIG="$DATA_ROOT/$BASE_DIR_NAME/resolved_config.yml"
SFINCS_IMAGE="docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release"
SFINCS_TIMEOUT_S=14400

# Full python binary paths, not `conda activate` - confirmed live 2026-09-23:
# `module load miniconda && eval "$(conda shell.bash hook)" && conda activate
# hydromt-sfincs-dev` silently stayed in (base) when run inside this script's
# own non-interactive `bash run_one_tile_v2.sh` subshell (plain `bash
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

# -- 2. build SFINCS inputs (hydromt-sfincs-dev env) --
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

# -- 3. bathtub + eikonal (gfm env - needs src/flood_model.py's older-hydromt import chain) --
"$GFM_PY" run_eikonal_on_sfincs_subgrid.py --tile-id "$TILE_ID" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || log_fail "run_eikonal_on_sfincs_subgrid.py failed (non-fatal - postprocess_tile_summary.py degrades gracefully on a missing method)"

# -- 4. SFINCS run (direct apptainer, staged to local scratch) --
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

# -- 5. postprocess (hmax.tif/flood_extent.tif) + per-tile summary.json (hydromt-sfincs-dev env) --
"$HYDROMT_SFINCS_DEV_PY" run_sfincs_tile.py --tile-id "$TILE_ID" --config "$CONFIG" --skip-run --base-dir-name "$BASE_DIR_NAME" \
  || log_fail "run_sfincs_tile.py postprocessing failed"
"$HYDROMT_SFINCS_DEV_PY" postprocess_tile_summary.py --tile-id "$TILE_ID" --set "$TILE_SET" --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
  || log_fail "postprocess_tile_summary.py failed"

echo "=== tile $TILE_ID (set $TILE_SET): done ==="
