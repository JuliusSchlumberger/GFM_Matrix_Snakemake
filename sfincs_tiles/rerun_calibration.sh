#!/bin/bash
# Rerun part of the final calibration (calibration_roughness_SFINCS) in place on Hydrax, after
# syncing the code to $GFM_CODE_ROOT. Replaces rerun_eikonal_calibration.sh and
# rerun_sfincs_calibration.sh (both kept as thin wrappers around this script).
#
#   --part eikonal  after a change to the eikonal side (run_calibration_models.py /
#                   src/flood_model.py). Per selected tile (tile_ids.txt) removes the
#                   done/failed/step markers plus every eikonal_on_subgrid_waterdepth_*_fsf*.tif
#                   and eikonal_solver_diagnostics.json; run_final_tile.sh then reruns only the
#                   eikonal sweep -> calibration_result.json. Submits the stage-2 jobs, then
#                   (POST=1) one post job (afterany) that runs compute_calibration_result_dedup.py
#                   and run_calibration_sweep_analysis.py for both base directories.
#   --part sfincs   after a change to the SFINCS forcing/mask code (build_line_forcing.py,
#                   build_sfincs_tile.py, connected_forcing.py). Removes the markers and
#                   sfincs.inp, so run_final_tile.sh rebuilds line forcing -> SFINCS model ->
#                   SFINCS run -> postprocessing -> calibration_result.json. Run the dedup and the
#                   sweep analysis by hand afterwards.
#
# Usage:
#   bash sfincs_tiles/rerun_calibration.sh --part eikonal        # reset + 20 jobs + post job
#   N_JOBS=40 bash sfincs_tiles/rerun_calibration.sh --part eikonal
#   POST=0 bash sfincs_tiles/rerun_calibration.sh --part eikonal # no post job
#   bash sfincs_tiles/rerun_calibration.sh --part sfincs
#   DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part <part>   # only report what would be reset
# Paths come from hydrax_env.sh (override e.g. GFM_PYTHON=... in the environment).

set -euo pipefail

PART=""
case "${1:-}" in
  --part) PART="${2:-}" ;;
  --part=*) PART="${1#--part=}" ;;
esac
if [ "$PART" != "eikonal" ] && [ "$PART" != "sfincs" ]; then
  echo "usage: bash $0 --part eikonal|sfincs" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/hydrax_env.sh"

BASE_DIR_NAME="${BASE_DIR_NAME:-calibration_roughness_SFINCS}"
N_JOBS="${N_JOBS:-20}"
DRY_RUN="${DRY_RUN:-0}"
BASE="$GFM_DATA_ROOT/$BASE_DIR_NAME"
CODE="$GFM_CODE_ROOT/sfincs_tiles"
GFM_PY="$GFM_PYTHON"

if [ "$PART" = "eikonal" ]; then
  DEDUP_DIR_NAME="${DEDUP_DIR_NAME:-${BASE_DIR_NAME}_dedup}"
  POST="${POST:-1}"
  HYDROMT_PY="$GFM_SFINCS_PYTHON"
  CONFIG="$BASE/resolved_config.yml"

  [ -f "$BASE/tile_ids.txt" ] || { echo "no $BASE/tile_ids.txt" >&2; exit 1; }
  grep -q "def solver_inputs" "$CODE/run_calibration_models.py" \
    || { echo "$CODE/run_calibration_models.py has no solver_inputs - sync the code first" >&2; exit 1; }

  n=0
  n_missing=0
  n_rasters=0
  while read -r t; do
    [ -n "$t" ] || continue
    if [ ! -d "$BASE/$t" ]; then
      n_missing=$((n_missing + 1))
      continue
    fi
    out="$BASE/$t/outputs"
    k=$(find "$out" -maxdepth 1 -name 'eikonal_on_subgrid_waterdepth_*_fsf*.tif' 2>/dev/null | wc -l)
    n_rasters=$((n_rasters + k))
    n=$((n + 1))
    [ "$DRY_RUN" = "1" ] && continue
    rm -f "$out/calibration_run.done" "$out/calibration_run.failed" "$out/calibration_run.step" \
          "$out"/eikonal_on_subgrid_waterdepth_*_fsf*.tif "$out/eikonal_solver_diagnostics.json"
  done < "$BASE/tile_ids.txt"

  if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: $n tile(s) would be reset, $n_rasters eikonal raster(s) removed ($n_missing listed without a tile directory)"
    exit 0
  fi
  echo "reset $n tile(s), removed $n_rasters eikonal raster(s) ($n_missing listed without a tile directory)"

  # SFINCS is skipped (sfincs_map.nc/hmax_subgrid.tif exist): only the eikonal sweep + result JSON
  cd "$CODE"
  JOBS="$BASE/hpc_jobs_stage2"
  rm -f "$JOBS"/final_*.sbatch "$JOBS"/tiles_*.txt  # stale job files of an earlier run would be submitted below
  "$GFM_PY" generate_calibration_jobs.py stage2 --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
      --n-jobs "$N_JOBS" --sfincs-min-per-mcell 0 --overhead-min 5

  ids=()
  for f in "$JOBS"/final_*.sbatch; do
    ids+=("$(sbatch --parsable "$f")")
  done
  echo "submitted ${#ids[@]} eikonal rerun job(s): ${ids[*]}"

  if [ "$POST" = "1" ]; then
    post="$JOBS/post_eikonal_rerun.sbatch"
    cat > "$post" <<EOF
#!/bin/bash
#SBATCH --job-name=cal_post_eikonal
#SBATCH --partition=4vcpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=0-12:00:00
#SBATCH --output=$JOBS/logs/cal_post_eikonal_%j.out
#SBATCH --error=$JOBS/logs/cal_post_eikonal_%j.err

set -uo pipefail
unset PROJ_LIB PROJ_DATA GDAL_DATA
cd $CODE || exit 1
$GFM_PY check_calibration_progress.py --config $CONFIG --base-dir-name $BASE_DIR_NAME
$HYDROMT_PY compute_calibration_result_dedup.py --config $CONFIG --base-dir-name $BASE_DIR_NAME \\
    --out-dir-name $DEDUP_DIR_NAME --workers 4 || exit 1
$GFM_PY run_calibration_sweep_analysis.py --config $CONFIG --base-dir-name $BASE_DIR_NAME
$GFM_PY run_calibration_sweep_analysis.py --config $CONFIG --base-dir-name $DEDUP_DIR_NAME \\
    --raster-base-dir-name $BASE_DIR_NAME
EOF
    dep=$(IFS=:; echo "${ids[*]}")
    pid=$(sbatch --parsable --dependency=afterany:"$dep" "$post")
    echo "post job (dedup + both analyses, after all of the above): $pid"
  fi
  echo "progress: $GFM_PY $CODE/check_calibration_progress.py --config $CONFIG --base-dir-name $BASE_DIR_NAME"
else
  [ -f "$BASE/tile_ids.txt" ] || { echo "no $BASE/tile_ids.txt" >&2; exit 1; }
  [ -f "$CODE/connected_forcing.py" ] || { echo "$CODE/connected_forcing.py missing - sync the code first" >&2; exit 1; }

  n=0
  n_missing=0
  while read -r t; do
    [ -n "$t" ] || continue
    if [ ! -d "$BASE/$t" ]; then
      n_missing=$((n_missing + 1))
      continue
    fi
    if [ "$DRY_RUN" = "1" ]; then
      n=$((n + 1))
      continue
    fi
    rm -f "$BASE/$t/outputs/calibration_run.done" "$BASE/$t/outputs/calibration_run.failed" \
          "$BASE/$t/outputs/calibration_run.step" "$BASE/$t/sfincs_model/sfincs.inp"
    n=$((n + 1))
  done < "$BASE/tile_ids.txt"

  if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: $n tile(s) would be reset ($n_missing listed without a tile directory)"
    exit 0
  fi
  echo "reset $n tile(s) ($n_missing listed without a tile directory)"

  # the eikonal sweep is skipped per friction factor (rasters exist), so its time estimate is ~0
  cd "$CODE"
  "$GFM_PY" generate_calibration_jobs.py stage2 --config "$BASE/resolved_config.yml" \
      --base-dir-name "$BASE_DIR_NAME" --n-jobs "$N_JOBS" --eikonal-min-per-mcell 1 --submit
  echo "progress: $GFM_PY $CODE/check_calibration_progress.py --config $BASE/resolved_config.yml --base-dir-name $BASE_DIR_NAME"
fi
