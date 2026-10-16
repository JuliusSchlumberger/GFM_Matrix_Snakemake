#!/bin/bash
# Rerun only the eikonal friction sweep of the final calibration (calibration_roughness_SFINCS) in
# place, after a change to the eikonal side (run_calibration_models.py / src/flood_model.py - e.g.
# the no-data frame fix, solver_inputs). Run on Hydrax, after syncing the code to /u/schlumbe/gfm_code.
#
# Per selected tile (tile_ids.txt) it removes the done/failed/step markers - which re-queues the
# tile - plus every eikonal_on_subgrid_waterdepth_*_fsf*.tif and eikonal_solver_diagnostics.json.
# run_final_tile.sh then skips everything that still exists (inputs, elevation/roughness, SFINCS
# model, sfincs_map.nc, hmax_subgrid.tif, bathtub raster) and reruns only
#   eikonal sweep (all friction factors) -> calibration_result.json
# The stage-2 jobs are written by generate_calibration_jobs.py stage2 (SFINCS time estimate 0) and
# submitted here, followed by one post job (afterany on all of them) that refreshes everything the
# analysis reads:
#   compute_calibration_result_dedup.py           -> calibration_roughness_SFINCS_dedup
#   run_calibration_sweep_analysis.py (both dirs) -> friction_sweep_*_metrics.csv, figures, tables
#
# Usage:
#   bash sfincs_tiles/rerun_eikonal_calibration.sh              # reset + submit 20 jobs + post job
#   N_JOBS=40 bash sfincs_tiles/rerun_eikonal_calibration.sh
#   POST=0 bash sfincs_tiles/rerun_eikonal_calibration.sh       # no post job (run the analysis by hand)
#   DRY_RUN=1 bash sfincs_tiles/rerun_eikonal_calibration.sh    # only report what would be reset

set -euo pipefail

BASE_DIR_NAME="${BASE_DIR_NAME:-calibration_roughness_SFINCS}"
DEDUP_DIR_NAME="${DEDUP_DIR_NAME:-${BASE_DIR_NAME}_dedup}"
N_JOBS="${N_JOBS:-20}"
POST="${POST:-1}"
DRY_RUN="${DRY_RUN:-0}"
DATA_ROOT="/p/11212688-004-global-floodmaps/modelling"
BASE="$DATA_ROOT/$BASE_DIR_NAME"
CODE="/u/schlumbe/gfm_code/sfincs_tiles"
GFM_PY="/u/schlumbe/.conda/envs/gfm/bin/python"
HYDROMT_PY="/u/schlumbe/.conda/envs/hydromt-sfincs-dev/bin/python"
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
