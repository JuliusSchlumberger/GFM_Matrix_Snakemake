#!/bin/bash
# Rerun only the SFINCS part of the final calibration (calibration_roughness_SFINCS) in place,
# after a change to the SFINCS forcing/mask code (build_line_forcing.py, build_sfincs_tile.py,
# connected_forcing.py). Run on Hydrax, after syncing the code to /u/schlumbe/gfm_code.
#
# Per selected tile (tile_ids.txt) it removes the done/failed/step markers - which re-queues the
# tile - and sfincs.inp, which makes run_final_tile.sh rebuild the line forcing and the SFINCS model
# (that step also clears sfincs_map.nc and hmax_subgrid.tif). Everything else is kept and skipped:
# copied inputs, elevation/roughness rasters, the eikonal/bathtub rasters. Then it writes and
# submits the stage-2 jobs (generate_calibration_jobs.py stage2); each tile reruns
#   line forcing -> SFINCS model build -> SFINCS run -> postprocessing -> calibration_result.json
#
# Afterwards: compute_calibration_result_dedup.py (overlap-corrected results) and
# run_calibration_sweep_analysis.py for both base directories.
#
# Usage:
#   bash sfincs_tiles/rerun_sfincs_calibration.sh              # reset + submit 20 jobs
#   N_JOBS=10 bash sfincs_tiles/rerun_sfincs_calibration.sh
#   DRY_RUN=1 bash sfincs_tiles/rerun_sfincs_calibration.sh    # only report what would be reset

set -euo pipefail

BASE_DIR_NAME="${BASE_DIR_NAME:-calibration_roughness_SFINCS}"
N_JOBS="${N_JOBS:-20}"
DRY_RUN="${DRY_RUN:-0}"
BASE="/p/11212688-004-global-floodmaps/modelling/$BASE_DIR_NAME"
CODE="/u/schlumbe/gfm_code/sfincs_tiles"
GFM_PY="/u/schlumbe/.conda/envs/gfm/bin/python"

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
