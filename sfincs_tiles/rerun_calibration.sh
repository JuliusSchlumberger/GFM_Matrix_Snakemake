#!/bin/bash
# Rerun part of the final calibration in place on the cluster, after syncing the code to $GFM_CODE_ROOT.
#   --part models   deletes the tile markers and runs one phase of run_calibration_models.py
#                   (PHASE=sweep | duration | selected, default sweep; existing runs are kept,
#                   RESET_RUNS=1 deletes every model raster and result first); POST=1 adds an
#                   afterany job running the dedup and the analysis (variant_selection.json) for both
#                   base dirs. duration and selected need the selection of the phase before.
#   --part sfincs   deletes markers + sfincs.inp, rebuilds forcing -> SFINCS model/run -> result;
#                   run the dedup and the analysis by hand afterwards.
# Usage:
#   RESET_RUNS=1 bash sfincs_tiles/rerun_calibration.sh --part models   # sweep from scratch + post job
#   PHASE=duration bash sfincs_tiles/rerun_calibration.sh --part models
#   PHASE=selected N_JOBS=40 bash sfincs_tiles/rerun_calibration.sh --part models
#   POST=0 bash sfincs_tiles/rerun_calibration.sh --part models         # no post job
#   bash sfincs_tiles/rerun_calibration.sh --part sfincs
#   DRY_RUN=1 bash sfincs_tiles/rerun_calibration.sh --part <part>      # only report what would be reset
# Paths and the post job's resources come from site_env.sh (override e.g. GFM_PYTHON=... in the environment).

set -euo pipefail

PART=""
case "${1:-}" in
  --part) PART="${2:-}" ;;
  --part=*) PART="${1#--part=}" ;;
esac
if [ "$PART" != "models" ] && [ "$PART" != "sfincs" ]; then
  echo "usage: bash $0 --part models|sfincs" >&2
  exit 2
fi

SITE_ENV="$(dirname "${BASH_SOURCE[0]}")/site_env.sh"
[ -f "$SITE_ENV" ] || { echo "sfincs_tiles/site_env.sh not found: copy sfincs_tiles/site_env.example.sh to sfincs_tiles/site_env.sh and fill in your site values" >&2; exit 1; }
source "$SITE_ENV"

BASE_DIR_NAME="${BASE_DIR_NAME:-calibration_roughness_SFINCS}"
N_JOBS="${N_JOBS:-20}"
DRY_RUN="${DRY_RUN:-0}"
BASE="$GFM_DATA_ROOT/$BASE_DIR_NAME"
CODE="$GFM_CODE_ROOT/sfincs_tiles"
GFM_PY="$GFM_PYTHON"

if [ "$PART" = "models" ]; then
  DEDUP_DIR_NAME="${DEDUP_DIR_NAME:-${BASE_DIR_NAME}_dedup}"
  POST="${POST:-1}"
  PHASE="${PHASE:-sweep}"
  RESET_RUNS="${RESET_RUNS:-0}"
  HYDROMT_PY="$GFM_SFINCS_PYTHON"
  CONFIG="$BASE/resolved_config.yml"

  [ -f "$BASE/tile_ids.txt" ] || { echo "no $BASE/tile_ids.txt" >&2; exit 1; }
  grep -q "def phase_runs" "$CODE/run_calibration_models.py" \
    || { echo "$CODE/run_calibration_models.py has no phase_runs - sync the code first" >&2; exit 1; }
  if [ "$PHASE" != "sweep" ] && [ ! -f "$BASE/variant_selection.json" ]; then
    echo "PHASE=$PHASE needs $BASE/variant_selection.json - run the sweep phase and its post job first" >&2
    exit 1
  fi

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
    n=$((n + 1))
    [ -d "$out" ] || continue  # never ran: nothing to reset (find would fail and abort under -e)
    if [ "$RESET_RUNS" = "1" ]; then
      k=$(find "$out" -maxdepth 1 -name '*_waterdepth_RP100_SLR_0*.tif' 2>/dev/null | wc -l)
      n_rasters=$((n_rasters + k))
    fi
    [ "$DRY_RUN" = "1" ] && continue
    rm -f "$out/calibration_run.done" "$out/calibration_run.failed" "$out/calibration_run.step"
    if [ "$RESET_RUNS" = "1" ]; then
      rm -f "$out"/*_waterdepth_RP100_SLR_0*.tif "$out/model_runs.json" "$out/eikonal_solver_diagnostics.json" \
            "$out/calibration_result.json"
    fi
  done < "$BASE/tile_ids.txt"

  if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: $n tile(s) would be reset for phase $PHASE, $n_rasters model raster(s) removed ($n_missing listed without a tile directory)"
    exit 0
  fi
  echo "reset $n tile(s) for phase $PHASE, removed $n_rasters model raster(s) ($n_missing listed without a tile directory)"

  # SFINCS outputs exist, so run_final_tile.sh skips SFINCS: estimate only the model runs.
  cd "$CODE"
  JOBS="$BASE/hpc_jobs_stage2"
  rm -f "$JOBS"/final_*.sbatch "$JOBS"/tiles_*.txt  # stale job files of an earlier run would be submitted below
  "$GFM_PY" generate_calibration_jobs.py stage2 --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \
      --n-jobs "$N_JOBS" --sfincs-min-per-mcell 0 --overhead-min 5 --phase "$PHASE"

  shopt -s nullglob
  sbatches=("$JOBS"/final_*.sbatch)
  shopt -u nullglob
  if [ "${#sbatches[@]}" -eq 0 ]; then
    echo "no final_*.sbatch written in $JOBS (nothing to rerun?) - nothing submitted"
    exit 0
  fi
  ids=()
  for f in "${sbatches[@]}"; do
    ids+=("$(sbatch --parsable "$f")")
  done
  echo "submitted ${#ids[@]} model job(s), phase $PHASE: ${ids[*]}"

  if [ "$POST" = "1" ]; then
    post="$JOBS/post_models_$PHASE.sbatch"
    cat > "$post" <<EOF
#!/bin/bash
#SBATCH --job-name=cal_post_$PHASE
#SBATCH --partition=$CAL_PARTITION
#SBATCH --cpus-per-task=$CAL_CPUS
#SBATCH --mem=$CAL_MEM
#SBATCH --time=$CAL_TIME
#SBATCH --output=$JOBS/logs/cal_post_${PHASE}_%j.out
#SBATCH --error=$JOBS/logs/cal_post_${PHASE}_%j.err

# No -e: the progress report and the non-dedup analysis must not stop the rest; they run
# before the dedup, which the dedup analysis reads and which ends the job when it fails.
set -uo pipefail
unset PROJ_LIB PROJ_DATA GDAL_DATA
cd "$CODE" || exit 1
"$GFM_PY" check_calibration_progress.py --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME"
"$GFM_PY" run_calibration_sweep_analysis.py --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME"
"$HYDROMT_PY" compute_calibration_result_dedup.py --config "$CONFIG" --base-dir-name "$BASE_DIR_NAME" \\
    --out-dir-name "$DEDUP_DIR_NAME" --workers $CAL_CPUS || exit 1
"$GFM_PY" run_calibration_sweep_analysis.py --config "$CONFIG" --base-dir-name "$DEDUP_DIR_NAME" \\
    --raster-base-dir-name "$BASE_DIR_NAME"
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

  # Eikonal rasters exist, so the sweep is skipped: its time estimate is ~0.
  cd "$CODE"
  "$GFM_PY" generate_calibration_jobs.py stage2 --config "$BASE/resolved_config.yml" \
      --base-dir-name "$BASE_DIR_NAME" --n-jobs "$N_JOBS" --eikonal-min-per-mcell 1 --submit
  echo "progress: $GFM_PY $CODE/check_calibration_progress.py --config $BASE/resolved_config.yml --base-dir-name $BASE_DIR_NAME"
fi
