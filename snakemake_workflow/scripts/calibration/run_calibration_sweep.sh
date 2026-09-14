#!/bin/bash
# Drives the full ESP/FRA/NOR HPC calibration sweep (docs/calibration_sweep_plan.md)
# through all (group, sweep_point) combinations sequentially, on Hydrax:
# build_run_config.py -> generate_hpc_preprocess_job.py --calibration ->
# submit -> wait for the SLURM queue to clear -> validate_country.py per
# country in the group -> next combination. Aggregates at the very end.
#
# "Sequential" here means one combination's ENTIRE chain (preprocessing ->
# simulation waves -> postprocessing -> exposure) runs to completion before
# the next one is submitted - NOT all 24 combinations queued at once. This
# avoids overlapping SLURM usage across combinations, at the cost of not
# using the cluster's own scheduler to interleave them.
#
# COMPLETION DETECTION IS APPROXIMATE: a combination is considered "done"
# once `squeue -u $USER` is completely empty - this only works cleanly if
# you are not ALSO running other, unrelated SLURM jobs under the same user
# at the same time. If you are, this will falsely think a combination
# finished early. There's no cheaper alternative here: the wave/postprocess/
# exposure job IDs are only known AFTER generate_jobs_and_dispatch.sbatch
# runs on the cluster (dynamically, not something this script can predict
# up front), and every sbatch script this pipeline generates uses fixed job
# NAMES (wave0_small_batch_000 etc.) shared across every combination, so
# name-based filtering can't disambiguate one combination's jobs from
# another's either.
#
# RESUMABLE: skips any (group, sweep_point) whose metrics CSV already
# exists for every country in that group, so re-running this script after
# an interruption (SSH drop, etc.) picks up where it left off rather than
# redoing already-validated combinations. Run this under `tmux`/`screen`/
# `nohup` on the login node - it's a long-running foreground process and
# will die with your SSH session otherwise.
#
# Usage:
#   bash run_calibration_sweep.sh [--poll-interval SECONDS]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

POLL_INTERVAL_S=60
if [ "${1:-}" = "--poll-interval" ]; then
    POLL_INTERVAL_S="$2"
fi

# NOTE: named SCENARIO_GROUPS, not GROUPS - GROUPS is a reserved bash
# special variable (the current user's real Unix group-ID list);
# assigning to it silently does nothing and ${GROUPS[@]} keeps returning
# the real system group list instead (confirmed live 2026-09-14 - every
# combination's "group" became a real GID and n_total blew up to
# len(real_groups) x len(SWEEP_POINTS)).
SCENARIO_GROUPS=(esp_fra_rp100 nor_rp250)
SWEEP_POINTS=(
    baseline
    friction_0.5 friction_2.0
    max_rounds_4 max_rounds_8 max_rounds_20
    obstacle_coupling_off obstacle_coupling_iter1 obstacle_coupling_iter3 obstacle_coupling_iter10
    waterlevel_eps_0.01 waterlevel_eps_0.10
)

declare -A GROUP_COUNTRIES=(
    [esp_fra_rp100]="ESP FRA"
    [nor_rp250]="NOR"
)

# Resolved once, reused for the per-combination resume check below -
# validation.output_dir is isolated per run_tag (build_run_config.py's own
# path_overrides), so the mere EXISTENCE of a metrics CSV under
# {ROOT}/calibration_esp_fra_nor/{run_tag}/validation/{country}/ already
# uniquely identifies it as belonging to that run_tag - no need to inspect
# file contents.
ROOT="$(python -c "import sys; sys.path.insert(0, 'src'); from config_utils import load_config; print(load_config('snakemake_workflow/config/config.yml')['paths']['root'])")"

wait_for_queue_empty() {
    echo "  waiting for SLURM queue to clear (polling every ${POLL_INTERVAL_S}s)..."
    while true; do
        # A transient squeue failure must NOT be read as "0 jobs" (which
        # would wrongly conclude the combination finished early and start
        # validating incomplete output) - retry instead of trusting it.
        n_jobs="$(squeue -u "$USER" -h 2>/dev/null | wc -l)" || { sleep "$POLL_INTERVAL_S"; continue; }
        [ "$n_jobs" -eq 0 ] && break
        sleep "$POLL_INTERVAL_S"
    done
    echo "  queue clear."
}

n_total=$((${#SCENARIO_GROUPS[@]} * ${#SWEEP_POINTS[@]}))
n_done=0

for group in "${SCENARIO_GROUPS[@]}"; do
    for sweep in "${SWEEP_POINTS[@]}"; do
        run_tag="${group}__${sweep}"
        n_done=$((n_done + 1))
        echo ""
        echo "=================================================="
        echo "=== [$n_done/$n_total] $run_tag ==="
        echo "=================================================="

        # Resume support: skip if every country in this group already has a
        # metrics CSV under this run_tag's own isolated validation dir.
        validation_dir="$ROOT/calibration_esp_fra_nor/$run_tag/validation"
        all_validated=true
        for country in ${GROUP_COUNTRIES[$group]}; do
            if ! ls "$validation_dir/$country"/metrics_*.csv >/dev/null 2>&1; then
                all_validated=false
                break
            fi
        done
        if [ "$all_validated" = true ]; then
            echo "  already validated for every country in this group - skipping."
            continue
        fi

        t_start=$(date +%s)
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] building run config..."
        build_out="$(python snakemake_workflow/scripts/calibration/build_run_config.py --group "$group" --sweep "$sweep")"
        echo "$build_out"
        config_path="$(echo "$build_out" | grep '^wrote ' | sed 's/^wrote //')"

        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] generating sbatch scripts (--calibration)..."
        gen_out="$(python snakemake_workflow/scripts/generate_hpc_preprocess_job.py --config "$config_path" --calibration)"
        echo "$gen_out"
        submit_script="$(echo "$gen_out" | grep '^Submit on Hydrax with: bash ' | sed 's/^Submit on Hydrax with: bash //')"

        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] submitting: $submit_script"
        bash "$submit_script"

        wait_for_queue_empty

        for country in ${GROUP_COUNTRIES[$group]}; do
            echo "  [$(date '+%Y-%m-%d %H:%M:%S')] validating $country..."
            python validation/validate_country.py --config "$config_path" --country "$country" --run-tag "$run_tag"
        done

        t_end=$(date +%s)
        echo "  [$run_tag] done in $(( (t_end - t_start) / 60 ))m $(( (t_end - t_start) % 60 ))s"
    done
done

echo ""
echo "=== All combinations done - aggregating ==="
python snakemake_workflow/scripts/calibration/aggregate_calibration_results.py
