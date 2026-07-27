#!/bin/bash
# Run the `simulate` Snakemake target grouped by geographic region (see
# src/regions.py: a continent, or a longitude-bisected half of an oversized
# one) instead of an arbitrary index range - each region is a real,
# analyzable chunk of global coverage, so partial results become usable for
# preliminary per-region analysis as soon as that region finishes, rather
# than at an arbitrary "job N of TOTAL" boundary.
#
# Sibling to run_simulate_batches.sh (index-based), not a replacement - use
# whichever grouping suits what you need at the time.
#
# Usage:
#   ./run_simulate_regions.sh [START_INDEX] [CORES] [MEM_MB]
#
#   Regions are discovered dynamically from list_regions.py - the single
#   source of truth shared with the Snakefile's own REGION_SIMULATION_OUTPUTS,
#   never hardcoded/duplicated here - sorted largest-first (biggest regions
#   start first). START_INDEX (default 1) skips the first N-1 regions in
#   that order, e.g. to resume after a completed region without re-verifying
#   it, the same idea as run_preprocess_batches.sh's START_BATCH. CORES/
#   MEM_MB default to 4/8000, same as run_simulate_batches.sh.
#
# Orchestration only: which region builds a tile has NO effect on where its
# output is stored (still model_outputs/{tile_id}/results/... exactly as
# always) or on postprocessing/the country-and-continent exposure analysis
# downstream (analysis/compute_exposure_analysis.py etc. aggregate from the
# population/geogunit rasters at global-grid resolution, entirely
# independent of this grouping and completely unaffected by it).
#
# OOM handling: NOT done here, deliberately - same reasoning as
# run_simulate_batches.sh. run_aqueduct.py already catches a tile's
# OutOfMemoryError itself, marks it in model_outputs/oom_tiles/, and writes
# an all-nodata placeholder instead of failing the job, so a plain
# `snakemake <region marker>` run always completes even with some tiles
# OOM'd - nothing here needs to retry or split anything for a region to
# finish. Once ALL regions (and/or index batches) have completed, run
# snakemake_workflow/run_pipeline.py ONCE, separately, targeting the full
# `simulate` rule - it re-invokes snakemake, progressively splitting
# whatever tiles ended up marked OOM and re-simulating the smaller pieces,
# to replace those nodata placeholders with real results where possible.
# Doing that retry-and-split loop PER REGION instead (an earlier version of
# this script did exactly that, via run_pipeline.py's --target/--batch
# flags) is redundant at best - it's a job that only needs to run once,
# globally, after coverage is complete - and at worst, run_pipeline.py's
# own --configfile forwarding actively broke every region invocation (see
# run_pipeline.py's own comment on that bug).
#
# Hang protection, and progress reporting: same design as
# run_simulate_batches.sh (DAG-build-hang watchdog via a one-shot "Job
# stats:" latch; a killed attempt is torn down with `taskkill /T` + a
# defensive `snakemake --unlock`) - see that script's header for the full
# reasoning, not repeated here. The one real difference: regions are NOT
# equal-sized like index batches are (a continent can be 2-3x another's
# tile count), so the overall-percentage figure is weighted by each
# region's actual job count from list_regions.py, not an assumed-equal share.

set -uo pipefail

START_INDEX="${1:-1}"
CORES="${2:-4}"
MEM_MB="${3:-8000}"
STALL_TIMEOUT_S=1200
POLL_INTERVAL_S=30
MAX_ATTEMPTS_PER_REGION=3
LOG_DIR="batch_logs_simulate_regions"
LIST_REGIONS="snakemake_workflow/list_regions.py"
LATENCY_WAIT_S=60

mkdir -p "$LOG_DIR"

# Discover regions once: name, job count, marker path - tab-separated,
# largest job count first (list_regions.py's own sort order).
mapfile -t REGION_LINES < <(python "$LIST_REGIONS")
NUM_REGIONS=${#REGION_LINES[@]}
if [ "$NUM_REGIONS" -eq 0 ]; then
    echo "list_regions.py returned no regions - aborting." >&2
    exit 1
fi
if [ "$START_INDEX" -lt 1 ] || [ "$START_INDEX" -gt "$NUM_REGIONS" ]; then
    echo "START_INDEX ($START_INDEX) must be between 1 and NUM_REGIONS ($NUM_REGIONS)." >&2
    exit 1
fi

declare -a REGION_NAMES REGION_JOBS REGION_MARKERS
TOTAL_JOBS=0
while IFS=$'\t' read -r name jobs marker; do
    # Defensive: strip a trailing \r from every field in case anything
    # upstream (a different Python/OS, a different capture method) ever
    # reintroduces one - list_regions.py forces \n-only output itself now,
    # but a marker path with an invisible trailing \r silently fails to
    # match the Snakefile's clean output pattern (confirmed the hard way:
    # MissingRuleException for what looks like, but isn't, the same path).
    name="${name%$'\r'}"
    jobs="${jobs%$'\r'}"
    marker="${marker%$'\r'}"
    REGION_NAMES+=("$name")
    REGION_JOBS+=("$jobs")
    REGION_MARKERS+=("$marker")
    TOTAL_JOBS=$((TOTAL_JOBS + jobs))
done < <(printf '%s\n' "${REGION_LINES[@]}")

# Cumulative job count of every region processed BEFORE index k (0-based),
# used to weight the overall-percentage figure by actual region size.
declare -a CUMULATIVE_BEFORE
CUMULATIVE_BEFORE[0]=0
for ((k = 1; k < NUM_REGIONS; k++)); do
    CUMULATIVE_BEFORE[k]=$(( ${CUMULATIVE_BEFORE[k - 1]} + ${REGION_JOBS[k - 1]} ))
done

run_one_region() {
    local region="$1"
    local marker="$2"
    local cumulative_before="$3"
    local log="$4"
    : > "$log"

    snakemake "$marker" --cores "$CORES" --resources "mem_mb=$MEM_MB" \
        --rerun-incomplete --rerun-triggers mtime --latency-wait "$LATENCY_WAIT_S" > "$log" 2>&1 &
    local pid=$!

    local last_size=-1
    local stalled_for=0
    local dag_built=false
    local last_progress=""

    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_INTERVAL_S"

        if [ "$dag_built" = false ] && grep -q "^Job stats:" "$log" 2>/dev/null; then
            dag_built=true
            echo "  [watchdog] DAG built - handing off to normal execution (no more stall checks this region)"
        fi

        if [ "$dag_built" = true ]; then
            local progress_line
            progress_line=$(grep -oE '[0-9]+ of [0-9]+ steps \([0-9]+%\) done' "$log" | tail -1)
            if [ -n "$progress_line" ] && [ "$progress_line" != "$last_progress" ]; then
                last_progress="$progress_line"
                local x overall
                x=$(echo "$progress_line" | grep -oE '^[0-9]+')
                overall=$(awk -v c="$cumulative_before" -v t="$TOTAL_JOBS" -v x="$x" \
                    'BEGIN { printf "%.1f", (c + x) / t * 100 }')
                echo "  [progress] region $region: $progress_line | overall: ~${overall}%"
            fi
            continue
        fi

        local size
        size=$(wc -c < "$log" 2>/dev/null || echo 0)
        if [ "$size" -eq "$last_size" ]; then
            stalled_for=$((stalled_for + POLL_INTERVAL_S))
        else
            stalled_for=0
            last_size=$size
        fi

        if [ "$stalled_for" -ge "$STALL_TIMEOUT_S" ]; then
            echo "  [watchdog] no DAG-build output for ${STALL_TIMEOUT_S}s - assuming hung, killing pid $pid (whole tree)"
            taskkill //F //T //PID "$pid" >/dev/null 2>&1
            wait "$pid" 2>/dev/null
            return 124
        fi
    done

    wait "$pid"
    return $?
}

echo "Running simulate by region: $((NUM_REGIONS - START_INDEX + 1)) of $NUM_REGIONS regions (starting at index $START_INDEX), $CORES cores, mem_mb=$MEM_MB each."
echo "Regions (largest first): ${REGION_NAMES[*]}"
echo "Logs -> $LOG_DIR/"

for ((idx = START_INDEX; idx <= NUM_REGIONS; idx++)); do
    region="${REGION_NAMES[idx - 1]}"
    marker="${REGION_MARKERS[idx - 1]}"
    cumulative_before="${CUMULATIVE_BEFORE[idx - 1]}"
    log="$LOG_DIR/$(echo "$region" | tr ' ' '_').log"
    echo ""
    echo "=== Region $idx/$NUM_REGIONS: $region (${REGION_JOBS[idx - 1]} jobs) ==="
    attempt=1
    while true; do
        run_one_region "$region" "$marker" "$cumulative_before" "$log"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "  region $region done"
            break
        elif [ "$rc" -eq 124 ] && [ "$attempt" -lt "$MAX_ATTEMPTS_PER_REGION" ]; then
            attempt=$((attempt + 1))
            echo "  region $region hung - retry attempt $attempt/$MAX_ATTEMPTS_PER_REGION"
            snakemake --unlock >/dev/null 2>&1 || true
        else
            echo ""
            echo "  region $region FAILED (exit $rc) after $attempt attempt(s)."
            echo "  See $log for details."
            echo "  Fix the issue, then either re-run this script with START_INDEX=$idx"
            echo "  or re-invoke just this region directly:"
            echo "    snakemake \"$marker\" --cores $CORES --resources mem_mb=$MEM_MB --rerun-incomplete --rerun-triggers mtime --latency-wait $LATENCY_WAIT_S"
            exit 1
        fi
    done
done

echo ""
echo "All regions completed successfully."
echo ""
echo "Next: once you're done running regions (this script and/or run_simulate_batches.sh),"
echo "run the OOM cleanup pass ONCE over the full simulate target:"
echo "  python snakemake_workflow/run_pipeline.py --target simulate --cores $CORES --mem-mb $MEM_MB"
