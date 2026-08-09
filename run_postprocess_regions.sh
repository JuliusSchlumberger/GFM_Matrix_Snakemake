#!/bin/bash
# Run the `postprocess_region` Snakemake target grouped by geographic region
# (see src/regions.py, src/chunks.py: a continent, or half of an oversized
# one) instead of requiring every tile globally to be simulated first - the
# postprocessing-side sibling of run_simulate_regions.sh.
#
# Usage:
#   ./run_postprocess_regions.sh [START_INDEX] [CORES] [MEM_MB]
#
#   Regions are discovered dynamically from list_region_chunks.py - the
#   single source of truth shared with the Snakefile's own
#   REGION_POSTPROCESS_OUTPUTS, never hardcoded/duplicated here - sorted
#   largest-first (biggest regions start first). START_INDEX (default 1)
#   skips the first N-1 regions in that order, e.g. to resume after a
#   completed region without re-verifying it, the same idea as
#   run_simulate_regions.sh's own START_INDEX. CORES/MEM_MB default to
#   4/8000, same as the other batch runners (postprocessing jobs are
#   lighter-weight than Aqueduct itself, but share the same --resources
#   mem_mb budget/scheduler).
#
# Region scoping is NOT the same shape as run_simulate_regions.sh's: a
# postprocessing chunk can straddle a region boundary (e.g. one chunk with
# tiles from both Europe_West and Europe_East), so each region only ever
# requests its "safe" chunks - those fully contained within that region
# (see src/chunks.py:safe_chunks_for_region and rule postprocess_region's
# own docstring) - never a chunk that also needs another region's tiles.
# "Partial" chunks are silently NOT requested by this script for either
# region touching them; list_region_chunks.py prints a note (to stderr)
# listing how many each region has outstanding. Once every region a
# straddling chunk touches has been simulated, run the plain, global
# `postprocess` target (or re-run this script after adding the missing
# region) to pick those up - nothing is lost, they're just not yet
# requestable per-region.
#
# This also means running this script for every region does NOT add up to
# the same thing as the plain `postprocess` target: the plot-only outputs
# (build_mosaic_vrt, plot_merged_results, plot_overlap_continent_diagnostics)
# mosaic every chunk globally by construction and can't be meaningfully
# scoped to one region - see rule postprocess_region's own docstring. Run
# `snakemake postprocess` once, globally, after every region (via this
# script and/or run_postprocess_regions.sh's simulate-side counterpart) has
# completed, to produce those.
#
# Hang protection, and progress reporting: same design as
# run_simulate_regions.sh (DAG-build-hang watchdog via a one-shot "Job
# stats:" latch; a killed attempt is torn down with `taskkill /T` + a
# defensive `snakemake --unlock`) - see that script's header for the full
# reasoning, not repeated here. As there, regions are NOT equal-sized, so
# the overall-percentage figure is weighted by each region's actual safe-
# chunk job count from list_region_chunks.py, not an assumed-equal share.

set -uo pipefail

START_INDEX="${1:-1}"
CORES="${2:-4}"
MEM_MB="${3:-8000}"
STALL_TIMEOUT_S=1200
POLL_INTERVAL_S=30
MAX_ATTEMPTS_PER_REGION=3
LOG_DIR="batch_logs_postprocess_regions"
LIST_REGION_CHUNKS="snakemake_workflow/list_region_chunks.py"
LATENCY_WAIT_S=60

mkdir -p "$LOG_DIR"

# Discover regions once: name, n_safe_chunks, n_partial_chunks, job count,
# marker path - tab-separated, largest job count first (list_region_chunks.
# py's own sort order). Any "# Note: ..." line list_region_chunks.py prints
# (to stderr, about partial chunks) passes straight through to this
# script's own stderr, not captured here.
mapfile -t REGION_LINES < <(python "$LIST_REGION_CHUNKS")
NUM_REGIONS=${#REGION_LINES[@]}
if [ "$NUM_REGIONS" -eq 0 ]; then
    echo "list_region_chunks.py returned no regions - aborting." >&2
    exit 1
fi
if [ "$START_INDEX" -lt 1 ] || [ "$START_INDEX" -gt "$NUM_REGIONS" ]; then
    echo "START_INDEX ($START_INDEX) must be between 1 and NUM_REGIONS ($NUM_REGIONS)." >&2
    exit 1
fi

declare -a REGION_NAMES REGION_SAFE REGION_PARTIAL REGION_JOBS REGION_MARKERS
TOTAL_JOBS=0
while IFS=$'\t' read -r name n_safe n_partial jobs marker; do
    # Defensive \r-stripping - same documented reasoning as
    # run_simulate_regions.sh's identical guard against list_regions.py's
    # own confirmed CRLF bug; list_region_chunks.py forces \n-only output
    # itself too, but this is cheap insurance against the same class of bug
    # reappearing from a different capture path.
    name="${name%$'\r'}"
    n_safe="${n_safe%$'\r'}"
    n_partial="${n_partial%$'\r'}"
    jobs="${jobs%$'\r'}"
    marker="${marker%$'\r'}"
    REGION_NAMES+=("$name")
    REGION_SAFE+=("$n_safe")
    REGION_PARTIAL+=("$n_partial")
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

echo "Running postprocess by region: $((NUM_REGIONS - START_INDEX + 1)) of $NUM_REGIONS regions (starting at index $START_INDEX), $CORES cores, mem_mb=$MEM_MB each."
echo "Regions (largest first): ${REGION_NAMES[*]}"
echo "Logs -> $LOG_DIR/"

for ((idx = START_INDEX; idx <= NUM_REGIONS; idx++)); do
    region="${REGION_NAMES[idx - 1]}"
    marker="${REGION_MARKERS[idx - 1]}"
    cumulative_before="${CUMULATIVE_BEFORE[idx - 1]}"
    log="$LOG_DIR/$(echo "$region" | tr ' ' '_').log"
    echo ""
    echo "=== Region $idx/$NUM_REGIONS: $region (${REGION_SAFE[idx - 1]} safe chunks, ${REGION_PARTIAL[idx - 1]} partial, ${REGION_JOBS[idx - 1]} jobs) ==="
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
echo "All regions completed successfully (safe chunks only - see header comment about partial chunks)."
echo ""
echo "Next: once every region touching a given chunk has been simulated,"
echo "run the plain, global postprocess target to pick up any remaining"
echo "partial chunks and produce the plots (which mosaic every chunk globally):"
echo "  snakemake postprocess --cores $CORES --resources mem_mb=$MEM_MB --rerun-incomplete --rerun-triggers mtime --latency-wait $LATENCY_WAIT_S"
