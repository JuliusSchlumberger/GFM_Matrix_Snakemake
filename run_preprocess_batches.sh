#!/bin/bash
# Run the `preprocess` Snakemake target in N sequential batches (via
# --batch preprocess=i/N), starting each one automatically as soon as the
# previous finishes - instead of manually invoking and checking on each
# batch yourself.
#
# Usage:
#   ./run_preprocess_batches.sh [TOTAL_BATCHES] [START_BATCH] [CORES]
#
#   TOTAL_BATCHES defaults to 20 (splitting the ~135k-job preprocess DAG
#   into ~6-7k jobs per batch keeps each individual DAG build fast - see
#   the --batch flag's own help text: "speeding up DAG computation").
#   START_BATCH defaults to 1 - set it higher to skip batches you already
#   know succeeded (e.g. from a previous run of this script), instead of
#   paying the few-minutes-per-batch "confirm nothing to do" cost Snakemake
#   still has to pay to verify that itself over P:\.
#   CORES defaults to 4, matching prior invocations.
#
# Hang protection:
#   A P:\ SMB stall can block Snakemake's "Building DAG of jobs..." phase
#   indefinitely with zero CPU progress and no exception raised - none of
#   the retry_transient_io/os.makedirs resilience fixes in the codebase can
#   help here, since those only trigger when a call fails fast, not when it
#   hangs forever. This script watches each batch's log for stalled output
#   ONLY until "Job stats:" appears (i.e. only during DAG building) and
#   kills+retries that batch if nothing is written for STALL_TIMEOUT_S
#   seconds. Once real job execution starts, the watchdog steps back
#   entirely - individual Aqueduct/extract jobs can legitimately run for a
#   long time, and a blind timeout there would abort genuinely-working
#   batches, which would be worse than the manual process this replaces.
#
# A batch failing for a real reason (not a hang) stops the whole script -
# fix the issue and re-run (already-completed batches are skipped fast by
# Snakemake since their outputs are up to date; only fix and resume from the
# batch number reported below by passing it as a one-off single-batch
# `snakemake ... --batch preprocess=<i>/<TOTAL_BATCHES>` call, or just
# re-run this script - it always starts from batch 1, but every prior batch
# will be a near-instant no-op).

set -uo pipefail

TOTAL_BATCHES="${1:-20}"
START_BATCH="${2:-1}"
CORES="${3:-4}"

if [ "$START_BATCH" -lt 1 ] || [ "$START_BATCH" -gt "$TOTAL_BATCHES" ]; then
    echo "START_BATCH ($START_BATCH) must be between 1 and TOTAL_BATCHES ($TOTAL_BATCHES)." >&2
    exit 1
fi

STALL_TIMEOUT_S=1200      # 20 min of silence during DAG-build -> assume hung
POLL_INTERVAL_S=30
MAX_ATTEMPTS_PER_BATCH=3
LOG_DIR="batch_logs"
# Snakemake's own post-job check: after a job's script exits, it waits up to
# --latency-wait seconds (default 5) for the declared output file to become
# visible before raising MissingOutputException. On P:\, a job can finish
# writing its output and still not be visible to Snakemake's process within
# that default window - the file IS there (confirmed: not a real failure,
# just P:\ propagation delay), so this only needed a more patient wait, not
# a retry. Distinct from the DAG-build hangs the watchdog above handles:
# this is Snakemake giving up too early on a file that does arrive, not a
# call that never returns at all.
LATENCY_WAIT_S=60

mkdir -p "$LOG_DIR"

run_one_batch() {
    local i="$1"
    local log="$LOG_DIR/batch_${i}.log"
    : > "$log"

    snakemake preprocess --cores "$CORES" --rerun-incomplete --rerun-triggers mtime \
        --latency-wait "$LATENCY_WAIT_S" \
        --batch "preprocess=${i}/${TOTAL_BATCHES}" > "$log" 2>&1 &
    local pid=$!

    local last_size=-1
    local stalled_for=0
    local dag_built=false
    local last_progress=""

    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_INTERVAL_S"

        if [ "$dag_built" = false ] && grep -q "^Job stats:" "$log" 2>/dev/null; then
            dag_built=true
            echo "  [watchdog] DAG built - handing off to normal execution (no more stall checks this batch)"
        fi

        if [ "$dag_built" = true ]; then
            # Snakemake prints "<done> of <total> steps (<pct>%) done" as jobs
            # finish - tail the log for the latest one and only print again
            # once it actually changes. Overall % treats all TOTAL_BATCHES
            # slices as roughly equal-sized (true by construction, since
            # --batch splits preprocess's input file list into N contiguous
            # chunks), so batch i being f fraction done maps to
            # ((i-1)+f)/TOTAL_BATCHES overall.
            local progress_line
            progress_line=$(grep -oE '[0-9]+ of [0-9]+ steps \([0-9]+%\) done' "$log" | tail -1)
            if [ -n "$progress_line" ] && [ "$progress_line" != "$last_progress" ]; then
                last_progress="$progress_line"
                local x y overall
                x=$(echo "$progress_line" | grep -oE '^[0-9]+')
                y=$(echo "$progress_line" | grep -oE 'of [0-9]+' | grep -oE '[0-9]+')
                overall=$(awk -v i="$i" -v n="$TOTAL_BATCHES" -v x="$x" -v y="$y" \
                    'BEGIN { f = (y > 0) ? x / y : 0; printf "%.1f", ((i - 1) + f) / n * 100 }')
                echo "  [progress] batch $i/$TOTAL_BATCHES: $progress_line | overall: ~${overall}%"
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
            echo "  [watchdog] no DAG-build output for ${STALL_TIMEOUT_S}s - assuming hung, killing pid $pid"
            kill -9 "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 124
        fi
    done

    wait "$pid"
    return $?
}

echo "Running preprocess: batches $START_BATCH-$TOTAL_BATCHES of $TOTAL_BATCHES total, $CORES cores each. Logs -> $LOG_DIR/"

for ((i = START_BATCH; i <= TOTAL_BATCHES; i++)); do
    echo ""
    echo "=== Batch $i/$TOTAL_BATCHES ==="
    attempt=1
    while true; do
        run_one_batch "$i"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "  batch $i/$TOTAL_BATCHES done"
            break
        elif [ "$rc" -eq 124 ] && [ "$attempt" -lt "$MAX_ATTEMPTS_PER_BATCH" ]; then
            attempt=$((attempt + 1))
            echo "  batch $i/$TOTAL_BATCHES hung - retry attempt $attempt/$MAX_ATTEMPTS_PER_BATCH"
        else
            echo ""
            echo "  batch $i/$TOTAL_BATCHES FAILED (exit $rc) after $attempt attempt(s)."
            echo "  See $LOG_DIR/batch_${i}.log for details."
            echo "  Fix the issue, then either re-run this script (earlier batches"
            echo "  no-op almost instantly) or re-invoke just this batch directly:"
            echo "    snakemake preprocess --cores $CORES --rerun-incomplete --rerun-triggers mtime --batch preprocess=${i}/${TOTAL_BATCHES}"
            exit 1
        fi
    done
done

echo ""
echo "All $TOTAL_BATCHES batches completed successfully."
