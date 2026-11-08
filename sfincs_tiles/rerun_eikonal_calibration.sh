#!/bin/bash
# Alias for `rerun_calibration.sh --part models` (PHASE, RESET_RUNS, ...); env vars pass through.
exec bash "$(dirname "${BASH_SOURCE[0]}")/rerun_calibration.sh" --part models "$@"
