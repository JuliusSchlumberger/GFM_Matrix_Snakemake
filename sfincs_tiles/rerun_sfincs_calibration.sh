#!/bin/bash
# Kept for existing commands/docs: same as `rerun_calibration.sh --part sfincs` (all env vars,
# e.g. N_JOBS / DRY_RUN, pass through).
exec bash "$(dirname "${BASH_SOURCE[0]}")/rerun_calibration.sh" --part sfincs "$@"
