#!/bin/bash
# Alias for `rerun_calibration.sh --part sfincs`; env vars pass through.
exec bash "$(dirname "${BASH_SOURCE[0]}")/rerun_calibration.sh" --part sfincs "$@"
