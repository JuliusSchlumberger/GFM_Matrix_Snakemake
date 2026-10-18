#!/bin/bash
# Alias for `rerun_calibration.sh --part eikonal`; env vars pass through.
exec bash "$(dirname "${BASH_SOURCE[0]}")/rerun_calibration.sh" --part eikonal "$@"
