# Hydrax paths for the sfincs_tiles/ (and calibration_studies/) HPC drivers - the single
# definition, sourced by run_final_tile.sh and rerun_calibration.sh, and read by
# generate_calibration_jobs.py / calibration_studies/generate_calibration_batch_jobs.py via
# gfm_config.hydrax_paths(). Each value can be overridden from the environment, e.g.
#   GFM_PYTHON=/other/env/bin/python bash rerun_calibration.sh --part eikonal
# Keep every line in the NAME="${NAME:-default}" form: gfm_config.hydrax_paths() parses it.

GFM_DATA_ROOT="${GFM_DATA_ROOT:-/p/11212688-004-global-floodmaps/modelling}"
GFM_CODE_ROOT="${GFM_CODE_ROOT:-/u/schlumbe/gfm_code}"
GFM_PYTHON="${GFM_PYTHON:-/u/schlumbe/.conda/envs/gfm/bin/python}"
GFM_SFINCS_PYTHON="${GFM_SFINCS_PYTHON:-/u/schlumbe/.conda/envs/hydromt-sfincs-dev/bin/python}"
