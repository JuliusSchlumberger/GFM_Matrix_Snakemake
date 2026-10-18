# Hydrax paths for the HPC drivers (sourced by the .sh drivers, parsed by gfm_config.hydrax_paths()).
# Keep every line in the NAME="${NAME:-default}" form: gfm_config.hydrax_paths() parses it.
# GFM_DATA_ROOT / GFM_CODE_ROOT must match paths.root / paths.code_root in
# snakemake_workflow/config/config_hpc.yml.

GFM_DATA_ROOT="${GFM_DATA_ROOT:-/p/11212688-004-global-floodmaps/modelling}"
GFM_CODE_ROOT="${GFM_CODE_ROOT:-/u/schlumbe/gfm_code}"
GFM_PYTHON="${GFM_PYTHON:-/u/schlumbe/.conda/envs/gfm/bin/python}"
GFM_SFINCS_PYTHON="${GFM_SFINCS_PYTHON:-/u/schlumbe/.conda/envs/hydromt-sfincs-dev/bin/python}"
