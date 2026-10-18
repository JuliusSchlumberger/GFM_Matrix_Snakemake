# Site values for the HPC shell drivers, worked example for Hydrax (Deltares).
# Copy to sfincs_tiles/site_env.sh (gitignored) and fill in your site values. Sourced by the
# .sh drivers and parsed by gfm_config.site_paths(), so keep every line in the
# NAME="${NAME:-default}" form; a set environment variable overrides the value.
# GFM_DATA_ROOT / GFM_CODE_ROOT must equal paths.root / paths.code_root in
# snakemake_workflow/config/config_hpc.yml (the calibration generators check this).

# Linux mount of the shared data root (paths.root).
GFM_DATA_ROOT="${GFM_DATA_ROOT:-/p/11212688-004-global-floodmaps/modelling}"
# Linux repo checkout the jobs run from (paths.code_root).
GFM_CODE_ROOT="${GFM_CODE_ROOT:-/u/schlumbe/gfm_code}"
# Python of the gfm env.
GFM_PYTHON="${GFM_PYTHON:-/u/schlumbe/.conda/envs/gfm/bin/python}"
# Python of the hydromt-sfincs-dev env.
GFM_SFINCS_PYTHON="${GFM_SFINCS_PYTHON:-/u/schlumbe/.conda/envs/hydromt-sfincs-dev/bin/python}"
# SFINCS container image run by run_final_tile.sh.
SFINCS_IMAGE="${SFINCS_IMAGE:-docker://deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release}"
# Container runtime that runs SFINCS_IMAGE (`<cmd> exec -B <dir>:/mnt/data <image> sfincs`).
CONTAINER_CMD="${CONTAINER_CMD:-apptainer}"
# SLURM partition of rerun_calibration.sh's post job (dedup + sweep analyses).
CAL_PARTITION="${CAL_PARTITION:-4vcpu}"
# CPUs of that post job (also the dedup's --workers).
CAL_CPUS="${CAL_CPUS:-4}"
# Memory of that post job.
CAL_MEM="${CAL_MEM:-30G}"
# Wall-clock limit of that post job (days-hours:minutes:seconds).
CAL_TIME="${CAL_TIME:-0-12:00:00}"
