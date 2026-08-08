#!/bin/bash
#SBATCH --job-name=rebalance_data
#SBATCH --output=/u/morenodu/ml_downscaling/code/logs/rebalance_data_output_%j.log
#SBATCH --error=/u/morenodu/ml_downscaling/code/logs/rebalance_data_error_%j.log
#SBATCH --time=0-01:30:00       # Adjust based on data size and number of files
#SBATCH --partition=16vcpu       # Or a partition with more cores/memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # Number of CPUs for Dask workers
#SBATCH --mem=30G               # Adjust memory; can be memory-intensive

echo "========================================================"
echo "Starting Data Rebalancing Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Submitted from: $(pwd)"
echo "Running on host: $(hostname)"
date
echo "========================================================"

# --- Configuration ---
CONDA_ENV_NAME="ml_downscaling"
PROJECT_DIR="/p/archivedprojects/11211481-023-ml-rain-adjust"
PYTHON_SCRIPT="/u/morenodu/ml_downscaling/code/rebalance_training_data.py"

# --- Input Data Configuration ---
INPUT_DATA_BASE_DIR="${PROJECT_DIR}/Data/preprocessed_data/regridded_to_mswep/preprocessed_data"

ERA5_TP_FILE="${INPUT_DATA_BASE_DIR}/era5_tp_1990_2024_subset_regridded_to_mswep_CRS4326_19900101_20241231.nc"
ERA5_TEMP_FILE="${INPUT_DATA_BASE_DIR}/era5_t_1990_2024_subset_regridded_to_mswep_CRS4326_19900101_20241231.nc"
ERA5_TCWV_FILE="${INPUT_DATA_BASE_DIR}/era5_tcwv_1990_2024_subset_regridded_to_mswep_CRS4326_19900101_20241231.nc"
ERA5_U_FILE="${INPUT_DATA_BASE_DIR}/era5_u_1990_2024_subset_regridded_to_mswep_CRS4326_19900101_20241231.nc"
ERA5_V_FILE="${INPUT_DATA_BASE_DIR}/era5_v_1990_2024_subset_regridded_to_mswep_CRS4326_19900101_20241231.nc"
ERA5_R_FILE="${INPUT_DATA_BASE_DIR}/era5_r_1990_2024_subset_regridded_to_mswep_CRS4326_19900101_20241231.nc"

DEM_FILE="${INPUT_DATA_BASE_DIR}/gebco_2024_n-10.0_s-45.0_w112.0_e155.0_regridded_to_mswep_CRS4326.nc"
MSWEP_REF_FILE="${INPUT_DATA_BASE_DIR}/mswep_precip_merged_1990_2024_processed_CRS4326_19900101_20241231.nc"

# --- Output Directory ---
REBALANCED_OUTPUT_DIR="${PROJECT_DIR}/Data/preprocessed_data/regridded_to_mswep_wet_bal"

# --- Rebalancing Parameters ---
PIXEL_PRECIP_THRESHOLD="1.0"
AREA_PERCENT_THRESHOLD="5.0"
TARGET_DRY_WET_RATIO="1:3"

# --- Processing Settings ---
DASK_WORKERS=${SLURM_CPUS_PER_TASK:-4}
LOG_LEVEL_PYTHON="INFO"

# --- Setup ---
mkdir -p "${REBALANCED_OUTPUT_DIR}"

echo "Loading Miniconda module..."
module load miniconda
if [ $? -ne 0 ]; then echo "ERROR: Failed to load miniconda module."; exit 1; fi

echo "Initializing Conda shell integration..."
eval "$(conda shell.bash hook)"
if [ $? -ne 0 ]; then echo "ERROR: Failed to initialize conda shell hook."; exit 1; fi

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then echo "ERROR: Failed to activate conda environment '${CONDA_ENV_NAME}'."; exit 1; fi

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# --- CRITICAL: Comprehensive Input File Checking ---
echo "========================================================"
echo "CHECKING ALL INPUT FILES"
echo "========================================================"

# Create arrays for easier processing
declare -A INPUT_FILES=(
    ["MSWEP_REF"]="${MSWEP_REF_FILE}"
    ["ERA5_TP"]="${ERA5_TP_FILE}"
    ["ERA5_TEMP"]="${ERA5_TEMP_FILE}"
    ["ERA5_TCWV"]="${ERA5_TCWV_FILE}"
    ["ERA5_U"]="${ERA5_U_FILE}"
    ["ERA5_V"]="${ERA5_V_FILE}"
    ["ERA5_R"]="${ERA5_R_FILE}"
    ["DEM"]="${DEM_FILE}"
)

MISSING_FILES=0
EXISTING_FILES=0

for file_type in "${!INPUT_FILES[@]}"; do
    file_path="${INPUT_FILES[$file_type]}"
    if [ -f "$file_path" ]; then
        file_size=$(du -h "$file_path" | cut -f1)
        echo "EXISTS: $file_type ($file_size)"
        echo "   Path: $file_path"
        EXISTING_FILES=$((EXISTING_FILES + 1))
    else
        echo "MISSING: $file_type"
        echo "   Expected path: $file_path"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

echo "--------------------------------------------------------"
echo "INPUT FILE SUMMARY:"
echo "  Total files expected: ${#INPUT_FILES[@]}"
echo "  Files found: $EXISTING_FILES"
echo "  Files missing: $MISSING_FILES"
echo "--------------------------------------------------------"

if [ $MISSING_FILES -gt 0 ]; then
    echo "WARNING: $MISSING_FILES input files are missing!"
    echo "The rebalancing script will skip missing files silently."
    echo "Consider completing the regridding step for missing files first."
    echo ""
fi

# --- Check Output Directory and Existing Files ---
echo "CHECKING OUTPUT DIRECTORY AND EXISTING FILES"
echo "Output directory: $REBALANCED_OUTPUT_DIR"

if [ -d "$REBALANCED_OUTPUT_DIR" ]; then
    existing_output_files=$(ls -1 "$REBALANCED_OUTPUT_DIR"/*.nc 2>/dev/null | wc -l)
    if [ $existing_output_files -gt 0 ]; then
        echo "Found $existing_output_files existing output files:"
        ls -lh "$REBALANCED_OUTPUT_DIR"/*.nc
        echo ""
        echo "These files will be OVERWRITTEN if the corresponding input files exist."
    else
        echo "Output directory exists but is empty."
    fi
else
    echo "Output directory will be created."
fi

echo "========================================================"
echo "CONFIGURATION SUMMARY"
echo "========================================================"
echo "  Pixel Precip Threshold: ${PIXEL_PRECIP_THRESHOLD}"
echo "  Area Percent Threshold: ${AREA_PERCENT_THRESHOLD}"
echo "  Target Dry:Wet Ratio: ${TARGET_DRY_WET_RATIO}"
echo "  Dask Workers: ${DASK_WORKERS}"
echo "  Python Log Level: ${LOG_LEVEL_PYTHON}"
echo "========================================================"

# --- Prompt for Continuation (optional - remove if running in batch) ---
# read -p "Continue with rebalancing? (y/n): " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "Rebalancing cancelled by user."
#     exit 0
# fi

# --- Build Python Command Arguments ---
CMD_ARGS=(
    "--mswep_ref_file" "${MSWEP_REF_FILE}"
    "--era5_tp_file" "${ERA5_TP_FILE}"
    "--era5_temp_file" "${ERA5_TEMP_FILE}"
    "--era5_tcwv_file" "${ERA5_TCWV_FILE}"
    "--era5_u_file" "${ERA5_U_FILE}"
    "--era5_v_file" "${ERA5_V_FILE}"
    "--era5_r_file" "${ERA5_R_FILE}"
    "--dem_file" "${DEM_FILE}"
    "--output_dir" "${REBALANCED_OUTPUT_DIR}"
    "--pixel_precip_threshold" "${PIXEL_PRECIP_THRESHOLD}"
    "--area_percent_threshold" "${AREA_PERCENT_THRESHOLD}"
    "--target_ratio_dry_wet" "${TARGET_DRY_WET_RATIO}"
    "--dask_workers" "${DASK_WORKERS}"
    "--log_level" "${LOG_LEVEL_PYTHON}"
)

# --- Execute Rebalancing Script ---
echo "========================================================"
echo "EXECUTING PYTHON REBALANCING SCRIPT"
echo "========================================================"
echo "Script: ${PYTHON_SCRIPT}"
echo "Start time: $(date)"

# Print command for debugging
echo "Command being executed:"
printf "python '%s'" "${PYTHON_SCRIPT}"
for arg in "${CMD_ARGS[@]}"; do
  printf " '%s'" "${arg}"
done
printf "\n\n"

# Execute with better error handling
python "${PYTHON_SCRIPT}" "${CMD_ARGS[@]}"
PYTHON_EXIT_CODE=$?

echo ""
echo "========================================================"
echo "PYTHON SCRIPT COMPLETION"
echo "========================================================"
echo "End time: $(date)"
echo "Exit code: ${PYTHON_EXIT_CODE}"

if [ ${PYTHON_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Python rebalancing script failed!"
    echo "Check the error logs above for details."
    exit ${PYTHON_EXIT_CODE}
else
    echo "Python rebalancing script completed successfully."
fi

# --- Final Output Verification ---
echo "========================================================"
echo "VERIFYING OUTPUT FILES"
echo "========================================================"

if [ -d "$REBALANCED_OUTPUT_DIR" ]; then
    output_files=$(ls -1 "$REBALANCED_OUTPUT_DIR"/*.nc 2>/dev/null)
    if [ -n "$output_files" ]; then
        output_count=$(echo "$output_files" | wc -l)
        echo "SUCCESS: Found $output_count output files:"
        ls -lh "$REBALANCED_OUTPUT_DIR"/*.nc
        
        total_size=$(du -sh "$REBALANCED_OUTPUT_DIR" | cut -f1)
        echo ""
        echo "Total output size: $total_size"
        
        # Check which variables were successfully processed
        echo ""
        echo "Variables successfully rebalanced:"
        for file in $output_files; do
            basename_file=$(basename "$file" .nc)
            variable_name=${basename_file%_rebalanced}
            echo " Worked $variable_name"
        done
        
    else
        echo "WARNING: No output files found in $REBALANCED_OUTPUT_DIR"
        echo "This suggests all input files were missing or there was an error."
    fi
else
    echo "ERROR: Output directory $REBALANCED_OUTPUT_DIR was not created!"
fi

echo "========================================================"
echo "Data Rebalancing Job Finished."
echo "Final status: $([ ${PYTHON_EXIT_CODE} -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
date
echo "========================================================"