#!/bin/bash
#SBATCH --job-name=ml_downscaling_h100
#SBATCH --output=/u/morenodu/ml_downscaling/code/logs/ml_train_output_%j.log
#SBATCH --error=/u/morenodu/ml_downscaling/code/logs/ml_train_error_%j.log
#SBATCH --time=0-06:00:00           
#SBATCH --partition=gpu               
#SBATCH --cpus-per-task=16          # Max 16 cores on gpu-shared
#SBATCH --mem=64G           

echo "Starting ML downscaling training job on H100"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: gpu"
date

# --- Configuration ---
CONDA_ENV_NAME="ml_downscaling"
PROJECT_DIR="/u/morenodu/ml_downscaling"
OUTPUT_DIR="${PROJECT_DIR}/output"
LOG_DIR="${PROJECT_DIR}/code/logs"

# Your Python script name - UPDATE THIS
PYTHON_SCRIPT="${PROJECT_DIR}/code/unet_comprehensive_crar.py"

# --- Setup ---
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Loading Miniconda module..."
module load nvidia/nvhpc/24.1
module load miniconda
if [ $? -ne 0 ]; then echo "ERROR: Failed to load miniconda module."; exit 1; fi

echo "Initializing Conda shell integration..."
eval "$(conda shell.bash hook)"
if [ $? -ne 0 ]; then echo "ERROR: Failed to initialize conda shell hook."; exit 1; fi

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then echo "ERROR: Failed to activate conda environment '${CONDA_ENV_NAME}'."; exit 1; fi

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=16                    # Use all 16 cores
export CUDA_VISIBLE_DEVICES=0               # Use the allocated GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Optimize GPU memory

# Display GPU information
echo "=== GPU Information ==="
nvidia-smi
echo "CUDA Version: $(nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-)"
echo "=== Python GPU Check ==="
python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# --- Run the ML training script ---
echo "=== Starting ML Training ==="
echo "Script: ${PYTHON_SCRIPT}"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

# Run with nice to ensure good cluster citizenship
nice -n 10 python "${PYTHON_SCRIPT}"

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "=== Training completed successfully ==="
else
    echo "ERROR: Training failed with exit code $exit_code"
    exit $exit_code
fi

echo "End time: $(date)"
echo "=== Job completed ==="