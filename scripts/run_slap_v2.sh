#!/bin/bash

#SBATCH --nodes=1                                      ## Node count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48                              ## Give the single process lots of CPU

#SBATCH --mem=100G                                      ## RAM per node
#SBATCH --time=2:00:00                                  ## Walltime
#SBATCH --gres=gpu:1                                    ## Number of GPUs
#SBATCH --exclude=neu[301,306]                          ## Exclude some nodes
#SBATCH --job-name=train_slap                            ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Stdout File
#SBATCH --error=slurm_outputs/%x/err_log_%x_%j.err      ## Stderr File

set -euo pipefail

# If not running under Slurm, auto-submit this script to avoid login-node execution.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[INFO] Not inside a Slurm job. Submitting via sbatch to avoid running on the login node..."
  exec sbatch "$0"
fi


cd /n/fs/jborz/projects/slap
module load cudatoolkit/12.6

# Use the correct Python executable from the virtual environment
PYTHON_EXE="/n/fs/jborz/projects/slap/.venv/bin/python"

# Add virtual environment bin to PATH so ninja and other tools are accessible
export PATH="/n/fs/jborz/projects/slap/.venv/bin:$PATH"

# CUDA memory allocator: reduce fragmentation for long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SLURM-specific threading
export SLURM_CPUS_PER_TASK=48
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ensure output directories exist
mkdir -p slurm_outputs/train_slap

python -m experiments.slap_train_pipeline_v2 --config-name gridworld_fixed ${CONFIG_OVERRIDES} seed=42 | tee run_log.txt