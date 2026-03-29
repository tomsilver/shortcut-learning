#!/bin/bash
# User-specific paths configuration
# This file is git-ignored - modify for your setup

# Virtual environment path
export SHORTCUT_VENV="$HOME/miniconda3/envs/slap_env"

# Output directory for SLURM jobs
export SHORTCUT_OUTPUT_DIR="/n/fs/recbench/slap_outputs"

# Collection cache directory
export SHORTCUT_COLLECTION_CACHE="/n/fs/recbench/slap_training_data"

# Library paths (for PyBullet/IKFast if needed)
export LAPACK_DIR="/usr/lib64"
export LIBGFORTRAN_DIR="/usr/lib64"
export BLAS_DIR="/usr/lib64"

# Optional: Module names to load (space-separated)
export SHORTCUT_MODULES="anaconda3/2024.02"
