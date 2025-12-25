#!/bin/bash
# User-specific paths configuration
# Copy this file to user_paths.sh and modify for your setup
# user_paths.sh is git-ignored so each user can have their own configuration

# Virtual environment path
export SHORTCUT_VENV="/scratch/gpfs/TSILVER/de7281/shortcut-learning-venv"

# Output directory for SLURM jobs
export SHORTCUT_OUTPUT_DIR="/scratch/gpfs/TSILVER/de7281/shortcut_learning"

# Collection cache directory
export SHORTCUT_COLLECTION_CACHE="/scratch/gpfs/TSILVER/de7281/collection_cache"

# Library paths (for PyBullet/IKFast if needed)
export LAPACK_DIR="/usr/lib64"
export LIBGFORTRAN_DIR="/usr/lib64"
export BLAS_DIR="/usr/lib64"

# Optional: Module names to load (space-separated)
export SHORTCUT_MODULES="intel-mkl/2024.2"
