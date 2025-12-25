#!/bin/bash
# User-specific paths configuration
# This file is git-ignored - modify for your setup

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
