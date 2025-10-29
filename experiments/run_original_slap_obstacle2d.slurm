#!/bin/bash
#SBATCH --job-name=original_slap_obstacle2d
#SBATCH --output=/scratch/gpfs/TSILVER/de7281/shortcut_learning/original_slap_%j.out
#SBATCH --error=/scratch/gpfs/TSILVER/de7281/shortcut_learning/original_slap_%j.err
#SBATCH --time=08:00:00  # Maximum runtime
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load necessary modules
module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python/3.8
module load cuda/11.3 2>/dev/null || true

# Set up paths
ORIGINAL_CODE_DIR="/home/de7281/thesis/shortcut-learning-isabel/SLAP/code"
OUR_VENV_DIR="/home/de7281/thesis/shortcut-learning/.venv"

# Use our existing venv which already has all dependencies
echo "Using existing venv with all dependencies..."
source "$OUR_VENV_DIR/bin/activate"

# Verify dependencies are available
if ! python -c "import torch; import numpy; import gymnasium" 2>/dev/null; then
    echo "ERROR: Required dependencies not found!"
    exit 1
fi

echo "Dependencies verified!"

# Navigate to the code directory
cd "$ORIGINAL_CODE_DIR"

echo "========================================="
echo "Running Original SLAP Paper Code"
echo "Environment: obstacle2d"
echo "Episodes per scenario: 1000"
echo "========================================="
echo ""

# Run the experiment with parameters matching the paper
# From config.json: episodes_per_scenario=1000, max_training_steps_per_shortcut=50
python -m src.tamp_improv.run_experiments --system=obstacle2d --episodes=1000

echo ""
echo "========================================="
echo "Experiment Complete!"
echo "========================================="
