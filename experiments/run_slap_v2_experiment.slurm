#!/bin/bash
#SBATCH --job-name=slap_v2_experiment  # Job name
#SBATCH --output=/scratch/gpfs/TSILVER/de7281/shortcut_learning/slap_v2_experiment_%j.out  # Output file with job ID
#SBATCH --error=/scratch/gpfs/TSILVER/de7281/shortcut_learning/slap_v2_experiment_%j.err  # Error file with job ID
#SBATCH --time=08:00:00  # Maximum runtime (hh:mm:ss) - longer for training
#SBATCH --partition=normal  # Partition to submit to
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=8  # More CPU cores for parallel training
#SBATCH --mem=32G  # More memory for larger graphs and training
#SBATCH --gres=gpu:1  # Request one GPU

# Load necessary modules
module load intel-mkl/2024.2  # Load Intel MKL for numpy

# Set library paths for IKFast (if using obstacle_tower)
export LAPACK_DIR="/usr/lib64"
export LIBGFORTRAN_DIR="/usr/lib64"
export BLAS_DIR="/usr/lib64"

# Activate virtual environment (created in scratch, can be used from anywhere)
source /scratch/gpfs/TSILVER/de7281/shortcut-learning-venv/bin/activate

# Navigate to the experiment directory
cd /home/de7281/thesis/shortcut-learning/experiments

# Run the SLAP V2 experiment with Hydra
# Override defaults to use slap_v2 approach with multi_rl_ppo_v2 policy
# Increase collection and training for better performance
python run_single_experiment.py \
    approach.approach_type=slap_v2 \
    policy.policy_type=multi_rl_ppo_v2 \
    system=obstacle2d \
    collection.states_per_node=100 \
    training.runs_per_shortcut=1000 \
    training.max_env_steps=50 \
    training.training_record_interval=10 \
    evaluation.num_episodes=100

# Note: Output files are already in /scratch/gpfs/TSILVER/de7281/shortcut_learning/
# Hydra creates its own output directory structure in experiments/outputs/
echo "Experiment complete! Output files are in /scratch/gpfs/TSILVER/de7281/shortcut_learning/"
echo "Hydra outputs are in: $(ls -td experiments/outputs/20*/*/ 2>/dev/null | head -1)"
