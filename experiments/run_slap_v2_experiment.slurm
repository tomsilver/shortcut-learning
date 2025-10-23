#!/bin/bash
#SBATCH --job-name=slap_v2_experiment  # Job name
#SBATCH --output=outputs/slap_v2_experiment.out  # Output file
#SBATCH --error=outputs/slap_v2_experiment.err  # Error file
#SBATCH --time=08:00:00  # Maximum runtime (hh:mm:ss) - longer for training
#SBATCH --partition=normal  # Partition to submit to
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=8  # More CPU cores for parallel training
#SBATCH --mem=32G  # More memory for larger graphs and training
#SBATCH --gres=gpu:1  # Request one GPU

# Load necessary modules
module load python/3.8  # Adjust to your Python version
module load cuda/11.3  # Adjust to your CUDA version

# Activate your virtual environment
source /home/de7281/thesis/shortcut-learning/.venv/bin/activate  # Adjust to your virtual environment path

# Navigate to the experiment directory
cd /home/de7281/thesis/shortcut-learning/experiments

# Run the SLAP V2 experiment with Hydra
# Override defaults to use slap_v2 approach with multi_rl_ppo_v2 policy
# Increase collection and training for better performance
python run_single_experiment.py \
    approach.approach_type=slap_v2 \
    policy.policy_type=multi_rl_ppo_v2 \
    system=obstacle2d \
    collection.states_per_node=20 \
    collection.perturbation_steps=10 \
    training.runs_per_shortcut=5 \
    training.max_env_steps=128 \
    evaluation.num_episodes=100
