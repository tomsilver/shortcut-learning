#!/bin/bash
#SBATCH --job-name=slap_v2_aggressive  # Job name
#SBATCH --output=outputs/slap_v2_aggressive.out  # Output file
#SBATCH --error=outputs/slap_v2_aggressive.err  # Error file
#SBATCH --time=16:00:00  # Maximum runtime (hh:mm:ss) - much longer for extensive training
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

# Run the SLAP V2 experiment with aggressive training parameters
# Goal: Beat pure_plan baseline (98% success, 25.88 avg episode length)
# Strategy: More training data + more training iterations
python run_single_experiment.py \
    approach.approach_type=slap_v2 \
    policy.policy_type=multi_rl_ppo_v2 \
    system=obstacle2d \
    collection.states_per_node=50 \
    collection.perturbation_steps=15 \
    training.runs_per_shortcut=10 \
    training.max_env_steps=256 \
    policy.n_epochs=10 \
    evaluation.num_episodes=100
