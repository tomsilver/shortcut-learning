#!/bin/bash
#SBATCH --job-name=pure_plan_experiment  # Job name
#SBATCH --output=outputs/pure_plan_experiment.out  # Output file
#SBATCH --error=outputs/pure_plan_experiment.err  # Error file
#SBATCH --time=02:00:00  # Maximum runtime (hh:mm:ss)
#SBATCH --partition=normal  # Partition to submit to
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=4  # Number of CPU cores per task
#SBATCH --mem=16G  # Memory per node
#SBATCH --gres=gpu:1  # Request one GPU

# Load necessary modules
module load python/3.8  # Adjust to your Python version
module load cuda/11.3  # Adjust to your CUDA version

# Activate your virtual environment
source /home/de7281/thesis/shortcut-learning/.venv/bin/activate  # Adjust to your virtual environment path

# Navigate to the experiment directory
cd /home/de7281/thesis/shortcut-learning/experiments

# Run the experiment with Hydra
python run_single_experiment.py approach.approach_type=slap system=obstacle2d training.skip_train=True
