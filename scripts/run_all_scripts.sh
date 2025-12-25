#!/bin/bash

# Define configurations and their overrides
# Each entry in CONFIGS_TO_RUN is a space-separated string:
# "config_name override1=value1 override2=value2 ..."
# Base config name
BASE_CONFIG="gridworld_fixed"

# Define parameter ranges
LATENT_DIMS=(12)
LEARNING_RATES=(1e-3)
GAMMAS=(0.9)
HEURISTIC_TYPES=("crl" "rollouts" "smart_rollouts" "dqn" "cmd")

# Loop through all combinations
for latent_dim in "${LATENT_DIMS[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
      for heuristic_type in "${HEURISTIC_TYPES[@]}"; do

        # Construct the override string
        CONFIG_OVERRIDES="heuristic.type=${heuristic_type} heuristic.latent_dim=${latent_dim} heuristic.learning_rate=${lr} heuristic.gamma=${gamma}"
        
        # Construct the WandB run name
        WANDB_RUN_NAME="${BASE_CONFIG}_HT${heuristic_type}_L${latent_dim}_LR${lr}_G${gamma}"
        
        echo "Submitting SLURM job for config: ${BASE_CONFIG} with overrides: ${CONFIG_OVERRIDES}"
        echo "WandB Run Name: ${WANDB_RUN_NAME}"

        # Export environment variables for the sbatch script
        export CONFIG_OVERRIDES="${CONFIG_OVERRIDES}"
        export WANDB_RUN_NAME="${WANDB_RUN_NAME}"
        
        # Submit the job
        sbatch scripts/run_slap_v2.sh --config-name "${BASE_CONFIG}"

        # Unset environment variables to prevent them from affecting subsequent sbatch calls
        unset CONFIG_OVERRIDES
        unset WANDB_RUN_NAME

        sleep 1 # Add a small delay to avoid overwhelming the SLURM scheduler
      done
    done
  done
done

echo "All SLURM jobs for ${BASE_CONFIG} combinations submitted."