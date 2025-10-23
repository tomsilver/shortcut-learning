# SLAP V2 Experiments

## Goal
Beat the pure planning baseline with SLAP V2 + multi-policy RL training.

## Baseline (Pure Planning)
- **Success Rate**: 98.00%
- **Avg Episode Length**: 25.88 steps
- **Approach**: Uses only classical planning, no learned shortcuts

## SLAP V2 Strategy
The SLAP V2 approach aims to reduce episode length by learning shortcuts that skip multiple classical planning steps. Key improvements over V1:
1. **Fixed planning graph**: Built once and reused, expanded adaptively
2. **More training data per shortcut**: Collect multiple states per node (vs V1's single state per episode)
3. **Multi-policy training**: One specialized policy per shortcut (vs single policy for all)
4. **Edge cost computation**: Measured during training, reused during evaluation

## Experiments

### Experiment 1: Standard SLAP V2
**File**: `run_slap_v2_experiment.slurm`

**Parameters**:
- `approach.approach_type=slap_v2`
- `policy.policy_type=multi_rl_ppo_v2`
- `collection.states_per_node=20` (2x default)
- `collection.perturbation_steps=10` (2x default)
- `training.runs_per_shortcut=5` (5x default)
- `training.max_env_steps=128` (2x default)
- `evaluation.num_episodes=100`

**Expected Runtime**: ~8 hours

**Strategy**: Moderate increase in collection and training to demonstrate that shortcuts can reduce episode length while maintaining high success rate.

**Success Criteria**:
- Success rate ≥ 95%
- Avg episode length < 25.88 (better than pure planning)

### Experiment 2: Aggressive SLAP V2
**File**: `run_slap_v2_aggressive_experiment.slurm`

**Parameters**:
- `approach.approach_type=slap_v2`
- `policy.policy_type=multi_rl_ppo_v2`
- `collection.states_per_node=50` (5x default)
- `collection.perturbation_steps=15` (3x default)
- `training.runs_per_shortcut=10` (10x default)
- `training.max_env_steps=256` (4x default)
- `policy.n_epochs=10` (2x default)
- `evaluation.num_episodes=100`

**Expected Runtime**: ~16 hours

**Strategy**: Aggressive training to maximize shortcut quality. More training data and iterations should lead to highly effective shortcuts that significantly reduce episode length.

**Success Criteria**:
- Success rate ≥ 95%
- Avg episode length < 20.00 (significant improvement over baseline)

## How to Run

```bash
# Standard experiment
sbatch run_slap_v2_experiment.slurm

# Aggressive experiment
sbatch run_slap_v2_aggressive_experiment.slurm
```

## How to Check Results

```bash
# Check job status
squeue -u $USER

# View output (live)
tail -f outputs/slap_v2_experiment.out

# View final metrics
grep -A 5 "Metrics:" outputs/slap_v2_experiment.out

# Compare to baseline
echo "Baseline: 98% success, 25.88 avg length"
grep -A 5 "Metrics:" outputs/slap_v2_experiment.out | grep "success_rate\|avg_episode_length"
```

## Expected Results

If the paper's claims hold, we should see:
1. **Success rate**: Comparable to baseline (~95-98%)
2. **Episode length**: Significantly shorter than baseline (target: 15-20 steps)
3. **Training time**: Longer than baseline (building comprehensive graph + training policies)
4. **Evaluation time**: Shorter than baseline (reusing fixed graph + fast shortcut execution)

The key insight: learned shortcuts should allow the agent to skip multiple planning steps, reducing the number of actions needed to reach the goal.

## Configuration Details

All experiments use the default configs from `experiments/conf/` with overrides specified in the SLURM files. Key config files:
- `approach/default.yaml`: Approach settings (planner, max steps, etc.)
- `policy/default.yaml`: Policy settings (learning rate, batch size, etc.)
- `collection/default.yaml`: Collection settings (states per node, perturbations)
- `training/default.yaml`: Training settings (runs per shortcut, max steps)
- `evaluation/default.yaml`: Evaluation settings (num episodes, max steps)
