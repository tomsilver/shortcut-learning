# SLAP Paper Comparison: Original vs Our Implementation

## Objective
Match the paper's ~17 steps performance by using the exact same parameters as the original SLAP implementation.

## Key Findings from Original SLAP Code

### 1. Initialization (Most Important!)
**Original paper uses DISCRETE randomization:**
- Block 1 (target): `x ∈ {0.0, 1.0}` (only 2 positions!)
- Block 2 (obstacle): `x ∈ {target_left, target_center, target_right}` (only 3 positions!)
- **Total: Only 6 possible initial states**

**Our obstacle2d_hard used CONTINUOUS randomization:**
- This creates scenarios where blocks can be very close together (<0.25 apart)
- Caused 100% skill failures in these cases
- We've now switched back to discrete initialization (regular obstacle2d)

### 2. Training Parameters from Original Code
From `/shortcut-learning-isabel/SLAP/code/train_data/Obstacle2DTAMPSystem/config.json`:

```json
{
  "episodes_per_scenario": 1000,
  "max_training_steps_per_shortcut": 50,
  "max_steps": 50,
  "num_rollouts_per_node": 1000,
  "max_steps_per_rollout": 100,
  "shortcut_success_threshold": 1
}
```

### 3. Skills Implementation
The original paper's `PickUpSkill` (lines 149-156 in `obstacle2d.py`) includes collision avoidance:
```python
# If too close to the other block, move away first
if (
    np.isclose(robot_y, other_block_y, atol=1e-3)
    and abs(robot_x - other_block_x) < (robot_width + block_width) / 2
    and not np.isclose(robot_x, other_block_x, atol=1e-3)
):
    dx = np.clip(robot_x - other_block_x, -0.1, 0.1)
    return np.array([dx, 0.0, -1.0])
```

**Our implementation already has this same logic!** (system.py lines 150-157)

## Experiment Results Timeline

### Previous Experiments
1. **Job 1710777** (100 episodes, 128 steps): 24.52 steps ❌
2. **Job 1711805** (100 episodes, 256 steps): 24.52 steps ❌
   - Shortcut 11 (2→4): Improved from 0% → 100% success by episode 340
   - Confirms more training episodes help!

### Current Experiments
1. **Job 1720760** (Our implementation):
   - Parameters: `runs_per_shortcut=1000`, `max_env_steps=50`
   - System: `obstacle2d` (discrete initialization)
   - Status: Running
   - Target: ~17 steps ✓

2. **Job 1720914** (Original SLAP code):
   - Parameters: `episodes=1000` (same as above)
   - Running original paper's code directly
   - Status: Pending
   - Target: ~17 steps (verification)

## Parameter Comparison

| Parameter | Previous | Paper/Current | Notes |
|-----------|----------|---------------|-------|
| `runs_per_shortcut` | 100 | **1000** | 10x more training! |
| `max_env_steps` | 256 | **50** | Shorter episodes |
| Initialization | Continuous (hard) | **Discrete** | Only 6 states |
| `states_per_node` | 100 | 100 | Same |
| `num_episodes` (eval) | 100 | 100 | Same |

## Expected Outcomes

If our reimplementation matches the paper:
- ✅ Average episode length: ~17 steps (down from 24.52)
- ✅ Success rate: 100%
- ✅ Reliable shortcuts learned with 1000 episodes

If the original code also gets ~17 steps:
- ✅ Confirms our understanding is correct
- ✅ Can study their debug outputs for insights
- ✅ Can move forward with improvements

## Next Steps

1. Wait for both experiments to complete (~8 hours each)
2. Compare results between our implementation and original
3. If both achieve ~17 steps, move on to improvements
4. If there are still differences, investigate debug outputs from original code
