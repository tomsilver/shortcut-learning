"""Debug script to identify why skills fail with continuous positions."""

from shortcut_learning.problems.obstacle2d_hard.env import Obstacle2DEnv
import numpy as np

print('Testing manual PickUp execution in obstacle2d_hard')
print('='*70)

env = Obstacle2DEnv()

failures = []
successes = []

for trial in range(30):
    obs, info = env.reset(seed=100+trial)
    initial_obs = obs.copy()

    # Extract initial state
    robot_x, robot_y = obs[0:2]
    robot_w, robot_h = obs[2:4]
    block1_x, block1_y = obs[4:6]
    block2_x, block2_y = obs[6:8]
    block_w, block_h = obs[8:10]
    target_x, target_y = obs[11:13]
    target_w, target_h = obs[13:15]

    # Manually implement PickUp logic for block1
    # Target: get robot above block1 and pick it up

    success = False
    steps = 0
    max_steps = 100

    for step in range(max_steps):
        # Target position: above block1
        target_robot_x = block1_x
        target_robot_y = block1_y + block_h/2 + robot_h/2

        # Move toward target position
        dx = target_robot_x - robot_x
        dy = target_robot_y - robot_y

        # Check if aligned
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            # Try to pick up
            action = np.array([0.0, 0.0, 1.0])
        elif abs(dy) > 0.01:
            # Move in y first
            action = np.array([0.0, np.clip(dy, -0.1, 0.1), -1.0])
        else:
            # Move in x
            action = np.array([np.clip(dx, -0.1, 0.1), 0.0, -1.0])

        obs, reward, term, trunc, _ = env.step(action)
        robot_x, robot_y = obs[0:2]
        gripper = obs[10]
        steps += 1

        # Check if picked up (gripper status > 0)
        if gripper > 0.5:
            success = True
            break

        if term or trunc:
            break

    # Analyze result
    block_dist = abs(initial_obs[4] - initial_obs[6])

    if success:
        successes.append({
            'trial': trial,
            'steps': steps,
            'block_dist': block_dist,
            'block1_x': initial_obs[4],
            'block2_x': initial_obs[6],
        })
    else:
        failures.append({
            'trial': trial,
            'steps': steps,
            'block_dist': block_dist,
            'robot_x': initial_obs[0],
            'block1_x': initial_obs[4],
            'block2_x': initial_obs[6],
            'target_x': initial_obs[11],
        })

        print(f'Trial {trial}: FAILED after {steps} steps')
        print(f'  Initial Robot x={initial_obs[0]:.3f}')
        print(f'  Block1 x={initial_obs[4]:.3f}')
        print(f'  Block2 x={initial_obs[6]:.3f}')
        print(f'  Block distance: {block_dist:.3f}')
        print(f'  Target x={initial_obs[11]:.3f}, range=[{initial_obs[11]-0.1:.3f}, {initial_obs[11]+0.1:.3f}]')

        # Check potential issues
        if block_dist < 0.25:
            print(f'  ⚠️  Blocks too close!')
        if initial_obs[4] < 0.15 or initial_obs[4] > 0.85:
            print(f'  ⚠️  Block1 near edge!')
        print()

print(f'\nSummary:')
print(f'  Successes: {len(successes)}/30 ({len(successes)/30*100:.1f}%)')
print(f'  Failures: {len(failures)}/30 ({len(failures)/30*100:.1f}%)')

if successes:
    avg_steps = np.mean([s['steps'] for s in successes])
    print(f'  Average steps for success: {avg_steps:.1f}')

if failures:
    print(f'\nFailure analysis:')
    avg_block_dist = np.mean([f['block_dist'] for f in failures])
    print(f'  Average block distance in failures: {avg_block_dist:.3f}')
    close_blocks = sum(1 for f in failures if f['block_dist'] < 0.25)
    print(f'  Failures with blocks < 0.25 apart: {close_blocks}/{len(failures)}')
