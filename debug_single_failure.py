"""Debug a single failure case in detail."""

from shortcut_learning.problems.obstacle2d_hard.env import Obstacle2DEnv
import numpy as np

print('Debugging single failure case in detail')
print('='*70)

env = Obstacle2DEnv()

# Use seed 103 which we know fails
obs, info = env.reset(seed=103)

robot_x, robot_y = obs[0:2]
robot_w, robot_h = obs[2:4]
block1_x, block1_y = obs[4:6]
block2_x, block2_y = obs[6:8]
block_w, block_h = obs[8:10]

print(f'Initial state:')
print(f'  Robot: ({robot_x:.3f}, {robot_y:.3f})')
print(f'  Block1: ({block1_x:.3f}, {block1_y:.3f})')
print(f'  Block2: ({block2_x:.3f}, {block2_y:.3f})')
print(f'  Block distance: {abs(block1_x - block2_x):.3f}')
print(f'  Robot-Block1 distance: {np.linalg.norm([robot_x - block1_x, robot_y - block1_y]):.3f}')
print()

target_robot_x = block1_x
target_robot_y = block1_y + block_h/2 + robot_h/2

print(f'Target robot position: ({target_robot_x:.3f}, {target_robot_y:.3f})')
print()

last_pos = (robot_x, robot_y)
stuck_count = 0

for step in range(20):  # Only show first 20 steps
    dx = target_robot_x - robot_x
    dy = target_robot_y - robot_y

    # Decide action
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        action = np.array([0.0, 0.0, 1.0])
        action_desc = 'PICKUP'
    elif abs(dy) > 0.01:
        action = np.array([0.0, np.clip(dy, -0.1, 0.1), -1.0])
        action_desc = f'MOVE_Y({action[1]:.3f})'
    else:
        action = np.array([np.clip(dx, -0.1, 0.1), 0.0, -1.0])
        action_desc = f'MOVE_X({action[0]:.3f})'

    obs, reward, term, trunc, _ = env.step(action)
    new_robot_x, new_robot_y = obs[0:2]
    gripper = obs[10]

    # Check if robot moved
    moved = abs(new_robot_x - last_pos[0]) > 0.001 or abs(new_robot_y - last_pos[1]) > 0.001

    print(f'Step {step}: {action_desc:15s}  Robot: ({robot_x:.3f},{robot_y:.3f}) -> ({new_robot_x:.3f},{new_robot_y:.3f})  ', end='')

    if not moved:
        print('⚠️  STUCK (collision?)')
        stuck_count += 1
        if stuck_count > 5:
            print('\\nRobot stuck for 5+ steps, giving up')
            break
    else:
        print(f'Moved {np.linalg.norm([new_robot_x - robot_x, new_robot_y - robot_y]):.3f}')
        stuck_count = 0
        last_pos = (new_robot_x, new_robot_y)

    robot_x, robot_y = new_robot_x, new_robot_y

    if gripper > 0.5:
        print(f'\\n✓ SUCCESS! Picked up block1 in {step+1} steps')
        break

    if term or trunc:
        print(f'\\nEpisode terminated')
        break
else:
    print(f'\\nFailed to pick up after 20 steps')

print()
print('Analyzing collision...')
print(f'  Robot width: {robot_w:.3f}, height: {robot_h:.3f}')
print(f'  Block width: {block_w:.3f}, height: {block_h:.3f}')
print(f'  Robot-block collision threshold: {(robot_w + block_w)/2:.3f}')
print(f'  Block-block collision threshold: {block_w:.3f}')
print(f'  Block1-Block2 horizontal distance: {abs(block1_x - block2_x):.3f}')

if abs(block1_x - block2_x) < block_w:
    print(f'  ⚠️  Blocks are closer than their width!')
    print(f'  When robot is at x={block1_x:.3f} (above block1), distance to block2 = {abs(block1_x - block2_x):.3f}')
    print(f'  Robot collision radius: {(robot_w + block_w)/2:.3f}')
    print(f'  COLLISION! Robot cannot fit between blocks or near them')
