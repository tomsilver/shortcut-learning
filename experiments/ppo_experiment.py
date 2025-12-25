import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class SimpleGrid(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size=11, start=(0, 0), goal=(5, 5), max_steps=50):
        super().__init__()
        self.size = size
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start.copy()
        self.steps = 0
        return self.pos.copy(), {}

    def step(self, action):
        self.steps += 1

        if action == 0:   # up
            self.pos[1] += 1
        elif action == 1: # down
            self.pos[1] -= 1
        elif action == 2: # right
            self.pos[0] += 1
        elif action == 3: # left
            self.pos[0] -= 1

        self.pos = np.clip(self.pos, 0, self.size - 1)

        terminated = np.array_equal(self.pos, self.goal)
        truncated = self.steps >= self.max_steps

        reward = 1.0 if terminated else 0.0
        if reward == 1:
            print(f"Reached goal in {self.steps} steps.")

        return self.pos.copy(), reward, terminated, truncated, {}

def random_rollout_success(env, n_rollouts=1000):
    successes = 0
    for _ in range(n_rollouts):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                successes += 1
                break
    return successes / n_rollouts

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def ppo_success_probability(
    env_fn,
    n_seeds=10,
    train_steps=1000,
    eval_episodes=5
):
    successes = 0

    for seed in range(n_seeds):
        print("Testing seed", seed)
        env = make_vec_env(env_fn, n_envs=1, seed=seed)

        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            gamma=0.99,
            n_steps=128,
            ent_coef=0.01,
        )

        model.learn(total_timesteps=train_steps)

        # Deterministic evaluation
        eval_env = make_vec_env(env_fn, n_envs=1)

        success = False
        for _ in range(eval_episodes):
            obs = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(action)
                done = dones[0]
                reward = rewards[0]
                if reward > 0:
                    success = True
                    break

        # eval_env = env_fn()   # raw Gymnasium env

        # obs, _ = eval_env.reset()
        # done = False
        # success= 
        # while not done:
        #     action, _ = model.predict(obs, deterministic=True)
        #     obs, reward, terminated, truncated, _ = eval_env.step(action)
        #     done = terminated or truncated
        #     if reward > 0:
        #         success = True
        #         break

        if success:
            successes += 1

    return successes / n_seeds

import matplotlib.pyplot as plt

def run_experiment():
    k = 30
    grid_size = 51
    start = (25, 25)

    distances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # distances = [2]
    num_seeds = 10

    R_vals = []
    P_vals = []

    for d in distances:
        print("Testing distance", d)
        goal = (start[0] + d, start[1])

        def env_fn():
            return SimpleGrid(
                size=grid_size,
                start=start,
                goal=goal,
                max_steps=k
            )

        env = env_fn()

        R = random_rollout_success(env, n_rollouts=2000)
        P = ppo_success_probability(env_fn, n_seeds=num_seeds, train_steps=1000)

        R_vals.append(R)
        P_vals.append(P)

        print(f"d={d}: R={R:.4f}, P={P:.4f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(R_vals, P_vals)
    plt.xlabel("R(s, g): Random rollout success")
    plt.ylabel("P(s, g): PPO convergence probability")
    plt.title("PPO vs Random Exploration")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("ppo_vs_random.png", dpi=300)
    plt.close()



    df = pd.DataFrame({
        "distance": distances,
        "R_random": R_vals,
        "P_ppo": P_vals
    })

    df.to_csv("ppo_vs_random.csv", index=False)

if __name__ == "__main__":
    run_experiment()
