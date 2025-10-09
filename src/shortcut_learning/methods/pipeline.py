"""Training utilities for improvisational approaches."""

import inspect
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from shortcut_learning.configs import (
    ApproachConfig,
    CollectionConfig,
    EvaluationConfig,
    PolicyConfig,
    TrainingConfig,
)
from shortcut_learning.methods.base_approach import BaseApproach
from shortcut_learning.methods.policies.base import Policy
from shortcut_learning.methods.policies.rl_ppo import RLPolicy
from shortcut_learning.methods.pure_rl_approach import PureRLApproach
from shortcut_learning.methods.random_approach import RandomApproach
from shortcut_learning.methods.training_data import TrainingData
from shortcut_learning.problems.base_tamp import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class Metrics:
    """Training and evaluation metrics."""

    success_rate: float
    avg_episode_length: float
    avg_reward: float
    collection_time: float = 0.0
    training_time: float = 0.0
    evaluation_time: float = 0.0


def initialize_policy(
    policy_config: PolicyConfig, seed: int = 42
) -> Policy[ObsType, ActType]:

    if policy_config.policy_type == "rl_ppo":
        return RLPolicy(seed, policy_config)


def initialize_approach(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach_config: ApproachConfig,
    policy_config: PolicyConfig,
    seed: int = 42,
) -> BaseApproach[ObsType, ActType]:
    if approach_config.approach_type == "random":
        return RandomApproach(system, seed, approach_config.approach_name)

    if approach_config.approach_type == "pure_rl":
        policy = initialize_policy(policy_config)
        return PureRLApproach(system, seed, approach_config.approach_name, policy)


def collect_approach(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: BaseApproach[ObsType, ActType],
    collect_config: CollectionConfig,
) -> TrainingData | None:
    # return collect_training_data(collect_config, ...)
    return None


def train_approach(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: BaseApproach[ObsType, ActType],
    train_config: TrainingConfig,
    train_data: TrainingData | None,
) -> BaseApproach[ObsType, ActType]:

    approach.train(train_data, train_config)

    return approach


def evaluate_approach(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: BaseApproach[ObsType, ActType],
    eval_config: EvaluationConfig,
) -> Metrics:

    # Run evaluation episodes
    print(f"\nEvaluating policy on {system.name}...")
    rewards = []
    lengths = []
    successes = []

    for episode in range(eval_config.num_episodes):
        print(f"\nEvaluation Episode {episode + 1}/{eval_config.num_episodes}")
        reward, length, success = run_evaluation_episode(
            system,
            approach,
            eval_config,
            episode_num=episode,
        )
        rewards.append(reward)
        lengths.append(length)
        successes.append(success)

        print(f"Current Success Rate: {sum(successes)/(episode+1):.2%}")
        print(f"Current Avg Episode Length: {np.mean(lengths):.2f}")
        print(f"Current Avg Reward: {np.mean(rewards):.2f}")

    return Metrics(
        success_rate=float(sum(successes) / len(successes)),
        avg_episode_length=float(np.mean(lengths)),
        avg_reward=float(np.mean(rewards)),
    )


def collect_train_evaluate_approach(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: BaseApproach[ObsType, ActType],
    collect_config: CollectionConfig,
    train_config: TrainingConfig,
    eval_config: EvaluationConfig,
):
    start_time = time.time()

    train_data = collect_approach(system, approach, collect_config)

    collect_time = time.time()

    trained_approach = train_approach(system, approach, train_config, train_data)

    train_time = time.time()

    metrics = evaluate_approach(system, trained_approach, eval_config)

    eval_time = time.time()

    metrics.collection_time = collect_time - start_time
    metrics.training_time = train_time - collect_time
    metrics.evaluation_time = eval_time - train_time

    return metrics


def run_evaluation_episode(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: BaseApproach[ObsType, ActType],
    eval_config: EvaluationConfig,
    episode_num: int = 0,
) -> tuple[float, int, bool]:
    """Run single evaluation episode."""
    # Set up rendering if available
    render_mode = getattr(system.env, "render_mode", None)
    can_render = render_mode is not None
    if eval_config.render and can_render:
        video_folder = Path(f"videos/{system.name}_{approach.name}_eval")
        video_folder.mkdir(parents=True, exist_ok=True)

        # Record only the base environment, not the planning environment
        recording_env = deepcopy(system.env)
        system.env = RecordVideo(
            recording_env,
            str(video_folder),
            episode_trigger=lambda _: True,
            name_prefix=f"episode_{episode_num}",
            disable_logger=True,
        )

    obs, info = system.reset()
    if (
        hasattr(approach, "reset")
        and "select_random_goal" in inspect.signature(approach.reset).parameters
    ):
        step_result = approach.reset(obs, info, select_random_goal=eval_config.select_random_goal)  # type: ignore[call-arg]  # pylint: disable=line-too-long
    else:
        step_result = approach.reset(obs, info)

    total_reward = 0.0
    step_count = 0
    success = False

    # Execute first action from the reset
    obs, reward, terminated, truncated, info = system.env.step(step_result.action)
    total_reward += float(reward)
    step_count += 1
    if step_result.terminate or terminated or truncated:
        success = step_result.terminate or terminated
        if eval_config.render and can_render:
            cast(Any, system.env).close()
            system.env = recording_env
        return total_reward, step_count, success

    # Rest of steps
    for _ in range(1, eval_config.max_steps):
        step_result = approach.step(obs, total_reward, False, False, info)
        obs, reward, terminated, truncated, info = system.env.step(step_result.action)
        total_reward += float(reward)
        step_count += 1
        if step_result.terminate or terminated or truncated:
            success = step_result.terminate or terminated
            break

    if eval_config.render and can_render:
        cast(Any, system.env).close()
        system.env = recording_env

    return total_reward, step_count, success


def pipeline_from_configs(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach_config: ApproachConfig,
    policy_config: PolicyConfig,
    collect_config: CollectionConfig,
    train_config: TrainingConfig,
    eval_config: EvaluationConfig,
) -> Metrics:
    approach = initialize_approach(system, approach_config, policy_config)

    return collect_train_evaluate_approach(
        system, approach, collect_config, train_config, eval_config
    )
