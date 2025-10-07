"""Configs for training."""

import inspect
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np
import torch

@dataclass
class ApproachConfig:
    """Configuration for general approach"""

    approach_type: str

    approach_name: str

@dataclass
class PolicyConfig:
    """Configuration for RL policy."""

    policy_type: str = "rl_ppo"

    learning_rate: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 5
    gamma: float = 0.99
    ent_coef: float = 0.01
    device: str = "cuda"



@dataclass
class TrainingConfig:
    """Configuration for training."""

    # General settings
    seed: int = 42
    num_episodes: int = 50
    max_steps: int = 100
    max_training_steps_per_shortcut: int = 50

    # Collection settings
    collect_episodes: int = 100
    episodes_per_scenario: int = 1
    force_collect: bool = False

    # Save/Load settings
    save_dir: str = "trained_policies"
    training_data_dir: str = "training_data"

    # Visualization settings
    render: bool = False
    record_training: bool = False
    training_record_interval: int = 50

    # Device settings
    device: str = "cuda"
    batch_size: int = 32

    # Policy-specific settings
    policy_config: dict[str, Any] | None = None

    # Shortcut information
    shortcut_info: list[dict[str, Any]] = field(default_factory=list)

    # Context size for augmenting observations
    max_atom_size: int = 12

    # Goal-conditioned training settings
    success_threshold: float = 0.01
    success_reward: float = 10.0
    step_penalty: float = -0.5

    # Action scaling
    action_scale: float = 1.0

    def get_training_data_path(self, system_name: str) -> Path:
        """Get path for training data for specific system."""
        return Path(self.training_data_dir) / system_name

@dataclass
class EvaluationConfig:

    seed: int = 42

    render: bool = False
    select_random_goal: bool = False
    max_steps: int = 100
    num_episodes: int = 100

@dataclass
class CollectionConfig:

    seed: int = 42
