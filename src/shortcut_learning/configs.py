"""Configs for training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ApproachConfig:
    """Configuration for general approach."""

    approach_type: str

    approach_name: str

    seed: int = 42

    planner_id: str = "pyperplan"
    max_skill_steps: int = 200
    max_atom_size: int = 12
    use_context_wrapper: bool = False

    debug_videos: bool = False


@dataclass
class PolicyConfig:
    """Configuration for RL policy."""

    policy_type: str = "rl_ppo"

    learning_rate: float = 1e-4
    batch_size: int = 16
    steps_per_episode: int = 64
    n_epochs: int = 1
    gamma: float = 0.99
    ent_coef: float = 0.01
    device: str = "cuda"


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # General settings
    seed: int = 42
    runs_per_shortcut: int = 1

    max_env_steps: int = 64

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

    skip_train: bool = False

    def get_training_data_path(self, system_name: str) -> Path:
        """Get path for training data for specific system."""
        return Path(self.training_data_dir) / system_name


@dataclass
class EvaluationConfig:
    """An evaluation config."""

    seed: int = 42

    render: bool = False
    select_random_goal: bool = False
    max_steps: int = 64
    num_episodes: int = 100


@dataclass
class CollectionConfig:
    """Config for data collection."""

    seed: int = 42
    max_shortcuts_per_graph: int = 100
    use_random_rollouts: bool = False
    num_rollouts_per_node: int = 50
    max_steps_per_rollout: int = 50
    shortcut_success_threshold: int = 1
    action_scale: float = 1.0
    collect_episodes: int = 10

    # V2 collection parameters
    states_per_node: int = 10  # Number of diverse states to collect per node
    perturbation_steps: int = 5  # Random steps to apply for state perturbation

    skip_collect: bool = False
