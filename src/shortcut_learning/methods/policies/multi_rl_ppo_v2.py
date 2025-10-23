"""Multi-Policy RL implementation for V2 (simplified)."""

from typing import TypeVar

import gymnasium as gym
import numpy as np
import torch

from shortcut_learning.configs import PolicyConfig, TrainingConfig
from shortcut_learning.methods.policies.base import Policy, PolicyContext
from shortcut_learning.methods.policies.rl_ppo import (
    RLPolicy,
    TrainingProgressCallback,
)
from shortcut_learning.methods.training_data import ShortcutTrainingData
from shortcut_learning.methods.wrappers import SLAPWrapperV2
from shortcut_learning.utils.gpu_parallel import GPUParallelTrainer

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def train_single_policy_v2(
    policy: RLPolicy,
    env: gym.Env,
    train_data: ShortcutTrainingData,
    train_config: TrainingConfig,
    policy_key: str | None = None,
):
    """Train a single policy for V2 (used in parallel training).

    Args:
        policy: The RLPolicy to train
        env: Wrapped environment (SLAPWrapperV2)
        train_data: Training data for this shortcut
        train_config: Training configuration
        policy_key: Optional key for logging (e.g., "shortcut_0")

    Returns:
        True on success
    """
    callback = TrainingProgressCallback(
        check_freq=train_config.training_record_interval,
        early_stopping=True,
        early_stopping_patience=1,
        early_stopping_threshold=0.8,
        policy_key=policy_key,
    )
    policy.train(env, train_config, train_data, callback=callback)
    return True


class MultiRLPolicyV2(Policy[ObsType, ActType]):
    """Simplified multi-policy for V2 architecture.

    Key simplifications vs V1:
    - Uses shortcut_id as policy key (no complex signature matching)
    - No object substitution logic
    - Simpler grouping based on shortcut pairs in training data
    - Each shortcut gets its own policy
    """

    def __init__(self, seed: int, config: PolicyConfig) -> None:
        """Initialize with a seed and config."""
        super().__init__(seed)
        self.config = config
        self.policies: dict[int, RLPolicy] = {}  # shortcut_id -> policy
        self._active_shortcut_id: int | None = None
        self._current_context: PolicyContext | None = None

        # Mapping from (source_node_id, target_node_id) -> shortcut_id
        self._node_pair_to_shortcut: dict[tuple[int, int], int] = {}

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training."""
        return True

    def initialize(self, env: gym.Env) -> None:
        """Initialize the policy."""
        # Not needed for V2 - policies are initialized during training
        pass

    def configure_context(self, context: PolicyContext) -> None:
        """Configure policy with context information (used during execution)."""
        self._current_context = context

        # Get source and target node IDs from context
        source_id = context.info.get("source_node_id")
        target_id = context.info.get("target_node_id")

        if source_id is not None and target_id is not None:
            node_pair = (source_id, target_id)
            if node_pair in self._node_pair_to_shortcut:
                self._active_shortcut_id = self._node_pair_to_shortcut[node_pair]
            else:
                self._active_shortcut_id = None
        else:
            self._active_shortcut_id = None

    def can_initiate(self) -> bool:
        """Check if we can handle the current context."""
        if not self._current_context:
            return False

        source_id = self._current_context.info.get("source_node_id")
        target_id = self._current_context.info.get("target_node_id")

        if source_id is None or target_id is None:
            return False

        return (source_id, target_id) in self._node_pair_to_shortcut

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from the appropriate policy."""
        if self._active_shortcut_id is None:
            raise ValueError("No active shortcut for current context")

        if self._active_shortcut_id not in self.policies:
            raise ValueError(f"No trained policy for shortcut {self._active_shortcut_id}")

        return self.policies[self._active_shortcut_id].get_action(obs)

    def train(
        self,
        env: gym.Env,
        train_config: TrainingConfig,
        train_data: ShortcutTrainingData,
        save_dir: str | None = None,
    ) -> None:
        """Train multiple specialized policies - one per shortcut.

        Supports parallel training across multiple GPUs if available.

        Args:
            env: Base environment (will be wrapped per-shortcut)
            train_config: Training configuration
            train_data: V2 training data with shortcuts
            save_dir: Optional directory to save models
        """
        print("\n=== Training Multi-Policy RL (V2) ===")
        print(f"Number of shortcuts: {len(train_data)}")
        print(f"Total training examples: {train_data.num_training_examples()}")

        # Get the perceiver from the environment
        perceiver = self._extract_perceiver(env)

        # Prepare all policies and environments for training
        policies_to_train = {}

        for shortcut_id, (source_node, target_node) in enumerate(train_data.shortcuts):
            # Store mapping
            self._node_pair_to_shortcut[(source_node.id, target_node.id)] = shortcut_id

            # Create wrapper for this specific shortcut
            single_shortcut_data = ShortcutTrainingData(
                shortcuts=[(source_node, target_node)],
                config=train_data.config,
            )

            # Unwrap to get base env
            base_env = self._get_base_env(env)

            # Create wrapper for this shortcut
            wrapped_env = SLAPWrapperV2(
                base_env,
                perceiver,
                max_episode_steps=train_config.max_env_steps,
                step_penalty=-0.1,
                achievement_bonus=1.0,
            )
            wrapped_env.configure_training(single_shortcut_data)

            # Create policy if it doesn't exist
            if shortcut_id not in self.policies:
                self.policies[shortcut_id] = RLPolicy(self._seed + shortcut_id, self.config)

            # Store for parallel training
            policies_to_train[shortcut_id] = (
                self.policies[shortcut_id],
                wrapped_env,
                single_shortcut_data,
                train_config,
            )

        # Decide whether to train in parallel or sequentially
        if (
            len(policies_to_train) > 1
            and torch.cuda.is_available()
        ):
            num_gpus = torch.cuda.device_count()
            print(f"\nUsing parallel training with {num_gpus} GPU(s)")
            self._train_parallel(policies_to_train, save_dir)
        else:
            print("\nTraining policies sequentially")
            self._train_sequential(policies_to_train)

        print(f"\n=== Completed training {len(self.policies)} policies ===")

    def _extract_perceiver(self, env: gym.Env):
        """Extract perceiver from environment by unwrapping."""
        if hasattr(env, 'perceiver'):
            return env.perceiver

        # Try to unwrap to find perceiver
        current_env = env
        while hasattr(current_env, 'env'):
            current_env = current_env.env
            if hasattr(current_env, 'perceiver'):
                return current_env.perceiver

        raise ValueError("Could not find perceiver in environment")

    def _get_base_env(self, env: gym.Env) -> gym.Env:
        """Unwrap environment to get base env."""
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

    def _train_sequential(self, policies_to_train: dict) -> None:
        """Train policies sequentially (one at a time)."""
        for shortcut_id, (policy, wrapped_env, single_shortcut_data, train_config) in policies_to_train.items():
            source_node, target_node = single_shortcut_data.shortcuts[0]
            print(f"\n--- Training policy for shortcut {shortcut_id} ---")
            print(f"  Source: node {source_node.id} ({len(source_node.states)} states)")
            print(f"  Target: node {target_node.id}")

            # Create callback
            callback = TrainingProgressCallback(
                check_freq=train_config.training_record_interval,
                policy_key=f"shortcut_{shortcut_id}",
            )

            # Train this policy
            print(f"  Training on {len(source_node.states)} examples...")
            policy.train(
                env=wrapped_env,
                train_config=train_config,
                train_data=single_shortcut_data,
                callback=callback,
            )

    def _train_parallel(self, policies_to_train: dict, save_dir: str | None = None) -> None:
        """Train policies in parallel across multiple GPUs."""
        # Use GPU parallel trainer
        trainer = GPUParallelTrainer(use_cuda=True)

        # Prepare kwargs for training function
        train_kwargs = {}
        if save_dir:
            train_kwargs["save_dir"] = save_dir

        # Train all policies in parallel
        results = trainer.train_policies(
            policies_to_train,
            train_single_policy_v2,
            **train_kwargs
        )

        # Handle results (e.g., saved model paths)
        saved_models = {}
        for shortcut_id, result in results.items():
            if isinstance(result, dict) and result.get("saved_path"):
                saved_models[shortcut_id] = result["saved_path"]

        trainer.close()

    def save(self, path: str) -> None:
        """Save all policies."""
        import os
        os.makedirs(path, exist_ok=True)

        for shortcut_id, policy in self.policies.items():
            policy_path = os.path.join(path, f"policy_{shortcut_id}.zip")
            policy.save(policy_path)

        # Save mapping
        import pickle
        mapping_path = os.path.join(path, "node_pair_mapping.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump(self._node_pair_to_shortcut, f)

    def load(self, path: str) -> None:
        """Load all policies."""
        import os
        import pickle

        # Load mapping
        mapping_path = os.path.join(path, "node_pair_mapping.pkl")
        with open(mapping_path, 'rb') as f:
            self._node_pair_to_shortcut = pickle.load(f)

        # Load policies
        for shortcut_id in set(self._node_pair_to_shortcut.values()):
            policy_path = os.path.join(path, f"policy_{shortcut_id}.zip")
            if os.path.exists(policy_path):
                self.policies[shortcut_id] = RLPolicy(self._seed + shortcut_id, self.config)
                self.policies[shortcut_id].load(policy_path)
