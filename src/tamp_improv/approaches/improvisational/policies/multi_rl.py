"""Multi-Policy RL implementation."""

import copy
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
from relational_structs import GroundAtom, Object, Predicate

from tamp_improv.approaches.improvisational.policies.base import (
    Policy,
    PolicyContext,
    TrainingData,
)
from tamp_improv.approaches.improvisational.policies.rl import (
    RLConfig,
    RLPolicy,
    TrainingProgressCallback,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MultiRLPolicy(Policy[ObsType, ActType]):
    """Policy that uses multiple specialized RL policies for different
    shortcuts."""

    def __init__(
        self,
        seed: int,
        config: RLConfig | None = None,
        enable_generalization: bool = False,
    ) -> None:
        """Initialize with a seed and optional config."""
        super().__init__(seed)
        self.env: gym.Env
        self.base_env: gym.Env
        self.config = config or RLConfig()
        self.enable_generalization = enable_generalization
        self.policies: dict[str, RLPolicy] = {}
        self._active_policy_key: str | None = None
        self._current_context: PolicyContext | None = None
        self._policy_patterns: dict[str, dict[str, set[GroundAtom]]] = {}
        self._current_substitution: dict[Object, Object] | None = None
        self._saved_models: dict[str, str] = {}

    def initialize(self, env: gym.Env) -> None:
        """Initialize the policy."""
        self.env = env
        self.base_env = self._get_base_env()

    def configure_context(self, context: PolicyContext) -> None:
        """Configure policy with context information."""
        self._current_context = context
        matching_policy = self._find_matching_policy(context)
        if matching_policy:
            self._active_policy_key = matching_policy
            self.policies[matching_policy].configure_context(context)
        else:
            self._active_policy_key = None

    def can_initiate(self) -> bool:
        """Check if we can handle the current context."""
        if not self._current_context:
            return False
        matching_policy = self._find_matching_policy(self._current_context)
        if matching_policy is None:
            # Debug: show what key we're looking for vs what's available
            key = self._get_policy_key(self._current_context)
            # print(f"    DEBUG can_initiate: Looking for key '{key}'")
            # print(f"    DEBUG can_initiate: Available keys ({len(self.policies)}): {list(self.policies.keys())}")
        return matching_policy is not None

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from the appropriate policy with selective feature
        extraction."""
        if not self._active_policy_key or self._active_policy_key not in self.policies:
            raise ValueError("No active policy for current context")

        # For graph observations, extract fixed vector of relevant object features
        if hasattr(obs, "nodes"):
            pattern = self._policy_patterns.get(self._active_policy_key, {})
            relevant_objects: set[str] = set()
            for atom in pattern.get("added_atoms", set()).union(
                pattern.get("deleted_atoms", set())
            ):
                for obj in atom.objects:
                    relevant_objects.add(obj.name)

            assert self._current_substitution is not None
            mapped_objects: set[str] = set()
            for obj_name in relevant_objects:
                for orig, subst in self._current_substitution.items():
                    if orig.name == obj_name:
                        mapped_objects.add(subst.name)
            assert mapped_objects is not None
            assert hasattr(self.base_env, "extract_relevant_object_features")
            feature_vector = self.base_env.extract_relevant_object_features(
                obs, mapped_objects
            )
            return self.policies[self._active_policy_key].get_action(feature_vector)

        # For non-graph observations, use the original observation
        return self.policies[self._active_policy_key].get_action(obs)

    def train(
        self, env: gym.Env, train_data: TrainingData | None, save_dir: str | None = None,
        system_cls: type | None = None, system_kwargs: dict | None = None,
    ) -> None:
        """Train multiple specialized policies."""
        assert train_data is not None
        print("\n=== Training Multi-Policy RL ===")
        print(f"Total training examples: {len(train_data.states)}")
        if len(train_data.states) == 0:
            print("No training data provided, skipping training.")
            return

        grouped_data = self._group_training_data(train_data)

        is_graph_based = hasattr(train_data.states[0], "nodes")

        # Determine parallel vs sequential
        import torch
        max_workers = self.config.num_parallel_workers
        use_parallel = max_workers > 0 and len(grouped_data) > 1

        policies_to_train = {}
        env_factories = {}  # Only used for parallel path
        for policy_key, group_data in grouped_data.items():
            print(f"Training examples: {len(group_data.states)}")

            if policy_key not in self.policies:
                self.policies[policy_key] = RLPolicy(self._seed, self.config)

            relevant_objects = None
            obs_space_shape = None
            if is_graph_based:
                pattern = self._policy_patterns[policy_key]
                relevant_objects = set()
                for atom in pattern.get("added_atoms", set()).union(
                    pattern.get("deleted_atoms", set())
                ):
                    for obj in atom.objects:
                        relevant_objects.add(obj.name)
                if isinstance(env, RecordVideo):
                    if hasattr(env.env, "set_relevant_objects"):
                        env.env.set_relevant_objects(relevant_objects)
                else:
                    if hasattr(env, "set_relevant_objects"):
                        env.set_relevant_objects(relevant_objects)
                self.base_env = self._get_base_env(env)

                assert hasattr(self.base_env, "extract_relevant_object_features")
                sample_state = group_data.states[0]
                sample_features = self.base_env.extract_relevant_object_features(
                    sample_state, relevant_objects
                )
                obs_space_shape = sample_features.shape

            if use_parallel and system_cls is not None:
                # Build a picklable factory for the worker to reconstruct the env
                env_factories[policy_key] = EnvFactory(
                    system_cls=system_cls,
                    system_kwargs=system_kwargs or {},
                    relevant_objects=relevant_objects,
                    obs_space_shape=obs_space_shape,
                    group_data=group_data,
                    max_episode_steps=self.config.max_episode_steps,
                )
            else:
                # Build env directly for sequential training
                if obs_space_shape is not None:
                    policy_env = copy.deepcopy(env)
                    policy_env.observation_space = Box(
                        low=-np.inf, high=np.inf, shape=obs_space_shape, dtype=np.float32,
                    )
                else:
                    policy_env = copy.deepcopy(env)
                self._configure_env_recursively(policy_env, group_data)

            policies_to_train[policy_key] = (
                self.policies[policy_key],
                group_data,
            )

        num_parallel_workers = min(len(policies_to_train), max_workers) if use_parallel else 1
        if num_parallel_workers > 1 and env_factories:
            import tempfile
            import torch.multiprocessing as mp
            try:
                mp.set_start_method("spawn")
            except RuntimeError:
                pass

            parallel_save_dir = save_dir or tempfile.mkdtemp(prefix="parallel_ppo_")
            print(f"\nUsing parallel training with {num_parallel_workers} workers")

            with mp.Pool(num_parallel_workers) as pool:
                async_results = []
                for policy_key, (policy, group_data) in policies_to_train.items():
                    factory = env_factories[policy_key]
                    ar = pool.apply_async(
                        _train_policy_in_process,
                        (policy, factory, group_data, policy_key, parallel_save_dir),
                    )
                    async_results.append((policy_key, ar))

                for policy_key, ar in async_results:
                    try:
                        result = ar.get()
                        if isinstance(result, dict) and result.get("saved_path"):
                            self.policies[policy_key].load(result["saved_path"])
                            print(f"  Loaded parallel-trained model for {policy_key}")
                        else:
                            print(f"  WARNING: No saved model for {policy_key}, result={result}")
                    except Exception as e:
                        print(f"  ERROR training {policy_key}: {e}")
        else:
            print("\nTraining policies sequentially")
            for policy_key, (
                policy,
                group_data,
            ) in policies_to_train.items():
                # Build env for this policy (sequential — can use deepcopy safely)
                obs_shape = None
                if is_graph_based:
                    pattern = self._policy_patterns[policy_key]
                    rel_objs = set()
                    for atom in pattern.get("added_atoms", set()).union(
                        pattern.get("deleted_atoms", set())
                    ):
                        for obj in atom.objects:
                            rel_objs.add(obj.name)
                    base = self._get_base_env(env)
                    sample_feats = base.extract_relevant_object_features(
                        group_data.states[0], rel_objs
                    )
                    obs_shape = sample_feats.shape

                policy_env = copy.deepcopy(env)
                if obs_shape is not None:
                    policy_env.observation_space = Box(
                        low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32,
                    )
                self._configure_env_recursively(policy_env, group_data)

                print(f"\nTraining policy for shortcut type: {policy_key}")
                result = train_single_policy(
                    policy, policy_env, group_data, policy_key=policy_key, save_dir=save_dir
                )
                if isinstance(result, dict) and result.get("best_checkpoint"):
                    best_checkpoint_path = result["best_checkpoint"]
                    if best_checkpoint_path:
                        policy.load(best_checkpoint_path)

        print(f"\nCompleted training {len(self.policies)} specialized policies")

    def _get_policy_key(self, context: PolicyContext) -> str:
        """Create a unique key for a policy based on the context."""
        source_atoms_str = sorted([str(atom) for atom in context.current_atoms])
        target_atoms_str = sorted([str(atom) for atom in context.goal_atoms])
        source_hash = hashlib.md5("|".join(source_atoms_str).encode()).hexdigest()[:8]
        target_hash = hashlib.md5("|".join(target_atoms_str).encode()).hexdigest()[:8]
        source_id = context.info.get("source_node_id", "")
        target_id = context.info.get("target_node_id", "")
        # if source_id != "" and target_id != "":
        #     return f"n{source_id}-to-n{target_id}_{source_hash}_{target_hash}"
        return f"{source_hash}_{target_hash}"

    def _find_matching_policy(self, context: PolicyContext) -> str | None:
        """Find a matching policy based on relevant atoms of the shortcut."""
        source_atoms = context.current_atoms
        target_atoms = context.goal_atoms
        added_atoms = target_atoms - source_atoms
        deleted_atoms = source_atoms - target_atoms

        transformed_test_atoms = set()
        for atom in added_atoms:
            transformed_test_atoms.add(self._transform_atom(atom, "ADD"))
        for atom in deleted_atoms:
            transformed_test_atoms.add(self._transform_atom(atom, "DEL"))

        # Try exact key match first
        key = self._get_policy_key(context)
        if key in self.policies:
            relevant_objects = set()
            for atom in (
                self._policy_patterns[key]
                .get("added_atoms", set())
                .union(self._policy_patterns[key].get("deleted_atoms", set()))
            ):
                relevant_objects.update(atom.objects)
            self._current_substitution = {obj: obj for obj in relevant_objects}
            return key

        # Try structural matching if enabled
        if self.enable_generalization:
            for policy_key, pattern_info in self._policy_patterns.items():
                transformed_train_atoms = set()
                if "transformed_atoms" in pattern_info:
                    transformed_train_atoms = pattern_info["transformed_atoms"]
                else:
                    for atom in pattern_info["added_atoms"]:
                        transformed_train_atoms.add(self._transform_atom(atom, "ADD"))
                    for atom in pattern_info["deleted_atoms"]:
                        transformed_train_atoms.add(self._transform_atom(atom, "DEL"))

                # Check predicate subsets with transformed predicates
                train_predicates = {
                    atom.predicate.name for atom in transformed_train_atoms
                }
                test_predicates = {
                    atom.predicate.name for atom in transformed_test_atoms
                }
                if not train_predicates.issubset(test_predicates):
                    continue

                # Find substitution
                match_found, substitution = find_atom_substitution(
                    transformed_train_atoms, transformed_test_atoms, self.base_env
                )
                if match_found:
                    self._current_substitution = substitution
                    return policy_key

        return None

    def _group_training_data(self, train_data: TrainingData) -> dict[str, TrainingData]:
        """Group training data by shortcut signature."""
        grouped: dict[str, dict] = {}
        shortcut_info = train_data.config.get("shortcut_info", [])

        for i in range(len(train_data)):
            current_atoms = train_data.current_atoms[i]
            goal_atoms = train_data.goal_atoms[i]
            added_atoms = goal_atoms - current_atoms
            deleted_atoms = current_atoms - goal_atoms
            transformed_atoms = set()
            for atom in added_atoms:
                transformed_atoms.add(self._transform_atom(atom, "ADD"))
            for atom in deleted_atoms:
                transformed_atoms.add(self._transform_atom(atom, "DEL"))

            info = {}
            if i < len(shortcut_info):
                info = shortcut_info[i]

            context: PolicyContext[ObsType, ActType] = PolicyContext(
                current_atoms=current_atoms, goal_atoms=goal_atoms, info=info
            )
            policy_key = self._get_policy_key(context)

            if policy_key not in grouped:
                grouped[policy_key] = {
                    "states": [],
                    "current_atoms": [],
                    "goal_atoms": [],
                    "pattern": {
                        "added_atoms": added_atoms,
                        "deleted_atoms": deleted_atoms,
                        "transformed_atoms": transformed_atoms,
                    },
                }

            grouped[policy_key]["states"].append(train_data.states[i])
            grouped[policy_key]["current_atoms"].append(current_atoms)
            grouped[policy_key]["goal_atoms"].append(goal_atoms)

        result = {}
        for key, group in grouped.items():
            result[key] = TrainingData(
                states=group["states"],
                current_atoms=group["current_atoms"],
                goal_atoms=group["goal_atoms"],
                config=train_data.config,
            )
            self._policy_patterns[key] = group["pattern"]
        return result

    def _transform_atom(self, atom: GroundAtom, prefix: str) -> GroundAtom:
        """Create a transformed atom with prefixed predicate name."""
        transformed_pred = Predicate(
            name=f"{prefix}-{atom.predicate.name}",
            types=atom.predicate.types,
        )
        return transformed_pred(atom.objects)

    def _configure_env_recursively(
        self, env: gym.Env, training_data: TrainingData
    ) -> None:
        """Recursively unwrap environment to configure the trainable
        wrapper."""
        if hasattr(env, "configure_training"):
            env.configure_training(training_data)
        if hasattr(env, "max_episode_steps"):
            env.max_episode_steps = self.config.max_episode_steps
        if hasattr(env, "env"):
            self._configure_env_recursively(env.env, training_data)

    def _get_base_env(self, env: gym.Env | None = None) -> gym.Env:
        """Get the base environment."""
        if env is not None:
            current_env = env
        else:
            current_env = self.env
        valid_base_env = False
        while hasattr(current_env, "env"):
            if hasattr(current_env, "reset_from_state"):
                valid_base_env = True
                break
            current_env = current_env.env
        if hasattr(current_env, "reset_from_state"):
            valid_base_env = True
        if not valid_base_env:
            raise AttributeError(
                "Could not find base environment with reset_from_state method"
            )
        return current_env

    def save(self, path: str) -> None:
        """Save all policies."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        already_saved = self._saved_models

        for key, policy in self.policies.items():
            if key in already_saved:
                continue
            safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
            policy_path = os.path.join(path, f"policy_{safe_key}")
            policy.save(policy_path)

        for policy_key, pattern in self._policy_patterns.items():
            pattern_file = os.path.join(path, f"pattern_{policy_key}.pkl")
            with open(pattern_file, "wb") as f:
                pickle.dump(pattern, f)

        manifest = {
            "policies": list(self.policies.keys()),
            "policy_count": len(self.policies),
        }
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

    def load(self, path: str) -> None:
        """Load all policies."""
        with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self._policy_patterns = {}
        for policy_key in manifest["policies"]:
            pattern_file = os.path.join(path, f"pattern_{policy_key}.pkl")
            if os.path.exists(pattern_file):
                with open(pattern_file, "rb") as f:
                    self._policy_patterns[policy_key] = pickle.load(f)

        self.policies = {}
        for key in manifest["policies"]:
            safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
            policy_path = os.path.join(path, f"policy_{safe_key}")
            policy: RLPolicy = RLPolicy(self._seed, self.config)
            policy.load(policy_path)
            self.policies[key] = policy

        print(f"Loaded {len(self.policies)} specialized policies")


class EnvFactory:
    """Picklable factory that reconstructs a training environment in a worker process."""

    def __init__(
        self,
        system_cls: type,
        system_kwargs: dict,
        relevant_objects: set[str] | None,
        obs_space_shape: tuple | None,
        group_data: TrainingData,
        max_episode_steps: int | None = None,
    ):
        self.system_cls = system_cls
        self.system_kwargs = system_kwargs
        self.relevant_objects = relevant_objects
        self.obs_space_shape = obs_space_shape
        self.group_data = group_data
        self.max_episode_steps = max_episode_steps

    def __call__(self) -> gym.Env:
        """Create a fresh environment in the current process."""
        import inspect
        create_fn = self.system_cls.create_default
        sig = inspect.signature(create_fn)
        valid_params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values()):
            filtered = self.system_kwargs
        else:
            filtered = {k: v for k, v in self.system_kwargs.items() if k in valid_params}
        system = create_fn(**filtered)
        env = system.wrapped_env

        if self.relevant_objects is not None and hasattr(env, "set_relevant_objects"):
            env.set_relevant_objects(self.relevant_objects)

        if self.obs_space_shape is not None:
            env.observation_space = Box(
                low=-np.inf, high=np.inf, shape=self.obs_space_shape, dtype=np.float32,
            )

        # Configure training data and max_episode_steps on the env
        current = env
        while current is not None:
            if hasattr(current, "configure_training"):
                current.configure_training(self.group_data)
            if self.max_episode_steps is not None and hasattr(current, "max_episode_steps"):
                current.max_episode_steps = self.max_episode_steps
            current = getattr(current, "env", None)

        return env


def _train_policy_in_process(
    policy: RLPolicy,
    env_factory: "EnvFactory",
    train_data: TrainingData,
    policy_key: str,
    save_dir: str | None = None,
):
    """Train a single policy in a worker process, reconstructing the env from factory."""
    import os, torch
    env = env_factory()
    callback = TrainingProgressCallback(
        check_freq=policy.config.training_record_interval,
        early_stopping=policy.config.early_stopping,
        early_stopping_patience=policy.config.early_stopping_patience,
        early_stopping_threshold=0.8,
        policy_key=policy_key,
    )
    policy.train(env, train_data, callback=callback)

    # Save model so main process can load it
    if save_dir and hasattr(policy, "model") and policy.model is not None:
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in policy_key)
        save_path = os.path.join(save_dir, f"policy_{safe_key}")
        os.makedirs(save_dir, exist_ok=True)
        policy.model.save(save_path)
        return {"success": True, "saved_path": save_path}
    return {"success": True}


def train_single_policy(
    policy: RLPolicy,
    env: gym.Env,
    train_data: TrainingData,
    policy_key: str | None = None,
    save_dir: str | None = None,
):
    """Train a single policy with a callback (sequential path)."""
    checkpoint_dir = None
    if save_dir:
        checkpoint_dir = str(Path(save_dir) / "checkpoints")
    callback = TrainingProgressCallback(
        check_freq=policy.config.training_record_interval,
        early_stopping=policy.config.early_stopping,
        early_stopping_patience=policy.config.early_stopping_patience,
        early_stopping_threshold=0.8,
        policy_key=policy_key,
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
    )
    policy.train(env, train_data, callback=callback)
    return {"success": True, "best_checkpoint": callback.best_checkpoint_path}


def find_atom_substitution(
    train_atoms: set[GroundAtom],
    test_atoms: set[GroundAtom],
    env: gym.Env | None = None,
) -> tuple[bool, dict[Object, Object]]:
    """Find if train_atoms can be mapped to a subset of test_atoms."""
    test_atoms_by_pred = defaultdict(list)
    for atom in test_atoms:
        test_atoms_by_pred[atom.predicate.name].append(atom)

    # Quick check: if there are enough atoms of each predicate type in test_atoms
    train_pred_counts = Counter(atom.predicate.name for atom in train_atoms)
    for pred_name, count in train_pred_counts.items():
        if len(test_atoms_by_pred[pred_name]) < count:
            return False, {}

    train_objs_by_type: dict[Any, list[Object]] = defaultdict(list)
    test_objs_by_type: dict[Any, list[Object]] = defaultdict(list)
    for atom in train_atoms:
        for obj in atom.objects:
            if obj not in train_objs_by_type[obj.type]:
                train_objs_by_type[obj.type].append(obj)
    for atom in test_atoms:
        for obj in atom.objects:
            if obj not in test_objs_by_type[obj.type]:
                test_objs_by_type[obj.type].append(obj)

    # Quick check: if there are enough test objects for each type
    for obj_type, objs in train_objs_by_type.items():
        if len(test_objs_by_type[obj_type]) < len(objs):
            return False, {}

    # Sort train objects to ensure deterministic behavior
    train_objects = []
    for obj_type in sorted(train_objs_by_type.keys(), key=lambda t: t.name):
        train_objects.extend(sorted(train_objs_by_type[obj_type], key=lambda o: o.name))

    return find_substitution_helper(
        train_atoms=train_atoms,
        test_atoms_by_pred=test_atoms_by_pred,
        remaining_train_objs=train_objects,
        test_objs_by_type=test_objs_by_type,
        partial_sub={},
        env=env,
    )


def find_substitution_helper(
    train_atoms: set[GroundAtom],
    test_atoms_by_pred: dict[str, list[GroundAtom]],
    remaining_train_objs: list[Object],
    test_objs_by_type: dict[Any, list[Object]],
    partial_sub: dict[Object, Object],
    env: gym.Env | None = None,
) -> tuple[bool, dict[Object, Object]]:
    """Helper to find_atom_substitution using backtracking search."""
    if not remaining_train_objs:
        return check_substitution_valid(
            train_atoms, test_atoms_by_pred, partial_sub, env
        )

    train_obj = remaining_train_objs[0]
    remaining = remaining_train_objs[1:]

    # Sort test objects to prioritize exact name matches
    candidates = list(test_objs_by_type[train_obj.type])
    candidates.sort(key=lambda obj: 0 if obj.name == train_obj.name else 1)

    for test_obj in test_objs_by_type[train_obj.type]:
        if test_obj in partial_sub.values():
            continue
        new_sub = partial_sub.copy()
        new_sub[train_obj] = test_obj
        success, final_sub = find_substitution_helper(
            train_atoms=train_atoms,
            test_atoms_by_pred=test_atoms_by_pred,
            remaining_train_objs=remaining,
            test_objs_by_type=test_objs_by_type,
            partial_sub=new_sub,
            env=env,
        )
        if success:
            return True, final_sub

    return False, {}


def check_substitution_valid(
    train_atoms: set[GroundAtom],
    test_atoms_by_pred: dict[str, list[GroundAtom]],
    substitution: dict[Object, Object],
    env: gym.Env | None = None,
) -> tuple[bool, dict[Object, Object]]:
    """Check if substitution maps all train_atoms to some subset of
    test_atoms."""
    for train_atom in train_atoms:
        pred_name = train_atom.predicate.name
        if pred_name not in test_atoms_by_pred:
            return False, {}
        subst_objs = tuple(substitution[obj] for obj in train_atom.objects)
        found_match = False
        for test_atom in test_atoms_by_pred[pred_name]:
            if tuple(test_atom.objects) == subst_objs:
                found_match = True
                break
        if not found_match:
            return False, {}
    if env is not None and hasattr(env, "get_object_category"):
        for orig, subst in substitution.items():
            orig_category = env.get_object_category(orig.name)
            subst_category = env.get_object_category(subst.name)
            if orig_category and subst_category and orig_category != subst_category:
                return False, {}

    return True, substitution
