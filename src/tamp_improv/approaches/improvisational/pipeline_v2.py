"""Simplified pipeline for SLAP training - V2 with unified heuristic interface.

This pipeline treats all heuristic methods uniformly through a common interface,
eliminating special cases and complexity. No caching - just clean execution.

Pipeline stages:
1. Collect data - collect_total_shortcuts → training_data + planning_graph + graph_distances
2. Train heuristic - heuristic.multi_train() → trained heuristic + training_history
   2.5. Test heuristic quality (if debug) → heuristic quality results
3. Prune with heuristic - heuristic.prune() → pruned_training_data
4. Create policy dictionary - creates D mapping (source, target) → policy wrapper
   - If use_multi_rl=True: trains MultiRL policies, then creates wrappers
   - If use_multi_rl=False: uses heuristic's actor directly (no training)
5. Add shortcuts to graph - converts D to LiftedOperators + Skills in system.components
   5.5. Test shortcut quality (if debug) → shortcut quality results
6. Evaluate - run_evaluation_episode() → evaluation results

The pipeline returns all results; the experiment handles saving.
"""

import copy
import itertools
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tamp_improv.approaches.improvisational.graph import PlanningGraphEdge
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    PDDLProblem,
    Variable
)
from task_then_motion_planning.structs import LiftedOperatorSkill

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic

from tamp_improv.approaches.improvisational.analyze import compute_true_node_distance, compute_all_edge_costs
from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.collection import collect_total_shortcuts
from tamp_improv.approaches.improvisational.graph_training import (
    compute_graph_distances,
    compute_first_edge_dict,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_cmd import (
    CMDHeuristic,
    CMDHeuristicConfig,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_crl import (
    CRLHeuristic,
    CRLHeuristicConfig,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_crl_v2 import (
    CRLv2Heuristic,
    CRLV2HeuristicConfig,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_cmd_v2 import (
    CMDv2Heuristic,
    CMDV2HeuristicConfig,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_dqn import (
    DQNHeuristic,
    DQNHeuristicConfig,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_sac_v2 import (
    SACv2Heuristic,
    SACV2HeuristicConfig,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_dsac_v2 import (
    DSACv2Heuristic,
    DSACv2HeuristicConfig,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_none import (
    NoneHeuristic,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_rollouts import (
    RolloutsHeuristic,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_smart_rollouts import (
    SmartRolloutsHeuristic,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
    Policy,
    PolicyContext,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.training import (
    Metrics,
    TrainingConfig,
    run_evaluation_episode,
    run_evaluation_episode_with_caching,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.utils.gpu_utils import set_torch_seed
import gymnasium as gym

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


# =============================================================================
# Policy Wrappers and Shortcut Skill
# =============================================================================


class MultiRLPolicyWrapper(Policy[ObsType, ActType]):
    """Wrapper that configures MultiRLPolicy to a specific (source, target) context."""

    def __init__(
        self,
        multi_rl_policy: MultiRLPolicy,
        context: PolicyContext,
    ):
        """Initialize wrapper.

        Args:
            multi_rl_policy: The trained MultiRLPolicy
            context: PolicyContext for this specific shortcut
        """
        self.policy = multi_rl_policy
        self.context = context

    def get_action(self, obs: ObsType) -> ActType:
        """Get action by configuring policy to this context."""
        self.policy.configure_context(self.context)
        return self.policy.get_action(obs)

    
    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""
        pass

    def can_initiate(self) -> bool:
        """Check whether the policy can be executed given the current
        context."""
        return True
    
    def save(self, path: str) -> None:
        """Save policy to disk."""
        pass

    def load(self, path: str) -> None:
        """Load policy from disk."""
        pass


class HeuristicPolicyWrapper(Policy[ObsType, ActType]):
    """Wrapper that uses heuristic's policy to navigate to a target node."""

    def __init__(
        self,
        heuristic: "BaseHeuristic",
        target_node_id: int,
    ):
        """Initialize wrapper.

        Args:
            heuristic: Trained heuristic with _select_action_with_actor method
            target_node_id: ID of the target node to navigate to
            env: Environment (needed for action selection)
        """
        self.heuristic = heuristic
        self.target_node_id = target_node_id

    def get_action(self, obs: ObsType) -> ActType:
        """Get action using heuristic's actor toward target node."""
        return self.heuristic.get_action(
            obs, self.target_node_id
        )

    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""
        pass

    def can_initiate(self) -> bool:
        """Check whether the policy can be executed given the current
        context."""
        return True
    
    def save(self, path: str) -> None:
        """Save policy to disk."""
        pass

    def load(self, path: str) -> None:
        """Load policy from disk."""
        pass


class ShortcutSkill(LiftedOperatorSkill[ObsType, ActType]):
    """Skill that wraps a policy wrapper to navigate between nodes."""

    def __init__(
        self,
        policy_wrapper: Policy[ObsType, ActType],
        operator: LiftedOperator,
        target_node_id: int,
        perceiver: Any,
        target_atoms: frozenset[GroundAtom],
    ):
        """Initialize shortcut skill.

        Args:
            policy_wrapper: Wrapper providing get_action(obs) interface
            operator: The shortcut operator this skill implements
            target_node_id: ID of the target node
            perceiver: Perceiver to check current atoms
            target_atoms: Atoms of the target node (for completion check)
        """
        self._policy_wrapper = policy_wrapper
        self._operator = operator
        self._target_node_id = target_node_id
        self._perceiver = perceiver
        self._target_atoms = target_atoms
        super().__init__()

    def _get_lifted_operator(self) -> LiftedOperator:
        """Return the operator this skill implements."""
        return self._operator

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: ObsType,
    ) -> ActType:
        """Use policy wrapper to select action toward target node."""
        return self._policy_wrapper.get_action(obs)


# =============================================================================
# Helper Functions: Create Heuristics and Policies
# =============================================================================

from dataclasses import fields


def dataclass_from_cfg(dataclass_type, cfg_section):
    field_names = {f.name for f in fields(dataclass_type)}

    kwargs = {k: v for k, v in cfg_section.items() if k in field_names}

    return dataclass_type(**kwargs)


def create_heuristic(
    training_data: GoalConditionedTrainingData,
    graph_distances: dict[tuple[int, int], float],
    system: ImprovisationalTAMPSystem,
    cfg: DictConfig,
    rng: np.random.Generator,
    first_edge_dict: dict[tuple[frozenset[GroundAtom], frozenset[GroundAtom]], PlanningGraphEdge],
) -> "BaseHeuristic":
    """Create a heuristic instance based on type.

    Args:
        heuristic_type: Type of heuristic ("rollouts", "v4", etc.)
        training_data: Training data from collection
        graph_distances: Graph distances
        system: TAMP system
        config: Configuration dictionary

    Returns:
        Heuristic instance
    """
    if cfg.heuristic.type == "none":
        return NoneHeuristic(
            training_data=training_data, graph_distances=graph_distances, system=system, rng=rng
        )
    elif cfg.heuristic.type == "rollouts":
        return RolloutsHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            num_rollouts=cfg.heuristic.rollouts.num_rollouts_per_node,
            max_steps_per_rollout=cfg.heuristic.rollouts.max_steps_per_rollout,
            threshold=cfg.heuristic.rollouts.success_threshold,
            action_scale=cfg.heuristic.rollouts.action_scale,
            seed=cfg.seed,
            rng=rng,
        )
    elif cfg.heuristic.type == "smart_rollouts":
        from tamp_improv.approaches.improvisational.heuristics.heuristic_smart_rollouts import SmartRolloutsConfig
        sr_config = dataclass_from_cfg(SmartRolloutsConfig, cfg.heuristic.rollouts)
        return SmartRolloutsHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            rng=rng,
            seed=cfg.seed,
            config=sr_config,
        )
    elif cfg.heuristic.type == "crl":
        # Extract config from heuristic.crl subsection
        crl_config = dataclass_from_cfg(CRLHeuristicConfig, cfg.heuristic.crl)
        crl_config.wandb_enabled = cfg.wandb_enabled
        crl_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("CRL Config:", crl_config)

        return CRLHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=crl_config,
            seed=cfg.seed,
            rng=rng,
        )
    elif cfg.heuristic.type == "crl_v2":
        # Extract config from heuristic.crl subsection (shared with crl)
        crl_v2_config = dataclass_from_cfg(CRLV2HeuristicConfig, cfg.heuristic.crl)
        crl_v2_config.wandb_enabled = cfg.wandb_enabled
        crl_v2_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("CRL V2 Config:", crl_v2_config)

        return CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=crl_v2_config,
            seed=cfg.seed,
            rng=rng,
            first_edge_dict=first_edge_dict,
        )
    elif cfg.heuristic.type == "dqn":
        # TODO: Implement V4 heuristic
        dqn_config = dataclass_from_cfg(DQNHeuristicConfig, cfg.heuristic.dqn)
        print("DQN Config:", dqn_config)

        return DQNHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=dqn_config,
            seed=cfg.seed,
        )
    elif cfg.heuristic.type == "sac_v2":
        sac_v2_config = dataclass_from_cfg(SACV2HeuristicConfig, cfg.heuristic.sac)
        sac_v2_config.wandb_enabled = cfg.wandb_enabled
        sac_v2_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("SAC V2 Config:", sac_v2_config)

        return SACv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=sac_v2_config,
            rng=rng,
            first_edge_dict=first_edge_dict,
        )
    elif cfg.heuristic.type == "dsac_v2":
        dsac_v2_config = dataclass_from_cfg(DSACv2HeuristicConfig, cfg.heuristic.sac)
        dsac_v2_config.wandb_enabled = cfg.wandb_enabled
        dsac_v2_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("DSAC V2 Config:", dsac_v2_config)

        return DSACv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=dsac_v2_config,
            rng=rng,
            first_edge_dict=first_edge_dict,
        )
    elif cfg.heuristic.type == "cmd":
        # TODO: Implement V4 heuristic
        cmd_config = dataclass_from_cfg(CMDHeuristicConfig, cfg.heuristic)
        cmd_config.wandb_enabled = cfg.wandb_enabled
        cmd_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("CMD Config:", cmd_config)

        return CMDHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=cmd_config,
            seed=cfg.seed,
        )
    elif cfg.heuristic.type == "cmd_v2":
        # TODO: Implement V4 heuristic
        cmd_config = dataclass_from_cfg(CMDV2HeuristicConfig, cfg.heuristic.crl)
        cmd_config.wandb_enabled = cfg.wandb_enabled
        cmd_config.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("CMD V2 Config:", cmd_config)
        return CMDv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=cmd_config,
            seed=cfg.seed,
            rng=rng,
            first_edge_dict=first_edge_dict,
        )
    else:
        raise ValueError(f"Unknown heuristic type: {cfg.heuristic.type}")


def create_policy(
    cfg: DictConfig,
) -> Policy:
    """Create a policy instance based on type.

    Args:
        policy_type: Type of policy ("multiRL")
        system: TAMP system
        config: Configuration dictionary with rl_* parameters

    Returns:
        Policy instance
    """
    if cfg.policy.type == "multiRL":
        # Extract RL config from parameters prefixed with rl_
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rl_config = RLConfig(
            learning_rate=cfg.policy.learning_rate,
            batch_size=cfg.policy.batch_size,
            n_epochs=cfg.policy.n_epochs,
            gamma=cfg.policy.gamma,
            ent_coef=cfg.policy.ent_coef,
            deterministic=cfg.policy.deterministic,
            device=device,
            episodes_per_scenario=cfg.policy.episodes_per_scenario,
            max_episode_steps=cfg.policy.max_episode_steps,
            training_record_interval=cfg.policy.training_record_interval,
            early_stopping=cfg.policy.early_stopping,
            early_stopping_patience=cfg.policy.early_stopping_patience,
            n_envs=cfg.policy.get("n_envs", 1),
            num_parallel_workers=cfg.policy.get("num_parallel_workers", 0),
        )
        return MultiRLPolicy(seed=cfg.seed, config=rl_config)
    else:
        raise ValueError(f"Unknown policy type: {cfg.policy.type}")


# =============================================================================
# Pipeline Results
# =============================================================================


@dataclass
class PipelineResults:
    """Serializable container for all pipeline outputs.

    Contains only primitives (ints, floats, strs, lists, dicts, numpy arrays)
    so it can be directly pickled without issues.
    """

    # === Grid / Environment Config ===
    grid_config: dict[str, Any] | None = None

    # === Pre-Training Data ===
    node_atoms: dict[int, list[str]] | None = None
    node_states: dict[int, list[list[float]]] | None = None
    unique_shortcuts: list[tuple[int, int]] | None = None
    graph_distances: dict[tuple[int, int], float] | None = None
    true_distances: dict[tuple[int, int], float] | None = None

    # === Per-Round Training Data ===
    # Each round dict has keys: critic_losses, actor_losses, buffer_size,
    # shortcut_rollouts, graph_distances, ucb_sample_counts,
    # estimated_distances, gains
    training_rounds: list[dict[str, Any]] = field(default_factory=list)

    # === After Training (Pruning) ===
    pruned_shortcuts: list[tuple[int, int]] | None = None
    all_shortcuts: list[tuple[int, int]] | None = None
    shortcut_quality_results: list[dict[str, Any]] | None = None

    # === Evaluation ===
    # Each episode dict has keys: success, num_steps, reward,
    # initial_node, goal_nodes, optimal_path_nodes
    eval_episodes: list[dict[str, Any]] = field(default_factory=list)
    avg_success_rate: float | None = None
    avg_steps: float | None = None
    avg_reward: float | None = None

    # === Latent Embeddings (optional, for visualization) ===
    # node_id -> (node_embedding, [(state_flat, state_embedding), ...])
    # All values are plain Python lists (serializable).
    latent_embeddings: dict[int, tuple[list[float], list[tuple[list[float], list[float]]]]] | None = None

    # === Timing ===
    times: dict[str, float] = field(default_factory=dict)

    # === Config ===
    config: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Helper functions for extracting serializable data
# -----------------------------------------------------------------------------


def _flatten_state(state: Any) -> list[float]:
    """Flatten a state observation to a list of floats.

    Handles graph observations (with .nodes attribute) and plain arrays.
    """
    if hasattr(state, "nodes"):
        return state.nodes.flatten().astype(np.float32).tolist()
    return np.array(state).flatten().astype(np.float32).tolist()


def _extract_grid_config(system: ImprovisationalTAMPSystem) -> dict[str, Any]:
    """Extract grid/environment layout config as serializable dict."""
    env = system.env
    if hasattr(env, "unwrapped"):
        env = env.unwrapped

    config: dict[str, Any] = {}
    for attr in [
        "num_cells", "cell_size", "grid_size", "max_velocity",
        "portal_radius", "num_states_per_cell", "num_teleporters",
    ]:
        if hasattr(env, attr):
            config[attr] = getattr(env, attr)

    # Portal positions — convert numpy arrays to plain lists
    if hasattr(env, "portal_positions"):
        config["portal_positions"] = [
            (p1.tolist() if hasattr(p1, "tolist") else list(p1),
             p2.tolist() if hasattr(p2, "tolist") else list(p2))
            for p1, p2 in env.portal_positions
        ]
    if hasattr(env, "portal_cell_pairs"):
        config["portal_cell_pairs"] = [
            (tuple(p1), tuple(p2)) for p1, p2 in env.portal_cell_pairs
        ]

    return config


def _extract_node_data(
    training_data: "GoalConditionedTrainingData",
) -> tuple[dict[int, list[str]], dict[int, list[list[float]]]]:
    """Extract node atoms and states as serializable dicts.

    Returns:
        (node_atoms_ser, node_states_ser) where atoms are str representations
        and states are lists of flattened coordinate lists.
    """
    node_atoms_ser: dict[int, list[str]] = {}
    for node_id, atoms in training_data.node_atoms.items():
        node_atoms_ser[node_id] = sorted(str(a) for a in atoms)

    node_states_ser: dict[int, list[list[float]]] = {}
    for node_id, states in training_data.node_states.items():
        node_states_ser[node_id] = [_flatten_state(s) for s in states]

    return node_atoms_ser, node_states_ser


def _extract_true_distances(
    system: ImprovisationalTAMPSystem,
    training_data: "GoalConditionedTrainingData",
) -> dict[tuple[int, int], float]:
    """Compute true distances for all node pairs with available states."""
    true_dists: dict[tuple[int, int], float] = {}
    node_ids = list(training_data.node_states.keys())
    for source_id in node_ids:
        for target_id in node_ids:
            print("Computing true distance from node", source_id, "to node", target_id)
            if source_id == target_id:
                true_dists[(source_id, target_id)] = 0.0
                continue
            source_states = training_data.node_states[source_id]
            target_atoms = training_data.node_atoms[target_id]
            if source_states and target_atoms:
                true_dists[(source_id, target_id)] = compute_true_node_distance(
                    system, source_states, target_atoms
                )
    return true_dists


def _extract_heuristic_round_data(
    heuristic: "BaseHeuristic",
    training_data: "GoalConditionedTrainingData",
) -> dict[str, Any]:
    """Extract UCB counts, gains, estimated distances from a heuristic.

    Returns dict with ucb_sample_counts, estimated_distances, gains.
    Values are None if the heuristic doesn't support them.
    """
    data: dict[str, Any] = {
        "ucb_sample_counts": None,
        "estimated_distances": None,
        "gains": None,
    }

    # UCB sample counts and gains (CMD v2 / CRL v2)
    if hasattr(heuristic, "node_pair_samples"):
        counts: dict[tuple[int, int], int] = {}
        for src, tgt in training_data.unique_shortcuts:
            counts[(src, tgt)] = int(heuristic.node_pair_samples[src, tgt])
        data["ucb_sample_counts"] = counts

    if hasattr(heuristic, "node_pair_gains"):
        gains: dict[tuple[int, int], float] = {}
        for src, tgt in training_data.unique_shortcuts:
            gains[(src, tgt)] = float(heuristic.node_pair_gains[src, tgt])
        data["gains"] = gains

    # Estimated distances
    if hasattr(heuristic, "estimate_node_distance"):
        est_dists: dict[tuple[int, int], float] = {}
        for src, tgt in training_data.unique_shortcuts:
            est_dists[(src, tgt)] = float(
                heuristic.estimate_node_distance(src, tgt)
            )
        data["estimated_distances"] = est_dists

    return data


# =============================================================================
# Stage 1: Collection
# =============================================================================


def collect_training_data(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> tuple[GoalConditionedTrainingData, dict[tuple[int, int], float],
           dict[tuple[frozenset[GroundAtom], frozenset[GroundAtom]], "PlanningGraphEdge"]]:
    """Stage 1: Collect all shortcuts and compute graph distances.

    Args:
        system: TAMP system
        approach: Improvisational TAMP approach
        config: Configuration dictionary
        rng: Random number generator

    Returns:
        Tuple of (training_data, graph_distances)
    """
    # print("\n" + "=" * 80)
    # print("STAGE 1: COLLECT TRAINING DATA")
    # print("=" * 80)

    # Collect shortcuts
    training_data = collect_total_shortcuts(
        system=system,
        approach=approach,
        cfg=cfg,
        rng=rng,
    )

    print(f"\nCollected {len(training_data.unique_shortcuts)} unique shortcuts")
    print(f"  ({len(training_data.valid_shortcuts)} state-node pairs total)")
    print(
        f"Planning graph has {len(training_data.graph.nodes) if training_data.graph else 0} nodes"
    )

    # Compute graph distances
    print("\nComputing graph distances...")
    graph_distances = compute_graph_distances(
        training_data.graph, exclude_shortcuts=True
    )
    print(f"Computed {len(graph_distances)} pairwise distances")

    print("\nGraph distance computation complete")

    print("\nComputing first edges for all node pairs...")
    first_edge_dict = compute_first_edge_dict(training_data.graph)

    return training_data, graph_distances, first_edge_dict


# =============================================================================
# Stage 2: Train Heuristic
# =============================================================================

def find_applicable_operators(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    current_atoms: set[GroundAtom],
    objects: set[Object]
) -> list[GroundOperator]:
    """Find all ground operators that are applicable in the current
    state."""
    applicable_ops = []
    domain_operators = system.get_domain().operators

    for lifted_op in domain_operators:
        valid_groundings = find_valid_groundings(lifted_op, objects)

        for grounding in valid_groundings:
            ground_op = lifted_op.ground(grounding)

            if ground_op.preconditions.issubset(current_atoms):
                applicable_ops.append(ground_op)

    return applicable_ops

def find_valid_groundings(
    lifted_op: LiftedOperator, objects: set[Object]
) -> list[tuple[Object, ...]]:
    """Find all valid groundings for a lifted operator."""
    # Shortcut operators use variables named "?<objname>" tied to specific objects.
    # Match by name instead of type to avoid O(|objects|^N) combinatorial explosion.
    if lifted_op.name.startswith("Shortcut_"):
        objects_by_name = {obj.name: obj for obj in objects}
        grounding = []
        for param in lifted_op.parameters:
            obj_name = param.name.lstrip("?")
            if obj_name not in objects_by_name:
                return []
            grounding.append(objects_by_name[obj_name])
        return [tuple(grounding)]

    objects_by_type: dict[Any, list[Object]] = {}
    for obj in objects:
        if obj.type not in objects_by_type:
            objects_by_type[obj.type] = []
        objects_by_type[obj.type].append(obj)

    param_objects = []
    for param in lifted_op.parameters:
        if param.type in objects_by_type:
            param_objects.append(objects_by_type[param.type])
        else:
            return []

    groundings = list(itertools.product(*param_objects))

    return groundings


def train_heuristic(
    heuristic: "BaseHeuristic",
    cfg: DictConfig,
    checkpoint_callback: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Stage 2: Train the heuristic.

    Args:
        heuristic: Initialized heuristic instance
        config: Configuration dictionary
        checkpoint_callback: If provided, called periodically during training to save heuristic state.

    Returns:
        List of per-round data dicts, each containing loss curves,
        shortcut rollouts, UCB data, graph distances, etc.
    """
    # print("\n" + "=" * 80)
    # print("STAGE 2: TRAIN HEURISTIC")
    # print("=" * 80)

    training_data = heuristic.training_data
    system = heuristic.system
    round_results: list[dict[str, Any]] = []

    if getattr(cfg.heuristic, "continuous_graduation", False):
        def _graduation_callback(h: "BaseHeuristic", source_id: int, target_id: int) -> None:
            """Add operator+skill to virtual_system and edge to internal_graph for one graduated pair."""
            wrapper = HeuristicPolicyWrapper(heuristic=h, target_node_id=target_id)
            add_shortcuts_to_graph(h.virtual_system, {(source_id, target_id): wrapper}, training_data)

            cost = h.estimate_node_distance(source_id, target_id) if hasattr(h, "estimate_node_distance") else 1.0
            source_atoms = training_data.node_atoms[source_id]
            target_atoms = training_data.node_atoms[target_id]
            source_node = h.internal_graph.node_map[frozenset(source_atoms)]
            target_node = h.internal_graph.node_map[frozenset(target_atoms)]

            obs, info = h.virtual_system.reset()
            objects, _, _ = h.virtual_system.perceiver.reset(obs, info)
            applicable_ops = find_applicable_operators(h.virtual_system, set(source_atoms), objects)
            for op in applicable_ops:
                next_atoms = set(source_atoms)
                next_atoms.difference_update(op.delete_effects)
                next_atoms.update(op.add_effects)
                if frozenset(next_atoms) == frozenset(target_atoms):
                    h.internal_graph.add_edge(source_node, target_node, op, is_shortcut=True, cost=cost, max_cost=cost)

        print("\n=== Heuristic Training (continuous graduation) ===")
        training_history = heuristic.train_continuous(
            graduate_fn=_graduation_callback,
            success_threshold=cfg.heuristic.success_threshold,
            checkpoint_callback=checkpoint_callback,
        )
        round_data: dict[str, Any] = {
            "critic_losses": training_history.get("critic_losses", []),
            "actor_losses": training_history.get("actor_losses", []),
            "buffer_size": training_history.get("buffer_size", 0),
            "shortcut_rollouts": None,
            "graph_distances": None,
        }
        round_data.update(_extract_heuristic_round_data(heuristic, training_data))
        round_results.append(round_data)
        print("\nHeuristic training complete")
        return round_results

    for i in range(cfg.heuristic.num_rounds):
        print(f"\n=== Heuristic Training Round {i+1}/{cfg.heuristic.num_rounds} ===")
        training_history = heuristic.train_one_round(checkpoint_callback=checkpoint_callback)

        # Build per-round data dict
        round_data: dict[str, Any] = {
            "critic_losses": training_history.get("critic_losses", []),
            "actor_losses": training_history.get("actor_losses", []),
            "buffer_size": training_history.get("buffer_size", 0),
            "shortcut_rollouts": None,
            "graph_distances": None,
        }

        # Extract UCB / gains / estimated distances from heuristic
        round_data.update(_extract_heuristic_round_data(heuristic, training_data))

        if cfg.heuristic.num_rounds > 1 and cfg.debug and cfg.heuristic.type in ["sac_v2", "crl_v2", "cmd_v2"]:
            all_data = heuristic.prune(max_shortcuts=None)
            policy = create_policy(cfg=cfg)

            # Compute stats for all shortcuts (includes rollout positions)
            
            policy_dict = create_policy_dictionary(
                system=system,
                heuristic=copy.deepcopy(heuristic),
                policy=policy,
                training_data=all_data,
                cfg=cfg,
                use_multi_rl=False,
            )

            shortcut_quality_results = test_shortcut_quality(
                system=system,
                policy_dict=policy_dict,
                training_data=all_data,
                cfg=cfg
            )

            round_data["shortcut_rollouts"] = shortcut_quality_results["pairs"]


        if cfg.heuristic.num_rounds > 1:
            # Begin by obtaining pruned training data (no pruning)
            pruned_data = heuristic.prune_by_success(success_threshold=cfg.heuristic.success_threshold, 
                                                     max_steps=cfg.collection.max_steps_per_edge)
            # Collect all policies for all shortcuts in pruned_data
            policy = create_policy(cfg=cfg)
            successful_policy_dict = create_policy_dictionary(
                system=system,
                heuristic=copy.deepcopy(heuristic),
                policy=policy,
                training_data=pruned_data,
                cfg=cfg,
                use_multi_rl=False,
            )

            
            # Make copy of system and add all new operators/skills for successful shortcuts to the copy
            virtual_system = copy.deepcopy(heuristic.system)
            virtual_graph = copy.deepcopy(heuristic.internal_graph)
            add_shortcuts_to_graph(virtual_system, successful_policy_dict, training_data)

            # Add edges one by one to training_data.graph
            for shortcut in successful_policy_dict.keys():
                source_id = shortcut[0]
                target_id = shortcut[1]
                cost = heuristic.estimate_node_distance(source_id, target_id) if hasattr(heuristic, "estimate_node_distance") else 1.0
                source_atoms = training_data.node_atoms[source_id]
                target_atoms = training_data.node_atoms[target_id]

                source_node = virtual_graph.node_map[frozenset(source_atoms)]
                target_node = virtual_graph.node_map[frozenset(target_atoms)]

                obs, info = virtual_system.reset()
                objects, _, _ = virtual_system.perceiver.reset(obs, info)

                applicable_ops = find_applicable_operators(
                    virtual_system, set(source_atoms), objects
                )

                for op in applicable_ops:
                    next_atoms = set(source_atoms)
                    next_atoms.difference_update(op.delete_effects)
                    next_atoms.update(op.add_effects)

                    next_atoms_frozen = frozenset(next_atoms)
                    if next_atoms_frozen == frozenset(target_atoms):
                        virtual_graph.add_edge(source_node, target_node, op, is_shortcut=True, cost=cost, max_cost=cost)

            # Calculate edge costs, graph distances, first edge dict for virtual system
            # Turn training_data.node_states into a dict of node_id -> list of states (instead of list of lists)
            print(f"Graph has {len(heuristic.internal_graph.edges)} edges before adding shortcuts")
            print(f"Graph has {len(virtual_graph.edges)} edges after adding shortcuts")
            print("Edges in virtual graph not in original graph:")
            for edge in virtual_graph.edges:
                if not any(e == edge for e in heuristic.internal_graph.edges):
                    print(f"  {edge.source.id}->{edge.target.id} (is shortcut? {edge.is_shortcut})")

            virtual_graph_distances = compute_graph_distances(virtual_graph, exclude_shortcuts=False)
            virtual_first_edge_dict = compute_first_edge_dict(virtual_graph)

            # Store graph distances for this round
            round_data["graph_distances"] = dict(virtual_graph_distances)

            # Record cumulative set of shortcuts in virtual_graph after this round
            round_data["virtual_shortcuts"] = [
                (edge.source.id, edge.target.id)
                for edge in virtual_graph.edges
                if edge.is_shortcut
            ]

            if i < cfg.heuristic.num_rounds - 1:
                heuristic.update_system(
                    new_system=virtual_system,
                    new_graph=virtual_graph,
                    new_graph_distances=virtual_graph_distances,
                    new_first_edge_dict=virtual_first_edge_dict,
                    reset_actor=False,
                )

        round_results.append(round_data)

    print("\nHeuristic training complete")

    return round_results


# =============================================================================
# Stage 2.5: Test Heuristic Quality (Optional)
# =============================================================================


def test_heuristic_quality(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    heuristic: "BaseHeuristic",
    training_data: GoalConditionedTrainingData,
    graph_distances: dict[tuple[int, int], float],
    cfg: DictConfig,
    times: dict[str, float] | None,
) -> dict[str, Any]:
    """Stage 2.5: Test heuristic quality on sample node pairs.

    Args:
        heuristic: Trained heuristic
        training_data: Full training data
        graph_distances: Graph distances
        config: Configuration dictionary

    Returns:
        Dictionary with heuristic quality results
    """
    # print("\n" + "=" * 80)
    # print("STAGE 2.5: TEST HEURISTIC QUALITY")
    # print("=" * 80)

    # If there are no nodes in the graph, skip plotting
    if not training_data.graph or not training_data.graph.nodes:
        print(
            "[WARN] No nodes in the planning graph. Skipping heuristic quality plotting."
        )
        return {
            "samples": [],
            "avg_estimated_distance": 0.0,
            "avg_graph_distance": 0.0,
            "avg_absolute_error": 0.0,
        }

    # Sample some node pairs to evaluate (use unique_shortcuts for node-node pairs)
    shortcuts = training_data.unique_shortcuts
    sample_indices = [i for i in range(len(shortcuts))]

    results = {
        "samples": [],
        "avg_estimated_distance": 0.0,
        "avg_graph_distance": 0.0,
        "avg_absolute_error": 0.0,
    }

    total_est = 0.0
    total_graph = 0.0
    total_abs_error = 0.0
    all_estimated_distances = []
    all_graph_distances = []
    all_true_distances = []

    print(f"\nEvaluating heuristic on {len(sample_indices)} sample shortcuts:")
    print(f"{'Source':>8} {'Target':>8} {'Estimated':>12} {'Graph':>12} {'True':>12}{'Error':>12}")
    print("-" * 60)

    for idx in sample_indices:
        source_id, target_id = shortcuts[idx]
        estimated_dist = heuristic.estimate_node_distance(source_id, target_id)
        graph_dist = graph_distances.get((source_id, target_id), float("inf"))
        true_dist = compute_true_node_distance(
            system,
            training_data.node_states[source_id],
            training_data.node_atoms[target_id],
        )
        abs_error = (
            abs(estimated_dist - graph_dist)
            if graph_dist != float("inf")
            else float("inf")
        )

        results["samples"].append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "estimated_distance": estimated_dist,
                "graph_distance": graph_dist,
                "true_distance": true_dist,
                "absolute_error": abs_error,
            }
        )

        print(
            f"{source_id:8d} {target_id:8d} {estimated_dist:12.2f} {graph_dist:12.2f} {true_dist:12.2f}  {abs_error:12.2f}"
        )
        # print(f"TRUE DISTANCE: {true_dist}")

        if graph_dist != float("inf"):
            total_est += estimated_dist
            total_graph += graph_dist
            total_abs_error += abs_error
            all_estimated_distances.append(estimated_dist)
            all_graph_distances.append(graph_dist)
            all_true_distances.append(true_dist)

    # Compute averages
    num_finite = sum(
        1 for s in results["samples"] if s["graph_distance"] != float("inf")
    )
    if num_finite > 0:
        results["avg_estimated_distance"] = total_est / num_finite
        results["avg_graph_distance"] = total_graph / num_finite
        results["avg_absolute_error"] = total_abs_error / num_finite

        # Compute min/max/correlation
        min_estimated_distance = np.min(all_estimated_distances)
        max_estimated_distance = np.max(all_estimated_distances)
        min_graph_distance = np.min(all_graph_distances)
        max_graph_distance = np.max(all_graph_distances)
        correlation_distance = np.corrcoef(
            all_estimated_distances, all_graph_distances
        )[0, 1]

        print("-" * 60)
        print(
            f"{'Average':>8} {'':<8} {results['avg_estimated_distance']:12.2f} {results['avg_graph_distance']:12.2f} {results['avg_absolute_error']:12.2f}"
        )
        print(f"{'Min Estimated':>28} {min_estimated_distance:12.2f}")
        print(f"{'Max Estimated':>28} {max_estimated_distance:12.2f}")
        print(f"{'Min Graph':>28} {min_graph_distance:12.2f}")
        print(f"{'Max Graph':>28} {max_graph_distance:12.2f}")
        print(f"{'Correlation':>28} {correlation_distance:12.2f}")

        # Get all unique node IDs from the graph
        all_graph_node_ids = sorted([node.id for node in training_data.graph.nodes])
        num_nodes = len(all_graph_node_ids)

        # Initialize dense matrices with NaN
        true_distances_matrix = np.full((num_nodes, num_nodes), np.nan)
        graph_distances_matrix = np.full((num_nodes, num_nodes), np.nan)
        estimated_distances_matrix = np.full((num_nodes, num_nodes), np.nan)

        node_id_to_idx = {node_id: i for i, node_id in enumerate(all_graph_node_ids)}

        # Populate matrices
        for sample in results["samples"]:
            src_idx = node_id_to_idx[sample["source_id"]]
            tgt_idx = node_id_to_idx[sample["target_id"]]
            true_distances_matrix[src_idx, tgt_idx] = sample["true_distance"]
            graph_distances_matrix[src_idx, tgt_idx] = sample["graph_distance"]
            estimated_distances_matrix[src_idx, tgt_idx] = sample["estimated_distance"]

        if cfg.wandb_enabled:
            _log_distance_plots_to_wandb(
                true_distances=true_distances_matrix,
                graph_distances=graph_distances_matrix,
                estimated_distances=estimated_distances_matrix,
                node_ids=all_graph_node_ids,
            )

        if cfg.wandb_enabled:
            wandb.log(
                {
                    "test/avg_estimated_distance": results["avg_estimated_distance"],
                    "test/avg_graph_distance": results["avg_graph_distance"],
                    "test/avg_absolute_error": results["avg_absolute_error"],
                    "test/min_estimated_distance": min_estimated_distance,
                    "test/max_estimated_distance": max_estimated_distance,
                    "test/min_graph_distance": min_graph_distance,
                    "test/max_graph_distance": max_graph_distance,
                    "test/correlation_distance": correlation_distance,
                    "times/heuristic_training_time": times["heuristic_training_time"] if times else 0.0,
                }
            )

    return results


def _log_distance_plots_to_wandb(
    true_distances: np.ndarray,
    graph_distances: np.ndarray,
    estimated_distances: np.ndarray,
    node_ids: list[int],
) -> None:
    """Helper to plot distance matrices and scatterplots and log to WandB."""
    # Create figure for heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Common settings for heatmaps
    vmin = 0
    vmax = max(
        np.nanmax(true_distances[np.isfinite(true_distances)]),
        np.nanmax(estimated_distances),
    )

    # Determine whether to show annotations (hide if grid is larger than 5x5)
    num_nodes = len(node_ids)
    show_annot = num_nodes <= 25  # 5x5 or smaller

    # Heatmap 1: True Distances
    sns.heatmap(
        true_distances,
        annot=show_annot,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=node_ids,
        yticklabels=node_ids,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Distance"},
        ax=axes[0],
    )
    axes[0].set_title("True Distances (Graph Search)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Target Node", fontsize=12)
    axes[0].set_ylabel("Source Node", fontsize=12)

    # Heatmap 2: Graph Distances
    sns.heatmap(
        graph_distances,
        annot=show_annot,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=node_ids,
        yticklabels=node_ids,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Distance"},
        ax=axes[1],
    )
    axes[1].set_title(
        "Graph Distances (Planning Graph)", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Target Node", fontsize=12)
    axes[1].set_ylabel("Source Node", fontsize=12)

    # Heatmap 3: Estimated Distances
    sns.heatmap(
        estimated_distances,
        annot=show_annot,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=node_ids,
        yticklabels=node_ids,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Distance"},
        ax=axes[2],
    )
    axes[2].set_title(
        "Learned Distances (V4 Heuristic)", fontsize=14, fontweight="bold"
    )
    axes[2].set_xlabel("Target Node", fontsize=12)
    axes[2].set_ylabel("Source Node", fontsize=12)

    plt.tight_layout()
    wandb.log({"test/distance_heatmaps": wandb.Image(fig)})
    plt.close(fig)  # Close the figure to free up memory

    # Create scatterplots comparing distances
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter out invalid distances
    valid_mask = np.isfinite(true_distances) & (true_distances > 0)
    valid_indices = np.where(valid_mask)

    true_flat = true_distances[valid_indices]
    graph_flat = graph_distances[valid_indices]
    estimated_flat = estimated_distances[valid_indices]

    # Scatterplot 1: Learned vs True
    axes[0].scatter(
        true_flat, estimated_flat, alpha=0.6, s=80, edgecolors="black", linewidth=0.5
    )

    # Add perfect prediction line
    max_val = max(np.max(true_flat), np.max(estimated_flat))
    axes[0].plot(
        [0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction"
    )

    # Compute and display correlation
    corr = np.corrcoef(true_flat, estimated_flat)[0, 1]
    mae = np.mean(np.abs(estimated_flat - true_flat))
    rmse = np.sqrt(np.mean((estimated_flat - true_flat) ** 2))

    axes[0].set_xlabel("True Distance", fontsize=12)
    axes[0].set_ylabel("Learned Distance", fontsize=12)
    axes[0].set_title(
        f"Learned vs True Distances\nCorr={corr:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Scatterplot 2: Graph vs True
    axes[1].scatter(
        true_flat,
        graph_flat,
        alpha=0.6,
        s=80,
        color="green",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add perfect prediction line
    max_val = max(np.max(true_flat), np.max(graph_flat))
    axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Match")

    # Compute correlation
    corr_graph = np.corrcoef(true_flat, graph_flat)[0, 1]
    mae_graph = np.mean(np.abs(graph_flat - true_flat))

    axes[1].set_xlabel("True Distance (Search)", fontsize=12)
    axes[1].set_ylabel("Graph Distance (Planning Graph)", fontsize=12)
    axes[1].set_title(
        f"Graph vs True Distances\nCorr={corr_graph:.3f}, MAE={mae_graph:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"test/distance_scatterplots": wandb.Image(fig)})
    plt.close(fig)  # Close the figure to free up memory

    # Error heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    error_matrix = estimated_distances - true_distances
    error_matrix[~valid_mask] = np.nan

    max_error = np.nanmax(np.abs(error_matrix))
    sns.heatmap(
        error_matrix,
        annot=show_annot,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        xticklabels=node_ids,
        yticklabels=node_ids,
        vmin=-max_error,
        vmax=max_error,
        cbar_kws={"label": "Error (Learned - True)"},
        ax=ax,
    )
    ax.set_title("Distance Estimation Error", fontsize=14, fontweight="bold")
    ax.set_xlabel("Target Node", fontsize=12)
    ax.set_ylabel("Source Node", fontsize=12)

    plt.tight_layout()
    wandb.log({"test/distance_error_heatmap": wandb.Image(fig)})
    plt.close(fig)  # Close the figure to free up memory


# =============================================================================
# Stage 3: Prune with Heuristic
# =============================================================================


def prune_with_heuristic(
    heuristic: "BaseHeuristic", max_shortcuts: int | None, use_multi_rl: bool = False
) -> GoalConditionedTrainingData:
    """Stage 3: Prune shortcuts using the heuristic.

    Args:
        heuristic: Trained heuristic
        config: Configuration dictionary

    Returns:
        Pruned training data
    """
    # print("\n" + "=" * 80)
    # print("STAGE 3: PRUNE WITH HEURISTIC")
    # print("=" * 80)

    # Call prune - interface is the same for all heuristics
    pruned_data = heuristic.prune(max_shortcuts=max_shortcuts, use_multi_rl=use_multi_rl)

    print(
        f"\nPruning complete: {len(pruned_data.unique_shortcuts)} unique shortcuts remaining"
    )
    print(f"  ({len(pruned_data.valid_shortcuts)} state-node pairs)")

    return pruned_data


# =============================================================================
# Stage 3.5: Random Selection
# =============================================================================


def random_selection(
    training_data: GoalConditionedTrainingData,
    max_shortcuts: int,
    rng: np.random.Generator,
) -> GoalConditionedTrainingData:
    """Stage 3.5: Randomly select up to max_shortcuts from pruned data.

    If max_shortcuts is 0, returns empty training data (pure planning mode).
    If max_shortcuts >= num shortcuts, returns all shortcuts unchanged.

    Note: We select from unique_shortcuts (node-node pairs) but keep all
    corresponding state-node pairs in valid_shortcuts for MultiRL training.

    Args:
        training_data: Pruned training data
        max_shortcuts: Maximum number of shortcuts to keep (0 = pure planning)
        rng: Random number generator

    Returns:
        Training data with randomly selected shortcuts
    """
    # print("\n" + "=" * 80)
    # print("STAGE 3.5: RANDOM SELECTION")
    # print("=" * 80)

    num_shortcuts = len(training_data.unique_shortcuts)

    # Handle edge cases
    if max_shortcuts == 0:
        print("max_shortcuts_per_graph = 0: Using pure planning (no shortcuts)")
        # Return empty training data
        return GoalConditionedTrainingData(
            states=[],
            current_atoms=[],
            goal_atoms=[],
            valid_shortcuts=[],
            unique_shortcuts=[],
            node_states=training_data.node_states,
            node_atoms=training_data.node_atoms,
            graph=training_data.graph,
            config={
                **training_data.config,
                "random_selection": True,
                "max_shortcuts_per_graph": max_shortcuts,
            },
        )

    if max_shortcuts >= num_shortcuts:
        print(
            f"max_shortcuts_per_graph ({max_shortcuts}) >= num shortcuts ({num_shortcuts}): Keeping all shortcuts"
        )
        return training_data

    # Random selection
    print(f"Randomly selecting {max_shortcuts} from {num_shortcuts} unique shortcuts")

    # Randomly select unique shortcuts (node-node pairs)
    print(training_data.unique_shortcuts)
    sample = rng.choice(
        training_data.unique_shortcuts, size=max_shortcuts, replace=False
    ).tolist()

    selected_unique_shortcuts = set((x, y) for x, y in sample)

    # Filter valid_shortcuts to keep only those matching selected unique shortcuts
    # valid_shortcuts has one entry per state-node pair, so we filter by node pairs
    selected_indices = []
    for i, (source_id, target_id) in enumerate(training_data.valid_shortcuts):
        if (source_id, target_id) in selected_unique_shortcuts:
            selected_indices.append(i)

    # Filter training data
    original_shortcut_info = training_data.config.get("shortcut_info", [])
    selected_shortcut_info = (
        [original_shortcut_info[i] for i in selected_indices]
        if original_shortcut_info
        else []
    )

    selected_data = GoalConditionedTrainingData(
        states=[training_data.states[i] for i in selected_indices],
        current_atoms=[training_data.current_atoms[i] for i in selected_indices],
        goal_atoms=[training_data.goal_atoms[i] for i in selected_indices],
        valid_shortcuts=[training_data.valid_shortcuts[i] for i in selected_indices],
        unique_shortcuts=list(selected_unique_shortcuts),
        node_states=training_data.node_states,
        node_atoms=training_data.node_atoms,
        graph=training_data.graph,
        config={
            **training_data.config,
            "shortcut_info": selected_shortcut_info,
            "random_selection": True,
            "max_shortcuts_per_graph": max_shortcuts,
        },
    )

    print(f"Selected {len(selected_data.unique_shortcuts)} unique shortcuts")
    print(f"  ({len(selected_data.valid_shortcuts)} state-node pairs)")

    return selected_data


# =============================================================================
# Stage 4: Train Policy
# =============================================================================


def train_policy(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy: Policy[ObsType, ActType],
    training_data: GoalConditionedTrainingData,
    cfg: DictConfig,
) -> None:
    """Stage 4: Train policy on final training data.

    Args:
        policy: Policy instance to train
        training_data: Final training data (after pruning and selection)
        system: TAMP system
        config: Configuration dictionary
        rng: Random number generator
    """
    # print("\n" + "=" * 80)
    # print("STAGE 4: TRAIN POLICY")
    # print("=" * 80)

    # Skip training if no shortcuts
    if len(training_data.valid_shortcuts) == 0:
        print("No shortcuts to train on - skipping policy training")
        return

    # Extract policy training parameters
    num_epochs = cfg.policy.n_epochs
    train_steps_per_shortcut = cfg.policy.max_episode_steps

    print(f"Training policy for {num_epochs} epochs")
    print(f"Max steps per shortcut: {train_steps_per_shortcut}")
    if hasattr(system.wrapped_env, "configure_training"):
        system.wrapped_env.configure_training(training_data)

    # Train policy
    policy.train(
        env=system.wrapped_env,
        train_data=training_data,
    )

    print("\nPolicy training complete")


# =============================================================================
# Stage 4b: Create Policy Dictionary
# =============================================================================


def create_policy_dictionary(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    heuristic: "BaseHeuristic",
    policy: MultiRLPolicy,
    training_data: GoalConditionedTrainingData,
    cfg: DictConfig,
    use_multi_rl: bool,
    system_cls: type | None = None,
    system_kwargs: dict | None = None,
) -> dict[tuple[int, int], Policy[ObsType, ActType]]:
    """Create dictionary D mapping (source, target) pairs to policy wrappers.

    If use_multi_rl=True, trains the MultiRL policy first, then creates wrappers.
    If use_multi_rl=False, creates wrappers using the heuristic's actor directly.

    Args:
        system: TAMP system
        heuristic: Trained heuristic (used if use_multi_rl=False)
        policy: MultiRLPolicy instance (trained here if use_multi_rl=True)
        training_data: Pruned training data with shortcuts
        cfg: Configuration dictionary
        use_multi_rl: If True, train and use MultiRL policies; if False, use heuristic

    Returns:
        Dictionary mapping (source_id, target_id) -> policy wrapper
    """
    # print("\n" + "=" * 80)
    # print("STAGE 4: CREATE POLICY DICTIONARY")
    # print("=" * 80)

    if len(training_data.unique_shortcuts) == 0:
        print("No shortcuts - returning empty dictionary")
        return {}

    D: dict[tuple[int, int], Policy[ObsType, ActType]] = {}

    if use_multi_rl:
        # First, train the MultiRL policy
        print("use_multi_rl=True: Training MultiRL policies...")
        print("-" * 40)

        if len(training_data.valid_shortcuts) > 0:
            num_epochs = cfg.policy.n_epochs
            train_steps_per_shortcut = cfg.policy.max_episode_steps

            print(f"Training policy for {num_epochs} epochs")
            print(f"Max steps per shortcut: {train_steps_per_shortcut}")

            if hasattr(system.wrapped_env, "configure_training"):
                system.wrapped_env.configure_training(training_data)

            policy.train(
                env=system.wrapped_env,
                train_data=training_data,
                system_cls=system_cls,
                system_kwargs=system_kwargs if system_cls else None,
            )
            print("Policy training complete")
        else:
            print("No training data - skipping policy training")

        # Now create wrappers from trained policies
        print("-" * 40)
        print("Creating policy dictionary from trained MultiRL policies...")
        for source_id, target_id in training_data.unique_shortcuts:
            source_atoms = training_data.node_atoms[source_id]
            target_atoms = training_data.node_atoms[target_id]

            # Create context for this shortcut
            context = PolicyContext(
                current_atoms=source_atoms,
                goal_atoms=target_atoms,
                info={"source_node_id": source_id, "target_node_id": target_id},
            )

            # Check if policy exists for this context
            key = policy._get_policy_key(context)
            if key in policy.policies:
                wrapper = MultiRLPolicyWrapper(policy, context)
                D[(source_id, target_id)] = wrapper
            else:
                print(f"  [WARN] No trained policy for shortcut {source_id}->{target_id}")
    else:
        # Use heuristic's actor directly - no training needed
        print("use_multi_rl=False: Using heuristic actor (no policy training)")
        print("-" * 40)
        print("Creating policy dictionary from heuristic actor...")
        for source_id, target_id in training_data.unique_shortcuts:
            wrapper = HeuristicPolicyWrapper(
                heuristic=heuristic,
                target_node_id=target_id,
            )
            D[(source_id, target_id)] = wrapper

    print(f"\nCreated {len(D)} policy wrappers")
    return D


# =============================================================================
# Stage 5: Add Shortcuts to Graph
# =============================================================================


def add_shortcuts_to_graph(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_dict: dict[tuple[int, int], Policy[ObsType, ActType]],
    training_data: GoalConditionedTrainingData,
) -> int:
    """Convert policy dictionary D to edges in the planning graph.

    For each (source, target) -> policy entry in D:
    - Create a LiftedOperator (preconditions = source atoms, effects = target atoms)
    - Create a ShortcutSkill wrapping the policy
    - Add operator and skill to system.components

    Args:
        system: TAMP system
        policy_dict: Dictionary mapping (source_id, target_id) -> policy wrapper
        training_data: Training data with node_atoms mapping

    Returns:
        Number of shortcuts added
    """
    # print("\n" + "=" * 80)
    # print("STAGE 5: ADD SHORTCUTS TO GRAPH")
    # print("=" * 80)

    if len(policy_dict) == 0:
        print("No policies in dictionary - skipping")
        return 0

    num_added = 0
    shortcut_count = 0

    for (source_id, target_id), policy_wrapper in policy_dict.items():
        source_atoms = training_data.node_atoms[source_id]
        target_atoms = training_data.node_atoms[target_id]

        # Create unique name for this shortcut
        shortcut_name = f"Shortcut_{shortcut_count}"
        shortcut_count += 1

        # Extract all unique objects from atoms to determine operator parameters
        all_objects: set[Object] = set()
        for atom in source_atoms | target_atoms:
            all_objects.update(atom.objects)

        # Create variables for each unique object
        parameters = [
            Variable(f"?{obj.name}", obj.type)
            for obj in sorted(all_objects, key=lambda o: o.name)
        ]

        # Handle edge case of no objects
        if not parameters:
            available_types = list(system.components.types)
            if available_types:
                parameters = [Variable("?obj", available_types[0])]
            else:
                print(f"  [WARN] Skipping shortcut {source_id}->{target_id}: no types available")
                continue

        # Create object -> variable mapping for lifting ground atoms
        obj_to_var = {
            obj: var
            for obj, var in zip(sorted(all_objects, key=lambda o: o.name), parameters)
        }

        # Lift ground atoms by replacing objects with variables
        def lift_atom(atom: GroundAtom) -> GroundAtom:
            lifted_objects = tuple(obj_to_var[obj] for obj in atom.objects)
            return atom.predicate(lifted_objects)

        lifted_source_atoms = {lift_atom(atom) for atom in source_atoms}
        lifted_target_atoms = {lift_atom(atom) for atom in target_atoms}

        # Determine add and delete effects
        add_effects = lifted_target_atoms - lifted_source_atoms
        delete_effects = lifted_source_atoms - lifted_target_atoms

        # Create the shortcut operator.
        # NOTE: _find_valid_groundings in base.py special-cases "Shortcut_*" operators
        # to match each variable ?<objname> directly to the object named <objname>,
        # giving exactly 1 grounding per node instead of O(|objects|^N).
        shortcut_operator = LiftedOperator(
            name=shortcut_name,
            parameters=parameters,
            preconditions=lifted_source_atoms,
            add_effects=add_effects,
            delete_effects=delete_effects,
        )

        # Create the skill wrapping the policy
        shortcut_skill = ShortcutSkill(
            policy_wrapper=policy_wrapper,
            operator=shortcut_operator,
            target_node_id=target_id,
            perceiver=system.perceiver,
            target_atoms=target_atoms,
        )

        # Check for collision with existing operators (re-graduation case)
        replaced = False
        for existing_op in system.components.operators:
            if existing_op == shortcut_operator:
                # Remove the stale skill for this operator and replace with the
                # fresher one from the most recent graduation.
                old_skills = {sk for sk in system.components.skills if sk.can_execute(existing_op)}
                system.components.skills -= old_skills
                system.components.skills.add(shortcut_skill)
                print(
                    f"  [INFO] Replaced skill for '{shortcut_name}' "
                    f"(re-graduated, {len(old_skills)} old skill(s) removed)"
                )
                replaced = True
                break

        if replaced:
            continue

        # Add operator and skill to system components
        system.components.operators.add(shortcut_operator)
        system.components.skills.add(shortcut_skill)
        num_added += 1

        # Debug output for first few shortcuts
        if num_added <= 3:
            print(f"  Added {shortcut_name}: node {source_id} -> node {target_id}")
            print(f"    Preconditions: {len(lifted_source_atoms)} atoms")
            print(f"    Add effects: {len(add_effects)} atoms")
            print(f"    Delete effects: {len(delete_effects)} atoms")

    print(f"\nAdded {num_added} shortcuts to planning graph")
    return num_added


# =============================================================================
# Stage 5.5: Test Shortcut Quality (Optional)
# =============================================================================


def test_shortcut_quality(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_dict: dict[tuple[int, int], Policy[ObsType, ActType]],
    training_data: GoalConditionedTrainingData,
    cfg: DictConfig,
) -> dict:
    """Test quality of trained shortcuts by running rollouts.

    Args:
        system: The TAMP system
        policy: Trained policy
        training_data: Training data with shortcuts
        config: Config with max_steps
    """
    from tamp_improv.approaches.improvisational.analyze import execute_shortcut_once

    print(f"\nTesting {len(training_data.unique_shortcuts)} shortcuts...")
    print("=" * 80)
    results = []
    successful_shortcut_paths = []

    max_steps = cfg.policy.max_episode_steps
    num_test_rollouts = cfg.collection.num_cost_rollouts # Test each shortcut multiple times

    success_counts = {}
    length_stats = {}

    for idx, (source_node, target_node) in enumerate(training_data.unique_shortcuts):
        policy = policy_dict[(source_node, target_node)]
        source_states = training_data.node_states[source_node]
        target_atoms = training_data.node_atoms[target_node]

        if source_states is None or target_atoms is None:
            continue

        # Ensure source_states is a list
        if not isinstance(source_states, list):
            source_states = [source_states]

        successes = 0
        lengths = []
        rollout_positions: list[list[list[float]]] = []

        for _ in range(num_test_rollouts):
            # Randomly sample a source state from the available states
            source_state = random.choice(source_states)

            # Execute shortcut once using helper function
            success, num_steps, path = execute_shortcut_once(
                policy=policy,
                system=system,
                start_state=source_state,
                goal_atoms=target_atoms,
                max_steps=max_steps,
                source_node_id=source_node,
                target_node_id=target_node,
            )

            if success:
                successes += 1
                lengths.append(num_steps)
                successful_shortcut_paths.append(path)

            # Store positions from this rollout (regardless of success)
            rollout_positions.append([_flatten_state(obs) for obs in path])

        success_rate = successes / num_test_rollouts
        avg_length = np.mean(lengths) if lengths else max_steps

        results.append(
            {
                "source_node": source_node,
                "target_node": target_node,
                "success_rate": success_rate,
                "avg_length": avg_length,
                "rollout_positions": rollout_positions,
            }
        )

        success_counts[(source_node, target_node)] = success_rate
        length_stats[(source_node, target_node)] = avg_length

        # Print progress every 5 shortcuts
        if (idx + 1) % 5 == 0 or (idx + 1) == len(training_data.unique_shortcuts):
            print(
                f"  Tested {idx + 1}/{len(training_data.unique_shortcuts)} shortcuts..."
            )

    # Print detailed results
    print("Mapping:", training_data.node_atoms)
    print("\nDetailed Results:")
    print("=" * 80)
    for (source_node, target_node), success_rate in success_counts.items():
        avg_length = length_stats[(source_node, target_node)]
        print(
            f"  Shortcut {source_node}->{target_node}: "
            f"success={success_rate:.1%}, avg_length={avg_length:.1f}"
        )

    # Summary statistics
    if success_counts:
        overall_success = np.mean(list(success_counts.values()))
        overall_length = np.mean(list(length_stats.values()))
        num_successful = sum(1 for sr in success_counts.values() if sr > 0.5)
        print("\n" + "=" * 80)
        print(f"Overall Statistics:")
        print(f"  Average success rate: {overall_success:.1%}")
        print(f"  Average length (when successful): {overall_length:.1f} steps")
        print(f"  Shortcuts with >50% success: {num_successful}/{len(success_counts)}")
        print("=" * 80)

    full_results = {
        "pairs": results,
        "avg_success_rate": overall_success,
        "avg_steps": overall_length,
        "successful_shortcut_paths": successful_shortcut_paths,
    }

    return full_results


# =============================================================================
# Stage 6: Evaluate
# =============================================================================


def evaluate_approach(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach,
    cfg: DictConfig,
) -> Metrics:
    """Stage 6: Evaluate the trained approach.

    Args:
        system: TAMP system
        approach: Trained approach
        config: Configuration dictionary
        num_eval_episodes: Number of evaluation episodes

    Returns:
        Evaluation metrics
    """
    # print("\n" + "=" * 80)
    # print("STAGE 5: EVALUATE")
    # print("=" * 80)

    num_eval_episodes = cfg.evaluation.num_episodes

    import psutil as _psutil
    print(f"Running {num_eval_episodes} evaluation episodes...")
    print(f"[MEM] eval entry: {_psutil.Process().memory_info().rss / 1e9:.2f} GB")

    # Create TrainingConfig for evaluation
    eval_config = TrainingConfig(
        render=cfg.evaluation.render,
        eval_max_steps=cfg.evaluation.max_episode_steps,
        fast_eval=cfg.evaluation.fast_eval,
    )
    print(f"[MEM] after TrainingConfig: {_psutil.Process().memory_info().rss / 1e9:.2f} GB")

    # Reseed env before eval so results are independent of training RNG state
    print("[MEM] calling env.reset...")
    system.env.reset(seed=cfg.seed + 9999)
    print(f"[MEM] after env.reset: {_psutil.Process().memory_info().rss / 1e9:.2f} GB")

    # Run evaluations
    rewards = []
    lengths = []
    successes = []
    all_episode_data = []

    for ep in range(num_eval_episodes):
        print(f"[DBG] evaluate_approach: starting ep {ep + 1}/{num_eval_episodes}")
        if (ep + 1) % 1 == 0:
            print(f"  Completed {ep + 1}/{num_eval_episodes} episodes")

        print(f"[DBG] evaluate_approach: about to run_evaluation_episode ep={ep}")
        reward, length, success, episode_data = run_evaluation_episode_with_caching(
            system=system,
            approach=approach,
            policy_name="MultiRL",
            config=eval_config,
            episode_num=ep,
        )
        print(f"[DBG] evaluate_approach: run_evaluation_episode done ep={ep}, success={success}, steps={length}")
        rewards.append(reward)
        lengths.append(length)
        successes.append(success)
        all_episode_data.append(episode_data)

    # Create Metrics object
    avg_metrics = Metrics(
        success_rate=sum(successes) / len(successes) if successes else 0.0,
        avg_episode_length=sum(lengths) / len(lengths) if lengths else 0.0,
        avg_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        episode_data=all_episode_data,
    )

    print("\nEvaluation complete:")
    print(f"  Success rate: {avg_metrics.success_rate:.2%}")
    print(f"  Avg steps: {avg_metrics.avg_episode_length:.1f}")
    print(f"  Avg reward: {avg_metrics.avg_reward:.3f}s")

    return avg_metrics


# =============================================================================
# Latent Embedding Extraction
# =============================================================================


def get_latent_embeddings(
    heuristic: "BaseHeuristic",
    n_states_per_node: int = 20,
) -> dict[int, tuple[list[float], list[tuple[list[float], list[float]]]]] | None:
    """Extract latent embeddings for all nodes and their associated states.

    Supports CRL v2 heuristics that expose ``sa_encoder`` and ``g_encoder``:
        node embedding  = g_encoder(atom_vec)
        state embedding = sa_encoder(state, zeros)  (action set to zero for simplicity)

    Args:
        heuristic: Trained heuristic instance.
        n_states_per_node: Maximum number of states to embed per node.

    Returns:
        Dict mapping ``node_id -> (node_embedding, [(state_flat, state_embedding), ...])``
        where all embeddings are plain Python lists (pickle-safe).
        Returns ``None`` if the heuristic does not expose CRL encoder networks.
    """
    import torch
    import torch.nn.functional as F

    if not (hasattr(heuristic, "sa_encoder") and hasattr(heuristic, "g_encoder")):
        return None

    device = torch.device(heuristic.config.device)
    normalize = getattr(heuristic.config, "normalize_embeddings", False)

    heuristic.sa_encoder.eval()
    heuristic.g_encoder.eval()

    result: dict[int, tuple[list[float], list[tuple[list[float], list[float]]]]] = {}

    with torch.no_grad():
        zero_action = torch.zeros(1, heuristic.action_dim).to(device)

        for node_id, atoms in heuristic._node_atoms_dict.items():
            atom_vec = heuristic.create_atom_vector(atoms)
            atom_tensor = torch.FloatTensor(atom_vec).unsqueeze(0).to(device)

            # Node embedding via goal encoder
            g_emb = heuristic.g_encoder(atom_tensor)
            if normalize:
                g_emb = F.normalize(g_emb, p=2, dim=-1)
            node_emb_list: list[float] = g_emb.squeeze(0).cpu().numpy().tolist()

            # State embeddings (zero action for all states)
            node_states = heuristic.training_data.node_states.get(node_id, [])
            sampled = random.sample(node_states, min(n_states_per_node, len(node_states)))

            state_emb_pairs: list[tuple[list[float], list[float]]] = []
            for state in sampled:
                state_flat = heuristic._flatten_state(state)
                state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)

                sa_emb = heuristic.sa_encoder(state_tensor, zero_action)
                if normalize:
                    sa_emb = F.normalize(sa_emb, p=2, dim=-1)

                state_emb_list: list[float] = sa_emb.squeeze(0).cpu().numpy().tolist()
                state_emb_pairs.append((state_flat.tolist(), state_emb_list))

            result[node_id] = (node_emb_list, state_emb_pairs)

    return result


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    cfg: DictConfig,
    output_dir: Path | None = None,
    system_cls: type | None = None,
    system_kwargs: dict | None = None,
) -> PipelineResults:
    """Run the complete SLAP pipeline.

    Args:
        system: TAMP system
        cfg: Configuration dictionary with all parameters
        output_dir: Hydra output directory for checkpointing (None disables checkpointing)

    Returns:
        PipelineResults with serializable outputs from the pipeline
    """
    results = PipelineResults()
    results.config = OmegaConf.to_container(cfg, resolve=True)

    ckpt_dir = Path(output_dir) if output_dir else None

    # Setup RNG
    seed = cfg.seed
    rng = np.random.Generator(np.random.PCG64(seed))
    set_torch_seed(seed)
    random.seed(seed)

    print("WANDB ENABLED:", cfg.wandb_enabled)

    if cfg.wandb_enabled:
        wandb_run_name = os.getenv("WANDB_RUN_NAME", None)
        wandb.init(
            project="slap_gridworld_fixed",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=wandb_run_name,
        )

    # Extract grid config before anything else
    results.grid_config = _extract_grid_config(system)

    # Create policy instance based on config
    policy = create_policy(cfg=cfg)

    # Create approach (wraps system + policy)
    approach = ImprovisationalTAMPApproach(
        system,
        policy,
        seed=seed,
        planner_id=cfg.collection.planner_id,
        max_skill_steps=cfg.collection.max_steps_per_edge,
    )

    times: dict[str, float] = {}

    # ── Resume from checkpoint ────────────────────────────────────────
    resume_path = getattr(cfg, "resume_from", None)
    if resume_path:
        resume_path = Path(resume_path)
        print(f"RESUMING FROM: {resume_path}")

        # Load training data
        td_path = resume_path / "training_data"
        if td_path.exists():
            print(f"  Loading training data from {td_path}")
            training_data = GoalConditionedTrainingData.load(td_path)
        elif cfg.collection.load_data and cfg.collection.training_data_path:
            print(f"  Loading training data from {cfg.collection.training_data_path}")
            training_data = GoalConditionedTrainingData.load(Path(cfg.collection.training_data_path))
        else:
            raise FileNotFoundError(f"No training data found at {td_path} and load_data not configured")
        graph_distances = compute_graph_distances(training_data.graph, exclude_shortcuts=True)
        first_edge_dict = compute_first_edge_dict(training_data.graph)

        # Store serializable pre-training data
        node_atoms_ser, node_states_ser = _extract_node_data(training_data)
        results.node_atoms = node_atoms_ser
        results.node_states = node_states_ser
        results.unique_shortcuts = list(training_data.unique_shortcuts)
        results.all_shortcuts = list(training_data.unique_shortcuts)
        results.graph_distances = dict(graph_distances)
        results.true_distances = {}

        # Create heuristic and load trained weights
        heuristic = create_heuristic(
            training_data=copy.deepcopy(training_data),
            graph_distances=graph_distances,
            system=copy.deepcopy(system),
            cfg=cfg,
            rng=rng,
            first_edge_dict=first_edge_dict,
        )
        heuristic_path = resume_path / "heuristic"
        if heuristic_path.exists() and hasattr(heuristic, "load"):
            print(f"  Loading trained heuristic from {heuristic_path}")
            heuristic.load(str(heuristic_path))
        else:
            raise FileNotFoundError(f"No heuristic checkpoint at {heuristic_path}")

        # Load previous results to carry over training metrics
        prev_results_path = resume_path / "results.pkl"
        if prev_results_path.exists():
            print(f"  Loading previous results from {prev_results_path}")
            with open(prev_results_path, "rb") as f:
                prev_results = pickle.load(f)
            results.training_rounds = getattr(prev_results, "training_rounds", [])
            results.true_distances = getattr(prev_results, "true_distances", {})
            results.latent_embeddings = getattr(prev_results, "latent_embeddings", None)
        else:
            results.training_rounds = []

        # Extract heuristic metrics from the loaded heuristic (distance estimates, gains, etc.)
        results.training_rounds.append(
            _extract_heuristic_round_data(heuristic, training_data)
        )

        print("  Skipping Stages 1-2, resuming at Stage 3 (pruning)")
        times["collection_time"] = 0.0
        times["heuristic_training_time"] = 0.0

    if not resume_path:
        # Stage 1: Collect training data
        print("STAGE 1: COLLECT TRAINING DATA")
        start = time.time()
        if cfg.collection.load_data:
            if cfg.collection.training_data_path is None:
                raise ValueError("load_data=True but training_data_path is not set in config")
            data_path = Path(cfg.collection.training_data_path)
            if not data_path.exists():
                raise FileNotFoundError(
                    f"load_data=True but training data not found at: {data_path}"
                )
            print(f"Loading training data from {data_path} ...")
            training_data = GoalConditionedTrainingData.load(data_path)
            gd_path = data_path / "graph_distances.pkl"
            fed_path = data_path / "first_edge_dict.pkl"
            if gd_path.exists() and fed_path.exists():
                print("Loading precomputed graph_distances and first_edge_dict ...")
                with open(gd_path, "rb") as f:
                    graph_distances = pickle.load(f)
                with open(fed_path, "rb") as f:
                    first_edge_dict = pickle.load(f)
            else:
                print("Computing graph_distances and first_edge_dict (no cached files found) ...")
                graph_distances = compute_graph_distances(training_data.graph, exclude_shortcuts=True)
                first_edge_dict = compute_first_edge_dict(training_data.graph)
            g = training_data.graph
            num_nodes = len(g.nodes) if g else 0
            num_edges = len(g.edges) if g else 0
            total_states = sum(
                (len(v) if isinstance(v, list) else 1)
                for v in training_data.node_states.values()
            )
            print(f"  Graph:            {num_nodes} nodes, {num_edges} edges")
            print(f"  Node states:      {len(training_data.node_states)} nodes with states ({total_states} total states)")
            print(f"  Unique shortcuts: {len(training_data.unique_shortcuts)}")
            print(f"  Valid shortcuts:  {len(training_data.valid_shortcuts)} (with per-state duplicates)")
            print(f"  Node atoms:       {len(training_data.node_atoms)} nodes with atoms")
        else:
            training_data, graph_distances, first_edge_dict = collect_training_data(
                system=system,
                approach=approach,
                cfg=cfg,
                rng=rng,
            )
            if cfg.collection.training_data_path is not None:
                data_path = Path(cfg.collection.training_data_path)
                data_path.mkdir(parents=True, exist_ok=True)
                training_data.save(data_path)
                print(f"Saved training data to {data_path}")
        times["collection_time"] = time.time() - start

        # Checkpoint: save training data so collection doesn't need to be re-run
        if ckpt_dir:
            td_path = ckpt_dir / "training_data"
            td_path.mkdir(parents=True, exist_ok=True)
            training_data.save(td_path)
            with open(ckpt_dir / "graph_distances.pkl", "wb") as f:
                pickle.dump(graph_distances, f)
            with open(ckpt_dir / "first_edge_dict.pkl", "wb") as f:
                pickle.dump(first_edge_dict, f)
            print(f"[CKPT] Saved Stage 1 training data to {ckpt_dir}")

        # Store serializable pre-training data
        node_atoms_ser, node_states_ser = _extract_node_data(training_data)
        results.node_atoms = node_atoms_ser
        results.node_states = node_states_ser
        results.unique_shortcuts = list(training_data.unique_shortcuts)
        results.all_shortcuts = list(training_data.unique_shortcuts)
        results.graph_distances = dict(graph_distances)

        # Compute and store true distances (only for environments that support it)
        from tamp_improv.benchmarks.gridworld_continuous import GridworldContinuousTAMPSystem
        if isinstance(system, GridworldContinuousTAMPSystem):
            print("Computing true distances for all node pairs...")
            results.true_distances = _extract_true_distances(system, training_data)
        else:
            print("Skipping true distance computation (not supported for this environment)")
            results.true_distances = {}

        # Create heuristic instance based on config
        start = time.time()
        if cfg.heuristic.type == "crl":
            heuristic_config = dataclass_from_cfg(CRLHeuristicConfig, cfg.heuristic)
        elif cfg.heuristic.type == "dqn":
            heuristic_config = dataclass_from_cfg(DQNHeuristicConfig, cfg.heuristic.dqn)
        elif cfg.heuristic.type == "cmd":
            heuristic_config = dataclass_from_cfg(CMDHeuristicConfig, cfg.heuristic)
        else:
            heuristic_config = None

        heuristic = create_heuristic(
            training_data=copy.deepcopy(training_data),
            graph_distances=graph_distances,
            system=copy.deepcopy(system),
            cfg=cfg,
            rng=rng,
            first_edge_dict=first_edge_dict,
        )

        # Stage 2: Train heuristic
        if cfg.heuristic.max_shortcuts_per_graph is None or cfg.heuristic.max_shortcuts_per_graph > 0:
            print("STAGE 2: TRAIN HEURISTIC")
            _ckpt_save_fn = None
            if ckpt_dir and hasattr(heuristic, "save"):
                _heuristic_ckpt_path = str(ckpt_dir / "heuristic")
                def _ckpt_save_fn():
                    heuristic.save(_heuristic_ckpt_path)
                    print(f"[CKPT] Saved heuristic checkpoint to {_heuristic_ckpt_path}")
            results.training_rounds = train_heuristic(
                heuristic=heuristic,
                cfg=cfg,
                checkpoint_callback=_ckpt_save_fn,
            )
            # Final heuristic save after training completes
            if ckpt_dir and hasattr(heuristic, "save"):
                heuristic.save(str(ckpt_dir / "heuristic"))
                print(f"[CKPT] Saved trained heuristic to {ckpt_dir / 'heuristic'}")
        else:
            print("STAGE 2: SKIP HEURISTIC TRAINING (max_shortcuts_per_graph is 0 or None)")

        times["heuristic_training_time"] = time.time() - start

    # Extract latent embeddings for visualization (CRL v2 only; None for others)
    if cfg.debug:
        results.latent_embeddings = get_latent_embeddings(heuristic)

    # Stage 2.5: Test heuristic quality (optional)
    if cfg.debug:
        print("STAGE 2.5: TEST HEURISTIC QUALITY")
        print(
            f"[DEBUG] training_data.graph.nodes (ids): {[node.id for node in training_data.graph.nodes]}"
        )
        print(
            f"[DEBUG] training_data.unique_shortcuts: {training_data.unique_shortcuts}"
        )
        test_heuristic_quality(
            system=system,
            heuristic=heuristic,
            training_data=training_data,
            graph_distances=graph_distances,
            cfg=cfg,
            times=times,
        )

    if cfg.eval_heuristic_only:
        results.times = times
        return results

    # Stage 3: Prune with heuristic
    print("STAGE 3: PRUNE WITH HEURISTIC")
    start = time.time()
    pruned_training_data = prune_with_heuristic(
        heuristic=heuristic,
        max_shortcuts=cfg.heuristic.max_shortcuts_per_graph,
        use_multi_rl=cfg.policy.use_multi_rl,
    )
    results.pruned_shortcuts = list(pruned_training_data.unique_shortcuts)
    times["heuristic_pruning_time"] = time.time() - start

    # Checkpoint: save pruned training data
    if ckpt_dir:
        ptd_path = ckpt_dir / "pruned_training_data"
        ptd_path.mkdir(parents=True, exist_ok=True)
        pruned_training_data.save(ptd_path)
        print(f"[CKPT] Saved pruned training data to {ptd_path}")

    # Stage 4: Create policy dictionary (conditionally trains MultiRL if use_multi_rl=True)
    print("STAGE 4: CREATE POLICY DICTIONARY")
    start = time.time()
    use_multi_rl = cfg.policy.use_multi_rl
    policy_dict = create_policy_dictionary(
        system=system,
        heuristic=copy.deepcopy(heuristic),
        policy=policy,
        training_data=pruned_training_data,
        cfg=cfg,
        use_multi_rl=use_multi_rl,
        system_cls=system_cls,
        system_kwargs=system_kwargs,
    )
    times["policy_training_time"] = time.time() - start

    # Checkpoint: save trained policies
    if ckpt_dir and use_multi_rl and hasattr(policy, "save"):
        policy.save(str(ckpt_dir / "multi_rl_policy"))
        print(f"[CKPT] Saved trained policies to {ckpt_dir / 'multi_rl_policy'}")

    import psutil as _psutil
    def _mem_gb() -> str:
        return f"{_psutil.Process().memory_info().rss / 1e9:.2f} GB"

    print(f"[MEM] After policy training: {_mem_gb()}")

    # Stage 5: Add shortcuts to graph
    print("STAGE 5: ADD SHORTCUTS TO GRAPH")
    virtual_system = copy.deepcopy(approach.system)
    print(f"[MEM] After deepcopy(approach.system): {_mem_gb()}")
    start = time.time()
    num_shortcuts_added = add_shortcuts_to_graph(
        system=virtual_system,
        policy_dict=policy_dict,
        training_data=pruned_training_data,
    )
    times["add_shortcuts_time"] = time.time() - start

    # Stage 5.5: Test shortcut quality — always run when shortcuts exist,
    # so we can populate the edge cost table for fast eval path planning.
    shortcut_quality_results = None
    if len(pruned_training_data.valid_shortcuts) > 0:
        print("STAGE 5.5: TEST SHORTCUT QUALITY")
        shortcut_quality_results = test_shortcut_quality(
            system=virtual_system,
            policy_dict=policy_dict,
            training_data=pruned_training_data,
            cfg=cfg,
        )

    if shortcut_quality_results is not None:
        results.shortcut_quality_results = shortcut_quality_results["pairs"]

    # Update the approach's system with the new graph containing shortcuts for evaluation
    approach.update_system(virtual_system)
    print(f"[MEM] After approach.update_system: {_mem_gb()}")

    # Build edge cost table from training graph for fast eval path planning
    approach.fast_eval = cfg.evaluation.fast_eval
    approach._edge_cost_table = {
        (edge.source.atoms, edge.target.atoms): edge.cost
        for edge in training_data.graph.edges
        if edge.cost is not None and edge.cost != float("inf")
    }
    # Add shortcut costs from Stage 5.5 quality test results
    if shortcut_quality_results is not None:
        for pair in shortcut_quality_results["pairs"]:
            src_atoms = training_data.node_atoms.get(pair["source_node"])
            tgt_atoms = training_data.node_atoms.get(pair["target_node"])
            if src_atoms is not None and tgt_atoms is not None:
                approach._edge_cost_table[(frozenset(src_atoms), frozenset(tgt_atoms))] = pair["avg_length"]
    print(f"fast_eval={cfg.evaluation.fast_eval}, built edge cost table with {len(approach._edge_cost_table)} entries")

    # Clear relevant_objects on the eval system's ImprovWrapper so base skills
    # receive full (unfiltered) observations — MultiRL training sets this during
    # training and it is not cleared on reset(), causing base skills to fail.
    if hasattr(system.env, "set_relevant_objects"):
        system.env.set_relevant_objects(None)

    # Stage 6: Evaluate
    print("STAGE 6: EVALUATE")
    start = time.time()
    eval_metrics = evaluate_approach(
        system=system,
        approach=approach,
        cfg=cfg,
    )
    times["evaluation_time"] = time.time() - start

    # Store serializable evaluation results
    results.eval_episodes = eval_metrics.episode_data
    results.avg_success_rate = eval_metrics.success_rate
    results.avg_steps = eval_metrics.avg_episode_length
    results.avg_reward = eval_metrics.avg_reward
    results.times = times

    if cfg.wandb_enabled:
        wandb.finish()

    return results
