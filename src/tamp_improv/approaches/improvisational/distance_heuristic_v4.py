"""Goal-conditioned distance heuristic V4 using contrastive state-node learning.

This module implements a learned distance function f(s, g) that estimates
the number of steps required to reach goal node g from source state s.

V4 improves on V3 by conditioning the policy on goal NODES rather than goal STATES.
This solves the problem where invalid state pairs (e.g., different robot/goal combinations)
could be sampled as shortcuts during training.

Key components:
- s_encoder: Neural network that maps states to k-dimensional embeddings
- g_encoder: Learnable matrix (S x k) where row i is the embedding of node i
- Contrastive loss: Aligns (current_state, future_node) pairs and separates negative pairs
- Normalized embeddings: All embeddings are L2-normalized to unit length
"""
import time
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, TypeVar, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class DistanceHeuristicV4Config:
    """Configuration for contrastive state-node distance heuristic."""

    # Network architecture
    latent_dim: int = 32  # Dimension of embedding space (k)
    hidden_dims: list[int] | None = None  # Hidden layer sizes [64, 64]
    normalize_embeddings: bool = True  # Whether to L2-normalize embeddings to unit sphere

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 256
    buffer_size: int = 10000
    grad_clip: float = 1.0  # Gradient clipping threshold

    # Contrastive learning
    gamma: float = 0.99  # For geometric sampling of future states
    repetition_factor: int = 4  # CRTR: repeat each trajectory this many times in batch
    policy_temperature: float = 1.0
    eval_temperature: float = 0.0

    # Training
    iters_per_epoch: int = 1  # Gradient steps per training call
    learn_frequency: int = 10  # Learn every N epochs

    # Multi-round training
    num_rounds: int = 1  # Number of pruning rounds (1 = no pruning, just normal train)
    keep_fraction: float = 0.5  # Fraction of node-node pairs to keep each round

    # Device
    device: str = "cuda"  # "cuda" or "cpu"


class StateEncoder(nn.Module):
    """Neural network that encodes states to k-dimensional embeddings.

    Maps state (state_dim-dimensional vector) -> embedding (k-dimensional vector)
    """

    def __init__(self, state_dim: int, latent_dim: int, hidden_dims: list[int] | None = None):
        """Initialize the state encoder network.

        Args:
            state_dim: Dimension of input state
            latent_dim: Dimension of output embedding (k)
            hidden_dims: List of hidden layer sizes
        """
        super(StateEncoder, self).__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Default hidden dims
        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Encode states to embeddings.

        Args:
            states: Batch of states (batch_size, state_dim) - float tensor

        Returns:
            Embeddings (batch_size, latent_dim)
        """
        return self.network(states)


class ContrastiveReplayBuffer:
    """Replay buffer for storing complete trajectories.

    Implements CRTR (Contrastive Representations for Temporal Reasoning) sampling
    to encourage learning temporal dynamics rather than static context.

    For V4: Stores trajectories as sequences of node IDs rather than states.
    """

    def __init__(
        self,
        capacity: int,
        gamma: float = 0.99,
        repetition_factor: int = 4,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of trajectories to store
            gamma: Discount factor (used for geometric sampling of future states)
            repetition_factor: Number of times to repeat each trajectory in a batch
                              (for CRTR within-trajectory negatives)
        """
        self.capacity = capacity
        self.gamma = gamma
        self.repetition_factor = repetition_factor

        # Storage for complete trajectories
        # Each trajectory is a dict with:
        #   - 'states': list of flattened states (NDArray)
        #   - 'nodes': list of node IDs (int)
        self.trajectories: deque = deque(maxlen=capacity)

        # Store metadata for each trajectory (start state, goal node)
        self.trajectory_metadata: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Return total number of trajectories stored."""
        return len(self.trajectories)

    def store_trajectory(
        self,
        states: list[NDArray],
        nodes: list[int],
        start_state: NDArray | None = None,
        goal_node: int | None = None
    ) -> None:
        """Store complete trajectory with metadata.

        Args:
            states: List of flattened states visited during episode
            nodes: List of node IDs corresponding to states
            start_state: Starting state of the trajectory
            goal_node: Goal node ID for the trajectory
        """
        if len(states) < 2 or len(states) != len(nodes):
            return

        # Store the trajectory
        self.trajectories.append({
            'states': states,
            'nodes': nodes,
        })

        # Store metadata
        self.trajectory_metadata.append({
            'start_state': start_state,
            'goal_node': goal_node,
        })

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample batch using CRTR strategy.

        Implements Algorithm 1 from "Contrastive Representations for Temporal Reasoning":
        For each sample in batch:
        1. Sample a trajectory uniformly
        2. Sample t0 uniformly from [0, T-1]
        3. Sample t1 from geometric distribution starting at t0+1

        Returns:
            Dictionary containing:
                - current_states: (batch_size, state_dim) - states at time t0
                - future_nodes: (batch_size,) - node IDs at time t1
        """
        if len(self.trajectories) == 0:
            raise ValueError("Buffer is empty")

        # Ensure batch_size is divisible by repetition_factor
        assert batch_size % self.repetition_factor == 0

        # Sample unique trajectory IDs
        num_unique_trajs = batch_size // self.repetition_factor
        unique_traj_ids = np.random.choice(
            len(self.trajectories),
            size=num_unique_trajs,
            replace=False if num_unique_trajs <= len(self.trajectories) else True
        )

        # Repeat each trajectory ID repetition_factor times
        traj_ids = np.repeat(unique_traj_ids, self.repetition_factor)

        current_states = []
        future_nodes = []

        # For each trajectory in the batch, sample (t0, t1) pair
        for traj_id in traj_ids:
            trajectory = self.trajectories[traj_id]
            states = trajectory['states']
            nodes = trajectory['nodes']
            traj_len = len(states)

            if traj_len < 2:
                current_states.append(states[0])
                future_nodes.append(nodes[-1])
                continue

            # Sample t0 uniformly from trajectory
            t0 = np.random.randint(0, traj_len - 1)

            # Sample t1 using geometric distribution
            num_future = traj_len - t0 - 1
            p = 1 - self.gamma
            offset = min(
                np.random.geometric(p) - 1,
                num_future - 1
            )
            # print("Offset:", offset)
            t1 = t0 + 1 + offset

            # print("Start node:", nodes[t0])
            # print("Goal node:", nodes[t1])

            current_states.append(states[t0])
            future_nodes.append(nodes[t1])

        # Stack into tensors
        current_states = torch.FloatTensor(np.array(current_states))
        future_nodes = torch.LongTensor(future_nodes)

        return {
            "current_states": current_states,
            "future_nodes": future_nodes,
        }


def contrastive_loss_normalized(
    encode_state_fn: Callable[[torch.Tensor], torch.Tensor],
    encode_node_fn: Callable[[torch.Tensor], torch.Tensor],
    current_states: torch.Tensor,
    future_nodes: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Compute contrastive loss with embeddings.

    This implements a simplified version of the contrastive loss:
    - phi = encode_state_fn(current_state)
    - psi = encode_node_fn(future_node)
    - l_align: ||phi - psi||^2 (align positive pairs)
    - l_unif: logsumexp over negative pair distances (uniformity)

    Args:
        encode_state_fn: Function to encode states to embeddings
        encode_node_fn: Function to encode node IDs to embeddings
        current_states: Batch of current states (batch_size, state_dim)
        future_nodes: Batch of future node IDs (batch_size,)

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Get embeddings using provided encoding functions
    phi = encode_state_fn(current_states)  # (batch_size, k)
    psi = encode_node_fn(future_nodes)  # (batch_size, k)

    batch_size = phi.shape[0]

    # Alignment loss: ||phi - psi||^2 for positive pairs
    l_align = torch.mean((phi - psi) ** 2, dim=1)  # (batch_size,)

    # Pairwise distances: ||phi[i] - psi[j]||^2 for all i, j
    pdist = torch.sum((phi[:, None] - psi[None]) ** 2, dim=-1)  # (batch_size, batch_size)

    # Uniformity loss: logsumexp over negative pairs
    # Mask out diagonal (positive pairs) with identity matrix
    I = torch.eye(batch_size, device=phi.device)

    l_unif = (
        torch.logsumexp(-(pdist * (1 - I)), dim=1) +
        torch.logsumexp(-(pdist.T * (1 - I)), dim=1)
    ) / 2.0

    # Combined contrastive loss
    loss = l_align + l_unif

    # Accuracy: how often is the closest future node the correct one?
    # We need to check if future_nodes[argmin_idx] == future_nodes[i] (the true node)
    # Note: pdist[i, j] = distance from phi[i] to psi[j], where psi[j] = embedding of future_nodes[j]
    # So argmin(pdist[i]) gives index j where phi[i] is closest to psi[j]
    # And future_nodes[j] is the node that phi[i] is closest to
    closest_node_indices = torch.argmin(pdist, dim=1)  # (batch_size,) - which psi is each phi closest to?
    predicted_nodes = future_nodes[closest_node_indices]  # (batch_size,) - which node is each state closest to?
    true_nodes = future_nodes  # (batch_size,) - which node should each state be close to?
    accuracy = torch.mean((predicted_nodes == true_nodes).float())

    # Total loss
    total_loss = loss.mean()

    # Metrics
    metrics = {
        'loss': total_loss.item(),
        'l_unif': l_unif.mean().item(),
        'l_align': l_align.mean().item(),
        'accuracy': accuracy.item(),
    }

    return total_loss, metrics


class DistanceHeuristicV4:
    """Contrastive state-node distance heuristic.

    Learns an embedding space where L2 distance correlates with trajectory distance.
    Conditions policy on goal NODES rather than goal STATES.
    """

    def __init__(
        self,
        env: Any,
        perceiver: Any,
        atoms_to_node: dict,
        node_to_states: dict,
        config: DistanceHeuristicV4Config | None = None,
        seed: int | None = None,
    ):
        """Initialize distance heuristic V4.

        Args:
            config: Configuration for training
            seed: Random seed
        """
        self.config = config or DistanceHeuristicV4Config()
        self.seed = seed
        self.env = env
        self.perceiver = perceiver
        self.atoms_to_node = atoms_to_node
        self.node_to_states = node_to_states

        # Create get_node function
        def get_node(state: ObsType) -> int:
            """Convert state to node ID."""
            atoms = self.perceiver.step(state)
            return self.atoms_to_node[frozenset(atoms)]



        self.get_node = get_node
        # Callable[[ObsType], int]
        self.num_nodes = len(atoms_to_node)

        if self.config.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Networks
        self.s_encoder: StateEncoder | None = None  # State encoder network
        self.g_encoder: torch.Tensor | None = None  # Goal node embedding matrix (S x k)
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ContrastiveReplayBuffer | None = None

        # Training state
        self.total_episodes = 0
        self.policy_temperature = config.policy_temperature

    def _encode_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Encode state tensor to embedding.

        Args:
            state_tensor: State tensor (batch_size, state_dim) or (state_dim,)

        Returns:
            Embedding tensor with same batch dimensions
        """
        emb = self.s_encoder(state_tensor)
        if self.config.normalize_embeddings:
            # Normalize to unit L2 norm
            return F.normalize(emb, p=2, dim=-1)
        return emb

    def _encode_node(self, node_id: int | torch.Tensor) -> torch.Tensor:
        """Encode node ID(s) to embedding.

        Args:
            node_id: Node ID (int) or tensor of node IDs (batch_size,)

        Returns:
            Embedding tensor (latent_dim,) or (batch_size, latent_dim)
        """
        emb = self.g_encoder[node_id]
        if self.config.normalize_embeddings:
            # Normalize to unit L2 norm
            return F.normalize(emb, p=2, dim=-1)
        return emb

    def _select_action_greedy(
        self,
        env: Any,
        current_state: ObsType,
        goal_node: int,
    ) -> int | None:
        """Select action using greedy policy over learned embedding space.

        Uses softmax over distances in embedding space (lower distance → higher probability).

        Args:
            env: Environment
            current_state: Current state
            goal_node: Goal node ID

        Returns:
            Selected action, or None if no valid actions
        """
        # Get available actions
        available_actions = list(range(env.action_space.n))

        if len(available_actions) == 0:
            return None

        # If networks not trained yet, use random policy
        if self.s_encoder is None or self.g_encoder is None:
            print("Outputting random action")
            return random.choice(available_actions)

        # Get goal node embedding
        with torch.no_grad():
            goal_emb = self._encode_node(goal_node).cpu().numpy()  # (k,)

        # Compute distances in embedding space for each action
        distances = []
        for action in available_actions:
            # Save current state
            env.reset_from_state(current_state)

            # Take action to get next state
            next_state, _, _, _, _ = env.step(action)

            # Encode next state
            next_flat = self._flatten_state(next_state)
            with torch.no_grad():
                next_tensor = torch.FloatTensor(next_flat).unsqueeze(0).to(self.device)
                next_emb = self._encode_state(next_tensor).squeeze(0).cpu().numpy()  # (k,)

            # Compute L2 distance to goal in embedding space
            dist = 0.5 * np.linalg.norm(next_emb - goal_emb)**2
            distances.append(dist)

            # print("Distance of next state:", next_flat[1:3], dist)

        distances = np.array(distances)

        if self.policy_temperature == 0:
            return available_actions[np.argmin(distances)]

        # Softmax over negative distances (lower distance = higher probability)
        logits = -distances / self.policy_temperature
        # Numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)
        # print("Current State:", self._flatten_state(current_state))
        # print("Actions:", available_actions)
        # print("Distances:", distances)
        # print("Probabilities:", probabilities)
        # Sample action
        action = np.random.choice(available_actions, p=probabilities)
        # print("Selected action:", action)
        return action

    def rollout(
        self,
        env: Any,
        start_state: ObsType,
        goal_node: int,
        max_steps: int = 100,
        temperature: float | None = None,
    ) -> tuple[list[NDArray], list[int], bool]:
        """Rollout the learned policy from start state to goal node.

        Args:
            env: Environment
            start_state: Starting state
            goal_node: Goal node ID to reach
            max_steps: Maximum number of steps
            temperature: Policy temperature (uses self.policy_temperature if None)

        Returns:
            Tuple of (states, nodes, success) where:
                - states: List of flattened states visited
                - nodes: List of node IDs visited
                - success: Whether the goal node was reached
        """
        if temperature is not None:
            old_temp = self.policy_temperature
            self.policy_temperature = temperature

        # Reset to start state
        current_state, _ = env.reset_from_state(start_state)

        states = [self._flatten_state(current_state)]
        # print(self.get_node)
        nodes = [self.get_node(current_state)]

        success = False


        for _ in range(max_steps):
            # Check if reached goal
            current_node = self.get_node(current_state)
            if current_node == goal_node:
                success = True
                break

            # Select action
            action = self._select_action_greedy(env, current_state, goal_node)
            if action is None:
                break

            # Step environment
            current_state, _, _, _, _ = env.step(action)

            # Store state and node
            states.append(self._flatten_state(current_state))
            nodes.append(self.get_node(current_state))

        # Restore temperature
        if temperature is not None:
            self.policy_temperature = old_temp

        return states, nodes, success

    def collect_trajectories(
        self,
        env: Any,
        state_node_pairs: list[tuple[ObsType, int]],
        num_trajectories: int,
        max_episode_steps: int = 100,
    ) -> tuple[list[dict], list[dict], dict]:
        """Collect trajectories by rolling out from state-node pairs.

        Uses learned policy (greedy over embedding space) if available,
        otherwise uses random policy.

        Args:
            env: Base environment (must support reset_from_state)
            state_node_pairs: List of (source_state, target_node) pairs
            num_trajectories: Number of trajectories to collect
            max_episode_steps: Maximum steps per trajectory

        Returns:
            Tuple of (trajectories, trajectory_metadata, stats_dict)
        """
        trajectories = []
        trajectory_metadata = []
        successes = []
        lengths = []

        for _ in range(num_trajectories):
            # Sample random state-node pair
            source_state, target_node = random.choice(state_node_pairs)

            # Rollout from source to target
            states, nodes, success = self.rollout(
                env, source_state, target_node, max_steps=max_episode_steps
            )

            trajectories.append({
                'states': states,
                'nodes': nodes,
            })
            trajectory_metadata.append({
                'start_state': self._flatten_state(source_state),
                'goal_node': target_node,
            })
            successes.append(success)
            lengths.append(len(states))

        # Compute statistics
        success_rate = np.mean(successes) if successes else 0.0
        avg_length = np.mean(lengths) if lengths else 0.0
        success_lengths = [l for l, s in zip(lengths, successes) if s]
        avg_success_length = np.mean(success_lengths) if success_lengths else 0.0

        stats = {
            'success_rate': success_rate,
            'avg_length': avg_length,
            'avg_success_length': avg_success_length,
            'num_successes': sum(successes),
            'num_trajectories': num_trajectories
        }

        return trajectories, trajectory_metadata, stats

    def multi_train(
        self,
        state_node_pairs: list[tuple[ObsType, int]],
        graph_distances: dict[tuple[int, int], float],
        num_rounds: int = 5,
        num_epochs_per_round: int = 200,
        trajectories_per_epoch: int = 10,
        max_episode_steps: int = 100,
        keep_fraction: float = 0.5,
    ) -> dict:
        """Multi-round training that iteratively prunes to promising shortcuts.

        For each round:
        1. Train on current subset of state-node pairs
        2. Compute estimated vs graph distances at NODE-NODE level
        3. Keep top keep_fraction of NODE-NODE pairs with best differences
        4. Keep ALL state-node pairs for the selected node-node pairs
        5. Continue training on this pruned subset

        This focuses learning on node-node pairs that show promise for shortcuts.

        Args:
            state_node_pairs: List of (source_state, target_node) pairs
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
            num_rounds: Number of pruning rounds
            num_epochs_per_round: Training epochs per round
            trajectories_per_epoch: Trajectories collected per epoch
            max_episode_steps: Max steps per trajectory
            keep_fraction: Fraction of NODE-NODE pairs to keep each round (0.5 = keep top 50%)

        Returns:
            Dictionary with combined training history across all rounds
        """
        print(f"\n{'='*80}")
        print(f"MULTI-ROUND TRAINING: {num_rounds} rounds, {num_epochs_per_round} epochs/round")
        print(f"Starting with {len(state_node_pairs)} state-node pairs")
        print(f"Keep fraction: {keep_fraction} (pruning {1-keep_fraction:.1%} each round)")
        print(f"{'='*80}\n")

        # Build mapping: (source_node, target_node) -> list of (source_state, target_node) pairs
        print("Building node-node to state-node mapping...")
        node_pair_to_state_pairs: dict[tuple[int, int], list[tuple[ObsType, int]]] = {}
        for state, target_node in state_node_pairs:
            source_node = self.get_node(state)
            key = (source_node, target_node)
            if key not in node_pair_to_state_pairs:
                node_pair_to_state_pairs[key] = []
            node_pair_to_state_pairs[key].append((state, target_node))

        print(f"Found {len(node_pair_to_state_pairs)} unique node-node pairs")
        print(f"Average {len(state_node_pairs) / len(node_pair_to_state_pairs):.1f} state-node pairs per node-node pair")

        # Combined history across all rounds
        combined_history = {
            'total_loss': [],
            'alignment_loss': [],
            'uniformity_loss': [],
            'accuracy': [],
            'success_rate': [],
            'avg_success_length': [],
            'learning_rate': [],
            'policy_temperature': [],
            'round_boundaries': [],  # Track where each round starts
            'num_state_pairs_per_round': [],  # Track state-node dataset size per round
            'num_node_pairs_per_round': [],  # Track node-node dataset size per round
            'distance_matrices': [],  # Distance matrix after each round (for debugging)
            'graph_distances': graph_distances,  # Store graph distances for reference
        }

        # Start with all node-node pairs
        current_node_pairs = list(node_pair_to_state_pairs.keys())

        for round_idx in range(num_rounds):
            # Get all state-node pairs for current node-node pairs
            current_state_pairs = []
            for node_pair in current_node_pairs:
                current_state_pairs.extend(node_pair_to_state_pairs[node_pair])

            print(f"\n{'='*80}")
            print(f"ROUND {round_idx + 1}/{num_rounds}")
            print(f"Training on {len(current_node_pairs)} node-node pairs")
            print(f"  -> {len(current_state_pairs)} state-node pairs")
            print(f"{'='*80}\n")

            # Mark start of this round in history
            combined_history['round_boundaries'].append(len(combined_history['total_loss']))
            combined_history['num_state_pairs_per_round'].append(len(current_state_pairs))
            combined_history['num_node_pairs_per_round'].append(len(current_node_pairs))

            # Train on current subset
            round_history = self.train(
                state_node_pairs=current_state_pairs,
                num_epochs=num_epochs_per_round,
                trajectories_per_epoch=trajectories_per_epoch,
                max_episode_steps=max_episode_steps,
            )

            # Append this round's history
            for key in ['total_loss', 'alignment_loss', 'uniformity_loss', 'accuracy',
                       'success_rate', 'avg_success_length', 'learning_rate', 'policy_temperature']:
                combined_history[key].extend(round_history[key])

            # Compute distance matrix for ALL node-node pairs (not just current ones)
            # This allows us to "bring back" previously pruned pairs if they become promising
            print(f"\n[DEBUG] Computing distance matrix for ALL {len(node_pair_to_state_pairs)} node pairs after round {round_idx + 1}...")
            all_node_pairs = list(node_pair_to_state_pairs.keys())
            distance_matrix = self._compute_distance_matrix(all_node_pairs)
            combined_history['distance_matrices'].append({
                'round': round_idx + 1,
                'distances': distance_matrix,
                'active_node_pairs': current_node_pairs.copy(),  # Which pairs we trained on
            })
            print(f"[DEBUG] Distance matrix computed with {len(distance_matrix)} entries")

            # If not the last round, prune to most promising NODE-NODE pairs
            # IMPORTANT: Evaluate ALL node pairs, not just current ones!
            if round_idx < num_rounds - 1:
                print(f"\n{'='*80}")
                print(f"PRUNING after round {round_idx + 1}")
                print(f"{'='*80}\n")

                # Call prune() to get selected node pairs
                selected_node_pairs_set = self.prune(
                    graph_distances,
                    keep_fraction=keep_fraction,
                    max_episode_steps=max_episode_steps,
                )
                current_node_pairs = list(selected_node_pairs_set)

                # Count resulting state-node pairs
                resulting_state_pairs = sum(
                    len(node_pair_to_state_pairs.get((src, tgt), []))
                    for src, tgt in current_node_pairs
                )

                print(f"Pruned node-node pairs: {len(all_node_pairs)} -> {len(current_node_pairs)}")
                print(f"Resulting state-node pairs: {resulting_state_pairs}")

        # Final state-node pairs
        final_state_pairs = []
        for node_pair in current_node_pairs:
            final_state_pairs.extend(node_pair_to_state_pairs[node_pair])

        print(f"\n{'='*80}")
        print(f"MULTI-ROUND TRAINING COMPLETE")
        print(f"Final node-node pairs: {len(current_node_pairs)}")
        print(f"Final state-node pairs: {len(final_state_pairs)}")
        print(f"{'='*80}\n")

        return combined_history

    def _compute_distance_matrix(self, node_pairs: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        """Compute estimated distances for all node pairs.

        Args:
            node_pairs: List of (source_node, target_node) pairs to compute distances for

        Returns:
            Dictionary mapping (source_node, target_node) -> estimated distance
        """
        distance_matrix = {}
        for source_node, target_node in node_pairs:
            est_dist = self.estimate_node_distance(source_node, target_node)
            distance_matrix[(source_node, target_node)] = est_dist
        return distance_matrix

    def train(
        self,
        state_node_pairs: list[tuple[ObsType, int]],
        num_epochs: int = 1000,
        trajectories_per_epoch: int = 10,
        max_episode_steps: int = 100,
    ) -> dict:
        """Train distance heuristic using contrastive state-node learning.

        Args:
            env: Base environment
            state_node_pairs: List of (source_state, target_node) pairs
            perceiver: Perceiver to convert states to atoms
            atoms_to_node: Dictionary mapping frozenset of atoms to node ID
            num_epochs: Number of training epochs
            trajectories_per_epoch: Number of trajectories to collect per epoch
            max_episode_steps: Maximum steps per episode

        Returns:
            Dictionary containing training history with keys:
                - total_loss: List of total loss values per epoch
                - alignment_loss: List of alignment loss values per epoch
                - uniformity_loss: List of uniformity loss values per epoch
                - accuracy: List of accuracy values per epoch
                - success_rate: List of policy success rates per epoch
                - avg_success_length: List of avg successful trajectory lengths per epoch
                - learning_rate: List of learning rates per epoch
                - policy_temperature: List of policy temperatures per epoch
        """
        print(f"\nTraining distance heuristic V4 on {len(state_node_pairs)} state-node pairs...")
        print(f"Device: {self.device}")
        print(f"Number of nodes: {self.num_nodes}")

        # Determine state dimension
        sample_state = state_node_pairs[0][0]
        state_flat = self._flatten_state(sample_state)
        state_dim = state_flat.shape[0]

        print(f"State dimension: {state_dim}")
        print(f"Latent dimension: {self.config.latent_dim}")

        # Create state encoder network
        self.s_encoder = StateEncoder(
            state_dim=state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims or [64, 64],
        ).to(self.device)

        # Initialize goal node embedding matrix (S x k)
        self.g_encoder = nn.Parameter(
            torch.randn(self.num_nodes, self.config.latent_dim, device=self.device) * 0.01
        )

        # Create optimizer for all parameters
        self.optimizer = torch.optim.Adam(
            list(self.s_encoder.parameters()) + [self.g_encoder],
            lr=self.config.learning_rate
        )

        # Create cosine annealing learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=self.config.learning_rate
        )

        # Create replay buffer
        self.replay_buffer = ContrastiveReplayBuffer(
            capacity=self.config.buffer_size,
            gamma=self.config.gamma,
            repetition_factor=self.config.repetition_factor,
        )

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Collecting {trajectories_per_epoch} trajectories per epoch")
        print(f"Using cosine annealing LR scheduler: {self.config.learning_rate} -> {self.config.learning_rate * 0.01}")

        # Temperature annealing parameters
        initial_temp = self.config.policy_temperature
        min_temp = self.config.eval_temperature

        # Initialize training history tracking
        training_history = {
            'total_loss': [],
            'alignment_loss': [],
            'uniformity_loss': [],
            'accuracy': [],
            'success_rate': [],
            'avg_success_length': [],
            'learning_rate': [],
            'policy_temperature': [],
        }

        # Training loop
        for epoch in range(num_epochs):
            print("Epoch", epoch)
            # Update policy temperature using cosine annealing
            progress = (epoch+1) / num_epochs
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            self.policy_temperature = min_temp + (initial_temp - min_temp) * cosine_factor

            # Collect trajectories
            trajectories, trajectory_metadata, traj_stats = self.collect_trajectories(
                self.env,
                state_node_pairs,
                num_trajectories=trajectories_per_epoch,
                max_episode_steps=max_episode_steps,
            )

            # Store trajectories in replay buffer with metadata
            for trajectory, metadata in zip(trajectories, trajectory_metadata):
                self.replay_buffer.store_trajectory(
                    states=trajectory['states'],
                    nodes=trajectory['nodes'],
                    start_state=metadata['start_state'],
                    goal_node=metadata['goal_node']
                )
                self.total_episodes += 1

            # Train if we have enough data (only every learn_frequency epochs)
            if (epoch > 0 and
                epoch % self.config.learn_frequency == 0 and
                len(self.replay_buffer) >= self.config.batch_size):

                metrics = self._train_step()

                # Track metrics in history
                current_lr = self.optimizer.param_groups[0]['lr']
                training_history['total_loss'].append(metrics['loss'])
                training_history['alignment_loss'].append(metrics['l_align'])
                training_history['uniformity_loss'].append(metrics['l_unif'])
                training_history['accuracy'].append(metrics['accuracy'])
                training_history['success_rate'].append(traj_stats['success_rate'])
                training_history['avg_success_length'].append(traj_stats['avg_success_length'])
                training_history['learning_rate'].append(current_lr)
                training_history['policy_temperature'].append(self.policy_temperature)

                # Logging after learning
                print(f"\n[Epoch {epoch}/{num_epochs}]")
                print(f"  Buffer size: {len(self.replay_buffer)}")
                print(f"  Learning rate: {current_lr:.6f}")
                print(f"  Policy temperature: {self.policy_temperature:.4f}")
                print(f"  --- Policy Performance ---")
                print(f"  Success rate: {traj_stats['success_rate']:.2%} ({traj_stats['num_successes']}/{traj_stats['num_trajectories']})")
                print(f"  Avg trajectory length: {traj_stats['avg_length']:.1f}")
                if traj_stats['num_successes'] > 0:
                    print(f"  Avg successful length: {traj_stats['avg_success_length']:.1f}")
                print(f"  --- Training Metrics ---")
                print(f"  Total loss: {metrics['loss']:.4f}")
                print(f"  Alignment loss: {metrics['l_align']:.4f}")
                print(f"  Uniformity loss: {metrics['l_unif']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")

            # Step the learning rate scheduler every epoch
            self.scheduler.step()

        print(f"\nTraining complete!")
        return training_history

    def _train_step(self) -> dict:
        """Perform one training step (batch update).

        Returns:
            Dictionary of training metrics
        """
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)
        # Move to device
        current_states = batch["current_states"].to(self.device)
        future_nodes = batch["future_nodes"].to(self.device)

        # Run gradient descent
        for _ in range(self.config.iters_per_epoch):
            self.optimizer.zero_grad()

            # Compute loss using encoding functions
            loss, metrics = contrastive_loss_normalized(
                self._encode_state,
                self._encode_node,
                current_states,
                future_nodes,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(self.s_encoder.parameters()) + [self.g_encoder],
                max_norm=self.config.grad_clip
            )

            # Update parameters
            self.optimizer.step()

        return metrics

    def latent_dist(self, source_state: ObsType, target_node: int) -> float:
        """Compute distance in latent space between state and node.

        Args:
            source_state: Source state
            target_node: Target node ID

        Returns:
            L2 distance in embedding space
        """
        if self.s_encoder is None or self.g_encoder is None:
            return 0.0

        source_flat = self._flatten_state(source_state)

        # Encode using encoding functions
        with torch.no_grad():
            source_tensor = torch.FloatTensor(source_flat).unsqueeze(0).to(self.device)
            source_emb = self._encode_state(source_tensor).squeeze(0)
            target_emb = self._encode_node(target_node)

            # Compute L2 distance in embedding space
            distance = torch.norm(source_emb - target_emb).item()

        return float(distance)

    def estimate_distance(self, source_state: ObsType, target_node: int) -> float:
        """Estimate trajectory distance from state to node.

        For V4, we use a simple transformation of the latent distance.
        This can be refined based on empirical results.

        Args:
            source_state: Source state
            target_node: Target node ID

        Returns:
            Estimated number of steps to reach target from source
        """
        d_sg_sq = (self.latent_dist(source_state, target_node))**2
        d_gg_sq = 0
        target_states = self.node_to_states[target_node]

        # print(target_states)

        n = 0
        for target_state in random.sample(target_states, min(100, len(target_states))):
            d_gg_sq += (self.latent_dist(target_state, target_node))**2 
            n += 1
        d_gg_sq /= n

        print("D_gg:", d_gg_sq, "D_sg:", d_sg_sq)

        return (1 / (2 * np.log(self.config.gamma))) * (d_gg_sq - d_sg_sq)

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        source_states = self.node_to_states[source_node]
        avg = 0
        n = 0
        for source_state in random.sample(source_states, min(100, len(source_states))):
            avg += self.estimate_distance(source_state, target_node)
            n += 1
        return avg / n

    def prune(
        self,
        graph_distances: dict[tuple[int, int], float],
        keep_fraction: float = 0.5,
        max_episode_steps: int | None = None,
    ) -> set[tuple[int, int]]:
        """Prune node pairs based on estimated vs graph distances.

        Returns the union of:
        1. Node pairs where estimated_distance < graph_distance (promising shortcuts)
        2. Bottom keep_fraction of node pairs by (estimated - graph) distance

        This ensures we keep both:
        - Shortcuts that look learnable (estimated < graph)
        - The most promising shortcuts overall (lowest estimated - graph)

        Additionally filters out pairs where estimated distance exceeds max_episode_steps,
        since those are unreachable within the episode limit.

        Args:
            graph_distances: Dictionary mapping (source_node, target_node) -> graph distance
            keep_fraction: Fraction of total node pairs to keep (0.0 to 1.0)
            max_episode_steps: Maximum episode steps (filters out pairs with est_dist > this)

        Returns:
            Set of (source_node, target_node) tuples to keep
        """
        print(f"\n[DEBUG] Pruning with keep_fraction={keep_fraction}, max_episode_steps={max_episode_steps}")

        # Score all node pairs: score = estimated_distance - graph_distance
        # Negative scores = shortcuts (estimated < graph)
        # Lower scores = more promising
        node_pair_scores = []
        filtered_by_max_steps = 0

        for (source_node, target_node), graph_dist in graph_distances.items():
            # Estimate distance using the heuristic
            est_dist = self.estimate_node_distance(source_node, target_node)

            # Filter out pairs that exceed max_episode_steps
            if max_episode_steps is not None and est_dist > max_episode_steps:
                filtered_by_max_steps += 1
                continue

            # Score: lower is better
            # Negative = shortcut (estimated < graph)
            score = est_dist - graph_dist

            node_pair_scores.append((score, source_node, target_node, est_dist))

        print(f"[DEBUG] Scored {len(node_pair_scores)} node pairs")
        if max_episode_steps is not None:
            print(f"[DEBUG] Filtered out {filtered_by_max_steps} pairs with est_dist > {max_episode_steps}")

        # Sort by score (ascending = most promising first)
        node_pair_scores.sort(key=lambda x: x[0])

        # Strategy 1: Keep all negative scores (shortcuts)
        shortcuts = set()
        for score, source_node, target_node, est_dist in node_pair_scores:
            if score < 0:
                shortcuts.add((source_node, target_node))

        print(f"[DEBUG] Found {len(shortcuts)} shortcuts (negative scores)")

        # Strategy 2: Keep top keep_fraction by score
        num_to_keep = max(1, int(len(node_pair_scores) * keep_fraction))
        top_pairs = set()
        for score, source_node, target_node, est_dist in node_pair_scores[:num_to_keep]:
            top_pairs.add((source_node, target_node))

        print(f"[DEBUG] Keeping top {num_to_keep} pairs ({keep_fraction:.1%})")

        # Return union
        result = shortcuts | top_pairs
        print(f"[DEBUG] Total pairs to keep: {len(result)} (union of shortcuts and top)")

        return result

    def _flatten_state(self, state: ObsType) -> np.ndarray:
        """Flatten state to array."""
        if hasattr(state, "nodes"):
            return state.nodes.flatten().astype(np.float32) #[1:3]
        return np.array(state).flatten().astype(np.float32)

    def save(self, path: str) -> None:
        """Save distance heuristic to disk."""
        os.makedirs(path, exist_ok=True)

        # Save networks
        torch.save(self.s_encoder.state_dict(), f"{path}/s_encoder.pt")
        torch.save(self.g_encoder, f"{path}/g_encoder.pt")

        # Save config
        with open(f"{path}/config.pkl", "wb") as f:
            pickle.dump(self.config, f)

        print(f"Distance heuristic V4 saved to {path}")

    def load(self, path: str, state_dim: int, num_nodes: int) -> None:
        """Load distance heuristic from disk."""
        # Load config
        with open(f"{path}/config.pkl", "rb") as f:
            self.config = pickle.load(f)

        self.num_nodes = num_nodes

        # Recreate networks
        self.s_encoder = StateEncoder(
            state_dim=state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims or [64, 64],
        ).to(self.device)

        # Load weights
        self.s_encoder.load_state_dict(torch.load(f"{path}/s_encoder.pt", map_location=self.device))
        self.s_encoder.eval()

        # Load g_encoder
        self.g_encoder = torch.load(f"{path}/g_encoder.pt", map_location=self.device)

        print(f"Distance heuristic V4 loaded from {path}")
