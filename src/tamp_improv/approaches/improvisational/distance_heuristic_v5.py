"""Goal-conditioned distance heuristic V5 using contrastive state-node learning.

This module implements a learned distance function f(s, g) that estimates
the number of steps required to reach goal node g from source state s.

V5 improves on V3 by conditioning the policy on goal NODES rather than goal STATES.
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
class DistanceHeuristicV5Config:
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

class GoalEncoder(nn.Module):
    """Embedding matrix for discrete goal nodes.

    Maps node ID (integer) -> embedding (k-dimensional vector).
    Since nodes are discrete, we use a simple embedding matrix rather than a neural network.
    """

    def __init__(self, num_nodes: int, latent_dim: int):
        """Initialize goal encoder.

        Args:
            num_nodes: Number of discrete goal nodes
            latent_dim: Dimension of output embedding (k)
        """
        super(GoalEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.out_dim = latent_dim

        # Learnable embedding matrix (num_nodes x latent_dim)
        self.embedding = nn.Embedding(num_nodes, latent_dim)

        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Encode node IDs to embeddings.

        Args:
            node_ids: Tensor of node IDs (batch_size,) or scalar

        Returns:
            Embeddings (batch_size, latent_dim) or (latent_dim,)
        """
        return self.embedding(node_ids)


class CostNet(nn.Module):
    """Scalar network that outputs c(g) for each goal node.

    This is a simple embedding that maps each goal node to a scalar cost value.
    In CMD-1, the energy function is f(s,g) = c(g) - d(s,g), where c(g) is
    an upper bound on the cost-to-go from any state to goal g.
    """

    def __init__(self, g_encoder: GoalEncoder):
        """Initialize cost network.

        Args:
            num_nodes: Number of discrete goal nodes
        """
        super(CostNet, self).__init__()

        self.g_encoder = g_encoder

        # Get output dimensions from encoders
        goal_enc_dim = g_encoder.latent_dim

        self.scalar_net = nn.Sequential(nn.Linear(goal_enc_dim, goal_enc_dim // 2),
                                        nn.ReLU(),
                                        nn.Linear(goal_enc_dim // 2, 1))


    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """Get cost value c(g) for goal nodes.

        Args:
            g: Tensor of node ids (batch_size,) or scalar

        Returns:
            Cost values (batch_size, 1) or (1,)
        """

        goal_encodings = self.g_encoder(g)

        return self.scalar_net(goal_encodings)
    

class MRN(nn.Module):
    """Metric Residual Network for learning quasi-metric distances.

    Combines symmetric and asymmetric components to learn d(s,g) that
    respects triangle inequality. Returns NEGATIVE distance (for energy function).
    """

    def __init__(self, s_encoder: StateEncoder, g_encoder: GoalEncoder, sym_dim: int = 64, asym_dim: int = 16):
        """Initialize MRN.

        Args:
            s_encoder: State encoder network
            g_encoder: Goal encoder network
            sym_dim: Dimension of symmetric component
            asym_dim: Dimension of asymmetric component
        """
        super().__init__()
        self.s_encoder = s_encoder
        self.g_encoder = g_encoder

        # Get output dimensions from encoders
        state_enc_dim = s_encoder.latent_dim
        goal_enc_dim = g_encoder.latent_dim

        # Symmetrical head
        self.s_sym = nn.Linear(state_enc_dim, sym_dim)
        self.g_sym = nn.Linear(goal_enc_dim, sym_dim)

        # Asymmetrical head
        self.s_asym = nn.Linear(state_enc_dim, asym_dim)
        self.g_asym = nn.Linear(goal_enc_dim, asym_dim)

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Compute negative distance -d(s,g).

        Args:
            s: State tensor (batch_size, state_dim) or (state_dim,)
            g: Goal node IDs (batch_size,) or scalar

        Returns:
            Negative distances (batch_size, 1) or (1,)
        """
        # Encode state and goal
        s_enc = self.s_encoder(s)
        g_enc = self.g_encoder(g)

        # Symmetric component
        sym_s = self.s_sym(s_enc)
        sym_g = self.g_sym(g_enc)
        dist_sym = (sym_s - sym_g).pow(2).mean(-1, keepdim=True)

        # Asymmetric component
        asym_s = self.s_asym(s_enc)
        asym_g = self.g_asym(g_enc)
        res = F.relu(asym_s - asym_g)
        dist_asym = (F.softmax(res, -1) * res).sum(-1, keepdim=True)

        # Total distance (both components are non-negative)
        dist = dist_sym + dist_asym

        # Return NEGATIVE distance for energy function
        return -dist
    
class ContrastiveReplayBuffer:
    """Replay buffer for storing complete trajectories.

    Implements CRTR (Contrastive Representations for Temporal Reasoning) sampling
    to encourage learning temporal dynamics rather than static context.

    For V5: Stores trajectories as sequences of node IDs rather than states.
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


def cmd_contrastive_loss(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    current_states: torch.Tensor,
    future_nodes: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute CMD-1 contrastive loss using energy-based formulation.

    Implements the InfoNCE loss using energy function f(s,g) = c(g) - d(s,g):
    - For each state s_i with positive goal g_pos_i, compute energies for all goals
    - The positive pair (s_i, g_pos_i) should have higher energy than negatives
    - Loss = cross_entropy(energies / temperature, labels)

    Args:
        energy_fn: Function that computes f(s,g) = c(g) - d(s,g)
        current_states: Batch of current states (batch_size, state_dim)
        future_nodes: Batch of positive goal nodes (batch_size,)
        temperature: Temperature for softmax

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    batch_size = current_states.shape[0]

    # Build energy matrix: energy[i, j] = f(s_i, g_j)
    # We compute energies for all (state, goal) pairs in the batch
    # Positive pairs are on the diagonal (i.e., f(s_i, g_i))

    # Create expanded tensors for pairwise computation
    states_expanded = current_states.unsqueeze(1).repeat(1, batch_size, 1)  # (B, B, state_dim)
    goals_expanded = future_nodes.unsqueeze(0).repeat(batch_size, 1)  # (B, B)

    # Flatten for batch computation
    states_flat = states_expanded.reshape(-1, current_states.shape[-1])  # (B*B, state_dim)
    goals_flat = goals_expanded.reshape(-1)  # (B*B,)

    # Compute all energies
    all_energies = energy_fn(states_flat, goals_flat)  # (B*B,)
    energy_matrix = all_energies.reshape(batch_size, batch_size)  # (B, B)

    # Logits for cross-entropy: positive is at index i for row i
    logits = energy_matrix / temperature  # (B, B)

    # Labels: positive pair is on diagonal
    labels = torch.arange(batch_size, dtype=torch.long, device=current_states.device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    # Accuracy: how often is the highest energy the correct goal?
    predicted_goals = torch.argmax(logits, dim=1)
    accuracy = torch.mean((predicted_goals == labels).float())

    # Additional metrics
    pos_energies = torch.diagonal(energy_matrix)  # (batch_size,)
    neg_energies = energy_matrix[~torch.eye(batch_size, dtype=bool, device=energy_matrix.device)]

    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'pos_energy_mean': pos_energies.mean().item(),
        'neg_energy_mean': neg_energies.mean().item(),
        'energy_gap': (pos_energies.mean() - neg_energies.mean()).item(),
    }

    return loss, metrics


class DistanceHeuristicV5:
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
        config: DistanceHeuristicV5Config | None = None,
        seed: int | None = None,
    ):
        """Initialize distance heuristic V5.

        Args:
            config: Configuration for training
            seed: Random seed
        """
        self.config = config or DistanceHeuristicV5Config()
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

        # Networks (will be initialized in train())
        self.s_encoder: StateEncoder | None = None  # State encoder network
        self.g_encoder: GoalEncoder | None = None  # Goal node encoder (embedding matrix)
        self.c_net: CostNet | None = None  # Cost network c(g)
        self.d_net: MRN | None = None  # Distance network d(s,g) (MRN)
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ContrastiveReplayBuffer | None = None

        # Training state
        self.total_episodes = 0
        self.policy_temperature = config.policy_temperature

    def energy(self, states: torch.Tensor, goal_nodes: torch.Tensor) -> torch.Tensor:
        """Compute energy function f(s,g) = c(g) - d(s,g).

        Args:
            states: State tensor (batch_size, state_dim)
            goal_nodes: Goal node IDs (batch_size,)

        Returns:
            Energy values (batch_size,)
        """
        # c(g): cost upper bound for each goal
        c_g = self.c_net(goal_nodes).squeeze(-1)  # (batch_size,)

        # -d(s,g): negative distance (MRN returns negative distance)
        neg_d_sg = self.d_net(states, goal_nodes).squeeze(-1)  # (batch_size,)

        print("C:", c_g.mean().item(), " -D:", neg_d_sg.mean().item())

        # f(s,g) = c(g) - d(s,g) = c(g) + (-d(s,g))
        energy = c_g + neg_d_sg

        return energy

    def _select_action_greedy(
        self,
        env: Any,
        current_state: ObsType,
        goal_node: int,
    ) -> int | None:
        """Select action using greedy policy over learned distance function.

        Uses softmax over distances d(s',g) (lower distance → higher probability).

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
        if self.d_net is None:
            print("Outputting random action")
            return random.choice(available_actions)

        # Create goal node tensor
        goal_tensor = torch.tensor([goal_node], dtype=torch.long, device=self.device)

        # Compute distances for each action
        distances = []
        for action in available_actions:
            # Save current state
            env.reset_from_state(current_state)

            # Take action to get next state
            next_state, _, _, _, _ = env.step(action)

            # Flatten and tensorize next state
            next_flat = self._flatten_state(next_state)
            next_tensor = torch.FloatTensor(next_flat).unsqueeze(0).to(self.device)

            # Compute distance: d(s',g) = -MRN(s',g)
            with torch.no_grad():
                neg_dist = self.d_net(next_tensor, goal_tensor).squeeze()  # MRN returns -d
                dist = -neg_dist.item()  # Get positive distance

            distances.append(dist)

        distances = np.array(distances)

        if self.policy_temperature == 0:
            return available_actions[np.argmin(distances)]

        # Softmax over negative distances (lower distance = higher probability)
        logits = -distances / self.policy_temperature
        # Numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)

        # Sample action
        action = np.random.choice(available_actions, p=probabilities)
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
        print(f"\nTraining distance heuristic V5 on {len(state_node_pairs)} state-node pairs...")
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

        # Create goal encoder (embedding matrix for discrete nodes)
        self.g_encoder = GoalEncoder(
            num_nodes=self.num_nodes,
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        # Create cost network c(g)
        self.c_net = CostNet(g_encoder=self.g_encoder).to(self.device)

        # Create distance network d(s,g) using MRN
        self.d_net = MRN(
            s_encoder=self.s_encoder,
            g_encoder=self.g_encoder,
            sym_dim=64,
            asym_dim=16,
        ).to(self.device)

        # Create optimizer for all parameters
        all_params = (
            list(self.s_encoder.parameters()) +
            list(self.g_encoder.parameters()) +
            list(self.c_net.parameters()) +
            list(self.d_net.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=self.config.learning_rate)

        # Create cosine annealing learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=self.config.learning_rate * 0.01
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
            'accuracy': [],
            'pos_energy_mean': [],
            'neg_energy_mean': [],
            'energy_gap': [],
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
                training_history['accuracy'].append(metrics['accuracy'])
                training_history['pos_energy_mean'].append(metrics['pos_energy_mean'])
                training_history['neg_energy_mean'].append(metrics['neg_energy_mean'])
                training_history['energy_gap'].append(metrics['energy_gap'])
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
                print(f"  Accuracy: {metrics['accuracy']:.2%}")
                print(f"  Positive energy: {metrics['pos_energy_mean']:.4f}")
                print(f"  Negative energy: {metrics['neg_energy_mean']:.4f}")
                print(f"  Energy gap: {metrics['energy_gap']:.4f}")

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

            # Compute loss using CMD-1 contrastive loss
            loss, metrics = cmd_contrastive_loss(
                self.energy,
                current_states,
                future_nodes,
                temperature=10.0,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            all_params = (
                list(self.s_encoder.parameters()) +
                list(self.g_encoder.parameters()) +
                list(self.c_net.parameters()) +
                list(self.d_net.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.config.grad_clip)

            # Update parameters
            self.optimizer.step()

        return metrics

    def estimate_distance(self, source_state: ObsType, target_node: int) -> float:
        """Estimate trajectory distance from state to node using learned distance function.

        In CMD-1/V5, we directly use the learned distance network d(s,g).

        Args:
            source_state: Source state
            target_node: Target node ID

        Returns:
            Estimated distance (number of steps) to reach target from source
        """
        if self.d_net is None:
            return 0.0

        source_flat = self._flatten_state(source_state)

        # Compute d(s,g) using the MRN
        with torch.no_grad():
            source_tensor = torch.FloatTensor(source_flat).unsqueeze(0).to(self.device)
            target_tensor = torch.tensor([target_node], dtype=torch.long, device=self.device)

            # MRN returns -d(s,g), so negate to get positive distance
            neg_distance = self.d_net(source_tensor, target_tensor).squeeze()
            distance = -neg_distance.item()

        return float(distance)

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        source_states = self.node_to_states[source_node]
        avg = 0
        n = 0
        for source_state in random.sample(source_states, min(100, len(source_states))):
            avg += self.estimate_distance(source_state, target_node)
            n += 1
        return avg / n

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
        torch.save(self.g_encoder.state_dict(), f"{path}/g_encoder.pt")
        torch.save(self.c_net.state_dict(), f"{path}/c_net.pt")
        torch.save(self.d_net.state_dict(), f"{path}/d_net.pt")

        # Save config
        with open(f"{path}/config.pkl", "wb") as f:
            pickle.dump(self.config, f)

        print(f"Distance heuristic V5 saved to {path}")

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

        self.g_encoder = GoalEncoder(
            num_nodes=self.num_nodes,
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        self.c_net = CostNet(num_nodes=self.num_nodes).to(self.device)

        self.d_net = MRN(
            s_encoder=self.s_encoder,
            g_encoder=self.g_encoder,
            sym_dim=64,
            asym_dim=16,
        ).to(self.device)

        # Load weights
        self.s_encoder.load_state_dict(torch.load(f"{path}/s_encoder.pt", map_location=self.device))
        self.s_encoder.eval()

        self.g_encoder.load_state_dict(torch.load(f"{path}/g_encoder.pt", map_location=self.device))
        self.g_encoder.eval()

        self.c_net.load_state_dict(torch.load(f"{path}/c_net.pt", map_location=self.device))
        self.c_net.eval()

        self.d_net.load_state_dict(torch.load(f"{path}/d_net.pt", map_location=self.device))
        self.d_net.eval()

        print(f"Distance heuristic V5 loaded from {path}")
