"""SLAP Approach V2 - Simplified architecture.

Key differences from V1:
- Planning graph built once and reused
- No path-dependent costs
- Shortcuts added after training
- Costs computed by sampling from training states
"""

import itertools
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from relational_structs import GroundAtom, GroundOperator, LiftedOperator, Object
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Skill

from shortcut_learning.configs import ApproachConfig, TrainingConfig
from shortcut_learning.methods.base_approach import (
    ActType,
    ApproachStepResult,
    BaseApproach,
    ObsType,
)
from shortcut_learning.methods.graph_utils import (
    PlanningGraph,
    PlanningGraphEdge,
)
from shortcut_learning.methods.policies.base import Policy, PolicyContext
from shortcut_learning.methods.training_data import ShortcutTrainingData
from shortcut_learning.problems.base_tamp import ImprovisationalTAMPSystem


class SLAPApproachV2(BaseApproach[ObsType, ActType]):
    """SLAP Approach V2 with simplified architecture.

    Pipeline:
    1. Build planning graph once (in first reset)
    2. Collect diverse states per node
    3. Train multi-policy on k shortcuts
    4. Add shortcuts to graph and compute costs
    5. Evaluate using Dijkstra on fixed graph
    """

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        config: ApproachConfig,
        policy: Policy[ObsType, ActType],
    ) -> None:
        super().__init__(system, config)
        self.config = config
        self.policy = policy
        self.name = config.approach_name

        # Planning graph (built once, reused)
        self.planning_graph: PlanningGraph | None = None
        self.graph_built = False
        self.costs_computed = False  # Track if we've computed costs for the graph

        # Shortcuts (populated during training)
        self.trained_shortcuts: list[tuple[int, int]] = []  # List of (source_id, target_id)
        self.shortcuts_added = False

        # Current execution state
        self._current_path: list[PlanningGraphEdge] = []
        self._current_edge: PlanningGraphEdge | None = None
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None
        self._goal_atoms: set[GroundAtom] = set()
        self._goal: set[GroundAtom] = set()
        self.policy_active = False

        # Config
        self._max_skill_steps = config.max_skill_steps
        self.debug_videos = config.debug_videos

        # Get domain for operator groundings
        self.domain = system.get_domain()

    # =========================================================================
    # GRAPH BUILDING METHODS (standalone, no V1 dependency)
    # =========================================================================

    def _build_planning_graph_from_state(
        self,
        objects: set[Object],
        init_atoms: set[GroundAtom],
        stop_at_goal: bool = True,
    ) -> PlanningGraph:
        """Build planning graph from a given initial state using BFS.

        Args:
            objects: Set of objects in the domain
            init_atoms: Initial symbolic state
            stop_at_goal: If True, stop BFS when goal is reached

        Returns:
            PlanningGraph
        """
        graph = PlanningGraph()
        initial_node = graph.add_node(init_atoms)
        visited_states = {frozenset(init_atoms): initial_node}
        queue = deque([(initial_node, 0)])  # Queue for BFS: [(node, depth)]
        node_count = 0
        max_nodes = 1300
        print(f"Building SUBGRAPH with max {max_nodes} nodes (node IDs will change after merge)...")

        # Breadth-first search to build the graph
        while queue and node_count < max_nodes:
            current_node, depth = queue.popleft()
            node_count += 1

            print(f"\n--- Node {node_count-1} at depth {depth} ---")
            print(f"Contains {len(current_node.atoms)} atoms: {current_node.atoms}")

            # Check if this is a goal state, stop search if requested
            if stop_at_goal and self._goal and self._goal.issubset(current_node.atoms):
                queue.clear()
                break

            # Find applicable ground operators using the domain's operators
            applicable_ops = self._find_applicable_operators(
                set(current_node.atoms), objects
            )

            # Apply each applicable operator to generate new states
            for op in applicable_ops:
                # Apply operator effects to get next state
                next_atoms = set(current_node.atoms)
                next_atoms.difference_update(op.delete_effects)
                next_atoms.update(op.add_effects)

                # Check if we've seen this state before
                next_atoms_frozen = frozenset(next_atoms)
                if next_atoms_frozen in visited_states:
                    # Create edge to existing node
                    next_node = visited_states[next_atoms_frozen]
                    graph.add_edge(current_node, next_node, op)
                else:
                    # Create new node and edge
                    next_node = graph.add_node(next_atoms)
                    visited_states[next_atoms_frozen] = next_node
                    graph.add_edge(current_node, next_node, op)
                    queue.append((next_node, depth + 1))

        print(
            f"Planning graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
        )

        # Print detailed graph structure
        print("\nGraph Edges:")
        for edge in graph.edges:
            op_str = f"{edge.operator.name}" if edge.operator else "SHORTCUT"
            print(f"  Node {edge.source.id} --[{op_str}]--> Node {edge.target.id}")
        return graph

    def _find_applicable_operators(
        self, current_atoms: set[GroundAtom], objects: set[Object]
    ) -> list[GroundOperator]:
        """Find all ground operators that are applicable in the current state."""
        applicable_ops = []
        domain_operators = self.domain.operators

        for lifted_op in domain_operators:
            # Find valid groundings using parameter types
            valid_groundings = self._find_valid_groundings(lifted_op, objects)

            for grounding in valid_groundings:
                ground_op = lifted_op.ground(grounding)

                # Check if preconditions are satisfied
                if ground_op.preconditions.issubset(current_atoms):
                    applicable_ops.append(ground_op)

        return applicable_ops

    def _find_valid_groundings(
        self, lifted_op: LiftedOperator, objects: set[Object]
    ) -> list[tuple[Object, ...]]:
        """Find all valid groundings for a lifted operator."""
        # Group objects by type
        objects_by_type: dict[Any, list[Object]] = {}
        for obj in objects:
            if obj.type not in objects_by_type:
                objects_by_type[obj.type] = []
            objects_by_type[obj.type].append(obj)

        # For each parameter, find objects of the right type
        param_objects = []
        for param in lifted_op.parameters:
            if param.type in objects_by_type:
                param_objects.append(objects_by_type[param.type])
            else:
                return []

        # Generate all possible groundings
        groundings = list(itertools.product(*param_objects))

        return groundings

    def _create_planning_graph(
        self, obs: ObsType, info: dict[str, Any], comprehensive: bool = False
    ) -> PlanningGraph:
        """Create planning graph.

        Args:
            obs: Current observation
            info: Environment info
            comprehensive: If True, sample multiple initial states to build a more
                         complete graph covering more of the state space.
        """
        # Get objects and init atoms from perceiver
        objects, atoms, goal = self.system.perceiver.reset(obs, info)
        self._goal = goal

        if not comprehensive:
            # Simple graph from current state (stop at goal)
            return self._build_planning_graph_from_state(objects, atoms, stop_at_goal=True)

        # Build comprehensive graph by sampling multiple initial states
        print("Building comprehensive planning graph for V2...")

        # Build initial graph from current state (don't stop at goal)
        graph = self._build_planning_graph_from_state(objects, atoms, stop_at_goal=False)

        # Try to expand graph by starting from other initial states
        num_samples = 5  # Sample a few different initial states
        for i in range(num_samples):
            print(f"  Sampling initial state {i+1}/{num_samples} for graph expansion...")
            sample_obs, sample_info = self.system.reset()
            sample_objects, sample_atoms, _ = self.system.perceiver.reset(sample_obs, sample_info)

            # Check if this state is already in the graph
            if frozenset(sample_atoms) in graph.node_map:
                print(f"    State already in graph (node {graph.node_map[frozenset(sample_atoms)].id})")
                continue

            # Build a subgraph from this state (don't stop at goal)
            subgraph = self._build_planning_graph_from_state(sample_objects, sample_atoms, stop_at_goal=False)

            # Merge subgraph into main graph
            self._merge_graphs(graph, subgraph)
            print(f"    Merged: graph now has {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        print(f"Final comprehensive graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph

    def _merge_graphs(self, main_graph: PlanningGraph, subgraph: PlanningGraph) -> None:
        """Merge a subgraph into the main graph.

        Adds nodes and edges from subgraph that don't already exist in main_graph.
        """
        # Map from old node to new node for subgraph nodes
        node_mapping = {}
        nodes_added = 0
        edges_added = 0

        # Add new nodes
        for node in subgraph.nodes:
            atoms_frozen = frozenset(node.atoms)
            if atoms_frozen in main_graph.node_map:
                # Node already exists
                existing_node = main_graph.node_map[atoms_frozen]
                node_mapping[node] = existing_node
                if node.id != existing_node.id:
                    print(f"    [MERGE] Subgraph node {node.id} ({len(node.atoms)} atoms) → main graph node {existing_node.id}")
            else:
                # Add new node
                new_node = main_graph.add_node(node.atoms)
                new_node.states = node.states.copy()  # Preserve any collected states
                node_mapping[node] = new_node
                nodes_added += 1
                print(f"    [MERGE] Subgraph node {node.id} ({len(node.atoms)} atoms) → NEW main graph node {new_node.id}")

        # Add new edges
        for edge in subgraph.edges:
            source = node_mapping[edge.source]
            target = node_mapping[edge.target]

            # Check if edge already exists
            existing_edges = main_graph.node_to_outgoing_edges.get(source, [])
            edge_exists = any(
                e.target == target and e.operator == edge.operator and e.is_shortcut == edge.is_shortcut
                for e in existing_edges
            )

            if not edge_exists:
                # Preserve the cost from the original edge
                main_graph.add_edge(source, target, edge.operator, cost=edge.cost, is_shortcut=edge.is_shortcut)
                edges_added += 1

        print(f"    Merge added {nodes_added} nodes and {edges_added} edges")

    def _get_skill(self, operator: GroundOperator) -> Skill:
        """Get skill for operator."""
        skills = [s for s in self.system.skills if s.can_execute(operator)]
        if not skills:
            raise TaskThenMotionPlanningFailure(
                f"No skill found for operator {operator.name}"
            )
        return skills[0]

    def _create_planning_env(self) -> gym.Env:
        """Create a separate environment instance for planning simulations."""
        import copy

        current_env = self.system.env
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
        base_env = current_env

        if hasattr(base_env, "clone"):
            planning_env = base_env.clone()
            print("Created planning environment using custom clone() method.")
            return planning_env
        planning_env = copy.deepcopy(base_env)
        print("No custom clone() found. Created planning environment using deepcopy().")
        return planning_env

    # =========================================================================
    # V2-SPECIFIC METHODS
    # =========================================================================

    def build_planning_graph(self, obs: ObsType, info: dict[str, Any], comprehensive: bool = True) -> None:
        """Build planning graph (call this once before collection/training).

        Separate from reset() to allow building the graph without planning.

        Args:
            obs: Current observation
            info: Environment info
            comprehensive: If True, sample multiple initial states to build a more
                         complete graph. Recommended for training to maximize coverage.
        """
        if not self.graph_built:
            print("Building planning graph (first time)...")
            self.planning_graph = self._create_planning_graph(obs, info, comprehensive=comprehensive)
            self.graph_built = True
            print(f"Planning graph built with {len(self.planning_graph.nodes)} nodes")

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult:
        """Reset for a new episode.

        V2 behavior:
        - Ensure planning graph is built (comprehensive at first)
        - Check if current initial state is in graph
        - If not, expand graph by merging in a new subgraph from this state
        - Find initial node, run Dijkstra, execute

        This adaptive strategy ensures:
        - Graph eventually covers full state space
        - Every episode is guaranteed to succeed (initial state always in graph)
        - Graph reuse is maximized (only expand when needed)
        """
        # Get current atoms and goal
        atoms = self.system.perceiver.step(obs)
        _, _, goal_atoms = self.system.perceiver.reset(obs, info)

        # Build comprehensive planning graph if not already built
        if not self.graph_built:
            self.build_planning_graph(obs, info, comprehensive=True)

        assert self.planning_graph is not None

        # Check if current initial state is in the graph
        atoms_frozen = frozenset(atoms)
        goal_atoms_frozen = frozenset(goal_atoms)

        # Check if we need to expand the graph
        initial_in_graph = atoms_frozen in self.planning_graph.node_map
        goal_in_graph = goal_atoms_frozen in self.planning_graph.node_map

        if not initial_in_graph or not goal_in_graph:
            missing = []
            if not initial_in_graph:
                missing.append("initial state")
            if not goal_in_graph:
                missing.append("goal state")

            print(f"Expanding planning graph: {' and '.join(missing)} not found")
            print(f"  Graph before: {len(self.planning_graph.nodes)} nodes, {len(self.planning_graph.edges)} edges")

            # Build a subgraph from this initial state
            # IMPORTANT: Don't stop at goal - explore fully to ensure connectivity
            objects, atoms_from_perceiver, _ = self.system.perceiver.reset(obs, info)
            subgraph = self._build_planning_graph_from_state(objects, atoms_from_perceiver, stop_at_goal=False)

            # Merge the subgraph
            self._merge_graphs(self.planning_graph, subgraph)

            print(f"  Graph after: {len(self.planning_graph.nodes)} nodes, {len(self.planning_graph.edges)} edges")

            # Re-add shortcuts to any newly added edges if we have trained shortcuts
            if self.shortcuts_added and self.trained_shortcuts:
                self._try_add_shortcuts_to_new_nodes()

            # Compute costs for new edges
            print("Computing costs for new edges...")
            self._compute_edge_costs_simple()

        # If this is the first reset and costs haven't been computed, compute them
        if not self.costs_computed:
            print("Computing costs for all edges (first time)...")
            self._compute_edge_costs_simple()
            self.costs_computed = True

        # Find shortest path (using main graph node IDs)
        initial_node = self.planning_graph.node_map.get(atoms_frozen)
        goal_node = self.planning_graph.node_map.get(goal_atoms_frozen)

        if initial_node:
            print(f"Initial state: node {initial_node.id} with {len(initial_node.atoms)} atoms")
        if goal_node:
            print(f"Goal state: exact match found at node {goal_node.id} with {len(goal_node.atoms)} atoms")
        else:
            # No exact match, but Dijkstra will find any node containing goal as subset
            goal_nodes_subset = [n for n in self.planning_graph.nodes if goal_atoms.issubset(n.atoms)]
            if goal_nodes_subset:
                print(f"Goal state: no exact match in graph (looking for {len(goal_atoms_frozen)} atoms)")
                print(f"  Will accept any of {len(goal_nodes_subset)} nodes containing goal as subset")
            else:
                print(f"Goal state: NOT IN GRAPH (has {len(goal_atoms_frozen)} atoms, no nodes contain these atoms)")

        try:
            self._current_path = self.planning_graph.find_shortest_path(atoms, goal_atoms)
        except AssertionError as e:
            # Print diagnostic info
            print(f"\nERROR: {e}")
            print(f"Initial atoms: {atoms}")
            print(f"Goal atoms: {goal_atoms}")
            print(f"\nGraph structure:")
            for node in self.planning_graph.nodes:
                out_edges = self.planning_graph.node_to_outgoing_edges.get(node, [])
                print(f"  Node {node.id}: {len(node.atoms)} atoms, {len(out_edges)} outgoing edges")
            raise

        if not self._current_path:
            raise TaskThenMotionPlanningFailure("No path found in planning graph")

        # Reset execution state
        self._current_edge = None
        self._current_operator = None
        self._current_skill = None
        self.policy_active = False

        # Return first action
        return self.step(obs, 0.0, False, False, info)

    def _compute_edge_costs_simple(self) -> None:
        """Compute costs for edges that don't have costs yet.

        For regular edges: Execute operator skill from source to target, measure steps.
        For shortcut edges: Skip (they get costs during training).
        """
        assert self.planning_graph is not None

        # Find all edges that need costs
        edges_to_compute = [e for e in self.planning_graph.edges if e.cost == float('inf') and not e.is_shortcut]

        if not edges_to_compute:
            print("  No new edges need cost computation")
            return

        print(f"  Computing costs for {len(edges_to_compute)} regular edges...")

        raw_env = self._create_planning_env()

        for edge in edges_to_compute:
            # For regular edges, measure cost by executing the operator's skill
            cost = self._measure_regular_edge_cost(raw_env, edge)
            edge.cost = cost

            if cost < float('inf'):
                print(f"    Edge {edge.source.id}→{edge.target.id} ({edge.operator.name if edge.operator else 'unknown'}): {cost:.1f} steps")
            else:
                print(f"    Edge {edge.source.id}→{edge.target.id} ({edge.operator.name if edge.operator else 'unknown'}): FAILED")

    def _measure_regular_edge_cost(
        self,
        env: gym.Env,
        edge: PlanningGraphEdge,
    ) -> float:
        """Measure cost of a regular (non-shortcut) edge by executing its operator.

        Returns number of steps, or inf if failed.
        """
        # We need a state that matches the source node's atoms
        # If the source node has stored states (from collection), use one
        # Otherwise, we need to find/create a matching state

        source_states = edge.source.states if hasattr(edge.source, 'states') else []

        if not source_states:
            # No stored states - we need to sample or search for a matching state
            # Try to find one by resetting the environment multiple times
            for _ in range(10):  # Try up to 10 times
                sample_obs, sample_info = self.system.reset()
                _, sample_atoms, _ = self.system.perceiver.reset(sample_obs, sample_info)

                if frozenset(sample_atoms) == frozenset(edge.source.atoms):
                    # Found a matching state!
                    source_states = [sample_obs]
                    break

            if not source_states:
                # Couldn't find a matching state, use default
                print(f"      Warning: No state found for node {edge.source.id}, using default cost")
                return 1.0

        # Group measurements by incoming edge for path-dependent costs
        # Use 2x max_skill_steps as the penalty for failures
        failure_penalty = 2 * self._max_skill_steps
        measurements_by_prev_edge = {}  # prev_edge_id -> list of costs
        all_measurements = []
        num_failures = 0

        for sample_idx in range(len(source_states)):
            initial_state = source_states[sample_idx]

            # Get which edge led to this state
            prev_edge_id = edge.source.state_incoming_edges[sample_idx] if sample_idx < len(edge.source.state_incoming_edges) else None

            # Reset environment to the source state
            env.reset_from_state(initial_state)

            # Get the skill for this operator
            assert edge.operator is not None, "Regular edge must have an operator"
            skill = self._get_skill(edge.operator)
            skill.reset(edge.operator)

            # Execute skill and measure steps
            obs = initial_state
            target_atoms = set(edge.target.atoms)
            succeeded = False

            for step in range(self._max_skill_steps):
                action = skill.get_action(obs)
                if action is None:
                    break  # Skill failed for this sample, try next

                obs, _, term, trunc, _ = env.step(action)
                atoms = self.system.perceiver.step(obs)

                # Check if reached target node
                if set(atoms) == target_atoms:
                    cost = float(step + 1)
                    all_measurements.append(cost)
                    succeeded = True

                    # Group by incoming edge
                    if prev_edge_id not in measurements_by_prev_edge:
                        measurements_by_prev_edge[prev_edge_id] = []
                    measurements_by_prev_edge[prev_edge_id].append(cost)
                    break

                if term or trunc:
                    break

            # If failed, count it as failure_penalty
            if not succeeded:
                num_failures += 1
                all_measurements.append(failure_penalty)
                if prev_edge_id not in measurements_by_prev_edge:
                    measurements_by_prev_edge[prev_edge_id] = []
                measurements_by_prev_edge[prev_edge_id].append(failure_penalty)

        # Store path-dependent costs
        for prev_edge_id, costs in measurements_by_prev_edge.items():
            if prev_edge_id is not None:
                edge.path_costs[(prev_edge_id,)] = sum(costs) / len(costs)

        # Return average of all measurements including failures
        if all_measurements:
            avg_cost = sum(all_measurements) / len(all_measurements)
            num_successes = len(all_measurements) - num_failures
            if len(all_measurements) > 1:
                individual_costs = ", ".join(f"{c:.1f}" for c in all_measurements)
                print(f"      Averaged {len(all_measurements)} measurements: [{individual_costs}] → {avg_cost:.1f}")
                if num_failures > 0:
                    success_rate = num_successes / len(all_measurements)
                    print(f"      Success rate: {num_successes}/{len(all_measurements)} = {success_rate:.1%}")
                if len(measurements_by_prev_edge) > 1:
                    print(f"      Path-dependent costs: {len(measurements_by_prev_edge)} different incoming edges")
            return avg_cost
        return float('inf')

    def _try_add_shortcuts_to_new_nodes(self) -> None:
        """Try to add trained shortcuts to newly added nodes in the graph.

        When we expand the graph with new nodes, we may be able to add
        shortcuts between existing nodes with trained shortcuts and new nodes,
        or between pairs of new nodes.
        """
        if not self.trained_shortcuts:
            return

        print("  Checking for new shortcut opportunities in expanded graph...")
        shortcuts_added = 0

        for shortcut_id, (source_id, target_id) in enumerate(self.trained_shortcuts):
            # Find source and target nodes by id
            source_node = next((n for n in self.planning_graph.nodes if n.id == source_id), None)
            target_node = next((n for n in self.planning_graph.nodes if n.id == target_id), None)

            if not source_node or not target_node:
                continue

            # Check if shortcut edge already exists
            existing_edges = self.planning_graph.node_to_outgoing_edges.get(source_node, [])
            shortcut_exists = any(
                e.target == target_node and e.is_shortcut and e.shortcut_id == shortcut_id
                for e in existing_edges
            )

            if not shortcut_exists:
                edge = self.planning_graph.add_edge(
                    source_node,
                    target_node,
                    operator=None,
                    is_shortcut=True,
                )
                edge.shortcut_id = shortcut_id
                # Use a reasonable default cost (could be refined later)
                edge.cost = 10.0
                shortcuts_added += 1

        if shortcuts_added > 0:
            print(f"  Added {shortcuts_added} new shortcut edges")

    def train(
        self,
        training_data: ShortcutTrainingData,
        train_config: TrainingConfig | None = None,
    ) -> None:
        """Train the multi-policy and add shortcuts to graph.

        V2 behavior:
        1. Train multi-policy on k shortcuts
        2. Add k shortcut edges to planning graph
        3. Compute costs by sampling from training states
        """
        print("\n" + "="*60)
        print("TRAINING V2")
        print("="*60)

        assert self.planning_graph is not None, "Must build planning graph first"
        assert not self.shortcuts_added, "Shortcuts already added!"

        # Extract shortcut info
        k = len(training_data.shortcuts)
        print(f"\nTraining multi-policy on {k} shortcuts...")

        # Use provided config or create default
        if train_config is None:
            train_config = TrainingConfig(
                max_env_steps=100,
                runs_per_shortcut=100,
                training_record_interval=100,
            )

        # Train policy on all shortcuts using V2 wrapper
        self._train_policy_v2(training_data, train_config)

        # Store trained shortcuts
        self.trained_shortcuts = [
            (source.id, target.id) for source, target in training_data.shortcuts
        ]

        # Add shortcut edges to graph
        print(f"\nAdding {k} shortcut edges to planning graph...")
        for shortcut_id, (source_node, target_node) in enumerate(training_data.shortcuts):
            edge = self.planning_graph.add_edge(
                source_node,
                target_node,
                operator=None,
                is_shortcut=True,
            )
            edge.shortcut_id = shortcut_id
            print(f"  Added shortcut {shortcut_id}: node {source_node.id} → node {target_node.id}")

        self.shortcuts_added = True

        # Compute costs for shortcuts
        print(f"\nComputing costs for {k} shortcuts...")
        self._compute_shortcut_costs(training_data)

        # Print summary of shortcuts for reference
        print(f"\n=== SHORTCUT SUMMARY (Main Graph Node IDs) ===")
        for shortcut_id, (source, target) in enumerate(training_data.shortcuts):
            edge = next((e for e in self.planning_graph.edges if e.is_shortcut and e.shortcut_id == shortcut_id), None)
            if edge:
                cost_str = f"{edge.cost:.1f} steps" if edge.cost < float('inf') else "FAILED"
                print(f"  Shortcut {shortcut_id}: node {source.id} → node {target.id} [{cost_str}]")

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60 + "\n")

    def _train_policy_v2(
        self,
        training_data: ShortcutTrainingData,
        train_config: TrainingConfig,
    ) -> None:
        """Train the multi-policy on all shortcuts.

        MultiRLPolicyV2 handles creating wrappers per shortcut internally.
        We just need to provide the base environment and training config.
        """
        # Create base training environment
        raw_env = self._create_planning_env()

        # Attach perceiver to environment for multi_rl_ppo_v2
        # The policy needs it to create SLAP wrappers
        if not hasattr(raw_env, 'perceiver'):
            raw_env.perceiver = self.system.perceiver

        # Train the multi-policy
        # MultiRLPolicyV2 will create one policy per shortcut
        if hasattr(self.policy, 'train'):
            print(f"Training multi-policy on {len(training_data)} shortcuts...")
            self.policy.train(
                env=raw_env,
                train_config=train_config,
                train_data=training_data,
            )
        else:
            print("Warning: Policy does not have a train() method")

    def _compute_shortcut_costs(
        self, training_data: ShortcutTrainingData
    ) -> None:
        """Compute costs for shortcuts by averaging over all training states.

        For each shortcut, use all states from source_node.states,
        execute the policy, measure steps, and average.
        """
        print("Computing shortcut costs from all available states...")

        raw_env = self._create_planning_env()

        for shortcut_id, (source_node, target_node) in enumerate(training_data.shortcuts):
            # Find the edge
            edge = None
            for e in self.planning_graph.edges:
                if e.source == source_node and e.target == target_node and e.is_shortcut:
                    edge = e
                    break

            if edge is None:
                print(f"  Warning: Could not find edge for shortcut {shortcut_id}")
                continue

            # Use all available states to measure cost
            source_states = source_node.states
            if not source_states:
                print(f"  Warning: No states for node {source_node.id}")
                edge.cost = float('inf')
                continue

            # Measure cost for each state, treating failures as high cost
            # Use 2x max_skill_steps as the penalty for failures
            failure_penalty = 2 * self._max_skill_steps
            costs = []
            for state in source_states:
                cost = self._measure_shortcut_cost(
                    raw_env, state, source_node, target_node, shortcut_id
                )
                # If failed (inf), use failure penalty instead
                if cost < float('inf'):
                    costs.append(cost)
                else:
                    costs.append(failure_penalty)

            # Average cost including failures
            if costs:
                avg_cost = np.mean(costs)
                num_successes = sum(1 for c in costs if c < failure_penalty)
                success_rate = num_successes / len(costs)
                edge.cost = avg_cost
                print(f"  Shortcut {shortcut_id} ({source_node.id}→{target_node.id}): {avg_cost:.1f} steps (success: {num_successes}/{len(source_states)} = {success_rate:.1%})")
            else:
                edge.cost = float('inf')
                print(f"  Shortcut {shortcut_id} ({source_node.id}→{target_node.id}): FAILED (no states)")

    def _measure_shortcut_cost(
        self,
        env: gym.Env,
        initial_state: Any,
        source_node: Any,
        target_node: Any,
        shortcut_id: int,
    ) -> float:
        """Measure cost of a shortcut from a given initial state.

        Returns number of steps, or inf if failed.
        """
        # Reset environment to initial state
        env.reset_from_state(initial_state)

        # Configure policy
        source_atoms = set(source_node.atoms)
        target_atoms = set(target_node.atoms)

        self.policy.configure_context(
            PolicyContext(
                goal_atoms=target_atoms,
                current_atoms=source_atoms,
                info={
                    "shortcut_id": shortcut_id,
                    "source_node_id": source_node.id,
                    "target_node_id": target_node.id,
                },
            )
        )

        # Execute policy
        obs = initial_state

        # Debug logging for specific shortcuts
        debug_shortcut = (source_node.id == 0 and target_node.id == 7) or (source_node.id == 2 and target_node.id == 4)
        if debug_shortcut:
            print(f"\n  [DEBUG] Measuring shortcut {source_node.id}→{target_node.id}")
            print(f"  [DEBUG] Source atoms ({len(source_atoms)}): {sorted([str(a) for a in source_atoms])}")
            print(f"  [DEBUG] Target atoms ({len(target_atoms)}): {sorted([str(a) for a in target_atoms])}")

        for step in range(self._max_skill_steps):
            action = self.policy.get_action(obs)
            if action is None:
                return float('inf')

            obs, _, term, trunc, _ = env.step(action)
            atoms = self.system.perceiver.step(obs)

            if debug_shortcut:
                print(f"  [DEBUG] Step {step + 1}: perceived {len(atoms)} atoms")
                # Show which target atoms are satisfied
                satisfied = target_atoms.intersection(atoms)
                missing = target_atoms - atoms
                extra = atoms - target_atoms
                print(f"  [DEBUG]   Satisfied: {len(satisfied)}/{len(target_atoms)}")
                if missing:
                    print(f"  [DEBUG]   Missing: {sorted([str(a) for a in missing])}")
                if len(extra) < 5:  # Only show if not too many
                    print(f"  [DEBUG]   Extra: {sorted([str(a) for a in extra])}")

            # Check if reached target (must match exactly)
            if target_atoms == atoms:
                if debug_shortcut:
                    print(f"  [DEBUG] SUCCESS at step {step + 1}!")
                return float(step + 1)

            if term or trunc:
                return float('inf')

        return float('inf')  # Timeout

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Execute one step."""
        atoms = self.system.perceiver.step(obs)

        # Get next edge if needed
        if not self._current_edge and self._current_path:
            self._current_edge = self._current_path.pop(0)

            if self._current_edge.is_shortcut:
                print(f"Using shortcut edge {self._current_edge.shortcut_id}")
                self.policy_active = True

                # Get goal atoms
                target_node = self._current_edge.target
                self._goal_atoms = set(target_node.atoms)

                # Configure policy
                self.policy.configure_context(
                    PolicyContext(
                        goal_atoms=self._goal_atoms,
                        current_atoms=atoms,
                        info={
                            "shortcut_id": self._current_edge.shortcut_id,
                            "source_node_id": self._current_edge.source.id,
                            "target_node_id": target_node.id,
                        },
                    )
                )

                return ApproachStepResult(action=self.policy.get_action(obs))

            # Regular edge - use operator skill
            self._current_operator = self._current_edge.operator

            if not self._current_operator:
                raise TaskThenMotionPlanningFailure("Edge has no operator")

            self._current_skill = self._get_skill(self._current_operator)
            self._current_skill.reset(self._current_operator)

        # Check if current edge's target state is achieved (exact match required)
        if self._current_edge and set(self._current_edge.target.atoms) == atoms:
            print("Edge target achieved")
            self._current_edge = None
            self.policy_active = False
            return self.step(obs, reward, terminated, truncated, info)

        # Execute current skill (for regular edges) or policy (for shortcuts)
        if self.policy_active:
            # Already handled above
            return ApproachStepResult(action=self.policy.get_action(obs))

        if not self._current_skill:
            raise TaskThenMotionPlanningFailure("No current skill")

        try:
            action = self._current_skill.get_action(obs)
            if action is None:
                print(f"No action returned by skill {self._current_skill}")
        except AssertionError as e:
            print(f"Assertion error in skill {self._current_skill}: {e}")
            action = None

        return ApproachStepResult(action=action)
