"""Pure RL baseline approach without using TAMP structure."""

import copy
import itertools
import os
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as iio
import numpy as np
from gymnasium.wrappers import RecordVideo
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    PDDLProblem,
)
from relational_structs.utils import parse_pddl_plan
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Skill
from tomsutils.pddl_planning import run_pddl_planner
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
    PlanningGraphNode,
)
from shortcut_learning.methods.policies.base import Policy, PolicyContext
from shortcut_learning.methods.training_data import ShortcutSignature, TrainingData
from shortcut_learning.methods.wrappers import (
    ContextAwareWrapper,
    GoalConditionedWrapper,
    SLAPWrapper,
)
from shortcut_learning.problems.base_tamp import ImprovisationalTAMPSystem


class SLAPApproach(BaseApproach[ObsType, ActType]):
    """General improvisational TAMP approach.

    This approach combines task-and-motion planning with learned
    policies for creating shortcuts between non-adjacent nodes in the
    plan.
    """

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        config: ApproachConfig,
        policy: Policy[ObsType, ActType],
    ) -> None:
        """Initialize approach."""
        super().__init__(system, config)
        self.debug_videos = config.debug_videos
        self.policy = policy
        self.planner_id = config.planner_id
        self._max_skill_steps = config.max_skill_steps
        self.use_context_wrapper = config.use_context_wrapper
        self.context_env = None
        if self.use_context_wrapper:
            if not isinstance(system.env, ContextAwareWrapper):
                self.context_env = ContextAwareWrapper(
                    system.env,
                    system.perceiver,
                    max_atom_size=config.max_atom_size,
                )
            else:
                self.context_env = system.env

            # Initialize policy with (context-aware) wrapped environment
            policy.initialize(self.context_env)
        else:
            policy.initialize(system.env)

        # Get domain
        self.domain = system.get_domain()

        # Initialize planning state
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None
        self._goal: set[GroundAtom] = set()
        self._custom_goal: set[GroundAtom] = set()

        # Graph-based planning state
        self.planning_graph: PlanningGraph | None = None
        self._current_path: list[PlanningGraphEdge] = []
        self._current_edge: PlanningGraphEdge | None = None
        self._goal_atoms: set[GroundAtom] = set()
        self.policy_active = False
        self.observed_states: dict[int, list[ObsType]] = {}

        # Shortcut signatures for similarity matching
        self.trained_signatures: list[ShortcutSignature] = []

        self.rng = np.random.default_rng(config.seed)

    def reset(
        self,
        obs: ObsType,
        info: dict[str, Any],
        select_random_goal: bool = False,
    ) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        objects, atoms, goal = self.system.perceiver.reset(obs, info)
        self._goal = goal
        self.observed_states = {}

        # Create planning graph
        self.planning_graph = self._create_planning_graph(objects, atoms)

        # If random goal selection, pick a random node as goal
        if select_random_goal:
            initial_node = self.planning_graph.node_map[frozenset(atoms)]
            higher_nodes = [
                n for n in self.planning_graph.nodes if n.id > initial_node.id
            ]
            random_index = self.rng.integers(0, len(higher_nodes))
            goal_node = higher_nodes[random_index]
            goal = set(goal_node.atoms)
            print(
                f"Selected random goal from node {goal_node.id} with atoms: {goal_node.atoms}"  # pylint: disable=line-too-long
            )
            self._custom_goal = goal

        # Store initial state in observed states
        initial_node = self.planning_graph.node_map[frozenset(atoms)]
        self.observed_states[initial_node.id] = []
        self.observed_states[initial_node.id].append(obs)

        # Try to add shortcuts
        if not self.training_mode:
            self._try_add_shortcuts(self.planning_graph)

        # Compute edge costs
        self._compute_planning_graph_edge_costs(obs, info)

        # Find shortest path
        self._current_path = self.planning_graph.find_shortest_path(atoms, goal)

        # Reset state
        self._current_operator = None
        self._current_skill = None
        self._current_edge = None
        self._goal_atoms = set()
        self.policy_active = False

        return self.step(obs, 0.0, False, False, info)

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Step approach with new observation."""
        atoms = self.system.perceiver.step(obs)
        using_goal_env, goal_env = self._using_goal_env(self.system.env)
        target_atom_vector = None
        current_atom_vector = None

        # Check if the custom goal (if any) has been achieved
        if self._custom_goal and self._custom_goal.issubset(atoms):
            print("Custom goal achieved!")
            # Return a no-op action to indicate success
            zero_action = np.zeros_like(self.system.env.action_space.sample())
            return ApproachStepResult(action=zero_action, terminate=True)

        # Check if policy achieved its goal
        if self.policy_active and self.planning_graph:
            current_node = self._current_edge.source if self._current_edge else None
            if current_node and self._goal_atoms == atoms:
                print("Policy successfully achieved goal atoms!")
                self._current_edge = None
                self.policy_active = False
                self._goal_atoms = set()
                return self.step(obs, reward, terminated, truncated, info)

            if using_goal_env and goal_env is not None:
                assert hasattr(
                    self.policy, "node_states"
                ), "Policy must have node_states"
                if self._current_edge and self._current_edge.target:
                    target_node_id = self._current_edge.target.id
                    target_atoms = set(self._current_edge.target.atoms)
                    if goal_env.use_atom_as_obs is True:
                        target_atom_vector = goal_env.create_atom_vector(target_atoms)
                        current_atom_vector = goal_env.create_atom_vector(atoms)
                    dict_obs = {
                        "observation": obs,
                        "achieved_goal": (
                            obs
                            if goal_env.use_atom_as_obs is False
                            else current_atom_vector
                        ),
                        "desired_goal": (
                            self.policy.node_states[target_node_id]
                            if goal_env.use_atom_as_obs is False
                            else target_atom_vector
                        ),
                    }
                    # type: ignore[arg-type] # pylint: disable=line-too-long
                    return ApproachStepResult(action=self.policy.get_action(dict_obs))
            elif self.use_context_wrapper and self.context_env is not None:
                self.context_env.set_context(atoms, self._goal_atoms)
                aug_obs = self.context_env.augment_observation(obs)
                return ApproachStepResult(action=self.policy.get_action(aug_obs))
            return ApproachStepResult(action=self.policy.get_action(obs))

        # Get next edge if needed
        assert self.planning_graph is not None
        if not self._current_edge and self._current_path:
            self._current_edge = self._current_path.pop(0)

            if self._current_edge.is_shortcut:
                print("Using shortcut edge")
                self.policy_active = True

                # Get goal nodes for the target node
                target_node = self._current_edge.target
                self._goal_atoms = set(target_node.atoms)

                # Configure policy and context wrapper
                self.policy.configure_context(
                    PolicyContext(
                        goal_atoms=self._goal_atoms,
                        current_atoms=atoms,
                        info={
                            "source_node_id": self._current_edge.source.id,
                            "target_node_id": target_node.id,
                        },
                    )
                )
                if using_goal_env and goal_env is not None:
                    target_state = self.policy.node_states[target_node.id]
                    target_atoms = set(self._current_edge.target.atoms)
                    if goal_env.use_atom_as_obs is True:
                        target_atom_vector = goal_env.create_atom_vector(target_atoms)
                        current_atom_vector = goal_env.create_atom_vector(atoms)
                    dict_obs = {
                        "observation": obs,
                        "achieved_goal": (
                            obs
                            if goal_env.use_atom_as_obs is False
                            else current_atom_vector
                        ),
                        "desired_goal": (
                            target_state
                            if goal_env.use_atom_as_obs is False
                            else target_atom_vector
                        ),
                    }
                    # type: ignore[arg-type] # pylint: disable=line-too-long
                    return ApproachStepResult(action=self.policy.get_action(dict_obs))
                if self.use_context_wrapper and self.context_env is not None:
                    self.context_env.set_context(atoms, self._goal_atoms)
                    aug_obs = self.context_env.augment_observation(obs)
                    return ApproachStepResult(action=self.policy.get_action(aug_obs))
                return ApproachStepResult(action=self.policy.get_action(obs))

            # Regular edge - use operator skill
            self._current_operator = self._current_edge.operator

            if not self._current_operator:
                raise TaskThenMotionPlanningFailure("Edge has no operator")

            # Get skill for the operator
            self._current_skill = self._get_skill(self._current_operator)
            self._current_skill.reset(self._current_operator)

        # Check if current edge's target state is achieved
        if self._current_edge and set(self._current_edge.target.atoms) == atoms:
            print("Edge target achieved")
            self._current_edge = None
            return self.step(obs, reward, terminated, truncated, info)

        # Execute current skill
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

    def _create_task_plan(
        self,
        objects: set[Object],
        init_atoms: set[GroundAtom],
        goal: set[GroundAtom],
    ) -> list[GroundOperator]:
        """Create task plan to achieve goal."""
        problem = PDDLProblem(
            self.domain.name, self.domain.name, objects, init_atoms, goal
        )
        plan_str = run_pddl_planner(
            str(self.domain), str(problem), planner=self.planner_id
        )
        if plan_str is None:
            raise TaskThenMotionPlanningFailure("No plan found")
        return parse_pddl_plan(plan_str, self.domain, problem)

    def _get_skill(self, operator: GroundOperator) -> Skill:
        """Get skill that can execute operator."""
        skills = [s for s in self.system.skills if s.can_execute(operator)]
        if not skills:
            raise TaskThenMotionPlanningFailure(
                f"No skill found for operator {operator.name}"
            )
        return skills[0]

    def _create_planning_graph(
        self,
        objects: set[Object],
        init_atoms: set[GroundAtom],
    ) -> PlanningGraph:
        """Create a tree-based planning graph by exploring possible action
        sequences.

        This builds a graph representing multiple possible plans rather
        than just following a single task plan. This method is domain-
        agnostic.
        """
        graph = PlanningGraph()
        initial_node = graph.add_node(init_atoms)
        visited_states = {frozenset(init_atoms): initial_node}
        queue = deque([(initial_node, 0)])  # Queue for BFS: [(node, depth)]
        node_count = 0
        max_nodes = 1300
        print(f"Building planning graph with max {max_nodes} nodes...")

        # Breadth-first search to build the graph
        while queue and node_count < max_nodes:
            current_node, depth = queue.popleft()
            node_count += 1

            print(f"\n--- Node {node_count-1} at depth {depth} ---")
            print(f"Contains {len(current_node.atoms)} atoms: {current_node.atoms}")

            # Check if this is a goal state, stop search if so
            # NOTE: we use the same assumption as PDDL to find the goal nodes of the
            # shortest sequences of symbolic actions
            if self._goal and self._goal.issubset(current_node.atoms):
                queue.clear()
                break
                # continue

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
                    # DEBUG: To get the desired node id in the graph by looking at op
                    # Instead of Atoms
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
        """Find all ground operators that are applicable in the current
        state."""
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
        """Find all valid groundings for a lifted operator.

        Args:
            lifted_op: The lifted operator to ground
            objects: Available objects in the domain

        Returns:
            List of valid parameter tuples for grounding
        """
        # Group objects by type
        objects_by_type: dict[Any, list[Object]] = {}
        for obj in objects:
            if obj.type not in objects_by_type:
                objects_by_type[obj.type] = []
            objects_by_type[obj.type].append(obj)

        # Print the parameter requirements for debugging
        param_types = []
        for param in lifted_op.parameters:
            param_types.append(f"{param.name} ({param.type.name})")

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

    def _try_add_shortcuts(self, graph: PlanningGraph) -> None:
        """Try to add shortcut edges to the graph."""
        using_goal_env, _ = self._using_goal_env(self.system.env)
        for source_node in graph.nodes:
            for target_node in graph.nodes:
                # Skip same node and existing edges
                if source_node == target_node:
                    continue
                if any(
                    edge.target == target_node
                    for edge in graph.node_to_outgoing_edges.get(source_node, [])
                ):
                    continue
                if target_node.id <= source_node.id:
                    continue

                source_atoms = set(source_node.atoms)
                target_atoms = set(target_node.atoms)
                if not target_atoms:
                    continue

                # Check if this is similar to a trained shortcut
                if self.trained_signatures and not using_goal_env:
                    current_sig = ShortcutSignature.from_context(
                        source_atoms, target_atoms
                    )
                    can_handle = False
                    best_similarity = 0.0

                    for trained_sig in self.trained_signatures:
                        similarity = current_sig.similarity(trained_sig)
                        if similarity > 0.99:  # Threshold to be tuned
                            can_handle = True
                            best_similarity = max(best_similarity, similarity)

                    if not can_handle:
                        continue

                # Configure context for environment and policy
                if self.use_context_wrapper and self.context_env is not None:
                    self.context_env.set_context(source_atoms, target_atoms)

                self.policy.configure_context(
                    PolicyContext(
                        goal_atoms=target_atoms,
                        current_atoms=source_atoms,
                        info={
                            "source_node_id": source_node.id,
                            "target_node_id": target_node.id,
                        },
                    )
                )
                if self.policy.can_initiate():
                    print(
                        f"Trying to add shortcut: {source_node.id} to {target_node.id}"
                    )
                    graph.add_edge(source_node, target_node, None, is_shortcut=True)

    def _compute_planning_graph_edge_costs(
        self,
        obs: ObsType,
        info: dict[str, Any],
    ) -> None:
        """Compute edge costs considering the path taken to reach each node.

        Explores all potential paths through the graph to find the
        optimal cost for each edge based on the path history.
        """
        assert self.planning_graph is not None

        debug = self.debug_videos

        if debug:
            edge_videos_dir = "videos/edge_computation_videos"
            os.makedirs(edge_videos_dir, exist_ok=True)

        output_dir = "videos/debug_frames"
        os.makedirs(output_dir, exist_ok=True)

        _, init_atoms, _ = self.system.perceiver.reset(obs, info)
        initial_node = self.planning_graph.node_map[frozenset(init_atoms)]

        # Map from (path, source_node, target_node) to observation and info
        path_states: dict[
            tuple[tuple[int, ...], PlanningGraphNode, PlanningGraphNode],
            tuple[ObsType, dict],
        ] = {}
        empty_path: tuple[int, ...] = tuple()
        path_states[(empty_path, initial_node, initial_node)] = (obs, info)

        raw_env = self._create_planning_env()
        using_goal_env, goal_env = self._using_goal_env(self.system.env)

        # BFS to explore all paths
        queue = [(initial_node, empty_path)]  # (node, path)
        explored_segments = set()
        while queue:
            node, path = queue.pop(0)
            if (path, node) in explored_segments:
                continue
            explored_segments.add((path, node))

            if (path, node, node) not in path_states:
                print(f"Warning: State not found for path {path} to node {node.id}")
                continue

            path_state, path_info = path_states[(path, node, node)]

            # Try each outgoing edge from this node
            for edge in self.planning_graph.node_to_outgoing_edges.get(node, []):
                if (path, node, edge.target) in path_states:
                    continue
                if edge.target.id <= node.id:
                    continue

                frames: list[Any] = []
                video_filename = ""
                if debug:
                    edge_type = "shortcut" if edge.is_shortcut else "regular"
                    path_str = (
                        "-".join(str(node_id) for node_id in path) if path else "start"
                    )
                    video_filename = f"{edge_videos_dir}/edge_{node.id}_to_{edge.target.id}_{edge_type}_via_{path_str}.mp4"  # pylint: disable=line-too-long

                raw_env.reset_from_state(path_state)  # type: ignore

                if debug and hasattr(raw_env, "render") and not self.training_mode:
                    try:
                        frames.append(raw_env.render())
                    except Exception as e: # pylint: disable=broad-except
                        print(f"Error rendering initial frame: {e}")

                _, init_atoms, _ = self.system.perceiver.reset(path_state, path_info)
                goal_atoms = set(edge.target.atoms)

                if edge.is_shortcut:
                    self.policy.configure_context(
                        PolicyContext(
                            goal_atoms=goal_atoms,
                            current_atoms=init_atoms,
                            info={
                                "source_node_id": node.id,
                                "target_node_id": edge.target.id,
                            },
                        )
                    )
                    if using_goal_env and goal_env is not None:
                        assert hasattr(
                            self.policy, "node_states"
                        ), "Policy must have node_states"
                        target_state = self.policy.node_states[edge.target.id]
                        target_atoms = set(edge.target.atoms)
                        if goal_env.use_atom_as_obs is True:
                            target_atom_vector = goal_env.create_atom_vector(
                                target_atoms
                            )
                            current_atom_vector = goal_env.create_atom_vector(
                                init_atoms
                            )
                        aug_obs = {
                            "observation": path_state,
                            "achieved_goal": (
                                path_state
                                if goal_env.use_atom_as_obs is False
                                else current_atom_vector
                            ),
                            "desired_goal": (
                                target_state
                                if goal_env.use_atom_as_obs is False
                                else target_atom_vector
                            ),
                        }
                    elif self.use_context_wrapper and self.context_env is not None:
                        self.context_env.set_context(init_atoms, goal_atoms)
                        # type: ignore[arg-type] # pylint: disable=line-too-long
                        aug_obs = self.context_env.augment_observation(path_state)
                    else:
                        aug_obs = path_state  # type: ignore[assignment]
                    skill: Policy | Skill = self.policy
                else:
                    assert edge.operator is not None
                    skill = self._get_skill(edge.operator)
                    skill.reset(edge.operator)
                    aug_obs = path_state  # type: ignore[assignment]

                # Execute the skill and track steps
                num_steps = 0
                curr_raw_obs = path_state
                curr_aug_obs = aug_obs
                frame_counter = 0
                # frame = raw_env.render()
                # iio.imwrite(
                #     f"{output_dir}/frame_{frame_counter:06d}.png", frame
                # )
                is_success = False
                for _ in range(self._max_skill_steps):
                    act = skill.get_action(curr_aug_obs)
                    if act is None:
                        print("No action returned by skill")
                        break
                    next_raw_obs, _, _, _, info = raw_env.step(act)
                    curr_raw_obs = next_raw_obs
                    atoms = self.system.perceiver.step(curr_raw_obs)
                    frame_counter += 1
                    # frame = raw_env.render()
                    # iio.imwrite(
                    #     f"{output_dir}/frame_{frame_counter:06d}.png", frame
                    # )

                    if debug and hasattr(raw_env, "render") and not self.training_mode:
                        frames.append(raw_env.render())

                    if edge.is_shortcut:
                        if using_goal_env and goal_env is not None:
                            target_state = self.policy.node_states[edge.target.id]
                            target_atoms = set(edge.target.atoms)
                            if goal_env.use_atom_as_obs is True:
                                target_atom_vector = goal_env.create_atom_vector(
                                    target_atoms
                                )
                                current_atom_vector = goal_env.create_atom_vector(atoms)
                            curr_aug_obs = {
                                "observation": curr_raw_obs,
                                "achieved_goal": (
                                    curr_raw_obs
                                    if goal_env.use_atom_as_obs is False
                                    else current_atom_vector
                                ),
                                "desired_goal": (
                                    target_state
                                    if goal_env.use_atom_as_obs is False
                                    else target_atom_vector
                                ),
                            }
                        elif self.use_context_wrapper and self.context_env is not None:
                            self.context_env.set_context(atoms, goal_atoms)
                            curr_aug_obs = self.context_env.augment_observation(
                                curr_raw_obs  # type: ignore[arg-type]
                            )
                        else:
                            # type: ignore[assignment]
                            curr_aug_obs = curr_raw_obs
                    else:
                        curr_aug_obs = curr_raw_obs  # type: ignore[assignment]

                    num_steps += 1

                    if goal_atoms == atoms:
                        # Store the observed state for the target node
                        target_id = edge.target.id
                        if target_id not in self.observed_states:
                            self.observed_states[target_id] = []
                        is_duplicate = False
                        if hasattr(curr_raw_obs, "nodes"):
                            for existing_obs in self.observed_states[target_id]:
                                assert hasattr(existing_obs, "nodes")
                                if np.array_equal(
                                    existing_obs.nodes, curr_raw_obs.nodes
                                ):
                                    is_duplicate = True
                                    break
                        elif isinstance(curr_raw_obs, np.ndarray):
                            for existing_obs in self.observed_states[target_id]:
                                assert isinstance(existing_obs, np.ndarray)
                                if np.array_equal(existing_obs, curr_raw_obs):
                                    is_duplicate = True
                                    break
                        else:
                            raise TypeError(
                                "Unsupported observation type for duplicate check"
                            )
                        if not is_duplicate:
                            self.observed_states[target_id].append(curr_raw_obs)

                        path_str = (
                            "-".join(str(node_id) for node_id in path)
                            if path
                            else "start"
                        )
                        print(
                            f"Added edge {edge.source.id} -> {edge.target.id} cost: {num_steps} via {path_str}. Is shortcut? {edge.is_shortcut}"  # pylint: disable=line-too-long
                        )
                        if debug and frames:
                            iio.mimsave(
                                video_filename.replace(".mp4", "_success.mp4"),
                                frames,
                                fps=5,
                            )
                        is_success = True
                        break  # success

                if not is_success:
                    # Edge expansion failed.
                    if debug and frames:
                        iio.mimsave(video_filename, frames, fps=5)
                    print(
                        f"Edge expansion failed: {edge.source.id} -> {edge.target.id}"
                    )
                    continue

                # Store cost for this specific path
                edge.costs[(path, node.id)] = num_steps
                if edge.cost == float("inf") or num_steps < edge.cost:
                    edge.cost = num_steps

                # Update path to include current node for next traversal
                new_path = path + (node.id,)
                path_states[(new_path, edge.target, edge.target)] = (curr_raw_obs, info)
                queue.append((edge.target, new_path))

        print("\nAll path costs:")
        for edge in self.planning_graph.edges:
            if edge.costs:
                cost_details = []
                for (p, _), cost in edge.costs.items():
                    path_str = "-".join(str(node_id) for node_id in p) if p else "start"
                    cost_details.append(f"via {path_str}: {cost}")
                print(
                    f"Edge {edge.source.id}->{edge.target.id}: {', '.join(cost_details)}"
                )

    def _create_planning_env(self) -> gym.Env:
        """Create a separate environment instance for planning simulations."""
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

    def _using_goal_env(
        self, env: gym.Env | None
    ) -> tuple[bool, GoalConditionedWrapper | None]:
        """Check if we're using the goal-conditioned wrapper and using node
        atoms as goals."""
        using_goal_env = False
        current_env = env
        while hasattr(current_env, "env") and current_env is not None:
            if isinstance(current_env, GoalConditionedWrapper):
                using_goal_env = True
                return using_goal_env, current_env
            current_env = current_env.env
        current_env = None
        return using_goal_env, current_env

    def train(self, train_data: TrainingData | None, config: TrainingConfig):

        if config.skip_train:
            return

        # obs, info = self.system.reset()
        # _, _, goal_atoms = self.system.perceiver.reset(obs, info)

        env_to_wrap = copy.deepcopy(self.system.env)
        if self.use_context_wrapper and self.context_env is not None:
            env_to_wrap = copy.deepcopy(self.context_env)

        training_env = SLAPWrapper(
            base_env=env_to_wrap,
            perceiver=self.system.perceiver,
            max_episode_steps=config.max_env_steps,
            step_penalty=config.step_penalty,
            achievement_bonus=config.success_reward,
            action_scale=config.action_scale,
        )

        render_mode = getattr(training_env, "_render_mode", None)
        can_render = render_mode is not None
        if config.record_training and can_render:
            video_folder = Path(f"videos/{self.system.name}_{self.name}_train")
            video_folder.mkdir(parents=True, exist_ok=True)
            training_env = RecordVideo(
                training_env,  # type: ignore[assignment]
                str(video_folder),
                episode_trigger=lambda x: x % config.training_record_interval == 0,
                name_prefix="training",
            )

        self.policy.train(training_env, config, train_data)
