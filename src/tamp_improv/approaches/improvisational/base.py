"""Base improvisational TAMP approach."""

from __future__ import annotations

import copy
import heapq
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
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

from tamp_improv.approaches.base import (
    ActType,
    ApproachStepResult,
    BaseApproach,
    ImprovisationalTAMPSystem,
    ObsType,
)
from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.policies.base import Policy, PolicyContext
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.benchmarks.context_wrapper import ContextAwareWrapper
from tamp_improv.benchmarks.goal_wrapper import GoalConditionedWrapper


@dataclass
class ShortcutSignature:
    """Domain-agnostic signature of a shortcut for matching purposes."""

    source_predicates: set[str]
    target_predicates: set[str]
    source_types: set[str]
    target_types: set[str]

    @classmethod
    def from_context(
        cls, source_atoms: set[GroundAtom], target_atoms: set[GroundAtom]
    ) -> ShortcutSignature:
        """Create signature from context."""
        source_preds = {atom.predicate.name for atom in source_atoms}
        target_preds = {atom.predicate.name for atom in target_atoms}
        source_types = set()
        for atom in source_atoms:
            for obj in atom.objects:
                source_types.add(obj.type.name)
        target_types = set()
        for atom in target_atoms:
            for obj in atom.objects:
                target_types.add(obj.type.name)
        return cls(source_preds, target_preds, source_types, target_types)

    def similarity(self, other: ShortcutSignature) -> float:
        """Calculate similarity score between signatures."""
        source_pred_sim = len(self.source_predicates & other.source_predicates) / max(
            len(self.source_predicates | other.source_predicates), 1
        )
        target_pred_sim = len(self.target_predicates & other.target_predicates) / max(
            len(self.target_predicates | other.target_predicates), 1
        )
        source_type_sim = len(self.source_types & other.source_types) / max(
            len(self.source_types | other.source_types), 1
        )
        target_type_sim = len(self.target_types & other.target_types) / max(
            len(self.target_types | other.target_types), 1
        )
        return (
            0.3 * source_pred_sim
            + 0.3 * target_pred_sim
            + 0.2 * source_type_sim
            + 0.2 * target_type_sim
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another ShortcutSignature."""
        if not isinstance(other, ShortcutSignature):
            return False
        return (
            self.source_predicates == other.source_predicates
            and self.target_predicates == other.target_predicates
            and self.source_types == other.source_types
            and self.target_types == other.target_types
        )

    def __hash__(self) -> int:
        """Hash function for ShortcutSignature."""
        return hash(
            (
                frozenset(self.source_predicates),
                frozenset(self.target_predicates),
                frozenset(self.source_types),
                frozenset(self.target_types),
            )
        )


class ImprovisationalTAMPApproach(BaseApproach[ObsType, ActType]):
    """General improvisational TAMP approach.

    This approach combines task-and-motion planning with learned
    policies for creating shortcuts between non-adjacent nodes in the
    plan.
    """

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        policy: Policy[ObsType, ActType],
        seed: int,
        planner_id: str = "pyperplan",
        max_skill_steps: int = 150,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy
        self.planner_id = planner_id
        self.max_skill_steps = max_skill_steps

        self.domain = system.get_domain()

        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None
        self._goal: set[GroundAtom] = set()

        self.planning_graph: PlanningGraph | None = None
        self.current_path: list[PlanningGraphEdge] = []
        self._current_edge: PlanningGraphEdge | None = None
        self._goal_atoms: set[GroundAtom] = set()
        self.policy_active = False
        self.observed_states: dict[int, list[ObsType]] = {}
        self.best_eval_path: list[PlanningGraphEdge] = []
        self.best_eval_total_steps: int = 0
        self.edge_action_cache: dict[tuple[int, int, tuple[int, ...]], list[Any]] = {}
        self._cached_actions: list[Any] = []   # pre-validated action sequence for current edge
        self._executed_path: tuple[int, ...] = ()  # path prefix used for cache key lookup

        self.trained_signatures: list[ShortcutSignature] = []

        self.rng = np.random.default_rng(seed)

        # For evaluation episode logging
        self.initial_node_id: int | None = None
        self.goal_node_ids: list[int] = []
        self.initial_node_atoms: list[str] = []
        self.goal_node_atoms_list: list[list[str]] = []
        self.best_eval_path_node_ids: list[int] = []
        self.best_eval_path_edge_details: list[dict[str, Any]] = []
        self.shortcuts_added_to_graph: int = 0

        # Cached planning env — created once, reused across all eval episodes
        self._planning_env: gym.Env | None = None

        # If True, use pre-built cost table for eval path (no physics simulation)
        self.fast_eval: bool = False

        # Cost table built from training graph edge costs.
        # Key: (frozenset(source_atoms), frozenset(target_atoms)) → min observed cost
        self._edge_cost_table: dict[tuple[frozenset, frozenset], float] = {}

        # Only initialize MultiRLPolicy here (needs base_env set up early)
        # Other policies will be initialized during training after wrappers are applied
        if isinstance(policy, MultiRLPolicy):
            policy.initialize(system.wrapped_env)

    def update_system(
        self, new_system: ImprovisationalTAMPSystem[ObsType, ActType]
    ):
        """Update the system for this approach."""
        self.system = new_system
        self.domain = new_system.get_domain()

    def reset(
        self,
        obs: ObsType,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        print("[DBG] approach.reset: entry")
        objects, atoms, goal = self.system.perceiver.reset(obs, info)
        print(f"[DBG] approach.reset: perceiver.reset done, {len(atoms)} atoms")
        self._goal = goal
        self.observed_states = {}
        self.edge_action_cache.clear()
        self.shortcuts_added_to_graph = 0

        print("[DBG] approach.reset: about to _create_planning_graph")
        self.planning_graph = self._create_planning_graph(objects, atoms)
        print(f"[DBG] approach.reset: _create_planning_graph done, {len(self.planning_graph.nodes)} nodes")

        initial_node = self.planning_graph.node_map[frozenset(atoms)]
        self.observed_states[initial_node.id] = []
        self.observed_states[initial_node.id].append(obs)

        # Always set node/atom fields so diagnostics are correct even on early exit
        self.initial_node_id = initial_node.id
        self.initial_node_atoms = sorted(str(a) for a in initial_node.atoms)
        goal_nodes = [n for n in self.planning_graph.nodes if goal.issubset(n.atoms)]
        self.goal_node_ids = [n.id for n in goal_nodes]
        self.goal_node_atoms_list = [sorted(str(a) for a in n.atoms) for n in goal_nodes]

        # Check if already at goal
        if goal.issubset(atoms):
            # Already at goal - episode immediately succeeds
            self.current_path = []
            self.best_eval_path = []
            self.best_eval_total_steps = 0
            self._current_operator = None
            self._current_skill = None
            self._current_edge = None
            self._goal_atoms = set()
            self.policy_active = False
            # Terminate with success
            return ApproachStepResult(
                action=self.system.wrapped_env.action_space.sample(),
                terminate=True,
                info={"already_at_goal": True},
            )

        # Compute edge costs and find shortest path
        print("Training mode:", self.training_mode)
        if not self.training_mode:
            print("[DBG] approach.reset: about to _try_add_shortcuts")
            self._try_add_shortcuts(self.planning_graph)
            print(f"[DBG] approach.reset: _try_add_shortcuts done, added={self.shortcuts_added_to_graph}")
            if self.fast_eval:
                print("[DBG] approach.reset: about to _compute_eval_path_fast")
                self.current_path = self._compute_eval_path_fast(obs, info, goal)
                print(f"[DBG] approach.reset: _compute_eval_path_fast done, path_len={len(self.current_path)}")
            else:
                print("[DBG] approach.reset: about to _compute_eval_path")
                self.current_path = self._compute_eval_path(obs, info, goal)
                print(f"[DBG] approach.reset: _compute_eval_path done, path_len={len(self.current_path)}")
            # print("Computed eval path:", self.current_path)
        else:
            print("[DBG] approach.reset: about to _compute_planning_graph_edge_costs")
            self._compute_planning_graph_edge_costs(obs, info)
            print("[DBG] approach.reset: about to find_shortest_path")
            self.current_path = self.planning_graph.find_shortest_path(atoms, goal)
            print("[DBG] approach.reset: find_shortest_path done")

        # If no path found, terminate with failure
        if not self.current_path:
            self._current_operator = None
            self._current_skill = None
            self._current_edge = None
            self._goal_atoms = set()
            self.policy_active = False
            return ApproachStepResult(
                action=self.system.wrapped_env.action_space.sample(),
                terminate=True,
                info={"no_path_found": True},
            )

        self.best_eval_path = list(self.current_path)
        self.best_eval_total_steps = int(
            sum((e.cost if e.cost != float("inf") else 0) for e in self.best_eval_path)
        )

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
        # print("Step")
        atoms = self.system.perceiver.step(obs)
        using_goal_env, goal_env = self._using_goal_env(self.system.wrapped_env)
        using_context_env, context_env = self._using_context_env(
            self.system.wrapped_env
        )
        target_vec = None
        current_vec = None

        # Check if policy achieved its goal
        if self.policy_active and self.planning_graph:
            current_node = self._current_edge.source if self._current_edge else None
            if current_node and self._goal_atoms == atoms:
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
                        target_vec = goal_env.create_atom_vector(target_atoms)
                        current_vec = goal_env.create_atom_vector(atoms)
                    else:
                        target_state = self.policy.node_states[target_node_id]
                        if isinstance(target_state, list):
                            target_state = target_state[0]  # Use first state
                        target_vec = goal_env.flatten_obs(target_state)
                        current_vec = goal_env.flatten_obs(obs)
                    dict_obs = {
                        "observation": goal_env.flatten_obs(obs),
                        "achieved_goal": current_vec,
                        "desired_goal": target_vec,
                    }
                    return ApproachStepResult(action=self.policy.get_action(dict_obs))  # type: ignore[arg-type] # pylint: disable=line-too-long
            elif using_context_env and context_env is not None:
                aug_obs = context_env.augment_observation(obs)
                return ApproachStepResult(action=self.policy.get_action(aug_obs))  # type: ignore[arg-type] # pylint: disable=line-too-long
            return ApproachStepResult(action=self.policy.get_action(obs))

        assert self.planning_graph is not None
        if not self._current_edge and self.current_path:
            self._current_edge = self.current_path.pop(0)

            # if self._current_edge.is_shortcut:
            #     self.policy_active = True

            #     target_node = self._current_edge.target
            #     self._goal_atoms = set(target_node.atoms)

            #     self.policy.configure_context(
            #         PolicyContext(
            #             goal_atoms=self._goal_atoms,
            #             current_atoms=atoms,
            #             info={
            #                 "source_node_id": self._current_edge.source.id,
            #                 "target_node_id": target_node.id,
            #             },
            #         )
            #     )
            #     if using_goal_env and goal_env is not None:
            #         target_state = self.policy.node_states[target_node.id]
            #         if isinstance(target_state, list):
            #             target_state = target_state[0]
            #         target_atoms = set(self._current_edge.target.atoms)
            #         if goal_env.use_atom_as_obs is True:
            #             target_vec = goal_env.create_atom_vector(target_atoms)
            #             current_vec = goal_env.create_atom_vector(atoms)
            #         else:
            #             target_vec = goal_env.flatten_obs(target_state)
            #             current_vec = goal_env.flatten_obs(obs)
            #         dict_obs = {
            #             "observation": goal_env.flatten_obs(obs),
            #             "achieved_goal": current_vec,
            #             "desired_goal": target_vec,
            #         }
            #         return ApproachStepResult(action=self.policy.get_action(dict_obs))  # type: ignore[arg-type] # pylint: disable=line-too-long
            #     if using_context_env and context_env is not None:
            #         aug_obs = context_env.augment_observation(obs)
            #         return ApproachStepResult(action=self.policy.get_action(aug_obs))  # type: ignore[arg-type] # pylint: disable=line-too-long
            #     return ApproachStepResult(action=self.policy.get_action(obs))

            self._current_operator = self._current_edge.operator
            if not self._current_operator:
                raise TaskThenMotionPlanningFailure("Edge has no operator")

            self._current_skill = self._get_skill(self._current_operator)
            self._current_skill.reset(self._current_operator)

        if self._current_edge and set(self._current_edge.target.atoms) == atoms:
            self._current_edge = None
            return self.step(obs, reward, terminated, truncated, info)

        if not self._current_skill:
            raise TaskThenMotionPlanningFailure("No current skill")

        try:
            # print("I am in state:", obs)
            # print("I am getting my current skill's action:", self._current_skill, "from edge:", self._current_edge)
            action = self._current_skill.get_action(obs)
            # print("I am executing action:", action)
        except AssertionError as e:
            print(f"Assertion error in skill {self._current_skill}: {e}")
            action = None
        if action is None:
            print(f"No action returned by skill {self._current_skill} — terminating episode as failure")
            return ApproachStepResult(
                action=self.system.wrapped_env.action_space.sample(),
                terminate=True,
                info={"skill_failed": True},
            )
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
        sequences."""
        graph = PlanningGraph()
        initial_node = graph.add_node(init_atoms)
        visited_states = {frozenset(init_atoms): initial_node}
        queue = deque([(initial_node, 0)])  # Queue for BFS: [(node, depth)]
        node_count = 0
        max_nodes = 1300

        # print(init_atoms, self._goal)

        while queue and node_count < max_nodes:
            current_node, depth = queue.popleft()
            node_count += 1

            # Check if this is a goal state, stop search if so
            # NOTE: we use the same assumption as PDDL to find the goal nodes of the
            # shortest sequences of symbolic actions
            # We remove this because it might not be super useful when shortcuts can be flawed?
            # if self._goal and self._goal.issubset(current_node.atoms):
            #     queue.clear()
            #     break

            applicable_ops = self._find_applicable_operators(
                set(current_node.atoms), objects
            )

            print("Current node:", current_node.id, "Depth:", depth)
            print("Applicable operators:", [op.name for op in applicable_ops])

            for op in applicable_ops:
                next_atoms = set(current_node.atoms)
                next_atoms.difference_update(op.delete_effects)
                next_atoms.update(op.add_effects)

                next_atoms_frozen = frozenset(next_atoms)
                # Check if operator is a shortcut by name
                is_shortcut = "Shortcut" in op.name
                # if is_shortcut:
                    # print(f"Found shortcut operator: {op.name}, from node {current_node.atoms} to new node {next_atoms}")
                if next_atoms_frozen in visited_states:
                    next_node = visited_states[next_atoms_frozen]
                    # print("I'm about to add an edge from", current_node.id, "to", next_node.id, "using operator", op.name, "is shortcut", is_shortcut)
                    graph.add_edge(current_node, next_node, op, is_shortcut=is_shortcut)
                else:
                    next_node = graph.add_node(next_atoms)
                    visited_states[next_atoms_frozen] = next_node
                    # print("I'm about to add an edge from", current_node.id, "to", next_node.id, "using operator", op.name, "is shortcut", is_shortcut)
                    graph.add_edge(current_node, next_node, op, is_shortcut=is_shortcut)
                    queue.append((next_node, depth + 1))

                
                # for e in graph.edges:
                #     if e.operator.name == "Shortcut_6":
                #         print("Edge with operator:", e.operator.name, "is shortcut", e.is_shortcut)

        print(
            f"Planning graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
        )

        # print("Edges of graph:", graph.edges)
        # for e in graph.edges:
        #     if e.operator.name == "Shortcut_6":
        #         print("Edge with operator:", e.operator.name, "is shortcut", e.is_shortcut)

        # print("\nGraph Edges:")
        # for edge in graph.edges:
        #     op_str = f"{edge.operator.name}" if edge.operator else "SHORTCUT"
        #     print(f"  Node {edge.source.id} --[{op_str}]--> Node {edge.target.id}")
        # raise Exception("Debugging - stop here")
        return graph

    def _find_applicable_operators(
        self, current_atoms: set[GroundAtom], objects: set[Object]
    ) -> list[GroundOperator]:
        """Find all ground operators that are applicable in the current
        state."""
        applicable_ops = []
        domain_operators = self.domain.operators

        for lifted_op in domain_operators:
            valid_groundings = self._find_valid_groundings(lifted_op, objects)

            for grounding in valid_groundings:
                ground_op = lifted_op.ground(grounding)

                if ground_op.preconditions.issubset(current_atoms):
                    applicable_ops.append(ground_op)

        return applicable_ops

    def _find_valid_groundings(
        self, lifted_op: LiftedOperator, objects: set[Object]
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

    def _try_add_shortcuts(self, graph: PlanningGraph) -> None:
        """Try to add shortcut edges to the graph."""
        using_goal_env, _ = self._using_goal_env(self.system.wrapped_env)
        if not self.training_mode and self.trained_signatures:
            print(
                f"DEBUG: Attempting to add shortcuts with {len(self.trained_signatures)} trained signatures"
            )
        for source_node in graph.nodes:
            for target_node in graph.nodes:
                if source_node == target_node:
                    continue
                if any(
                    edge.target == target_node
                    for edge in graph.node_to_outgoing_edges.get(source_node, [])
                ):
                    continue
                # if target_node.id <= source_node.id:
                #     continue

                source_atoms = set(source_node.atoms)
                target_atoms = set(target_node.atoms)
                # print(source_atoms, target_atoms)
                if not target_atoms:
                    continue

                if self.trained_signatures and not using_goal_env:
                    current_sig = ShortcutSignature.from_context(
                        source_atoms, target_atoms
                    )
                    # print(current_sig)
                    can_handle = False
                    best_similarity = 0.0

                    for trained_sig in self.trained_signatures:
                        similarity = current_sig.similarity(trained_sig)
                        if similarity > 0.99:
                            can_handle = True
                            best_similarity = max(best_similarity, similarity)
                    # print("Best sim:", best_similarity)

                    if not can_handle:
                        # if self.training_mode:
                        #     print(f"  DEBUG: Skipping {source_node.id}→{target_node.id} - no matching signature (best sim: {best_similarity:.2f})")
                        continue

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
                    if not self.training_mode:
                        print(
                            f"  DEBUG: Adding shortcut {source_node.id}→{target_node.id} (sim: {best_similarity:.2f})"
                        )
                    graph.add_edge(source_node, target_node, None, is_shortcut=True)
                    self.shortcuts_added_to_graph += 1
                elif not self.training_mode:
                    # print(f"  DEBUG: Matched signature for {source_node.id}→{target_node.id} but policy.can_initiate() = False")
                    pass

        if not self.training_mode and self.trained_signatures:
            print(f"DEBUG: Added {self.shortcuts_added_to_graph} shortcuts to graph")

    @property
    def eval_path(self) -> list[PlanningGraphEdge]:
        return self.best_eval_path

    def _compute_eval_path(
        self,
        obs: ObsType,
        info: dict[str, Any],
        goal: set[GroundAtom],
    ) -> list[PlanningGraphEdge]:
        """Efficiently compute shortest path during evaluation."""
        assert self.planning_graph is not None

        l = list(self.planning_graph.node_map.keys())
        print("Node map:")
        for i in range(len(l)):
            print(f"  Node {i}: {l[i]}")

        _, init_atoms, _ = self.system.perceiver.reset(obs, info)
        initial_node = self.planning_graph.node_map[frozenset(init_atoms)]
        goal_nodes = [
            node for node in self.planning_graph.nodes if goal.issubset(node.atoms)
        ]

        self.initial_node_id = initial_node.id
        self.goal_node_ids = [node.id for node in goal_nodes]
        self.initial_node_atoms = sorted(str(a) for a in initial_node.atoms)
        self.goal_node_atoms_list = [
            sorted(str(a) for a in node.atoms) for node in goal_nodes
        ]

        # print("Initial node:", initial_node)
        # print("Initial obs:", obs)
        # print("Goal nodes:", goal_nodes)

        if not goal_nodes:
            print("No goal nodes found in planning graph")
            return []

        # Check if already at goal
        if initial_node in goal_nodes:
            print("Already at goal node - no path needed")
            return []

        raw_env = self._create_planning_env()
        # CRITICAL: Reset planning env to match current state before planning
        # Otherwise the deepcopy will have a different random initialization
        raw_env.reset_from_state(obs)  # type: ignore
        using_goal_env, goal_env = self._using_goal_env(self.system.wrapped_env)
        using_context_env, context_env = self._using_context_env(
            self.system.wrapped_env
        )

        # (total_cost, counter, node, path_tuple, path_edges, path_state_info)
        counter = itertools.count()
        empty_path: tuple[int, ...] = tuple()
        pq: list[
            tuple[
                int,
                int,
                PlanningGraphNode,
                tuple[int, ...],
                list[PlanningGraphEdge],
                tuple[ObsType, dict[str, Any]],
            ]
        ] = [(0, next(counter), initial_node, empty_path, [], (obs, info))]

        distances: dict[tuple[int, tuple[int, ...]], float] = {}
        distances[(initial_node.id, empty_path)] = 0

        best_goal_cost = float("inf")
        best_goal_path: list[PlanningGraphEdge] = []
        best_goal_node: PlanningGraphNode | None = None

        max_path_length = len(self.planning_graph.nodes) * 2  # Prevent infinite loops

        while pq:
            (
                current_cost,
                _,
                current_node,
                current_path,
                path_edges,
                (path_state, path_info),
            ) = heapq.heappop(pq)

            # print("Current node:", current_node.id, "with cost:", current_cost)
            # print("Current path:", current_path)
            # print("Path edges:", [f"{e.source.id}->{e.target.id}" for e in path_edges])

            if current_cost >= best_goal_cost:
                break

            if len(current_path) > max_path_length:
                continue

            state_key = (current_node.id, current_path)
            if state_key in distances and current_cost > distances[state_key]:
                continue

            # print("Outgoing edges from node", current_node.id, "are:", 
            #       [f"{edge.source.id}->{edge.target.id}" for edge in 
            #        self.planning_graph.node_to_outgoing_edges.get(current_node, [])])

            for edge in self.planning_graph.node_to_outgoing_edges.get(
                current_node, []
            ):
                # print("  Considering edge:", f"{edge.source.id}->{edge.target.id}")
                # Check if target node is already in current path to prevent cycles
                if edge.target.id in current_path:
                    continue

                edge_cost, end_state, end_info, success = self._execute_edge(
                    edge,
                    path_state,
                    path_info,
                    raw_env,
                    using_goal_env,
                    goal_env,
                    using_context_env,
                    context_env,
                    current_path,
                )

                if not success:
                    print(
                        f"    Edge {current_node.id} -> {edge.target.id} execution failed."  # pylint: disable=line-too-long
                    )
                    continue

                if current_node.id == 0 and edge.target.id == 3:
                    print("    Debug: Edge from initial node to node 3 executed with cost:", edge_cost)
                    print("Nodes 0 and 3 atoms:", current_node.atoms, edge.target.atoms)

                new_total_cost = current_cost + edge_cost
                print(
                    f"    Edge {current_node.id} -> {edge.target.id} executed with cost {edge_cost}. Is shortcut? {edge.is_shortcut}."  # pylint: disable=line-too-long
                )

                new_path_edges = path_edges + [edge]

                if edge.target in goal_nodes:
                    if new_total_cost < best_goal_cost:
                        best_goal_cost = new_total_cost
                        best_goal_path = new_path_edges
                        best_goal_node = edge.target
                    else:
                        continue

                if new_total_cost >= best_goal_cost:
                    continue

                edge.costs[(current_path, current_node.id)] = edge_cost
                if edge.cost == float("inf") or edge_cost < edge.cost:
                    edge.cost = edge_cost

                new_path = current_path + (current_node.id,)
                new_state_key = (edge.target.id, new_path)

                # Only add to queue if this is a better path to this (node, path) state
                if (
                    new_state_key not in distances
                    or new_total_cost < distances[new_state_key]
                ):
                    distances[new_state_key] = new_total_cost
                    heapq.heappush(  # type: ignore[misc]
                        pq,
                        (
                            new_total_cost,
                            next(counter),
                            edge.target,
                            new_path,
                            new_path_edges,
                            (end_state, end_info),
                        ),
                    )

        if best_goal_cost < float("inf"):
            assert best_goal_node is not None
            node_ids = [edge.source.id for edge in best_goal_path]
            node_ids.append(best_goal_path[-1].target.id)
            node_str = " -> ".join(map(str, node_ids))
            shortcut_edges = [
                f"{edge.source.id} -> {edge.target.id}"
                for edge in best_goal_path
                if edge.is_shortcut
            ]
            self.best_eval_path_node_ids = node_ids
            self.best_eval_path_edge_details = []
            current_path_tuple = empty_path
            for i, edge in enumerate(best_goal_path):
                path_key = (current_path_tuple, edge.source.id)
                cost_for_this_path_segment = edge.costs.get(path_key, float("inf"))
                self.best_eval_path_edge_details.append(
                    {
                        "source": edge.source.id,
                        "target": edge.target.id,
                        "is_shortcut": edge.is_shortcut,
                        "cost": cost_for_this_path_segment,
                    }
                )
                current_path_tuple = current_path_tuple + (edge.source.id,)
            if shortcut_edges:
                shortcut_str = ", ".join(shortcut_edges)
                print(
                    f"Optimal path found with cost {best_goal_cost}: {node_str} (with shortcut(s) {shortcut_str})"  # pylint: disable=line-too-long
                )
            else:
                print(f"Optimal path found with cost {best_goal_cost}: {node_str}")
            return best_goal_path

        print("No path found to goal")
        return []

    def _compute_eval_path_fast(
        self,
        obs: ObsType,
        info: dict[str, Any],
        goal: set[GroundAtom],
    ) -> list[PlanningGraphEdge]:
        """Compute eval path using pre-built edge cost table — no physics simulation.

        Costs are looked up by (frozenset(source_atoms), frozenset(target_atoms)).
        Regular edges unseen in training get cost=inf (excluded from search).
        Shortcut edges unseen in training get shortcut_default_cost.
        Edges with operator=None (from _try_add_shortcuts) get cost=inf since
        base.py:step() cannot execute them.
        """
        assert self.planning_graph is not None

        _, init_atoms, _ = self.system.perceiver.reset(obs, info)
        initial_node = self.planning_graph.node_map[frozenset(init_atoms)]
        goal_nodes = [n for n in self.planning_graph.nodes if goal.issubset(n.atoms)]

        self.initial_node_id = initial_node.id
        self.goal_node_ids = [n.id for n in goal_nodes]
        self.initial_node_atoms = sorted(str(a) for a in initial_node.atoms)
        self.goal_node_atoms_list = [sorted(str(a) for a in n.atoms) for n in goal_nodes]

        if not goal_nodes:
            print("No goal nodes found in planning graph (fast path)")
            return []
        if initial_node in goal_nodes:
            print("Already at goal node (fast path)")
            return []

        for edge in self.planning_graph.edges:
            key = (edge.source.atoms, edge.target.atoms)
            if edge.operator is None:
                # _try_add_shortcuts edges have no operator — step() cannot execute them
                edge.max_cost = float("inf")
                edge.cost = float("inf")
            else:
                edge.max_cost = self._edge_cost_table.get(key, float("inf"))
                edge.cost = self._edge_cost_table.get(key, float("inf"))

        path = self.planning_graph.find_shortest_path(init_atoms, goal)
        if path:
            node_ids = [edge.source.id for edge in path]
            node_ids.append(path[-1].target.id)
            node_str = " -> ".join(map(str, node_ids))
            total_cost = sum(e.cost for e in path if e.cost != float("inf"))
            shortcut_edges = [
                f"{e.source.id} -> {e.target.id}"
                for e in path if e.is_shortcut
            ]
            if shortcut_edges:
                print(f"[FAST] Optimal path found with cost {total_cost}: {node_str} (with shortcut(s) {', '.join(shortcut_edges)})")
            else:
                print(f"[FAST] Optimal path found with cost {total_cost}: {node_str}")
        else:
            print("[FAST] No path found")
        return path

    def _execute_edge(
        self,
        edge: PlanningGraphEdge,
        start_state: ObsType,
        start_info: dict[str, Any],
        raw_env: gym.Env,
        using_goal_env: bool,
        goal_env: GoalConditionedWrapper | None,
        using_context_env: bool,
        context_env: ContextAwareWrapper | None,
        current_path: tuple[int, ...] = tuple(),
    ) -> tuple[float, ObsType, dict[str, Any], bool]:
        """Execute a single edge and return the cost and end state."""
        raw_env.reset_from_state(start_state)  # type: ignore

        _, init_atoms, _ = self.system.perceiver.reset(start_state, start_info)
        goal_atoms = set(edge.target.atoms)
        actions: list[Any] = []

        # print("Executing edge:", edge, init_atoms, goal_atoms)

        # if edge.is_shortcut:
        #     self.policy.configure_context(
        #         PolicyContext(
        #             goal_atoms=goal_atoms,
        #             current_atoms=init_atoms,
        #             info={
        #                 "source_node_id": edge.source.id,
        #                 "target_node_id": edge.target.id,
        #             },
        #         )
        #     )
        #     if using_goal_env and goal_env is not None:
        #         assert hasattr(
        #             self.policy, "node_states"
        #         ), "Policy must have node_states"
        #         target_state = self.policy.node_states[edge.target.id]
        #         if isinstance(target_state, list):
        #             target_state = target_state[0]
        #         target_atoms_set = set(edge.target.atoms)
        #         if goal_env.use_atom_as_obs is True:
        #             target_vec = goal_env.create_atom_vector(target_atoms_set)
        #             current_vec = goal_env.create_atom_vector(init_atoms)
        #         else:
        #             target_vec = goal_env.flatten_obs(target_state)
        #             current_vec = goal_env.flatten_obs(start_state)
        #         aug_obs = {
        #             "observation": goal_env.flatten_obs(start_state),
        #             "achieved_goal": current_vec,
        #             "desired_goal": target_vec,
        #         }
        #     elif using_context_env and context_env is not None:
        #         aug_obs = context_env.augment_observation(start_state)  # type: ignore[assignment]  # pylint: disable=line-too-long
        #     else:
        #         aug_obs = start_state  # type: ignore[assignment]
        #     skill: Policy | Skill = self.policy
        # else:
        #     assert edge.operator is not None
        #     skill = self._get_skill(edge.operator)
        #     skill.reset(edge.operator)
        #     aug_obs = start_state  # type: ignore[assignment]
        #     # print("Skill:", skill)
        
        assert edge.operator is not None
        skill = self._get_skill(edge.operator)
        skill.reset(edge.operator)
        aug_obs = start_state  # type: ignore[assignment]
        # print("Skill:", skill)

        num_steps = 0
        curr_raw_obs = start_state
        curr_aug_obs = aug_obs
        for _ in range(self.max_skill_steps):
            try:
                act = skill.get_action(curr_aug_obs)
            except Exception as e:
                print(f"Skill raised {type(e).__name__}: {e}")
                return float("inf"), start_state, start_info, False
            # print("Action:", act)
            if act is None:
                print("No action returned by skill")
                return float("inf"), start_state, start_info, False
            actions.append(copy.deepcopy(act))
            next_raw_obs, _, _, _, info = raw_env.step(act)
            curr_raw_obs = next_raw_obs
            atoms = self.system.perceiver.step(curr_raw_obs)
            # print("State:", curr_raw_obs, atoms)

            # if edge.is_shortcut:
            #     if using_goal_env and goal_env is not None:
            #         target_state = self.policy.node_states[edge.target.id]
            #         if isinstance(target_state, list):
            #             target_state = target_state[0]
            #         target_atoms_set = set(edge.target.atoms)
            #         if goal_env.use_atom_as_obs is True:
            #             target_vec = goal_env.create_atom_vector(target_atoms_set)
            #             current_vec = goal_env.create_atom_vector(atoms)
            #         else:
            #             target_vec = goal_env.flatten_obs(target_state)
            #             current_vec = goal_env.flatten_obs(curr_raw_obs)
            #         curr_aug_obs = {
            #             "observation": goal_env.flatten_obs(curr_raw_obs),
            #             "achieved_goal": current_vec,
            #             "desired_goal": target_vec,
            #         }
            #     elif using_context_env and context_env is not None:
            #         curr_aug_obs = context_env.augment_observation(curr_raw_obs)  # type: ignore[assignment]  # pylint: disable=line-too-long
            #     else:
            #         curr_aug_obs = curr_raw_obs  # type: ignore[assignment]
            # else:
            curr_aug_obs = curr_raw_obs  # type: ignore[assignment]

            num_steps += 1

            if goal_atoms == atoms:
                target_id = edge.target.id
                if target_id not in self.observed_states:
                    self.observed_states[target_id] = []
                is_duplicate = False
                if hasattr(curr_raw_obs, "nodes"):
                    for existing_obs in self.observed_states[target_id]:
                        assert hasattr(existing_obs, "nodes")
                        if np.array_equal(existing_obs.nodes, curr_raw_obs.nodes):
                            is_duplicate = True
                            break
                elif isinstance(curr_raw_obs, np.ndarray):
                    for existing_obs in self.observed_states[target_id]:
                        assert isinstance(existing_obs, np.ndarray)
                        if np.array_equal(existing_obs, curr_raw_obs):
                            is_duplicate = True
                            break
                else:
                    raise TypeError("Unsupported observation type for duplicate check")

                if not is_duplicate:
                    self.observed_states[target_id].append(curr_raw_obs)

                key = (edge.source.id, edge.target.id, current_path)
                self.edge_action_cache[key] = actions.copy()

                return num_steps, curr_raw_obs, info, True

        # Skill timed out
        return float("inf"), start_state, start_info, False

    def _compute_planning_graph_edge_costs(
        self,
        obs: ObsType,
        info: dict[str, Any],
    ) -> None:
        """Compute edge costs considering the path taken to reach each node."""
        assert self.planning_graph is not None

        _, init_atoms, _ = self.system.perceiver.reset(obs, info)
        initial_node = self.planning_graph.node_map[frozenset(init_atoms)]

        path_states: dict[
            tuple[tuple[int, ...], PlanningGraphNode, PlanningGraphNode],
            tuple[ObsType, dict],
        ] = {}
        empty_path: tuple[int, ...] = tuple()
        path_states[(empty_path, initial_node, initial_node)] = (obs, info)

        raw_env = self._create_planning_env()
        using_goal_env, goal_env = self._using_goal_env(self.system.wrapped_env)
        using_context_env, context_env = self._using_context_env(
            self.system.wrapped_env
        )

        queue = [(initial_node, empty_path)]
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

            for edge in self.planning_graph.node_to_outgoing_edges.get(node, []):
                if (path, node, edge.target) in path_states:
                    continue
                if edge.target.id <= node.id:
                    continue

                raw_env.reset_from_state(path_state)  # type: ignore
                _ = self.system.perceiver.reset(path_state, path_info)

                edge_cost, end_state, _, success = self._execute_edge(
                    edge,
                    path_state,
                    path_info,
                    raw_env,
                    using_goal_env,
                    goal_env,
                    using_context_env,
                    context_env,
                    path,
                )

                if not success:
                    print(
                        f"Edge expansion failed: {edge.source.id} -> {edge.target.id}"
                    )
                    continue

                edge.costs[(path, node.id)] = edge_cost
                if edge.cost == float("inf") or edge_cost < edge.cost:
                    edge.cost = edge_cost

                path_str = (
                    "-".join(str(node_id) for node_id in path) if path else "start"
                )
                print(
                    f"Added edge {edge.source.id} -> {edge.target.id} cost: {edge_cost} via {path_str}. Is shortcut? {edge.is_shortcut}"  # pylint: disable=line-too-long
                )

                new_path = path + (node.id,)
                path_states[(new_path, edge.target, edge.target)] = (end_state, info)
                queue.append((edge.target, new_path))

    def _create_planning_env(self) -> gym.Env:
        """Create (or return cached) separate environment instance for planning.

        The environment is created once via deepcopy or clone() and reused
        across all subsequent planning calls to avoid repeated C++ heap
        allocations (e.g. PyBullet physics servers) that would cause OOM.
        """
        if self._planning_env is not None:
            print("[DBG] _create_planning_env: returning cached env")
            return self._planning_env

        print("[DBG] _create_planning_env: creating new planning env (first call)")
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
        print(f"[DBG] _create_planning_env: base_env type={type(base_env).__name__}, has_clone={hasattr(base_env, 'clone')}")
        if hasattr(base_env, "clone"):
            print("[DBG] _create_planning_env: calling base_env.clone()")
            planning_env = base_env.clone()
            print("[DBG] _create_planning_env: clone() done")
        else:
            print("[DBG] _create_planning_env: calling copy.deepcopy(base_env)")
            planning_env = copy.deepcopy(base_env)
            print("[DBG] _create_planning_env: deepcopy done")
        self._planning_env = planning_env
        print("[DBG] _create_planning_env: cached and returning")
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

    def _using_context_env(
        self, env: gym.Env | None
    ) -> tuple[bool, ContextAwareWrapper | None]:
        """Check if we're using the context-aware wrapper."""
        using_context_env = False
        current_env = env
        while hasattr(current_env, "env") and current_env is not None:
            if isinstance(current_env, ContextAwareWrapper):
                using_context_env = True
                return using_context_env, current_env
            current_env = current_env.env
        current_env = None
        return using_context_env, current_env
