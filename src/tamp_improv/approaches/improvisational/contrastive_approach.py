"""Contrastive learning approach for TAMP using distance heuristic V4.

This approach uses a single trained distance heuristic (V4) to create shortcuts
between high-level nodes in the planning graph, rather than training separate
RL policies for each shortcut.
"""

from __future__ import annotations

from typing import Any, Generic, Sequence

import gymnasium as gym
from relational_structs import GroundAtom, LiftedOperator, Object, Variable
from task_then_motion_planning.structs import LiftedOperatorSkill

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.distance_heuristic_v4 import (
    DistanceHeuristicV4,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem, ObsType, ActType


class ContrastiveShortcutSkill(LiftedOperatorSkill[ObsType, ActType]):
    """Skill that uses distance heuristic to navigate to a target node."""

    def __init__(
        self,
        heuristic: DistanceHeuristicV4,
        operator: LiftedOperator,
        target_node_id: int,
        env: gym.Env,
        temperature: float = 0.0,
    ):
        """Initialize contrastive shortcut skill.

        Args:
            heuristic: Trained distance heuristic V4
            operator: The shortcut operator this skill implements
            target_node_id: ID of the target node to navigate to
            env: Environment (needed for action enumeration)
            temperature: Temperature for action selection (0 = greedy)
        """
        self._heuristic = heuristic
        self._operator = operator
        self._target_node_id = target_node_id
        self._env = env
        self._temperature = temperature
        super().__init__()

    def _get_lifted_operator(self) -> LiftedOperator:
        """Return the operator this skill implements."""
        return self._operator

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: ObsType,
    ) -> ActType:
        """Use heuristic to select action toward target node."""
        # Temporarily set heuristic temperature
        old_temp = self._heuristic.policy_temperature
        self._heuristic.policy_temperature = self._temperature

        # Call heuristic's greedy action selection
        action = self._heuristic._select_action_greedy(
            env=self._env,
            current_state=obs,
            goal_node=self._target_node_id,
        )

        # Restore temperature
        self._heuristic.policy_temperature = old_temp

        # Debug: check if we're at goal already
        current_node = self._heuristic.get_node(obs)
        # print(f"[DEBUG ContrastiveSkill {self._operator.name}] Current node: {current_node}, Target node: {self._target_node_id}")
        if current_node == self._target_node_id:
            # print(f"[DEBUG ContrastiveSkill] Already at goal node {self._target_node_id}, returning None")
            return None  # Signal completion

        # Return action or default if None
        if action is None:
            # print(f"[DEBUG ContrastiveSkill] Heuristic returned None for node {current_node} -> {self._target_node_id}, sampling random")
            # Fall back to first valid action
            return self._env.action_space.sample()
        # print(f"[DEBUG ContrastiveSkill] Returning action {action}")
        return action


class ContrastiveApproach(ImprovisationalTAMPApproach[ObsType, ActType]):
    """TAMP approach using contrastive distance heuristic for shortcuts.

    This approach extends ImprovisationalTAMPApproach but instead of training
    separate RL policies for each shortcut, it uses a single trained distance
    heuristic to create skills for navigating between nodes.
    """

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        heuristic: DistanceHeuristicV4,
        seed: int,
    ):
        """Initialize contrastive approach.

        Args:
            system: TAMP system
            heuristic: Trained distance heuristic V4
            seed: Random seed
        """
        # Initialize with a dummy policy (we won't use it for shortcuts)
        from tamp_improv.approaches.improvisational.policies.multi_rl import (
            MultiRLPolicy,
        )
        from tamp_improv.approaches.improvisational.policies.rl import RLConfig

        dummy_policy = MultiRLPolicy(seed=seed, config=RLConfig())
        super().__init__(system, dummy_policy, seed)

        self.heuristic = heuristic
        self._shortcut_count = 0

    def add_shortcut(
        self,
        source_atoms: frozenset[GroundAtom],
        target_atoms: frozenset[GroundAtom],
        target_node_id: int,
        temperature: float = 0.0,
    ) -> None:
        """Add a shortcut from source node to target node.

        Creates a new operator and skill that use the distance heuristic
        to navigate from the source node to the target node.

        Args:
            source_atoms: Atoms of the source node (preconditions)
            target_atoms: Atoms of the target node (effects)
            target_node_id: ID of the target node in the planning graph
            temperature: Temperature for action selection (0 = greedy)
        """
        # Create a unique name for this shortcut
        shortcut_name = f"ContrastiveShortcut_{self._shortcut_count}"
        self._shortcut_count += 1

        # Extract all unique objects from the atoms to determine operator parameters
        # This works for any domain (not just single-robot gridworld)
        all_objects = set()
        for atom in source_atoms | target_atoms:
            all_objects.update(atom.objects)

        # Create variables for each unique object
        parameters = [Variable(f"?{obj.name}", obj.type) for obj in sorted(all_objects, key=lambda o: o.name)]

        # If no objects found, this is likely an issue - but create a dummy variable
        if not parameters:
            # Use the first available type
            available_types = list(self.system.components.types)
            if available_types:
                parameters = [Variable("?obj", available_types[0])]
            else:
                raise ValueError("No types available in system components")

        # Create object -> variable mapping for lifting ground atoms
        obj_to_var = {obj: var for obj, var in zip(sorted(all_objects, key=lambda o: o.name), parameters)}

        # Lift the ground atoms by replacing objects with variables
        def lift_atom(atom: GroundAtom) -> GroundAtom:
            """Replace objects in atom with corresponding variables."""
            lifted_objects = tuple(obj_to_var[obj] for obj in atom.objects)
            # Call the predicate with the lifted objects to create a lifted atom
            return atom.predicate(lifted_objects)

        lifted_source_atoms = {lift_atom(atom) for atom in source_atoms}
        lifted_target_atoms = {lift_atom(atom) for atom in target_atoms}

        # Determine add and delete effects
        add_effects = lifted_target_atoms - lifted_source_atoms  # Atoms being ADDED
        delete_effects = lifted_source_atoms - lifted_target_atoms  # Atoms being DELETED

        # Create the shortcut operator
        shortcut_operator = LiftedOperator(
            name=shortcut_name,
            parameters=parameters,
            preconditions=lifted_source_atoms,
            add_effects=add_effects,
            delete_effects=delete_effects,
        )

        # Check if an operator with the same PDDL signature already exists
        # Operator.__eq__ compares str(operator), which is the PDDL string
        # This prevents shortcuts from colliding with existing operators
        for existing_op in self.system.components.operators:
            if existing_op == shortcut_operator:
                print(
                    f"[WARNING] Skipping shortcut '{shortcut_name}' - "
                    f"operator signature collides with existing operator '{existing_op.name}'"
                )
                return

        # Create the skill that uses the heuristic
        shortcut_skill = ContrastiveShortcutSkill(
            heuristic=self.heuristic,
            operator=shortcut_operator,
            target_node_id=target_node_id,
            env=self.system.env,
            temperature=temperature,
        )

        # Add operator and skill to system components
        self.system.components.operators.add(shortcut_operator)
        self.system.components.skills.add(shortcut_skill)

        # Debug output for first few shortcuts
        if self._shortcut_count <= 5:
            print(f"[DEBUG] Added shortcut {shortcut_name}: {len(source_atoms)} precond atoms -> {len(target_atoms)} target atoms, targeting node {target_node_id}")
            print(f"  Preconditions: {lifted_source_atoms}")
            print(f"  Add effects: {add_effects}")
            print(f"  Delete effects: {delete_effects}")

        print(
            f"[INFO] Added shortcut '{shortcut_name}': "
            f"{len(source_atoms)} atoms -> {len(target_atoms)} atoms "
            f"(target_node={target_node_id})"
        )

    def add_shortcuts_from_training_data(
        self,
        training_data: Any,
        distance_threshold: float = 5.0,
        max_shortcuts: int = 50,
    ) -> None:
        """Add shortcuts based on trained heuristic estimates.

        Selects promising node pairs based on low estimated distances
        and adds them as shortcuts.

        Args:
            training_data: Training data containing node information
            distance_threshold: Maximum estimated distance to consider
            max_shortcuts: Maximum number of shortcuts to add
        """
        node_atoms = training_data.node_atoms
        node_states = training_data.node_states

        # Evaluate all pairs and sort by estimated distance
        candidates = []

        for source_id, source_atoms in node_atoms.items():
            # Get a representative state for the source node
            if source_id not in node_states or len(node_states[source_id]) == 0:
                continue
            source_state = node_states[source_id][0]

            for target_id, target_atoms in node_atoms.items():
                if source_id == target_id:
                    continue

                # Estimate distance using the heuristic
                estimated_dist = self.heuristic.latent_dist(source_state, target_id)

                if estimated_dist <= distance_threshold:
                    candidates.append(
                        (estimated_dist, source_id, target_id, source_atoms, target_atoms)
                    )

        # Sort by estimated distance (lowest first)
        candidates.sort(key=lambda x: x[0])

        # Add top candidates as shortcuts
        num_added = 0
        for est_dist, source_id, target_id, source_atoms, target_atoms in candidates[
            :max_shortcuts
        ]:
            self.add_shortcut(
                source_atoms=source_atoms,
                target_atoms=target_atoms,
                target_node_id=target_id,
                temperature=0.0,
            )
            num_added += 1

            if num_added >= max_shortcuts:
                break

        print(
            f"[INFO] Added {num_added} shortcuts from {len(candidates)} candidates "
            f"(distance threshold: {distance_threshold})"
        )
