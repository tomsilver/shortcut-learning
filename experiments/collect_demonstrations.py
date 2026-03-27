"""Collect SLAP-style skill demonstrations from ObstacleTower for diverse (start, goal) pairs.

For each episode:
1. Reset the environment to a random configuration and build the TAMP planning graph.
2. BFS through the graph, executing skills, to collect one low-level state per
   reachable node.
3. Enumerate all (source, target) node pairs where target is reachable from source
   via skill-only edges.
4. Sample --pairs_per_episode of these pairs at random.
5. For each pair: reset the env to the source node's state, execute the skills along
   the BFS path while recording (obs, action) pairs, and save successful trajectories.

Each saved trajectory is a dict:
    observations  : list of raw env observations, length T+1
    actions       : list of actions,              length T
    start_atoms   : list[str] — atom strings for source node
    goal_atoms    : list[str] — atom strings for target node
    start_node_id : int
    target_node_id: int
    n_steps       : int (= T)
    episode_idx   : int
    episode_seed  : int

Usage:
    python experiments/collect_demonstrations.py
    python experiments/collect_demonstrations.py \\
        --num_episodes 500 --pairs_per_episode 8 \\
        --output_dir data/demonstrations/obstacle_tower
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import GraphInstance

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.policies.base import Policy, PolicyContext
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


# ---------------------------------------------------------------------------
# Minimal dummy policy — needed only to instantiate ImprovisationalTAMPApproach
# so we can call _create_planning_graph without touching the RL machinery.
# ---------------------------------------------------------------------------

class NullPolicy(Policy):
    """Does nothing. Only here to satisfy ImprovisationalTAMPApproach.__init__."""

    node_states: dict = {}

    def initialize(self, env: Any) -> None:  # noqa: D102
        pass

    def can_initiate(self) -> bool:  # noqa: D102
        return False

    def get_action(self, obs: Any) -> Any:  # noqa: D102
        return None

    def save(self, path: str) -> None:  # noqa: D102
        pass

    def load(self, path: str) -> None:  # noqa: D102
        pass


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _collect_node_states(
    system: GraphObstacleTowerTAMPSystem,
    graph: PlanningGraph,
    initial_obs: Any,
    max_steps_per_skill: int,
) -> dict[int, Any]:
    """BFS through the planning graph to collect one state per reachable node.

    Returns {node_id: raw_obs}.
    """
    _, initial_atoms, _ = system.perceiver.reset(initial_obs, {})
    initial_node = graph.node_map.get(frozenset(initial_atoms))
    if initial_node is None:
        return {}

    node_states: dict[int, Any] = {initial_node.id: initial_obs}
    queue: deque[tuple[PlanningGraphNode, Any]] = deque([(initial_node, initial_obs)])
    visited: set[int] = {initial_node.id}

    while queue:
        current_node, current_state = queue.popleft()

        for edge in graph.node_to_outgoing_edges.get(current_node, []):
            if edge.is_shortcut or edge.operator is None:
                continue
            if edge.target.id in visited:
                continue

            skill = next(
                (s for s in system.skills if s.can_execute(edge.operator)), None
            )
            if skill is None:
                continue

            obs, _ = system.env.reset_from_state(current_state)
            skill.reset(edge.operator)
            target_atoms_frozen = frozenset(edge.target.atoms)
            success = False

            for _ in range(max_steps_per_skill):
                try:
                    action = skill.get_action(obs)
                except Exception:
                    break
                if action is None:
                    break
                obs, _, _, _, _ = system.env.step(action)
                if frozenset(system.perceiver.step(obs)) == target_atoms_frozen:
                    success = True
                    break

            if success:
                visited.add(edge.target.id)
                node_states[edge.target.id] = obs
                queue.append((edge.target, obs))

    return node_states


def _find_all_reachable_pairs(
    graph: PlanningGraph,
    node_states: dict[int, Any],
) -> list[tuple[PlanningGraphNode, PlanningGraphNode, list[PlanningGraphEdge]]]:
    """Return all (source, target, skill_path) triples where:
    - both nodes have a known state,
    - source != target, and
    - target is reachable from source via skill-only edges.
    """
    result: list[
        tuple[PlanningGraphNode, PlanningGraphNode, list[PlanningGraphEdge]]
    ] = []

    for start_node in graph.nodes:
        if start_node.id not in node_states:
            continue

        # BFS from start_node over skill-only edges
        queue: deque[tuple[PlanningGraphNode, list[PlanningGraphEdge]]] = deque(
            [(start_node, [])]
        )
        visited: set[int] = {start_node.id}

        while queue:
            node, path = queue.popleft()

            # Every reachable node (other than start) with a known state is a
            # valid target.
            if node.id != start_node.id and node.id in node_states:
                result.append((start_node, node, path))

            for edge in graph.node_to_outgoing_edges.get(node, []):
                if edge.is_shortcut or edge.operator is None:
                    continue
                if edge.target.id not in visited:
                    visited.add(edge.target.id)
                    queue.append((edge.target, path + [edge]))

    return result


# ---------------------------------------------------------------------------
# Trajectory execution
# ---------------------------------------------------------------------------

def _execute_path_and_collect(
    system: GraphObstacleTowerTAMPSystem,
    path: list[PlanningGraphEdge],
    start_state: Any,
    max_steps_per_skill: int,
) -> tuple[list[Any], list[Any]] | None:
    """Execute a skill path from start_state, collecting (obs, action) pairs.

    Returns (observations, actions) with len(observations) == len(actions) + 1,
    or None if any edge fails.

    The path is executed as a single continuous rollout (no reset_from_state
    between edges), so the trajectory is physically coherent.
    """
    obs, _ = system.env.reset_from_state(start_state)
    all_obs: list[Any] = [obs]
    all_actions: list[Any] = []

    for edge in path:
        if edge.operator is None:
            return None

        skill = next(
            (s for s in system.skills if s.can_execute(edge.operator)), None
        )
        if skill is None:
            return None

        skill.reset(edge.operator)
        target_atoms_frozen = frozenset(edge.target.atoms)
        success = False

        for _ in range(max_steps_per_skill):
            try:
                action = skill.get_action(obs)
            except (AssertionError, Exception):
                return None
            if action is None:
                return None

            all_actions.append(action)
            obs, _, _, _, _ = system.env.step(action)
            all_obs.append(obs)

            if frozenset(system.perceiver.step(obs)) == target_atoms_frozen:
                success = True
                break

        if not success:
            return None

    return all_obs, all_actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect demonstrations from ObstacleTower."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000,
        help="Number of env resets (each gives ~pairs_per_episode trajectories).",
    )
    parser.add_argument(
        "--pairs_per_episode", type=int, default=10,
        help="Max (source, target) pairs to attempt per episode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str,
        default="data/demonstrations/obstacle_tower",
    )
    parser.add_argument("--num_obstacle_blocks", type=int, default=3)
    parser.add_argument(
        "--max_steps_per_skill", type=int, default=150,
        help="Max low-level steps to attempt each skill edge.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("Creating ObstacleTower system...")
    system = GraphObstacleTowerTAMPSystem.create_default(
        seed=args.seed,
        render_mode=None,
        num_obstacle_blocks=args.num_obstacle_blocks,
    )

    # We only need the approach for _create_planning_graph; NullPolicy is fine.
    approach = ImprovisationalTAMPApproach(
        system=system,
        policy=NullPolicy(seed=args.seed),
        seed=args.seed,
    )

    traj_count = 0
    total_attempted = 0
    total_succeeded = 0
    t_start = time.time()

    for episode_idx in range(args.num_episodes):
        episode_seed = int(rng.integers(0, 2**31))

        # --- Reset environment --------------------------------------------------
        obs, info = None, None
        for attempt in range(5):
            try:
                obs, info = system.reset(seed=episode_seed + attempt)
                objects, atoms, _ = system.perceiver.reset(obs, info)
                break
            except AssertionError as e:
                print(
                    f"  Episode {episode_idx}: reset failed "
                    f"(attempt {attempt + 1}/5): {e}"
                )
        if obs is None:
            print(f"  Episode {episode_idx}: all resets failed — skipping.")
            continue

        # --- Build planning graph ----------------------------------------------
        # Suppress the noisy per-node prints from _create_planning_graph by
        # redirecting them, or just tolerate them.
        approach.planning_graph = approach._create_planning_graph(objects, atoms)
        graph = approach.planning_graph
        n_nodes = len(graph.nodes)
        n_edges = len(graph.edges)
        print(
            f"\nEpisode {episode_idx}/{args.num_episodes}: "
            f"{n_nodes} nodes, {n_edges} edges"
        )

        # --- Collect one state per reachable node ------------------------------
        node_states = _collect_node_states(
            system=system,
            graph=graph,
            initial_obs=obs,
            max_steps_per_skill=args.max_steps_per_skill,
        )
        print(
            f"  States collected: {len(node_states)}/{n_nodes} nodes reachable"
        )

        if len(node_states) < 2:
            continue

        # --- Enumerate reachable (source, target) pairs ------------------------
        all_pairs = _find_all_reachable_pairs(graph, node_states)
        if not all_pairs:
            print("  No reachable pairs found.")
            continue
        print(f"  Reachable pairs: {len(all_pairs)}")

        # --- Sample and execute ------------------------------------------------
        n_sample = min(args.pairs_per_episode, len(all_pairs))
        chosen_indices = rng.choice(len(all_pairs), size=n_sample, replace=False)

        for idx in chosen_indices:
            source_node, target_node, path = all_pairs[idx]
            total_attempted += 1

            result = _execute_path_and_collect(
                system=system,
                path=path,
                start_state=node_states[source_node.id],
                max_steps_per_skill=args.max_steps_per_skill,
            )

            if result is None:
                continue

            observations, actions = result
            assert len(observations) == len(actions) + 1

            traj: dict[str, Any] = {
                "observations": observations,
                "actions": actions,
                "start_atoms": sorted(str(a) for a in source_node.atoms),
                "goal_atoms": sorted(str(a) for a in target_node.atoms),
                "start_node_id": source_node.id,
                "target_node_id": target_node.id,
                "n_steps": len(actions),
                "episode_idx": episode_idx,
                "episode_seed": episode_seed,
            }

            traj_path = output_dir / f"traj_{traj_count:06d}.pkl"
            with open(traj_path, "wb") as f:
                pickle.dump(traj, f)

            traj_count += 1
            total_succeeded += 1

        elapsed = time.time() - t_start
        print(
            f"  Saved so far: {traj_count} trajectories  "
            f"({total_succeeded}/{total_attempted} attempts succeeded)  "
            f"[{elapsed:.0f}s elapsed]"
        )

    # --- Write metadata --------------------------------------------------------
    metadata = {
        "num_episodes": args.num_episodes,
        "pairs_per_episode": args.pairs_per_episode,
        "seed": args.seed,
        "num_obstacle_blocks": args.num_obstacle_blocks,
        "max_steps_per_skill": args.max_steps_per_skill,
        "total_trajectories": traj_count,
        "total_attempted": total_attempted,
        "total_succeeded": total_succeeded,
        "success_rate": total_succeeded / max(total_attempted, 1),
        "elapsed_seconds": time.time() - t_start,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. {traj_count} trajectories saved to {output_dir}/")
    print(f"Success rate: {metadata['success_rate']:.1%}")
    print(f"Total time: {metadata['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
