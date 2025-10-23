"""Planning graph representation for improvisational TAMP."""

import heapq
import itertools
from dataclasses import dataclass, field

from relational_structs import GroundAtom, GroundOperator


@dataclass
class PlanningGraphNode:
    """Node in the planning graph representing a set of atoms."""

    atoms: frozenset[GroundAtom]
    id: int
    states: list = field(default_factory=list)  # Store multiple low-level states

    def __hash__(self) -> int:
        return hash(self.atoms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlanningGraphNode):
            return False
        return self.atoms == other.atoms


@dataclass
class PlanningGraphEdge:
    """Edge in the planning graph representing a transition."""

    source: PlanningGraphNode
    target: PlanningGraphNode
    operator: GroundOperator | None = None
    cost: float = float("inf")
    is_shortcut: bool = False
    shortcut_id: int | None = None  # Index into multi-policy for this shortcut

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.operator))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlanningGraphEdge):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.operator == other.operator
        )


class PlanningGraph:
    """Graph representation of a task plan."""

    def __init__(self) -> None:
        self.nodes: list[PlanningGraphNode] = []
        self.edges: list[PlanningGraphEdge] = []
        self.node_to_incoming_edges: dict[
            PlanningGraphNode, list[PlanningGraphEdge]
        ] = {}
        self.node_to_outgoing_edges: dict[
            PlanningGraphNode, list[PlanningGraphEdge]
        ] = {}
        self.node_map: dict[frozenset[GroundAtom], PlanningGraphNode] = {}
        self.goal_nodes: list[PlanningGraphNode] = []

    def add_node(self, atoms: set[GroundAtom]) -> PlanningGraphNode:
        """Add a node to the graph."""
        frozen_atoms = frozenset(atoms)
        assert frozen_atoms not in self.node_map
        node_id = len(self.nodes)
        node = PlanningGraphNode(frozen_atoms, node_id)
        self.nodes.append(node)
        self.node_map[frozen_atoms] = node
        self.node_to_incoming_edges[node] = []
        self.node_to_outgoing_edges[node] = []
        return node

    def add_edge(
        self,
        source: PlanningGraphNode,
        target: PlanningGraphNode,
        operator: GroundOperator | None = None,
        cost: float = float("inf"),
        is_shortcut: bool = False,
    ) -> PlanningGraphEdge:
        """Add an edge to the graph."""
        edge = PlanningGraphEdge(source, target, operator, cost, is_shortcut)
        self.edges.append(edge)
        self.node_to_incoming_edges[edge.target].append(edge)
        self.node_to_outgoing_edges[edge.source].append(edge)
        return edge

    def find_shortest_path(
        self, init_atoms: set[GroundAtom], goal: set[GroundAtom]
    ) -> list[PlanningGraphEdge]:
        """Find shortest path from initial node to goal node using standard Dijkstra."""
        if not self.nodes:
            return []

        initial_node = self.node_map[frozenset(init_atoms)]
        goal_nodes = [node for node in self.nodes if goal.issubset(node.atoms)]
        assert goal_nodes, "No goal node found"

        # Standard Dijkstra's algorithm
        distances: dict[PlanningGraphNode, float] = {initial_node: 0.0}
        previous: dict[PlanningGraphNode, tuple[PlanningGraphNode, PlanningGraphEdge] | None] = {
            initial_node: None
        }

        # Priority queue for Dijkstra's algorithm
        counter = itertools.count()
        queue: list[tuple[float, int, PlanningGraphNode]] = [
            (0, next(counter), initial_node)
        ]

        while queue:
            current_dist, _, current_node = heapq.heappop(queue)

            # Skip if we've already found a better path
            if current_dist > distances.get(current_node, float("inf")):
                continue

            # Check if we reached a goal
            if current_node in goal_nodes:
                # We can stop since Dijkstra guarantees this is optimal
                break

            # Check all outgoing edges
            for edge in [e for e in self.edges if e.source == current_node]:
                if edge.cost == float("inf"):
                    continue

                new_dist = current_dist + edge.cost

                # If we found a better path to the target, update
                if new_dist < distances.get(edge.target, float("inf")):
                    distances[edge.target] = new_dist
                    previous[edge.target] = (current_node, edge)
                    heapq.heappush(queue, (new_dist, next(counter), edge.target))

        # Find the best goal node
        reachable_goals = [g for g in goal_nodes if g in distances]
        assert reachable_goals, "No goal node reachable"
        best_goal = min(reachable_goals, key=lambda g: distances[g])

        # Reconstruct path
        path = []
        current_node = best_goal
        while previous[current_node] is not None:
            prev_node, edge = previous[current_node]
            path.append(edge)
            current_node = prev_node
        path.reverse()

        # Print path information
        total_cost = distances[best_goal]
        print(f"Shortest path's cost: {total_cost}")
        path_details = [
            f"{edge.source.id}->{edge.target.id} [cost: {edge.cost}]"
            for edge in path
        ]
        print(f"Path details: {' -> '.join(path_details)}")

        return path
