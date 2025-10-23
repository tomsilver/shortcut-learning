# V2 Architecture Overview

## Key Insight
The planning graph structure is **fixed per domain** - it doesn't change between episodes. Only the initial state, initial node, and goal change per episode.

## Data Structures

### PlanningGraphNode (in graph_utils.py)
```python
@dataclass
class PlanningGraphNode:
    atoms: frozenset[GroundAtom]
    id: int
    states: list  # NEW: Multiple low-level states per node
```

### PlanningGraphEdge (in graph_utils.py)
```python
@dataclass
class PlanningGraphEdge:
    source: PlanningGraphNode
    target: PlanningGraphNode
    operator: GroundOperator | None
    cost: float  # Single cost (no path-dependent costs)
    is_shortcut: bool
    shortcut_id: int | None  # Maps to multi-policy index
```

### ShortcutTrainingData (in training_data.py)
```python
@dataclass
class ShortcutTrainingData:
    shortcuts: list[tuple[PlanningGraphNode, PlanningGraphNode]]  # k pairs
    config: dict[str, Any]
```

## Pipeline Flow

### 1. INITIALIZATION
```python
system = create_system()
approach = SLAPApproachV2(system, config)
```

### 2. BUILD PLANNING GRAPH (once)
```python
obs, info = system.reset()
approach.reset(obs, info)  # Builds approach.planning_graph
```

### 3. COLLECT (once)
```python
training_data = collect_training_data_v2(approach, config)
```

**What happens:**
- For each node, collect m diverse low-level states
  - Execute paths to reach the node
  - Apply perturbations to get state diversity
  - Store in `node.states`
- Select k shortcut pairs randomly
- Return `ShortcutTrainingData` with the k pairs

**Result:**
- Each node has m states
- We have k shortcut pairs selected
- Total training examples: k × m

### 4. TRAIN (once)
```python
approach.train(training_data)
```

**What happens:**
- Train multi-RL policy on k shortcuts
  - For each shortcut i: (source_node, target_node)
  - Training data: all states in source_node.states → target_node.atoms
  - This gives m examples per shortcut
- After training, add k shortcut edges to planning graph
  - Each edge has shortcut_id = i
- Compute costs for each shortcut
  - Sample from training states
  - Execute policy and measure steps
  - Average over multiple samples
  - Store in edge.cost
- Store trained shortcut mapping in approach

**Result:**
- Multi-policy trained on k tasks
- Planning graph has k shortcut edges with costs
- Ready for evaluation

### 5. EVALUATE (many episodes)
```python
for episode in range(num_episodes):
    obs, info = system.reset()

    # Find initial node and goal
    atoms = system.perceiver.step(obs)
    initial_node = planning_graph.node_map[frozenset(atoms)]
    goal_node = ...  # Based on goal

    # Run Dijkstra (costs already computed!)
    path = planning_graph.find_shortest_path(atoms, goal)

    # Execute path
    for edge in path:
        if edge.is_shortcut:
            # Use multi-policy with shortcut_id
            policy_id = edge.shortcut_id
            use_policy(policy_id, target_atoms=edge.target.atoms)
        else:
            # Use regular skill
            execute_skill(edge.operator)
```

## Configuration

### collection/default.yaml
```yaml
states_per_node: 10  # m = number of states per node
perturbation_steps: 5  # random steps for diversity
max_shortcuts_per_graph: 100  # k = max shortcuts to train
```

## Benefits of V2

1. **More training data**: m examples per shortcut (instead of 1)
2. **Cleaner structure**: Nodes store states, shortcuts are just pairs
3. **One planning graph**: Built once, reused for all episodes
4. **Simple costs**: One cost per edge (no path-dependent complexity)
5. **Clear separation**: Collect → Train → Evaluate

## What Changes Between Episodes

- **Initial continuous state**: Different positions/velocities
- **Initial node**: Which abstract state we start in
- **Goal**: Which node we're trying to reach

## What Stays Fixed

- **Planning graph structure**: Same nodes and regular edges
- **Shortcut edges**: Same k shortcuts with same costs
- **Trained policies**: Same multi-policy for all episodes
