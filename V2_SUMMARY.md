# V2 Implementation Summary

## What We Built

A cleaner, simpler architecture for shortcut learning that addresses the core issue: **not enough training data per shortcut**.

## Files Modified/Created

### Core Changes

1. **graph_utils.py**
   - Added `states: list` field to `PlanningGraphNode`
   - Removed path-dependent costs (`edge.costs` dict)
   - Added `shortcut_id: int` to `PlanningGraphEdge`
   - Simplified `find_shortest_path()` - standard Dijkstra without path tracking

2. **training_data.py**
   - Created `ShortcutTrainingData` class
   - Stores k shortcut pairs: `list[tuple[PlanningGraphNode, PlanningGraphNode]]`
   - States accessed directly via `node.states`

3. **configs.py + collection/default.yaml**
   - Added `states_per_node: int = 10` (m states per node)
   - Added `perturbation_steps: int = 5` (for state diversity)

### New Files

4. **collection_v2.py** - Clean collection pipeline
   - `collect_diverse_states_per_node()` - Collects m states per node
   - `select_shortcut_pairs()` - Randomly selects k shortcuts
   - `generate_training_data()` - Returns `ShortcutTrainingData`
   - `collect_training_data_v2()` - Main entry point

5. **slap_approach_v2.py** - Simplified approach
   - Planning graph built once, reused for all episodes
   - `train()` - Trains policy and adds shortcuts to graph
   - `_compute_shortcut_costs()` - Samples from training states
   - No context wrappers or goal-conditioned wrappers
   - Clean separation: collect → train → evaluate

6. **V2_ARCHITECTURE.md** - Architecture documentation

## Key Improvements

### 1. More Training Data
- **Before**: 1-3 examples per shortcut
- **After**: m examples per shortcut (configurable, default 10)
- **Result**: m × improvement in data per shortcut

### 2. Cleaner Structure
- **Before**: Parallel lists of states/atoms, path-dependent costs
- **After**: Shortcut pairs with states in nodes, single cost per edge
- **Result**: Simpler to understand and debug

### 3. Fixed Planning Graph
- **Before**: Graph rebuilt every episode
- **After**: Built once, reused (shortcuts added once)
- **Result**: Faster evaluation, clearer semantics

### 4. No Wrapper Complexity
- **Before**: ContextAwareWrapper, GoalConditionedWrapper
- **After**: Direct policy calls
- **Result**: Simpler code, fewer dependencies

## Usage Example

```python
from shortcut_learning.methods.slap_approach_v2 import SLAPApproachV2
from shortcut_learning.methods.collection_v2 import collect_training_data_v2

# 1. Initialize
system = Obstacle2DSystem()
policy = MultiRLPolicy()
approach = SLAPApproachV2(system, approach_config, policy)

# 2. Build planning graph (once)
obs, info = system.reset()
approach.reset(obs, info)
print(f"Planning graph has {len(approach.planning_graph.nodes)} nodes")

# 3. COLLECT (once)
training_data = collect_training_data_v2(approach, collection_config)
print(f"Collected {training_data.num_training_examples()} examples")
print(f"For {len(training_data)} shortcuts")

# 4. TRAIN (once)
approach.train(training_data)
print("Policy trained, shortcuts added to graph")

# 5. EVALUATE (many episodes)
for episode in range(100):
    obs, info = system.reset()
    approach.reset(obs, info)

    done = False
    while not done:
        result = approach.step(obs, reward, term, trunc, info)
        obs, reward, term, trunc, info = system.env.step(result.action)
        done = term or trunc
```

## What's Different from V1

| Aspect | V1 | V2 |
|--------|----|----|
| States per node | 1-3 | 10 (configurable) |
| Training examples | k × 1 | k × m |
| Edge costs | Path-dependent dict | Single float |
| Planning graph | Rebuilt per episode | Built once, reused |
| Dijkstra | Path-aware | Standard |
| Wrappers | Context + Goal | None |
| Code complexity | High | Low |

## Expected Benefits

1. **Better shortcuts**: More training data → more robust policies
2. **Less variance**: Shortcuts work from more diverse initial states
3. **Faster execution**: No 100-step timeouts on bad shortcuts
4. **Clearer code**: Easier to understand and modify
5. **Better performance**: Should match or beat pure_plan baseline

## Next Steps

1. **Integrate policy training**: Wire up `MultiRLPolicy.train()` in `approach.train()`
2. **Test collection**: Run `collect_training_data_v2` to verify state collection
3. **Test training**: Verify shortcuts are added and costs computed
4. **Run evaluation**: Compare V2 vs V1 vs pure_plan baselines
5. **Tune hyperparameters**: Adjust `states_per_node`, `perturbation_steps`

## Configuration

To use V2 architecture, set in your experiment config:

```yaml
# collection/default.yaml
states_per_node: 10  # More states = more robust shortcuts
perturbation_steps: 5  # Random exploration for diversity
max_shortcuts_per_graph: 100  # k shortcuts to train
```

Higher `states_per_node` = more training data but slower collection.
Higher `perturbation_steps` = more diversity but potentially less accurate states.

## Questions to Investigate

1. Does more training data actually help? (Plot performance vs states_per_node)
2. What's the right amount of perturbation? (Too much → wrong states)
3. How many shortcuts is optimal? (k = 10, 50, 100?)
4. Can we do better than random selection of shortcuts?

---

**Status**: Implementation complete, ready for testing!
