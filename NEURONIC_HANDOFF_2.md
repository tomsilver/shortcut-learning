# Neuronic Handoff - April 1, 2026

## What needs to happen on Neuronic

1. **Set up the environment** — clone/pull the latest code from `continuous-graduation` branch, install dependencies, make sure `slap_env` conda env is ready.

2. **Run PyBullet experiments with parallel PPO** — the main motivation for moving to Neuronic is more CPU cores. The parallel training (`num_parallel_workers > 0`) spawns one process per shortcut policy, and each process reconstructs its own PyBullet environment from scratch via `EnvFactory`. This avoids the "Not connected to physics server" error from pickling PyBullet envs. Request `--cpus-per-task=8` or more in SLURM to actually benefit from parallelism.

3. **Run gridworld sweeps** — ch5/ch6/ch7 sweep files are updated with `time=71:59:00`. These are CPU-only jobs.

## Key changes since last handoff

### Checkpointing & Resume
- Pipeline saves training data, heuristic, pruned data, and policies at each stage boundary
- Mid-training heuristic checkpoint every 100 epochs (overwrites single file)
- `resume_from=/path/to/previous/output` config option skips Stages 1-2, loads heuristic, runs Stages 3-6 fresh
- Previous results (training rounds, true distances, embeddings) are carried over to the new output

### Eval Fixes
- **Single-edge path bug fixed**: `approach.current_path.pop(0)` was consuming the only edge before `run_evaluation_episode_with_caching` could use it. Now uses `approach.best_eval_path` (a copy made before pop).
- **Stale goal atoms fixed**: `initial_node_id`, `goal_node_ids`, `initial_node_atoms`, `goal_node_atoms_list` are now set in `approach.reset()` BEFORE the `already_at_goal` early return, so diagnostics always show correct values.
- **Eval diagnostics**: Each eval episode prints start position, atoms, goal atoms, termination reason.

### Distance Estimation
- **Medoids removed**: All heuristics now use multi-state averaging (`node_distance_samples`, default 10) instead of single-medoid estimation. `_update_medoids()` calls removed from all training loops.
- This gave consistently better correlation in testing.

### Pruning Improvements
- **Success-weighted pruning**: `score = p * gain` where `p` is either empirical success rate (heuristic actor) or estimated PPO probability (multi_rl).
- For multi_rl pruning: `estimate_probability()` uses Brownian motion argument (`p_rr = e^(-d^2 / 2*max_steps)`) converted to PPO probability via step function.
- For heuristic-only pruning: `p` = empirical success rate from sliding window, gated by `num_reliability_trials` (now 10 in all configs).
- Smart rollouts uses its own empirical `p_rr` from rollout data, always applies PPO conversion (since it's always paired with multi_rl).

### Continuous Graduation
- **Cooldown replaces window clearing**: After graduation, the pair enters a cooldown of `k` samples instead of having its success window cleared. This preserves accurate success rates for pruning.

### Parallel PPO Training
- `num_parallel_workers` in policy config (default 0 = sequential, >0 = parallel)
- Uses `torch.multiprocessing.Pool` with `spawn` start method
- Each worker reconstructs its own environment via `EnvFactory` (picklable recipe: system_cls + system_kwargs)
- Workers save trained models to disk, main process loads them back
- Works with PyBullet (each worker creates its own physics server)
- For best results: set `--cpus-per-task` >= `num_parallel_workers` in SLURM

### Config Changes
- `num_reliability_trials: 10` in all configs
- `node_distance_samples: 10` (default in all heuristic dataclasses)
- `num_parallel_workers: 0` in policy section (set >0 to enable)
- `resume_from: null` at top level of all configs
- PyBullet configs: `load_data: true` with training data paths set
- Gridworld: `batch_size: 16`, `n_envs: 1` (matching paper)

## Important: Torch version differences
- Della has torch 2.10.0+cu128, Neuronic has torch 2.11.0+cu130
- These produce **systematically different** training outcomes on identical code + seed
- This is NOT a bug — it's RL seed sensitivity amplified by different RNG sequences across torch versions
- Evaluate across multiple seeds, don't draw conclusions from single runs

## File inventory of changes
- `pipeline_v2.py` — checkpointing, resume_from, system_cls/kwargs threading, use_multi_rl in prune
- `heuristic_{sac,dsac,cmd,crl}_v2.py` — multi-state averaging, success-weighted pruning, estimate_probability, graduation cooldown, checkpoint callback
- `heuristic_{none,rollouts,smart_rollouts,dqn_v2,sac_v3}.py` — **kwargs compatibility for train_one_round/prune, no-op save/load
- `multi_rl.py` — parallel training via EnvFactory + mp.Pool
- `rl.py` — num_parallel_workers, node_distance_samples config fields
- `gpu_parallel.py` — GPUParallelTrainer (CPU-parallel support added)
- `base.py` — goal/node atom fields set before early return in reset()
- `training.py` — eval diagnostics, best_eval_path fix
- `visualize_results.py` — shortcut quality from Stage 5.5, skipped episode diagnostics
- `slap_train_pipeline_v2.py` — passes output_dir, system_cls, system_kwargs to pipeline
