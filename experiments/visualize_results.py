"""Visualization script for SLAP pipeline results.

Loads a PipelineResults pickle and generates plots:
1. Shortcut quality per round (success rates, avg steps)
2. Heuristic metrics per round (gain, estimated distance, sample counts)
3. Estimated vs true distance scatterplot
4. Pruned shortcuts displayed on the grid
5. Eval rollout distributions (shortcut-using vs not)
6. Eval trajectories on the grid

Usage:
    python visualize_results.py <results_dir>
    python visualize_results.py <results_dir> --output-dir /tmp/plots
"""

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> Any:
    """Load PipelineResults from a results directory."""
    results_path = results_dir / "results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with open(results_path, "rb") as f:
        return pickle.load(f)


def atoms_to_rc(atoms: list[str]) -> tuple[int, int] | None:
    """Parse a list of atom strings to extract (row, col)."""
    row, col = None, None
    for atom_str in atoms:
        m = re.search(r"InRow(\d+)", atom_str) or re.search(r"Row(\d+)", atom_str)
        if m:
            row = int(m.group(1))
        m = re.search(r"InCol(\d+)", atom_str) or re.search(r"Col(\d+)", atom_str)
        if m:
            col = int(m.group(1))
    if row is not None and col is not None:
        return (row, col)
    return None


def node_id_to_rc(
    node_id: int, node_atoms: dict[int, list[str]]
) -> tuple[int, int] | None:
    """Parse node_atoms strings to extract (row, col) for a node."""
    atoms = node_atoms.get(node_id)
    if atoms is None:
        return None
    return atoms_to_rc(atoms)


def shortcut_label(
    src: int, tgt: int, node_atoms: dict[int, list[str]]
) -> str:
    """Human-readable label like '(1,2)->(3,0)' for a shortcut."""
    src_rc = node_id_to_rc(src, node_atoms)
    tgt_rc = node_id_to_rc(tgt, node_atoms)
    s = f"({src_rc[0]},{src_rc[1]})" if src_rc else str(src)
    t = f"({tgt_rc[0]},{tgt_rc[1]})" if tgt_rc else str(tgt)
    return f"{s}\u2192{t}"


def _cell_center(row: int, col: int, grid_config: dict[str, Any]) -> tuple[float, float]:
    """Return the (x, y) center of a cell given its (row, col)."""
    cell_size = grid_config.get("cell_size", grid_config.get("num_states_per_cell", 1))
    x = (col + 0.5) * cell_size
    y = (row + 0.5) * cell_size
    return x, y


def draw_grid(
    ax: plt.Axes,
    grid_config: dict[str, Any],
    node_atoms: dict[int, list[str]] | None = None,
    draw_portals: bool = True,
) -> None:
    """Draw gridworld cells, labels, and portals onto an axes."""
    num_cells = grid_config["num_cells"]
    cell_size = grid_config.get("cell_size", grid_config.get("num_states_per_cell", 1))
    grid_size = num_cells * cell_size

    # Cell boundaries
    for i in range(num_cells + 1):
        pos = i * cell_size
        ax.axhline(pos, color="gray", linewidth=0.8, alpha=0.4)
        ax.axvline(pos, color="gray", linewidth=0.8, alpha=0.4)

    # Cell labels — show (row, col) and node id
    if node_atoms is not None:
        rc_to_nid: dict[tuple[int, int], int] = {}
        for nid in node_atoms:
            rc = node_id_to_rc(nid, node_atoms)
            if rc is not None:
                rc_to_nid[rc] = nid
        for row in range(num_cells):
            for col in range(num_cells):
                cx, cy = _cell_center(row, col, grid_config)
                nid = rc_to_nid.get((row, col))
                label = f"({row},{col})" if nid is None else f"({row},{col})\nn{nid}"
                ax.text(cx, cy, label, ha="center", va="center", fontsize=7, alpha=0.45)

    # Portals
    if draw_portals and "portal_positions" in grid_config:
        colors = ["red", "green", "blue", "orange", "purple", "brown"]
        for idx, (p1, p2) in enumerate(grid_config["portal_positions"]):
            c = colors[idx % len(colors)]
            ax.plot(p1[0], p1[1], "o", color=c, markersize=8, markeredgecolor="k",
                    markeredgewidth=1, zorder=5)
            ax.plot(p2[0], p2[1], "o", color=c, markersize=8, markeredgecolor="k",
                    markeredgewidth=1, zorder=5)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color=c, linewidth=1.2, alpha=0.5)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")


def _top_k_annotation(
    pairs: dict[tuple[int, int], float],
    node_atoms: dict[int, list[str]] | None,
    fmt: str = ".2f",
    k: int = 5,
    descending: bool = True,
) -> str:
    """Build a 'Top shortcuts' text box string from a {(src,tgt): value} dict."""
    ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=descending)[:k]
    lines = ["Top shortcuts:"]
    for (s, t), val in ranked:
        lbl = shortcut_label(s, t, node_atoms) if node_atoms else f"{s}\u2192{t}"
        lines.append(f"{lbl}: {val:{fmt}}")
    return "\n".join(lines)


def _episode_uses_shortcuts(ep: dict[str, Any]) -> bool:
    """Return True if the episode's optimal path contains a shortcut edge."""
    edges = ep.get("optimal_path_edges")
    if not edges:
        return False
    return any(e.get("is_shortcut", False) for e in edges)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_shortcut_quality_per_round(results: Any, save_dir: Path) -> None:
    """Per-round histograms of shortcut success rates and avg steps.

    Generates one PNG per round with two histograms side-by-side,
    plus labels for the top shortcuts.
    """
    rounds = results.training_rounds
    rounds_with_rollouts = [
        (i, r) for i, r in enumerate(rounds) if r.get("shortcut_rollouts")
    ]
    if not rounds_with_rollouts:
        print("[SKIP] No shortcut rollout data in training_rounds.")
        return

    out = save_dir / "shortcut_quality"
    out.mkdir(exist_ok=True)

    for round_idx, rd in rounds_with_rollouts:
        rollouts = rd["shortcut_rollouts"]
        success_rates = [sr["success_rate"] for sr in rollouts]
        avg_steps = [sr.get("avg_length", 0.0) for sr in rollouts]

        # Build per-shortcut dicts for top-K annotations
        sr_dict = {(sr["source_node"], sr["target_node"]): sr["success_rate"]
                   for sr in rollouts}
        steps_dict = {(sr["source_node"], sr["target_node"]): sr.get("avg_length", 0.0)
                      for sr in rollouts if sr.get("avg_length", 0.0) > 0}

        _, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Success rate histogram
        ax = axes[0]
        ax.hist(success_rates, bins=np.linspace(0, 1, 21), color="steelblue",
                edgecolor="k", linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Success Rate")
        ax.set_ylabel("Count")
        ax.set_title(f"Round {round_idx}: Shortcut Success Rates")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3, axis="y")
        ax.text(0.98, 0.95, _top_k_annotation(sr_dict, results.node_atoms, fmt=".0%"),
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        # Avg steps histogram
        ax = axes[1]
        steps_arr = np.array(avg_steps)
        nonzero = steps_arr[steps_arr > 0]
        if len(nonzero) > 0:
            ax.hist(nonzero, bins=20, color="coral", edgecolor="k",
                    linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Avg Steps (successful rollouts)")
        ax.set_ylabel("Count")
        ax.set_title(f"Round {round_idx}: Shortcut Avg Steps")
        ax.grid(True, alpha=0.3, axis="y")
        if steps_dict:
            ax.text(0.98, 0.95,
                    _top_k_annotation(steps_dict, results.node_atoms, fmt=".1f", descending=False),
                    transform=ax.transAxes, fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        fname = out / f"round_{round_idx}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[SAVED] {len(rounds_with_rollouts)} shortcut quality plots in {out}/")


def plot_heuristic_metrics_per_round(results: Any, save_dir: Path) -> None:
    """Per-round histograms of gain, estimated distance, and sample counts.

    Generates one PNG per round with up to 3 histogram panels.
    """
    rounds = results.training_rounds
    if not rounds:
        print("[SKIP] No training_rounds data.")
        return

    # Figure out which metrics exist across any round
    has_gains = any(r.get("gains") for r in rounds)
    has_est = any(r.get("estimated_distances") for r in rounds)
    has_ucb = any(r.get("ucb_sample_counts") for r in rounds)

    n_panels = sum([has_gains, has_est, has_ucb])
    if n_panels == 0:
        print("[SKIP] No heuristic metric data (gains/estimated_distances/ucb_sample_counts).")
        return

    out = save_dir / "heuristic_metrics"
    out.mkdir(exist_ok=True)

    for round_idx, rd in enumerate(rounds):
        # Collect available data for this round
        # Each entry: (title, pair_dict, xlabel, color, fmt, descending)
        panels: list[tuple[str, dict, str, str, str, bool]] = []
        if has_gains and rd.get("gains"):
            panels.append(("Gain", rd["gains"], "Gain", "mediumpurple", ".2f", True))
        if has_est and rd.get("estimated_distances"):
            panels.append(("Estimated Distance", rd["estimated_distances"],
                           "Estimated Distance", "seagreen", ".2f", False))
        if has_ucb and rd.get("ucb_sample_counts"):
            panels.append(("UCB Sample Count", rd["ucb_sample_counts"],
                           "Sample Count", "goldenrod", ".0f", True))

        if not panels:
            continue

        _, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
        if len(panels) == 1:
            axes = [axes]

        for ax, (title, pair_dict, xlabel, color, fmt, desc) in zip(axes, panels):
            values = list(pair_dict.values())
            ax.hist(values, bins=20, color=color, edgecolor="k",
                    linewidth=0.5, alpha=0.8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.set_title(f"Round {round_idx}: {title}")
            ax.grid(True, alpha=0.3, axis="y")
            # Top-K shortcuts annotation
            ax.text(0.98, 0.95,
                    _top_k_annotation(pair_dict, results.node_atoms, fmt=fmt, descending=desc),
                    transform=ax.transAxes, fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        fname = out / f"round_{round_idx}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[SAVED] heuristic metric plots in {out}/")


def _scatter_distances(
    ax: plt.Axes,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    is_pruned: np.ndarray,
    xlabel: str,
    ylabel: str,
    title_prefix: str,
) -> None:
    """Draw a distance scatter with y=x line, pruned coloring, and stats."""
    mask_np = ~is_pruned
    if mask_np.any():
        ax.scatter(x_arr[mask_np], y_arr[mask_np], alpha=0.5, s=40,
                   color="gray", edgecolors="k", linewidth=0.3, label="Not pruned")
    if is_pruned.any():
        ax.scatter(x_arr[is_pruned], y_arr[is_pruned], alpha=0.8, s=70,
                   color="tab:blue", edgecolors="k", linewidth=0.5, label="Pruned (kept)")

    lo = min(x_arr.min(), y_arr.min()) * 0.9
    hi = max(x_arr.max(), y_arr.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")

    corr = np.corrcoef(x_arr, y_arr)[0, 1] if len(x_arr) > 1 else float("nan")
    mae = np.mean(np.abs(y_arr - x_arr))
    ax.set_title(f"{title_prefix}\nCorr={corr:.3f}  MAE={mae:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_estimated_vs_true_distance(results: Any, save_dir: Path) -> None:
    """Scatter of estimated vs true distance AND estimated vs graph distance."""
    # Get estimated distances from the last training round
    est_dists: dict[tuple[int, int], float] | None = None
    for rd in reversed(results.training_rounds):
        if rd.get("estimated_distances"):
            est_dists = rd["estimated_distances"]
            break
    if not est_dists:
        print("[SKIP] No estimated_distances in any training round.")
        return

    true_dists = results.true_distances or {}
    graph_dists = results.graph_distances or {}
    pruned_set = set(results.pruned_shortcuts) if results.pruned_shortcuts else set()

    has_true = bool(set(true_dists) & set(est_dists))
    has_graph = bool(set(graph_dists) & set(est_dists))

    if not has_true and not has_graph:
        print("[SKIP] No overlapping keys for distance scatterplots.")
        return

    n_cols = sum([has_true, has_graph])
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]
    ax_idx = 0

    if has_true:
        keys = sorted(set(true_dists) & set(est_dists))
        true_arr = np.array([true_dists[k] for k in keys])
        est_arr = np.array([est_dists[k] for k in keys])
        is_pruned = np.array([k in pruned_set for k in keys])
        _scatter_distances(axes[ax_idx], true_arr, est_arr, is_pruned,
                           "True Distance", "Estimated Distance",
                           "Estimated vs True Distance")
        ax_idx += 1

    if has_graph:
        keys = sorted(set(graph_dists) & set(est_dists))
        graph_arr = np.array([graph_dists[k] for k in keys])
        est_arr = np.array([est_dists[k] for k in keys])
        is_pruned = np.array([k in pruned_set for k in keys])
        _scatter_distances(axes[ax_idx], graph_arr, est_arr, is_pruned,
                           "Graph Distance", "Estimated Distance",
                           "Estimated vs Graph Distance")

    plt.tight_layout()
    plt.savefig(save_dir / "distance_scatterplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_dir / 'distance_scatterplots.png'}")


def plot_pruned_shortcuts_on_grid(results: Any, save_dir: Path) -> None:
    """Draw pruned shortcuts as dotted arrows on the grid."""
    if not results.grid_config or not results.node_atoms:
        print("[SKIP] Missing grid_config or node_atoms for grid plot.")
        return
    pruned = results.pruned_shortcuts
    if not pruned:
        print("[SKIP] No pruned_shortcuts to display.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_grid(ax, results.grid_config, results.node_atoms, draw_portals=True)

    # Get last-round rollout data for labeling success rates
    sr_map: dict[tuple[int, int], float] = {}
    for rd in reversed(results.training_rounds):
        if rd.get("shortcut_rollouts"):
            for sr in rd["shortcut_rollouts"]:
                sr_map[(sr["source_node"], sr["target_node"])] = sr["success_rate"]
            break

    for src, tgt in pruned:
        src_rc = node_id_to_rc(src, results.node_atoms)
        tgt_rc = node_id_to_rc(tgt, results.node_atoms)
        if src_rc is None or tgt_rc is None:
            continue
        x0, y0 = _cell_center(src_rc[0], src_rc[1], results.grid_config)
        x1, y1 = _cell_center(tgt_rc[0], tgt_rc[1], results.grid_config)
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=15,
            linestyle="--", linewidth=1.8, color="tab:blue", alpha=0.8,
            connectionstyle="arc3,rad=0.15",
        )
        ax.add_patch(arrow)
        # Label with success rate
        sr_val = sr_map.get((src, tgt))
        if sr_val is not None:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"{sr_val:.0%}", fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_title(f"Pruned Shortcuts ({len(pruned)} kept)")
    plt.tight_layout()
    plt.savefig(save_dir / "pruned_shortcuts_on_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_dir / 'pruned_shortcuts_on_grid.png'}")


def plot_virtual_shortcuts_per_round(results: Any, save_dir: Path) -> None:
    """For each training round, draw the cumulative set of virtually added shortcuts."""
    if not results.grid_config or not results.node_atoms:
        print("[SKIP] Missing grid_config or node_atoms for virtual shortcuts plot.")
        return

    rounds_with_data = [
        (i, rd) for i, rd in enumerate(results.training_rounds)
        if rd.get("virtual_shortcuts")
    ]
    if not rounds_with_data:
        print("[SKIP] No virtual_shortcuts recorded in any training round.")
        return

    n = len(rounds_with_data)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 10), squeeze=False)

    for col, (i, rd) in enumerate(rounds_with_data):
        ax = axes[0][col]
        draw_grid(ax, results.grid_config, results.node_atoms, draw_portals=True)
        shortcuts = rd["virtual_shortcuts"]

        for src, tgt in shortcuts:
            src_rc = node_id_to_rc(src, results.node_atoms)
            tgt_rc = node_id_to_rc(tgt, results.node_atoms)
            if src_rc is None or tgt_rc is None:
                continue
            x0, y0 = _cell_center(src_rc[0], src_rc[1], results.grid_config)
            x1, y1 = _cell_center(tgt_rc[0], tgt_rc[1], results.grid_config)
            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle="-|>", mutation_scale=15,
                linestyle="--", linewidth=1.8, color="tab:orange", alpha=0.8,
                connectionstyle="arc3,rad=0.15",
            )
            ax.add_patch(arrow)

        ax.set_title(f"Round {i + 1} — {len(shortcuts)} virtual shortcuts")

    plt.suptitle("Cumulative virtual shortcuts per round", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "virtual_shortcuts_per_round.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_dir / 'virtual_shortcuts_per_round.png'}")


def _pure_planning_distance(
    ep: dict[str, Any],
    grid_config: dict[str, Any],
) -> float | None:
    """Estimate the pure-planning distance for an eval episode.

    Approximated as the Manhattan distance from the start position to the
    closest point on the goal cell.  Uses goal_node_atoms_list (atom strings)
    rather than goal_nodes (ephemeral IDs).
    """
    positions = ep.get("trajectory_positions")
    goal_atoms_list = ep.get("goal_node_atoms_list")
    print(f"Goal atoms: {goal_atoms_list}")
    if not positions or not goal_atoms_list:
        return None

    

    start = np.array(positions[0][:2])  # [x, y]
    cell_size = grid_config.get("cell_size", grid_config.get("num_states_per_cell", 1))

    best_dist = float("inf")
    for goal_atoms in goal_atoms_list:
        rc = atoms_to_rc(goal_atoms)
        if rc is None:
            continue
        # Goal cell bounding box
        x_lo = rc[1] * cell_size
        x_hi = (rc[1] + 1) * cell_size
        y_lo = rc[0] * cell_size
        y_hi = (rc[0] + 1) * cell_size
        # Closest point on the cell to start
        cx = np.clip(start[0], x_lo, x_hi)
        cy = np.clip(start[1], y_lo, y_hi)
        dist = abs(start[0] - cx) + abs(start[1] - cy)  # Manhattan distance
        # dist = np.sqrt((start[0] - cx) ** 2 + (start[1] - cy) ** 2)
        best_dist = min(best_dist, dist)

    return best_dist if best_dist < float("inf") else None


def plot_eval_distributions(results: Any, save_dir: Path) -> None:
    """Eval episode distributions, colored by whether shortcuts were used.

    Includes raw steps, reward, success rate, and advantage over pure planning.
    """
    episodes = results.eval_episodes
    if not episodes:
        print("[SKIP] No eval_episodes.")
        return

    uses_sc = np.array([_episode_uses_shortcuts(ep) for ep in episodes])
    steps = np.array([ep["true_steps"] for ep in episodes])
    rewards = np.array([ep["reward"] for ep in episodes])
    successes = np.array([ep["success"] for ep in episodes])

    # Compute advantage (pure_planning_dist - actual_steps)
    # Positive = shortcut saved steps vs pure planning
    can_compute_advantage = results.grid_config is not None
    advantages: np.ndarray | None = None
    adv_mask: np.ndarray | None = None  # which episodes have valid advantage
    if can_compute_advantage:
        adv_list: list[float | None] = []
        for i, ep in enumerate(episodes):
            pp = _pure_planning_distance(ep, results.grid_config)
            print(f"Episode {i}: pure_planning_dist={pp}, true_steps={ep['true_steps']}, success={ep['success']}")
            if pp is not None and ep["success"]:
                if pp - ep["true_steps"] > 0 and not _episode_uses_shortcuts(ep):
                    print(f"  >> Large positive advantage but no shortcuts used? Check episode data.")
                    print(ep)
                
                if ep["true_steps"] == 0:
                    print(f"  >> Warning: true_steps=0, cannot compute advantage. Check episode data.")
                    adv_list.append(None)
                else:   
                    adv_list.append(pp - ep["true_steps"])
                
            else:
                adv_list.append(None)
        adv_mask = np.array([a is not None for a in adv_list])
        advantages = np.array([a if a is not None else 0.0 for a in adv_list])

    has_advantage = advantages is not None and adv_mask is not None and adv_mask.any()
    n_cols = 4 if has_advantage else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5))

    # Steps histogram
    ax = axes[0]
    bins = np.linspace(0, steps.max() + 1, 25)
    if uses_sc.any():
        ax.hist(steps[uses_sc], bins=bins, alpha=0.7, color="tab:blue",
                label="With shortcuts", edgecolor="k", linewidth=0.5)
    if (~uses_sc).any():
        ax.hist(steps[~uses_sc], bins=bins, alpha=0.5, color="tab:gray",
                label="No shortcuts", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Steps"); ax.set_ylabel("Count")
    ax.set_title("Eval Episode Steps")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Reward histogram
    ax = axes[1]
    bins_r = np.linspace(rewards.min(), rewards.max() + 0.1, 25)
    if uses_sc.any():
        ax.hist(rewards[uses_sc], bins=bins_r, alpha=0.7, color="tab:blue",
                label="With shortcuts", edgecolor="k", linewidth=0.5)
    if (~uses_sc).any():
        ax.hist(rewards[~uses_sc], bins=bins_r, alpha=0.5, color="tab:gray",
                label="No shortcuts", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Reward"); ax.set_ylabel("Count")
    ax.set_title("Eval Episode Reward")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Success rate bar
    ax = axes[2]
    groups = {"With shortcuts": uses_sc, "No shortcuts": ~uses_sc}
    x_pos = 0
    for label, mask in groups.items():
        if mask.any():
            sr = successes[mask].mean()
            n = mask.sum()
            color = "tab:blue" if "With" in label else "tab:gray"
            ax.bar(x_pos, sr, width=0.6, color=color, edgecolor="k",
                   label=f"{label} (n={n})")
            ax.text(x_pos, sr + 0.02, f"{sr:.0%}", ha="center", fontsize=10)
            x_pos += 1
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate")
    ax.set_ylim(0, 1.15)
    ax.set_xticks([])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # Advantage histogram (pure planning steps - actual steps)
    if has_advantage:
        ax = axes[3]
        valid = adv_mask
        sc_valid = valid & uses_sc
        no_sc_valid = valid & ~uses_sc
        all_adv = advantages[valid]
        lo = all_adv.min() - 1
        hi = all_adv.max() + 1
        bins_a = np.linspace(lo, hi, 25)
        if sc_valid.any():
            ax.hist(advantages[sc_valid], bins=bins_a, alpha=0.7, color="tab:blue",
                    label="With shortcuts", edgecolor="k", linewidth=0.5)
        if no_sc_valid.any():
            ax.hist(advantages[no_sc_valid], bins=bins_a, alpha=0.5, color="tab:gray",
                    label="No shortcuts", edgecolor="k", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Advantage (planning dist \u2212 actual steps)")
        ax.set_ylabel("Count")
        ax.set_title("Advantage over Pure Planning\n(positive = saved steps)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        # Summary stats
        if sc_valid.any():
            mean_sc = advantages[sc_valid].mean()
            ax.text(0.98, 0.95, f"Shortcut mean: {mean_sc:+.1f}",
                    transform=ax.transAxes, fontsize=8, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_dir / "eval_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_dir / 'eval_distributions.png'}")


def plot_eval_trajectories(results: Any, save_dir: Path, max_plots: int = 20) -> None:
    """Plot each eval trajectory as a path on the grid."""
    episodes = results.eval_episodes
    if not episodes:
        print("[SKIP] No eval_episodes.")
        return
    if not results.grid_config or not results.node_atoms:
        print("[SKIP] Missing grid_config or node_atoms for trajectory plots.")
        return

    traj_dir = save_dir / "eval_trajectories"
    traj_dir.mkdir(exist_ok=True)

    num_cells = results.grid_config["num_cells"]
    cell_size = results.grid_config.get(
        "cell_size", results.grid_config.get("num_states_per_cell", 1)
    )

    for idx, ep in enumerate(episodes[:max_plots]):
        positions = ep.get("trajectory_positions")
        if not positions or len(positions) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        draw_grid(ax, results.grid_config, results.node_atoms, draw_portals=True)

        pos_arr = np.array(positions)
        xs, ys = pos_arr[:, 0], pos_arr[:, 1]

        # Highlight goal cell (use atoms, not ephemeral IDs)
        goal_atoms_list = ep.get("goal_node_atoms_list", [])
        for goal_atoms in goal_atoms_list:
            rc = atoms_to_rc(goal_atoms)
            if rc is not None:
                gx = rc[1] * cell_size
                gy = rc[0] * cell_size
                ax.add_patch(Rectangle(
                    (gx, gy), cell_size, cell_size,
                    facecolor="blue", alpha=0.12, edgecolor="blue",
                    linewidth=2, linestyle="--",
                ))

        # Color gradient by timestep
        colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))
        for i in range(len(xs) - 1):
            ax.plot(xs[i:i + 2], ys[i:i + 2], color=colors[i], linewidth=1.5, alpha=0.8)

        # Start / end markers
        ax.scatter(xs[0], ys[0], c="green", s=120, marker="o", edgecolors="k",
                   linewidth=1.5, label="Start", zorder=10)
        ax.scatter(xs[-1], ys[-1], c="red", s=120, marker="X", edgecolors="k",
                   linewidth=1.5, label="End", zorder=10)

        success_str = "Success" if ep["success"] else "Failure"
        sc_str = " [shortcut]" if _episode_uses_shortcuts(ep) else ""
        ax.set_title(
            f"Eval #{idx} ({success_str}{sc_str})  "
            f"Steps={ep['true_steps']}  Reward={ep['reward']:.1f}",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        plt.savefig(traj_dir / f"eval_trajectory_{idx}.png", dpi=120, bbox_inches="tight")
        plt.close()

    print(f"[SAVED] {min(max_plots, len(episodes))} trajectory plots in {traj_dir}/")


def plot_pruned_shortcut_rollouts(results: Any, save_dir: Path) -> None:
    """For each pruned shortcut, plot one example rollout trajectory on the grid."""
    pruned = results.pruned_shortcuts
    if not pruned:
        print("[SKIP] No pruned_shortcuts.")
        return
    if not results.grid_config or not results.node_atoms:
        print("[SKIP] Missing grid_config or node_atoms.")
        return

    # Find the last round that has shortcut_rollouts
    last_rollouts: list[dict[str, Any]] | None = None
    for rd in reversed(results.training_rounds):
        if rd.get("shortcut_rollouts"):
            last_rollouts = rd["shortcut_rollouts"]
            break
    if not last_rollouts:
        print("[SKIP] No shortcut_rollouts in any training round.")
        return

    # Index rollouts by (source, target)
    rollout_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for sr in last_rollouts:
        rollout_by_pair[(sr["source_node"], sr["target_node"])] = sr

    cell_size = results.grid_config.get(
        "cell_size", results.grid_config.get("num_states_per_cell", 1)
    )

    out = save_dir / "shortcut_rollouts"
    out.mkdir(exist_ok=True)
    count = 0

    for idx, (src, tgt) in enumerate(pruned):
        sr = rollout_by_pair.get((src, tgt))
        if sr is None:
            continue
        positions_list = sr.get("rollout_positions")
        if not positions_list or len(positions_list) == 0:
            continue

        # Take the first rollout trajectory
        positions = positions_list[0]
        if len(positions) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        draw_grid(ax, results.grid_config, results.node_atoms, draw_portals=True)

        # Highlight source and target cells
        src_rc = node_id_to_rc(src, results.node_atoms)
        tgt_rc = node_id_to_rc(tgt, results.node_atoms)
        if src_rc is not None:
            ax.add_patch(Rectangle(
                (src_rc[1] * cell_size, src_rc[0] * cell_size), cell_size, cell_size,
                facecolor="green", alpha=0.12, edgecolor="green",
                linewidth=2, linestyle="--",
            ))
        if tgt_rc is not None:
            ax.add_patch(Rectangle(
                (tgt_rc[1] * cell_size, tgt_rc[0] * cell_size), cell_size, cell_size,
                facecolor="blue", alpha=0.12, edgecolor="blue",
                linewidth=2, linestyle="--",
            ))

        pos_arr = np.array(positions)
        xs, ys = pos_arr[:, 0], pos_arr[:, 1]

        # Color gradient by timestep
        colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))
        for i in range(len(xs) - 1):
            ax.plot(xs[i:i + 2], ys[i:i + 2], color=colors[i], linewidth=1.5, alpha=0.8)

        ax.scatter(xs[0], ys[0], c="green", s=120, marker="o", edgecolors="k",
                   linewidth=1.5, label="Start", zorder=10)
        ax.scatter(xs[-1], ys[-1], c="red", s=120, marker="X", edgecolors="k",
                   linewidth=1.5, label="End", zorder=10)

        lbl = shortcut_label(src, tgt, results.node_atoms)
        sr_val = sr["success_rate"]
        avg_len = sr.get("avg_length", 0.0)
        ax.set_title(
            f"Shortcut {lbl}  (SR={sr_val:.0%}, avg_steps={avg_len:.1f})\n"
            f"Rollout: {len(positions)} steps",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        plt.savefig(out / f"shortcut_rollout_{idx}.png", dpi=120, bbox_inches="tight")
        plt.close()
        count += 1

    print(f"[SAVED] {count} shortcut rollout plots in {out}/")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: Any) -> None:
    """Print key summary stats to stdout."""
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    if results.config:
        print(f"  Config: {results.config.get('heuristic', {}).get('name', '?')}")
    if results.unique_shortcuts is not None:
        print(f"  Candidate shortcuts: {len(results.unique_shortcuts)}")
    if results.pruned_shortcuts is not None:
        print(f"  Pruned shortcuts:    {len(results.pruned_shortcuts)}")
    if results.training_rounds:
        print(f"  Training rounds:     {len(results.training_rounds)}")
    if results.avg_success_rate is not None:
        print(f"  Eval success rate:   {results.avg_success_rate:.2%}")
    if results.avg_steps is not None:
        print(f"  Eval avg steps:      {results.avg_steps:.1f}")
    if results.avg_reward is not None:
        print(f"  Eval avg reward:     {results.avg_reward:.2f}")
    if results.times:
        total = sum(results.times.values())
        print(f"  Total time:          {total:.1f}s")
        for k, v in results.times.items():
            print(f"    {k}: {v:.1f}s")
    print("=" * 60 + "\n")


def plot_loss_curves_per_round(results: Any, save_dir: Path) -> None:
    """Plot actor and critic loss curves for each training round."""
    rounds = results.training_rounds
    if not rounds:
        print("[SKIP] No training_rounds for loss curves.")
        return

    loss_dir = save_dir / "loss_curves"
    loss_dir.mkdir(parents=True, exist_ok=True)

    for i, rd in enumerate(rounds):
        critic_losses = rd.get("critic_losses", [])
        actor_losses = rd.get("actor_losses", [])
        if not critic_losses and not actor_losses:
            continue

        n_panels = sum([bool(critic_losses), bool(actor_losses)])
        if n_panels == 0:
            continue

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        panel = 0
        if critic_losses:
            axes[panel].plot(critic_losses, linewidth=0.8, color="tab:blue")
            axes[panel].set_title(f"Round {i} — Critic Loss")
            axes[panel].set_xlabel("Training step")
            axes[panel].set_ylabel("Loss")
            axes[panel].grid(True, alpha=0.3)
            panel += 1

        if actor_losses:
            axes[panel].plot(actor_losses, linewidth=0.8, color="tab:orange")
            axes[panel].set_title(f"Round {i} — Actor Loss")
            axes[panel].set_xlabel("Training step")
            axes[panel].set_ylabel("Loss")
            axes[panel].grid(True, alpha=0.3)
            panel += 1

        plt.tight_layout()
        plt.savefig(loss_dir / f"round_{i}.png", dpi=120, bbox_inches="tight")
        plt.close()

    print(f"[SAVED] Loss curves in {loss_dir}/")


# ---------------------------------------------------------------------------
# Latent space
# ---------------------------------------------------------------------------


def plot_latent_space(results: Any, save_dir: Path, method: str = "0-1") -> None:
    """Plot latent space embeddings: nodes as X markers, states as dots.

    Nodes and their corresponding states share a color.  Node positions are
    annotated with (row, col) when grid atoms are available.  Dimensionality
    reduction: first two dims when latent_dim <= 2, otherwise t-SNE (falls
    back to first two dims if sklearn is not installed).
    """
    emb = getattr(results, "latent_embeddings", None)
    if emb is None:
        print("[SKIP] latent_embeddings not available in results")
        return

    node_atoms: dict[int, list[str]] = results.node_atoms or {}
    node_ids = sorted(emb.keys())
    if not node_ids:
        return

    # Collect embeddings
    node_embs = np.array([emb[nid][0] for nid in node_ids])  # (n_nodes, D)

    state_embs_list: list[list[float]] = []
    state_node_idx: list[int] = []  # index into node_ids
    for idx, nid in enumerate(node_ids):
        for _, state_emb in emb[nid][1]:
            state_embs_list.append(state_emb)
            state_node_idx.append(idx)

    D = node_embs.shape[1]
    state_embs = (
        np.array(state_embs_list) if state_embs_list else np.zeros((0, D))
    )

    # 2-D projection
    if D > 2:
        if method == "tsne":
            from sklearn.manifold import TSNE  # noqa: PLC0415

            all_embs = (
                np.vstack([node_embs, state_embs])
                if len(state_embs) > 0
                else node_embs
            )
            perplexity = min(30, max(2, all_embs.shape[0] // 4))
            coords = TSNE(
                n_components=2, perplexity=perplexity, random_state=42
            ).fit_transform(all_embs)
            node_coords = coords[: len(node_ids)]
            state_coords = coords[len(node_ids) :]
            method_label = "t-SNE"
        elif method == "0-1":
            node_coords = node_embs[:, :2]
            state_coords = state_embs[:, :2] if len(state_embs) > 0 else state_embs
            method_label = "dims 0–1"
        else:
            raise ValueError(f"Unknown method for latent space projection: {method}")
    else:
        node_coords = node_embs
        state_coords = state_embs
        method_label = "embedding dims"

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(node_ids), 1)))
    color_map = {nid: colors[i] for i, nid in enumerate(node_ids)}

    fig, ax = plt.subplots(figsize=(10, 8))

    # States as semi-transparent dots
    for i, node_idx in enumerate(state_node_idx):
        nid = node_ids[node_idx]
        ax.scatter(
            state_coords[i, 0],
            state_coords[i, 1],
            color=color_map[nid],
            alpha=0.35,
            s=25,
            zorder=2,
        )

    # Nodes as X markers with (row,col) labels
    for i, nid in enumerate(node_ids):
        x, y = node_coords[i]
        ax.scatter(
            x, y,
            color=color_map[nid],
            marker="X",
            s=220,
            zorder=4,
            edgecolors="black",
            linewidths=0.8,
        )
        rc = node_id_to_rc(nid, node_atoms)
        label = f"({rc[0]},{rc[1]})" if rc else f"n{nid}"
        ax.annotate(
            label, (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="bold",
            zorder=5,
        )

    ax.set_xlabel(f"{method_label} dim 0")
    ax.set_ylabel(f"{method_label} dim 1")
    ax.set_title(
        f"Latent Space ({method_label})\nNodes = ✕   States = ●   (color = node)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = save_dir / "latent_space.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SLAP pipeline results")
    parser.add_argument("results_dir", type=str, help="Directory containing results.pkl")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output plots (default: <results_dir>/plots)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    print(f"Saving plots to:      {output_dir}\n")

    results = load_results(results_dir)
    print_summary(results)

    plot_shortcut_quality_per_round(results, output_dir)
    plot_heuristic_metrics_per_round(results, output_dir)
    plot_estimated_vs_true_distance(results, output_dir)
    plot_pruned_shortcuts_on_grid(results, output_dir)
    plot_virtual_shortcuts_per_round(results, output_dir)
    plot_eval_distributions(results, output_dir)
    plot_eval_trajectories(results, output_dir)
    plot_pruned_shortcut_rollouts(results, output_dir)
    plot_loss_curves_per_round(results, output_dir)
    plot_latent_space(results, output_dir, method="0-1")

    print("\n[DONE] All visualizations generated.")


if __name__ == "__main__":
    main()
