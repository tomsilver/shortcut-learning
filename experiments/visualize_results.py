"""Visualization script for V4 distance heuristic training results.

This script loads results from a training run and generates visualizations including:
1. Training progress curves (when training history is available)
2. Distance matrix comparisons (heatmaps, scatterplots)
3. Rollout trajectory analysis
4. Summary statistics

Usage:
    python visualize_results.py <results_dir>

Example:
    python visualize_results.py /scratch/gpfs/TSILVER/de7281/shortcut_learning/outputs/heuristic_run_2025-12-04_23-59-56_job2758566
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec


def load_results(results_dir: Path) -> dict[str, Any]:
    """Load results from a training run.

    Args:
        results_dir: Directory containing the results

    Returns:
        Dictionary containing all results
    """
    results_path = results_dir / "heuristic_v4_results.pkl"

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    return results


def plot_training_history(results: dict[str, Any], save_dir: Path, show: bool = True) -> None:
    """Plot training progress curves.

    Plots total loss, alignment loss, uniformity loss, success rate,
    and average successful trajectory length over epochs.

    Args:
        results: Results dictionary
        save_dir: Directory to save plots
        show: Whether to display the plot interactively
    """
    # Check if training history is available
    if 'training_history' not in results:
        print("[WARNING] No training history found in results. Skipping training plots.")
        print("          To enable training plots, modify distance_heuristic_v4.py to save training history.")
        return

    history = results['training_history']

    # Create figure with 2x3 subplots
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    epochs = np.arange(len(history['total_loss']))

    # Plot 1: Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['total_loss'], linewidth=2, color='tab:blue')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Alignment Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['alignment_loss'], linewidth=2, color='tab:green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Alignment Loss', fontsize=12)
    ax2.set_title('Alignment Loss (L_align)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Uniformity Loss
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['uniformity_loss'], linewidth=2, color='tab:red')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Uniformity Loss', fontsize=12)
    ax3.set_title('Uniformity Loss (L_unif)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Success Rate
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, np.array(history['success_rate']) * 100, linewidth=2, color='tab:purple')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Success Rate (%)', fontsize=12)
    ax4.set_title('Policy Success Rate', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    # Plot 5: Average Successful Length
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history['avg_success_length'], linewidth=2, color='tab:orange')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Avg. Steps', fontsize=12)
    ax5.set_title('Average Successful Trajectory Length', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Accuracy
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, np.array(history['accuracy']) * 100, linewidth=2, color='tab:brown')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Accuracy (%)', fontsize=12)
    ax6.set_title('Embedding Accuracy', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 105])

    plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'training_history.pdf', bbox_inches='tight')
    print(f"[SAVED] Training history plots: {save_dir / 'training_history.png'}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_distance_matrices(results: dict[str, Any], save_dir: Path, show: bool = True) -> None:
    """Plot distance matrices as heatmaps and scatterplots.

    Visualizes true distances, graph distances, and learned distances
    as heatmaps, and compares them via scatterplots.

    Args:
        results: Results dictionary
        save_dir: Directory to save plots
        show: Whether to display the plot interactively
    """
    dist_eval = results['distance_evaluation']

    true_distances = dist_eval['true_distances']
    graph_distances = dist_eval['graph_distances']
    estimated_distances = np.maximum(dist_eval['estimated_distances'], 0) 
    # estimated_distances = (np.maximum(dist_eval['estimated_distances'], 0)) / np.max(dist_eval['estimated_distances']) * np.max(true_distances)
    print(true_distances)
    print(dist_eval['estimated_distances'])
    node_ids = dist_eval['node_ids']

    # Create figure for heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Common settings for heatmaps
    vmin = 0
    vmax = max(np.nanmax(true_distances[np.isfinite(true_distances)]),
               np.nanmax(estimated_distances))

    # Determine whether to show annotations (hide if grid is larger than 5x5)
    num_nodes = len(node_ids)
    show_annot = num_nodes <= 25  # 5x5 or smaller

    # Heatmap 1: True Distances
    sns.heatmap(true_distances, annot=show_annot, fmt='.1f', cmap='YlOrRd',
                xticklabels=node_ids, yticklabels=node_ids,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Distance'},
                ax=axes[0])
    axes[0].set_title('True Distances (Graph Search)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Target Node', fontsize=12)
    axes[0].set_ylabel('Source Node', fontsize=12)

    # Heatmap 2: Graph Distances
    sns.heatmap(graph_distances, annot=show_annot, fmt='.1f', cmap='YlOrRd',
                xticklabels=node_ids, yticklabels=node_ids,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Distance'},
                ax=axes[1])
    axes[1].set_title('Graph Distances (Planning Graph)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Target Node', fontsize=12)
    axes[1].set_ylabel('Source Node', fontsize=12)

    # Heatmap 3: Estimated Distances
    sns.heatmap(estimated_distances, annot=show_annot, fmt='.1f', cmap='YlOrRd',
                xticklabels=node_ids, yticklabels=node_ids,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Distance'},
                ax=axes[2])
    axes[2].set_title('Learned Distances (V4 Heuristic)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Target Node', fontsize=12)
    axes[2].set_ylabel('Source Node', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_dir / 'distance_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'distance_heatmaps.pdf', bbox_inches='tight')
    print(f"[SAVED] Distance heatmaps: {save_dir / 'distance_heatmaps.png'}")
    if show:
        plt.show()
    else:
        plt.close()

    # Create scatterplots comparing distances
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter out invalid distances
    valid_mask = np.isfinite(true_distances) & (true_distances > 0)
    valid_indices = np.where(valid_mask)

    true_flat = true_distances[valid_indices]
    graph_flat = graph_distances[valid_indices]
    estimated_flat = estimated_distances[valid_indices]

    # Scatterplot 1: Learned vs True
    axes[0].scatter(true_flat, estimated_flat, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

    # Add perfect prediction line
    max_val = max(np.max(true_flat), np.max(estimated_flat))
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Compute and display correlation
    corr = np.corrcoef(true_flat, estimated_flat)[0, 1]
    mae = np.mean(np.abs(estimated_flat - true_flat))
    rmse = np.sqrt(np.mean((estimated_flat - true_flat)**2))

    axes[0].set_xlabel('True Distance', fontsize=12)
    axes[0].set_ylabel('Learned Distance', fontsize=12)
    axes[0].set_title(f'Learned vs True Distances\nCorr={corr:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}',
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Scatterplot 2: Graph vs True
    axes[1].scatter(true_flat, graph_flat, alpha=0.6, s=80, color='green',
                    edgecolors='black', linewidth=0.5)

    # Add perfect prediction line
    max_val = max(np.max(true_flat), np.max(graph_flat))
    axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Match')

    # Compute correlation
    corr_graph = np.corrcoef(true_flat, graph_flat)[0, 1]
    mae_graph = np.mean(np.abs(graph_flat - true_flat))

    axes[1].set_xlabel('True Distance (Search)', fontsize=12)
    axes[1].set_ylabel('Graph Distance (Planning Graph)', fontsize=12)
    axes[1].set_title(f'Graph vs True Distances\nCorr={corr_graph:.3f}, MAE={mae_graph:.2f}',
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'distance_scatterplots.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'distance_scatterplots.pdf', bbox_inches='tight')
    print(f"[SAVED] Distance scatterplots: {save_dir / 'distance_scatterplots.png'}")
    if show:
        plt.show()
    else:
        plt.close()

    # Error heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    error_matrix = estimated_distances - true_distances
    error_matrix[~valid_mask] = np.nan

    max_error = np.nanmax(np.abs(error_matrix))
    sns.heatmap(error_matrix, annot=show_annot, fmt='.1f', cmap='RdBu_r', center=0,
                xticklabels=node_ids, yticklabels=node_ids,
                vmin=-max_error, vmax=max_error,
                cbar_kws={'label': 'Error (Learned - True)'},
                ax=ax)
    ax.set_title('Distance Estimation Error', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Node', fontsize=12)
    ax.set_ylabel('Source Node', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_dir / 'distance_error_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'distance_error_heatmap.pdf', bbox_inches='tight')
    print(f"[SAVED] Distance error heatmap: {save_dir / 'distance_error_heatmap.png'}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_rollout_analysis(results: dict[str, Any], save_dir: Path, show: bool = True) -> None:
    """Plot rollout trajectory analysis.

    Analyzes the relationship between trajectory length and true distance,
    showing policy efficiency.

    Args:
        results: Results dictionary
        save_dir: Directory to save plots
        show: Whether to display the plot interactively
    """
    rollout_eval = results['rollout_evaluation']
    rollout_results = rollout_eval['rollout_results']

    # Extract data
    trajectory_lengths = []
    true_lengths = []
    successes = []

    for result in rollout_results:
        trajectory_lengths.append(result['trajectory_length'])
        true_lengths.append(result['true_length'])
        successes.append(result['success'])

    trajectory_lengths = np.array(trajectory_lengths)
    true_lengths = np.array(true_lengths)
    successes = np.array(successes)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatterplot: Trajectory length vs True length
    colors = ['green' if s else 'red' for s in successes]
    axes[0].scatter(true_lengths, trajectory_lengths, c=colors, alpha=0.6, s=100,
                    edgecolors='black', linewidth=1)

    # Add perfect efficiency line (trajectory = true)
    max_val = max(np.max(true_lengths), np.max(trajectory_lengths))
    axes[0].plot([0, np.max(true_lengths)], [0, np.max(true_lengths)], 'b--', linewidth=2, label='Perfect Efficiency')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Success'),
        Patch(facecolor='red', edgecolor='black', label='Failure'),
        plt.Line2D([0], [0], color='b', linestyle='--', linewidth=2, label='Perfect Efficiency')
    ]
    axes[0].legend(handles=legend_elements, fontsize=10)

    # Compute efficiency for successful rollouts
    success_mask = successes == True
    if success_mask.any():
        efficiency = true_lengths[success_mask] / trajectory_lengths[success_mask]
        avg_efficiency = np.mean(efficiency)

        axes[0].set_title(f'Rollout Trajectory Analysis\nAvg. Efficiency (Success): {avg_efficiency:.2%}',
                          fontsize=14, fontweight='bold')
    else:
        axes[0].set_title('Rollout Trajectory Analysis', fontsize=14, fontweight='bold')

    axes[0].set_xlabel('True Distance (Optimal Steps)', fontsize=12)
    axes[0].set_ylabel('Trajectory Length (Actual Steps)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Histogram: Distribution of trajectory lengths
    success_lengths = trajectory_lengths[success_mask]
    failure_lengths = trajectory_lengths[~success_mask]

    bins = np.arange(0, max(trajectory_lengths) + 5, 5)
    axes[1].hist(success_lengths, bins=bins, alpha=0.7, color='green', label='Success', edgecolor='black')
    axes[1].hist(failure_lengths, bins=bins, alpha=0.7, color='red', label='Failure', edgecolor='black')

    axes[1].set_xlabel('Trajectory Length', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distribution of Trajectory Lengths', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'rollout_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'rollout_analysis.pdf', bbox_inches='tight')
    print(f"[SAVED] Rollout analysis: {save_dir / 'rollout_analysis.png'}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory_error_histogram(results: dict[str, Any], save_dir: Path, show: bool = True) -> None:
    """Plot histogram of trajectory length errors.

    Shows the distribution of (trajectory_length - true_length) for all rollouts.

    Args:
        results: Results dictionary
        save_dir: Directory to save plots
        show: Whether to display the plot interactively
    """
    rollout_eval = results['rollout_evaluation']
    rollout_results = rollout_eval['rollout_results']

    # Compute errors
    errors = []
    success_errors = []
    failure_errors = []

    for result in rollout_results:
        error = result['trajectory_length'] - result['true_length']
        errors.append(error)

        if result['success']:
            success_errors.append(error)
        else:
            failure_errors.append(error)

    errors = np.array(errors)
    success_errors = np.array(success_errors)
    failure_errors = np.array(failure_errors)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Overall histogram
    axes[0].hist(errors, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2,
                    label=f'Mean Error: {np.mean(errors):.1f}')
    axes[0].set_xlabel('Trajectory Length Error (Actual - Optimal)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of Trajectory Errors', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Success vs Failure
    if len(success_errors) > 0:
        axes[1].hist(success_errors, bins=15, alpha=0.7, color='green',
                     label=f'Success (n={len(success_errors)})', edgecolor='black')
    if len(failure_errors) > 0:
        axes[1].hist(failure_errors, bins=15, alpha=0.7, color='red',
                     label=f'Failure (n={len(failure_errors)})', edgecolor='black')

    axes[1].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].set_xlabel('Trajectory Length Error (Actual - Optimal)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Trajectory Errors by Outcome', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f'Overall Stats:\n'
    stats_text += f'Mean Error: {np.mean(errors):.2f}\n'
    stats_text += f'Median Error: {np.median(errors):.2f}\n'
    stats_text += f'Std Dev: {np.std(errors):.2f}'

    if len(success_errors) > 0:
        stats_text += f'\n\nSuccess Stats:\n'
        stats_text += f'Mean: {np.mean(success_errors):.2f}\n'
        stats_text += f'Median: {np.median(success_errors):.2f}'

    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_dir / 'trajectory_error_histogram.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'trajectory_error_histogram.pdf', bbox_inches='tight')
    print(f"[SAVED] Trajectory error histogram: {save_dir / 'trajectory_error_histogram.png'}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory_on_grid(results: dict[str, Any], save_dir: Path,
                           rollout_idx: int = 0, show: bool = True) -> None:
    """Plot a trajectory on the gridworld.

    Args:
        results: Results dictionary
        save_dir: Directory to save plots
        rollout_idx: Index of rollout to visualize
        show: Whether to display the plot interactively
    """
    rollout_results = results['rollout_evaluation']['rollout_results']

    if rollout_idx >= len(rollout_results):
        print(f"[WARNING] Rollout index {rollout_idx} out of range (max: {len(rollout_results)-1})")
        return

    rollout = rollout_results[rollout_idx]

    # Check if all_states is available
    if 'all_states' not in rollout:
        print("[WARNING] Trajectory states not available in results.")
        print("          To enable trajectory visualization, ensure rollout() saves all_states.")
        return

    all_states = np.array(rollout['all_states'])  # Shape: (T, 2)
    all_nodes = rollout.get('all_nodes', None)

    # Get grid configuration
    cfg = results['config']
    num_cells = cfg['num_cells']
    num_states_per_cell = cfg['num_states_per_cell']
    grid_size = num_cells * num_states_per_cell

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw grid lines
    for i in range(num_cells + 1):
        pos = i * num_states_per_cell
        ax.axhline(pos, color='gray', linewidth=1, alpha=0.3)
        ax.axvline(pos, color='gray', linewidth=1, alpha=0.3)

    # Draw cell labels
    for i in range(num_cells):
        for j in range(num_cells):
            center_x = (j + 0.5) * num_states_per_cell
            center_y = (i + 0.5) * num_states_per_cell
            ax.text(center_x, center_y, f'({i},{j})',
                   ha='center', va='center', fontsize=8, alpha=0.3)

    # Highlight goal cell if we have goal_atoms info
    goal_atoms = rollout.get('goal_atoms', None)
    if goal_atoms is not None:
        # Parse goal atoms to determine target cell
        # Atoms are like: InRow1(robot0), InCol2(robot0)
        goal_row = None
        goal_col = None

        for atom in goal_atoms:
            atom_str = str(atom)
            if "InRow" in atom_str or "Row" in atom_str:
                # Extract row number from atom like "InRow1(robot0)" or "Row1(robot0)"
                for i in range(num_cells):
                    if f"Row{i}" in atom_str or f"InRow{i}" in atom_str:
                        goal_row = i
                        break
            elif "InCol" in atom_str or "Col" in atom_str:
                # Extract col number from atom like "InCol2(robot0)" or "Col2(robot0)"
                for i in range(num_cells):
                    if f"Col{i}" in atom_str or f"InCol{i}" in atom_str:
                        goal_col = i
                        break

        if goal_row is not None and goal_col is not None:
            # Highlight the goal cell
            goal_cell_x_min = goal_col * num_states_per_cell
            goal_cell_x_max = (goal_col + 1) * num_states_per_cell
            goal_cell_y_min = goal_row * num_states_per_cell
            goal_cell_y_max = (goal_row + 1) * num_states_per_cell

            # Draw a shaded rectangle for the goal cell
            from matplotlib.patches import Rectangle
            goal_rect = Rectangle((goal_cell_x_min, goal_cell_y_min),
                                  num_states_per_cell, num_states_per_cell,
                                  linewidth=3, edgecolor='blue', facecolor='blue',
                                  alpha=0.15, linestyle='--', label='Goal Cell')
            ax.add_patch(goal_rect)

            # Mark the center of the goal cell
            goal_center_x = (goal_col + 0.5) * num_states_per_cell
            goal_center_y = (goal_row + 0.5) * num_states_per_cell
            ax.scatter(goal_center_x, goal_center_y, c='blue', s=300, marker='*',
                      edgecolors='black', linewidth=2, label='Goal Center', zorder=9)

    # Plot trajectory
    xs = all_states[:, 0]
    ys = all_states[:, 1]

    # Color trajectory by timestep (gradient)
    colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))

    for i in range(len(xs) - 1):
        ax.plot(xs[i:i+2], ys[i:i+2], color=colors[i], linewidth=2, alpha=0.7)

    # Mark start and end
    ax.scatter(xs[0], ys[0], c='green', s=200, marker='o',
              edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax.scatter(xs[-1], ys[-1], c='red', s=200, marker='X',
              edgecolors='black', linewidth=2, label='End', zorder=10)

    # Add timestep labels at key points
    step_interval = max(1, len(xs) // 10)
    for i in range(0, len(xs), step_interval):
        ax.text(xs[i], ys[i], str(i), fontsize=8, ha='center', va='bottom')

    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)

    success_str = "Success" if rollout['success'] else "Failure"
    title = f"Rollout #{rollout_idx} Trajectory ({success_str})\n"
    title += f"Length: {rollout['trajectory_length']}, Optimal: {rollout['true_length']:.0f}, "
    title += f"Error: {rollout['trajectory_length'] - rollout['true_length']:.0f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_dir / f'trajectory_rollout_{rollout_idx}.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / f'trajectory_rollout_{rollout_idx}.pdf', bbox_inches='tight')
    print(f"[SAVED] Trajectory visualization: {save_dir / f'trajectory_rollout_{rollout_idx}.png'}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_embeddings_tsne(results: dict[str, Any], heuristic_path: Path,
                         save_dir: Path, show: bool = True,
                         projection_method: str = 'raw') -> None:
    """Plot visualization of learned embeddings.

    Loads the trained heuristic and visualizes state and node embeddings
    in 2D using various projection methods.

    Args:
        results: Results dictionary
        heuristic_path: Path to saved heuristic pickle file
        save_dir: Directory to save plots
        show: Whether to display the plot interactively
        projection_method: Method for 2D projection. Options:
            - 'raw': Use first 2 embedding dimensions (default)
            - 'tsne': Use t-SNE dimensionality reduction
            - 'umap': Use UMAP dimensionality reduction
    """
    try:
        import torch
        if projection_method == 'tsne':
            from sklearn.manifold import TSNE
        elif projection_method == 'umap':
            try:
                import umap
            except ImportError:
                print("[ERROR] umap-learn not installed. Install with: pip install umap-learn")
                print("        Falling back to raw embeddings.")
                projection_method = 'raw'
    except ImportError:
        print("[ERROR] torch not available. Cannot plot embeddings.")
        return

    # Load the heuristic
    print(f"[INFO] Loading heuristic from {heuristic_path}...")

    # Check if it's a directory (new format) or a pickle file (old format)
    if heuristic_path.is_dir():
        # New format: directory with separate files
        try:
            from tamp_improv.approaches.improvisational.distance_heuristic_v4 import (
                DistanceHeuristicV4, StateEncoder
            )
            import torch
            import torch.nn as nn

            # Load config
            with open(heuristic_path / 'config.pkl', 'rb') as f:
                config = pickle.load(f)

            # Load encoders
            state_dim = 2  # For gridworld_fixed
            s_encoder = StateEncoder(
                state_dim=state_dim,
                latent_dim=config.latent_dim,
                hidden_dims=config.hidden_dims or [64, 64],
            )
            s_encoder.load_state_dict(torch.load(heuristic_path / 's_encoder.pt', map_location='cpu'))
            s_encoder.eval()

            g_encoder = torch.load(heuristic_path / 'g_encoder.pt', map_location='cpu')

            # Create a minimal object to hold the encoders
            class HeuristicWrapper:
                def __init__(self, s_enc, g_enc):
                    self.s_encoder = s_enc
                    self.g_encoder = g_enc
                    self.device = 'cpu'

            heuristic = HeuristicWrapper(s_encoder, g_encoder)

        except Exception as e:
            print(f"[ERROR] Could not load heuristic from directory: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Old format: single pickle file
        try:
            with open(heuristic_path, 'rb') as f:
                heuristic = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Could not load heuristic: {e}")
            return

        # Check if heuristic has encoders
        if not hasattr(heuristic, 's_encoder') or not hasattr(heuristic, 'g_encoder'):
            print("[ERROR] Heuristic does not have s_encoder or g_encoder attributes.")
            return

        if heuristic.s_encoder is None or heuristic.g_encoder is None:
            print("[ERROR] Encoders are None. Make sure heuristic was trained.")
            return

    # Get rollout states to embed
    rollout_results = results['rollout_evaluation']['rollout_results']

    # Collect all states from all rollouts
    all_states_list = []
    all_positions = []  # (x, y) positions

    for rollout in rollout_results:
        if 'all_states' in rollout:
            states = np.array(rollout['all_states'])
            all_states_list.extend(states)
            all_positions.extend(states)

    if len(all_states_list) == 0:
        print("[WARNING] No trajectory states available for embedding visualization.")
        return

    all_states_array = np.array(all_states_list)  # Shape: (N, 2)
    all_positions = np.array(all_positions)  # Shape: (N, 2)

    print(f"[INFO] Embedding {len(all_states_array)} states...")

    # Encode states
    heuristic.s_encoder.eval()
    with torch.no_grad():
        states_tensor = torch.FloatTensor(all_states_array).to(heuristic.device)
        state_embeddings = heuristic.s_encoder(states_tensor).cpu().numpy()  # Shape: (N, k)

    # Get node embeddings
    with torch.no_grad():
        # g_encoder is a Parameter tensor, need to convert properly
        if isinstance(heuristic.g_encoder, torch.nn.Parameter):
            node_embeddings = heuristic.g_encoder.data.cpu().numpy()  # Shape: (num_nodes, k)
        else:
            node_embeddings = heuristic.g_encoder.cpu().numpy()  # Shape: (num_nodes, k)

    # Parse node_atoms to get cell coordinates for labeling
    node_atoms = results['training_data'].get('node_atoms', {})
    node_labels = {}  # Dict[int, str] mapping node_id to "(row, col)"

    for node_id, atoms in node_atoms.items():
        # Parse atoms to find InRow and InCol predicates
        row = None
        col = None
        for atom in atoms:
            atom_str = str(atom)
            # Look for InRow{i} or Row{i}
            if "InRow" in atom_str or "Row" in atom_str:
                for i in range(100):  # Assume max 100 cells
                    if f"Row{i}" in atom_str or f"InRow{i}" in atom_str:
                        row = i
                        break
            # Look for InCol{j} or Col{j}
            elif "InCol" in atom_str or "Col" in atom_str:
                for i in range(100):
                    if f"Col{i}" in atom_str or f"InCol{i}" in atom_str:
                        col = i
                        break

        if row is not None and col is not None:
            node_labels[node_id] = f"({row},{col})"
        else:
            node_labels[node_id] = f"N{node_id}"

    print(f"[INFO] Parsed node labels: {node_labels}")

    # Debug: Print embedding statistics (before normalization)
    print(f"[DEBUG] State embeddings shape: {state_embeddings.shape}")
    print(f"[DEBUG] State embeddings mean: {np.mean(state_embeddings, axis=0)[:5]}...")  # First 5 dims
    print(f"[DEBUG] State embeddings std: {np.std(state_embeddings, axis=0)[:5]}...")
    print(f"[DEBUG] Node embeddings shape: {node_embeddings.shape}")
    print(f"[DEBUG] Node embeddings (before normalization):")
    for i, node_emb in enumerate(node_embeddings):
        print(f"  Node {i}: [{node_emb[0]:.4f}, {node_emb[1]:.4f}, ...]")
    print(f"[DEBUG] Node embedding mean: {np.mean(node_embeddings, axis=0)[:5]}...")
    print(f"[DEBUG] Node embedding std across nodes: {np.std(node_embeddings, axis=0)[:5]}...")

    # Normalize embeddings (as done in contrastive loss)
    print(f"\n[INFO] Normalizing embeddings (L2 norm)...")
    state_embeddings = state_embeddings / (np.linalg.norm(state_embeddings, axis=1, keepdims=True) + 1e-8)
    node_embeddings = node_embeddings / (np.linalg.norm(node_embeddings, axis=1, keepdims=True) + 1e-8)

    # Debug: Print after normalization
    print(f"[DEBUG] Node embeddings (after normalization):")
    for i, node_emb in enumerate(node_embeddings):
        print(f"  Node {i}: [{node_emb[0]:.4f}, {node_emb[1]:.4f}, ...] (norm={np.linalg.norm(node_emb):.4f})")
    print(f"[DEBUG] Node embedding std across nodes (after norm): {np.std(node_embeddings, axis=0)[:5]}...")

    # Project to 2D
    all_embeddings = np.vstack([state_embeddings, node_embeddings])  # Shape: (N + num_nodes, k)

    if projection_method == 'tsne':
        # Use t-SNE for dimensionality reduction
        print(f"[INFO] Running t-SNE on {all_embeddings.shape[0]} embeddings...")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # Split back into states and nodes
        state_embeddings_2d = embeddings_2d[:len(state_embeddings)]
        node_embeddings_2d = embeddings_2d[len(state_embeddings):]
        method_name = "t-SNE"

    elif projection_method == 'umap':
        # Use UMAP for dimensionality reduction
        print(f"[INFO] Running UMAP on {all_embeddings.shape[0]} embeddings...")
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(all_embeddings)

        # Split back into states and nodes
        state_embeddings_2d = embeddings_2d[:len(state_embeddings)]
        node_embeddings_2d = embeddings_2d[len(state_embeddings):]
        method_name = "UMAP"

    else:  # 'raw' or default
        # Just use first 2 dimensions of embeddings
        print(f"[INFO] Using first 2 embedding dimensions (out of {state_embeddings.shape[1]})...")
        state_embeddings_2d = state_embeddings[:, :2]
        node_embeddings_2d = node_embeddings[:, :2]
        method_name = "Embedding Dims 0-1"

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Color by X position
    scatter1 = axes[0].scatter(state_embeddings_2d[:, 0], state_embeddings_2d[:, 1],
                               c=all_positions[:, 0], cmap='viridis', alpha=0.6, s=20)
    axes[0].scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                   label='Node Embeddings', alpha=0.9)
    # Add text labels for nodes
    for node_id, label in node_labels.items():
        if node_id < len(node_embeddings_2d):
            axes[0].annotate(label, (node_embeddings_2d[node_id, 0], node_embeddings_2d[node_id, 1]),
                           textcoords="offset points", xytext=(0, 10), ha='center',
                           fontsize=8, fontweight='bold', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    plt.colorbar(scatter1, ax=axes[0], label='X Position')
    axes[0].set_xlabel(f'{method_name} Dimension 1', fontsize=12)
    axes[0].set_ylabel(f'{method_name} Dimension 2', fontsize=12)
    axes[0].set_title(f'Embeddings ({method_name}) - Colored by X Position', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Color by Y position
    scatter2 = axes[1].scatter(state_embeddings_2d[:, 0], state_embeddings_2d[:, 1],
                               c=all_positions[:, 1], cmap='plasma', alpha=0.6, s=20)
    axes[1].scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                   label='Node Embeddings', alpha=0.9)
    # Add text labels for nodes
    for node_id, label in node_labels.items():
        if node_id < len(node_embeddings_2d):
            axes[1].annotate(label, (node_embeddings_2d[node_id, 0], node_embeddings_2d[node_id, 1]),
                           textcoords="offset points", xytext=(0, 10), ha='center',
                           fontsize=8, fontweight='bold', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    plt.colorbar(scatter2, ax=axes[1], label='Y Position')
    axes[1].set_xlabel(f'{method_name} Dimension 1', fontsize=12)
    axes[1].set_ylabel(f'{method_name} Dimension 2', fontsize=12)
    axes[1].set_title(f'Embeddings ({method_name}) - Colored by Y Position', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Color by distance from origin
    distances = np.sqrt(all_positions[:, 0]**2 + all_positions[:, 1]**2)
    scatter3 = axes[2].scatter(state_embeddings_2d[:, 0], state_embeddings_2d[:, 1],
                               c=distances, cmap='coolwarm', alpha=0.6, s=20)
    axes[2].scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                   label='Node Embeddings', alpha=0.9)
    # Add text labels for nodes
    for node_id, label in node_labels.items():
        if node_id < len(node_embeddings_2d):
            axes[2].annotate(label, (node_embeddings_2d[node_id, 0], node_embeddings_2d[node_id, 1]),
                           textcoords="offset points", xytext=(0, 10), ha='center',
                           fontsize=8, fontweight='bold', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    plt.colorbar(scatter3, ax=axes[2], label='Distance from Origin')
    axes[2].set_xlabel(f'{method_name} Dimension 1', fontsize=12)
    axes[2].set_ylabel(f'{method_name} Dimension 2', fontsize=12)
    axes[2].set_title(f'Embeddings ({method_name}) - Colored by Distance', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate filename based on projection method
    if projection_method == 'tsne':
        filename = 'embeddings_tsne.png'
        filename_pdf = 'embeddings_tsne.pdf'
    elif projection_method == 'umap':
        filename = 'embeddings_umap.png'
        filename_pdf = 'embeddings_umap.pdf'
    else:
        filename = 'embeddings_2d.png'
        filename_pdf = 'embeddings_2d.pdf'

    plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / filename_pdf, bbox_inches='tight')
    print(f"[SAVED] Embeddings visualization: {save_dir / filename}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_gridworld_environment(results: dict[str, Any], save_dir: Path, show: bool = True) -> None:
    """Plot the gridworld environment with cells and portals.

    Shows the grid structure with:
    - Each cell labeled by its node ID
    - Portal positions marked with dots
    - Portal pairs connected with dotted lines

    Args:
        results: Results dictionary
        save_dir: Directory to save plots
        show: Whether to display the plot interactively
    """
    # Extract environment info
    env_info = results.get('environment', {})
    portal_positions = env_info.get('portal_positions', None)
    num_cells = env_info.get('num_cells', None)
    num_states_per_cell = env_info.get('num_states_per_cell', None)

    if num_cells is None or num_states_per_cell is None:
        print("[WARNING] Environment info not available. Skipping environment plot.")
        return

    grid_size = num_cells * num_states_per_cell

    # Parse node_atoms to get cell coordinates for each node
    node_atoms = results['training_data'].get('node_atoms', {})
    node_cells = {}  # Dict[int, Tuple[int, int]] mapping node_id to (row, col)

    for node_id, atoms in node_atoms.items():
        row = None
        col = None
        for atom in atoms:
            atom_str = str(atom)
            if "InRow" in atom_str or "Row" in atom_str:
                for i in range(100):
                    if f"Row{i}" in atom_str or f"InRow{i}" in atom_str:
                        row = i
                        break
            elif "InCol" in atom_str or "Col" in atom_str:
                for i in range(100):
                    if f"Col{i}" in atom_str or f"InCol{i}" in atom_str:
                        col = i
                        break

        if row is not None and col is not None:
            node_cells[node_id] = (row, col)

    # Create figure
    _, ax = plt.subplots(figsize=(10, 10))

    # Draw grid cells
    for row in range(num_cells):
        for col in range(num_cells):
            # Draw cell boundaries
            x_start = col * num_states_per_cell
            y_start = row * num_states_per_cell

            # Draw rectangle for cell
            rect = plt.Rectangle((x_start, y_start), num_states_per_cell, num_states_per_cell,
                                fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            # Find node_id for this cell and label it
            cell_center_x = x_start + num_states_per_cell / 2
            cell_center_y = y_start + num_states_per_cell / 2

            # Find node ID for this cell
            node_id = None
            for nid, (r, c) in node_cells.items():
                if r == row and c == col:
                    node_id = nid
                    break

            if node_id is not None:
                ax.text(cell_center_x, cell_center_y, f"{node_id}",
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

    # Draw portals if available
    if portal_positions is not None and len(portal_positions) > 0:
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']

        for idx, (pos1, pos2) in enumerate(portal_positions):
            color = colors[idx % len(colors)]

            # Draw portal dots
            ax.plot(pos1[0], pos1[1], 'o', color=color, markersize=15,
                   markeredgecolor='black', markeredgewidth=2, label=f'Portal {idx+1}')
            ax.plot(pos2[0], pos2[1], 'o', color=color, markersize=15,
                   markeredgecolor='black', markeredgewidth=2)

            # Draw connecting line
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                   linestyle='--', color=color, linewidth=2, alpha=0.7)

    # Set axis properties
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Gridworld Environment ({num_cells}x{num_cells} cells, {num_states_per_cell}x{num_states_per_cell} states/cell)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    if portal_positions is not None and len(portal_positions) > 0:
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save
    filename = 'gridworld_environment.png'
    filename_pdf = 'gridworld_environment.pdf'
    plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / filename_pdf, bbox_inches='tight')
    print(f"[SAVED] Gridworld environment: {save_dir / filename}")

    if show:
        plt.show()
    else:
        plt.close()


def print_summary_statistics(results: dict[str, Any]) -> None:
    """Print summary statistics from the results.

    Args:
        results: Results dictionary
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Configuration
    print("\n[Configuration]")
    cfg = results['config']
    print(f"  Environment: {cfg['env_name']}")
    print(f"  Grid size: {cfg['num_cells']}x{cfg['num_cells']}")
    print(f"  Latent dimension: {cfg['latent_dim']}")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Training epochs: {cfg['num_epochs']}")
    print(f"  Trajectories per epoch: {cfg['trajectories_per_epoch']}")

    # Training data
    print("\n[Training Data]")
    train_data = results['training_data']
    print(f"  State-node pairs: {train_data['num_state_node_pairs']}")
    print(f"  Number of nodes: {train_data['num_nodes']}")
    print(f"  Number of edges: {train_data['num_edges']}")
    print(f"  Number of shortcuts: {train_data['num_shortcuts']}")

    # Rollout evaluation
    print("\n[Rollout Evaluation]")
    rollout_eval = results['rollout_evaluation']
    print(f"  Success rate: {rollout_eval['success_rate']:.2f}%")
    print(f"  Avg trajectory length: {rollout_eval['avg_length']:.1f}")
    print(f"  Avg successful length: {rollout_eval['avg_success_length']:.1f}")
    print(f"  Number of rollouts: {len(rollout_eval['rollout_results'])}")

    # Distance evaluation
    print("\n[Distance Evaluation]")
    dist_eval = results['distance_evaluation']
    print(f"  MAE (Mean Absolute Error): {dist_eval['mae']:.3f}")
    print(f"  RMSE (Root Mean Squared Error): {dist_eval['rmse']:.3f}")
    print(f"  Correlation: {dist_eval['correlation']:.3f}")

    # Compute efficiency metrics
    rollout_results = rollout_eval['rollout_results']
    successes = [r['success'] for r in rollout_results]
    if any(successes):
        efficiencies = [
            r['true_length'] / r['trajectory_length']
            for r in rollout_results if r['success']
        ]
        print(f"  Avg policy efficiency (success): {np.mean(efficiencies):.2%}")
        print(f"  Min policy efficiency: {np.min(efficiencies):.2%}")
        print(f"  Max policy efficiency: {np.max(efficiencies):.2%}")

    print("\n" + "="*80 + "\n")


def main():
    """Main function to generate all visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize V4 distance heuristic training results"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to directory containing results (heuristic_v4_results.pkl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as results_dir)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (default: False, just save)"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open saved PNG files with default viewer (works in VS Code)"
    )

    args = parser.parse_args()

    # Configure matplotlib backend for interactive display if requested
    if args.show:
        # Try to use TkAgg backend for interactive display
        # If that fails, fall back to the default
        try:
            matplotlib.use('TkAgg')
        except Exception:
            try:
                matplotlib.use('Qt5Agg')
            except Exception:
                print("[WARNING] Could not set interactive backend. Plots may not display.")
                print("          Make sure you have a display connection (e.g., X11 forwarding)")
                print("          or run this on a machine with a GUI.")

    # Convert to Path objects
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from: {results_dir}")
    print(f"Saving plots to: {output_dir}")
    if args.show:
        print(f"Displaying plots interactively")
    print()

    # Load results
    results = load_results(results_dir)

    # Print summary statistics
    print_summary_statistics(results)

    # Generate visualizations
    print("[INFO] Generating visualizations...")

    # 0. Gridworld environment (shows grid structure and portals)
    plot_gridworld_environment(results, output_dir, show=args.show)

    # 1. Training history plots (if available)
    plot_training_history(results, output_dir, show=args.show)

    # 2. Distance matrix visualizations
    plot_distance_matrices(results, output_dir, show=args.show)

    # 3. Rollout analysis
    plot_rollout_analysis(results, output_dir, show=args.show)

    # 4. Trajectory error histogram
    plot_trajectory_error_histogram(results, output_dir, show=args.show)

    # 5. Trajectory visualization (first few successful and failed rollouts)
    rollout_results = results['rollout_evaluation']['rollout_results']
    if rollout_results and 'all_states' in rollout_results[0]:
        # Plot first successful rollout
        for idx, rollout in enumerate(rollout_results):
            if rollout['success']:
                plot_trajectory_on_grid(results, output_dir, rollout_idx=idx, show=args.show)
                break

        # Plot first failed rollout
        for idx, rollout in enumerate(rollout_results):
            if not rollout['success']:
                plot_trajectory_on_grid(results, output_dir, rollout_idx=idx, show=args.show)
                break

    # 6. Embedding visualizations (if heuristic is available)
    heuristic_path = results_dir / 'distance_heuristic_v4.pkl'
    if heuristic_path.exists():
        print("[INFO] Generating embedding visualizations...")
        print("[INFO] Note: This requires setting PYTHONPATH=/path/to/src before running.")

        # Try each projection method
        for method in ['raw']:
            try:
                plot_embeddings_tsne(results, heuristic_path, output_dir,
                                   show=args.show, projection_method=method)
            except Exception as e:
                print(f"[WARNING] Could not generate {method.upper()} plot: {e}")
                if method == 'umap':
                    print("          To enable UMAP plots, install: pip install umap-learn")
                else:
                    print("          To enable embedding plots, run with:")
                    print("          export PYTHONPATH=/home/de7281/thesis/tamp_physical_improvisation/src:$PYTHONPATH")

    print("\n[SUCCESS] All visualizations generated!")
    print(f"[INFO] Plots saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
