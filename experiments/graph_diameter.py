"""Print the graph diameter (max finite shortest-path distance) for training datasets.

Usage:
    python -m experiments.graph_diameter
    python -m experiments.graph_diameter --data-root /other/path
"""

import argparse
from pathlib import Path

import numpy as np

from tamp_improv.approaches.improvisational.graph_training import compute_graph_distances
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData

DEFAULT_DATA_ROOT = Path("/n/fs/recbench/slap_training_data")

ENVS = ["cleanup_table", "cluttered_drawer", "obstacle_tower"]


def graph_diameter(data_path: Path) -> tuple[float, float, int]:
    """Return (diameter, mean_finite_dist, n_finite_pairs) for the dataset."""
    training_data = GoalConditionedTrainingData.load(data_path)
    graph_distances = compute_graph_distances(training_data.graph, exclude_shortcuts=True)
    finite = [d for d in graph_distances.values() if np.isfinite(d)]
    if not finite:
        return float("nan"), float("nan"), 0
    return float(max(finite)), float(np.mean(finite)), len(finite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print graph diameter for each environment.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing per-environment training data folders.",
    )
    args = parser.parse_args()

    for env in ENVS:
        data_path = args.data_root / env
        if not data_path.exists():
            print(f"{env}: data path not found ({data_path})")
            continue
        print(f"{env}: loading...", flush=True)
        diameter, mean_dist, n_pairs = graph_diameter(data_path)
        print(
            f"{env}: diameter={diameter:.1f}  mean={mean_dist:.1f}  "
            f"finite_pairs={n_pairs}"
        )


if __name__ == "__main__":
    main()
