"""Compare avg_steps across experiments by name.

Looks up experiments under the outputs directory by matching directories named
sweep_<name> (no job ID suffix).

Usage:
    python -m experiments.compare_results name1 name2 name3 ...
    python -m experiments.compare_results name1 name2 --metric avg_reward
    python -m experiments.compare_results name1 name2 --outputs-dir /path/to/outputs
"""

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

DEFAULT_OUTPUTS_DIR = Path("/n/fs/recbench/slap_outputs/outputs")


def parse_results(results_path: Path) -> dict[str, str]:
    """Parse a results.txt file into a dict."""
    result = {}
    for line in results_path.read_text().splitlines():
        if ": " in line:
            key, _, value = line.partition(": ")
            result[key.strip()] = value.strip()
    return result


def summarize_outputs(outputs_dir: Path, csv_path: Path) -> None:
    """Scan outputs_dir for sweep_<name> dirs and write a CSV summary."""
    pattern = re.compile(r"^sweep_(.+)$")
    job_id_suffix = re.compile(r"_\d+$")
    rows = []
    for d in sorted(outputs_dir.iterdir()):
        m = pattern.match(d.name)
        if not m or not d.is_dir():
            continue
        name = m.group(1)
        if job_id_suffix.search(name):
            continue
        results_file = d / "results.txt"
        if not results_file.exists():
            rows.append({"name": name, "avg_steps": "no results found", "success_rate": "no results found"})
        else:
            results = parse_results(results_file)
            rows.append({
                "name": name,
                "avg_steps": results.get("avg_steps", ""),
                "success_rate": results.get("success_rate", ""),
            })

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "avg_steps", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {csv_path}")


_TRAINING_TIME_KEYS = [
    "collection_time",
    "heuristic_training_time",
    "heuristic_pruning_time",
    "policy_training_time",
    "add_shortcuts_time",
]
_ALL_TIME_KEYS = _TRAINING_TIME_KEYS + ["evaluation_time"]


def load_times(run_dir: Path) -> dict[str, float] | None:
    """Load all stage times from results.pkl, or None if unavailable."""
    pkl_path = run_dir / "results.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
        times = getattr(results, "times", None)
        if not times:
            return None
        return {k: times.get(k, 0.0) for k in _ALL_TIME_KEYS}
    except Exception:
        return None


def load_episode_steps(run_dir: Path) -> list[float] | None:
    """Load per-episode step counts from results.pkl, or None if unavailable."""
    pkl_path = run_dir / "results.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
        episodes = getattr(results, "eval_episodes", None)
        # print("Loaded results.pkl, eval_episodes:", episodes)
        if not episodes:
            return None
        return [ep["true_steps"] for ep in episodes if "true_steps" in ep]
    except Exception:
        return None


def find_latest_run(name: str, outputs_dir: Path) -> Path | None:
    """Find the directory named sweep_<name>."""
    d = outputs_dir / f"sweep_{name}"
    if d.is_dir() and (d / "results.txt").exists():
        return d
    return None


def aggregate_seeds(
    base_name: str, outputs_dir: Path, metric: str = "avg_steps"
) -> tuple[float, float, int, list[float]] | None:
    """Aggregate results across sweep_{base_name}_s* seed directories.

    Returns (mean, std, n_seeds, all_episode_steps) or None if no seed dirs found.
    std is the cross-seed standard deviation (between-seed variance).
    """
    seed_dirs = sorted(outputs_dir.glob(f"sweep_{base_name}_s*"))
    seed_dirs = [d for d in seed_dirs if d.is_dir() and (d / "results.txt").exists()]
    if not seed_dirs:
        return None

    seed_values: list[float] = []
    all_episode_steps: list[float] = []
    for d in seed_dirs:
        results = parse_results(d / "results.txt")
        if metric not in results:
            continue
        seed_values.append(float(results[metric]))
        eps = load_episode_steps(d)
        if eps:
            all_episode_steps.extend(eps)

    if not seed_values:
        return None

    mean = float(np.mean(seed_values))
    if len(all_episode_steps) > 1:
        sem = float(np.std(all_episode_steps, ddof=1) / np.sqrt(len(all_episode_steps)))
    else:
        sem = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
    return mean, sem, len(seed_values), all_episode_steps


def _plot_eval_time(names: list[str], outputs_dir: Path, output: Path) -> None:
    """Bar chart of evaluation wall-clock time per experiment (minutes)."""
    labels, values, missing = [], [], []

    for name in names:
        run_dir = find_latest_run(name, outputs_dir)
        if run_dir is None:
            print(f"WARNING: no results found for '{name}'", file=sys.stderr)
            missing.append(name)
            continue
        times = load_times(run_dir)
        if not times or times.get("evaluation_time", 0.0) == 0.0:
            print(f"WARNING: no evaluation_time for '{name}'", file=sys.stderr)
            missing.append(name)
            continue
        labels.append(name)
        values.append(times["evaluation_time"] / 60.0)

    if not labels:
        print("No eval timing data to plot.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Evaluation time (minutes)")
    ax.set_title("Evaluation time by experiment")
    if missing:
        ax.set_xlabel(f"Missing: {', '.join(missing)}", fontsize=8, color="red")
    plt.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved to {output}")


def _plot_training_times(names: list[str], outputs_dir: Path, output: Path) -> None:
    """Stacked bar chart of training time breakdown per experiment (hours)."""
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    labels = []
    stage_data: dict[str, list[float]] = {k: [] for k in _TRAINING_TIME_KEYS}
    missing = []

    for name in names:
        run_dir = find_latest_run(name, outputs_dir)
        if run_dir is None:
            print(f"WARNING: no results found for '{name}'", file=sys.stderr)
            missing.append(name)
            continue
        times = load_times(run_dir)
        if times is None:
            print(f"WARNING: no timing data in results.pkl for '{name}'", file=sys.stderr)
            missing.append(name)
            continue
        labels.append(name)
        for k in _TRAINING_TIME_KEYS:
            stage_data[k].append(times[k] / 3600.0)  # seconds → hours

    if not labels:
        print("No timing data to plot.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    bottoms = np.zeros(len(labels))
    stage_labels = ["Collection", "Heuristic training", "Pruning", "Policy training", "Add shortcuts"]
    for (key, label, color) in zip(_TRAINING_TIME_KEYS, stage_labels, colors):
        vals = np.array(stage_data[key])
        ax.bar(range(len(labels)), vals, bottom=bottoms, label=label, color=color)
        bottoms += vals

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Training time (hours)")
    ax.set_title("Training time breakdown by experiment")
    ax.legend(fontsize=8)
    if missing:
        ax.set_xlabel(f"Missing: {', '.join(missing)}", fontsize=8, color="red")

    plt.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment results.")
    parser.add_argument("names", nargs="*", help="Experiment names to compare")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Scan all sweep_* dirs and write results_summary.csv instead of plotting",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results_summary.csv"),
        help="CSV path for --summarize (default: results_summary.csv)",
    )
    parser.add_argument(
        "--metric", default="avg_steps", help="Metric to plot (default: avg_steps)"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Root outputs directory",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("comparison.png"), help="Output image path"
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Experiment name to use as baseline; bars show %% improvement over it",
    )
    parser.add_argument(
        "--plot-time",
        action="store_true",
        help="Plot stacked bar chart of training time breakdown instead of avg_steps",
    )
    parser.add_argument(
        "--plot-eval-time",
        action="store_true",
        help="Plot bar chart of evaluation wall-clock time per experiment",
    )
    args = parser.parse_args()

    if args.summarize:
        summarize_outputs(args.outputs_dir, args.summary_output)
        return

    if not args.names:
        parser.error("Provide experiment names, or use --summarize")

    if args.plot_time:
        _plot_training_times(args.names, args.outputs_dir, args.output)
        return

    if args.plot_eval_time:
        _plot_eval_time(args.names, args.outputs_dir, args.output)
        return

    values = []
    stds = []
    labels = []
    missing = []

    for name in args.names:
        # Try multi-seed aggregation first
        agg = aggregate_seeds(name, args.outputs_dir, args.metric)
        if agg is not None:
            mean, std, n_seeds, all_eps = agg
            print(f"{name}: {n_seeds} seeds ({len(all_eps)} episodes), mean={mean:.2f}, sem={std:.2f}")
            labels.append(name)
            values.append(mean)
            stds.append(std)
            continue

        # Fall back to single run
        run_dir = find_latest_run(name, args.outputs_dir)
        if run_dir is None:
            print(f"WARNING: no results found for '{name}'", file=sys.stderr)
            missing.append(name)
            continue
        results = parse_results(run_dir / "results.txt")
        if args.metric not in results:
            print(
                f"WARNING: metric '{args.metric}' not in results for '{name}'",
                file=sys.stderr,
            )
            missing.append(name)
            continue
        labels.append(name)
        values.append(float(results[args.metric]))
        episode_steps = load_episode_steps(run_dir)
        if episode_steps and len(episode_steps) > 1:
            stds.append(float(np.std(episode_steps, ddof=1) / np.sqrt(len(episode_steps))))
        else:
            stds.append(0.0)

    if not values:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    # Resolve baseline
    baseline_value = None
    if args.baseline is not None:
        # Try multi-seed aggregation first, then fall back to single run
        baseline_agg = aggregate_seeds(args.baseline, args.outputs_dir, args.metric)
        if baseline_agg is not None:
            baseline_value, _, _, _ = baseline_agg
            print(f"baseline '{args.baseline}': mean={baseline_value:.2f} (from seeds)")
        else:
            baseline_dir = find_latest_run(args.baseline, args.outputs_dir)
            if baseline_dir is None:
                print(f"ERROR: baseline '{args.baseline}' not found", file=sys.stderr)
                sys.exit(1)
            baseline_results = parse_results(baseline_dir / "results.txt")
            if args.metric not in baseline_results:
                print(f"ERROR: baseline has no {args.metric}", file=sys.stderr)
                sys.exit(1)
            baseline_value = float(baseline_results[args.metric])

    plot_values = values
    plot_stds = stds
    ylabel = args.metric
    title = f"{args.metric} by experiment"
    if baseline_value is not None:
        plot_values = [(baseline_value - v) / baseline_value * 100 for v in values]
        plot_stds = [s / baseline_value * 100 for s in stds]
        ylabel = f"% improvement in avg_steps over '{args.baseline}'"
        title = f"Improvement over {args.baseline}"

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    bars = ax.bar(range(len(labels)), plot_values,
                  yerr=plot_stds if any(s > 0 for s in plot_stds) else None,
                  capsize=3, error_kw={"linewidth": 1.0})
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if baseline_value is not None:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars, plot_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}{'%' if baseline_value is not None else ''}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )

    if missing:
        ax.set_xlabel(f"Missing: {', '.join(missing)}", fontsize=8, color="red")

    plt.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
