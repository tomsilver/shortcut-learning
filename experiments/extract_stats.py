"""Extract per-run statistics from sweep output directories into a CSV.

Usage:
    python -m experiments.extract_stats "sweep_mg_*_ch5_*" [--outputs-dir PATH] [--out FILE]

Scans matching sweep_* directories, loads results.pkl from each, and writes a
CSV row per run with eval stats (success rate, avg steps, shortcut fraction,
and their standard errors) plus runtime for each pipeline phase.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUTS_DIR = Path("/n/fs/iterativesl/slap_outputs/outputs")

PHASE_KEYS = [
    "collection_time",
    "heuristic_training_time",
    "heuristic_pruning_time",
    "policy_training_time",
    "add_shortcuts_time",
    "evaluation_time",
]


def _episode_uses_shortcuts(ep: dict[str, Any]) -> bool:
    edges = ep.get("optimal_path_edges")
    if not edges:
        return False
    return any(e.get("is_shortcut", False) for e in edges)


def _proportion_se(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(np.sqrt(p * (1.0 - p) / n))


def _mean_se(values: np.ndarray) -> float:
    n = len(values)
    if n <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(n))


# Per-sample table row from test_heuristic_quality:
#       0       10        28.68        12.00        19.72         16.68
# Columns: source, target, estimated, graph, true, |estimated - graph|
_SAMPLE_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
)
# "Source   Target   Estimated     Graph      True     Error" — marks the
# start of a new sample table, letting us reset accumulated samples so we
# keep only the most recent block.
_HEADER_RE = re.compile(r"^\s*Source\s+Target\s+Estimated\s+Graph\s+True")


def _parse_distance_stats(run_dir: Path) -> tuple[float, float, float, float]:
    """Return (mae_vs_true, corr_vs_true, mae_vs_graph, corr_vs_graph).

    Parses the per-sample table from the newest job_*.out file in run_dir
    and computes correlation and MAE directly. Uses only the LAST sample
    table in the file, since multiple rounds print these stats and we want
    the final values.
    """
    candidates = sorted(run_dir.glob("job_*.out"))
    if not candidates:
        return (float("nan"),) * 4
    log = candidates[-1]

    estimated: list[float] = []
    graph: list[float] = []
    true: list[float] = []
    try:
        with open(log) as f:
            for line in f:
                if _HEADER_RE.match(line):
                    estimated.clear()
                    graph.clear()
                    true.clear()
                    continue
                m = _SAMPLE_RE.match(line)
                if m:
                    est_v = float(m.group(3))
                    graph_v = float(m.group(4))
                    true_v = float(m.group(5))
                    # Skip inf graph distances (not useful for corr/MAE)
                    if not (np.isfinite(est_v) and np.isfinite(graph_v) and np.isfinite(true_v)):
                        continue
                    estimated.append(est_v)
                    graph.append(graph_v)
                    true.append(true_v)
    except OSError:
        return (float("nan"),) * 4

    if len(estimated) < 2:
        return (float("nan"),) * 4

    est = np.array(estimated)
    gra = np.array(graph)
    tru = np.array(true)

    mae_true = float(np.mean(np.abs(est - tru)))
    mae_graph = float(np.mean(np.abs(est - gra)))
    with np.errstate(invalid="ignore"):
        corr_true = float(np.corrcoef(est, tru)[0, 1])
        corr_graph = float(np.corrcoef(est, gra)[0, 1])
    return mae_true, corr_true, mae_graph, corr_graph


def extract_run_stats(run_dir: Path) -> dict[str, Any] | None:
    results_path = run_dir / "results.pkl"
    if not results_path.exists():
        return None

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    episodes = getattr(results, "eval_episodes", None) or []
    n = len(episodes)

    successes = np.array([bool(ep.get("success", False)) for ep in episodes], dtype=float)
    steps = np.array([float(ep.get("true_steps", 0)) for ep in episodes], dtype=float)
    uses_sc = np.array([_episode_uses_shortcuts(ep) for ep in episodes], dtype=float)

    success_rate = float(successes.mean()) if n else float("nan")
    avg_steps = float(steps.mean()) if n else float("nan")
    shortcut_frac = float(uses_sc.mean()) if n else float("nan")

    mae_true, corr_true, mae_graph, corr_graph = _parse_distance_stats(run_dir)

    # Shortcut quality: avg success rate and avg length of pruned shortcuts
    sq = getattr(results, "shortcut_quality_results", None) or []
    sq_success_rates = np.array([float(s.get("success_rate", 0.0)) for s in sq])
    sq_avg_lengths = np.array(
        [float(s.get("avg_length", 0.0)) for s in sq if s.get("avg_length", 0.0) > 0]
    )
    shortcut_success_rate = (
        float(sq_success_rates.mean()) if sq_success_rates.size else float("nan")
    )
    shortcut_success_rate_se = _mean_se(sq_success_rates) if sq_success_rates.size else float("nan")
    shortcut_avg_length = (
        float(sq_avg_lengths.mean()) if sq_avg_lengths.size else float("nan")
    )
    shortcut_avg_length_se = _mean_se(sq_avg_lengths) if sq_avg_lengths.size else float("nan")

    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "n_episodes": n,
        "success_rate": success_rate,
        "success_rate_se": _proportion_se(success_rate, n),
        "avg_steps": avg_steps,
        "avg_steps_se": _mean_se(steps),
        "shortcut_frac": shortcut_frac,
        "shortcut_frac_se": _proportion_se(shortcut_frac, n),
        "distance_mae_vs_true": mae_true,
        "distance_corr_vs_true": corr_true,
        "distance_mae_vs_graph": mae_graph,
        "distance_corr_vs_graph": corr_graph,
        "shortcut_success_rate": shortcut_success_rate,
        "shortcut_success_rate_se": shortcut_success_rate_se,
        "shortcut_avg_length": shortcut_avg_length,
        "shortcut_avg_length_se": shortcut_avg_length_se,
        "n_shortcuts_evaluated": len(sq),
    }

    times = getattr(results, "times", None) or {}
    for key in PHASE_KEYS:
        row[f"t_{key[:-5] if key.endswith('_time') else key}"] = float(times.get(key, 0.0))

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pattern",
        help='Glob pattern for sweep dirs (e.g. "sweep_mg_*_ch5_*").',
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help=f"Parent directory containing sweep_* runs (default: {DEFAULT_OUTPUTS_DIR}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <pattern>_stats.csv in current dir).",
    )
    args = parser.parse_args()

    out_path = args.out or Path(
        args.pattern.replace("*", "").replace("/", "").strip("_") + "_stats.csv"
    )

    run_dirs = sorted(args.outputs_dir.glob(args.pattern))
    if not run_dirs:
        print(f"[WARN] No directories match {args.outputs_dir}/{args.pattern}")
        return

    rows: list[dict[str, Any]] = []
    n_missing = 0
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        row = extract_run_stats(run_dir)
        if row is None:
            print(f"[SKIP] {run_dir.name}: no results.pkl")
            n_missing += 1
            continue
        rows.append(row)

    if not rows:
        print("[WARN] No rows extracted.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[SAVED] {len(rows)} rows → {out_path}")
    if n_missing:
        print(f"  ({n_missing} dirs skipped due to missing results.pkl)")


if __name__ == "__main__":
    main()
