"""Generic bar chart of a metric (or % improvement over baseline) across sweep runs.

Edit the CONFIG block below to pick experiments, labels, baseline, and titles,
then run:

    python -m experiments.plot_sweep_bars

Reads stats from experiments/sweep_stats.csv (produced by extract_stats.py).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Experiment name → abbreviation shown on the x-axis. Keys must match the
# `run_name` column of sweep_stats.csv exactly.
#
# Best configurations (min avg_steps at 100% success, 10-shortcut SAC only).
# Populated from sweep_stats.csv on 2026-04-13. Swap in as needed.
#
#   gw_ch5 best  (note: ch5 use_multi_rl=false bug — pre-rerun)
#     sweep_gw_sac_10_ch5_s42           # 25.66 ± 1.40
#     sweep_gw_dsac_10_ch5_s42          # 26.51 ± 1.49
#     sweep_gw_crl_10_ch5_s42           # 26.86 ± 1.52
#     sweep_gw_cmd_10_ch5_s42           # 26.80 ± 1.49
#
#   gw_ch6/7 best
#     sweep_gw_sac_nores_ch7_s42        # 25.26 ± 1.35
#     sweep_gw_dsac_full_lam05_ch7_s42  # 25.89 ± 1.42
#     sweep_gw_crl_noucb_ch7_s42        # 26.14 ± 1.43
#     sweep_gw_cmd_nores_ch7_s42        # 25.84 ± 1.39
#
#   mg_ch5 best  (note: ch5 use_multi_rl=false bug — pre-rerun)
#     sweep_mg_sac_naive_ch5_s42        # 26.23 ± 1.42
#     sweep_mg_dsac_10_ch5_s42          # 26.62 ± 1.48
#     sweep_mg_crl_10_ch5_s42           # 26.84 ± 1.53
#     sweep_mg_cmd_10_ch5_s42           # 26.89 ± 1.52
#
#   mg_ch6/7 best
#     sweep_mg_sac_noucb_ch7_s42        # 25.72 ± 1.38
#     sweep_mg_dsac_10_unif_ch6_s42     # 26.14 ± 1.43
#     sweep_mg_crl_noucb_ch7_s42        # 26.52 ± 1.47
#     sweep_mg_cmd_10_unif_ch6_s42      # 26.09 ± 1.42
# mapping = {
#     "sweep_gw_sac_naive_ch5_s42": "Naive",
#     "sweep_gw_sac_estimate_ch5_s42": "Estimate",
#     "sweep_gw_sac_exact_ch5_s42": "Exact"
# }

# mapping = {
#     "sweep_mg_none_10_ch5_s42": "Random",
#     "sweep_mg_routs_10_ch5_s42": "Rollouts",
#     "sweep_mg_srouts_10_ch5_s42": "Smart Rollouts",
#     "sweep_mg_sac_naive_ch5_s42": "SAC",
#     "sweep_mg_dsac_10_ch5_s42": "DSAC",
#     "sweep_mg_crl_10_ch5_s42": "CRL",
#     "sweep_mg_cmd_10_ch5_s42": "CMD",
# }

# mapping = {
#     "sweep_gw_none_10_ch5_s42": "Random",
#     "sweep_gw_routs_10_ch5_s42": "SLAP",
#     "sweep_gw_sac_10_ch5_s42": "SLIP",
#     "sweep_gw_sac_full_ch7_s42": "SLIDE",
# }

# mapping = {
#     "sweep_mg_sac_10_unif_ch6_s42": "Uniform",
#     "sweep_mg_sac_10_nsucb10_ch6_s42": "Deterministic UCB",
#     "sweep_mg_sac_10_ucb10_ch6_s42": "Stochastic UCB",
# }

# mapping = {
#     "sweep_gw_sac_10_unif_ch6_s42": "Uniform",
#     "sweep_gw_sac_10_ucb1_ch6_s42": "Weight=1",
#     "sweep_gw_sac_10_ucb10_ch6_s42": "Weight=10",
#     "sweep_gw_sac_10_ucb100_ch6_s42": "Weight=100",
# }
# mapping = {
#     "sweep_mg_srouts_10_ch5_s42": "Smart Rollouts",
#     "sweep_mg_sac_10_ch5_s42": "SAC",
#     "sweep_mg_dsac_10_ch5_s42": "DSAC",
#     "sweep_mg_crl_10_ch5_s42": "CRL",
#     "sweep_mg_cmd_10_ch5_s42": "CMD",
# }

# mapping = {
#     "sweep_gw_srouts_10_ch5_s42": "Smart Rollouts",
#     "sweep_gw_sac_10_ch5_s42": "SAC",
#     "sweep_gw_dsac_10_ch5_s42": "DSAC",
#     "sweep_gw_crl_10_ch5_s42": "CRL",
#     "sweep_gw_cmd_10_ch5_s42": "CMD",
# }
# mapping = {
#     "sweep_mg_sac_noucb_ch7_s42": "SAC",
#     "sweep_mg_dsac_10_unif_ch6_s42": "DSAC",
#     "sweep_mg_crl_noucb_ch7_s42": "CRL",
#     "sweep_mg_cmd_10_unif_ch6_s42": "CMD",
# }

# mapping = {
#     "sweep_gw_sac_nores_ch7_s42": "SAC",
#     "sweep_gw_dsac_full_lam05_ch7_s42": "DSAC",
#     "sweep_gw_crl_noucb_ch7_s42": "CRL",
#     "sweep_gw_cmd_nores_ch7_s42": "CMD",
# }

# mapping = {
#     "sweep_mg_sac_10_ch5_s42": "10",
#     "sweep_mg_sac_15_ch5_s42": "15",
#     "sweep_mg_sac_20_ch5_s42": "20",
#     "sweep_mg_sac_25_ch5_s42": "25",
#     "sweep_mg_sac_30_ch5_s42": "30",

# }

# series = {
#     "Full": {"SAC": "sweep_mg_sac_full_ch7_s42", "DSAC": "sweep_mg_dsac_full_ch7_s42", "CRL": "sweep_mg_crl_full_ch7_s42", "CMD": "sweep_mg_cmd_full_ch7_s42"},
#     "No-Res": {"SAC": "sweep_mg_sac_nores_ch7_s42", "DSAC": "sweep_mg_dsac_nores_ch7_s42", "CRL": "sweep_mg_crl_nores_ch7_s42", "CMD": "sweep_mg_cmd_nores_ch7_s42"},
#     "No-UCB": {"SAC": "sweep_mg_sac_noucb_ch7_s42", "DSAC": "sweep_mg_dsac_noucb_ch7_s42", "CRL": "sweep_mg_crl_noucb_ch7_s42", "CMD": "sweep_mg_cmd_noucb_ch7_s42"},
#     "No-Promo": {"SAC": "sweep_mg_sac_nograd_ch7_s42", "DSAC": "sweep_mg_dsac_nograd_ch7_s42", "CRL": "sweep_mg_crl_nograd_ch7_s42", "CMD": "sweep_mg_cmd_nograd_ch7_s42"},
#     "None": {"SAC": "sweep_mg_sac_10_unif_ch6_s42", "DSAC": "sweep_mg_dsac_10_unif_ch6_s42", "CRL": "sweep_mg_crl_10_unif_ch6_s42", "CMD": "sweep_mg_cmd_10_unif_ch6_s42"},
# }
# Mapping that reduces the above series to just the SAC entries
# mapping = {
#     "sweep_mg_sac_full_ch7_s42": "Full",
#     "sweep_mg_sac_nores_ch7_s42": "No-Res",
#     "sweep_mg_sac_noucb_ch7_s42": "No-UCB",
#     "sweep_mg_sac_nograd_ch7_s42": "No-Promo",
#     "sweep_mg_sac_10_unif_ch6_s42": "None",
# }


# mapping = {
#     "sweep_mg_sac_nores_ch7_s42": "No-Res",
#     "sweep_mg_sac_full_ch7_s42": "Lam-0",
#     "sweep_mg_sac_full_lam05_ch7_s42": "Lam-0.5",
#     "sweep_mg_sac_full_lam1_ch7_s42": "Lam-1",
# }

# mapping = {
#     "sweep_ot_none_ch5_s42": "Random",
#     "sweep_ot_routs_ch5_s42": "Rollouts",
#     "sweep_ot_srouts_ch5_s42": "Smart Rollouts",
#     "sweep_ot_sac_ch5_s42": "SAC",
#     "sweep_ot_dsac_ch5_s42": "DSAC",
#     "sweep_ot_crl_ch5_s42": "CRL",
#     "sweep_ot_cmd_ch5_s42": "CMD",

# }

mapping = {
    "sweep_o2_sac_unif_ch6_s42": "None",
    "sweep_o2_sac_ucb01_ch6_s42": "UCB0.1",
    "sweep_o2_sac_ucb1_ch6_s42": "UCB1",
    "sweep_o2_sac_ucb100_ch6_s42": "UCB100",
    "sweep_o2_sac_ucb_ch7_s42": "Residual",
    "sweep_o2_sac_unif_ch7_s42": "Residual+UCB1"
}
# mapping = None

# Optional paired (grouped) bar chart config. When non-None, overrides `mapping`
# and groups multiple bars under each x-axis label. Format:
# series = {
#     "SLIP":   {"SAC": "sweep_gw_sac_10_ch5_s42", "DSAC": "sweep_gw_dsac_10_ch5_s42", "CRL": "sweep_gw_crl_10_ch5_s42", "CMD": "sweep_gw_cmd_10_ch5_s42"},
#     "SLIDE": {"SAC": "sweep_gw_sac_nores_ch7_s42", "DSAC": "sweep_gw_dsac_full_lam05_ch7_s42", "CRL": "sweep_gw_crl_noucb_ch7_s42", "CMD": "sweep_gw_cmd_nores_ch7_s42"},
# }

# series = {
#     "SLIP":   {"SAC": "sweep_mg_sac_naive_ch5_s42", "DSAC": "sweep_mg_dsac_10_ch5_s42", "CRL": "sweep_mg_crl_10_ch5_s42", "CMD": "sweep_mg_cmd_10_ch5_s42"},
#     "SLIDE": {"SAC": "sweep_mg_sac_noucb_ch7_s42", "DSAC": "sweep_mg_dsac_10_unif_ch6_s42", "CRL": "sweep_mg_crl_noucb_ch7_s42", "CMD": "sweep_mg_cmd_10_unif_ch6_s42"},
# }

# series = {
#     "SLAP":   {"0": "sweep_gw_routs_10_ch5_s42", "1": "sweep_mg_routs_10_ch5_s42", "2": "sweep_mg_routs_10_filter2_ch5_s42", "3": "sweep_mg_routs_10_filter3_ch5_s42"},
#     "SROUTS": {"0": "sweep_gw_srouts_10_ch5_s42", "1": "sweep_mg_srouts_10_ch5_s42", "2": "sweep_mg_srouts_10_filter2_ch5_s42", "3": "sweep_mg_srouts_10_filter3_ch5_s42"},
#     "SAC": {"0": "sweep_gw_sac_10_ch5_s42", "1": "sweep_mg_sac_naive_ch5_s42", "2": "sweep_mg_sac_10_filter2_ch5_s42", "3": "sweep_mg_sac_10_filter3_ch5_s42"},
# }

# series = {
#     "Rollouts":   {"10": "sweep_gw_routs_10_ch5_s42", "15": "sweep_gw_routs_15_ch5_s42", "20": "sweep_gw_routs_20_ch5_s42", "25": "sweep_gw_routs_25_ch5_s42", "30": "sweep_gw_routs_30_ch5_s42"},
#     "Smart Rollouts": {"10": "sweep_gw_srouts_10_ch5_s42", "15": "sweep_gw_srouts_15_ch5_s42", "20": "sweep_gw_srouts_20_ch5_s42", "25": "sweep_gw_srouts_25_ch5_s42", "30": "sweep_gw_srouts_30_ch5_s42"},
#     "SAC": {"10": "sweep_gw_sac_10_ch5_s42", "15": "sweep_gw_sac_15_ch5_s42", "20": "sweep_gw_sac_20_ch5_s42", "25": "sweep_gw_sac_25_ch5_s42", "30": "sweep_gw_sac_30_ch5_s42"},
# }
# Outer keys are series labels (appear in legend). Inner keys are x-axis group
# labels (abbreviations). Inner values are run names. Only supported in
# mode="metric".

# series = {
#     "SAC": {"Uniform": "sweep_gw_sac_10_unif_ch6_s42", "UCB1": "sweep_gw_sac_10_ucb1_ch6_s42", "UCB10": "sweep_gw_sac_10_ucb10_ch6_s42", "UCB100": "sweep_gw_sac_10_ucb100_ch6_s42"},
#     "DSAC": {"Uniform": "sweep_gw_dsac_10_unif_ch6_s42", "UCB1": "sweep_gw_dsac_10_ucb1_ch6_s42", "UCB10": "sweep_gw_dsac_10_ucb10_ch6_s42", "UCB100": "sweep_gw_dsac_10_ucb100_ch6_s42"},
#     "CRL": {"Uniform": "sweep_gw_crl_10_unif_ch6_s42", "UCB1": "sweep_gw_crl_10_ucb1_ch6_s42", "UCB10": "sweep_gw_crl_10_ucb10_ch6_s42", "UCB100": "sweep_gw_crl_10_ucb100_ch6_s42"},
#     "CMD": {"Uniform": "sweep_gw_cmd_10_unif_ch6_s42", "UCB1": "sweep_gw_cmd_10_ucb1_ch6_s42", "UCB10": "sweep_gw_cmd_10_ucb10_ch6_s42", "UCB100": "sweep_gw_cmd_10_ucb100_ch6_s42"}
# }

# Reverse of the above series where the series are the sampling mechanisims and the groups are the methods (SAC, DSAC, etc). Just to verify that the code doesn't rely on any particular ordering.
# series = {
#     "Uniform": {"SAC": "sweep_mg_sac_10_unif_ch6_s42", "DSAC": "sweep_mg_dsac_10_unif_ch6_s42", "CRL": "sweep_mg_crl_10_unif_ch6_s42", "CMD": "sweep_mg_cmd_10_unif_ch6_s42"},
#     "UCB1": {"SAC": "sweep_mg_sac_10_ucb1_ch6_s42", "DSAC": "sweep_mg_dsac_10_ucb1_ch6_s42", "CRL": "sweep_mg_crl_10_ucb1_ch6_s42", "CMD": "sweep_mg_cmd_10_ucb1_ch6_s42"},
#     "UCB10": {"SAC": "sweep_mg_sac_10_ucb10_ch6_s42", "DSAC": "sweep_mg_dsac_10_ucb10_ch6_s42", "CRL": "sweep_mg_crl_10_ucb10_ch6_s42", "CMD": "sweep_mg_cmd_10_ucb10_ch6_s42"},
#     "UCB100": {"SAC": "sweep_mg_sac_10_ucb100_ch6_s42", "DSAC": "sweep_mg_dsac_10_ucb100_ch6_s42", "CRL": "sweep_mg_crl_10_ucb100_ch6_s42", "CMD": "sweep_mg_cmd_10_ucb100_ch6_s42"}
# }

# series = {
#     "Full": {"SAC": "sweep_mg_sac_full_ch7_s42", "DSAC": "sweep_mg_dsac_full_ch7_s42", "CRL": "sweep_mg_crl_full_ch7_s42", "CMD": "sweep_mg_cmd_full_ch7_s42"},
#     "No-Res": {"SAC": "sweep_mg_sac_nores_ch7_s42", "DSAC": "sweep_mg_dsac_nores_ch7_s42", "CRL": "sweep_mg_crl_nores_ch7_s42", "CMD": "sweep_mg_cmd_nores_ch7_s42"},
#     "No-UCB": {"SAC": "sweep_mg_sac_noucb_ch7_s42", "DSAC": "sweep_mg_dsac_noucb_ch7_s42", "CRL": "sweep_mg_crl_noucb_ch7_s42", "CMD": "sweep_mg_cmd_noucb_ch7_s42"},
#     "No-Promo": {"SAC": "sweep_mg_sac_nograd_ch7_s42", "DSAC": "sweep_mg_dsac_nograd_ch7_s42", "CRL": "sweep_mg_crl_nograd_ch7_s42", "CMD": "sweep_mg_cmd_nograd_ch7_s42"},
#     "None": {"SAC": "sweep_mg_sac_10_unif_ch6_s42", "DSAC": "sweep_mg_dsac_10_unif_ch6_s42", "CRL": "sweep_mg_crl_10_unif_ch6_s42", "CMD": "sweep_mg_cmd_10_unif_ch6_s42"},
# }

# series = {
#     "Full": {"SAC": "sweep_mg_sac_full_ch7_s42", "DSAC": "sweep_mg_dsac_full_ch7_s42", "CRL": "sweep_mg_crl_full_ch7_s42", "CMD": "sweep_mg_cmd_full_ch7_s42"},
#     "No-Res": {"SAC": "sweep_mg_sac_nores_ch7_s42", "DSAC": "sweep_mg_dsac_nores_ch7_s42", "CRL": "sweep_mg_crl_nores_ch7_s42", "CMD": "sweep_mg_cmd_nores_ch7_s42"},
#     "No-UCB": {"SAC": "sweep_mg_sac_noucb_ch7_s42", "DSAC": "sweep_mg_dsac_noucb_ch7_s42", "CRL": "sweep_mg_crl_noucb_ch7_s42", "CMD": "sweep_mg_cmd_noucb_ch7_s42"},
#     "No-Promo": {"SAC": "sweep_mg_sac_nograd_ch7_s42", "DSAC": "sweep_mg_dsac_nograd_ch7_s42", "CRL": "sweep_mg_crl_nograd_ch7_s42", "CMD": "sweep_mg_cmd_nograd_ch7_s42"},
#     "None": {"SAC": "sweep_mg_sac_10_unif_ch6_s42", "DSAC": "sweep_mg_dsac_10_unif_ch6_s42", "CRL": "sweep_mg_crl_10_unif_ch6_s42", "CMD": "sweep_mg_cmd_10_unif_ch6_s42"},
# }


# Turn these two mappings into one series where the outer group is Ch5 vs Ch7 and the inner groups are the shortcut numbers. Just to verify that the code doesn't rely on any particular ordering.
# series = {
#     "SLIP": {"10": "sweep_gw_sac_10_ch5_s42", "15": "sweep_gw_sac_15_ch5_s42", "20": "sweep_gw_sac_20_ch5_s42", "25": "sweep_gw_sac_25_ch5_s42", "30": "sweep_gw_sac_30_ch5_s42"},
#     "SLIDE": {"10": "sweep_gw_sac_10_full_scale_ch7_s42", "15": "sweep_gw_sac_15_full_scale_ch7_s42", "20": "sweep_gw_sac_20_full_scale_ch7_s42", "25": "sweep_gw_sac_25_full_scale_ch7_s42", "30": "sweep_gw_sac_30_full_scale_ch7_s42"}

# }

# series = {
#     "SLAP":   {"0": "sweep_gw_routs_10_ch5_s42", "1": "sweep_mg_routs_10_ch5_s42", "2": "sweep_mg_routs_10_filter2_ch5_s42", "3": "sweep_mg_routs_10_filter3_ch5_s42"},
#     "SLIP": {"0": "sweep_gw_sac_10_ch5_s42", "1": "sweep_mg_sac_naive_ch5_s42", "2": "sweep_mg_sac_10_filter2_ch5_s42", "3": "sweep_mg_sac_10_filter3_ch5_s42"},
#     "SLIDE": {"0": "sweep_gw_sac_full_ch7_s42", "1": "sweep_mg_sac_full_ch7_s42", "2": "sweep_mg_sac_full_filter2_ch7_s42", "3": "sweep_mg_sac_full_filter3_ch7_s42"},
# }
series = None

# Plot mode:
#   "metric" — single bar per run (or paired bars if `series` is set), height = value of `metric` column
#   "time"   — stacked bar with t_heuristic_training + t_policy_training + t_evaluation
mode = "metric"

# Plot kind: "bar" or "line". Line is only supported with mode="metric"
# (works for both single `mapping` and paired `series`).
kind = "bar"

# Metric to plot (only used when mode=="metric"). Must be a numeric column in sweep_stats.csv.
# metric = "avg_steps"
# metric = "t_evaluation"
# metric = "t_policy_training"
# metric = "distance_mae_vs_true"
metric = "shortcut_frac"

# Standard-error column for error bars. Set to None to disable.
# metric_se = "avg_steps_se"
metric_se = None
# metric_se = "shortcut_frac_se"

# Set to a (experiment_name, abbreviation) tuple to plot percent improvement
# relative to it. The abbreviation is used in the y-axis label.
# improvement = 100 * (baseline - value) / baseline (higher = better for
# metrics where lower is better like avg_steps). Set to None to plot raw values.
# baseline = ("sweep_gw_sac_10_ch5_s42", "SLIP")
# baseline = ("sweep_cd_pure_plan_ch5_s42", "Pure Planning")
baseline = None

# "lower_is_better" or "higher_is_better" — only affects percent improvement
# sign convention.
direction = "lower_is_better"

title = "SLIDE Shortcut Usage on Obstacle2D"
# xtitle = "Residual Architecture"
# xtitle = "Max Number of Shortcuts"
xtitle = "Guidance System"
# ytitle = "Average episode length (steps)"
# ytitle = "Shortcut Success Rate"
# ytitle = "MAE"
# ytitle = "Time (s)"
ytitle = "Shortcut Fraction"

# Output path. Default: alongside the script.
out_path = Path(__file__).parent / "sweep_bars.png"

# Path to the stats CSV.
stats_csv = Path(__file__).parent / "sweep_stats.csv"
# ──────────────────────────────────────────────────────────────────────────────


def _load_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with open(csv_path) as f:
        return {row["run_name"]: row for row in csv.DictReader(f)}


def _get_float(row: dict[str, str], col: str) -> float:
    val = row.get(col, "")
    return float(val) if val not in (None, "") else float("nan")


def _compute_metric(
    run_name: str,
    rows: dict[str, dict[str, str]],
    base_val: float | None,
) -> tuple[float, float]:
    """Return (plot_value, plot_error) for a single run, applying baseline transform."""
    val = _get_float(rows[run_name], metric)
    err = _get_float(rows[run_name], metric_se) if metric_se else 0.0
    if base_val is not None:
        if direction == "lower_is_better":
            val = 100.0 * (base_val - val) / base_val
        else:
            val = 100.0 * (val - base_val) / base_val
        err = abs(err * (100.0 / base_val)) if metric_se else 0.0
    return val, err


def main() -> None:
    if not stats_csv.exists():
        raise FileNotFoundError(f"Stats CSV not found: {stats_csv}")
    rows = _load_rows(stats_csv)

    if series is not None and mode == "time":
        raise ValueError("Paired `series` is not supported with mode='time'.")
    if kind == "line" and mode == "time":
        raise ValueError("kind='line' is only supported with mode='metric'.")
    if kind not in ("bar", "line"):
        raise ValueError(f"kind must be 'bar' or 'line', got {kind!r}")

    all_runs: set[str] = set()
    if series is not None:
        for s_map in series.values():
            all_runs.update(s_map.values())
    else:
        all_runs.update(mapping.keys())
    missing = [name for name in all_runs if name not in rows]
    if missing:
        raise KeyError(f"Experiments not found in {stats_csv.name}: {missing}")

    baseline_name: str | None = None
    baseline_abbrev: str | None = None
    base_val: float | None = None
    if baseline is not None:
        baseline_name, baseline_abbrev = baseline
        if baseline_name not in rows:
            raise KeyError(f"Baseline '{baseline_name}' not found in {stats_csv.name}")
        base_val = _get_float(rows[baseline_name], metric)
        if base_val == 0 or not np.isfinite(base_val):
            raise ValueError(f"Baseline value for {metric} is invalid: {base_val}")

    # ── Paired metric mode ──────────────────────────────────────────────
    if series is not None:
        group_labels: list[str] = []
        for s_map in series.values():
            for k in s_map:
                if k not in group_labels:
                    group_labels.append(k)

        n_groups = len(group_labels)
        n_series = len(series)
        x = np.arange(n_groups)
        bar_width = 0.8 / n_series

        fig, ax = plt.subplots(figsize=(max(6, 1.3 * n_groups), 5))
        palette = plt.cm.tab10(np.linspace(0, 1, max(n_series, 2)))

        # Collect values for all series up-front so we can share y-scaling logic
        all_vals_list = []
        for si, (series_label, s_map) in enumerate(series.items()):
            vals = np.full(n_groups, np.nan)
            errs = np.zeros(n_groups)
            for gi, group in enumerate(group_labels):
                if group in s_map:
                    v, e = _compute_metric(s_map[group], rows, base_val)
                    vals[gi] = v
                    errs[gi] = e
            all_vals_list.append((series_label, vals, errs))

        global_vmax = max(
            (float(np.nanmax(np.abs(v))) for _, v, _ in all_vals_list if np.any(np.isfinite(v))),
            default=1.0,
        )

        for si, (series_label, vals, errs) in enumerate(all_vals_list):
            mask = np.isfinite(vals)
            if kind == "bar":
                offset = (si - (n_series - 1) / 2) * bar_width
                ax.bar(
                    x[mask] + offset,
                    vals[mask],
                    width=bar_width,
                    yerr=errs[mask] if metric_se else None,
                    capsize=3,
                    color=palette[si],
                    edgecolor="black",
                    linewidth=0.8,
                    label=series_label,
                )
                x_label = x + offset
            else:  # line
                ax.errorbar(
                    x[mask],
                    vals[mask],
                    yerr=errs[mask] if metric_se else None,
                    marker="o",
                    markersize=7,
                    linewidth=2,
                    capsize=3,
                    color=palette[si],
                    label=series_label,
                )
                x_label = x
            for gi in np.where(mask)[0]:
                v = vals[gi]
                ax.text(
                    x_label[gi],
                    v + (0.01 * global_vmax if v >= 0 else -0.01 * global_vmax),
                    f"{v:.1f}" + ("%" if baseline is not None else ""),
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=0 if n_groups <= 8 else 30, ha="center")
        ax.set_xlabel(xtitle)
        ax.set_ylabel(
            ytitle if baseline is None
            else f"% improvement over {baseline_abbrev}"
        )
        ax.set_title(title)
        if baseline is not None:
            ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {out_path}")
        return

    # ── Single-series modes (metric or time) ────────────────────────────
    labels = list(mapping.values())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))

    if mode == "time":
        # Stacked bar: heuristic training + policy training + evaluation
        phase_cols = [
            ("t_heuristic_training", "Heuristic training", "tab:blue"),
            ("t_policy_training", "Policy training", "tab:orange"),
            ("t_evaluation", "Evaluation", "tab:green"),
        ]
        segments = {
            col: np.array([_get_float(rows[name], col) for name in mapping])
            for col, _, _ in phase_cols
        }
        bottoms = np.zeros(len(labels))
        for col, label, color in phase_cols:
            vals = segments[col]
            ax.bar(
                x, vals, bottom=bottoms, label=label,
                color=color, edgecolor="black", linewidth=0.8,
            )
            bottoms = bottoms + vals

        # Numeric total on top of each bar
        for xi, total in zip(x, bottoms):
            ax.text(
                xi, total + 0.01 * bottoms.max(),
                f"{total:.0f}s",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0 if len(labels) <= 8 else 30, ha="center")
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.set_title(title)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    else:
        plot_vals = np.empty(len(mapping))
        plot_errs = np.empty(len(mapping))
        for i, name in enumerate(mapping):
            plot_vals[i], plot_errs[i] = _compute_metric(name, rows, base_val)

        mask = np.isfinite(plot_vals)
        n_missing = int((~mask).sum())
        if n_missing:
            missing_labels = [lbl for lbl, ok in zip(labels, mask) if not ok]
            print(f"[WARN] Skipping {n_missing} run(s) with NaN {metric}: {missing_labels}")

        if kind == "bar":
            ax.bar(
                x[mask], plot_vals[mask],
                yerr=plot_errs[mask] if metric_se else None,
                capsize=4,
                color="tab:blue", edgecolor="black", linewidth=0.8,
            )
        else:
            ax.errorbar(
                x[mask], plot_vals[mask],
                yerr=plot_errs[mask] if metric_se else None,
                marker="o", markersize=7, linewidth=2,
                capsize=4, color="tab:blue",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0 if len(labels) <= 8 else 30, ha="center")
        ax.set_xlabel(xtitle)
        ax.set_ylabel(
            ytitle if baseline is None
            else f"% improvement over {baseline_abbrev}"
        )
        ax.set_title(title)
        if baseline is not None:
            ax.axhline(0, color="black", linewidth=0.8)

        vmax = float(np.abs(plot_vals[mask]).max()) if mask.any() else 1.0
        for xi, v in zip(x[mask], plot_vals[mask]):
            ax.text(
                xi,
                v + (0.01 * vmax if v >= 0 else -0.01 * vmax),
                f"{v:.1f}" + ("%" if baseline is not None else ""),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
