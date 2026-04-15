"""Generate a LaTeX table of sweep statistics from sweep_stats.csv.

Columns: environment, method, success_rate ± SE, avg_steps ± SE, shortcut_frac ± SE.

For ch5, each (env, method) has a single run and is used as-is.
For ch6/ch7, each (env, method) has multiple variants (unif, ucb1, ucb100,
nores, nograd, ...). This script picks the variant with the lowest avg_steps.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ENV_ORDER = ["o2", "cd", "ct", "ot"]
ENV_LABEL = {
    "o2": "Obstacle2D",
    "cd": "ClutteredDrawer",
    "ct": "CleanupTable",
    "ot": "ObstacleTower",
}
METHOD_ORDER = [
    ("pure_plan", "Pure planning"),
    ("none", "None"),
    ("routs", "Rollouts"),
    ("srouts", "Smart rollouts"),
    ("sac", "SAC"),
    ("dsac", "DSAC"),
    ("crl", "CRL"),
    ("cmd", "CMD"),
]
METHOD_KEYS = {m for m, _ in METHOD_ORDER}


def parse_run_name(run_name: str, chapter: str) -> tuple[str, str] | None:
    """Parse 'sweep_<env>_<method>[_variant]_<chapter>_s42' → (env, method).

    chapter is one of 'ch5', 'ch6', 'ch7'.
    Returns None if the run does not match.
    """
    name = run_name[len("sweep_"):] if run_name.startswith("sweep_") else run_name
    suffix = f"_{chapter}_s42"
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]

    env = None
    for e in ENV_ORDER:
        if stem.startswith(e + "_"):
            env = e
            break
    if env is None:
        return None
    rest = stem[len(env) + 1:]

    # rest is either "<method>" or "<method>_<variant>". Match the longest
    # method key that is a prefix of rest (handles pure_plan vs pure, etc.).
    for method in sorted(METHOD_KEYS, key=len, reverse=True):
        if rest == method or rest.startswith(method + "_"):
            return env, method
    return None


def fmt(value: str, se: str, decimals: int = 2) -> str:
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "--"
    if math.isnan(v):
        return "--"
    try:
        s = float(se)
    except (ValueError, TypeError):
        return f"{v:.{decimals}f}"
    if math.isnan(s):
        return f"{v:.{decimals}f}"
    return f"${v:.{decimals}f} \\pm {s:.{decimals}f}$"


def _score(row: dict[str, str]) -> tuple[float, float] | None:
    """Sort key for picking the 'best' variant.

    Higher shortcut_frac is better; within ties, lower avg_steps is better.
    Returned tuple is (-shortcut_frac, avg_steps); smaller is better.
    """
    try:
        frac = float(row["shortcut_frac"])
        steps = float(row["avg_steps"])
    except (ValueError, TypeError):
        return None
    if math.isnan(frac) or math.isnan(steps):
        return None
    return (-frac, steps)


def load_best(csv_path: str, chapters: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    """Return best run per (env, method) across all chapters.

    'Best' = highest shortcut_frac, tie-broken by lowest avg_steps.
    """
    best: dict[tuple[str, str], dict[str, str]] = {}
    best_score: dict[tuple[str, str], tuple[float, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = None
            for chapter in chapters:
                parsed = parse_run_name(row["run_name"], chapter)
                if parsed is not None:
                    break
            if parsed is None:
                continue
            score = _score(row)
            if score is None:
                continue
            if parsed not in best_score or score < best_score[parsed]:
                best[parsed] = row
                best_score[parsed] = score
    return best


def render_table(rows: dict[tuple[str, str], dict[str, str]], caption_chapter: str) -> str:
    lines: list[str] = []
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Environment & Method & Success rate & Avg.\\ steps & Shortcut frac.\\ \\\\")
    lines.append("\\midrule")

    for env in ENV_ORDER:
        first_in_env = True
        wrote_any = False
        for method_key, method_label in METHOD_ORDER:
            row = rows.get((env, method_key))
            if row is None:
                continue
            wrote_any = True
            env_cell = ENV_LABEL[env] if first_in_env else ""
            first_in_env = False
            success = fmt(row["success_rate"], row["success_rate_se"], decimals=2)
            steps = fmt(row["avg_steps"], row["avg_steps_se"], decimals=1)
            shortcut = fmt(row["shortcut_frac"], row["shortcut_frac_se"], decimals=2)
            best_run = row["run_name"].replace("_", "\\_")
            lines.append(
                f"{env_cell} & {method_label} & {success} & {steps} & {shortcut} \\\\"
            )
        if wrote_any:
            lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="experiments/sweep_stats.csv")
    parser.add_argument("--chapters", nargs="+", choices=["ch5", "ch6", "ch7"],
                        default=["ch5"],
                        help="Which chapter(s) to include (best across them)")
    parser.add_argument("--output", default=None, help="Optional .tex output path")
    args = parser.parse_args()

    rows = load_best(args.csv, args.chapters)
    tex = render_table(rows, "+".join(args.chapters))

    if args.output:
        Path(args.output).write_text(tex + "\n")
        print(f"Wrote {args.output}")
    else:
        print(tex)


if __name__ == "__main__":
    main()
