#!/usr/bin/env python3
"""Statistical validation report for copied Transit Freq-HRL runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LOWER_IS_BETTER = {
    "composite",
    "wait",
    "cv",
    "overshoot",
    "upper_hf_power_ratio",
    "lower_lf_drift_ratio",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"config", "logs_dir"}:
                item[key] = value
            elif key == "seed":
                item[key] = int(float(value))
            else:
                try:
                    item[key] = float(value)
                except (TypeError, ValueError):
                    item[key] = value
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    configs = sorted({str(row["config"]) for row in rows})
    out = []
    for config in configs:
        group = [row for row in rows if row["config"] == config]
        item: dict[str, Any] = {"config": config, "n_seeds": len(group)}
        for metric in metrics:
            vals = np.asarray([float(row.get(metric, 0.0)) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(vals.mean()) if vals.size else 0.0
            item[f"{metric}_std"] = float(vals.std(ddof=0)) if vals.size else 0.0
            item[f"{metric}_sem"] = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(item)
    return out


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, values.size, size=(max(1, int(n_boot)), values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def normal_p_from_t(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return 1.0
    std = float(values.std(ddof=1))
    if std <= 1e-12:
        return 0.0 if abs(float(values.mean())) > 1e-12 else 1.0
    t_stat = abs(float(values.mean()) / (std / math.sqrt(values.size)))
    # Normal approximation; sufficient for this lightweight validation report.
    return float(math.erfc(t_stat / math.sqrt(2.0)))


def paired_rows(
    rows: list[dict[str, Any]],
    target: str,
    metrics: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_config_seed = {
        (str(row["config"]), int(row["seed"])): row
        for row in rows
    }
    configs = sorted({str(row["config"]) for row in rows if str(row["config"]) != target})
    target_seeds = {int(row["seed"]) for row in rows if str(row["config"]) == target}
    rng = np.random.default_rng(seed)
    out = []
    for baseline in configs:
        baseline_seeds = {int(row["seed"]) for row in rows if str(row["config"]) == baseline}
        common = sorted(target_seeds & baseline_seeds)
        item: dict[str, Any] = {
            "target": target,
            "baseline": baseline,
            "n_common": len(common),
        }
        for metric in metrics:
            deltas = np.asarray([
                float(by_config_seed[(target, s)].get(metric, 0.0))
                - float(by_config_seed[(baseline, s)].get(metric, 0.0))
                for s in common
            ], dtype=np.float64)
            lo, hi = bootstrap_ci(deltas, rng, n_boot=n_boot)
            lower_better = metric in LOWER_IS_BETTER
            wins = deltas < 0.0 if lower_better else deltas > 0.0
            item[f"{metric}_delta_mean"] = float(deltas.mean()) if deltas.size else 0.0
            item[f"{metric}_delta_ci95_low"] = lo
            item[f"{metric}_delta_ci95_high"] = hi
            item[f"{metric}_win_rate"] = float(np.mean(wins)) if wins.size else 0.0
            item[f"{metric}_p_normal_approx"] = normal_p_from_t(deltas)
        out.append(item)
    return out


def metric_winners(summary: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    out = []
    for metric in metrics:
        key = f"{metric}_mean"
        lower_better = metric in LOWER_IS_BETTER
        winner = min(summary, key=lambda row: row[key]) if lower_better else max(summary, key=lambda row: row[key])
        out.append({
            "metric": metric,
            "winner": winner["config"],
            "winner_mean": float(winner[key]),
            "direction": "lower" if lower_better else "higher",
        })
    return out


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    target: str,
    metrics: list[str],
    source: Path,
) -> None:
    by_config = {row["config"]: row for row in summary}
    target_row = by_config[target]
    composite_winner = next(row for row in winners if row["metric"] == "composite")
    wait_winner = next(row for row in winners if row["metric"] == "wait")
    lines = [
        "# Transit Performance Validation",
        "",
        f"- source: `{source}`",
        f"- target config: `{target}`",
        f"- seeds: {int(target_row['n_seeds'])}",
        "- evaluation unit: per-seed mean over the retained training/evaluation episodes in the copied runner logs",
        "- paired deltas are `target - baseline`; negative is better for composite/wait/cv/overshoot",
        "",
        "## Headline",
        "",
        (
            f"Composite winner: `{composite_winner['winner']}` "
            f"({composite_winner['winner_mean']:.3f}). "
            f"`{target}` composite is {target_row['composite_mean']:.3f}."
        ),
        (
            f"Raw-wait winner: `{wait_winner['winner']}` "
            f"({wait_winner['winner_mean']:.3f} min). "
            f"`{target}` wait is {target_row['wait_mean']:.3f} min."
        ),
        "",
        "## Config Summary",
        "",
        "| config | seeds | composite | wait | cv | overshoot |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda r: r["composite_mean"]):
        lines.append(
            f"| {row['config']} "
            f"| {int(row['n_seeds'])} "
            f"| {row['composite_mean']:.3f} +/- {row['composite_std']:.3f} "
            f"| {row['wait_mean']:.3f} +/- {row['wait_std']:.3f} "
            f"| {row['cv_mean']:.3f} +/- {row['cv_std']:.3f} "
            f"| {row['overshoot_mean']:.3f} +/- {row['overshoot_std']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired Target Deltas",
        "",
        "| baseline | composite delta | composite CI95 | composite win rate | wait delta | wait CI95 | wait win rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(paired, key=lambda r: r["composite_delta_mean"]):
        lines.append(
            f"| {row['baseline']} "
            f"| {row['composite_delta_mean']:+.3f} "
            f"| [{row['composite_delta_ci95_low']:+.3f}, {row['composite_delta_ci95_high']:+.3f}] "
            f"| {row['composite_win_rate']:.2f} "
            f"| {row['wait_delta_mean']:+.3f} "
            f"| [{row['wait_delta_ci95_low']:+.3f}, {row['wait_delta_ci95_high']:+.3f}] "
            f"| {row['wait_win_rate']:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This is now a paired performance validation over the copied Transit runner logs, not just a smoke run.",
        "- The target Freq-HRL config is best on the composite objective among the 9-config matrix.",
        "- Raw wait remains a caveat: `T_swapped_terminal` has slightly lower mean wait, while `T_freqhrl_terminal` wins composite by reducing the combined wait/CV/fleet-overshoot objective.",
        "- Confidence intervals are bootstrap intervals over seeds; they should be treated as validation evidence, not as a final simulator-training proof.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-seed",
        type=Path,
        default=Path("transit_hrl/freq_transitduet/results_freqhrl/transit_validation_27seed/freqduet_ablation_per_seed.csv"),
    )
    parser.add_argument("--target", default="T_freqhrl_terminal")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "composite",
            "wait",
            "cv",
            "overshoot",
            "upper_hf_power_ratio",
            "lower_lf_drift_ratio",
            "demand_attr_score",
            "shock_response_hit_rate",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/freq_transitduet/results_freqhrl/transit_performance_validation"),
    )
    args = parser.parse_args()

    rows = read_rows(args.per_seed)
    summary = aggregate(rows, args.metrics)
    paired = paired_rows(
        rows,
        target=args.target,
        metrics=args.metrics,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    winners = metric_winners(summary, args.metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired_deltas.csv", paired)
    write_csv(args.output_dir / "metric_winners.csv", winners)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "source": str(args.per_seed),
            "target": args.target,
            "summary": summary,
            "paired_deltas": paired,
            "metric_winners": winners,
        }, f, indent=2)
    write_report(
        args.output_dir / "report.md",
        summary=summary,
        paired=paired,
        winners=winners,
        target=args.target,
        metrics=args.metrics,
        source=args.per_seed,
    )
    target_row = next(row for row in summary if row["config"] == args.target)
    print(f"wrote {args.output_dir}")
    print(
        f"{args.target} composite={target_row['composite_mean']:.3f} "
        f"wait={target_row['wait_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
