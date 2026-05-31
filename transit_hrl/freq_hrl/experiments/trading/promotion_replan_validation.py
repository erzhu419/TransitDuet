"""Validate promotion-triggered high-level replanning for plan-curve control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .performance_validation import run_baseline, write_csv

METRICS = [
    "total_return",
    "sharpe",
    "post_shift_cum_pnl_120",
    "recovery_cost_120",
    "recovery_regret_120",
    "promotion_count",
    "plan_curve_decisions",
    "plan_curve_forced_replans",
    "plan_curve_reuse_ratio",
    "plan_curve_desired_gap_mean",
    "LowerLFDrift",
]


def variants(strength_min: float) -> dict[str, dict[str, Any]]:
    common = {
        "baseline": "freq_hrl",
        "scenario": "promotion_recovery",
        "promotion_threshold": 0.00025,
        "promotion_ratio": 0.20,
        "promotion_window_s": 5 * 60.0,
        "promotion_regime_threshold": 8e-05,
        "promotion_mid_gain": 0.5,
        "promotion_adapt_gain": 0.05,
        "promotion_residual_plan_gain": 0.5,
        "promotion_speed_boost": 0.05,
        "plan_curve_enable": True,
        "plan_curve_horizon_s": 15 * 60.0,
        "plan_curve_replan_interval_s": 5 * 60.0,
        "plan_curve_basis_dim": 2,
        "plan_curve_desired_change_threshold": 0.25,
    }
    return {
        "interval_plan": {
            **common,
            "promotion_force_replan": False,
            "promotion_replan_strength_min": float(strength_min),
        },
        "promotion_replan": {
            **common,
            "promotion_force_replan": True,
            "promotion_replan_strength_min": float(strength_min),
        },
    }


def run_variant(seed: int, steps: int, assets: int, variant: str, strength_min: float) -> dict[str, Any]:
    spec = dict(variants(strength_min)[variant])
    row = run_baseline(seed=seed, steps=steps, n_assets=assets, **spec)
    row["variant"] = variant
    return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in ("interval_plan", "promotion_replan"):
        group = [row for row in rows if row["variant"] == variant]
        item: dict[str, Any] = {"variant": variant, "n": len(group)}
        for metric in METRICS:
            vals = np.asarray([float(row.get(metric, 0.0)) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(vals.mean()) if vals.size else 0.0
            item[f"{metric}_std"] = float(vals.std(ddof=0)) if vals.size else 0.0
        out.append(item)
    return out


def paired_delta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["variant"], int(row["seed"])): row for row in rows}
    seeds = sorted(
        {int(row["seed"]) for row in rows if row["variant"] == "interval_plan"}
        & {int(row["seed"]) for row in rows if row["variant"] == "promotion_replan"}
    )
    out: dict[str, Any] = {"variant": "promotion_replan", "reference": "interval_plan", "n_common": len(seeds)}
    for metric in METRICS:
        vals = np.asarray([
            float(by_key[("promotion_replan", seed)].get(metric, 0.0))
            - float(by_key[("interval_plan", seed)].get(metric, 0.0))
            for seed in seeds
        ], dtype=np.float64)
        out[f"{metric}_delta_mean"] = float(vals.mean()) if vals.size else 0.0
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: list[dict[str, Any]], delta: dict[str, Any], seeds: list[int]) -> None:
    by_variant = {row["variant"]: row for row in summary}
    interval = by_variant["interval_plan"]
    replanned = by_variant["promotion_replan"]
    lines = [
        "# Promotion Replan Validation",
        "",
        f"- seeds: {seeds}",
        "- scenario: `promotion_recovery`",
        "- comparison: interval-only plan curve vs promotion-triggered forced replan",
        f"- forced replan count mean: {replanned['plan_curve_forced_replans_mean']:.2f}",
        f"- return delta: {delta['total_return_delta_mean']:+.4f}",
        f"- Sharpe delta: {delta['sharpe_delta_mean']:+.3f}",
        f"- post-shift-120 PnL delta: {delta['post_shift_cum_pnl_120_delta_mean']:+.5f}",
        f"- recovery-regret delta: {delta['recovery_regret_120_delta_mean']:+.5f}",
        f"- LowerLFDrift delta: {delta['LowerLFDrift_delta_mean']:+.4f}",
        "",
        "| variant | return | Sharpe | post_shift_120 | recovery_regret | forced_replans | plan_reuse | LowerLFDrift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (interval, replanned):
        lines.append(
            f"| {row['variant']} "
            f"| {row['total_return_mean']:.4f} "
            f"| {row['sharpe_mean']:.3f} "
            f"| {row['post_shift_cum_pnl_120_mean']:.5f} "
            f"| {row['recovery_regret_120_mean']:.5f} "
            f"| {row['plan_curve_forced_replans_mean']:.2f} "
            f"| {row['plan_curve_reuse_ratio_mean']:.3f} "
            f"| {row['LowerLFDrift_mean']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--promotion-replan-strength-min", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_promotion_replan"))
    args = parser.parse_args()
    rows = [
        run_variant(
            seed=int(seed),
            steps=int(args.steps),
            assets=int(args.assets),
            variant=variant,
            strength_min=float(args.promotion_replan_strength_min),
        )
        for seed in args.seeds
        for variant in ("interval_plan", "promotion_replan")
    ]
    summary = aggregate(rows)
    delta = paired_delta(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_rows(args.output_dir / "summary.csv", summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"per_seed": rows, "summary": summary, "paired_delta": delta}, f, indent=2)
    write_report(args.output_dir / "report.md", summary, delta, list(args.seeds))
    print(f"wrote {args.output_dir}")
    print(
        "promotion_replan "
        f"return_delta={delta['total_return_delta_mean']:+.4f} "
        f"post_shift_delta={delta['post_shift_cum_pnl_120_delta_mean']:+.5f} "
        f"forced={next(row for row in summary if row['variant'] == 'promotion_replan')['plan_curve_forced_replans_mean']:.2f}"
    )


if __name__ == "__main__":
    main()
