"""Run a causal encoder ablation for the Freq-HRL trading protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .performance_validation import SCENARIOS, run_baseline


ENCODER_METHODS = ("ema", "fourier", "state_space", "haar_wavelet")


def aggregate_by_encoder(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metrics = [
        key for key in rows[0].keys()
        if key not in {"baseline", "seed", "scenario", "freq_method"}
        and isinstance(rows[0].get(key), (int, float, np.integer, np.floating))
    ]
    out = []
    for method in ENCODER_METHODS:
        group = [row for row in rows if row.get("freq_method") == method]
        if not group:
            continue
        summary: dict[str, Any] = {"freq_method": method, "n": len(group)}
        scenarios = sorted({str(row.get("scenario", "")) for row in group})
        if len(scenarios) == 1:
            summary["scenario"] = scenarios[0]
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(vals.mean())
            summary[f"{metric}_std"] = float(vals.std(ddof=0))
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: list[dict[str, Any]], seeds: list[int], steps: int, scenario: str) -> None:
    if not summary:
        path.write_text("# Trading Encoder Ablation\n\nNo rows.\n", encoding="utf-8")
        return
    best = max(summary, key=lambda row: row["sharpe_mean"])
    lines = [
        "# Trading Encoder Ablation",
        "",
        "Causal decomposer ablation for the same `freq_hrl` protocol.",
        "",
        f"- seeds: {seeds}",
        f"- bars per seed: {steps}",
        f"- scenario: `{scenario}`",
        f"- best Sharpe encoder: `{best['freq_method']}` ({best['sharpe_mean']:.3f})",
        "",
        "| encoder | return | Sharpe | max DD | turnover | PromotionDelay | UpperHFPower | LowerLFDrift | FocusScore |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['freq_method']} "
            f"| {row['total_return_mean']:.4f} "
            f"| {row['sharpe_mean']:.3f} "
            f"| {row['max_drawdown_mean']:.4f} "
            f"| {row['turnover_mean']:.2f} "
            f"| {row['PromotionDelay_mean']:.1f} "
            f"| {row['UpperHFPower_mean']:.4f} "
            f"| {row['LowerLFDrift_mean']:.3f} "
            f"| {row['FocusScore_mean']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_encoder_ablation(
    seeds: list[int],
    steps: int,
    assets: int,
    scenario: str,
    methods: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = tuple(methods or ENCODER_METHODS)
    rows = []
    for method in selected:
        if method not in ENCODER_METHODS:
            raise ValueError(f"unknown encoder method: {method}")
        for seed in seeds:
            rows.append(
                run_baseline(
                    seed=seed,
                    baseline="freq_hrl",
                    steps=steps,
                    n_assets=assets,
                    scenario=scenario,
                    freq_method=method,
                )
            )
    return rows, aggregate_by_encoder(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", choices=SCENARIOS, default="persistent_shift")
    parser.add_argument("--methods", nargs="+", choices=ENCODER_METHODS, default=list(ENCODER_METHODS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/trading_encoder_ablation"),
    )
    args = parser.parse_args()

    rows, summary = run_encoder_ablation(
        seeds=list(args.seeds),
        steps=args.steps,
        assets=args.assets,
        scenario=args.scenario,
        methods=list(args.methods),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"per_seed": rows, "summary": summary}, f, indent=2)
    write_report(args.output_dir / "report.md", summary, list(args.seeds), args.steps, args.scenario)

    best = max(summary, key=lambda row: row["sharpe_mean"])
    print(f"wrote {args.output_dir}")
    print(
        "best_encoder="
        f"{best['freq_method']} sharpe={best['sharpe_mean']:.3f} "
        f"return={best['total_return_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
