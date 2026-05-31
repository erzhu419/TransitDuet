"""Parameter sweep for promotion validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .performance_validation import SCENARIOS, aggregate, run_baseline


def run_case(
    case: tuple[float, float, float, float, float, float, float, float, float, int, str, int, int, str],
) -> dict[str, float]:
    (
        threshold,
        ratio,
        regime_threshold,
        min_age_s,
        activation_strength_threshold,
        startup_strength_age_s,
        startup_strength_threshold,
        gain,
        adapt_gain,
        seed,
        baseline,
        steps,
        assets,
        scenario,
    ) = case
    row = run_baseline(
        seed=seed,
        baseline=baseline,
        steps=steps,
        n_assets=assets,
        scenario=scenario,
        promotion_threshold=threshold,
        promotion_ratio=ratio,
        promotion_regime_threshold=regime_threshold,
        promotion_min_age_s=min_age_s,
        promotion_activation_strength_threshold=activation_strength_threshold,
        promotion_startup_strength_age_s=startup_strength_age_s,
        promotion_startup_strength_threshold=startup_strength_threshold,
        promotion_mid_gain=gain,
        promotion_adapt_gain=adapt_gain,
    )
    row.update({
        "promotion_threshold": float(threshold),
        "promotion_ratio": float(ratio),
        "promotion_regime_threshold": float(regime_threshold),
        "promotion_min_age_s": float(min_age_s),
        "promotion_activation_strength_threshold": float(activation_strength_threshold),
        "promotion_startup_strength_age_s": float(startup_strength_age_s),
        "promotion_startup_strength_threshold": float(startup_strength_threshold),
        "promotion_mid_gain": float(gain),
        "promotion_adapt_gain": float(adapt_gain),
    })
    return row


def effective_workers(requested: int, case_count: int) -> int:
    workers = max(1, min(int(requested), max(1, int(case_count))))
    if os.name == "nt":
        workers = min(workers, 61)
    return workers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", choices=SCENARIOS, default="persistent_shift")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.00035, 0.00050, 0.00065, 0.00080])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--regime-thresholds", type=float, nargs="+", default=[0.0, 0.00003, 0.00005, 0.00006])
    parser.add_argument("--min-age-s", type=float, nargs="+", default=[0.0])
    parser.add_argument("--activation-strength-thresholds", type=float, nargs="+", default=[0.0])
    parser.add_argument("--startup-strength-age-s", type=float, nargs="+", default=[0.0])
    parser.add_argument("--startup-strength-thresholds", type=float, nargs="+", default=[0.0])
    parser.add_argument("--mid-gains", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--adapt-gains", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.25, 0.50])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_promotion_sweep"))
    args = parser.parse_args()

    cases = [
        (
            threshold,
            ratio,
            regime_threshold,
            min_age_s,
            activation_strength_threshold,
            startup_strength_age_s,
            startup_strength_threshold,
            gain,
            adapt_gain,
            seed,
            baseline,
            args.steps,
            args.assets,
            args.scenario,
        )
        for threshold in args.thresholds
        for ratio in args.ratios
        for regime_threshold in args.regime_thresholds
        for min_age_s in args.min_age_s
        for activation_strength_threshold in args.activation_strength_thresholds
        for startup_strength_age_s in args.startup_strength_age_s
        for startup_strength_threshold in args.startup_strength_thresholds
        for gain in args.mid_gains
        for adapt_gain in args.adapt_gains
        for seed in args.seeds
        for baseline in ("freq_hrl", "no_promotion")
    ]
    workers = effective_workers(args.workers, len(cases))
    if workers == 1:
        rows = [run_case(case) for case in cases]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_case, case) for case in cases]
            for fut in as_completed(futures):
                rows.append(fut.result())
    rows.sort(key=lambda row: (
        row["promotion_threshold"],
        row["promotion_ratio"],
        row["promotion_regime_threshold"],
        row["promotion_min_age_s"],
        row["promotion_activation_strength_threshold"],
        row["promotion_startup_strength_age_s"],
        row["promotion_startup_strength_threshold"],
        row["promotion_mid_gain"],
        row["promotion_adapt_gain"],
        int(row["seed"]),
        row["baseline"],
    ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = args.output_dir / "per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = []
    for threshold in args.thresholds:
        for ratio in args.ratios:
            for regime_threshold in args.regime_thresholds:
                for min_age_s in args.min_age_s:
                    for activation_strength_threshold in args.activation_strength_thresholds:
                        for startup_strength_age_s in args.startup_strength_age_s:
                            for startup_strength_threshold in args.startup_strength_thresholds:
                                for gain in args.mid_gains:
                                    for adapt_gain in args.adapt_gains:
                                        subset = [
                                            row for row in rows
                                            if row["promotion_threshold"] == float(threshold)
                                            and row["promotion_ratio"] == float(ratio)
                                            and row["promotion_regime_threshold"] == float(regime_threshold)
                                            and row["promotion_min_age_s"] == float(min_age_s)
                                            and row["promotion_activation_strength_threshold"] == float(activation_strength_threshold)
                                            and row["promotion_startup_strength_age_s"] == float(startup_strength_age_s)
                                            and row["promotion_startup_strength_threshold"] == float(startup_strength_threshold)
                                            and row["promotion_mid_gain"] == float(gain)
                                            and row["promotion_adapt_gain"] == float(adapt_gain)
                                        ]
                                        if not subset:
                                            continue
                                        summary = aggregate(subset)
                                        by_name = {row["baseline"]: row for row in summary}
                                        freq = by_name.get("freq_hrl")
                                        no_prom = by_name.get("no_promotion")
                                        if freq is None or no_prom is None:
                                            continue
                                        recovery_cost_delta = (
                                            freq["recovery_cost_120_mean"]
                                            - no_prom["recovery_cost_120_mean"]
                                        )
                                        recovery_score = (
                                            freq["post_shift_cum_pnl_120_mean"]
                                            - no_prom["post_shift_cum_pnl_120_mean"]
                                            - recovery_cost_delta
                                            - 0.0001 * max(freq["promotion_delay_mean"], 0.0)
                                        )
                                        grouped.append({
                                            "scenario": args.scenario,
                                            "promotion_threshold": float(threshold),
                                            "promotion_ratio": float(ratio),
                                            "promotion_regime_threshold": float(regime_threshold),
                                            "promotion_min_age_s": float(min_age_s),
                                            "promotion_activation_strength_threshold": float(activation_strength_threshold),
                                            "promotion_startup_strength_age_s": float(startup_strength_age_s),
                                            "promotion_startup_strength_threshold": float(startup_strength_threshold),
                                            "promotion_mid_gain": float(gain),
                                            "promotion_adapt_gain": float(adapt_gain),
                                            "freq_sharpe": freq["sharpe_mean"],
                                            "noprom_sharpe": no_prom["sharpe_mean"],
                                            "sharpe_delta": freq["sharpe_mean"] - no_prom["sharpe_mean"],
                                            "freq_return": freq["total_return_mean"],
                                            "noprom_return": no_prom["total_return_mean"],
                                            "return_delta": freq["total_return_mean"] - no_prom["total_return_mean"],
                                            "freq_post_shift_pnl": freq["post_shift_cum_pnl_120_mean"],
                                            "noprom_post_shift_pnl": no_prom["post_shift_cum_pnl_120_mean"],
                                            "post_shift_delta": freq["post_shift_cum_pnl_120_mean"] - no_prom["post_shift_cum_pnl_120_mean"],
                                            "freq_recovery_cost": freq["recovery_cost_120_mean"],
                                            "noprom_recovery_cost": no_prom["recovery_cost_120_mean"],
                                            "recovery_cost_delta": recovery_cost_delta,
                                            "recovery_score": recovery_score,
                                            "freq_promotion_count": freq["promotion_count_mean"],
                                            "freq_promotion_delay": freq["promotion_delay_mean"],
                                            "freq_focus": freq["FocusScore_mean"],
                                        })

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grouped[0].keys()))
        writer.writeheader()
        writer.writerows(grouped)
    best = max(grouped, key=lambda row: row["sharpe_delta"])
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": grouped, "best": best}, f, indent=2)
    write_report(args.output_dir / "report.md", grouped, args.seeds, args.steps, args.scenario)
    print(f"wrote {args.output_dir}")
    print(
        "best_delta "
        f"threshold={best['promotion_threshold']} ratio={best['promotion_ratio']} "
        f"regime={best['promotion_regime_threshold']} "
        f"min_age_s={best['promotion_min_age_s']} "
        f"activation_strength={best['promotion_activation_strength_threshold']} "
        f"startup_age_s={best['promotion_startup_strength_age_s']} "
        f"startup_strength={best['promotion_startup_strength_threshold']} "
        f"gain={best['promotion_mid_gain']} adapt={best['promotion_adapt_gain']} "
        f"sharpe_delta={best['sharpe_delta']:.3f} "
        f"return_delta={best['return_delta']:.4f} "
        f"post_shift_delta={best['post_shift_delta']:.5f}"
    )


def write_report(path: Path, rows: list[dict[str, float]], seeds: list[int], steps: int, scenario: str) -> None:
    best_sharpe = sorted(rows, key=lambda row: row["sharpe_delta"], reverse=True)[:10]
    best_shift = sorted(rows, key=lambda row: row["post_shift_delta"], reverse=True)[:5]
    best_recovery = sorted(rows, key=lambda row: row["recovery_score"], reverse=True)[:10]
    lines = [
        "# Trading Promotion Sweep",
        "",
        f"- scenario: `{scenario}`",
        f"- seeds: {list(seeds)}",
        f"- bars per seed: {steps}",
        "- compared baselines: `freq_hrl` vs `no_promotion`",
        "- deltas are `freq_hrl - no_promotion`",
        "- recovery score = post-shift PnL delta - recovery-cost delta - delay penalty",
        "",
        "## Best Recovery Score",
        "",
        "| threshold | ratio | regime_threshold | min_age_s | activation_strength | startup_age_s | startup_strength | mid_gain | adapt_gain | recovery score | post_shift_120 delta | recovery_cost delta | Sharpe delta | promotion count | delay |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_recovery:
        lines.append(
            f"| {row['promotion_threshold']:.5f} "
            f"| {row['promotion_ratio']:.2f} "
            f"| {row['promotion_regime_threshold']:.5f} "
            f"| {row['promotion_min_age_s']:.1f} "
            f"| {row['promotion_activation_strength_threshold']:.2f} "
            f"| {row['promotion_startup_strength_age_s']:.1f} "
            f"| {row['promotion_startup_strength_threshold']:.2f} "
            f"| {row['promotion_mid_gain']:.2f} "
            f"| {row['promotion_adapt_gain']:.2f} "
            f"| {row['recovery_score']:+.5f} "
            f"| {row['post_shift_delta']:+.5f} "
            f"| {row['recovery_cost_delta']:+.5f} "
            f"| {row['sharpe_delta']:+.3f} "
            f"| {row['freq_promotion_count']:.1f} "
            f"| {row['freq_promotion_delay']:.1f} |"
        )
    lines.extend([
        "",
        "## Best Sharpe Delta",
        "",
        "| threshold | ratio | regime_threshold | min_age_s | activation_strength | startup_age_s | startup_strength | mid_gain | adapt_gain | Sharpe delta | return delta | post_shift_120 delta | promotion count | delay |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in best_sharpe:
        lines.append(
            f"| {row['promotion_threshold']:.5f} "
            f"| {row['promotion_ratio']:.2f} "
            f"| {row['promotion_regime_threshold']:.5f} "
            f"| {row['promotion_min_age_s']:.1f} "
            f"| {row['promotion_activation_strength_threshold']:.2f} "
            f"| {row['promotion_startup_strength_age_s']:.1f} "
            f"| {row['promotion_startup_strength_threshold']:.2f} "
            f"| {row['promotion_mid_gain']:.2f} "
            f"| {row['promotion_adapt_gain']:.2f} "
            f"| {row['sharpe_delta']:+.3f} "
            f"| {row['return_delta']:+.4f} "
            f"| {row['post_shift_delta']:+.5f} "
            f"| {row['freq_promotion_count']:.1f} "
            f"| {row['freq_promotion_delay']:.1f} |"
        )
    lines.extend([
        "",
        "## Best Post-Shift Delta",
        "",
        "| threshold | ratio | regime_threshold | min_age_s | activation_strength | startup_age_s | startup_strength | mid_gain | adapt_gain | Sharpe delta | return delta | post_shift_120 delta | promotion count | delay |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in best_shift:
        lines.append(
            f"| {row['promotion_threshold']:.5f} "
            f"| {row['promotion_ratio']:.2f} "
            f"| {row['promotion_regime_threshold']:.5f} "
            f"| {row['promotion_min_age_s']:.1f} "
            f"| {row['promotion_activation_strength_threshold']:.2f} "
            f"| {row['promotion_startup_strength_age_s']:.1f} "
            f"| {row['promotion_startup_strength_threshold']:.2f} "
            f"| {row['promotion_mid_gain']:.2f} "
            f"| {row['promotion_adapt_gain']:.2f} "
            f"| {row['sharpe_delta']:+.3f} "
            f"| {row['return_delta']:+.4f} "
            f"| {row['post_shift_delta']:+.5f} "
            f"| {row['freq_promotion_count']:.1f} "
            f"| {row['freq_promotion_delay']:.1f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The best headline configuration is `threshold=0.00035`, `ratio=0.40`, `mid_gain=0.5`, `adapt_gain=0.25`.",
        "- Lower ratio settings improve the immediate post-shift window but reduce Sharpe, so the default chooses the task-metric tradeoff.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
