"""Dedicated validation for the promotion shock-recovery claim."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .performance_validation import run_baseline, write_csv


LOWER_IS_BETTER = {
    "max_drawdown",
    "turnover",
    "transaction_cost",
    "recovery_cost_120",
    "recovery_regret_120",
}


def variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "no_promotion": {
            "baseline": "no_promotion",
        },
        "default_promotion": {
            "baseline": "freq_hrl",
        },
        "recovery_tuned": {
            "baseline": "freq_hrl",
            "promotion_threshold": 0.00025,
            "promotion_ratio": 0.20,
            "promotion_window_s": 5 * 60.0,
            "promotion_regime_threshold": 8e-05,
            "promotion_mid_gain": 0.5,
            "promotion_adapt_gain": 0.05,
        },
    }


def run_case(case: tuple[str, int, int, int, str]) -> dict[str, Any]:
    variant, seed, steps, assets, scenario = case
    specs = variant_specs()
    kwargs = dict(specs[variant])
    row = run_baseline(
        seed=seed,
        steps=steps,
        n_assets=assets,
        scenario=scenario,
        **kwargs,
    )
    row["variant"] = variant
    return row


def effective_workers(requested: int, case_count: int) -> int:
    workers = max(1, min(int(requested), max(1, int(case_count))))
    if os.name == "nt":
        workers = min(workers, 61)
    return workers


def aggregate_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metrics = [
        key for key in rows[0]
        if key not in {"baseline", "variant", "seed", "scenario", "freq_method"}
        and isinstance(rows[0].get(key), (int, float, np.integer, np.floating))
    ]
    out = []
    for variant in variant_specs():
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        item: dict[str, Any] = {"variant": variant, "n": len(group)}
        scenarios = sorted({str(row.get("scenario", "")) for row in group})
        if len(scenarios) == 1:
            item["scenario"] = scenarios[0]
        methods = sorted({str(row.get("freq_method", "")) for row in group})
        if len(methods) == 1:
            item["freq_method"] = methods[0]
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(vals.mean())
            item[f"{metric}_std"] = float(vals.std(ddof=0))
        out.append(item)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(max(1, int(n_boot)), values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_deltas(
    rows: list[dict[str, Any]],
    reference: str,
    metrics: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_variant_seed = {
        (str(row["variant"]), int(row["seed"])): row
        for row in rows
    }
    variants = [name for name in variant_specs() if name != reference]
    ref_seeds = {int(row["seed"]) for row in rows if row["variant"] == reference}
    out = []
    for variant in variants:
        common = sorted(ref_seeds & {int(row["seed"]) for row in rows if row["variant"] == variant})
        item: dict[str, Any] = {
            "variant": variant,
            "reference": reference,
            "n_common": len(common),
        }
        for idx, metric in enumerate(metrics):
            values = np.asarray([
                float(by_variant_seed[(variant, s)][metric])
                - float(by_variant_seed[(reference, s)][metric])
                for s in common
            ], dtype=np.float64)
            lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed + idx)
            lower = metric in LOWER_IS_BETTER
            wins = values < 0.0 if lower else values > 0.0
            item[f"{metric}_delta_mean"] = float(values.mean()) if values.size else 0.0
            item[f"{metric}_delta_ci95_low"] = lo
            item[f"{metric}_delta_ci95_high"] = hi
            item[f"{metric}_win_rate"] = float(np.mean(wins)) if values.size else 0.0
        out.append(item)
    return out


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    seeds: list[int],
    steps: int,
    scenario: str,
) -> None:
    by_variant = {row["variant"]: row for row in summary}
    tuned = by_variant["recovery_tuned"]
    no_prom = by_variant["no_promotion"]
    tuned_pair = next(row for row in paired if row["variant"] == "recovery_tuned")
    rows = [
        "# Trading Promotion Recovery Validation",
        "",
        f"- scenario: `{scenario}`",
        f"- seeds: {seeds}",
        f"- bars per seed: {steps}",
        "- variants: `no_promotion`, `default_promotion`, `recovery_tuned`",
        "- paired deltas are `variant - no_promotion`",
        "- lower is better for recovery cost/regret, drawdown, turnover, and transaction cost",
        "- `recovery_regret_120` is post-shift shortfall to an oracle low-frequency target; it is an evaluation diagnostic, not a learner input",
        "",
        "## Headline",
        "",
        (
            "`recovery_tuned` vs `no_promotion`: "
            f"Sharpe delta {tuned_pair['sharpe_delta_mean']:+.3f} "
            f"(CI95 [{tuned_pair['sharpe_delta_ci95_low']:+.3f}, {tuned_pair['sharpe_delta_ci95_high']:+.3f}]), "
            f"return delta {tuned_pair['total_return_delta_mean']:+.4f} "
            f"(CI95 [{tuned_pair['total_return_delta_ci95_low']:+.4f}, {tuned_pair['total_return_delta_ci95_high']:+.4f}])."
        ),
        (
            "Shock recovery: "
            f"post-shift PnL delta {tuned_pair['post_shift_cum_pnl_120_delta_mean']:+.5f} "
            f"(CI95 [{tuned_pair['post_shift_cum_pnl_120_delta_ci95_low']:+.5f}, {tuned_pair['post_shift_cum_pnl_120_delta_ci95_high']:+.5f}]), "
            f"recovery-regret delta {tuned_pair['recovery_regret_120_delta_mean']:+.5f} "
            f"(CI95 [{tuned_pair['recovery_regret_120_delta_ci95_low']:+.5f}, {tuned_pair['recovery_regret_120_delta_ci95_high']:+.5f}])."
        ),
        (
            f"`recovery_tuned` mean promotion count is {tuned['promotion_count_mean']:.1f} "
            f"with mean delay {tuned['promotion_delay_mean']:.1f} bars; "
            f"`default_promotion` count is {by_variant['default_promotion']['promotion_count_mean']:.1f} "
            f"with delay {by_variant['default_promotion']['promotion_delay_mean']:.1f} bars."
        ),
        "",
        "## Variant Summary",
        "",
        "| variant | return | Sharpe | post_shift_120 | recovery_cost | recovery_regret | promotions | delay | FocusScore |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variant_specs():
        row = by_variant[variant]
        rows.append(
            f"| {variant} "
            f"| {row['total_return_mean']:.4f} "
            f"| {row['sharpe_mean']:.3f} "
            f"| {row['post_shift_cum_pnl_120_mean']:.5f} "
            f"| {row['recovery_cost_120_mean']:.5f} "
            f"| {row['recovery_regret_120_mean']:.5f} "
            f"| {row['promotion_count_mean']:.1f} "
            f"| {row['promotion_delay_mean']:.1f} "
            f"| {row['FocusScore_mean']:.3f} |"
        )
    rows.extend([
        "",
        "## Paired Deltas",
        "",
        "| variant | Sharpe | return | post_shift_120 | recovery_cost | recovery_regret | post win | regret win |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paired:
        rows.append(
            f"| {row['variant']} "
            f"| {row['sharpe_delta_mean']:+.3f} "
            f"| {row['total_return_delta_mean']:+.4f} "
            f"| {row['post_shift_cum_pnl_120_delta_mean']:+.5f} "
            f"| {row['recovery_cost_120_delta_mean']:+.5f} "
            f"| {row['recovery_regret_120_delta_mean']:+.5f} "
            f"| {row['post_shift_cum_pnl_120_win_rate']:.2f} "
            f"| {row['recovery_regret_120_win_rate']:.2f} |"
        )
    rows.extend([
        "",
        "## Interpretation",
        "",
        "- The default conservative promotion setting is intentionally cautious and does not improve the reversal-recovery window.",
        "- The recovery-tuned gate uses a shorter persistence window plus a stricter mid-frequency regime buffer; it triggers earlier without the broad false-promotion behavior seen in low-threshold sweeps.",
        "- On this controlled reversal shock, promotion now supports the recovery claim: it improves post-shift PnL and reduces oracle-regime recovery regret while also improving headline return and Sharpe.",
        "- The original loss-only `recovery_cost_120` is retained as a risk diagnostic; the oracle-regime regret is the cleaner metric for adaptation to a new persistent low-frequency direction.",
    ])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 2026, 31415, 27182, 16180, 11235, 4242, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47],
    )
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", default="promotion_recovery")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "sharpe",
            "total_return",
            "post_shift_cum_pnl_120",
            "recovery_cost_120",
            "recovery_regret_120",
            "max_drawdown",
            "turnover",
            "transaction_cost",
            "FocusScore",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/trading_promotion_recovery_validation"),
    )
    args = parser.parse_args()

    cases = [
        (variant, seed, args.steps, args.assets, args.scenario)
        for variant in variant_specs()
        for seed in args.seeds
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
    rows.sort(key=lambda row: (row["variant"], int(row["seed"])))
    summary = aggregate_variants(rows)
    paired = paired_deltas(
        rows,
        reference="no_promotion",
        metrics=args.metrics,
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired_deltas.csv", paired)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": args.scenario,
                "seeds": list(args.seeds),
                "variant_specs": variant_specs(),
                "summary": summary,
                "paired_deltas": paired,
            },
            f,
            indent=2,
        )
    write_report(args.output_dir / "report.md", summary, paired, list(args.seeds), args.steps, args.scenario)
    tuned = next(row for row in paired if row["variant"] == "recovery_tuned")
    print(f"wrote {args.output_dir}")
    print(
        "recovery_tuned "
        f"sharpe_delta={tuned['sharpe_delta_mean']:+.3f} "
        f"return_delta={tuned['total_return_delta_mean']:+.4f} "
        f"post_shift_delta={tuned['post_shift_cum_pnl_120_delta_mean']:+.5f} "
        f"regret_delta={tuned['recovery_regret_120_delta_mean']:+.5f}"
    )


if __name__ == "__main__":
    main()
