"""Scenario pressure-test matrix for synthetic trading validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .performance_validation import BASELINES, SCENARIOS, run_baseline


def run_case(case: tuple[str, int, str, int, int, float, float, float, float]) -> dict[str, Any]:
    (
        scenario,
        seed,
        baseline,
        steps,
        assets,
        promotion_min_age_s,
        promotion_activation_strength_threshold,
        promotion_startup_strength_age_s,
        promotion_startup_strength_threshold,
    ) = case
    return run_baseline(
        seed=seed,
        baseline=baseline,
        steps=steps,
        n_assets=assets,
        scenario=scenario,
        promotion_min_age_s=promotion_min_age_s,
        promotion_activation_strength_threshold=promotion_activation_strength_threshold,
        promotion_startup_strength_age_s=promotion_startup_strength_age_s,
        promotion_startup_strength_threshold=promotion_startup_strength_threshold,
    )


def effective_workers(requested: int, case_count: int) -> int:
    workers = max(1, min(int(requested), max(1, int(case_count))))
    if os.name == "nt":
        workers = min(workers, 61)
    return workers


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        key for key in rows[0]
        if key not in {"baseline", "seed", "scenario", "freq_method"}
        and isinstance(rows[0].get(key), (int, float, np.integer, np.floating))
    ]
    out = []
    for scenario in SCENARIOS:
        for baseline in BASELINES:
            group = [
                row for row in rows
                if row["scenario"] == scenario and row["baseline"] == baseline
            ]
            if not group:
                continue
            item = {"scenario": scenario, "baseline": baseline, "n": len(group)}
            for metric in metrics:
                vals = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
                item[f"{metric}_mean"] = float(vals.mean())
                item[f"{metric}_std"] = float(vals.std(ddof=0))
            out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    seeds: list[int],
    steps: int,
    promotion_min_age_s: float,
    promotion_activation_strength_threshold: float,
    promotion_startup_strength_age_s: float,
    promotion_startup_strength_threshold: float,
) -> None:
    lines = [
        "# Trading Pressure-Test Matrix",
        "",
        f"- scenarios: {list(SCENARIOS)}",
        f"- seeds: {list(seeds)}",
        f"- bars per seed: {steps}",
        f"- promotion min age: {promotion_min_age_s}",
        f"- promotion activation strength threshold: {promotion_activation_strength_threshold}",
        f"- promotion startup strength age: {promotion_startup_strength_age_s}",
        f"- promotion startup strength threshold: {promotion_startup_strength_threshold}",
        "- policies are deterministic heuristics, not trained RL policies",
        "",
        "## Scenario Winners",
        "",
        "| scenario | best Sharpe | best return | Freq-HRL Sharpe | Freq-HRL return | LF-only Sharpe | NoPromotion Sharpe |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in summary:
        by_scenario.setdefault(row["scenario"], []).append(row)
    for scenario in SCENARIOS:
        rows = by_scenario.get(scenario, [])
        if not rows:
            continue
        best_sharpe = max(rows, key=lambda row: row["sharpe_mean"])
        best_return = max(rows, key=lambda row: row["total_return_mean"])
        by_name = {row["baseline"]: row for row in rows}
        freq = by_name.get("freq_hrl", {})
        lf = by_name.get("lf_upper_only", {})
        nop = by_name.get("no_promotion", {})
        lines.append(
            f"| {scenario} "
            f"| {best_sharpe['baseline']} ({best_sharpe['sharpe_mean']:.3f}) "
            f"| {best_return['baseline']} ({best_return['total_return_mean']:.4f}) "
            f"| {freq.get('sharpe_mean', 0.0):.3f} "
            f"| {freq.get('total_return_mean', 0.0):.4f} "
            f"| {lf.get('sharpe_mean', 0.0):.3f} "
            f"| {nop.get('sharpe_mean', 0.0):.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This matrix checks whether the frequency-responsibility claim survives beyond the default persistent-shift setting.",
        "- `lf_upper_only` is tracked explicitly because it is close to `freq_hrl` in the default validation.",
        "- Rows where `freq_hrl` is not the scenario winner should drive the next policy or promotion tuning pass.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--promotion-min-age-s", type=float, default=0.0)
    parser.add_argument("--promotion-activation-strength-threshold", type=float, default=0.0)
    parser.add_argument("--promotion-startup-strength-age-s", type=float, default=0.0)
    parser.add_argument("--promotion-startup-strength-threshold", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_pressure_matrix"))
    args = parser.parse_args()

    cases = [
        (
            scenario,
            seed,
            baseline,
            args.steps,
            args.assets,
            args.promotion_min_age_s,
            args.promotion_activation_strength_threshold,
            args.promotion_startup_strength_age_s,
            args.promotion_startup_strength_threshold,
        )
        for scenario in args.scenarios
        for seed in args.seeds
        for baseline in args.baselines
    ]
    rows = []
    workers = effective_workers(args.workers, len(cases))
    if workers == 1:
        rows = [run_case(case) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_case, case) for case in cases]
            for fut in as_completed(futures):
                rows.append(fut.result())
    rows.sort(key=lambda row: (
        list(SCENARIOS).index(row["scenario"]),
        int(row["seed"]),
        list(BASELINES).index(row["baseline"]),
    ))
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"per_seed": rows, "summary": summary}, f, indent=2)
    write_report(
        args.output_dir / "report.md",
        summary,
        args.seeds,
        args.steps,
        args.promotion_min_age_s,
        args.promotion_activation_strength_threshold,
        args.promotion_startup_strength_age_s,
        args.promotion_startup_strength_threshold,
    )
    print(f"wrote {args.output_dir}")
    for scenario in args.scenarios:
        rows_s = [row for row in summary if row["scenario"] == scenario]
        if rows_s:
            best = max(rows_s, key=lambda row: row["sharpe_mean"])
            print(f"{scenario}: best={best['baseline']} sharpe={best['sharpe_mean']:.3f}")


if __name__ == "__main__":
    main()
