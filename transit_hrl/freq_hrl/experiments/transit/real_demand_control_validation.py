"""Replay real AFC/APC demand traces through the Transit Freq-HRL PPO control loop."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.transit.ppo_surrogate import train_transit_surrogate_ppo
from freq_hrl.experiments.transit.public_afc_demand_validation import (
    DEFAULT_AFC_ENDPOINT,
    fetch_mta_hourly_ridership,
    rows_to_station_hour_series,
)
from freq_hrl.experiments.transit.public_apc_demand_validation import (
    DEFAULT_APC_ENDPOINT,
    fetch_halifax_apc_boardings,
    rows_to_route_halfhour_series,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "base_real_ema": {
        "tracker_method": "ema",
        "plan_basis_dim": 0,
        "include_native_lower_context": False,
        "upper_decision_interval": 1,
    },
    "full_real_freqhrl": {
        "tracker_method": "dynamic_harmonic_nb",
        "plan_basis_dim": 2,
        "include_native_lower_context": True,
        "upper_decision_interval": 8,
        "promotion_forced_replan": True,
        "promotion_replan_strength_min": 0.10,
        "promotion_residual_threshold": 0.55,
        "promotion_persistence_ratio": 0.20,
        "plan_coefficient_scale_s": 1.0,
        "lower_lf_effect_filter_window": 12,
        "lower_lf_effect_filter_gain": 1.0,
        "lower_lf_raw_recenter_gain": 1.0,
        "wait_upper_weight": 0.005,
        "wait_lower_weight": 0.010,
        "wait_lower_board_credit_weight": 0.10,
        "wait_credit_control_gain": 2.0,
        "lower_lf_constraint_coef": 0.02,
        "lower_lf_constraint_target": 0.55,
        "lower_lf_dual_lr": 0.02,
        "lower_lf_objective_weight": 0.02,
    },
}


def control_objective(row: dict[str, Any]) -> float:
    return (
        float(row["reward_mean"])
        - 0.08 * float(row["LowerLFDrift"])
        - 0.04 * float(row["wait_proxy"])
        - 0.01 * float(row["hold_mean"])
    )


def _window_trace(
    series: dict[str, np.ndarray],
    *,
    seed: int,
    steps: int,
    corridors: int,
    demand_scale: float,
) -> np.ndarray:
    if not series:
        raise ValueError("no real demand series available")
    values = [np.asarray(arr, dtype=np.float64).reshape(-1) for arr in series.values()]
    min_len = min(arr.size for arr in values)
    if min_len <= 0:
        raise ValueError("empty real demand series")
    stacked = np.stack([arr[:min_len] for arr in values[:max(1, int(corridors))]], axis=1)
    if stacked.shape[1] < int(corridors):
        repeats = int(np.ceil(float(corridors) / max(float(stacked.shape[1]), 1.0)))
        stacked = np.tile(stacked, (1, repeats))
    stacked = stacked[:, :int(corridors)]
    if stacked.shape[0] < int(steps):
        repeats = int(np.ceil(float(steps) / max(float(stacked.shape[0]), 1.0)))
        stacked = np.tile(stacked, (repeats, 1))
    max_offset = max(stacked.shape[0] - int(steps), 0)
    offset = int(seed * 37) % max(max_offset + 1, 1)
    trace = stacked[offset:offset + int(steps), :].copy()
    denom = np.maximum(np.percentile(trace, 95, axis=0), 1.0)
    return np.maximum(trace / denom * float(demand_scale), 0.0)


def build_trace_map(
    series: dict[str, np.ndarray],
    seeds: list[int],
    *,
    steps: int,
    corridors: int,
    demand_scale: float,
) -> dict[int, np.ndarray]:
    return {
        int(seed): _window_trace(
            series,
            seed=int(seed),
            steps=int(steps),
            corridors=int(corridors),
            demand_scale=float(demand_scale),
        )
        for seed in seeds
    }


def load_real_demand_series(
    source: str,
    *,
    afc_cache_csv: Path | None,
    apc_cache_csv: Path | None,
    max_series: int,
    min_bins: int,
    afc_start: str,
    afc_end: str,
    apc_start: str,
    apc_end: str,
    limit: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if source == "afc":
        rows = fetch_mta_hourly_ridership(
            endpoint=DEFAULT_AFC_ENDPOINT,
            start=str(afc_start),
            end=str(afc_end),
            cache_csv=afc_cache_csv,
            limit=int(limit),
        )
        series = rows_to_station_hour_series(
            rows,
            max_series=int(max_series),
            min_hours=int(min_bins),
        )
        return series, {
            "source": "afc",
            "source_endpoint": DEFAULT_AFC_ENDPOINT,
            "start": str(afc_start),
            "end": str(afc_end),
            "rows": len(rows),
            "boundary": "AFC station entries, not onboard load or OD",
        }
    if source == "apc":
        rows = fetch_halifax_apc_boardings(
            endpoint=DEFAULT_APC_ENDPOINT,
            start=str(apc_start),
            end=str(apc_end),
            cache_csv=apc_cache_csv,
            limit=int(limit),
        )
        series = rows_to_route_halfhour_series(
            rows,
            max_series=int(max_series),
            min_bins=int(min_bins),
        )
        return series, {
            "source": "apc",
            "source_endpoint": DEFAULT_APC_ENDPOINT,
            "start": str(apc_start),
            "end": str(apc_end),
            "rows": len(rows),
            "boundary": "APC route boardings, not onboard occupancy/alighting/OD",
        }
    raise ValueError(f"unknown real demand source: {source}")


def paired_checks(rows: list[dict[str, Any]], min_pairs: int = 3) -> list[dict[str, Any]]:
    checks = []
    for metric, lower_is_better in [
        ("control_objective", False),
        ("reward_mean", False),
        ("wait_proxy", True),
        ("LowerLFDrift", True),
        ("RawLowerLFDriftAbs", True),
    ]:
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("source", "seed"),
            metric=metric,
            treatment="full_real_freqhrl",
            control="base_real_ema",
            lower_is_better=lower_is_better,
        )
        checks.append({
            "check": f"real_demand_control_{metric}",
            **stats,
            "status": claim_status(stats, min_pairs=int(min_pairs)),
        })
    return checks


def run_validation(
    output_dir: Path,
    *,
    sources: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    iterations: int,
    corridors: int,
    optimizer_seed: int,
    max_series: int,
    min_bins: int,
    demand_scale: float,
    afc_cache_csv: Path | None,
    apc_cache_csv: Path | None,
    afc_start: str,
    afc_end: str,
    apc_start: str,
    apc_end: str,
    limit: int,
    min_pairs: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    metadata: list[dict[str, Any]] = []
    all_seeds = sorted({int(seed) for seed in [*train_seeds, *eval_seeds]})
    for source_idx, source in enumerate(sources):
        series, source_meta = load_real_demand_series(
            source,
            afc_cache_csv=afc_cache_csv,
            apc_cache_csv=apc_cache_csv,
            max_series=max_series,
            min_bins=min_bins,
            afc_start=afc_start,
            afc_end=afc_end,
            apc_start=apc_start,
            apc_end=apc_end,
            limit=limit,
        )
        metadata.append({**source_meta, "series": len(series)})
        traces = build_trace_map(
            series,
            all_seeds,
            steps=int(steps),
            corridors=int(corridors),
            demand_scale=float(demand_scale),
        )
        for variant_idx, (variant, overrides) in enumerate(VARIANTS.items()):
            payload, per_seed, _ = train_transit_surrogate_ppo(
                train_seeds=list(train_seeds),
                eval_seeds=list(eval_seeds),
                steps=int(steps),
                corridors=int(corridors),
                scenario=f"real_{source}",
                iterations=int(iterations),
                seed=int(optimizer_seed) + 100 * source_idx + variant_idx,
                demand_traces=traces,
                **overrides,
            )
            payloads[f"{source}:{variant}"] = {"model": payload, "per_seed": per_seed}
            for row in per_seed:
                item = dict(row)
                item["source"] = str(source)
                item["variant"] = str(variant)
                item["control_objective"] = control_objective(item)
                all_rows.append(item)
    checks = paired_checks(all_rows, min_pairs=int(min_pairs))
    summary = {
        "rows": len(all_rows),
        "sources": list(sources),
        "train_seeds": list(train_seeds),
        "eval_seeds": list(eval_seeds),
        "steps": int(steps),
        "iterations": int(iterations),
        "corridors": int(corridors),
    }
    payload = {
        "metadata": metadata,
        "summary": summary,
        "rows": all_rows,
        "paired_checks": checks,
        "payloads": payloads,
    }
    write_outputs(output_dir, payload)
    return payload


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    rows = payload["rows"]
    if rows:
        with (output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    checks = payload["paired_checks"]
    if checks:
        with (output_dir / "paired_checks.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(checks)
    write_report(output_dir / "report.md", payload)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Real AFC/APC Demand Control Validation",
        "",
        "This replays real passenger-demand traces through the shared Transit PPO surrogate control loop.",
        "Boundary: this is real-demand control replay, not native OD/onboard-load simulation.",
        "",
        "## Sources",
        "",
        "| source | rows | series | window | boundary |",
        "|---|---:|---:|---|---|",
    ]
    for meta in payload["metadata"]:
        lines.append(
            f"| {meta['source']} | {meta['rows']} | {meta['series']} "
            f"| {meta['start']} to {meta['end']} | {meta['boundary']} |"
        )
    lines.extend([
        "",
        "## Paired Checks",
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["paired_checks"]:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['metric']} "
            f"| {row['n_common']} | {row['delta_mean']:+.4f} "
            f"| {row['delta_ci95_low']:+.4f} | {row['delta_ci95_high']:+.4f} "
            f"| {row['win_rate']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["afc", "apc"])
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 131, 151])
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--corridors", type=int, default=2)
    parser.add_argument("--optimizer-seed", type=int, default=4040)
    parser.add_argument("--max-series", type=int, default=4)
    parser.add_argument("--min-bins", type=int, default=48)
    parser.add_argument("--demand-scale", type=float, default=18.0)
    parser.add_argument("--afc-cache-csv", type=Path, default=Path("transit_hrl/data/public_afc_mta/hourly_ridership.csv"))
    parser.add_argument("--apc-cache-csv", type=Path, default=Path("transit_hrl/data/public_apc_halifax/route_boardings.csv"))
    parser.add_argument("--afc-start", default="2024-10-01T00:00:00")
    parser.add_argument("--afc-end", default="2024-10-15T00:00:00")
    parser.add_argument("--apc-start", default="2026-01-01")
    parser.add_argument("--apc-end", default="2026-02-01")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/transit_real_demand_control"))
    args = parser.parse_args()
    payload = run_validation(
        output_dir=args.output_dir,
        sources=list(args.sources),
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        steps=int(args.steps),
        iterations=int(args.iterations),
        corridors=int(args.corridors),
        optimizer_seed=int(args.optimizer_seed),
        max_series=int(args.max_series),
        min_bins=int(args.min_bins),
        demand_scale=float(args.demand_scale),
        afc_cache_csv=args.afc_cache_csv,
        apc_cache_csv=args.apc_cache_csv,
        afc_start=str(args.afc_start),
        afc_end=str(args.afc_end),
        apc_start=str(args.apc_start),
        apc_end=str(args.apc_end),
        limit=int(args.limit),
        min_pairs=int(args.min_pairs),
    )
    objective = next(row for row in payload["paired_checks"] if row["metric"] == "control_objective")
    print(
        "real_demand_control "
        f"objective_delta={objective['delta_mean']:+.4f} "
        f"status={objective['status']}"
    )


if __name__ == "__main__":
    main()
