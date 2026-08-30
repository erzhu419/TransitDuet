#!/usr/bin/env python3
"""Analyze the frozen 3x3 MuJoCo v14.18 router mechanism screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from scripts import mujoco_v14_18_router_mechanism_screen_spec as spec


ROOT = Path(__file__).resolve().parents[1]


def cell_relative_dir(run_name: str, environment: str, seed: int) -> Path:
    return Path("results") / run_name / "cells" / environment / f"replicate_{seed}"


def _candidate_registry(payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    if payload.get("profile") != spec.PROFILE:
        raise ValueError("router probe profile does not match the frozen screen")
    if tuple(map(float, payload.get("gains", []))) != spec.ACTOR_GAINS:
        raise ValueError("router probe actor gains do not match the frozen screen")
    if tuple(map(float, payload.get("router_strengths", []))) != spec.ROUTER_STRENGTHS:
        raise ValueError("router probe strength grid does not match the frozen screen")
    candidates = list(payload.get("candidates", []))
    registry = {
        float(item["router_strength"]): dict(item)
        for item in candidates
        if float(item["upper_gain"]) == 1.0
        and float(item["lower_gain"]) == 1.0
    }
    if set(registry) != set(spec.ROUTER_STRENGTHS):
        raise ValueError("router probe candidate registry is incomplete or duplicated")
    return registry


def _relative_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate == baseline else -1.0
    return (baseline - candidate) / baseline


def analyze_payloads(
    payloads: Iterable[tuple[str, int, dict[str, Any]]],
) -> dict[str, Any]:
    cells = []
    seen = set()
    for environment, seed, payload in payloads:
        key = (str(environment), int(seed))
        if key in seen:
            raise ValueError(f"duplicate v14.18 cell: {key}")
        seen.add(key)
        registry = _candidate_registry(payload)
        baseline = registry[spec.BASELINE_ROUTER_STRENGTH]
        if int(baseline["reward_violation_count"]) != 0:
            raise ValueError(f"v14.18 baseline is not reward-safe: {key}")
        baseline_merit = float(baseline["frequency_violation_merit"])
        rows = []
        for strength in spec.ROUTER_STRENGTHS:
            candidate = registry[strength]
            merit = float(candidate["frequency_violation_merit"])
            reward_safe = int(candidate["reward_violation_count"]) == 0
            strict_improvement = (
                merit < baseline_merit - spec.STRICT_IMPROVEMENT_TOLERANCE
            )
            rows.append({
                "router_strength": float(strength),
                "reward_violation_count": int(
                    candidate["reward_violation_count"]
                ),
                "frequency_violation_count": int(
                    candidate["frequency_violation_count"]
                ),
                "frequency_violation_merit": merit,
                "worst_frequency_violation": float(
                    candidate["worst_frequency_violation"]
                ),
                "reward_safe": reward_safe,
                "strict_merit_improvement": strict_improvement,
                "relative_merit_reduction": _relative_reduction(
                    baseline_merit, merit
                ),
            })
        cells.append({
            "environment": key[0],
            "optimizer_seed": key[1],
            "discovery_cell": key == spec.DISCOVERY_CELL,
            "baseline_frequency_violation_merit": baseline_merit,
            "candidates": rows,
        })

    expected = {
        (environment, int(seed))
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"v14.18 cell registry mismatch: missing={missing}, extra={extra}")

    aggregates = []
    for strength in spec.ROUTER_STRENGTHS:
        rows = [
            next(
                row for row in cell["candidates"]
                if row["router_strength"] == strength
            )
            for cell in cells
        ]
        reductions = [float(row["relative_merit_reduction"]) for row in rows]
        reward_safe_count = sum(bool(row["reward_safe"]) for row in rows)
        improvement_count = sum(
            bool(row["strict_merit_improvement"]) for row in rows
        )
        joint_count = sum(
            bool(row["reward_safe"] and row["strict_merit_improvement"])
            for row in rows
        )
        aggregates.append({
            "router_strength": float(strength),
            "cell_count": len(rows),
            "reward_safe_count": reward_safe_count,
            "strict_merit_improvement_count": improvement_count,
            "joint_success_count": joint_count,
            "universal_qualified": joint_count == spec.EXPECTED_CELL_COUNT,
            "minimum_relative_merit_reduction": min(reductions),
            "median_relative_merit_reduction": statistics.median(reductions),
            "mean_relative_merit_reduction": statistics.fmean(reductions),
            "total_frequency_violation_count": sum(
                int(row["frequency_violation_count"]) for row in rows
            ),
            "maximum_worst_frequency_violation": max(
                float(row["worst_frequency_violation"]) for row in rows
            ),
        })

    qualified = [
        row for row in aggregates
        if row["universal_qualified"]
        and row["router_strength"] != spec.BASELINE_ROUTER_STRENGTH
    ]
    qualified.sort(key=lambda row: (
        float(row["minimum_relative_merit_reduction"]),
        float(row["median_relative_merit_reduction"]),
        float(row["mean_relative_merit_reduction"]),
        -int(row["total_frequency_violation_count"]),
        -float(row["router_strength"]),
    ), reverse=True)
    nominated = qualified[0] if qualified else None
    untouched = [cell for cell in cells if not cell["discovery_cell"]]
    return {
        "analysis_version": "mujoco_v14_18_router_mechanism_analysis_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cells),
        "untouched_cell_count": len(untouched),
        "status": (
            "global_router_strength_nominated"
            if nominated is not None
            else "no_global_router_strength_nominated"
        ),
        "nominated_global_router_strength": (
            None if nominated is None else nominated["router_strength"]
        ),
        "nominated_aggregate": nominated,
        "strength_aggregates": aggregates,
        "cells": sorted(
            cells, key=lambda row: (row["environment"], row["optimizer_seed"])
        ),
    }


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    payloads = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            path = ROOT / cell_relative_dir(run_name, environment, seed) / "probe.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing v14.18 probe result: {path}")
            payloads.append((environment, seed, json.loads(path.read_text())))
    result = analyze_payloads(payloads)
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "router_mechanism_screen.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (target / "router_mechanism_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "environment", "optimizer_seed", "discovery_cell",
            "router_strength", "reward_violation_count",
            "frequency_violation_count", "frequency_violation_merit",
            "worst_frequency_violation", "reward_safe",
            "strict_merit_improvement", "relative_merit_reduction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in result["cells"]:
            for candidate in cell["candidates"]:
                writer.writerow({
                    "environment": cell["environment"],
                    "optimizer_seed": cell["optimizer_seed"],
                    "discovery_cell": cell["discovery_cell"],
                    **candidate,
                })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_name, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "cell_count": result["cell_count"],
        "nominated_global_router_strength": result[
            "nominated_global_router_strength"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
