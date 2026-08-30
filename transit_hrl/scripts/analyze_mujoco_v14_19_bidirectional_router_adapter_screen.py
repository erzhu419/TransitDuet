#!/usr/bin/env python3
"""Analyze the frozen v14.19 bidirectional router adapter screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scripts import mujoco_v14_19_bidirectional_router_adapter_screen_spec as spec


ROOT = Path(__file__).resolve().parents[1]


def cell_relative_dir(run_name: str, environment: str, seed: int) -> Path:
    return Path("results") / run_name / "cells" / environment / f"replicate_{seed}"


def _candidate_registry(payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    if payload.get("profile") != spec.PROFILE:
        raise ValueError("v14.19 probe profile does not match the frozen screen")
    if tuple(map(float, payload.get("gains", []))) != spec.ACTOR_GAINS:
        raise ValueError("v14.19 actor gains do not match the frozen screen")
    if tuple(map(float, payload.get("router_strengths", []))) != spec.ROUTER_STRENGTHS:
        raise ValueError("v14.19 router grid does not match the frozen screen")
    candidates = [
        dict(item) for item in payload.get("candidates", [])
        if float(item["upper_gain"]) == 1.0
        and float(item["lower_gain"]) == 1.0
    ]
    registry = {
        float(item["router_strength"]): item for item in candidates
    }
    if (
        len(candidates) != len(spec.ROUTER_STRENGTHS)
        or len(registry) != len(candidates)
        or set(registry) != set(spec.ROUTER_STRENGTHS)
    ):
        raise ValueError("v14.19 candidate registry is incomplete or duplicated")
    return registry


def _selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    strength = float(candidate["router_strength"])
    return (
        float(candidate["frequency_violation_merit"]),
        float(candidate["worst_frequency_violation"]),
        float(candidate["frequency_violation_count"]),
        abs(strength - spec.BASELINE_ROUTER_STRENGTH),
        strength,
    )


def analyze_payloads(
    payloads: Iterable[tuple[str, int, dict[str, Any]]],
) -> dict[str, Any]:
    cells = []
    seen = set()
    for environment, seed, payload in payloads:
        key = (str(environment), int(seed))
        if key in seen:
            raise ValueError(f"duplicate v14.19 cell: {key}")
        seen.add(key)
        registry = _candidate_registry(payload)
        baseline = registry[spec.BASELINE_ROUTER_STRENGTH]
        baseline_merit = float(baseline["frequency_violation_merit"])
        if int(baseline["reward_violation_count"]) != 0:
            raise ValueError(f"v14.19 baseline is not reward-safe: {key}")
        candidates = []
        for strength in spec.ROUTER_STRENGTHS:
            raw = registry[strength]
            merit = float(raw["frequency_violation_merit"])
            row = {
                "router_strength": float(strength),
                "reward_violation_count": int(raw["reward_violation_count"]),
                "frequency_violation_count": int(
                    raw["frequency_violation_count"]
                ),
                "frequency_violation_merit": merit,
                "worst_frequency_violation": float(
                    raw["worst_frequency_violation"]
                ),
                "eligible": bool(
                    strength != spec.BASELINE_ROUTER_STRENGTH
                    and int(raw["reward_violation_count"]) == 0
                    and merit < baseline_merit - spec.STRICT_IMPROVEMENT_TOLERANCE
                ),
                "relative_merit_reduction": (
                    (baseline_merit - merit) / baseline_merit
                    if baseline_merit > 0.0
                    else (0.0 if merit == baseline_merit else -1.0)
                ),
            }
            candidates.append(row)
        eligible = sorted(
            (row for row in candidates if row["eligible"]),
            key=_selection_key,
        )
        selected = eligible[0] if eligible else None
        cells.append({
            "environment": key[0],
            "optimizer_seed": key[1],
            "baseline_frequency_violation_merit": baseline_merit,
            "eligible_candidate_count": len(eligible),
            "selected_candidate": selected,
            "candidates": candidates,
        })

    expected = {
        (environment, int(seed))
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    }
    if seen != expected:
        raise ValueError(
            "v14.19 cell registry mismatch: "
            f"missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
        )
    selected = [cell["selected_candidate"] for cell in cells]
    complete = all(candidate is not None for candidate in selected)
    reductions = [
        float(candidate["relative_merit_reduction"])
        for candidate in selected if candidate is not None
    ]
    strengths = [
        float(candidate["router_strength"])
        for candidate in selected if candidate is not None
    ]
    direction_counts = Counter(
        "lower" if strength < spec.BASELINE_ROUTER_STRENGTH else "higher"
        for strength in strengths
    )
    all_reward_safe = all(
        int(row["reward_violation_count"]) == 0
        for cell in cells for row in cell["candidates"]
    )
    return {
        "analysis_version": "mujoco_v14_19_router_adapter_analysis_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cells),
        "status": (
            "bidirectional_router_adapter_mechanism_supported"
            if complete else "bidirectional_router_adapter_mechanism_not_supported"
        ),
        "supported_cell_count": sum(candidate is not None for candidate in selected),
        "all_grid_candidates_reward_safe": all_reward_safe,
        "selected_strength_counts": {
            str(key): value for key, value in sorted(Counter(strengths).items())
        },
        "selected_direction_counts": dict(sorted(direction_counts.items())),
        "minimum_selected_relative_merit_reduction": (
            min(reductions) if reductions else None
        ),
        "median_selected_relative_merit_reduction": (
            statistics.median(reductions) if reductions else None
        ),
        "mean_selected_relative_merit_reduction": (
            statistics.fmean(reductions) if reductions else None
        ),
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
                raise FileNotFoundError(f"missing v14.19 probe result: {path}")
            payloads.append((environment, seed, json.loads(path.read_text())))
    result = analyze_payloads(payloads)
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "bidirectional_router_adapter_screen.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (target / "bidirectional_router_adapter_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "environment", "optimizer_seed", "selected", "router_strength",
            "reward_violation_count", "frequency_violation_count",
            "frequency_violation_merit", "worst_frequency_violation",
            "eligible", "relative_merit_reduction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in result["cells"]:
            selected_strength = (
                None if cell["selected_candidate"] is None
                else cell["selected_candidate"]["router_strength"]
            )
            for candidate in cell["candidates"]:
                writer.writerow({
                    "environment": cell["environment"],
                    "optimizer_seed": cell["optimizer_seed"],
                    "selected": candidate["router_strength"] == selected_strength,
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
        "supported_cell_count": result["supported_cell_count"],
        "selected_strength_counts": result["selected_strength_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
