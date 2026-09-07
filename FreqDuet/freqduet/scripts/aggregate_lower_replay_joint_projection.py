#!/usr/bin/env python3
"""Strictly aggregate the V23 replay joint-projection audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_aggregate_gain_screen import (
    FACTORIAL_SPECS,
    TRAIN_SEEDS,
)


def _run_identity(result: dict[str, object]) -> tuple[str, int]:
    checkpoint = Path(str(result.get("checkpoint", "")))
    match = re.fullmatch(
        r"(.+)_seed([0-9]+)", checkpoint.parent.parent.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint run identity: {checkpoint}")
    return match.group(1), int(match.group(2))


def _mean_range(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("projection aggregate contains non-finite values")
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _audit_row(
    result: dict[str, object],
    *,
    config: str,
    seed: int,
) -> dict[str, object]:
    if result.get("schema") != "freqduet-replay-joint-kl-projection-v1":
        raise ValueError(f"{config} seed {seed} has the wrong schema")
    if int(result.get("checkpoint_episode", -1)) != 39:
        raise ValueError(f"{config} seed {seed} is not checkpoint episode 39")
    if result.get("action_bins_s") != [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]:
        raise ValueError(f"{config} seed {seed} has the wrong action library")
    if not np.isclose(float(result["regularity_cost_limit"]), 0.05):
        raise ValueError(f"{config} seed {seed} changed the regularity budget")
    if not np.isclose(float(result["passenger_cost_limit"]), 0.08):
        raise ValueError(f"{config} seed {seed} changed the passenger budget")
    projection = result["projection"]
    if not projection.get("joint_feasible"):
        raise ValueError(f"{config} seed {seed} has no joint frontier")
    if not projection.get("converged"):
        raise ValueError(f"{config} seed {seed} projection did not converge")
    projected = result.get("projected_policy")
    if not isinstance(projected, dict):
        raise ValueError(f"{config} seed {seed} has no projected policy")
    projected_regularity = float(projected["expected_regularity_cost_mean"])
    projected_passenger = float(projected["expected_passenger_cost_mean"])
    if projected_regularity > 0.05 + 1e-8:
        raise ValueError(f"{config} seed {seed} projection misses regularity")
    if projected_passenger > 0.08 + 1e-8:
        raise ValueError(f"{config} seed {seed} projection misses passenger")
    learned = result["learned_policy"]
    multipliers = projection["multipliers"]
    return {
        "config": config,
        "train_seed": int(seed),
        "constraint_cost_mode": str(result["constraint_cost_mode"]),
        "valid_transitions": int(result["valid_transitions"]),
        "per_state_deterministic_joint_feasible_fraction": float(
            result["per_state_deterministic_joint_feasible_fraction"]),
        "learned_regularity_cost": float(
            learned["expected_regularity_cost_mean"]),
        "learned_passenger_cost": float(
            learned["expected_passenger_cost_mean"]),
        "learned_action_mean_s": float(learned["expected_action_mean_s"]),
        "learned_entropy": float(learned["entropy_mean"]),
        "projected_regularity_cost": projected_regularity,
        "projected_passenger_cost": projected_passenger,
        "projected_action_mean_s": float(projected["expected_action_mean_s"]),
        "projected_entropy": float(projected["entropy_mean"]),
        "regularity_projection_multiplier": float(multipliers[0]),
        "passenger_projection_multiplier": float(multipliers[1]),
        "projection_iterations": int(projection["iterations"]),
        "projection_kl": float(projected["kl_from_learned_mean"]),
        "projection_probability_l1_shift": float(
            projected["probability_l1_shift_mean"]),
        "projection_argmax_changed_fraction": float(
            projected["argmax_changed_fraction"]),
        "projection_action_change_mean_s": float(
            projected["expected_action_change_mean_s"]),
        "learned_joint_budget_feasible": bool(
            float(learned["expected_regularity_cost_mean"]) <= 0.05 + 1e-8
            and float(learned["expected_passenger_cost_mean"]) <= 0.08 + 1e-8),
    }


def aggregate_joint_projection_audit(
    root: str | Path,
    *,
    specs: Iterable[tuple[str, str, str, float, bool]] = FACTORIAL_SPECS,
    train_seeds: Sequence[int] = TRAIN_SEEDS,
) -> dict[str, object]:
    root = Path(root)
    specs = list(specs)
    expected = {
        (config, int(seed))
        for config, *_ in specs
        for seed in train_seeds
    }
    results: dict[tuple[str, int], dict[str, object]] = {}
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text())
        identity = _run_identity(result)
        if identity not in expected:
            raise ValueError(f"unexpected projection result {identity} at {path}")
        if identity in results:
            raise ValueError(f"duplicate projection result {identity}")
        results[identity] = result
    missing = sorted(expected - set(results))
    if missing:
        raise ValueError(f"missing projection results: {missing}")

    rows = []
    candidates = []
    for config, allocation, update, rho, promotion_eligible in specs:
        candidate_rows = [
            _audit_row(
                results[(config, int(seed))],
                config=config,
                seed=int(seed),
            )
            for seed in train_seeds
        ]
        rows.extend(candidate_rows)
        numeric_fields = [
            key for key, value in candidate_rows[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and key != "train_seed"
        ]
        candidates.append({
            "config": config,
            "allocation": allocation,
            "dual_update_mode": update,
            "augmented_lagrangian_rho": float(rho),
            "promotion_eligible": bool(promotion_eligible),
            "run_count": len(candidate_rows),
            "learned_joint_budget_feasible_seed_fraction": float(np.mean([
                row["learned_joint_budget_feasible"]
                for row in candidate_rows
            ])),
            "metrics": {
                field: _mean_range([
                    float(row[field]) for row in candidate_rows
                ])
                for field in numeric_fields
            },
        })

    per_state_fractions = [
        float(row["per_state_deterministic_joint_feasible_fraction"])
        for row in rows
    ]
    return {
        "schema": "freqduet-replay-joint-kl-projection-aggregate-v1",
        "root": str(root),
        "candidate_count": len(candidates),
        "run_count": len(rows),
        "all_joint_frontiers_feasible": True,
        "all_projections_converged": True,
        "batch_projection_supported": True,
        "per_state_projection_supported": bool(
            min(per_state_fractions) >= 0.95),
        "per_state_deterministic_joint_feasible_fraction": _mean_range(
            per_state_fractions),
        "candidates": candidates,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    result = aggregate_joint_projection_audit(args.root)
    csv_path = args.csv or args.out.with_suffix(".csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(result["rows"]).to_csv(csv_path, index=False)
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "run_count": result["run_count"],
        "batch_projection_supported": result["batch_projection_supported"],
        "per_state_projection_supported": result[
            "per_state_projection_supported"],
        "json": str(args.out),
        "csv": str(csv_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
