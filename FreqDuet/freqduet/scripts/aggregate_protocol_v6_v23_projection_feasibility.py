#!/usr/bin/env python3
"""Strictly aggregate the 16-checkpoint V23 projection-feasibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.audit_lower_replay_v23_projection_feasibility import (
    MINIBATCH_COUNT,
    MINIBATCH_SAMPLE_SEED,
    MINIBATCH_SIZE,
    PASSENGER_REPLAY_TARGET,
    REGULARITY_REPLAY_TARGET,
)
from scripts.audit_protocol_v6_aggregate_gain_screen import (
    FACTORIAL_SPECS,
    TRAIN_SEEDS,
)


AGGREGATE_SPECS = tuple(
    spec for spec in FACTORIAL_SPECS if spec[1] == "aggregate")


def _metric(summary: dict[str, object], name: str, bound: str) -> float:
    value = summary.get("metrics", {}).get(name)
    if not isinstance(value, dict) or value.get(bound) is None:
        return float("nan")
    return float(value[bound])


def aggregate_v23_projection_feasibility(
    root: str | Path,
    *,
    specs: Sequence[tuple[str, str, str, float, bool]] = AGGREGATE_SPECS,
    train_seeds: Sequence[int] = TRAIN_SEEDS,
) -> dict[str, object]:
    root = Path(root)
    expected = {
        (config, int(seed))
        for config, *_ in specs
        for seed in train_seeds
    }
    results: dict[tuple[str, int], dict[str, object]] = {}
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text())
        pair = (str(result.get("config")), int(result.get("train_seed", -1)))
        if pair not in expected:
            raise ValueError(f"unexpected V23 feasibility result {pair}")
        if pair in results:
            raise ValueError(f"duplicate V23 feasibility result {pair}")
        results[pair] = result
    missing = sorted(expected - set(results))
    if missing:
        raise ValueError(f"missing V23 feasibility results: {missing}")

    rows = []
    candidate_summaries = []
    for config, allocation, update, rho, _ in specs:
        candidate_rows = []
        for seed in train_seeds:
            result = results[(config, int(seed))]
            if result.get("schema") != (
                    "freqduet-v23-projection-feasibility-v1"):
                raise ValueError(f"wrong schema for {config} seed {seed}")
            if result.get("constraint_cost_mode") != (
                    "hf_aggregate_gain_shortfall_v4"):
                raise ValueError(f"wrong cost mode for {config} seed {seed}")
            if int(result.get("checkpoint_episode", -1)) != 39:
                raise ValueError(f"wrong checkpoint for {config} seed {seed}")
            if result.get("action_bins_s") != [
                    0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]:
                raise ValueError(f"wrong actions for {config} seed {seed}")
            expected_scalars = {
                "configured_regularity_budget": 0.05,
                "configured_passenger_budget": 0.08,
                "locked_regularity_replay_target": REGULARITY_REPLAY_TARGET,
                "locked_passenger_replay_target": PASSENGER_REPLAY_TARGET,
            }
            for key, expected_value in expected_scalars.items():
                if not np.isclose(float(result.get(key, np.nan)), expected_value):
                    raise ValueError(f"{config} seed {seed} changed {key}")
            minibatches = result.get("minibatches", {})
            if (
                    int(minibatches.get("batch_size", -1)) != MINIBATCH_SIZE
                    or int(minibatches.get("batch_count", -1))
                    != MINIBATCH_COUNT
                    or int(minibatches.get("sample_seed", -1))
                    != MINIBATCH_SAMPLE_SEED):
                raise ValueError(
                    f"{config} seed {seed} changed minibatch protocol")
            full = result.get("full_replay", {})
            minibatch_counts_pass = all(
                int(minibatches.get(key, -1)) == MINIBATCH_COUNT
                for key in (
                    "pass_count", "joint_feasible_count",
                    "converged_count", "budget_satisfied_count"))
            computed_pass = bool(
                full.get("passes") is True
                and minibatches.get("passes") is True
                and minibatch_counts_pass)
            if bool(result.get("passes")) != computed_pass:
                raise ValueError(
                    f"{config} seed {seed} has inconsistent pass fields")
            if computed_pass and (
                    float(full["projected_regularity_cost"])
                    > REGULARITY_REPLAY_TARGET + 1e-8
                    or float(full["projected_passenger_cost"])
                    > PASSENGER_REPLAY_TARGET + 1e-8
                    or _metric(
                        minibatches, "projected_regularity_cost", "maximum")
                    > REGULARITY_REPLAY_TARGET + 1e-8
                    or _metric(
                        minibatches, "projected_passenger_cost", "maximum")
                    > PASSENGER_REPLAY_TARGET + 1e-8):
                raise ValueError(
                    f"{config} seed {seed} exceeds a locked target")
            row = {
                "config": config,
                "train_seed": int(seed),
                "dual_update_mode": update,
                "augmented_lagrangian_rho": float(rho),
                "passes": bool(result.get("passes")),
                "full_replay_passes": bool(full.get("passes")),
                "full_replay_joint_feasible": bool(
                    full.get("joint_feasible")),
                "full_replay_converged": bool(full.get("converged")),
                "full_replay_learned_regularity_cost": float(
                    full.get("learned_regularity_cost", np.nan)),
                "full_replay_learned_passenger_cost": float(
                    full.get("learned_passenger_cost", np.nan)),
                "full_replay_projected_regularity_cost": float(
                    full.get("projected_regularity_cost", np.nan)),
                "full_replay_projected_passenger_cost": float(
                    full.get("projected_passenger_cost", np.nan)),
                "full_replay_projection_action_change_mean_s": float(
                    full.get("projection_action_change_mean_s", np.nan)),
                "full_replay_projection_kl": float(
                    full.get("projection_kl", np.nan)),
                "minibatch_pass_count": int(
                    minibatches.get("pass_count", -1)),
                "minibatch_joint_feasible_count": int(
                    minibatches.get("joint_feasible_count", -1)),
                "minibatch_converged_count": int(
                    minibatches.get("converged_count", -1)),
                "minibatch_projected_regularity_max": _metric(
                    minibatches, "projected_regularity_cost", "maximum"),
                "minibatch_projected_passenger_max": _metric(
                    minibatches, "projected_passenger_cost", "maximum"),
                "minibatch_projection_action_change_mean_s": _metric(
                    minibatches, "projection_action_change_mean_s", "mean"),
                "minibatch_projection_kl_mean": _metric(
                    minibatches, "projection_kl", "mean"),
            }
            rows.append(row)
            candidate_rows.append(row)
        candidate_summaries.append({
            "config": config,
            "allocation": allocation,
            "dual_update_mode": update,
            "augmented_lagrangian_rho": float(rho),
            "checkpoint_count": len(candidate_rows),
            "pass_count": int(sum(row["passes"] for row in candidate_rows)),
            "minimum_minibatch_pass_count": int(min(
                row["minibatch_pass_count"] for row in candidate_rows)),
        })

    passed = all(row["passes"] for row in rows)
    return {
        "schema": "freqduet-v23-projection-feasibility-aggregate-v1",
        "status": "pass" if passed else "no_pass",
        "v23_training_projection_supported": bool(passed),
        "checkpoint_count": len(rows),
        "minibatch_count": len(rows) * MINIBATCH_COUNT,
        "candidate_summaries": candidate_summaries,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = aggregate_v23_projection_feasibility(args.root)
    csv_path = args.csv or args.out.with_suffix(".csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(result["rows"]).to_csv(csv_path, index=False)
    print(json.dumps({
        "status": result["status"],
        "checkpoint_count": result["checkpoint_count"],
        "minibatch_count": result["minibatch_count"],
        "json": str(args.out),
        "csv": str(csv_path),
    }, indent=2, sort_keys=True))
    if args.require_pass and result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
