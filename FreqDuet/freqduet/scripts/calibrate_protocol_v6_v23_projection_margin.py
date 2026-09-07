#!/usr/bin/env python3
"""Lock V23 replay targets from the observed V22 replay-to-frozen shift."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_aggregate_gain_screen import (
    EVAL_SEEDS,
    FACTORIAL_SPECS,
    TRAIN_SEEDS,
)


REGULARITY_BUDGET = 0.05
PASSENGER_BUDGET = 0.08
MARGIN_QUANTUM = 0.001
SELECTED_ALLOCATION = "aggregate"


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("margin calibration contains non-finite values")
    return {
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _ceiling_margin(maximum_shift: float, quantum: float) -> float:
    if not np.isfinite(maximum_shift):
        raise ValueError("maximum shift must be finite")
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError("margin quantum must be finite and positive")
    positive = max(float(maximum_shift), 0.0)
    units = math.ceil(positive / float(quantum) - 1e-12)
    return round(units * float(quantum), 12)


def calibrate_projection_margin(
    projection_summary: dict[str, object],
    frozen_per_eval: pd.DataFrame,
    *,
    specs: Sequence[tuple[str, str, str, float, bool]] = FACTORIAL_SPECS,
    train_seeds: Sequence[int] = TRAIN_SEEDS,
    eval_seeds: Sequence[int] = EVAL_SEEDS,
    selected_allocation: str = SELECTED_ALLOCATION,
    margin_quantum: float = MARGIN_QUANTUM,
) -> dict[str, object]:
    if projection_summary.get("schema") != (
            "freqduet-replay-joint-kl-projection-aggregate-v1"):
        raise ValueError("wrong joint-projection aggregate schema")
    for key in (
            "all_joint_frontiers_feasible",
            "all_projections_converged",
            "batch_projection_supported"):
        if projection_summary.get(key) is not True:
            raise ValueError(f"joint-projection prerequisite failed: {key}")

    specs = list(specs)
    config_meta = {
        config: {
            "allocation": allocation,
            "dual_update_mode": update,
            "augmented_lagrangian_rho": float(rho),
        }
        for config, allocation, update, rho, _ in specs
    }
    if selected_allocation not in {
            meta["allocation"] for meta in config_meta.values()}:
        raise ValueError("selected allocation is absent from the factorial")

    expected_pairs = {
        (config, int(seed))
        for config in config_meta
        for seed in train_seeds
    }
    projection_rows = projection_summary.get("rows")
    if not isinstance(projection_rows, list):
        raise ValueError("joint-projection aggregate has no row inventory")
    replay_by_pair: dict[tuple[str, int], dict[str, object]] = {}
    for row in projection_rows:
        if not isinstance(row, dict):
            raise ValueError("joint-projection row is not an object")
        pair = (str(row.get("config")), int(row.get("train_seed", -1)))
        if pair not in expected_pairs:
            raise ValueError(f"unexpected replay pair: {pair}")
        if pair in replay_by_pair:
            raise ValueError(f"duplicate replay pair: {pair}")
        replay_by_pair[pair] = row
    missing_replay = sorted(expected_pairs - set(replay_by_pair))
    if missing_replay:
        raise ValueError(f"missing replay pairs: {missing_replay}")

    required_columns = {
        "config", "train_seed", "eval_seed", "ep", "checkpoint_ep",
        "lower_policy_frozen", "lower_critic_frozen",
        "lower_regularity_gain_floor_expected_shortfall_mean",
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
        "lower_regularity_passenger_expected_cost_mean",
    }
    missing_columns = sorted(required_columns - set(frozen_per_eval.columns))
    if missing_columns:
        raise ValueError(f"frozen evaluation is missing: {missing_columns}")

    frozen = frozen_per_eval.loc[
        frozen_per_eval["config"].astype(str).isin(config_meta)
    ].copy()
    frozen["config"] = frozen["config"].astype(str)
    frozen["train_seed"] = pd.to_numeric(
        frozen["train_seed"], errors="raise").astype(int)
    frozen["eval_seed"] = pd.to_numeric(
        frozen["eval_seed"], errors="raise").astype(int)
    expected_triples = {
        (config, int(train_seed), int(eval_seed))
        for config in config_meta
        for train_seed in train_seeds
        for eval_seed in eval_seeds
    }
    actual_triples = list(zip(
        frozen["config"], frozen["train_seed"], frozen["eval_seed"]))
    if len(actual_triples) != len(set(actual_triples)):
        raise ValueError("duplicate frozen config/train/eval row")
    actual_set = set(actual_triples)
    if actual_set != expected_triples:
        raise ValueError(
            "frozen inventory mismatch: "
            f"missing={sorted(expected_triples - actual_set)} "
            f"unexpected={sorted(actual_set - expected_triples)}")
    for column in ("ep", "checkpoint_ep"):
        values = pd.to_numeric(frozen[column], errors="raise")
        if not (values == 39).all():
            raise ValueError(f"frozen evaluation changed {column}")
    for column in ("lower_policy_frozen", "lower_critic_frozen"):
        values = pd.to_numeric(frozen[column], errors="raise")
        if not (values == 1.0).all():
            raise ValueError(f"frozen evaluation changed {column}")

    rows: list[dict[str, object]] = []
    for frozen_row in frozen.to_dict(orient="records"):
        config = str(frozen_row["config"])
        train_seed = int(frozen_row["train_seed"])
        replay = replay_by_pair[(config, train_seed)]
        allocation = str(config_meta[config]["allocation"])
        expected_mode = (
            "hf_aggregate_gain_shortfall_v4"
            if allocation == "aggregate"
            else "hf_relative_gain_shortfall_v3")
        if str(replay.get("constraint_cost_mode")) != expected_mode:
            raise ValueError(
                f"constraint mode disagrees for {config} seed {train_seed}")
        frozen_regularity_column = (
            "lower_regularity_gain_floor_"
            "expected_aggregate_shortfall_ratio"
            if allocation == "aggregate"
            else "lower_regularity_gain_floor_expected_shortfall_mean")
        replay_regularity = float(replay["learned_regularity_cost"])
        replay_passenger = float(replay["learned_passenger_cost"])
        frozen_regularity = float(frozen_row[frozen_regularity_column])
        frozen_passenger = float(
            frozen_row["lower_regularity_passenger_expected_cost_mean"])
        values = np.asarray([
            replay_regularity, replay_passenger,
            frozen_regularity, frozen_passenger,
        ])
        if not np.isfinite(values).all():
            raise ValueError(
                f"non-finite transfer row for {config} seed {train_seed}")
        rows.append({
            "config": config,
            "allocation": allocation,
            "dual_update_mode": config_meta[config]["dual_update_mode"],
            "augmented_lagrangian_rho": config_meta[config][
                "augmented_lagrangian_rho"],
            "train_seed": train_seed,
            "eval_seed": int(frozen_row["eval_seed"]),
            "replay_regularity_cost": replay_regularity,
            "frozen_regularity_cost": frozen_regularity,
            "regularity_shift": frozen_regularity - replay_regularity,
            "replay_passenger_cost": replay_passenger,
            "frozen_passenger_cost": frozen_passenger,
            "passenger_shift": frozen_passenger - replay_passenger,
        })

    allocation_summaries = []
    for allocation in sorted({row["allocation"] for row in rows}):
        selected = [row for row in rows if row["allocation"] == allocation]
        regularity = _distribution([
            float(row["regularity_shift"]) for row in selected])
        passenger = _distribution([
            float(row["passenger_shift"]) for row in selected])
        regularity_margin = _ceiling_margin(
            regularity["maximum"], margin_quantum)
        passenger_margin = _ceiling_margin(
            passenger["maximum"], margin_quantum)
        regularity_target = round(
            REGULARITY_BUDGET - regularity_margin, 12)
        passenger_target = round(
            PASSENGER_BUDGET - passenger_margin, 12)
        if regularity_target <= 0.0 or passenger_target <= 0.0:
            raise ValueError("calibrated replay target is not positive")
        allocation_summaries.append({
            "allocation": allocation,
            "row_count": len(selected),
            "regularity_shift": regularity,
            "passenger_shift": passenger,
            "regularity_margin": regularity_margin,
            "passenger_margin": passenger_margin,
            "regularity_replay_target": regularity_target,
            "passenger_replay_target": passenger_target,
        })

    selected_summary = next(
        row for row in allocation_summaries
        if row["allocation"] == selected_allocation)
    return {
        "schema": "freqduet-v23-replay-to-frozen-margin-v1",
        "selected_allocation": selected_allocation,
        "margin_rule": "ceil_positive_max_shift_v1",
        "margin_quantum": float(margin_quantum),
        "regularity_budget": REGULARITY_BUDGET,
        "passenger_budget": PASSENGER_BUDGET,
        "locked_regularity_replay_target": selected_summary[
            "regularity_replay_target"],
        "locked_passenger_replay_target": selected_summary[
            "passenger_replay_target"],
        "factorial_checkpoint_count": len(expected_pairs),
        "frozen_rollout_count": len(rows),
        "allocation_summaries": allocation_summaries,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection_summary", type=Path)
    parser.add_argument("frozen_per_eval", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    projection = json.loads(args.projection_summary.read_text())
    frozen = pd.read_csv(args.frozen_per_eval)
    result = calibrate_projection_margin(projection, frozen)
    result["sources"] = {
        "projection_summary": str(args.projection_summary),
        "frozen_per_eval": str(args.frozen_per_eval),
    }
    csv_path = args.csv or args.out.with_suffix(".csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(result["rows"]).to_csv(csv_path, index=False)
    print(json.dumps({
        "selected_allocation": result["selected_allocation"],
        "regularity_replay_target": result[
            "locked_regularity_replay_target"],
        "passenger_replay_target": result[
            "locked_passenger_replay_target"],
        "json": str(args.out),
        "csv": str(csv_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
