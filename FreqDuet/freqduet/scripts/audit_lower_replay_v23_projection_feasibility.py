#!/usr/bin/env python3
"""Audit V23's locked joint projection on full replay and train-size batches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.audit_lower_replay_joint_projection import (
    _entropy,
    _joint_kl_projection,
    _load_replay_projection_table,
)


REGULARITY_REPLAY_TARGET = 0.036
PASSENGER_REPLAY_TARGET = 0.075
MINIBATCH_SIZE = 512
MINIBATCH_COUNT = 256
MINIBATCH_SAMPLE_SEED = 230908
TOLERANCE = 1e-8


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return None
    if not np.isfinite(array).all():
        raise ValueError("feasibility audit contains non-finite values")
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _run_identity(checkpoint: str | Path) -> tuple[str, int]:
    checkpoint = Path(checkpoint)
    match = re.fullmatch(r"(.+)_seed([0-9]+)", checkpoint.parent.parent.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint run identity: {checkpoint}")
    return match.group(1), int(match.group(2))


def _project_indices(
    table: dict[str, object],
    indices: np.ndarray,
    *,
    regularity_target: float,
    passenger_target: float,
) -> dict[str, object]:
    indices = np.asarray(indices, dtype=np.int64)
    valid_mask = table["valid"][indices]
    valid_indices = indices[valid_mask]
    if not valid_indices.size:
        return {
            "passes": False,
            "failure": "no_causal_valid_transition",
            "sampled_transitions": int(indices.size),
            "valid_transitions": 0,
            "valid_fraction": 0.0,
        }

    required_gain_mean = float(
        table["required_gain"][valid_indices].mean())
    if required_gain_mean <= 0.0:
        return {
            "passes": False,
            "failure": "zero_required_gain",
            "sampled_transitions": int(indices.size),
            "valid_transitions": int(valid_indices.size),
            "valid_fraction": float(valid_indices.size / indices.size),
        }
    regularity = (
        table["absolute_shortfall"][valid_indices] / required_gain_mean)
    passenger = table["passenger_costs"][valid_indices]
    probabilities = table["probabilities"][valid_indices]
    feasible = table["feasible"][valid_indices]
    projection = _joint_kl_projection(
        probabilities,
        regularity,
        passenger,
        feasible,
        regularity_limit=regularity_target,
        passenger_limit=passenger_target,
        tolerance=TOLERANCE,
    )
    summary: dict[str, object] = {
        "sampled_transitions": int(indices.size),
        "valid_transitions": int(valid_indices.size),
        "valid_fraction": float(valid_indices.size / indices.size),
        "required_gain_mean": required_gain_mean,
        "joint_feasible": bool(projection["joint_feasible"]),
        "converged": bool(projection["converged"]),
        "iterations": int(projection["iterations"]),
        "frontier": projection["frontier"],
    }
    if not projection["joint_feasible"]:
        summary.update({"passes": False, "failure": "no_joint_frontier"})
        return summary
    if not projection["converged"]:
        summary.update({"passes": False, "failure": "projection_not_converged"})
        return summary

    projected = projection["probabilities"]
    projected_costs = np.asarray(projection["expected_costs"], dtype=float)
    budget_satisfied = bool(
        projected_costs[0] <= regularity_target + TOLERANCE
        and projected_costs[1] <= passenger_target + TOLERANCE)
    learned_action = (
        probabilities * table["action_bins_s"][None, :]).sum(axis=1)
    projected_action = (
        projected * table["action_bins_s"][None, :]).sum(axis=1)
    kl_by_state = np.sum(
        projected * (
            np.log(np.maximum(projected, 1e-300))
            - np.log(np.maximum(probabilities, 1e-300))),
        axis=1,
    )
    learned_regularity = float(
        (probabilities * regularity).sum(axis=1).mean())
    learned_passenger = float(
        (probabilities * passenger).sum(axis=1).mean())
    summary.update({
        "passes": budget_satisfied,
        "failure": None if budget_satisfied else "projected_budget_violation",
        "budget_satisfied": budget_satisfied,
        "multipliers": projection["multipliers"],
        "learned_regularity_cost": learned_regularity,
        "learned_passenger_cost": learned_passenger,
        "projected_regularity_cost": float(projected_costs[0]),
        "projected_passenger_cost": float(projected_costs[1]),
        "learned_action_mean_s": float(learned_action.mean()),
        "projected_action_mean_s": float(projected_action.mean()),
        "projection_action_change_mean_s": float(
            (projected_action - learned_action).mean()),
        "learned_entropy": float(_entropy(probabilities).mean()),
        "projected_entropy": float(_entropy(projected).mean()),
        "projection_kl": float(kl_by_state.mean()),
        "projection_probability_l1_shift": float(
            np.abs(projected - probabilities).sum(axis=1).mean()),
        "projection_argmax_changed_fraction": float(np.mean(
            projected.argmax(axis=1) != probabilities.argmax(axis=1))),
    })
    return summary


def _audit_minibatches(
    table: dict[str, object],
    *,
    regularity_target: float,
    passenger_target: float,
    batch_size: int,
    batch_count: int,
    sample_seed: int,
) -> dict[str, object]:
    replay_size = int(len(table["valid"]))
    if batch_size < 1 or batch_size > replay_size:
        raise ValueError("minibatch size must lie within the replay inventory")
    if batch_count < 1:
        raise ValueError("minibatch count must be positive")
    rng = np.random.default_rng(int(sample_seed))
    rows = []
    for batch_index in range(int(batch_count)):
        indices = rng.choice(replay_size, size=int(batch_size), replace=False)
        row = _project_indices(
            table,
            indices,
            regularity_target=regularity_target,
            passenger_target=passenger_target,
        )
        row["batch_index"] = batch_index
        rows.append(row)

    successful = [row for row in rows if row.get("passes") is True]
    metrics = {}
    for name in (
            "valid_transitions", "valid_fraction", "required_gain_mean",
            "iterations", "learned_regularity_cost",
            "learned_passenger_cost", "projected_regularity_cost",
            "projected_passenger_cost", "learned_action_mean_s",
            "projected_action_mean_s", "projection_action_change_mean_s",
            "learned_entropy", "projected_entropy", "projection_kl",
            "projection_probability_l1_shift",
            "projection_argmax_changed_fraction"):
        values = [float(row[name]) for row in successful if name in row]
        metrics[name] = _distribution(values)
    regularity_multipliers = [
        float(row["multipliers"][0]) for row in successful]
    passenger_multipliers = [
        float(row["multipliers"][1]) for row in successful]
    metrics["regularity_projection_multiplier"] = _distribution(
        regularity_multipliers)
    metrics["passenger_projection_multiplier"] = _distribution(
        passenger_multipliers)
    return {
        "batch_size": int(batch_size),
        "batch_count": int(batch_count),
        "sample_seed": int(sample_seed),
        "joint_feasible_count": int(sum(
            row.get("joint_feasible") is True for row in rows)),
        "converged_count": int(sum(
            row.get("converged") is True for row in rows)),
        "budget_satisfied_count": int(sum(
            row.get("budget_satisfied") is True for row in rows)),
        "pass_count": len(successful),
        "passes": len(successful) == len(rows),
        "failed_batches": [
            {
                "batch_index": int(row["batch_index"]),
                "failure": str(row.get("failure")),
                "valid_transitions": int(row["valid_transitions"]),
            }
            for row in rows if row.get("passes") is not True
        ],
        "metrics": metrics,
    }


def audit_v23_projection_feasibility(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    regularity_target: float = REGULARITY_REPLAY_TARGET,
    passenger_target: float = PASSENGER_REPLAY_TARGET,
    batch_size: int = MINIBATCH_SIZE,
    batch_count: int = MINIBATCH_COUNT,
    sample_seed: int = MINIBATCH_SAMPLE_SEED,
) -> dict[str, object]:
    table = _load_replay_projection_table(checkpoint_path, config_path)
    if table["constraint_cost_mode"] != "hf_aggregate_gain_shortfall_v4":
        raise ValueError("V23 feasibility audit requires aggregate gain cost")
    if not np.isclose(float(table["configured_regularity_limit"]), 0.05):
        raise ValueError("V23 source changed the registered regularity budget")
    if not np.isclose(float(table["configured_passenger_limit"]), 0.08):
        raise ValueError("V23 source changed the registered passenger budget")
    if not (0.0 < regularity_target < 0.05):
        raise ValueError("V23 regularity replay target must lie in (0, 0.05)")
    if not (0.0 < passenger_target < 0.08):
        raise ValueError("V23 passenger replay target must lie in (0, 0.08)")
    config_name, train_seed = _run_identity(table["checkpoint"])
    if Path(str(table["config"])).stem != config_name:
        raise ValueError("checkpoint and config identities disagree")

    full = _project_indices(
        table,
        np.arange(len(table["valid"]), dtype=np.int64),
        regularity_target=regularity_target,
        passenger_target=passenger_target,
    )
    minibatches = _audit_minibatches(
        table,
        regularity_target=regularity_target,
        passenger_target=passenger_target,
        batch_size=batch_size,
        batch_count=batch_count,
        sample_seed=sample_seed,
    )
    return {
        "schema": "freqduet-v23-projection-feasibility-v1",
        "checkpoint": table["checkpoint"],
        "config": config_name,
        "train_seed": train_seed,
        "checkpoint_episode": int(table["checkpoint_episode"]),
        "constraint_cost_mode": table["constraint_cost_mode"],
        "action_bins_s": table["action_bins_s"].tolist(),
        "configured_regularity_budget": float(
            table["configured_regularity_limit"]),
        "configured_passenger_budget": float(
            table["configured_passenger_limit"]),
        "locked_regularity_replay_target": float(regularity_target),
        "locked_passenger_replay_target": float(passenger_target),
        "full_replay": full,
        "minibatches": minibatches,
        "passes": bool(full.get("passes") and minibatches["passes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regularity-target", type=float,
                        default=REGULARITY_REPLAY_TARGET)
    parser.add_argument("--passenger-target", type=float,
                        default=PASSENGER_REPLAY_TARGET)
    parser.add_argument("--batch-size", type=int, default=MINIBATCH_SIZE)
    parser.add_argument("--batch-count", type=int, default=MINIBATCH_COUNT)
    parser.add_argument("--sample-seed", type=int,
                        default=MINIBATCH_SAMPLE_SEED)
    args = parser.parse_args()
    result = audit_v23_projection_feasibility(
        args.checkpoint,
        args.config,
        regularity_target=args.regularity_target,
        passenger_target=args.passenger_target,
        batch_size=args.batch_size,
        batch_count=args.batch_count,
        sample_seed=args.sample_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "Audit complete: FREQDUET_V23_PROJECTION_FEASIBILITY_COMPLETE "
        f"pass={result['passes']} out={args.out}")


if __name__ == "__main__":
    main()
