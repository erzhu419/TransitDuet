#!/usr/bin/env python3
"""Audit exact joint KL projection on a V21/V22 replay action table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_lower_replay_allocation import (
    REQUIRED_CONTEXT,
    _finite_vector,
    _load_config,
    resolve_config_path,
)
from scripts.audit_lower_replay_gain_floor_frontier import (
    _gain_floor_cost_arrays,
    _minimum_primary_at_constraint,
    _policy_probabilities,
)


def _joint_kl_projection(
    base_probabilities: np.ndarray,
    regularity_costs: np.ndarray,
    passenger_costs: np.ndarray,
    feasible: np.ndarray,
    *,
    regularity_limit: float,
    passenger_limit: float,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> dict[str, object]:
    """Project categorical rows onto two global expected-cost constraints."""
    base = np.asarray(base_probabilities, dtype=np.float64)
    regularity = np.asarray(regularity_costs, dtype=np.float64)
    passenger = np.asarray(passenger_costs, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    if not (base.shape == regularity.shape == passenger.shape == feasible.shape):
        raise ValueError("projection arrays must have the same shape")
    if base.ndim != 2 or not base.shape[0] or not base.shape[1]:
        raise ValueError("projection arrays must be nonempty state-action tables")
    if not (np.isfinite(base).all() and np.isfinite(regularity).all()
            and np.isfinite(passenger).all()):
        raise ValueError("projection inputs must be finite")
    if (base < 0.0).any() or not feasible.any(axis=1).all():
        raise ValueError("projection needs nonnegative probabilities and feasible rows")
    limits = np.asarray(
        [regularity_limit, passenger_limit], dtype=np.float64)
    if not np.isfinite(limits).all() or (limits < 0.0).any():
        raise ValueError("projection limits must be finite and nonnegative")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("projection tolerance and iterations must be positive")

    weights = np.full(base.shape[0], 1.0 / base.shape[0])

    support_floor = 1e-12
    base = np.where(feasible, np.maximum(base, support_floor), 0.0)
    base /= base.sum(axis=1, keepdims=True)
    log_base = np.full_like(base, -np.inf)
    log_base[feasible] = np.log(base[feasible])
    costs = np.stack((regularity, passenger), axis=0)

    def distribution(multiplier: np.ndarray) -> np.ndarray:
        logits = (
            log_base
            - multiplier[0] * regularity
            - multiplier[1] * passenger)
        logits = np.where(feasible, logits, -np.inf)
        maximum = np.max(logits, axis=1, keepdims=True)
        mass = np.where(feasible, np.exp(logits - maximum), 0.0)
        return mass / mass.sum(axis=1, keepdims=True)

    def expected_costs(probabilities: np.ndarray) -> np.ndarray:
        per_state = np.einsum("na,kna->kn", probabilities, costs)
        return per_state @ weights

    frontier = _minimum_primary_at_constraint(
        passenger, regularity, feasible, float(regularity_limit))
    joint_feasible = bool(
        frontier["feasible"]
        and frontier["minimum_primary_mean"] is not None
        and float(frontier["minimum_primary_mean"])
        <= passenger_limit + tolerance)
    if not joint_feasible:
        return {
            "joint_feasible": False,
            "frontier": frontier,
            "converged": False,
            "iterations": 0,
            "multipliers": None,
            "expected_costs": None,
            "probabilities": None,
        }

    multipliers = np.zeros(2, dtype=np.float64)
    converged = False
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        iterations = iteration
        for constraint in range(2):
            trial = multipliers.copy()
            trial[constraint] = 0.0
            if expected_costs(distribution(trial))[constraint] <= (
                    limits[constraint] + tolerance):
                multipliers[constraint] = 0.0
                continue
            low = 0.0
            high = max(1.0, float(multipliers[constraint]))
            trial[constraint] = high
            cost = expected_costs(distribution(trial))[constraint]
            while cost > limits[constraint] + tolerance and high < 1e8:
                low = high
                high *= 2.0
                trial[constraint] = high
                cost = expected_costs(distribution(trial))[constraint]
            if cost > limits[constraint] + tolerance:
                multipliers[constraint] = high
                continue
            for _ in range(70):
                midpoint = 0.5 * (low + high)
                trial[constraint] = midpoint
                if expected_costs(distribution(trial))[constraint] > (
                        limits[constraint]):
                    low = midpoint
                else:
                    high = midpoint
            multipliers[constraint] = high

        probabilities = distribution(multipliers)
        residual = expected_costs(probabilities) - limits
        kkt = max(
            abs(float(residual[index]))
            if multipliers[index] > tolerance
            else max(float(residual[index]), 0.0)
            for index in range(2)
        )
        if kkt <= tolerance:
            converged = True
            break

    probabilities = distribution(multipliers)
    projected_costs = expected_costs(probabilities)
    return {
        "joint_feasible": True,
        "frontier": frontier,
        "converged": bool(converged),
        "iterations": int(iterations),
        "multipliers": multipliers.tolist(),
        "expected_costs": projected_costs.tolist(),
        "constraint_residuals": (projected_costs - limits).tolist(),
        "probabilities": probabilities,
    }


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return -np.sum(
        np.where(probabilities > 0.0,
                 probabilities * np.log(np.maximum(probabilities, 1e-300)),
                 0.0),
        axis=1,
    )


def audit_replay_joint_projection(
    checkpoint_path: str | Path,
    config_path: str | Path,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = resolve_config_path(config_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("format") != "freqduet-exact-training-state-v4":
        raise ValueError("not a FreqDuet exact v4 training checkpoint")
    lower_state = state.get("lower_trainer", {})
    contract = lower_state.get("regularity_policy_contract", {})
    floor_contract = contract.get("regularity_gain_floor", {}) or {}
    passenger_contract = contract.get("passenger_holding_constraint", {}) or {}
    cost_mode = contract.get("constraint_cost_mode")
    if cost_mode not in {
            "hf_relative_gain_shortfall_v3",
            "hf_aggregate_gain_shortfall_v4"}:
        raise ValueError("checkpoint does not use a registered gain-floor cost")
    if floor_contract.get("enabled") is not True:
        raise ValueError("checkpoint has no enabled gain floor")
    if passenger_contract.get("enabled") is not True:
        raise ValueError("checkpoint has no enabled passenger constraint")

    replay = state.get("lower_replay_buffer", {}).get("buffer", [])
    if not replay:
        raise ValueError("checkpoint lower replay buffer is empty")
    replay_states = []
    for index, transition in enumerate(replay):
        if len(transition) != 7:
            raise ValueError(f"replay transition {index} is not a 7-tuple")
        replay_states.append(
            _finite_vector(transition[0], name=f"state[{index}]")
            .astype(np.float32, copy=False))
    dimensions = {row.size for row in replay_states}
    if len(dimensions) != 1:
        raise ValueError("replay states do not share one dimensionality")
    states = np.stack(replay_states)

    config = _load_config(config_path)
    context_cfg = config.get("frequency", {}).get("lower_context", {}) or {}
    features = [str(name) for name in context_cfg.get("features", [])]
    missing = sorted(REQUIRED_CONTEXT.difference(features))
    if not bool(context_cfg.get("enable", False)) or missing:
        raise ValueError(
            "config lacks required lower causal context: " + ", ".join(missing))
    target_offset = features.index("regularity_hold_target_norm")
    base_state_dim = int(contract["target_feature_index"]) - target_offset
    expected_valid_index = (
        base_state_dim + features.index("regularity_hold_target_valid"))
    if int(contract["valid_feature_index"]) != expected_valid_index:
        raise ValueError("config context order does not match checkpoint contract")

    action_bins = (
        lower_state["policy"]["action_bins"].detach().cpu().numpy().reshape(-1)
        .astype(np.float64, copy=False))
    registered_bins = np.asarray([0, 5, 10, 15, 20, 30, 45], dtype=float)
    if not np.array_equal(action_bins, registered_bins):
        raise ValueError("checkpoint does not use the registered seven actions")
    probabilities = _policy_probabilities(states, lower_state, action_bins)

    valid = states[:, int(contract["valid_feature_index"])] >= 0.5
    if not valid.any():
        raise ValueError("checkpoint replay has no causal-valid state")
    target_s = np.clip(
        states[:, int(contract["target_feature_index"])], 0.0, 1.0
    ) * float(contract["action_target_scale_s"])
    headway_s = np.maximum(
        states[:, int(contract["target_headway_feature_index"])]
        * float(contract["target_headway_scale_s"]),
        1.0,
    )
    absolute_costs = np.minimum(
        ((action_bins[None, :] - target_s[:, None]) / headway_s[:, None]) ** 2,
        float(contract["cost_cap"]),
    )
    zero_hold_cost = np.minimum(
        (target_s / headway_s) ** 2, float(contract["cost_cap"]))
    hf_energy = np.maximum(
        states[:, int(floor_contract["hf_energy_feature_index"])], 0.0)
    scaled_hf = (
        hf_energy / float(floor_contract["hf_energy_scale"])
    ) ** float(floor_contract["hf_energy_exponent"])
    required_fraction = (
        float(floor_contract["base_fraction"])
        + float(floor_contract["hf_increment"])
        * scaled_hf / (1.0 + scaled_hf))
    floor = _gain_floor_cost_arrays(
        zero_hold_cost, absolute_costs, required_fraction)

    load = np.clip(
        states[:, int(passenger_contract["load_feature_index"])],
        0.0,
        float(passenger_contract["load_clip"]),
    )
    passenger_costs = (
        load[:, None] * action_bins[None, :]
        / float(passenger_contract["action_norm_s"]))
    feasible = np.ones_like(passenger_costs, dtype=bool)
    limit_index = lower_state.get("action_limit_feature_index")
    if limit_index is not None:
        action_limit_s = np.clip(states[:, int(limit_index)], 0.0, 1.0)
        action_limit_s *= float(action_bins.max())
        feasible = action_bins[None, :] <= action_limit_s[:, None] + 1e-6
        feasible[:, int(np.argmin(np.abs(action_bins)))] = True

    valid_probabilities = probabilities[valid]
    valid_passenger = passenger_costs[valid]
    valid_feasible = feasible[valid]
    required_gain_mean = float(floor["required_gain"][valid].mean())
    if required_gain_mean <= 0.0:
        raise ValueError("replay has no positive required regularity gain")
    if cost_mode == "hf_aggregate_gain_shortfall_v4":
        valid_regularity = (
            floor["absolute_shortfall"][valid] / required_gain_mean)
    else:
        valid_regularity = floor["relative_shortfall"][valid]

    regularity_limit = float(contract["cost_limit"])
    passenger_limit = float(passenger_contract["cost_limit"])
    projection = _joint_kl_projection(
        valid_probabilities,
        valid_regularity,
        valid_passenger,
        valid_feasible,
        regularity_limit=regularity_limit,
        passenger_limit=passenger_limit,
    )
    if not projection["joint_feasible"] or not projection["converged"]:
        projected_summary = None
    else:
        projected = projection.pop("probabilities")
        learned_action = (
            valid_probabilities * action_bins[None, :]).sum(axis=1)
        projected_action = (projected * action_bins[None, :]).sum(axis=1)
        learned_argmax = valid_probabilities.argmax(axis=1)
        projected_argmax = projected.argmax(axis=1)
        kl_by_state = np.sum(
            projected * (
                np.log(np.maximum(projected, 1e-300))
                - np.log(np.maximum(valid_probabilities, 1e-300))),
            axis=1,
        )
        projected_summary = {
            "expected_action_mean_s": float(projected_action.mean()),
            "expected_regularity_cost_mean": float(
                projection["expected_costs"][0]),
            "expected_passenger_cost_mean": float(
                projection["expected_costs"][1]),
            "entropy_mean": float(_entropy(projected).mean()),
            "kl_from_learned_mean": float(kl_by_state.mean()),
            "probability_l1_shift_mean": float(
                np.abs(projected - valid_probabilities).sum(axis=1).mean()),
            "argmax_changed_fraction": float(
                np.mean(projected_argmax != learned_argmax)),
            "expected_action_change_mean_s": float(
                (projected_action - learned_action).mean()),
        }

    learned_regularity = (
        valid_probabilities * valid_regularity).sum(axis=1)
    learned_passenger = (
        valid_probabilities * valid_passenger).sum(axis=1)
    learned_action = (
        valid_probabilities * action_bins[None, :]).sum(axis=1)
    deterministic_joint = np.any(
        valid_feasible
        & (valid_regularity <= regularity_limit + 1e-12)
        & (valid_passenger <= passenger_limit + 1e-12),
        axis=1,
    )
    return {
        "schema": "freqduet-replay-joint-kl-projection-v1",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "checkpoint_episode": int(state.get("episode", -1)),
        "constraint_cost_mode": str(cost_mode),
        "replay_transitions": int(len(states)),
        "valid_transitions": int(valid.sum()),
        "action_bins_s": action_bins.tolist(),
        "regularity_cost_limit": regularity_limit,
        "passenger_cost_limit": passenger_limit,
        "required_gain_mean": required_gain_mean,
        "per_state_deterministic_joint_feasible_fraction": float(
            deterministic_joint.mean()),
        "learned_policy": {
            "expected_action_mean_s": float(learned_action.mean()),
            "expected_regularity_cost_mean": float(
                learned_regularity.mean()),
            "expected_passenger_cost_mean": float(learned_passenger.mean()),
            "entropy_mean": float(_entropy(valid_probabilities).mean()),
        },
        "projection": {
            key: value for key, value in projection.items()
            if key != "probabilities"
        },
        "projected_policy": projected_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_replay_joint_projection(args.checkpoint, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    projection = result["projection"]
    print(
        "Audit complete: FREQDUET_JOINT_PROJECTION_COMPLETE "
        f"valid={result['valid_transitions']} "
        f"feasible={projection['joint_feasible']} "
        f"converged={projection['converged']} out={args.out}"
    )


if __name__ == "__main__":
    main()
