#!/usr/bin/env python3
"""Audit the exact V21 passenger--regularity feasible action frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lower.resac_lagrangian import CategoricalPolicy
from scripts.audit_lower_replay_allocation import (
    REQUIRED_CONTEXT,
    _finite_vector,
    _load_config,
    resolve_config_path,
)


def _minimum_primary_at_constraint(
    primary_costs: np.ndarray,
    constraint_costs: np.ndarray,
    feasible: np.ndarray,
    constraint_limit: float,
) -> dict[str, object]:
    """Solve a one-constraint finite-action policy LP through its dual."""
    primary = np.asarray(primary_costs, dtype=np.float64)
    constraint = np.asarray(constraint_costs, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    if primary.shape != constraint.shape or primary.shape != feasible.shape:
        raise ValueError("frontier arrays must have the same shape")
    if primary.ndim != 2 or not primary.shape[0] or not primary.shape[1]:
        raise ValueError("frontier costs must be a nonempty state-action matrix")
    if not np.isfinite(primary).all() or not np.isfinite(constraint).all():
        raise ValueError("frontier costs must be finite")
    if not feasible.any(axis=1).all():
        raise ValueError("every frontier state needs a feasible action")
    if not np.isfinite(constraint_limit) or constraint_limit < 0.0:
        raise ValueError("constraint limit must be finite and nonnegative")

    def select(multiplier: float) -> tuple[float, float, np.ndarray]:
        objective = primary + float(multiplier) * constraint
        objective = np.where(feasible, objective, np.inf)
        indexes = objective.argmin(axis=1)
        rows = np.arange(primary.shape[0])
        return (
            float(primary[rows, indexes].mean()),
            float(constraint[rows, indexes].mean()),
            indexes,
        )

    primary_zero, constraint_zero, indexes_zero = select(0.0)
    minimum_constraint = float(
        np.where(feasible, constraint, np.inf).min(axis=1).mean())
    result: dict[str, object] = {
        "constraint_limit": float(constraint_limit),
        "unconstrained_primary_mean": primary_zero,
        "unconstrained_constraint_mean": constraint_zero,
        "minimum_constraint_mean": minimum_constraint,
    }
    tolerance = 1e-10
    if minimum_constraint > constraint_limit + tolerance:
        result.update({
            "feasible": False,
            "minimum_primary_mean": None,
            "achieved_constraint_mean": None,
            "dual_multiplier_low": None,
            "dual_multiplier_high": None,
            "mixture_weight_high_constraint_policy": None,
        })
        return result
    if constraint_zero <= constraint_limit + tolerance:
        result.update({
            "feasible": True,
            "minimum_primary_mean": primary_zero,
            "achieved_constraint_mean": constraint_zero,
            "dual_multiplier_low": 0.0,
            "dual_multiplier_high": 0.0,
            "mixture_weight_high_constraint_policy": 1.0,
        })
        return result

    low = 0.0
    low_point = (primary_zero, constraint_zero, indexes_zero)
    high = 1.0
    high_point = select(high)
    while high_point[1] > constraint_limit + tolerance and high < 1e12:
        low, low_point = high, high_point
        high *= 2.0
        high_point = select(high)
    if high_point[1] > constraint_limit + tolerance:
        raise RuntimeError("dual search did not reach the feasible frontier")

    for _ in range(80):
        midpoint = 0.5 * (low + high)
        point = select(midpoint)
        if point[1] > constraint_limit:
            low, low_point = midpoint, point
        else:
            high, high_point = midpoint, point

    primary_low, constraint_low, _ = low_point
    primary_high, constraint_high, _ = high_point
    gap = constraint_low - constraint_high
    if gap <= tolerance:
        weight_low = 0.0
    else:
        weight_low = np.clip(
            (constraint_limit - constraint_high) / gap, 0.0, 1.0)
    minimum_primary = (
        weight_low * primary_low + (1.0 - weight_low) * primary_high)
    achieved_constraint = (
        weight_low * constraint_low
        + (1.0 - weight_low) * constraint_high)
    result.update({
        "feasible": True,
        "minimum_primary_mean": float(minimum_primary),
        "achieved_constraint_mean": float(achieved_constraint),
        "dual_multiplier_low": float(low),
        "dual_multiplier_high": float(high),
        "mixture_weight_high_constraint_policy": float(weight_low),
        "high_constraint_endpoint": {
            "primary_mean": float(primary_low),
            "constraint_mean": float(constraint_low),
        },
        "low_constraint_endpoint": {
            "primary_mean": float(primary_high),
            "constraint_mean": float(constraint_high),
        },
    })
    return result


def _gain_floor_cost_arrays(
    zero_hold_cost: np.ndarray,
    absolute_action_costs: np.ndarray,
    required_fraction: np.ndarray,
) -> dict[str, np.ndarray]:
    zero_cost = np.asarray(zero_hold_cost, dtype=np.float64).reshape(-1, 1)
    absolute = np.asarray(absolute_action_costs, dtype=np.float64)
    required = np.asarray(required_fraction, dtype=np.float64).reshape(-1, 1)
    if absolute.shape[0] != zero_cost.shape[0] or required.shape != zero_cost.shape:
        raise ValueError("gain-floor state dimensions do not match")
    positive_gain = np.maximum(zero_cost - absolute, 0.0)
    maximum_gain = positive_gain.max(axis=1)
    eligible = maximum_gain > 1e-12
    gain_fraction = np.ones_like(positive_gain)
    gain_fraction[eligible] = (
        positive_gain[eligible] / maximum_gain[eligible, None])
    relative_shortfall = np.maximum(required - gain_fraction, 0.0)
    relative_shortfall[~eligible] = 0.0
    required_gain = required.reshape(-1) * maximum_gain
    absolute_shortfall = np.maximum(
        required_gain[:, None] - positive_gain, 0.0)
    absolute_shortfall[~eligible] = 0.0
    return {
        "positive_gain": positive_gain,
        "maximum_gain": maximum_gain,
        "eligible": eligible,
        "gain_fraction": gain_fraction,
        "relative_shortfall": relative_shortfall,
        "required_gain": required_gain,
        "absolute_shortfall": absolute_shortfall,
    }


def _policy_probabilities(
    states: np.ndarray,
    lower_state: dict,
    action_bins: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    policy_state = lower_state["policy"]
    hidden_dim = int(policy_state["fc1.weight"].shape[0])
    policy = CategoricalPolicy(
        states.shape[1],
        action_bins,
        hidden_dim=hidden_dim,
        action_limit_feature_index=lower_state.get(
            "action_limit_feature_index"),
    )
    policy.load_state_dict(policy_state)
    policy.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(states), int(batch_size)):
            batch = torch.from_numpy(states[start:start + batch_size]).float()
            probs, _, _ = policy.dist_info(batch)
            parts.append(probs.cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)


def _gain_band_summary(
    maximum_gain: np.ndarray,
    eligible: np.ndarray,
    expected_relative_shortfall: np.ndarray,
    expected_absolute_shortfall: np.ndarray,
    required_gain: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    gains = maximum_gain[eligible]
    if not gains.size:
        raise ValueError("gain-floor audit has no eligible replay state")
    quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(gains, q))
        for q in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    }
    eligible_indexes = np.flatnonzero(eligible)
    ordered = eligible_indexes[np.argsort(maximum_gain[eligible_indexes])]
    total_gain = float(maximum_gain[eligible].sum())
    total_required = float(required_gain[eligible].sum())
    total_relative = float(expected_relative_shortfall[eligible].sum())
    total_absolute = float(expected_absolute_shortfall[eligible].sum())
    bands = []
    for index, members in enumerate(np.array_split(ordered, 4), start=1):
        bands.append({
            "quartile": index,
            "count": int(members.size),
            "maximum_gain_mean": float(maximum_gain[members].mean()),
            "maximum_gain_mass_share": float(
                maximum_gain[members].sum() / max(total_gain, 1e-15)),
            "required_gain_mass_share": float(
                required_gain[members].sum() / max(total_required, 1e-15)),
            "learned_relative_shortfall_mass_share": float(
                expected_relative_shortfall[members].sum()
                / max(total_relative, 1e-15)),
            "learned_absolute_shortfall_mass_share": float(
                expected_absolute_shortfall[members].sum()
                / max(total_absolute, 1e-15)),
        })
    return quantiles, bands


def audit_gain_floor_frontier(
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
    if contract.get("constraint_cost_mode") != "hf_relative_gain_shortfall_v3":
        raise ValueError("checkpoint does not use the V21 gain-floor cost")
    if floor_contract.get("enabled") is not True:
        raise ValueError("checkpoint has no enabled V21 gain floor")
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
    feature_indexes = {
        name: base_state_dim + features.index(name) for name in REQUIRED_CONTEXT
    }
    if int(contract["valid_feature_index"]) != feature_indexes[
            "regularity_hold_target_valid"]:
        raise ValueError("config context order does not match checkpoint contract")

    action_bins = (
        lower_state["policy"]["action_bins"].detach().cpu().numpy().reshape(-1)
        .astype(np.float64, copy=False))
    if action_bins.size != 7 or not np.array_equal(
            action_bins, np.asarray([0, 5, 10, 15, 20, 30, 45], dtype=float)):
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
    hf_pressure = scaled_hf / (1.0 + scaled_hf)
    required_fraction = (
        float(floor_contract["base_fraction"])
        + float(floor_contract["hf_increment"]) * hf_pressure)
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
    valid_relative = floor["relative_shortfall"][valid]
    valid_absolute = floor["absolute_shortfall"][valid]
    valid_passenger = passenger_costs[valid]
    valid_feasible = feasible[valid]
    required_gain_mean = float(floor["required_gain"][valid].mean())
    if required_gain_mean <= 0.0:
        raise ValueError("V21 replay has no positive required regularity gain")
    aggregate_weighted_shortfall = valid_absolute / required_gain_mean
    expected_relative_by_state = (
        valid_probabilities * valid_relative).sum(axis=1)
    expected_absolute_by_state = (
        valid_probabilities * valid_absolute).sum(axis=1)
    expected_passenger_by_state = (
        valid_probabilities * valid_passenger).sum(axis=1)

    relative_limit = float(contract["cost_limit"])
    passenger_limit = float(passenger_contract["cost_limit"])
    relative_frontier = {
        "minimum_passenger_at_floor_limit": _minimum_primary_at_constraint(
            valid_passenger, valid_relative, valid_feasible, relative_limit),
        "minimum_floor_at_passenger_limit": _minimum_primary_at_constraint(
            valid_relative, valid_passenger, valid_feasible, passenger_limit),
    }
    weighted_frontier = {
        "minimum_passenger_at_floor_limit": _minimum_primary_at_constraint(
            valid_passenger,
            aggregate_weighted_shortfall,
            valid_feasible,
            relative_limit,
        ),
        "minimum_floor_at_passenger_limit": _minimum_primary_at_constraint(
            aggregate_weighted_shortfall,
            valid_passenger,
            valid_feasible,
            passenger_limit,
        ),
    }
    for frontier in (relative_frontier, weighted_frontier):
        minimum = frontier["minimum_passenger_at_floor_limit"]
        frontier["joint_budget_feasible"] = bool(
            minimum["feasible"]
            and minimum["minimum_primary_mean"] is not None
            and float(minimum["minimum_primary_mean"])
            <= passenger_limit + 1e-10)

    quantiles, gain_bands = _gain_band_summary(
        floor["maximum_gain"][valid],
        floor["eligible"][valid],
        expected_relative_by_state,
        expected_absolute_by_state,
        floor["required_gain"][valid],
    )
    regularity_log_lambda = lower_state.get("log_regularity_lambda")
    passenger_log_lambda = lower_state.get("log_regularity_passenger_lambda")
    learned_action = (
        valid_probabilities * action_bins[None, :]).sum(axis=1)
    return {
        "schema": "freqduet-v21-replay-gain-floor-frontier-v1",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "checkpoint_episode": int(state.get("episode", -1)),
        "replay_transitions": int(len(states)),
        "valid_transitions": int(valid.sum()),
        "action_bins_s": action_bins.tolist(),
        "relative_floor_limit": relative_limit,
        "passenger_cost_limit": passenger_limit,
        "required_gain_mean": required_gain_mean,
        "eligible_fraction": float(floor["eligible"][valid].mean()),
        "maximum_gain_quantiles": quantiles,
        "learned_policy": {
            "expected_action_mean_s": float(learned_action.mean()),
            "expected_relative_floor_shortfall_mean": float(
                expected_relative_by_state.mean()),
            "expected_aggregate_weighted_shortfall_ratio": float(
                expected_absolute_by_state.mean() / required_gain_mean),
            "expected_passenger_cost_mean": float(
                expected_passenger_by_state.mean()),
            "regularity_lambda": float(
                regularity_log_lambda.exp().item()),
            "passenger_lambda": float(passenger_log_lambda.exp().item()),
        },
        "relative_floor_frontier": relative_frontier,
        "aggregate_gain_weighted_frontier": weighted_frontier,
        "maximum_gain_quartiles": gain_bands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_gain_floor_frontier(args.checkpoint, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "Audit complete: FREQDUET_V21_FRONTIER_COMPLETE "
        f"valid={result['valid_transitions']} out={args.out}"
    )


if __name__ == "__main__":
    main()
