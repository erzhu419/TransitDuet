#!/usr/bin/env python3
"""Audit how a trained lower policy allocates holding across causal states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONTEXT = {
    "load",
    "capacity",
    "queue",
    "regularity_hold_target_norm",
    "regularity_hold_target_valid",
}


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    name = str(value)
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    candidate = ROOT / "configs_freqduet" / filename
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"config not found: {value}")


def _load_config(path: Path) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runner_v3 import load_config

    return load_config(str(path))


def _finite_vector(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _summary(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, object]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0}
    selected = {name: value[mask] for name, value in values.items()}
    target_positive = selected["target_s"] > 1e-9
    capture = (
        selected["action_s"][target_positive]
        / selected["target_s"][target_positive]
    )
    return {
        "count": count,
        "action_mean_s": float(selected["action_s"].mean()),
        "action_median_s": float(np.median(selected["action_s"])),
        "target_mean_s": float(selected["target_s"].mean()),
        "target_capture_ratio_mean": (
            float(capture.mean()) if capture.size else None),
        "load_mean": float(selected["load"].mean()),
        "capacity_mean": float(selected["capacity"].mean()),
        "queue_mean": float(selected["queue"].mean()),
        "absolute_cost_mean": float(selected["absolute_cost"].mean()),
        "zero_hold_cost_mean": float(selected["zero_hold_cost"].mean()),
        "signed_regularity_gain_mean": float(
            selected["signed_regularity_gain"].mean()),
        "positive_regularity_gain_mean": float(
            selected["positive_regularity_gain"].mean()),
        "zero_hold_regret_mean": float(selected["zero_hold_regret"].mean()),
        "zero_hold_regret_fraction": float(
            (selected["zero_hold_regret"] > 1e-12).mean()),
    }


def audit_lower_replay_allocation(
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
    if contract.get("enabled") is not True:
        raise ValueError("checkpoint has no enabled causal regularity contract")
    replay = state.get("lower_replay_buffer", {}).get("buffer", [])
    if not replay:
        raise ValueError("checkpoint lower replay buffer is empty")

    config = _load_config(config_path)
    context_cfg = config.get("frequency", {}).get("lower_context", {}) or {}
    features = [str(name) for name in context_cfg.get("features", [])]
    missing = sorted(REQUIRED_CONTEXT.difference(features))
    if not bool(context_cfg.get("enable", False)) or missing:
        raise ValueError(
            "config lacks required lower causal context: " + ", ".join(missing)
        )

    states = []
    actions = []
    for index, transition in enumerate(replay):
        if len(transition) != 7:
            raise ValueError(f"replay transition {index} is not a 7-tuple")
        states.append(_finite_vector(transition[0], name=f"state[{index}]"))
        action = _finite_vector(transition[1], name=f"action[{index}]")
        if action.size != 1:
            raise ValueError(f"replay action {index} is not scalar")
        actions.append(float(action[0]))
    state_dims = {row.size for row in states}
    if len(state_dims) != 1:
        raise ValueError("replay states do not share one dimensionality")
    states_array = np.stack(states)
    actions_array = np.asarray(actions, dtype=np.float64)

    target_offset = features.index("regularity_hold_target_norm")
    base_state_dim = int(contract["target_feature_index"]) - target_offset
    if base_state_dim < 8:
        raise ValueError("regularity contract implies an invalid base state size")
    feature_indexes = {
        name: base_state_dim + features.index(name) for name in REQUIRED_CONTEXT
    }
    expected_valid = feature_indexes["regularity_hold_target_valid"]
    if int(contract["valid_feature_index"]) != expected_valid:
        raise ValueError("config context order does not match checkpoint contract")
    if max(feature_indexes.values()) >= states_array.shape[1]:
        raise ValueError("checkpoint state is shorter than its causal context")

    headway_index = int(contract["target_headway_feature_index"])
    if not 0 <= headway_index < states_array.shape[1]:
        raise ValueError("checkpoint headway feature index is out of range")
    action_scale = float(contract["action_target_scale_s"])
    headway_scale = float(contract["target_headway_scale_s"])
    cost_cap = float(contract["cost_cap"])
    if min(action_scale, headway_scale, cost_cap) <= 0.0:
        raise ValueError("regularity contract scales must be positive")

    target_s = np.clip(
        states_array[:, feature_indexes["regularity_hold_target_norm"]], 0.0, 1.0
    ) * action_scale
    target_headway_s = np.maximum(
        states_array[:, headway_index] * headway_scale, 1.0
    )
    absolute_cost = np.minimum(
        ((actions_array - target_s) / target_headway_s) ** 2, cost_cap
    )
    zero_hold_cost = np.minimum(
        (target_s / target_headway_s) ** 2, cost_cap
    )
    signed_gain = zero_hold_cost - absolute_cost
    values = {
        "action_s": actions_array,
        "target_s": target_s,
        "load": states_array[:, feature_indexes["load"]],
        "capacity": states_array[:, feature_indexes["capacity"]],
        "queue": states_array[:, feature_indexes["queue"]],
        "absolute_cost": absolute_cost,
        "zero_hold_cost": zero_hold_cost,
        "signed_regularity_gain": signed_gain,
        "positive_regularity_gain": np.maximum(signed_gain, 0.0),
        "zero_hold_regret": np.maximum(-signed_gain, 0.0),
    }
    valid = (
        states_array[:, feature_indexes["regularity_hold_target_valid"]] >= 0.5
    )

    load_bands = [
        ("low_0_033", values["load"] < 1.0 / 3.0),
        (
            "mid_033_067",
            (values["load"] >= 1.0 / 3.0) & (values["load"] < 2.0 / 3.0),
        ),
        ("high_067_plus", values["load"] >= 2.0 / 3.0),
    ]
    target_bands = [
        ("zero", target_s <= 1e-9),
        ("low_0_15", (target_s > 1e-9) & (target_s <= 15.0)),
        ("mid_15_30", (target_s > 15.0) & (target_s <= 30.0)),
        ("high_30_plus", target_s > 30.0),
    ]
    by_load = {
        name: _summary(values, valid & mask) for name, mask in load_bands
    }
    by_target = {
        name: _summary(values, valid & mask) for name, mask in target_bands
    }
    joint = []
    for target_name, target_mask in target_bands:
        for load_name, load_mask in load_bands:
            joint.append(
                {
                    "target_band": target_name,
                    "load_band": load_name,
                    **_summary(values, valid & target_mask & load_mask),
                }
            )

    valid_values = {name: value[valid] for name, value in values.items()}
    return {
        "schema": "freqduet-lower-replay-allocation-audit-v1",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "checkpoint_episode": int(state.get("episode", -1)),
        "replay_transitions": int(len(replay)),
        "valid_transitions": int(valid.sum()),
        "base_state_dim": base_state_dim,
        "context_features": features,
        "regularity_contract": contract,
        "overall": _summary(values, np.ones(len(replay), dtype=bool)),
        "valid_overall": _summary(values, valid),
        "valid_correlations": {
            "action_vs_target": _pearson(
                valid_values["action_s"], valid_values["target_s"]
            ),
            "action_vs_load": _pearson(
                valid_values["action_s"], valid_values["load"]
            ),
            "action_vs_capacity": _pearson(
                valid_values["action_s"], valid_values["capacity"]
            ),
            "regularity_gain_vs_load": _pearson(
                valid_values["signed_regularity_gain"], valid_values["load"]
            ),
        },
        "valid_by_load": by_load,
        "valid_by_target": by_target,
        "valid_by_target_and_load": joint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_lower_replay_allocation(args.checkpoint, args.config)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
