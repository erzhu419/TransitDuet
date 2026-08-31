#!/usr/bin/env python3
"""Audit how a trained lower policy allocates holding across causal states."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONTEXT = {
    "load",
    "capacity",
    "queue",
    "regularity_hold_target_norm",
    "regularity_hold_target_valid",
}
HF_FEATURE_NAMES = (
    "local_hf_residual_norm",
    "delta_local_hf_residual_norm",
    "local_hf_energy_norm",
    "global_hf_energy_norm",
)
HF_LAYOUT_START = {
    "high": 0,
    "hf": 0,
    "split": 0,
    "high_prior": 0,
    "hf_prior": 0,
    "high_context": 0,
    "hf_context": 0,
    "all": 3,
    "allfreq": 3,
    "all_freq": 3,
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


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_config(path: Path, seen: set[Path] | None = None) -> dict:
    path = path.resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"cyclic config inheritance at {path}")
    seen.add(path)
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    parent_value = payload.pop("_extends", None)
    if parent_value is None:
        return payload
    parent = Path(str(parent_value))
    if not parent.is_absolute():
        candidates = [path.parent / parent, path.parent.parent / parent, ROOT / parent]
        parent = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not parent.is_file():
        raise FileNotFoundError(f"parent config not found: {parent_value}")
    return _deep_merge(_load_config(parent, seen), payload)


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
    result = {
        "count": count,
        "action_mean_s": float(selected["action_s"].mean()),
        "action_median_s": float(np.median(selected["action_s"])),
        "target_mean_s": float(selected["target_s"].mean()),
        "action_minus_target_mean_s": float(
            selected["action_minus_target_s"].mean()),
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
    if "local_hf_residual_norm" in selected:
        result.update({
            "local_hf_residual_norm_mean": float(
                selected["local_hf_residual_norm"].mean()),
            "positive_local_hf_residual_norm_mean": float(
                selected["positive_local_hf_residual_norm"].mean()),
            "delta_local_hf_residual_norm_mean": float(
                selected["delta_local_hf_residual_norm"].mean()),
            "local_hf_energy_norm_mean": float(
                selected["local_hf_energy_norm"].mean()),
            "global_hf_energy_norm_mean": float(
                selected["global_hf_energy_norm"].mean()),
            "positive_local_hf_fraction": float(
                (selected["local_hf_residual_norm"] > 0.0).mean()),
            "hf_active_fraction": float(selected["hf_active"].mean()),
        })
    return result


def _positive_quantile_bands(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    zero_tolerance: float = 0.0,
) -> tuple[dict[str, object], list[tuple[str, np.ndarray]]]:
    positive = values[valid & (values > zero_tolerance)]
    if positive.size:
        q33, q67 = np.quantile(positive, [1.0 / 3.0, 2.0 / 3.0])
    else:
        q33 = q67 = float(zero_tolerance)
    bands = [
        ("nonpositive_or_zero", values <= zero_tolerance),
        (
            "positive_low_0_033",
            (values > zero_tolerance) & (values <= q33),
        ),
        (
            "positive_mid_033_067",
            (values > q33) & (values <= q67),
        ),
        ("positive_high_067_100", values > q67),
    ]
    metadata = {
        "zero_tolerance": float(zero_tolerance),
        "positive_count": int(positive.size),
        "positive_q33": float(q33),
        "positive_q67": float(q67),
    }
    return metadata, bands


def _masked_pearson(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    return _pearson(left[mask], right[mask])


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

    frequency_cfg = config.get("frequency", {}) or {}
    lower_mode = str(frequency_cfg.get("lower_mode", "high")).strip().lower()
    if not bool(frequency_cfg.get("enable", False)) or not bool(
        frequency_cfg.get("lower_features", False)
    ):
        raise ValueError("config has no enabled lower frequency features")
    if lower_mode not in HF_LAYOUT_START:
        raise ValueError(
            "lower frequency mode has no auditable HF feature layout: "
            f"{lower_mode}"
        )
    frequency_start = base_state_dim + len(features)
    hf_start = frequency_start + HF_LAYOUT_START[lower_mode]
    hf_feature_indexes = {
        name: hf_start + offset
        for offset, name in enumerate(HF_FEATURE_NAMES)
    }
    if max(hf_feature_indexes.values()) >= states_array.shape[1]:
        raise ValueError("checkpoint state is shorter than its HF feature layout")

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
    local_hf_residual = states_array[
        :, hf_feature_indexes["local_hf_residual_norm"]
    ]
    local_hf_energy = np.maximum(
        states_array[:, hf_feature_indexes["local_hf_energy_norm"]], 0.0
    )
    values = {
        "action_s": actions_array,
        "target_s": target_s,
        "action_minus_target_s": actions_array - target_s,
        "load": states_array[:, feature_indexes["load"]],
        "capacity": states_array[:, feature_indexes["capacity"]],
        "queue": states_array[:, feature_indexes["queue"]],
        "absolute_cost": absolute_cost,
        "zero_hold_cost": zero_hold_cost,
        "signed_regularity_gain": signed_gain,
        "positive_regularity_gain": np.maximum(signed_gain, 0.0),
        "zero_hold_regret": np.maximum(-signed_gain, 0.0),
        "local_hf_residual_norm": local_hf_residual,
        "positive_local_hf_residual_norm": np.maximum(
            local_hf_residual, 0.0
        ),
        "delta_local_hf_residual_norm": states_array[
            :, hf_feature_indexes["delta_local_hf_residual_norm"]
        ],
        "local_hf_energy_norm": local_hf_energy,
        "global_hf_energy_norm": np.maximum(
            states_array[:, hf_feature_indexes["global_hf_energy_norm"]], 0.0
        ),
        "hf_active": (
            (local_hf_residual > 0.0) & (local_hf_energy > 1e-12)
        ).astype(np.float64),
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
    residual_band_metadata, residual_bands = _positive_quantile_bands(
        values["local_hf_residual_norm"], valid
    )
    energy_band_metadata, energy_bands = _positive_quantile_bands(
        values["local_hf_energy_norm"], valid, zero_tolerance=1e-12
    )
    by_hf_residual = {
        name: _summary(values, valid & mask)
        for name, mask in residual_bands
    }
    by_hf_energy = {
        name: _summary(values, valid & mask)
        for name, mask in energy_bands
    }
    by_hf_activity = {
        "inactive": _summary(values, valid & (values["hf_active"] < 0.5)),
        "active": _summary(values, valid & (values["hf_active"] >= 0.5)),
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

    target_and_hf_energy = []
    for target_name, target_mask in target_bands:
        for energy_name, energy_mask in energy_bands:
            target_and_hf_energy.append(
                {
                    "target_band": target_name,
                    "hf_energy_band": energy_name,
                    **_summary(values, valid & target_mask & energy_mask),
                }
            )

    valid_values = {name: value[valid] for name, value in values.items()}
    positive_target = valid_values["target_s"] > 1e-9
    target_capture = np.zeros_like(valid_values["target_s"])
    target_capture[positive_target] = (
        valid_values["action_s"][positive_target]
        / valid_values["target_s"][positive_target]
    )
    return {
        "schema": "freqduet-lower-replay-allocation-audit-v2",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "checkpoint_episode": int(state.get("episode", -1)),
        "replay_transitions": int(len(replay)),
        "valid_transitions": int(valid.sum()),
        "base_state_dim": base_state_dim,
        "context_features": features,
        "frequency_mode": lower_mode,
        "frequency_start_index": frequency_start,
        "hf_feature_indexes": hf_feature_indexes,
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
            "action_vs_local_hf_residual": _pearson(
                valid_values["action_s"],
                valid_values["local_hf_residual_norm"],
            ),
            "action_vs_local_hf_energy": _pearson(
                valid_values["action_s"], valid_values["local_hf_energy_norm"]
            ),
            "action_minus_target_vs_local_hf_residual": _pearson(
                valid_values["action_minus_target_s"],
                valid_values["local_hf_residual_norm"],
            ),
            "action_minus_target_vs_local_hf_energy": _pearson(
                valid_values["action_minus_target_s"],
                valid_values["local_hf_energy_norm"],
            ),
            "target_capture_vs_local_hf_residual": _masked_pearson(
                target_capture,
                valid_values["local_hf_residual_norm"],
                positive_target,
            ),
            "target_capture_vs_local_hf_energy": _masked_pearson(
                target_capture,
                valid_values["local_hf_energy_norm"],
                positive_target,
            ),
            "regularity_gain_vs_local_hf_residual": _pearson(
                valid_values["signed_regularity_gain"],
                valid_values["local_hf_residual_norm"],
            ),
            "regularity_gain_vs_local_hf_energy": _pearson(
                valid_values["signed_regularity_gain"],
                valid_values["local_hf_energy_norm"],
            ),
        },
        "valid_by_load": by_load,
        "valid_by_target": by_target,
        "hf_residual_band_boundaries": residual_band_metadata,
        "hf_energy_band_boundaries": energy_band_metadata,
        "valid_by_hf_residual": by_hf_residual,
        "valid_by_hf_energy": by_hf_energy,
        "valid_by_hf_activity": by_hf_activity,
        "valid_by_target_and_load": joint,
        "valid_by_target_and_hf_energy": target_and_hf_energy,
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
