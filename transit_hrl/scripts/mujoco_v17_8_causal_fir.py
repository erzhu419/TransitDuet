"""Causal multivariate FIR fitting utilities for the v17.8 diagnostic."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from freq_hrl.experiments.mujoco.full_horizon_responsibility_oracle import (
    responsibility_frequency_powers,
)


def causal_fir_features(total_action: Any, *, window: int) -> np.ndarray:
    """Return current-and-past action features with zero prehistory."""

    total = _action_trace(total_action, role="total action")
    width = int(window)
    if width < 1:
        raise ValueError("causal FIR window must be positive")
    length, dimension = total.shape
    features = np.zeros((length, width * dimension), dtype=np.float64)
    for lag in range(min(width, length)):
        features[lag:, lag * dimension:(lag + 1) * dimension] = total[
            :length - lag
        ]
    return features


def fit_causal_fir(
    total_actions: Iterable[Any],
    target_upper_actions: Iterable[Any],
    *,
    window: int,
    ridge_penalty: float,
    feature_scale_floor: float,
) -> dict[str, Any]:
    """Fit a zero-state multivariate FIR with normalized ridge regression."""

    totals = [
        _action_trace(value, role="fit total action")
        for value in total_actions
    ]
    targets = [
        _action_trace(value, role="fit target upper action")
        for value in target_upper_actions
    ]
    if not totals or len(totals) != len(targets):
        raise ValueError("FIR fitting requires aligned non-empty path lists")
    dimension = int(totals[0].shape[1])
    for total, target in zip(totals, targets, strict=True):
        if total.shape != target.shape or total.shape[1] != dimension:
            raise ValueError("FIR fitting paths must share an action dimension")
    penalty = float(ridge_penalty)
    scale_floor = float(feature_scale_floor)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("ridge penalty must be positive and finite")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("feature scale floor must be positive and finite")

    features = np.concatenate(
        [causal_fir_features(total, window=window) for total in totals],
        axis=0,
    )
    outputs = np.concatenate(targets, axis=0)
    feature_scale = np.maximum(
        np.sqrt(np.mean(np.square(features), axis=0)), scale_floor
    )
    normalized = features / feature_scale
    sample_count = float(normalized.shape[0])
    gram = normalized.T @ normalized / sample_count
    cross = normalized.T @ outputs / sample_count
    normalized_coefficients = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0], dtype=np.float64),
        cross,
    )
    coefficients = normalized_coefficients / feature_scale[:, None]
    width = int(window)
    return {
        "window": width,
        "ridge_penalty": penalty,
        "action_dimension": dimension,
        "fit_path_count": len(totals),
        "fit_sample_count": int(features.shape[0]),
        "coefficients": coefficients.reshape(width, dimension, dimension),
    }


def apply_causal_fir(
    total_action: Any,
    model: dict[str, Any],
    *,
    output_gain: float,
    upper_action_limit: float,
    lower_action_limit: float,
) -> dict[str, np.ndarray]:
    """Apply a fitted FIR and project each split onto physical boxes."""

    total = _action_trace(total_action, role="evaluation total action")
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    window = int(model["window"])
    dimension = int(model["action_dimension"])
    gain = float(output_gain)
    upper_limit = float(upper_action_limit)
    lower_limit = float(lower_action_limit)
    if coefficients.shape != (window, dimension, dimension):
        raise ValueError("FIR coefficient tensor has an invalid shape")
    if total.shape[1] != dimension:
        raise ValueError("FIR model and total action dimensions do not align")
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("FIR output gain must be positive and finite")
    if (
        not np.isfinite(upper_limit)
        or upper_limit <= 0.0
        or not np.isfinite(lower_limit)
        or lower_limit <= 0.0
    ):
        raise ValueError("component action limits must be positive and finite")
    if np.max(np.abs(total)) > upper_limit + lower_limit + 1e-10:
        raise ValueError("total action exceeds the component reconstruction box")

    flattened = coefficients.reshape(window * dimension, dimension)
    raw_upper = gain * causal_fir_features(total, window=window) @ flattened
    physical_low = np.maximum(-upper_limit, total - lower_limit)
    physical_high = np.minimum(upper_limit, total + lower_limit)
    if np.any(physical_low > physical_high + 1e-12):
        raise RuntimeError("total action has no bounded responsibility split")
    upper = np.clip(raw_upper, physical_low, physical_high)
    lower = total - upper
    return {
        "raw_upper": raw_upper,
        "upper": upper,
        "lower": lower,
        "total": total,
    }


def evaluate_causal_fir_split(
    total_action: Any,
    model: dict[str, Any],
    *,
    output_gain: float,
    upper_action_limit: float,
    lower_action_limit: float,
    upper_window: int,
    lower_window: int,
    upper_power_budget: float,
    lower_power_budget: float,
    power_tolerance: float,
) -> dict[str, Any]:
    split = apply_causal_fir(
        total_action,
        model,
        output_gain=output_gain,
        upper_action_limit=upper_action_limit,
        lower_action_limit=lower_action_limit,
    )
    upper_power, lower_power = responsibility_frequency_powers(
        split["total"],
        split["upper"],
        upper_window=int(upper_window),
        lower_window=int(lower_window),
    )
    reconstruction_error = float(np.max(np.abs(
        split["upper"] + split["lower"] - split["total"]
    )))
    bound_violation = float(max(
        np.max(np.maximum(np.abs(split["upper"]) - upper_action_limit, 0.0)),
        np.max(np.maximum(np.abs(split["lower"]) - lower_action_limit, 0.0)),
    ))
    finite = bool(
        np.all(np.isfinite(split["upper"]))
        and np.all(np.isfinite(split["lower"]))
        and np.isfinite(upper_power)
        and np.isfinite(lower_power)
    )
    tolerance = float(power_tolerance)
    upper_pass = bool(
        finite and upper_power <= float(upper_power_budget) + tolerance
    )
    lower_pass = bool(
        finite and lower_power <= float(lower_power_budget) + tolerance
    )
    return {
        "finite": finite,
        "upper_power": float(upper_power),
        "lower_power": float(lower_power),
        "upper_budget_pass": upper_pass,
        "lower_budget_pass": lower_pass,
        "joint_budget_pass": bool(upper_pass and lower_pass),
        "reconstruction_error_max": reconstruction_error,
        "bound_violation_max": bound_violation,
    }


def apply_causal_fir_with_prefix_high_frequency_budget(
    total_action: Any,
    model: dict[str, Any],
    *,
    output_gain: float,
    upper_action_limit: float,
    lower_action_limit: float,
    upper_window: int,
    upper_rms_budget: float,
    power_tolerance: float,
) -> dict[str, Any]:
    """Project only causal upper innovations onto the prefix HPF budget."""

    base = apply_causal_fir(
        total_action,
        model,
        output_gain=output_gain,
        upper_action_limit=upper_action_limit,
        lower_action_limit=lower_action_limit,
    )
    total = base["total"]
    desired = base["raw_upper"]
    width = int(upper_window)
    budget = float(upper_rms_budget)
    tolerance = float(power_tolerance)
    if width < 2:
        raise ValueError("prefix upper window must be at least two")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("prefix upper RMS budget must be positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("prefix power tolerance must be positive and finite")

    length, dimension = total.shape
    upper = np.empty_like(total)
    residuals = np.empty_like(total)
    accumulated_energy = 0.0
    feasible_steps = 0
    projection_energy = 0.0
    prefix_powers = []
    for index in range(length):
        physical_low = np.maximum(
            -float(upper_action_limit),
            total[index] - float(lower_action_limit),
        )
        physical_high = np.minimum(
            float(upper_action_limit),
            total[index] + float(lower_action_limit),
        )
        count = min(index + 1, width)
        history = upper[max(0, index - width + 1):index]
        if count == 1:
            selected_upper = np.clip(
                desired[index], physical_low, physical_high
            )
            residual = np.zeros(dimension, dtype=np.float64)
            feasible = True
        else:
            coefficient = 1.0 - 1.0 / float(count)
            offset = -np.sum(history, axis=0) / float(count)
            residual_low = coefficient * physical_low + offset
            residual_high = coefficient * physical_high + offset
            low = np.minimum(residual_low, residual_high)
            high = np.maximum(residual_low, residual_high)
            target = np.clip(
                coefficient * desired[index] + offset, low, high
            )
            allowed_energy = max(
                (index + 1) * dimension * budget ** 2
                - accumulated_energy,
                0.0,
            )
            minimum = np.clip(np.zeros_like(target), low, high)
            minimum_energy = float(np.sum(np.square(minimum)))
            feasible = bool(
                minimum_energy <= allowed_energy + tolerance
            )
            if not feasible:
                selected_residual = minimum
            elif float(np.sum(np.square(target))) <= allowed_energy:
                selected_residual = target
            else:
                multiplier_low = 0.0
                multiplier_high = 1.0
                selected_residual = np.clip(
                    target / (1.0 + multiplier_high), low, high
                )
                while (
                    float(np.sum(np.square(selected_residual)))
                    > allowed_energy
                ):
                    multiplier_high *= 10.0
                    selected_residual = np.clip(
                        target / (1.0 + multiplier_high), low, high
                    )
                for _ in range(48):
                    midpoint = 0.5 * (multiplier_low + multiplier_high)
                    trial = np.clip(target / (1.0 + midpoint), low, high)
                    if float(np.sum(np.square(trial))) <= allowed_energy:
                        multiplier_high = midpoint
                        selected_residual = trial
                    else:
                        multiplier_low = midpoint
            selected_upper = np.clip(
                (selected_residual - offset) / coefficient,
                physical_low,
                physical_high,
            )
            residual = coefficient * selected_upper + offset
        upper[index] = selected_upper
        residuals[index] = residual
        accumulated_energy += float(np.sum(np.square(residual)))
        feasible_steps += int(feasible)
        projection_energy += float(np.sum(np.square(
            selected_upper - np.clip(
                desired[index], physical_low, physical_high
            )
        )))
        prefix_powers.append(
            accumulated_energy / float((index + 1) * dimension)
        )
    lower = total - upper
    return {
        "raw_upper": desired,
        "upper": upper,
        "lower": lower,
        "total": total,
        "upper_high_frequency_residual": residuals,
        "prefix_upper_budget_feasible_rate": float(
            feasible_steps / length
        ),
        "prefix_upper_power_max": float(max(prefix_powers)),
        "projection_rms": float(np.sqrt(
            projection_energy / float(length * dimension)
        )),
    }


def evaluate_causal_fir_prefix_split(
    total_action: Any,
    model: dict[str, Any],
    *,
    output_gain: float,
    upper_action_limit: float,
    lower_action_limit: float,
    upper_window: int,
    lower_window: int,
    upper_rms_budget: float,
    lower_rms_budget: float,
    power_tolerance: float,
) -> dict[str, Any]:
    split = apply_causal_fir_with_prefix_high_frequency_budget(
        total_action,
        model,
        output_gain=output_gain,
        upper_action_limit=upper_action_limit,
        lower_action_limit=lower_action_limit,
        upper_window=upper_window,
        upper_rms_budget=upper_rms_budget,
        power_tolerance=power_tolerance,
    )
    upper_power, lower_power = responsibility_frequency_powers(
        split["total"],
        split["upper"],
        upper_window=int(upper_window),
        lower_window=int(lower_window),
    )
    reconstruction_error = float(np.max(np.abs(
        split["upper"] + split["lower"] - split["total"]
    )))
    bound_violation = float(max(
        np.max(np.maximum(np.abs(split["upper"]) - upper_action_limit, 0.0)),
        np.max(np.maximum(np.abs(split["lower"]) - lower_action_limit, 0.0)),
    ))
    finite = bool(
        np.all(np.isfinite(split["upper"]))
        and np.all(np.isfinite(split["lower"]))
        and np.isfinite(upper_power)
        and np.isfinite(lower_power)
    )
    upper_budget = float(upper_rms_budget) ** 2
    lower_budget = float(lower_rms_budget) ** 2
    tolerance = float(power_tolerance)
    upper_pass = bool(finite and upper_power <= upper_budget + tolerance)
    lower_pass = bool(finite and lower_power <= lower_budget + tolerance)
    return {
        "finite": finite,
        "upper_power": float(upper_power),
        "lower_power": float(lower_power),
        "upper_budget_pass": upper_pass,
        "lower_budget_pass": lower_pass,
        "joint_budget_pass": bool(upper_pass and lower_pass),
        "reconstruction_error_max": reconstruction_error,
        "bound_violation_max": bound_violation,
        "prefix_upper_budget_feasible_rate": float(
            split["prefix_upper_budget_feasible_rate"]
        ),
        "prefix_upper_power_max": float(split["prefix_upper_power_max"]),
        "projection_rms": float(split["projection_rms"]),
    }


def candidate_id(window: int, ridge_penalty: float, output_gain: float) -> str:
    return (
        f"fir_w{int(window)}_ridge{float(ridge_penalty):.0e}_"
        f"gain{float(output_gain):.2f}"
    )


def _action_trace(value: Any, *, role: str) -> np.ndarray:
    trace = np.asarray(value, dtype=np.float64)
    if trace.ndim != 2 or not trace.shape[0] or not trace.shape[1]:
        raise ValueError(f"{role} must be a non-empty matrix")
    if not np.all(np.isfinite(trace)):
        raise ValueError(f"{role} must be finite")
    return trace
