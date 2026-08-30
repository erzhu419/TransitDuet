"""Causal residual adapter for hierarchical total-action proposals."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def causal_actor_features(
    total_action: Any,
    proposed_upper_action: Any,
    *,
    window: int,
) -> np.ndarray:
    """Return lagged upper/lower proposal features with zero prehistory."""

    total = _action_trace(total_action, role="total action")
    upper = _action_trace(proposed_upper_action, role="proposed upper action")
    if upper.shape != total.shape:
        raise ValueError("total and proposed upper actions must align")
    width = int(window)
    if width < 1:
        raise ValueError("actor adapter window must be positive")
    lower = total - upper
    proposal = np.concatenate((upper, lower), axis=1)
    length, proposal_dimension = proposal.shape
    features = np.zeros(
        (length, width * proposal_dimension), dtype=np.float64
    )
    for lag in range(min(width, length)):
        start = lag * proposal_dimension
        features[lag:, start:start + proposal_dimension] = proposal[
            :length - lag
        ]
    return features


def fit_causal_actor_adapter(
    total_actions: Iterable[Any],
    proposed_upper_actions: Iterable[Any],
    target_total_corrections: Iterable[Any],
    path_weights: Iterable[float],
    *,
    window: int,
    ridge_penalty: float,
    feature_scale_floor: float,
) -> dict[str, Any]:
    """Fit a path-balanced weighted ridge adapter.

    Every path contributes its declared weight regardless of trajectory length.
    This prevents long zero-target paths from dominating short actor-floor paths.
    """

    totals = [_action_trace(value, role="fit total") for value in total_actions]
    uppers = [
        _action_trace(value, role="fit proposed upper")
        for value in proposed_upper_actions
    ]
    targets = [
        _action_trace(value, role="fit target correction")
        for value in target_total_corrections
    ]
    weights = np.asarray(list(path_weights), dtype=np.float64)
    if (
        not totals
        or len(totals) != len(uppers)
        or len(totals) != len(targets)
        or weights.shape != (len(totals),)
    ):
        raise ValueError("actor adapter fitting inputs must be aligned")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("actor adapter path weights must be positive and finite")
    dimension = int(totals[0].shape[1])
    feature_blocks = []
    sample_weight_blocks = []
    for total, upper, target, path_weight in zip(
        totals, uppers, targets, weights, strict=True
    ):
        if (
            upper.shape != total.shape
            or target.shape != total.shape
            or total.shape[1] != dimension
        ):
            raise ValueError("actor adapter paths must share an action dimension")
        feature_blocks.append(
            causal_actor_features(total, upper, window=int(window))
        )
        sample_weight_blocks.append(np.full(
            total.shape[0],
            float(path_weight) / float(total.shape[0]),
            dtype=np.float64,
        ))
    penalty = float(ridge_penalty)
    scale_floor = float(feature_scale_floor)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("actor adapter ridge penalty must be positive and finite")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("actor adapter scale floor must be positive and finite")

    features = np.concatenate(feature_blocks, axis=0)
    outputs = np.concatenate(targets, axis=0)
    sample_weights = np.concatenate(sample_weight_blocks)
    weight_sum = float(np.sum(sample_weights))
    feature_scale = np.maximum(
        np.sqrt(np.sum(
            sample_weights[:, None] * np.square(features), axis=0
        ) / weight_sum),
        scale_floor,
    )
    normalized = features / feature_scale
    weighted_features = normalized * sample_weights[:, None]
    gram = normalized.T @ weighted_features / weight_sum
    cross = weighted_features.T @ outputs / weight_sum
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
        "proposal_dimension": 2 * dimension,
        "fit_path_count": len(totals),
        "fit_actor_floor_weight": float(np.sum(weights)),
        "coefficients": coefficients.reshape(
            width, 2 * dimension, dimension
        ),
    }


def apply_causal_actor_adapter(
    total_action: Any,
    proposed_upper_action: Any,
    model: dict[str, Any],
    *,
    output_gain: float,
    correction_abs_limit: float,
    component_sum_limit: float = 2.0,
    executed_action_limit: float = 1.0,
) -> dict[str, np.ndarray]:
    """Apply the adapter before nominal and environment action clipping."""

    total = _action_trace(total_action, role="evaluation total")
    upper = _action_trace(
        proposed_upper_action, role="evaluation proposed upper"
    )
    if upper.shape != total.shape:
        raise ValueError("evaluation total and proposed upper actions must align")
    width = int(model["window"])
    dimension = int(model["action_dimension"])
    proposal_dimension = int(model["proposal_dimension"])
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if total.shape[1] != dimension:
        raise ValueError("actor adapter model and actions do not align")
    if coefficients.shape != (width, proposal_dimension, dimension):
        raise ValueError("actor adapter coefficient tensor has an invalid shape")
    if proposal_dimension != 2 * dimension:
        raise ValueError("actor adapter proposal dimension is invalid")
    gain = _positive_finite(output_gain, "output gain")
    correction_limit = _positive_finite(
        correction_abs_limit, "correction absolute limit"
    )
    sum_limit = _positive_finite(component_sum_limit, "component sum limit")
    executed_limit = _positive_finite(
        executed_action_limit, "executed action limit"
    )
    if np.max(np.abs(total)) > sum_limit + 1e-10:
        raise ValueError("input total action exceeds the component sum limit")

    features = causal_actor_features(total, upper, window=width)
    raw = gain * features @ coefficients.reshape(
        width * proposal_dimension, dimension
    )
    trusted = np.clip(raw, -correction_limit, correction_limit)
    corrected = np.clip(total + trusted, -sum_limit, sum_limit)
    correction = corrected - total
    executed_reference = np.clip(total, -executed_limit, executed_limit)
    executed_corrected = np.clip(
        corrected, -executed_limit, executed_limit
    )
    return {
        "raw_correction": raw,
        "correction": correction,
        "corrected_total": corrected,
        "executed_reference": executed_reference,
        "executed_corrected": executed_corrected,
        "executed_correction": executed_corrected - executed_reference,
    }


def _action_trace(value: Any, *, role: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{role} must be a finite non-empty 2D array")
    return array


def _positive_finite(value: float, role: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{role} must be positive and finite")
    return number
