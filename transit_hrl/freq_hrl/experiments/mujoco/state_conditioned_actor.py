"""Causal state-conditioned residual actor for hierarchical action proposals."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


def causal_state_actor_features(
    lower_policy_state: Any,
    total_action: Any,
    proposed_upper_action: Any,
    *,
    proposal_window: int,
) -> np.ndarray:
    """Combine the current policy state with current and past proposals."""

    state = _matrix(lower_policy_state, role="lower policy state")
    total = _matrix(total_action, role="total action")
    upper = _matrix(proposed_upper_action, role="proposed upper action")
    if state.shape[0] != total.shape[0] or upper.shape != total.shape:
        raise ValueError("state-conditioned actor traces must align")
    width = int(proposal_window)
    if width < 1:
        raise ValueError("state-conditioned actor proposal window must be positive")
    lower = total - upper
    proposal = np.concatenate((upper, lower), axis=1)
    length, proposal_dimension = proposal.shape
    history = np.zeros(
        (length, width * proposal_dimension), dtype=np.float64
    )
    for lag in range(min(width, length)):
        start = lag * proposal_dimension
        history[lag:, start:start + proposal_dimension] = proposal[
            :length - lag
        ]
    return np.concatenate((state, history), axis=1)


def fit_state_conditioned_actor(
    lower_policy_states: Iterable[Any],
    total_actions: Iterable[Any],
    proposed_upper_actions: Iterable[Any],
    target_total_corrections: Iterable[Any],
    path_weights: Iterable[float],
    *,
    proposal_window: int,
    hidden_dim: int,
    hidden_layers: int,
    correction_abs_limit: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    random_seed: int,
    feature_scale_floor: float = 1e-6,
) -> dict[str, Any]:
    """Fit a deterministic path-balanced bounded residual MLP on CPU."""

    states = [
        _matrix(value, role="fit lower policy state")
        for value in lower_policy_states
    ]
    totals = [_matrix(value, role="fit total action") for value in total_actions]
    uppers = [
        _matrix(value, role="fit proposed upper action")
        for value in proposed_upper_actions
    ]
    targets = [
        _matrix(value, role="fit target correction")
        for value in target_total_corrections
    ]
    weights = np.asarray(list(path_weights), dtype=np.float64)
    count = len(states)
    if (
        count < 1
        or len(totals) != count
        or len(uppers) != count
        or len(targets) != count
        or weights.shape != (count,)
    ):
        raise ValueError("state-conditioned actor fitting inputs must align")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("state-conditioned actor path weights must be positive")
    width = int(proposal_window)
    hidden = int(hidden_dim)
    depth = int(hidden_layers)
    epoch_count = int(epochs)
    seed = int(random_seed)
    limit = _positive_finite(correction_abs_limit, "correction limit")
    rate = _positive_finite(learning_rate, "learning rate")
    decay = float(weight_decay)
    scale_floor = _positive_finite(feature_scale_floor, "feature scale floor")
    if hidden < 1 or depth < 1 or epoch_count < 1:
        raise ValueError("state-conditioned actor architecture is invalid")
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("state-conditioned actor weight decay is invalid")
    action_dimension = int(totals[0].shape[1])
    state_dimension = int(states[0].shape[1])
    feature_blocks: list[np.ndarray] = []
    sample_weight_blocks: list[np.ndarray] = []
    for state, total, upper, target, path_weight in zip(
        states, totals, uppers, targets, weights, strict=True
    ):
        if (
            state.shape[1] != state_dimension
            or total.shape[1] != action_dimension
            or upper.shape != total.shape
            or target.shape != total.shape
            or state.shape[0] != total.shape[0]
        ):
            raise ValueError(
                "state-conditioned actor paths must share dimensions"
            )
        feature_blocks.append(causal_state_actor_features(
            state,
            total,
            upper,
            proposal_window=width,
        ))
        sample_weight_blocks.append(np.full(
            total.shape[0],
            float(path_weight) / float(total.shape[0]),
            dtype=np.float64,
        ))
    features = np.concatenate(feature_blocks, axis=0)
    outputs = np.concatenate(targets, axis=0)
    sample_weights = np.concatenate(sample_weight_blocks)
    weight_sum = float(np.sum(sample_weights))
    feature_mean = np.sum(
        sample_weights[:, None] * features, axis=0
    ) / weight_sum
    centered = features - feature_mean
    feature_scale = np.maximum(
        np.sqrt(np.sum(
            sample_weights[:, None] * np.square(centered), axis=0
        ) / weight_sum),
        scale_floor,
    )
    normalized = (features - feature_mean) / feature_scale
    normalized_targets = np.clip(outputs / limit, -1.0, 1.0)
    normalized_weights = sample_weights / weight_sum

    torch.manual_seed(seed)
    model = _ResidualMLP(
        input_dim=int(normalized.shape[1]),
        action_dim=action_dimension,
        hidden_dim=hidden,
        hidden_layers=depth,
    ).to(device="cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rate, weight_decay=decay
    )
    x = torch.as_tensor(normalized, dtype=torch.float32)
    y = torch.as_tensor(normalized_targets, dtype=torch.float32)
    w = torch.as_tensor(normalized_weights, dtype=torch.float32)
    final_loss = float("nan")
    for _ in range(epoch_count):
        optimizer.zero_grad(set_to_none=True)
        prediction = torch.tanh(model(x))
        per_sample = torch.mean(torch.square(prediction - y), dim=1)
        loss = torch.sum(w * per_sample)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    if not np.isfinite(final_loss):
        raise RuntimeError("state-conditioned actor training diverged")
    layers = []
    for layer in model.linear_layers:
        layers.append({
            "weight": layer.weight.detach().cpu().numpy().astype(
                np.float64
            ),
            "bias": layer.bias.detach().cpu().numpy().astype(np.float64),
        })
    return {
        "model_type": "causal_state_residual_mlp_v1",
        "proposal_window": width,
        "state_dimension": state_dimension,
        "action_dimension": action_dimension,
        "input_dimension": int(normalized.shape[1]),
        "hidden_dim": hidden,
        "hidden_layers": depth,
        "correction_abs_limit": limit,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "layers": layers,
        "fit_path_count": count,
        "fit_total_path_weight": float(np.sum(weights)),
        "training_loss": final_loss,
        "training_epochs": epoch_count,
        "training_seed": seed,
    }


def apply_state_conditioned_actor(
    lower_policy_state: Any,
    total_action: Any,
    proposed_upper_action: Any,
    model: dict[str, Any],
    *,
    component_sum_limit: float = 2.0,
    executed_action_limit: float = 1.0,
) -> dict[str, np.ndarray]:
    """Apply a fitted residual before nominal and environment clipping."""

    features = causal_state_actor_features(
        lower_policy_state,
        total_action,
        proposed_upper_action,
        proposal_window=int(model["proposal_window"]),
    )
    total = _matrix(total_action, role="evaluation total action")
    state = _matrix(lower_policy_state, role="evaluation lower policy state")
    if int(model["state_dimension"]) != state.shape[1]:
        raise ValueError("state-conditioned actor state dimension mismatch")
    if int(model["action_dimension"]) != total.shape[1]:
        raise ValueError("state-conditioned actor action dimension mismatch")
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    if mean.shape != (features.shape[1],) or scale.shape != mean.shape:
        raise ValueError("state-conditioned actor normalization is invalid")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("state-conditioned actor normalization is non-finite")
    if np.any(scale <= 0.0):
        raise ValueError("state-conditioned actor normalization scale is invalid")
    hidden = (features - mean) / scale
    layers = list(model["layers"])
    if len(layers) != int(model["hidden_layers"]) + 1:
        raise ValueError("state-conditioned actor layer count is invalid")
    for index, payload in enumerate(layers):
        weight = np.asarray(payload["weight"], dtype=np.float64)
        bias = np.asarray(payload["bias"], dtype=np.float64)
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("state-conditioned actor layer shape is invalid")
        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("state-conditioned actor layers do not compose")
        hidden = hidden @ weight.T + bias
        if index + 1 < len(layers):
            hidden = np.tanh(hidden)
    limit = _positive_finite(
        model["correction_abs_limit"], "correction limit"
    )
    raw_correction = limit * np.tanh(hidden)
    sum_limit = _positive_finite(component_sum_limit, "component sum limit")
    executed_limit = _positive_finite(
        executed_action_limit, "executed action limit"
    )
    corrected = np.clip(total + raw_correction, -sum_limit, sum_limit)
    correction = corrected - total
    reference_executed = np.clip(total, -executed_limit, executed_limit)
    corrected_executed = np.clip(
        corrected, -executed_limit, executed_limit
    )
    return {
        "raw_correction": raw_correction,
        "correction": correction,
        "corrected_total": corrected,
        "executed_reference": reference_executed,
        "executed_corrected": corrected_executed,
        "executed_correction": corrected_executed - reference_executed,
    }


class _ResidualMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        dimensions = [int(input_dim)]
        dimensions.extend([int(hidden_dim)] * int(hidden_layers))
        dimensions.append(int(action_dim))
        self.linear_layers = nn.ModuleList([
            nn.Linear(source, target)
            for source, target in zip(
                dimensions[:-1], dimensions[1:], strict=True
            )
        ])
        nn.init.zeros_(self.linear_layers[-1].weight)
        nn.init.zeros_(self.linear_layers[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = value
        for layer in self.linear_layers[:-1]:
            hidden = torch.tanh(layer(hidden))
        return self.linear_layers[-1](hidden)


def _matrix(value: Any, *, role: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{role} must be a finite non-empty matrix")
    return array


def _positive_finite(value: Any, role: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{role} must be positive and finite")
    return number
