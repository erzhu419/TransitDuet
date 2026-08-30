"""Causal raw-policy responsibility targets and actor-head distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ResponsibilityDistillationTargets:
    """Raw actor targets for one causally segmented trajectory."""

    upper_raw: np.ndarray
    lower_raw: np.ndarray
    upper_action: np.ndarray
    lower_action: np.ndarray
    macro_index: np.ndarray
    reconstruction_rms: float
    reconstruction_max_abs: float
    feasible_width_minimum: float


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive_scale(value: Any, action_dim: int, *, name: str) -> np.ndarray:
    scale = np.asarray(value, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full(int(action_dim), float(scale), dtype=np.float64)
    else:
        scale = scale.reshape(-1)
    if (
        scale.shape != (int(action_dim),)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError(f"{name} must be positive, finite, and action aligned")
    return scale


def _macro_index(durations: Any, lower_count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(durations, dtype=np.float64).reshape(-1)
    rounded = np.rint(values).astype(np.int64)
    if (
        values.size < 1
        or not np.all(np.isfinite(values))
        or np.any(rounded < 1)
        or not np.allclose(values, rounded, rtol=0.0, atol=1e-9)
        or int(np.sum(rounded)) != int(lower_count)
    ):
        raise ValueError(
            "upper durations must be positive integers covering lower actions"
        )
    return np.repeat(np.arange(values.size, dtype=np.int64), rounded), rounded


def _inverse_squash(action: np.ndarray, scale: np.ndarray) -> np.ndarray:
    ratio = np.asarray(action, dtype=np.float64) / np.asarray(
        scale, dtype=np.float64
    )
    if np.any(np.abs(ratio) > 1.0 + 1e-9):
        raise ValueError("distillation action exceeds its actor scale")
    return np.arctanh(np.clip(ratio, -1.0 + 1e-7, 1.0 - 1e-7))


def causal_macro_responsibility_targets(
    upper_raw: Any,
    lower_raw: Any,
    upper_durations: Any,
    *,
    upper_action_scale: Any = 1.0,
    lower_action_scale: Any = 1.0,
    slow_alpha: float = 0.25,
    transfer_strength: float = 1.0,
) -> ResponsibilityDistillationTargets:
    """Build strictly causal, exactly reconstructing upper/lower targets.

    The slow target available at macro ``m`` contains only total actions from
    completed macros ``< m``. The desired upper action is projected onto the
    intersection for which every lower complement in the current macro remains
    inside its action scale. The lower target is then the exact complement of
    the original pre-clipping total action.
    """

    upper = _finite_matrix(upper_raw, name="upper_raw")
    lower = _finite_matrix(lower_raw, name="lower_raw")
    if upper.shape[1] != lower.shape[1]:
        raise ValueError("upper and lower actions must share an action dimension")
    macro_index, durations = _macro_index(upper_durations, lower.shape[0])
    if upper.shape[0] != durations.size:
        raise ValueError("upper actions and durations must align")
    alpha = float(slow_alpha)
    strength = float(transfer_strength)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("slow_alpha must be in (0, 1]")
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("transfer_strength must be in [0, 1]")

    action_dim = upper.shape[1]
    upper_scale = _positive_scale(
        upper_action_scale, action_dim, name="upper_action_scale"
    )
    lower_scale = _positive_scale(
        lower_action_scale, action_dim, name="lower_action_scale"
    )
    upper_action = np.tanh(upper) * upper_scale
    lower_action = np.tanh(lower) * lower_scale
    repeated_upper = upper_action[macro_index]
    original_total = repeated_upper + lower_action

    target_upper = np.empty_like(upper_action)
    target_lower = np.empty_like(lower_action)
    slow_total = upper_action[0].copy()
    minimum_width = float("inf")
    start = 0
    for macro, duration in enumerate(durations):
        stop = start + int(duration)
        macro_total = original_total[start:stop]
        desired = (
            (1.0 - strength) * upper_action[macro]
            + strength * slow_total
        )
        lower_bound = np.maximum(
            -upper_scale,
            np.max(macro_total - lower_scale, axis=0),
        )
        upper_bound = np.minimum(
            upper_scale,
            np.min(macro_total + lower_scale, axis=0),
        )
        if np.any(lower_bound > upper_bound + 1e-9):
            raise RuntimeError("macro responsibility target has no feasible split")
        width = np.maximum(upper_bound - lower_bound, 0.0)
        minimum_width = min(minimum_width, float(np.min(width)))
        projected = np.minimum(np.maximum(desired, lower_bound), upper_bound)
        complement = macro_total - projected
        if np.any(np.abs(complement) > lower_scale + 1e-8):
            raise RuntimeError("projected lower complement exceeds its action scale")
        target_upper[macro] = projected
        target_lower[start:stop] = complement
        slow_total = (
            (1.0 - alpha) * slow_total
            + alpha * np.mean(macro_total, axis=0)
        )
        start = stop

    reconstructed = target_upper[macro_index] + target_lower
    error = reconstructed - original_total
    return ResponsibilityDistillationTargets(
        upper_raw=_inverse_squash(target_upper, upper_scale),
        lower_raw=_inverse_squash(target_lower, lower_scale),
        upper_action=target_upper,
        lower_action=target_lower,
        macro_index=macro_index,
        reconstruction_rms=float(np.sqrt(np.mean(np.square(error)))),
        reconstruction_max_abs=float(np.max(np.abs(error))),
        feasible_width_minimum=float(minimum_width),
    )


def _actor_features_and_head(
    actor: nn.Module, states: np.ndarray
) -> tuple[np.ndarray, nn.Linear]:
    network = getattr(actor, "net", None)
    if not isinstance(network, nn.Sequential) or not isinstance(
        network[-1], nn.Linear
    ):
        raise TypeError("responsibility distillation requires an MLP Gaussian actor")
    state = _finite_matrix(states, name="actor states")
    device = network[-1].weight.device
    dtype = network[-1].weight.dtype
    with torch.no_grad():
        tensor = torch.as_tensor(state, dtype=dtype, device=device)
        features = tensor if len(network) == 1 else network[:-1](tensor)
    return features.detach().cpu().numpy().astype(np.float64), network[-1]


def _actor_raw_actions(actor: nn.Module, states: np.ndarray) -> np.ndarray:
    network = getattr(actor, "net", None)
    if not isinstance(network, nn.Sequential):
        raise TypeError("responsibility distillation requires an MLP Gaussian actor")
    state = _finite_matrix(states, name="actor states")
    parameter = next(network.parameters())
    with torch.no_grad():
        output = network(torch.as_tensor(
            state, dtype=parameter.dtype, device=parameter.device
        ))
    return output.detach().cpu().numpy().astype(np.float64)


def fit_actor_output_head(
    actor: nn.Module,
    states: Any,
    target_raw_actions: Any,
    *,
    ridge: float = 1e-3,
    blend: float = 1.0,
) -> dict[str, float]:
    """Fit one MLP actor output head toward raw-action targets."""

    target = _finite_matrix(target_raw_actions, name="target_raw_actions")
    features, head = _actor_features_and_head(
        actor, _finite_matrix(states, name="states")
    )
    if features.shape[0] != target.shape[0] or head.out_features != target.shape[1]:
        raise ValueError("actor features and distillation targets must align")
    penalty = float(ridge)
    fraction = float(blend)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("blend must be in [0, 1]")

    design = np.concatenate(
        [features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1
    )
    prior = np.concatenate(
        [
            head.weight.detach().cpu().numpy().astype(np.float64).T,
            head.bias.detach().cpu().numpy().astype(np.float64)[None, :],
        ],
        axis=0,
    )
    gram = design.T @ design
    rhs = design.T @ target
    if penalty > 0.0:
        gram = gram + penalty * np.eye(gram.shape[0], dtype=np.float64)
        rhs = rhs + penalty * prior
    fitted = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    candidate = prior + fraction * (fitted - prior)
    before = design @ prior
    after = design @ candidate
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(
            candidate[:-1].T, dtype=head.weight.dtype, device=head.weight.device
        ))
        head.bias.copy_(torch.as_tensor(
            candidate[-1], dtype=head.bias.dtype, device=head.bias.device
        ))
    return {
        "sample_count": float(features.shape[0]),
        "target_mse_before": float(np.mean(np.square(before - target))),
        "target_mse_after": float(np.mean(np.square(after - target))),
        "parameter_delta_rms": float(
            np.sqrt(np.mean(np.square(candidate - prior)))
        ),
        "design_rank": float(np.linalg.matrix_rank(design)),
    }


def distill_hierarchical_actor_heads(
    model: Any,
    trajectories: Sequence[Any],
    *,
    upper_action_scale: Any,
    lower_action_scale: Any,
    slow_alpha: float,
    transfer_strength: float,
    ridge: float,
    blend: float,
    lower_action_context_start: int | None = None,
) -> dict[str, Any]:
    """Distill causal responsibility targets from frozen SMDP trajectories."""

    if not trajectories:
        raise ValueError("responsibility distillation requires trajectories")
    upper_states: list[np.ndarray] = []
    lower_states: list[np.ndarray] = []
    upper_targets: list[np.ndarray] = []
    target_diagnostics: list[ResponsibilityDistillationTargets] = []
    trajectory_data: list[dict[str, np.ndarray]] = []
    for trajectory in trajectories:
        upper_batch = trajectory.upper
        lower_batch = trajectory.lower
        targets = causal_macro_responsibility_targets(
            upper_batch.action,
            lower_batch.action,
            upper_batch.duration,
            upper_action_scale=upper_action_scale,
            lower_action_scale=lower_action_scale,
            slow_alpha=slow_alpha,
            transfer_strength=transfer_strength,
        )
        upper_states.append(np.asarray(upper_batch.state, dtype=np.float64))
        upper_targets.append(targets.upper_raw)
        target_diagnostics.append(targets)
        trajectory_data.append({
            "upper_raw": np.asarray(upper_batch.action, dtype=np.float64),
            "lower_raw": np.asarray(lower_batch.action, dtype=np.float64),
            "lower_state": np.asarray(lower_batch.state, dtype=np.float64),
            "macro_index": targets.macro_index,
            "teacher_upper_action": targets.upper_action,
        })

    upper_fit = fit_actor_output_head(
        model.upper_actor,
        np.concatenate(upper_states, axis=0),
        np.concatenate(upper_targets, axis=0),
        ridge=ridge,
        blend=blend,
    )
    action_dim = int(target_diagnostics[0].upper_action.shape[1])
    upper_scale = _positive_scale(
        upper_action_scale, action_dim, name="upper_action_scale"
    )
    lower_scale = _positive_scale(
        lower_action_scale, action_dim, name="lower_action_scale"
    )
    context_start = (
        None if lower_action_context_start is None
        else int(lower_action_context_start)
    )
    student_reconstruction_errors: list[np.ndarray] = []
    upper_teacher_errors: list[np.ndarray] = []
    context_shifts: list[np.ndarray] = []
    lower_targets: list[np.ndarray] = []
    for states, data in zip(upper_states, trajectory_data, strict=True):
        fitted_upper_raw = _actor_raw_actions(model.upper_actor, states)
        fitted_upper_action = np.tanh(fitted_upper_raw) * upper_scale
        macro_index = data["macro_index"].astype(np.int64, copy=False)
        original_total = (
            np.tanh(data["upper_raw"]) * upper_scale
        )[macro_index] + np.tanh(data["lower_raw"]) * lower_scale
        repeated_fitted_upper = fitted_upper_action[macro_index]
        desired_lower = original_total - repeated_fitted_upper
        fitted_lower_target = np.minimum(
            np.maximum(desired_lower, -lower_scale * (1.0 - 1e-7)),
            lower_scale * (1.0 - 1e-7),
        )
        student_reconstruction_errors.append(
            repeated_fitted_upper + fitted_lower_target - original_total
        )
        upper_teacher_errors.append(
            fitted_upper_action - data["teacher_upper_action"]
        )
        lower_state = data["lower_state"].copy()
        if context_start is not None:
            context_stop = context_start + action_dim
            if context_start < 0 or context_stop > lower_state.shape[1]:
                raise ValueError(
                    "lower action context slice is outside the lower state"
                )
            context_shifts.append(
                lower_state[:, context_start:context_stop]
                - repeated_fitted_upper
            )
            lower_state[:, context_start:context_stop] = repeated_fitted_upper
        lower_states.append(lower_state)
        lower_targets.append(_inverse_squash(
            fitted_lower_target, lower_scale
        ))

    lower_fit = fit_actor_output_head(
        model.lower_actor,
        np.concatenate(lower_states, axis=0),
        np.concatenate(lower_targets, axis=0),
        ridge=ridge,
        blend=blend,
    )
    return {
        "contract": (
            "causal_macro_raw_policy_counterfactual_context_"
            "responsibility_distillation_v2"
        ),
        "trajectory_count": int(len(trajectories)),
        "slow_alpha": float(slow_alpha),
        "transfer_strength": float(transfer_strength),
        "ridge": float(ridge),
        "blend": float(blend),
        "target_reconstruction_rms_max": float(max(
            item.reconstruction_rms for item in target_diagnostics
        )),
        "target_reconstruction_max_abs": float(max(
            item.reconstruction_max_abs for item in target_diagnostics
        )),
        "target_feasible_width_minimum": float(min(
            item.feasible_width_minimum for item in target_diagnostics
        )),
        "student_upper_teacher_action_mse": float(np.mean(np.square(
            np.concatenate(upper_teacher_errors, axis=0)
        ))),
        "student_target_reconstruction_rms_max": float(max(
            np.sqrt(np.mean(np.square(error)))
            for error in student_reconstruction_errors
        )),
        "student_target_reconstruction_max_abs": float(max(
            np.max(np.abs(error)) for error in student_reconstruction_errors
        )),
        "lower_action_context_counterfactual": bool(
            context_start is not None
        ),
        "lower_action_context_shift_rms": float(
            np.sqrt(np.mean(np.square(np.concatenate(context_shifts, axis=0))))
            if context_shifts else 0.0
        ),
        "upper_fit": upper_fit,
        "lower_fit": lower_fit,
    }
