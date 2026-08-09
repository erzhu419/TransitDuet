"""Differentiable frequency diagnostics for deterministic policy deployment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


DEPLOYMENT_ACTION_TRANSFORMS = ("identity", "tanh")
DEPLOYMENT_FREQUENCY_BANDS = ("low", "high")


@dataclass(frozen=True)
class DeploymentFrequencyStats:
    """A deployed action sequence's aggregate frequency constraint values."""

    power: torch.Tensor
    power_budget: torch.Tensor
    signed_excess: torch.Tensor
    violation: torch.Tensor
    normalized_signed_excess: torch.Tensor
    normalized_violation: torch.Tensor
    primitive_steps: int
    segment_count: int


def deterministic_actor_action(
    actor: torch.nn.Module,
    state: torch.Tensor,
    *,
    transform: str,
    scale: float,
) -> torch.Tensor:
    """Return the differentiable action used by deterministic deployment."""

    mode = str(transform)
    if mode not in DEPLOYMENT_ACTION_TRANSFORMS:
        raise ValueError(f"unknown deployment action transform: {mode}")
    mean = actor.distribution(state).mean
    action = torch.tanh(mean) if mode == "tanh" else mean
    return float(scale) * action


def _expanded_primitive_sequence(
    actions: torch.Tensor,
    duration: torch.Tensor,
    done: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 2 or int(actions.shape[0]) < 1:
        raise ValueError("deployment actions must have shape (n, action_dim)")
    durations = duration.reshape(-1).to(device=actions.device, dtype=torch.long)
    boundaries = done.reshape(-1).to(device=actions.device, dtype=torch.bool)
    if (
        int(durations.numel()) != int(actions.shape[0])
        or int(boundaries.numel()) != int(actions.shape[0])
    ):
        raise ValueError("deployment duration/done vectors must align with actions")
    if bool(torch.any(durations < 1).detach().cpu().item()):
        raise ValueError("deployment durations must be positive")
    primitive = torch.repeat_interleave(actions, durations, dim=0)
    primitive_boundaries = torch.zeros(
        int(primitive.shape[0]), dtype=torch.bool, device=actions.device
    )
    ends = torch.cumsum(durations, dim=0) - 1
    primitive_boundaries[ends[boundaries]] = True
    primitive_boundaries[-1] = True
    return primitive, primitive_boundaries


def _causal_rolling_low(sequence: torch.Tensor, window: int) -> torch.Tensor:
    length = int(sequence.shape[0])
    if length < 1:
        raise ValueError("deployment frequency segment cannot be empty")
    width = int(window)
    if width < 1:
        raise ValueError("deployment frequency window must be positive")
    prefix = torch.cat((
        torch.zeros(
            (1, int(sequence.shape[1])),
            dtype=sequence.dtype,
            device=sequence.device,
        ),
        torch.cumsum(sequence, dim=0),
    ), dim=0)
    end = torch.arange(1, length + 1, device=sequence.device)
    start = torch.clamp(end - width, min=0)
    count = (end - start).to(dtype=sequence.dtype).unsqueeze(-1)
    return (prefix[end] - prefix[start]) / count


def deployment_frequency_stats(
    actions: torch.Tensor,
    duration: torch.Tensor,
    done: torch.Tensor,
    *,
    window: int,
    band: str,
    rms_budget: float | None = None,
    power_budget: torch.Tensor | float | None = None,
) -> DeploymentFrequencyStats:
    """Compute an episode-reset causal frequency power and signed budget gap."""

    mode = str(band)
    if mode not in DEPLOYMENT_FREQUENCY_BANDS:
        raise ValueError(f"unknown deployment frequency band: {mode}")
    if (rms_budget is None) == (power_budget is None):
        raise ValueError(
            "provide exactly one deployment RMS or power budget"
        )
    if rms_budget is not None:
        budget = float(rms_budget)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("deployment frequency RMS budget must be positive")
        power_budget_t = torch.as_tensor(
            budget * budget, dtype=actions.dtype, device=actions.device
        )
    else:
        power_budget_t = torch.as_tensor(
            power_budget, dtype=actions.dtype, device=actions.device
        ).reshape(())
        if (
            not bool(torch.isfinite(power_budget_t).detach().cpu().item())
            or float(power_budget_t.detach().cpu().item()) <= 0.0
        ):
            raise ValueError(
                "deployment frequency power budget must be positive and finite"
            )
    primitive, boundaries = _expanded_primitive_sequence(
        actions, duration, done
    )
    components: list[torch.Tensor] = []
    start = 0
    boundary_indices = torch.nonzero(boundaries, as_tuple=False).reshape(-1)
    for boundary in boundary_indices.detach().cpu().tolist():
        end = int(boundary) + 1
        segment = primitive[start:end]
        low = _causal_rolling_low(segment, int(window))
        components.append(low if mode == "low" else segment - low)
        start = end
    if start != int(primitive.shape[0]):
        raise RuntimeError("deployment frequency boundaries did not cover trace")
    component = torch.cat(components, dim=0)
    power = torch.mean(component.square())
    signed_excess = power - power_budget_t
    normalized_signed_excess = power / power_budget_t - 1.0
    return DeploymentFrequencyStats(
        power=power,
        power_budget=power_budget_t,
        signed_excess=signed_excess,
        violation=F.relu(signed_excess),
        normalized_signed_excess=normalized_signed_excess,
        normalized_violation=F.relu(normalized_signed_excess),
        primitive_steps=int(primitive.shape[0]),
        segment_count=len(components),
    )
