"""Parameter-budget matching for flat and two-level actor-critic models."""

from __future__ import annotations


def mlp_parameter_count(in_dim: int, out_dim: int, hidden_dim: int) -> int:
    hidden = int(hidden_dim)
    if hidden < 1:
        return int(in_dim * out_dim + out_dim)
    return int(
        in_dim * hidden
        + hidden
        + hidden * hidden
        + hidden
        + hidden * out_dim
        + out_dim
    )


def flat_actor_critic_parameter_count(
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> int:
    return int(
        mlp_parameter_count(state_dim, action_dim, hidden_dim)
        + action_dim
        + mlp_parameter_count(state_dim, 1, hidden_dim)
    )


def hierarchical_actor_critic_parameter_count(
    *,
    upper_state_dim: int,
    lower_state_dim: int,
    goal_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> int:
    return int(
        mlp_parameter_count(upper_state_dim, goal_dim, hidden_dim)
        + goal_dim
        + mlp_parameter_count(lower_state_dim, action_dim, hidden_dim)
        + action_dim
        + mlp_parameter_count(upper_state_dim, 1, hidden_dim)
        + mlp_parameter_count(lower_state_dim, 1, hidden_dim)
    )


def matched_hierarchical_hidden_dim(
    *,
    target_parameter_count: int,
    upper_state_dim: int,
    lower_state_dim: int,
    goal_dim: int,
    action_dim: int,
    maximum_hidden_dim: int = 256,
) -> tuple[int, int, float]:
    target = int(target_parameter_count)
    if target < 1:
        raise ValueError("target parameter count must be positive")
    candidates = []
    for hidden in range(1, int(maximum_hidden_dim) + 1):
        actual = hierarchical_actor_critic_parameter_count(
            upper_state_dim=upper_state_dim,
            lower_state_dim=lower_state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            hidden_dim=hidden,
        )
        candidates.append((abs(actual - target), hidden, actual))
    _, hidden, actual = min(candidates)
    return hidden, actual, float(actual / target)
