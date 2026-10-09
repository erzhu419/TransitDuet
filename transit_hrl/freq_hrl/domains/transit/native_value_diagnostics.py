"""Frozen native upper critic and temporal-credit diagnostics."""

import numpy as np
import torch


def critic_action_curve(trainer, states, goals):
    """Query a physical-action grid at identical recorded states, without RNG."""
    device = next(trainer.q_net.parameters()).device
    states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    means, stds = [], []
    with torch.no_grad():
        for goal in goals:
            actions = torch.full((len(states), 1), float(goal), device=device)
            values = trainer.q_net(states, actions)
            means.append(values.mean(dim=0).cpu().numpy())
            stds.append(values.std(dim=0).cpu().numpy())
    means, stds = np.stack(means), np.stack(stds)
    lcb = means + trainer.beta * stds
    if not np.isfinite(lcb).all():
        raise RuntimeError("Nonfinite native critic curve")
    best = np.argmax(lcb, axis=0)
    return {str(goal): {"q_mean": float(np.mean(means[index])),
        "ensemble_std_mean": float(np.mean(stds[index])),
        "lcb_mean": float(np.mean(lcb[index])),
        "lcb_argmax_state_fraction": float(np.mean(best == index))}
        for index, goal in enumerate(goals)}


def credit_ledger(transitions):
    durations = [row["duration_s"] for row in transitions]
    waits = [row["interval_wait_cost"] for row in transitions]
    wait_sum = sum(waits)
    return {"transitions": len(transitions),
        "duration_median_s": float(np.median(durations)),
        "duration_p95_s": float(np.quantile(durations, .95)),
        "last_duration_s": float(durations[-1]),
        "last_wait_credit_fraction": float(waits[-1] / wait_sum) if wait_sum else 0.0,
        "system_reward_sum": float(sum(row["system_reward"] for row in transitions)),
        "gap_credit_sum": float(sum(row["gap_credit"] for row in transitions)),
        "interval_reward_sum": float(sum(row["interval_reward"] for row in transitions))}


def critic_input_scale(trainer, states, action_s=60.0):
    """Compare first-layer state/action contributions at recorded causal states."""
    weight = trainer.q_net.weights[0]
    state = torch.as_tensor(np.stack(states), dtype=torch.float32, device=weight.device)
    action = torch.full((len(state), 1), float(action_s), device=weight.device)
    if trainer.critic_action_units == "unit":
        action = trainer.q_net.action_coordinates(action)
    with torch.no_grad():
        state_part = torch.einsum("bd,kdh->kbh", state, weight[:, :state.shape[1]])
        action_part = torch.einsum("bd,kdh->kbh", action, weight[:, state.shape[1]:])
    return {"critic_action_units": trainer.critic_action_units, "physical_action_s": action_s,
        "state_contribution_abs_mean": float(state_part.abs().mean()),
        "action_contribution_abs_mean": float(action_part.abs().mean())}
