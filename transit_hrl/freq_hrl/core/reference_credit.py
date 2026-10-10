"""Action-independent reference subtraction for finite-episode prefix credit."""

import numpy as np


def paired_prefix_credit(controlled, reference):
    """Keep factual transitions; subtract reference rewards at matching clocks."""
    left, right = controlled.transitions, reference.transitions
    if not (controlled.pending is None and reference.pending is None and left and len(left) == len(right)
            and controlled.scale == reference.scale and controlled.initial_cost == reference.initial_cost
            and all(a["duration_s"] == b["duration_s"] and a["done"] == b["done"] for a, b in zip(left, right))):
        raise ValueError("Reference subtraction requires finalized ledgers with identical decision clocks")
    transitions = [{**a, "reward": a["reward"] - b["reward"]} for a, b in zip(left, right)]
    rewards = np.asarray([t["reward"] for t in transitions])
    expected = controlled.scale * (right[-1]["cost_after"] - left[-1]["cost_after"])
    replay_sum = float(np.asarray(rewards, dtype=np.float32).sum(dtype=np.float64))
    if not (np.isclose(rewards.sum(), expected, rtol=1e-10, atol=1e-8)
            and np.isclose(replay_sum, expected, rtol=1e-6, atol=1e-4)):
        raise RuntimeError("Reference credit changed the terminal physical objective")
    return transitions, {"decisions": len(left), "duration_s": sum(t["duration_s"] for t in left),
        "terminal_transitions": sum(t["done"] for t in left), "paired_reward_sum": float(rewards.sum()),
        "replay_reward_sum": replay_sum, "terminal_advantage": float(expected),
        "controlled_final_cost": float(left[-1]["cost_after"]), "reference_final_cost": float(right[-1]["cost_after"]),
        "raw_reward_std": float(np.std([t["reward"] for t in left])),
        "reference_reward_std": float(np.std([t["reward"] for t in right])), "paired_reward_std": float(rewards.std())}
