"""Passive actor probes for the preserved native Transit feature layout."""

import numpy as np
import torch


FEATURE_BLOCKS = {
    "upper": {"dynamic_band": (1, 11), "all_frequency": (1, 11, 12, 13, 14, 15)},
    "lower": {"dynamic_band": (29, 30, 31, 32)},
}
ACTION_BOUNDS = {"upper": (-120.0, 120.0), "lower": (0.0, 60.0)}


class NativePolicyProbe:
    """Record actor inputs without changing sampling or network state.

    Zero-input sensitivities are evaluated in batches after the physical
    episode. An optional zero-action intervention retains the original actor
    call, but substitutes its actuator command, not its features or weights.
    """

    def __init__(self, policy, level, *, neutral=False, fixed_action_s=None):
        self.policy = policy
        self.level = level
        self.neutral = neutral
        self.fixed_action_s = fixed_action_s
        self.original_get_action = policy.get_action
        self.states = []

    def get_action(self, state, deterministic=False):
        values = state.detach().cpu().numpy() if torch.is_tensor(state) else state
        self.states.append(np.asarray(values, dtype=np.float32).reshape(-1).copy())
        action = self.original_get_action(state, deterministic=deterministic)
        if self.fixed_action_s is not None:
            return np.full_like(action, self.fixed_action_s)
        return np.zeros_like(action) if self.neutral else action

    def summarize(self):
        states = np.stack(self.states)
        width = 16 if self.level == "upper" else 33
        if states.shape[1] != width or not np.isfinite(states).all():
            raise ValueError(f"Unexpected native {self.level} actor inputs")
        device = next(self.policy.parameters()).device
        proposals, means, deviations = [], [], {key: [] for key in FEATURE_BLOCKS[self.level]}
        with torch.no_grad():
            for start in range(0, len(states), 512):
                batch = torch.as_tensor(states[start:start + 512], device=device)
                mean, _ = self.policy.forward(batch)
                action = np.asarray(self.original_get_action(batch, deterministic=True)).reshape(-1)
                proposals.append(action)
                means.append(mean.cpu().numpy().reshape(-1))
                for name, indices in FEATURE_BLOCKS[self.level].items():
                    changed = batch.clone()
                    changed[:, list(indices)] = 0.0
                    alternative = np.asarray(self.original_get_action(changed, deterministic=True)).reshape(-1)
                    deviations[name].append(np.abs(action - alternative))
        proposals = np.concatenate(proposals)
        low, high = ACTION_BOUNDS[self.level]
        result = {
            "actor_calls": len(states),
            "input_abs_mean": np.mean(np.abs(states), axis=0).tolist(),
            "input_abs_max": np.max(np.abs(states), axis=0).tolist(),
            "proposal_mean_s": float(np.mean(proposals)),
            "proposal_std_s": float(np.std(proposals)),
            "proposal_edge_fraction_1pct": float(np.mean(
                (proposals <= low + .01 * (high - low))
                | (proposals >= high - .01 * (high - low)))),
            "latent_mean_abs_p95": float(np.quantile(np.abs(np.concatenate(means)), .95)),
            "zero_input_effect_s": {
                name: {"mean_abs": float(np.mean(np.concatenate(values))),
                       "p95_abs": float(np.quantile(np.concatenate(values), .95))}
                for name, values in deviations.items()},
            "neutral_action": self.neutral,
            "fixed_action_s": self.fixed_action_s,
        }
        return result
