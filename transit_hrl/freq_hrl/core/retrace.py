"""Finite off-policy traces over completed decision trajectories."""

from collections import deque
import random

import numpy as np
import torch


class SequenceReplayBuffer:
    """Uniform start-state sampling; terminal transitions stop each sequence."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done, duration_steps=1.0, *, behavior_log_prob):
        if not np.isfinite(behavior_log_prob) or not np.isfinite(duration_steps) or duration_steps <= 0:
            raise ValueError("Trace replay requires finite behavior density and positive duration")
        self.buffer.append((np.array(state, dtype=np.float32), np.array(action, dtype=np.float32),
            float(reward), np.array(next_state, dtype=np.float32), float(done),
            float(duration_steps), float(behavior_log_prob)))

    def sample_sequences(self, batch_size, horizon):
        rows = list(self.buffer)
        sequences, masks = [], []
        for start in random.sample(range(len(rows)), batch_size):
            sequence = []
            for row in rows[start:start + horizon]:
                if sequence and not np.array_equal(sequence[-1][3], row[0]):
                    raise ValueError("Trace replay contains disconnected decision states")
                sequence.append(row)
                if row[4]:
                    break
            masks.append([1.0] * len(sequence) + [0.0] * (horizon - len(sequence)))
            sequences.append(sequence + [sequence[-1]] * (horizon - len(sequence)))
        names = ("state", "action", "reward", "next_state", "done", "duration", "behavior_log_prob")
        return {**{name: np.asarray([[row[index] for row in sequence] for sequence in sequences],
                dtype=np.float32) for index, name in enumerate(names)}, "valid": np.asarray(masks, dtype=np.float32)}

    def __len__(self):
        return len(self.buffer)


def retrace_targets(reward, discount, next_value, action_value, log_pi, log_mu, valid, trace_lambda):
    """Backward form of Q + sum(prod(discount*c) * Bellman residual).

    All tensors are [batch, time]. ``next_value`` may be a soft value, including
    its entropy term. Terminal discounts are zero; a truncated sequence retains
    its last bootstrap. The next action's ratio weights continuation, not c_t.
    """
    coefficient = float(trace_lambda) * torch.exp(torch.clamp(log_pi - log_mu, max=0))
    result = torch.zeros_like(reward)
    for step in range(reward.shape[1] - 1, -1, -1):
        target = reward[:, step] + discount[:, step] * next_value[:, step]
        if step + 1 < reward.shape[1]:
            target = target + discount[:, step] * valid[:, step + 1] * coefficient[:, step + 1] * (
                result[:, step + 1] - action_value[:, step + 1])
        result[:, step] = target
    return result, coefficient
