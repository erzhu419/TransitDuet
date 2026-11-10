"""
lower/cost_replay_buffer.py
============================
Replay buffer supporting (s, a, r, c, s', done, trip_id) 7-tuples.

The extra fields vs standard replay buffer:
  - cost:    Lagrangian constraint signal (headway deviation²)
  - trip_id: which dispatch trip this transition belongs to (for TAP)
"""

import numpy as np
import random
from collections import deque


class CostReplayBuffer:
    """
    Replay buffer for DSAC-Lagrangian training.
    Stores transitions with cost and trip_id for TAP integration.
    """

    def __init__(self, capacity=1_000_000, seed=None):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
        self.last_episode_step = 0
        self._rng = random if seed is None else random.Random(int(seed))

    def push(self, state, action, reward, cost, next_state, done, trip_id=0):
        """
        Store a transition.

        Args:
            state: np.array
            action: float or np.array
            reward: float
            cost: float (Lagrangian constraint violation)
            next_state: np.array
            done: bool
            trip_id: int (which timetable trip this bus belongs to)
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array([action], dtype=np.float32).reshape(-1),
            float(reward),
            float(cost),
            np.array(next_state, dtype=np.float32),
            float(done),
            int(trip_id),
        ))

    def sample(self, batch_size):
        """
        Random sample from buffer.

        Returns:
            state:      np.array (batch, state_dim)
            action:     np.array (batch, action_dim)
            reward:     np.array (batch, 1)
            cost:       np.array (batch, 1)
            next_state: np.array (batch, state_dim)
            done:       np.array (batch, 1)
            trip_id:    np.array (batch,) int
        """
        batch = self._rng.sample(self.buffer, batch_size)
        states, actions, rewards, costs, next_states, dones, trip_ids = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(costs).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1),
            np.array(trip_ids),
        )

    def __len__(self):
        return len(self.buffer)

    def rng_state(self):
        return None if self._rng is random else self._rng.getstate()

    def set_rng_state(self, state):
        if state is not None:
            if self._rng is random:
                raise ValueError("cannot restore an isolated state into global RNG")
            self._rng.setstate(state)

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "buffer": list(self.buffer),
            "last_episode_step": int(self.last_episode_step),
            "rng_state": self.rng_state(),
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("lower replay capacity does not match checkpoint")
        self.buffer = deque(state["buffer"], maxlen=self.capacity)
        self.last_episode_step = int(state.get("last_episode_step", 0))
        self.set_rng_state(state.get("rng_state"))
