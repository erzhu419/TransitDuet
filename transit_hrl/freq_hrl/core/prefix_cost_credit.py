"""Undiscounted finite-episode credit for an unchanged terminal cost."""

import numpy as np


class PrefixCostCredit:
    """Close one transition per actual plan decision, never per lower step."""

    def __init__(self, scale=100.0):
        self.scale, self.pending, self.transitions = float(scale), None, []
        self.initial_cost = None

    def _close(self, state, cost, time, done):
        previous = self.pending
        if previous is None:
            return
        if time < previous["time"] or not np.isfinite(cost):
            raise ValueError("Prefix costs need finite values and ordered decision times")
        self.transitions.append({"state": previous["state"], "action": previous["action"],
            "reward": self.scale * (previous["cost"] - cost),
            "next_state": np.asarray(state, dtype=np.float32).copy(), "done": bool(done),
            "duration_s": float(time - previous["time"]),
            "cost_before": previous["cost"], "cost_after": float(cost)})

    def begin(self, state, action, cost, time):
        if not np.isfinite(cost):
            raise ValueError("Nonfinite causal prefix cost")
        self._close(state, cost, time, False)
        if self.initial_cost is None:
            self.initial_cost = float(cost)
        self.pending = {"state": np.asarray(state, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(), "cost": float(cost), "time": float(time)}

    def finish(self, cost, time):
        if self.pending is None:
            raise ValueError("No upper plan decisions to finalize")
        self._close(self.pending["state"], cost, time, True)
        self.pending = None
        total = sum(t["reward"] for t in self.transitions)
        expected = self.scale * (self.initial_cost - cost)
        if not np.isclose(total, expected, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Upper credit does not telescope to terminal service cost")
        return {"decisions": len(self.transitions), "reward_sum": total,
            "initial_cost": self.initial_cost, "final_cost": float(cost),
            "duration_s": sum(t["duration_s"] for t in self.transitions),
            "terminal_transitions": sum(t["done"] for t in self.transitions)}
