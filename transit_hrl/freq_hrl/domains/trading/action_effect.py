"""Trading action-effect operators."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...core.leakage import ActionEffectOperator, as_2d


class TradingActionEffectOperator(ActionEffectOperator):
    """Map portfolio targets and executions to frequency-constrained effects."""

    def __init__(self, target_history: Sequence[Any] | None = None) -> None:
        self.target_history = None if target_history is None else as_2d(target_history)

    def upper_effect(self, upper_action_history: Sequence[Any]) -> np.ndarray:
        # Target weights/inventory should stay low-pass.
        return as_2d(upper_action_history)

    def lower_effect(self, lower_action_history: Sequence[Any]) -> np.ndarray:
        # Executions accumulate into inventory.  If target history is supplied,
        # constrain the drift relative to the high-level target.
        inventory = np.cumsum(as_2d(lower_action_history), axis=0)
        if self.target_history is None:
            return inventory
        n = min(inventory.shape[0], self.target_history.shape[0])
        return inventory[:n] - self.target_history[:n]
