"""Transit action-effect operators."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...core.leakage import ActionEffectOperator, as_2d


class TransitActionEffectOperator(ActionEffectOperator):
    """Map transit plans and holding actions to frequency-constrained effects."""

    def upper_effect(self, upper_action_history: Sequence[Any]) -> np.ndarray:
        # Headway/timetable plan deltas are already the low-frequency effect.
        return as_2d(upper_action_history)

    def lower_effect(self, lower_action_history: Sequence[Any]) -> np.ndarray:
        # Holding seconds accumulate into downstream schedule drift.
        return np.cumsum(np.maximum(as_2d(lower_action_history), 0.0), axis=0)
