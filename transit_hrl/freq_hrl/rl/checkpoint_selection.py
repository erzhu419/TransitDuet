"""Auditable validation-checkpoint selection for shared RL trainers."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


class RobustValidationCheckpointSelector:
    """Select checkpoints from a smoothed, materially improved validation trace.

    Training still consumes the complete registered budget.  The selector only
    determines which already-observed state is frozen for downstream tuning or
    held-out evaluation.
    """

    PROTOCOL_VERSION = "trailing_mean_material_improvement_v1"
    MINIMUM_ITERATION_PROTOCOL_VERSION = (
        "trailing_mean_material_improvement_minimum_iteration_v2"
    )
    LEGACY_PROTOCOL_VERSION = "disjoint_validation_paths"

    def __init__(
        self,
        *,
        initial_score: float,
        initial_state: Any,
        smoothing_window: int = 1,
        min_delta: float = 0.0,
        minimum_eligible_iteration: int = -1,
    ) -> None:
        if isinstance(smoothing_window, bool) or int(smoothing_window) < 1:
            raise ValueError("checkpoint smoothing_window must be positive")
        if (
            not np.isfinite(float(min_delta))
            or float(min_delta) < 0.0
        ):
            raise ValueError("checkpoint min_delta must be finite and non-negative")
        if not np.isfinite(float(initial_score)):
            raise ValueError("initial checkpoint score must be finite")
        if (
            isinstance(minimum_eligible_iteration, bool)
            or int(minimum_eligible_iteration) < -1
        ):
            raise ValueError(
                "checkpoint minimum_eligible_iteration must be at least -1"
            )

        self.smoothing_window = int(smoothing_window)
        self.min_delta = float(min_delta)
        self.minimum_eligible_iteration = int(minimum_eligible_iteration)
        self.initial_score = float(initial_score)
        self.best_score = float(initial_score)
        self.selected_raw_score = float(initial_score)
        self.selected_iteration = -1
        self.last_material_improvement_iteration = -1
        self.best_state = copy.deepcopy(initial_state)
        self._scores = [float(initial_score)]
        self._has_eligible_selection = self.minimum_eligible_iteration < 0

    def initial_history_fields(self) -> dict[str, Any]:
        eligible = self.minimum_eligible_iteration < 0
        return {
            "checkpoint_evaluation_performed": True,
            "checkpoint_selection_score": self.initial_score,
            "checkpoint_selection_eligible": eligible,
            "checkpoint_selected": eligible,
        }

    @property
    def protocol_version(self) -> str:
        if self.minimum_eligible_iteration >= 0:
            return self.MINIMUM_ITERATION_PROTOCOL_VERSION
        if self.smoothing_window == 1 and self.min_delta == 0.0:
            return self.LEGACY_PROTOCOL_VERSION
        return self.PROTOCOL_VERSION

    @property
    def has_eligible_selection(self) -> bool:
        return bool(self._has_eligible_selection)

    def consider(
        self,
        *,
        score: float,
        state: Any,
        iteration: int,
    ) -> dict[str, Any]:
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("checkpoint validation score must be finite")
        if int(iteration) < 0:
            raise ValueError("checkpoint iteration must be non-negative")

        self._scores.append(value)
        eligible = bool(
            len(self._scores) >= self.smoothing_window
            and int(iteration) >= self.minimum_eligible_iteration
        )
        selection_score = float(np.mean(
            self._scores[-self.smoothing_window:]
        ))
        selected = bool(
            eligible
            and (
                not self._has_eligible_selection
                or selection_score > self.best_score + self.min_delta
            )
        )
        if selected:
            self.best_score = selection_score
            self.selected_raw_score = value
            self.selected_iteration = int(iteration)
            self.last_material_improvement_iteration = int(iteration)
            self.best_state = copy.deepcopy(state)
            self._has_eligible_selection = True
        return {
            "checkpoint_selection_score": selection_score,
            "checkpoint_selection_eligible": eligible,
            "checkpoint_selected": selected,
        }

    def metadata(self, *, total_iterations: int) -> dict[str, Any]:
        count = int(total_iterations)
        if count < 1:
            raise ValueError("total_iterations must be positive")
        plateau_tail = (
            count
            if self.last_material_improvement_iteration < 0
            else count - self.last_material_improvement_iteration - 1
        )
        return {
            "checkpoint_selection_protocol": self.protocol_version,
            "checkpoint_smoothing_window": self.smoothing_window,
            "checkpoint_min_delta": self.min_delta,
            "checkpoint_minimum_eligible_iteration": (
                self.minimum_eligible_iteration
            ),
            "checkpoint_has_eligible_selection": bool(
                self._has_eligible_selection
            ),
            "checkpoint_selection_score": float(self.best_score),
            "selected_checkpoint_raw_score": float(self.selected_raw_score),
            "last_material_improvement_iteration": int(
                self.last_material_improvement_iteration
            ),
            "checkpoint_plateau_tail_iterations": int(plateau_tail),
            "checkpoint_validation_observation_count": len(self._scores),
        }
