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


class StateAlignedLexicographicCheckpointSelector:
    """Select a state with its own multi-objective validation rank.

    Every rank component is maximized in the declared order. Unlike the
    historical trailing-mean selector, the rank attached to a checkpoint is
    computed only from that checkpoint's validation paths, so a favorable
    temporal average cannot select an individually poor state.
    """

    PROTOCOL_VERSION = "state_aligned_lexicographic_validation_v1"

    def __init__(
        self,
        *,
        initial_score: float,
        initial_rank: tuple[float, ...],
        rank_names: tuple[str, ...],
        initial_state: Any,
        minimum_eligible_iteration: int = -1,
    ) -> None:
        self.rank_names = tuple(map(str, rank_names))
        self.initial_rank = self._validated_rank(initial_rank)
        if (
            not self.rank_names
            or len(self.rank_names) != len(self.initial_rank)
            or len(set(self.rank_names)) != len(self.rank_names)
            or any(not name for name in self.rank_names)
        ):
            raise ValueError(
                "checkpoint rank names must be unique, non-empty, and aligned"
            )
        if not np.isfinite(float(initial_score)):
            raise ValueError("initial checkpoint score must be finite")
        if (
            isinstance(minimum_eligible_iteration, bool)
            or int(minimum_eligible_iteration) < -1
        ):
            raise ValueError(
                "checkpoint minimum_eligible_iteration must be at least -1"
            )
        self.minimum_eligible_iteration = int(minimum_eligible_iteration)
        self.initial_score = float(initial_score)
        self.best_score = float(initial_score)
        self.selected_raw_score = float(initial_score)
        self.best_rank = self.initial_rank
        self.selected_iteration = -1
        self.last_material_improvement_iteration = -1
        self.best_state = copy.deepcopy(initial_state)
        self._observation_count = 1
        self._has_eligible_selection = self.minimum_eligible_iteration < 0

    @staticmethod
    def _validated_rank(rank: tuple[float, ...]) -> tuple[float, ...]:
        values = tuple(float(value) for value in rank)
        if not values or not np.all(np.isfinite(values)):
            raise ValueError("checkpoint rank must contain finite values")
        return values

    def _rank_payload(self, rank: tuple[float, ...]) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.rank_names, rank)
        }

    def initial_history_fields(self) -> dict[str, Any]:
        eligible = self.minimum_eligible_iteration < 0
        return {
            "checkpoint_evaluation_performed": True,
            "checkpoint_selection_score": self.initial_score,
            "checkpoint_selection_rank": self._rank_payload(
                self.initial_rank
            ),
            "checkpoint_selection_eligible": eligible,
            "checkpoint_selected": eligible,
        }

    @property
    def protocol_version(self) -> str:
        return self.PROTOCOL_VERSION

    @property
    def has_eligible_selection(self) -> bool:
        return bool(self._has_eligible_selection)

    def consider(
        self,
        *,
        score: float,
        rank: tuple[float, ...],
        state: Any,
        iteration: int,
    ) -> dict[str, Any]:
        value = float(score)
        candidate_rank = self._validated_rank(rank)
        if len(candidate_rank) != len(self.rank_names):
            raise ValueError("checkpoint candidate rank is misaligned")
        if not np.isfinite(value):
            raise ValueError("checkpoint validation score must be finite")
        if int(iteration) < 0:
            raise ValueError("checkpoint iteration must be non-negative")
        self._observation_count += 1
        eligible = int(iteration) >= self.minimum_eligible_iteration
        selected = bool(
            eligible
            and (
                not self._has_eligible_selection
                or candidate_rank > self.best_rank
            )
        )
        if selected:
            self.best_score = value
            self.selected_raw_score = value
            self.best_rank = candidate_rank
            self.selected_iteration = int(iteration)
            self.last_material_improvement_iteration = int(iteration)
            self.best_state = copy.deepcopy(state)
            self._has_eligible_selection = True
        return {
            "checkpoint_selection_score": value,
            "checkpoint_selection_rank": self._rank_payload(candidate_rank),
            "checkpoint_selection_eligible": bool(eligible),
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
            "checkpoint_smoothing_window": 1,
            "checkpoint_min_delta": 0.0,
            "checkpoint_minimum_eligible_iteration": (
                self.minimum_eligible_iteration
            ),
            "checkpoint_has_eligible_selection": bool(
                self._has_eligible_selection
            ),
            "checkpoint_selection_score": float(self.best_score),
            "selected_checkpoint_raw_score": float(
                self.selected_raw_score
            ),
            "checkpoint_rank_names": list(self.rank_names),
            "checkpoint_initial_rank": self._rank_payload(self.initial_rank),
            "checkpoint_selected_rank": self._rank_payload(self.best_rank),
            "last_material_improvement_iteration": int(
                self.last_material_improvement_iteration
            ),
            "checkpoint_plateau_tail_iterations": int(plateau_tail),
            "checkpoint_validation_observation_count": int(
                self._observation_count
            ),
        }
