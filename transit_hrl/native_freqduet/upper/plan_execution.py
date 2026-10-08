"""Execution contract for rolling upper-level timetable plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


_REPLAY_ACTION_SOURCES = {"executed", "policy_command"}


@dataclass(frozen=True)
class UpperPlanExecutionContract:
    """Keep policy commands, smoothed plans, state context, and time aligned."""

    replay_action_source: str = "executed"
    include_plan_context: bool = False
    duration_discount: bool = False
    duration_base_s: float = 900.0
    duration_min_steps: float = 0.25
    duration_max_steps: float = 4.0

    def __post_init__(self) -> None:
        source = str(self.replay_action_source).strip().lower()
        if source not in _REPLAY_ACTION_SOURCES:
            raise ValueError(
                "upper.timetable_planner.execution_contract."
                "replay_action_source must be one of "
                f"{sorted(_REPLAY_ACTION_SOURCES)}"
            )
        if not np.isfinite(self.duration_base_s) or self.duration_base_s <= 0.0:
            raise ValueError("duration_base_s must be finite and positive")
        if not np.isfinite(self.duration_min_steps) or self.duration_min_steps <= 0.0:
            raise ValueError("duration_min_steps must be finite and positive")
        if (
            not np.isfinite(self.duration_max_steps)
            or self.duration_max_steps < self.duration_min_steps
        ):
            raise ValueError(
                "duration_max_steps must be finite and no smaller than "
                "duration_min_steps"
            )
        object.__setattr__(self, "replay_action_source", source)

    @classmethod
    def from_config(
        cls, planner_config: Mapping | None
    ) -> "UpperPlanExecutionContract":
        cfg = dict((planner_config or {}).get("execution_contract", {}) or {})
        return cls(
            replay_action_source=cfg.get("replay_action_source", "executed"),
            include_plan_context=bool(cfg.get("include_plan_context", False)),
            duration_discount=bool(cfg.get("duration_discount", False)),
            duration_base_s=float(cfg.get("duration_base_s", 900.0)),
            duration_min_steps=float(cfg.get("duration_min_steps", 0.25)),
            duration_max_steps=float(cfg.get("duration_max_steps", 4.0)),
        )

    def context_dim(self, action_dim: int) -> int:
        return int(action_dim) + 2 if self.include_plan_context else 0

    def plan_context(
        self,
        active_plan: Mapping | None,
        decision_time_s: float,
        action_low,
        action_high,
        replan_interval_s: float,
    ) -> np.ndarray:
        """Return normalized previous executed plan, age, and presence flag."""
        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if low.shape != high.shape or np.any(high <= low):
            raise ValueError("upper action bounds must be aligned and increasing")
        if not self.include_plan_context:
            return np.zeros(0, dtype=np.float32)
        if active_plan is None:
            return np.zeros(low.size + 2, dtype=np.float32)

        action = np.asarray(active_plan.get("action"), dtype=np.float32).reshape(-1)
        if action.shape != low.shape:
            raise ValueError(
                "active timetable plan action does not match configured action bounds"
            )
        action_unit = 2.0 * (np.clip(action, low, high) - low) / (high - low) - 1.0
        interval = max(float(replan_interval_s), 1.0)
        age = max(
            0.0,
            float(decision_time_s) - float(active_plan.get("origin", decision_time_s)),
        )
        age_norm = float(np.clip(age / interval, 0.0, 2.0))
        return np.concatenate(
            [action_unit, np.asarray([age_norm, 1.0], dtype=np.float32)]
        ).astype(np.float32)

    def replay_action(self, policy_command, executed_action) -> np.ndarray:
        command = np.asarray(policy_command, dtype=np.float32).reshape(-1)
        executed = np.asarray(executed_action, dtype=np.float32).reshape(-1)
        if command.shape != executed.shape:
            raise ValueError("policy command and executed action shapes differ")
        if self.replay_action_source == "policy_command":
            return command.copy()
        return executed.copy()

    def duration_steps(self, elapsed_s: float) -> float:
        if not self.duration_discount:
            return 1.0
        steps = max(float(elapsed_s), 0.0) / self.duration_base_s
        return float(np.clip(
            steps, self.duration_min_steps, self.duration_max_steps
        ))
