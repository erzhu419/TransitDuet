"""Causal service-outcome credit for upper timetable-plan intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


_ASSIGNMENT_MODES = {"additive", "local_mean"}


@dataclass
class _IntervalAccumulator:
    start_time_s: float
    direction: bool | None
    sampled_duration_s: float = 0.0
    waiting_exposure_s: float = 0.0
    fleet_excess_exposure_bus_s: float = 0.0
    headway_abs_deviation_sum: float = 0.0
    headway_sample_count: int = 0


class UpperIntervalOutcomeTracker:
    """Partition physical outcomes by the upper plan that could cause them.

    Directional planner streams accumulate directional queues, buses and
    headway events. A global planner stream accumulates network-wide values.
    The additive scoring mode partitions the episode objective across
    intervals; ``local_mean`` is retained as a denser experimental variant.
    """

    def __init__(
        self,
        enabled: bool = False,
        assignment_mode: str = "additive",
        wait_weight: float = 1.0,
        headway_weight: float = 1.0,
        fleet_weight: float = 1.0,
        reward_scale: float = 1.0,
        wait_reference_min: float = 10.0,
        local_wait_queue_norm: float = 100.0,
        headway_reference: float = 1.0,
        fleet_reference: float = 1.0,
        component_clip: float = 4.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.assignment_mode = str(assignment_mode).strip().lower()
        if self.assignment_mode not in _ASSIGNMENT_MODES:
            raise ValueError(
                "upper.interval_credit.assignment_mode must be additive or "
                "local_mean")
        self.wait_weight = self._nonnegative(wait_weight, "wait_weight")
        self.headway_weight = self._nonnegative(
            headway_weight, "headway_weight")
        self.fleet_weight = self._nonnegative(fleet_weight, "fleet_weight")
        self.reward_scale = self._nonnegative(reward_scale, "reward_scale")
        self.wait_reference_min = self._positive(
            wait_reference_min, "wait_reference_min")
        self.local_wait_queue_norm = self._positive(
            local_wait_queue_norm, "local_wait_queue_norm")
        self.headway_reference = self._positive(
            headway_reference, "headway_reference")
        self.fleet_reference = self._positive(
            fleet_reference, "fleet_reference")
        self.component_clip = self._positive(
            component_clip, "component_clip")
        self.reset()

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    @staticmethod
    def _positive(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None
    ) -> "UpperIntervalOutcomeTracker":
        cfg = dict(config or {})
        weights = dict(cfg.get("weights", {}) or {})
        return cls(
            enabled=cfg.get("enable", False),
            assignment_mode=cfg.get("assignment_mode", "additive"),
            wait_weight=weights.get("wait", cfg.get("wait_weight", 1.0)),
            headway_weight=weights.get(
                "headway", cfg.get("headway_weight", 1.0)),
            fleet_weight=weights.get(
                "fleet", cfg.get("fleet_weight", 1.0)),
            reward_scale=cfg.get("reward_scale", 1.0),
            wait_reference_min=cfg.get("wait_reference_min", 10.0),
            local_wait_queue_norm=cfg.get("local_wait_queue_norm", 100.0),
            headway_reference=cfg.get("headway_reference", 1.0),
            fleet_reference=cfg.get("fleet_reference", 1.0),
            component_clip=cfg.get("component_clip", 4.0),
        )

    def reset(self) -> None:
        self._active: dict[Hashable, _IntervalAccumulator] = {}
        self._headway_event_cursor = 0

    @staticmethod
    def _stream_direction(stream_key: Hashable) -> bool | None:
        if isinstance(stream_key, (bool, np.bool_)):
            return bool(stream_key)
        return None

    def begin(self, stream_key: Hashable, start_time_s: float) -> None:
        if not self.enabled:
            return
        if stream_key in self._active:
            raise RuntimeError(
                f"upper interval stream {stream_key!r} was opened twice")
        self._active[stream_key] = _IntervalAccumulator(
            start_time_s=float(start_time_s),
            direction=self._stream_direction(stream_key),
        )

    def record_step(
        self,
        *,
        dt_s: float,
        waiting_by_direction: Mapping[bool, float],
        fleet_by_direction: Mapping[bool, float],
        n_fleet_target: float,
        headway_events: Sequence[Mapping[str, Any]],
    ) -> None:
        if not self.enabled:
            return
        dt_s = max(float(dt_s), 0.0)
        new_events = headway_events[self._headway_event_cursor:]
        self._headway_event_cursor = len(headway_events)
        if not self._active or dt_s <= 0.0:
            return

        target_global = max(float(n_fleet_target), 1.0)
        for accumulator in self._active.values():
            direction = accumulator.direction
            if direction is None:
                waiting = sum(float(v) for v in waiting_by_direction.values())
                fleet = sum(float(v) for v in fleet_by_direction.values())
                fleet_target = target_global
            else:
                waiting = float(waiting_by_direction.get(direction, 0.0))
                fleet = float(fleet_by_direction.get(direction, 0.0))
                fleet_target = max(target_global / 2.0, 1.0)

            accumulator.sampled_duration_s += dt_s
            accumulator.waiting_exposure_s += max(waiting, 0.0) * dt_s
            accumulator.fleet_excess_exposure_bus_s += (
                max(0.0, fleet - fleet_target) * dt_s)

            for event in new_events:
                if direction is not None and bool(
                        event.get("direction")) != direction:
                    continue
                headway_s = event.get("headway_s")
                target_s = event.get("target_headway_s")
                if headway_s is None or target_s is None:
                    continue
                headway_s = float(headway_s)
                target_s = float(target_s)
                if not np.isfinite(headway_s) or not np.isfinite(target_s):
                    continue
                target_s = max(target_s, 1.0)
                deviation = abs(headway_s - target_s) / target_s
                accumulator.headway_abs_deviation_sum += min(
                    deviation, self.component_clip)
                accumulator.headway_sample_count += 1

    def close(
        self, stream_key: Hashable, end_time_s: float
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        accumulator = self._active.pop(stream_key, None)
        if accumulator is None:
            return None
        wall_duration_s = max(
            0.0, float(end_time_s) - accumulator.start_time_s)
        coverage = accumulator.sampled_duration_s / max(wall_duration_s, 1.0)
        return {
            "stream_key": stream_key,
            "direction": accumulator.direction,
            "start_time_s": accumulator.start_time_s,
            "end_time_s": float(end_time_s),
            "wall_duration_s": wall_duration_s,
            "sampled_duration_s": accumulator.sampled_duration_s,
            "coverage": float(np.clip(coverage, 0.0, 1.0)),
            "waiting_exposure_s": accumulator.waiting_exposure_s,
            "fleet_excess_exposure_bus_s": (
                accumulator.fleet_excess_exposure_bus_s),
            "headway_abs_deviation_sum": (
                accumulator.headway_abs_deviation_sum),
            "headway_sample_count": accumulator.headway_sample_count,
        }

    def score(
        self,
        outcome: Mapping[str, Any] | None,
        *,
        passengers_generated: int,
        episode_headway_samples: int,
        episode_duration_s: float,
        n_fleet_target: float,
    ) -> dict[str, float]:
        if not self.enabled or outcome is None:
            return {
                "reward": 0.0,
                "wait_cost": 0.0,
                "headway_cost": 0.0,
                "fleet_cost": 0.0,
            }

        sampled_duration_s = max(
            float(outcome.get("sampled_duration_s", 0.0)), 1.0)
        waiting_exposure_s = max(
            float(outcome.get("waiting_exposure_s", 0.0)), 0.0)
        headway_sum = max(
            float(outcome.get("headway_abs_deviation_sum", 0.0)), 0.0)
        headway_count = max(
            int(outcome.get("headway_sample_count", 0)), 0)
        fleet_exposure = max(
            float(outcome.get("fleet_excess_exposure_bus_s", 0.0)), 0.0)
        direction = outcome.get("direction")
        fleet_target = max(
            float(n_fleet_target) / (2.0 if direction is not None else 1.0),
            1.0,
        )

        if self.assignment_mode == "additive":
            wait_cost = waiting_exposure_s / (
                max(int(passengers_generated), 1)
                * 60.0
                * self.wait_reference_min
            )
            headway_cost = headway_sum / (
                max(int(episode_headway_samples), 1)
                * self.headway_reference
            )
            fleet_cost = fleet_exposure / (
                max(float(episode_duration_s), 1.0)
                * fleet_target
                * self.fleet_reference
            )
        else:
            wait_cost = (
                waiting_exposure_s / sampled_duration_s
                / self.local_wait_queue_norm
            )
            headway_cost = (
                headway_sum / max(headway_count, 1)
                / self.headway_reference
            )
            fleet_cost = (
                fleet_exposure / sampled_duration_s
                / fleet_target
                / self.fleet_reference
            )

        wait_cost = float(np.clip(wait_cost, 0.0, self.component_clip))
        headway_cost = float(np.clip(
            headway_cost, 0.0, self.component_clip))
        fleet_cost = float(np.clip(fleet_cost, 0.0, self.component_clip))
        reward = -self.reward_scale * (
            self.wait_weight * wait_cost
            + self.headway_weight * headway_cost
            + self.fleet_weight * fleet_cost
        )
        return {
            "reward": float(reward),
            "wait_cost": wait_cost,
            "headway_cost": headway_cost,
            "fleet_cost": fleet_cost,
        }
