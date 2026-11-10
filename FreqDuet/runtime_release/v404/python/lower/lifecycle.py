"""Trip-boundary handling for the shared lower-level bus controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, MutableMapping


@dataclass(frozen=True)
class CompletedTripEvent:
    bus_id: int
    trip_id: int
    direction: bool
    pending_states_dropped: int
    pending_action_dropped: bool
    feedback_finalized: bool
    pending_state: Any | None
    pending_action: Any | None
    previous_action_s: float
    terminal_reward: float | None
    terminal_cost: float | None
    last_board_wait_sum_s: float
    last_board_lf_wait_sum_s: float
    last_board_hf_wait_sum_s: float
    last_board_lf_mass: float
    last_board_hf_mass: float
    last_board_count: int
    last_board_station_id: int
    forward_headway: float
    backward_headway: float
    target_headway: float

    @property
    def _target_headway(self) -> float:
        """Expose the bus-compatible name used by lower value diagnostics."""
        return self.target_headway


class LowerEpisodeLifecycle:
    """Close per-bus state and feedback when a physical trip terminates."""

    BOUNDARY_MODES = {"legacy", "reset"}
    FEEDBACK_MODES = {"episode_end", "trip_end"}

    def __init__(
        self,
        boundary_mode: str = "legacy",
        feedback_mode: str = "episode_end",
    ) -> None:
        self.boundary_mode = str(boundary_mode).strip().lower()
        self.feedback_mode = str(feedback_mode).strip().lower()
        if self.boundary_mode not in self.BOUNDARY_MODES:
            raise ValueError(
                "lower.trip_boundary_mode must be one of "
                f"{sorted(self.BOUNDARY_MODES)}"
            )
        if self.feedback_mode not in self.FEEDBACK_MODES:
            raise ValueError(
                "coupling.holding_feedback_finalize_mode must be one of "
                f"{sorted(self.FEEDBACK_MODES)}"
            )
        self._closed: set[tuple[int, int]] = set()

    def reset_episode(self) -> None:
        self._closed.clear()

    def process(
        self,
        buses: Iterable[Any],
        state_dict: MutableMapping[int, list],
        action_dict: MutableMapping[int, Any],
        last_action: MutableMapping[int, float],
        holding_feedback: Any,
    ) -> list[CompletedTripEvent]:
        events: list[CompletedTripEvent] = []
        for bus in buses:
            if bool(getattr(bus, "on_route", False)):
                continue
            bus_id = int(getattr(bus, "bus_id", -1))
            completed_trip_id = getattr(bus, "last_completed_trip_id", None)
            if completed_trip_id is None:
                completed_trip_id = getattr(bus, "trip_id", -1)
            trip_id = int(completed_trip_id)
            marker = (bus_id, trip_id)
            if marker in self._closed:
                continue
            self._closed.add(marker)

            completed_direction = getattr(
                bus, "last_completed_direction", None
            )
            if completed_direction is None:
                completed_direction = not bool(getattr(bus, "direction", False))
            completed_direction = bool(completed_direction)

            feedback_finalized = False
            if self.feedback_mode == "trip_end":
                feedback_finalized = bool(holding_feedback.finalize_trip(
                    trip_id,
                    completed_direction,
                    actions=list(getattr(bus, "applied_actions", []) or []),
                ))

            states = state_dict.setdefault(bus_id, [])
            pending_state = None
            if states:
                candidate = states[-1]
                pending_state = (
                    candidate.copy() if hasattr(candidate, "copy") else candidate)
            action_candidate = action_dict.get(bus_id)
            pending_action_value = (
                action_candidate.copy()
                if hasattr(action_candidate, "copy") else action_candidate)
            previous_action_s = float(last_action.get(bus_id, 0.0))

            pending_states = 0
            pending_action = False
            if self.boundary_mode == "reset":
                pending_states = len(states)
                states.clear()
                pending_action = action_candidate is not None
                action_dict[bus_id] = None
                last_action[bus_id] = 0.0

            def optional_float(name: str) -> float | None:
                value = getattr(bus, name, None)
                return None if value is None else float(value)

            completed_station_id = getattr(
                bus, "last_completed_station_id", None)
            if completed_station_id is None:
                completed_station_id = -1

            events.append(CompletedTripEvent(
                bus_id=bus_id,
                trip_id=trip_id,
                direction=completed_direction,
                pending_states_dropped=pending_states,
                pending_action_dropped=pending_action,
                feedback_finalized=feedback_finalized,
                pending_state=pending_state,
                pending_action=pending_action_value,
                previous_action_s=previous_action_s,
                terminal_reward=optional_float("last_completed_reward"),
                terminal_cost=optional_float("last_completed_cost"),
                last_board_wait_sum_s=float(getattr(
                    bus, "last_completed_board_wait_sum_s", 0.0)),
                last_board_lf_wait_sum_s=float(getattr(
                    bus, "last_completed_board_lf_wait_sum_s", 0.0)),
                last_board_hf_wait_sum_s=float(getattr(
                    bus, "last_completed_board_hf_wait_sum_s", 0.0)),
                last_board_lf_mass=float(getattr(
                    bus, "last_completed_board_lf_mass", 0.0)),
                last_board_hf_mass=float(getattr(
                    bus, "last_completed_board_hf_mass", 0.0)),
                last_board_count=int(getattr(
                    bus, "last_completed_board_count", 0)),
                last_board_station_id=int(completed_station_id),
                forward_headway=float(getattr(bus, "forward_headway", 360.0)),
                backward_headway=float(getattr(
                    bus, "backward_headway", 360.0)),
                target_headway=float(getattr(
                    bus, "last_completed_target_headway", 360.0) or 360.0),
            ))
        return events
