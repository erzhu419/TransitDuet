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
                    trip_id, completed_direction
                ))

            pending_states = 0
            pending_action = False
            if self.boundary_mode == "reset":
                states = state_dict.setdefault(bus_id, [])
                pending_states = len(states)
                states.clear()
                pending_action = action_dict.get(bus_id) is not None
                action_dict[bus_id] = None
                last_action[bus_id] = 0.0

            events.append(CompletedTripEvent(
                bus_id=bus_id,
                trip_id=trip_id,
                direction=completed_direction,
                pending_states_dropped=pending_states,
                pending_action_dropped=pending_action,
                feedback_finalized=feedback_finalized,
            ))
        return events
