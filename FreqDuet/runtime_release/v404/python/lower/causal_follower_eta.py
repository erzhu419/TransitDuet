"""Same-time AVL estimate of the following bus's departure gap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AVLVehicleSnapshot:
    bus_id: int
    direction: bool
    on_route: bool
    progress_m: float
    launch_time_s: float
    current_speed_mps: float
    route_speed_mps: float


@dataclass(frozen=True)
class FollowerDepartureEstimate:
    departure_gap_s: float | None
    eta_s: float | None
    spatial_gap_m: float | None
    speed_mps: float | None
    follower_bus_id: int | None
    source: str
    valid: bool


def estimate_follower_departure_gap(
    *,
    current_bus_id: int,
    current_direction: bool,
    current_progress_m: float,
    current_time_s: float,
    service_dwell_proxy_s: float,
    vehicles: Iterable[AVLVehicleSnapshot],
) -> FollowerDepartureEstimate:
    """Estimate the nearest physical follower using only current AVL state.

    The follower must be active, travel in the same direction, and be spatially
    behind the controlled bus. Its observed journey-average speed is preferred
    because it already reflects upstream dwell and holding. Current speed and a
    conservative fraction of the static segment speed are causal fallbacks.
    """

    progress = _finite_nonnegative(current_progress_m)
    now = _finite_nonnegative(current_time_s)
    dwell = _finite_nonnegative(service_dwell_proxy_s)
    if progress is None or now is None or dwell is None:
        return _invalid("invalid_current_avl")

    candidates: list[tuple[float, AVLVehicleSnapshot]] = []
    for vehicle in vehicles:
        if int(vehicle.bus_id) == int(current_bus_id):
            continue
        if not bool(vehicle.on_route):
            continue
        if bool(vehicle.direction) != bool(current_direction):
            continue
        follower_progress = _finite_nonnegative(vehicle.progress_m)
        if follower_progress is None:
            continue
        gap = progress - follower_progress
        if gap > 1e-6:
            candidates.append((float(gap), vehicle))

    if not candidates:
        return _invalid("no_same_direction_avl_follower")

    spatial_gap, follower = min(candidates, key=lambda item: item[0])
    elapsed = now - float(follower.launch_time_s)
    journey_speed = (
        float(follower.progress_m) / elapsed
        if np.isfinite(elapsed) and elapsed > 1e-6
        and float(follower.progress_m) > 0.0
        else None
    )
    current_speed = _finite_positive(follower.current_speed_mps)
    route_speed = _finite_positive(follower.route_speed_mps)

    if journey_speed is not None and np.isfinite(journey_speed) \
            and journey_speed > 0.25:
        speed = float(journey_speed)
        source = "same_time_avl_journey_speed_eta"
    elif current_speed is not None:
        speed = float(current_speed)
        source = "same_time_avl_current_speed_eta"
    elif route_speed is not None:
        speed = 0.5 * float(route_speed)
        source = "same_time_avl_route_speed_eta"
    else:
        return _invalid("follower_speed_unavailable")

    eta = spatial_gap / max(speed, 0.25)
    departure_gap = eta + dwell
    if not np.isfinite(departure_gap) or departure_gap < 0.0:
        return _invalid("invalid_follower_eta")
    return FollowerDepartureEstimate(
        departure_gap_s=float(departure_gap),
        eta_s=float(eta),
        spatial_gap_m=float(spatial_gap),
        speed_mps=float(speed),
        follower_bus_id=int(follower.bus_id),
        source=source,
        valid=True,
    )


def _invalid(source: str) -> FollowerDepartureEstimate:
    return FollowerDepartureEstimate(
        departure_gap_s=None,
        eta_s=None,
        spatial_gap_m=None,
        speed_mps=None,
        follower_bus_id=None,
        source=str(source),
        valid=False,
    )


def _finite_nonnegative(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) and result >= 0.0 else None


def _finite_positive(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) and result > 0.0 else None
