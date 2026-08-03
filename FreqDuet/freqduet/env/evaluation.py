"""Evaluation protocol and service-quality metrics for FreqDuet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class EpisodeProtocol:
    """Fixed scenario clock shared by every policy in a comparison."""

    service_start_hour: int
    service_end_hour: int
    demand_end_time_s: int
    evaluation_end_time_s: int
    allow_early_finish: bool = True

    @classmethod
    def from_config(
        cls,
        env_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        timetable_last_launch_s: float,
    ) -> "EpisodeProtocol":
        start = int(env_config.get("service_start_hour", 6))
        end = int(env_config.get("service_end_hour", 19))
        end_unwrapped = end if end >= start else end + 24
        service_hours = end_unwrapped - start + 1
        service_window_end = service_hours * 3600
        configured_demand_end = env_config.get("demand_end_time_s")
        if configured_demand_end is None:
            demand_tail_s = max(0, int(env_config.get("demand_tail_s", 0)))
            if float(timetable_last_launch_s) > 0.0:
                demand_end = min(
                    service_window_end,
                    int(np.ceil(float(timetable_last_launch_s))) + demand_tail_s,
                )
            else:
                demand_end = service_window_end
        else:
            demand_end = int(configured_demand_end)
        if demand_end <= 0:
            raise ValueError("demand_end_time_s must be positive")

        explicit_end = env_config.get(
            "evaluation_end_time_s", data_config.get("max_time")
        )
        if explicit_end is None:
            clearance = int(env_config.get("clearance_time_s", 4 * 3600))
            explicit_end = max(demand_end, int(np.ceil(timetable_last_launch_s))) + clearance
        evaluation_end = max(demand_end, int(explicit_end))
        return cls(
            service_start_hour=start,
            service_end_hour=end,
            demand_end_time_s=demand_end,
            evaluation_end_time_s=evaluation_end,
            allow_early_finish=bool(env_config.get("allow_early_finish", True)),
        )

    def demand_active(self, current_time_s: float) -> bool:
        return 0.0 <= float(current_time_s) < float(self.demand_end_time_s)

    def should_terminate(
        self,
        current_time_s: float,
        all_trips_launched: bool,
        any_bus_on_route: bool,
    ) -> tuple[bool, str | None]:
        if float(current_time_s) >= float(self.evaluation_end_time_s):
            return True, "evaluation_horizon"
        cleared = all_trips_launched and not any_bus_on_route
        if (
            self.allow_early_finish
            and float(current_time_s) >= float(self.demand_end_time_s)
            and cleared
        ):
            return True, "service_cleared"
        return False, None


class HeadwayEventRecorder:
    """Collect inter-arrival headways at stop events, not bus snapshots."""

    def __init__(self) -> None:
        self._last_arrival: dict[tuple[int, bool], float] = {}
        self.headways_s: list[float] = []
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        station_id: int,
        direction: bool,
        arrival_time_s: float,
        trip_id: int,
        target_headway_s: float | None = None,
    ) -> None:
        key = (int(station_id), bool(direction))
        time_s = float(arrival_time_s)
        previous = self._last_arrival.get(key)
        headway_s = None
        if previous is not None:
            headway_s = max(0.0, time_s - previous)
            self.headways_s.append(headway_s)
        self._last_arrival[key] = time_s
        self.events.append({
            "station_id": key[0],
            "direction": key[1],
            "arrival_time_s": time_s,
            "trip_id": int(trip_id),
            "headway_s": headway_s,
            "target_headway_s": (
                None if target_headway_s is None
                else float(target_headway_s)
            ),
        })

    def previous_arrival_time(
        self, station_id: int, direction: bool
    ) -> float | None:
        """Return the last causal arrival before the caller records its event."""
        value = self._last_arrival.get((int(station_id), bool(direction)))
        return None if value is None else float(value)

    def summary(self) -> dict[str, float | int]:
        values = np.asarray(self.headways_s, dtype=np.float64)
        if values.size == 0:
            return {
                "headway_event_count": len(self.events),
                "headway_sample_count": 0,
                "headway_mean_s": 0.0,
                "headway_std_s": 0.0,
                "headway_cv": 0.0,
            }
        mean = float(values.mean())
        std = float(values.std())
        return {
            "headway_event_count": len(self.events),
            "headway_sample_count": int(values.size),
            "headway_mean_s": mean,
            "headway_std_s": std,
            "headway_cv": std / max(mean, 1.0),
        }


def compute_wait_metrics(
    stations: Iterable[Any],
    censor_time_s: float,
) -> dict[str, float | int]:
    """Compute boarded-only and fixed-horizon restricted waiting metrics."""

    observed_waits: list[float] = []
    adjusted_waits: list[float] = []
    observed_lf_wait_sum_s = 0.0
    observed_hf_wait_sum_s = 0.0
    restricted_lf_wait_sum_s = 0.0
    restricted_hf_wait_sum_s = 0.0
    observed_in_vehicle_s: list[float] = []
    observed_journey_s: list[float] = []
    restricted_in_vehicle_sum_s = 0.0
    restricted_journey_sum_s = 0.0
    lf_mass = 0.0
    hf_mass = 0.0
    max_share_error = 0.0
    generated = 0
    boarded = 0
    arrived = 0
    for station in stations:
        for passenger in station.total_passenger:
            generated += 1
            appear = float(passenger.appear_time)
            low_share = float(getattr(passenger, "frequency_low_share", 1.0))
            high_share = float(getattr(passenger, "frequency_high_share", 0.0))
            max_share_error = max(
                max_share_error, abs(low_share + high_share - 1.0))
            if low_share < 0.0 or high_share < 0.0:
                raise ValueError("passenger frequency shares must be non-negative")
            lf_mass += low_share
            hf_mass += high_share
            boarding = getattr(passenger, "boarding_time", None)
            if boarding is None:
                wait_s = max(0.0, float(censor_time_s) - appear)
                adjusted_waits.append(wait_s)
                restricted_lf_wait_sum_s += low_share * wait_s
                restricted_hf_wait_sum_s += high_share * wait_s
                restricted_journey_sum_s += wait_s
                continue
            wait_s = max(0.0, float(boarding) - appear)
            observed_waits.append(wait_s)
            adjusted_waits.append(wait_s)
            observed_lf_wait_sum_s += low_share * wait_s
            observed_hf_wait_sum_s += high_share * wait_s
            restricted_lf_wait_sum_s += low_share * wait_s
            restricted_hf_wait_sum_s += high_share * wait_s
            boarded += 1
            arrive = getattr(passenger, "arrive_time", None)
            if arrive is None:
                in_vehicle_s = max(0.0, float(censor_time_s) - float(boarding))
                restricted_in_vehicle_sum_s += in_vehicle_s
                restricted_journey_sum_s += max(
                    0.0, float(censor_time_s) - appear)
                continue
            in_vehicle_s = max(0.0, float(arrive) - float(boarding))
            journey_s = max(0.0, float(arrive) - appear)
            observed_in_vehicle_s.append(in_vehicle_s)
            observed_journey_s.append(journey_s)
            restricted_in_vehicle_sum_s += in_vehicle_s
            restricted_journey_sum_s += journey_s
            arrived += 1

    unserved = generated - boarded
    observed_mean = float(np.mean(observed_waits)) if observed_waits else 0.0
    adjusted_mean = float(np.mean(adjusted_waits)) if adjusted_waits else 0.0
    return {
        "passengers_generated": generated,
        "passengers_boarded": boarded,
        "passengers_arrived": arrived,
        "passengers_unserved": unserved,
        "passenger_unserved_rate": unserved / max(generated, 1),
        "avg_wait_observed_min": observed_mean / 60.0,
        "restricted_wait_horizon_min": adjusted_mean / 60.0,
        # Compatibility alias for protocol-v2 scripts created before the
        # metric was given its precise restricted-wait name.
        "avg_wait_censored_min": adjusted_mean / 60.0,
        "avg_wait_lf_observed_min": (
            observed_lf_wait_sum_s / max(boarded, 1) / 60.0
        ),
        "avg_wait_hf_observed_min": (
            observed_hf_wait_sum_s / max(boarded, 1) / 60.0
        ),
        "restricted_wait_lf_horizon_min": (
            restricted_lf_wait_sum_s / max(generated, 1) / 60.0
        ),
        "restricted_wait_hf_horizon_min": (
            restricted_hf_wait_sum_s / max(generated, 1) / 60.0
        ),
        "frequency_lf_passenger_mass": lf_mass,
        "frequency_hf_passenger_mass": hf_mass,
        "frequency_share_max_error": max_share_error,
        "avg_in_vehicle_observed_min": (
            float(np.mean(observed_in_vehicle_s)) / 60.0
            if observed_in_vehicle_s else 0.0
        ),
        "restricted_in_vehicle_horizon_min": (
            restricted_in_vehicle_sum_s / max(generated, 1) / 60.0
        ),
        "avg_total_journey_observed_min": (
            float(np.mean(observed_journey_s)) / 60.0
            if observed_journey_s else 0.0
        ),
        "restricted_total_journey_horizon_min": (
            restricted_journey_sum_s / max(generated, 1) / 60.0
        ),
    }


def composite_service_cost(
    avg_wait_min: float,
    peak_fleet: float,
    headway_cv: float,
    n_fleet: float,
    passenger_unserved_rate: float = 0.0,
    trip_completion_rate: float = 1.0,
    weights: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Selection cost with explicit service-reliability penalties.

    Individual components must still be reported. This scalar is for training
    and model selection, not a replacement for multi-outcome inference.
    """

    configured = {
        "wait": 1.0,
        "fleet": 1.0,
        "headway": 1.0,
        "unserved": 5.0,
        "incomplete_service": 5.0,
    }
    if weights:
        configured.update({str(key): float(value) for key, value in weights.items()})
        if "unlaunched" in weights and "incomplete_service" not in weights:
            configured["incomplete_service"] = float(weights["unlaunched"])
    overshoot = max(0.0, float(peak_fleet) - float(n_fleet))
    components = {
        "wait": float(avg_wait_min) / 10.0,
        "fleet": overshoot ** 2 / max(float(n_fleet), 1.0),
        "headway": float(headway_cv),
        "unserved": max(0.0, float(passenger_unserved_rate)),
        "incomplete_service": max(
            0.0, 1.0 - float(trip_completion_rate)),
    }
    total = sum(configured[name] * value for name, value in components.items())
    return float(total), components


def normalize_wait_metric(value: str | None) -> str:
    """Normalize the configured wait basis used for scalar model selection."""
    aliases = {
        "boarded": "observed",
        "boarded_only": "observed",
        "censored": "restricted",
        "restricted_horizon": "restricted",
    }
    metric = aliases.get(str(value or "observed").strip().lower(),
                         str(value or "observed").strip().lower())
    if metric not in {"observed", "restricted"}:
        raise ValueError(
            "wait metric must be 'observed' or 'restricted'")
    return metric


def service_cost_views(
    measurement_details: Mapping[str, Any],
    peak_fleet: float,
    headway_cv: float,
    n_fleet: float,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Compute boarded-only and restricted-horizon costs on one outcome."""
    common = {
        "peak_fleet": peak_fleet,
        "headway_cv": headway_cv,
        "n_fleet": n_fleet,
        "passenger_unserved_rate": float(
            measurement_details.get("passenger_unserved_rate", 0.0)),
        "trip_completion_rate": float(
            measurement_details.get("trip_completion_rate", 1.0)),
        "weights": weights,
    }
    observed, _ = composite_service_cost(
        avg_wait_min=float(measurement_details["avg_wait_observed_min"]),
        **common,
    )
    restricted, _ = composite_service_cost(
        avg_wait_min=float(
            measurement_details["restricted_wait_horizon_min"]),
        **common,
    )
    return {"observed": float(observed), "restricted": float(restricted)}
