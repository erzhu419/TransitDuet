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
        self._last_arrival_trip: dict[tuple[int, bool], int] = {}
        self._last_departure: dict[tuple[int, bool], dict[str, float | int]] = {}
        self._action_ready: dict[tuple[int, bool, int], float] = {}
        self._follower_forecast_decisions = 0
        self._follower_forecasts: list[dict[str, Any]] = []
        self._follower_forecasts_by_current: dict[
            tuple[int, bool, int], list[dict[str, Any]]
        ] = {}
        self._follower_forecasts_by_follower_ready: dict[
            tuple[int, bool, int], list[dict[str, Any]]
        ] = {}
        self._follower_forecasts_by_follower_action: dict[
            tuple[int, bool, int], list[dict[str, Any]]
        ] = {}
        self._follower_forecasts_by_follower_departure: dict[
            tuple[int, bool, int], list[dict[str, Any]]
        ] = {}
        self._executed_holding_actions: dict[
            tuple[int, bool, int], float
        ] = {}
        self._resolved_follower_forecasts: list[dict[str, Any]] = []
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
        self._last_arrival_trip[key] = int(trip_id)
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

    def previous_arrival_event(
        self, station_id: int, direction: bool
    ) -> dict[str, float | int] | None:
        key = (int(station_id), bool(direction))
        if key not in self._last_arrival:
            return None
        return {
            "time_s": float(self._last_arrival[key]),
            "trip_id": int(self._last_arrival_trip[key]),
        }

    def record_departure(
        self,
        station_id: int,
        direction: bool,
        departure_time_s: float,
        trip_id: int,
    ) -> None:
        station = int(station_id)
        direction_value = bool(direction)
        trip = int(trip_id)
        departure = float(departure_time_s)
        self._last_departure[(station, direction_value)] = {
            "time_s": departure,
            "trip_id": int(trip_id),
        }
        key = (station, direction_value, trip)
        touched: list[dict[str, Any]] = []
        for row in self._follower_forecasts_by_current.pop(key, []):
            row["current_departure_time_s"] = departure
            touched.append(row)
        for row in self._follower_forecasts_by_follower_departure.pop(
                key, []):
            row["follower_departure_time_s"] = departure
            touched.append(row)
        for row in touched:
            self._resolve_follower_forecast(row)

    def record_action_ready(
        self,
        station_id: int,
        direction: bool,
        action_ready_time_s: float,
        trip_id: int,
    ) -> None:
        """Resolve forecasts at the pre-control point they actually predict."""
        key = (int(station_id), bool(direction), int(trip_id))
        ready = float(action_ready_time_s)
        self._action_ready[key] = ready
        for row in self._follower_forecasts_by_follower_ready.pop(key, []):
            row["follower_action_ready_time_s"] = ready
            self._resolve_follower_forecast(row)

    def record_executed_holding_action(
        self,
        station_id: int,
        direction: bool,
        trip_id: int,
        action_s: float,
    ) -> None:
        """Record the final physical action, excluding simulator tick delay."""
        action = float(action_s)
        if not np.isfinite(action) or action < 0.0:
            return
        key = (int(station_id), bool(direction), int(trip_id))
        self._executed_holding_actions[key] = action
        for row in self._follower_forecasts_by_follower_action.pop(key, []):
            row["follower_action_s"] = action
            self._update_follower_followup(row)

    def record_follower_departure_forecast(
        self,
        *,
        station_id: int,
        direction: bool,
        decision_time_s: float | None,
        current_trip_id: int | None,
        follower_trip_id: int | None,
        predicted_follower_gap_s: float | None,
        forward_departure_gap_s: float | None,
        action_s: float,
        action_cap_s: float,
        source: str | None,
        base_predicted_follower_gap_s: float | None = None,
        calibration_requested_adjustment_s: float = 0.0,
        calibration_effective_adjustment_s: float = 0.0,
        calibration_features: Iterable[float] | None = None,
        calibration_active: bool = False,
        calibration_history_episodes: int = 0,
        calibration_mode: str = "disabled",
    ) -> bool:
        """Register one causal AVL forecast for later departure calibration."""
        self._follower_forecast_decisions += 1
        source_value = str(source or "unavailable").strip().lower()
        numeric = (
            decision_time_s,
            predicted_follower_gap_s,
            forward_departure_gap_s,
            action_s,
            action_cap_s,
        )
        if (
            current_trip_id is None
            or follower_trip_id is None
            or int(current_trip_id) == int(follower_trip_id)
            or not source_value.startswith("same_time_avl_")
            or any(value is None for value in numeric)
        ):
            return False
        decision, predicted_gap, forward_gap, action, action_cap = (
            float(value) for value in numeric
        )
        base_predicted_gap = (
            predicted_gap if base_predicted_follower_gap_s is None
            else float(base_predicted_follower_gap_s)
        )
        requested_adjustment = float(calibration_requested_adjustment_s)
        effective_adjustment = float(calibration_effective_adjustment_s)
        if (
            not np.isfinite(np.asarray(
                [
                    decision,
                    predicted_gap,
                    base_predicted_gap,
                    forward_gap,
                    action,
                    action_cap,
                    requested_adjustment,
                    effective_adjustment,
                ]
            )).all()
            or min(
                decision,
                predicted_gap,
                base_predicted_gap,
                forward_gap,
                action,
            ) < 0.0
            or action_cap <= 0.0
        ):
            return False

        feature_values = None
        if calibration_features is not None:
            candidate = np.asarray(
                list(calibration_features), dtype=np.float64).reshape(-1)
            if candidate.size and np.isfinite(candidate).all():
                feature_values = tuple(float(value) for value in candidate)

        row: dict[str, Any] = {
            "station_id": int(station_id),
            "direction": bool(direction),
            "decision_time_s": decision,
            "current_trip_id": int(current_trip_id),
            "follower_trip_id": int(follower_trip_id),
            "predicted_follower_gap_s": predicted_gap,
            "base_predicted_follower_gap_s": base_predicted_gap,
            "forward_departure_gap_s": forward_gap,
            "action_s": action,
            "action_cap_s": action_cap,
            "source": source_value,
            "calibration_requested_adjustment_s": requested_adjustment,
            "calibration_effective_adjustment_s": effective_adjustment,
            "calibration_features": feature_values,
            "calibration_active": bool(calibration_active),
            "calibration_history_episodes": int(
                calibration_history_episodes),
            "calibration_mode": str(calibration_mode),
            "current_departure_time_s": None,
            "follower_action_ready_time_s": None,
            "follower_action_s": None,
            "follower_departure_time_s": None,
            "resolved": False,
        }
        self._follower_forecasts.append(row)
        current_key = (
            int(station_id), bool(direction), int(current_trip_id))
        follower_key = (
            int(station_id), bool(direction), int(follower_trip_id))
        self._follower_forecasts_by_current.setdefault(
            current_key, []).append(row)
        self._follower_forecasts_by_follower_ready.setdefault(
            follower_key, []).append(row)
        self._follower_forecasts_by_follower_action.setdefault(
            follower_key, []).append(row)
        self._follower_forecasts_by_follower_departure.setdefault(
            follower_key, []).append(row)
        if follower_key in self._action_ready:
            row["follower_action_ready_time_s"] = self._action_ready[
                follower_key]
            self._resolve_follower_forecast(row)
        if follower_key in self._executed_holding_actions:
            row["follower_action_s"] = self._executed_holding_actions[
                follower_key]
            self._update_follower_followup(row)
        return True

    def _resolve_follower_forecast(self, row: dict[str, Any]) -> None:
        if row["resolved"]:
            self._update_follower_followup(row)
            return
        current_departure = row["current_departure_time_s"]
        follower_ready = row["follower_action_ready_time_s"]
        if current_departure is None or follower_ready is None:
            return

        actual_raw_gap = float(follower_ready) - row["decision_time_s"]
        actual_post_hold_gap = (
            float(follower_ready) - float(current_departure))
        predicted_post_hold_gap = max(
            row["predicted_follower_gap_s"] - row["action_s"], 0.0)
        predicted_target = float(np.clip(
            0.5 * (
                row["predicted_follower_gap_s"]
                - row["forward_departure_gap_s"]
            ),
            0.0,
            row["action_cap_s"],
        ))
        base_predicted_target = float(np.clip(
            0.5 * (
                row["base_predicted_follower_gap_s"]
                - row["forward_departure_gap_s"]
            ),
            0.0,
            row["action_cap_s"],
        ))
        realized_target = float(np.clip(
            0.5 * (
                actual_raw_gap - row["forward_departure_gap_s"]
            ),
            0.0,
            row["action_cap_s"],
        ))
        row.update({
            "actual_follower_gap_s": actual_raw_gap,
            "raw_gap_prediction_error_s": (
                row["predicted_follower_gap_s"] - actual_raw_gap),
            "base_gap_prediction_error_s": (
                row["base_predicted_follower_gap_s"] - actual_raw_gap),
            "predicted_post_hold_gap_s": predicted_post_hold_gap,
            "actual_post_hold_gap_s": actual_post_hold_gap,
            "post_hold_gap_prediction_error_s": (
                predicted_post_hold_gap - actual_post_hold_gap),
            "predicted_target_action_s": predicted_target,
            "base_predicted_target_action_s": base_predicted_target,
            "realized_target_action_s": realized_target,
            "target_action_prediction_error_s": (
                predicted_target - realized_target),
            "base_target_action_prediction_error_s": (
                base_predicted_target - realized_target),
            "calibration_target_adjustment_s": (
                predicted_target - base_predicted_target),
            "departure_timing_error_s": (
                float(current_departure)
                - row["decision_time_s"]
                - row["action_s"]),
            "hold_need_false_positive": float(
                predicted_target > 1e-9 and realized_target <= 1e-9),
            "hold_need_false_negative": float(
                predicted_target <= 1e-9 and realized_target > 1e-9),
            "base_hold_need_false_positive": float(
                base_predicted_target > 1e-9 and realized_target <= 1e-9),
            "base_hold_need_false_negative": float(
                base_predicted_target <= 1e-9 and realized_target > 1e-9),
            "resolved": True,
        })
        self._resolved_follower_forecasts.append(row)
        self._update_follower_followup(row)

    @staticmethod
    def _update_follower_followup(row: dict[str, Any]) -> None:
        action = row.get("follower_action_s")
        if action is not None:
            row["follower_future_hold_s"] = max(float(action), 0.0)
        ready = row.get("follower_action_ready_time_s")
        departure = row.get("follower_departure_time_s")
        if ready is None or departure is None or action is None:
            return
        row["follower_action_execution_error_s"] = (
            float(departure) - float(ready) - float(action))

    def previous_departure_event(
        self, station_id: int, direction: bool
    ) -> dict[str, float | int] | None:
        event = self._last_departure.get((int(station_id), bool(direction)))
        return None if event is None else dict(event)

    def follower_forecast_calibration_samples(self) -> list[dict[str, Any]]:
        """Return completed matched rows for a post-episode history update."""
        return [
            dict(row)
            for row in self._resolved_follower_forecasts
            if row.get("calibration_features") is not None
        ]

    def summary(self) -> dict[str, float | int]:
        values = np.asarray(self.headways_s, dtype=np.float64)
        if values.size == 0:
            result = {
                "headway_event_count": len(self.events),
                "headway_sample_count": 0,
                "headway_mean_s": 0.0,
                "headway_std_s": 0.0,
                "headway_cv": 0.0,
            }
        else:
            mean = float(values.mean())
            std = float(values.std())
            result = {
                "headway_event_count": len(self.events),
                "headway_sample_count": int(values.size),
                "headway_mean_s": mean,
                "headway_std_s": std,
                "headway_cv": std / max(mean, 1.0),
            }
        result.update(self._follower_forecast_summary())
        return result

    def _follower_forecast_summary(self) -> dict[str, float | int]:
        registered = len(self._follower_forecasts)
        resolved = len(self._resolved_follower_forecasts)
        action_resolved_rows = [
            row for row in self._resolved_follower_forecasts
            if "follower_future_hold_s" in row
        ]
        departure_resolved_rows = [
            row for row in self._resolved_follower_forecasts
            if "follower_action_execution_error_s" in row
        ]
        result: dict[str, float | int] = {
            "follower_forecast_decision_count": int(
                self._follower_forecast_decisions),
            "follower_forecast_registered_count": int(registered),
            "follower_forecast_resolved_count": int(resolved),
            "follower_forecast_valid_rate": float(
                registered / max(self._follower_forecast_decisions, 1)),
            "follower_forecast_resolution_rate": float(
                resolved / max(registered, 1)),
            "follower_forecast_action_resolved_count": int(
                len(action_resolved_rows)),
            "follower_forecast_departure_resolved_count": int(
                len(departure_resolved_rows)),
        }
        metric_names = (
            "predicted_follower_gap_s",
            "base_predicted_follower_gap_s",
            "actual_follower_gap_s",
            "raw_gap_prediction_error_s",
            "base_gap_prediction_error_s",
            "post_hold_gap_prediction_error_s",
            "predicted_target_action_s",
            "base_predicted_target_action_s",
            "realized_target_action_s",
            "target_action_prediction_error_s",
            "base_target_action_prediction_error_s",
            "calibration_requested_adjustment_s",
            "calibration_effective_adjustment_s",
            "calibration_target_adjustment_s",
            "calibration_active",
            "calibration_history_episodes",
            "departure_timing_error_s",
            "hold_need_false_positive",
            "hold_need_false_negative",
            "base_hold_need_false_positive",
            "base_hold_need_false_negative",
        )
        if not resolved:
            for name in metric_names:
                result[f"follower_forecast_{name}_mean"] = 0.0
            result.update({
                "follower_forecast_raw_gap_prediction_mae_s": 0.0,
                "follower_forecast_raw_gap_prediction_rmse_s": 0.0,
                "follower_forecast_raw_gap_prediction_p90_abs_s": 0.0,
                "follower_forecast_base_gap_prediction_mae_s": 0.0,
                "follower_forecast_base_gap_prediction_rmse_s": 0.0,
                "follower_forecast_target_action_prediction_mae_s": 0.0,
                "follower_forecast_base_target_action_prediction_mae_s": 0.0,
                "follower_forecast_calibration_requested_adjustment_abs_mean_s": 0.0,
                "follower_forecast_calibration_target_adjustment_abs_mean_s": 0.0,
                "follower_forecast_follower_future_hold_s_mean": 0.0,
                "follower_forecast_follower_future_hold_positive_rate": 0.0,
                "follower_forecast_follower_action_execution_error_s_mean": 0.0,
            })
            return result

        arrays = {
            name: np.asarray(
                [row[name] for row in self._resolved_follower_forecasts],
                dtype=np.float64,
            )
            for name in metric_names
        }
        for name, array in arrays.items():
            result[f"follower_forecast_{name}_mean"] = float(array.mean())
        raw_error = arrays["raw_gap_prediction_error_s"]
        base_gap_error = arrays["base_gap_prediction_error_s"]
        target_error = arrays["target_action_prediction_error_s"]
        base_target_error = arrays["base_target_action_prediction_error_s"]
        result.update({
            "follower_forecast_raw_gap_prediction_mae_s": float(
                np.abs(raw_error).mean()),
            "follower_forecast_raw_gap_prediction_rmse_s": float(
                np.sqrt(np.mean(raw_error ** 2))),
            "follower_forecast_raw_gap_prediction_p90_abs_s": float(
                np.quantile(np.abs(raw_error), 0.9)),
            "follower_forecast_base_gap_prediction_mae_s": float(
                np.abs(base_gap_error).mean()),
            "follower_forecast_base_gap_prediction_rmse_s": float(
                np.sqrt(np.mean(base_gap_error ** 2))),
            "follower_forecast_target_action_prediction_mae_s": float(
                np.abs(target_error).mean()),
            "follower_forecast_base_target_action_prediction_mae_s": float(
                np.abs(base_target_error).mean()),
            "follower_forecast_calibration_requested_adjustment_abs_mean_s": float(
                np.abs(arrays[
                    "calibration_requested_adjustment_s"]).mean()),
            "follower_forecast_calibration_target_adjustment_abs_mean_s": float(
                np.abs(arrays["calibration_target_adjustment_s"]).mean()),
        })
        future_holds = np.asarray(
            [row["follower_future_hold_s"] for row in action_resolved_rows],
            dtype=np.float64,
        )
        execution_errors = np.asarray([
            row["follower_action_execution_error_s"]
            for row in departure_resolved_rows
        ], dtype=np.float64)
        result.update({
            "follower_forecast_follower_future_hold_s_mean": (
                float(future_holds.mean()) if future_holds.size else 0.0),
            "follower_forecast_follower_future_hold_positive_rate": (
                float((future_holds > 1e-9).mean())
                if future_holds.size else 0.0),
            "follower_forecast_follower_action_execution_error_s_mean": (
                float(execution_errors.mean())
                if execution_errors.size else 0.0),
        })
        return result


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
