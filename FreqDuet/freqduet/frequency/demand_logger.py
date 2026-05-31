"""Demand/frequency trace logging for FreqDuet Phase-0 audits.

The logger is intentionally passive: it records realized passenger arrivals,
station queues, boarding wait, lower holding actions, and the online frequency
state without changing the environment dynamics or policy inputs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


class DemandEventLogger:
    """Append aggregate and station-level demand-frequency traces to CSV."""

    AGGREGATE_HEADER = [
        "ep",
        "time_s",
        "bin_index",
        "arrivals",
        "active_station_count",
        "queue_total",
        "queue_max",
        "boarded",
        "board_wait_mean_s",
        "lower_action_count",
        "lower_action_mean_s",
        "headway_mean_s",
        "target_headway_mean_s",
        "holding_bus_count",
        "on_route_bus_count",
        "freq_low_demand",
        "freq_low_slope",
        "freq_low_forecast",
        "freq_high_energy",
        "freq_middle",
        "freq_middle_energy",
        "freq_promotion_flag",
        "freq_promotion_strength",
        "freq_promotion_age",
        "freq_promotion_score",
        "freq_promotion_direction",
    ]

    STATION_HEADER = [
        "ep",
        "time_s",
        "bin_index",
        "station_id",
        "direction",
        "arrivals",
        "queue",
        "boarded",
        "board_wait_mean_s",
        "lower_action_count",
        "lower_action_mean_s",
        "local_low",
        "local_high",
        "local_high_raw",
        "local_high_feature",
        "local_high_noise_floor",
        "local_high_energy",
        "local_high_energy_feature",
        "local_high_prior_share",
        "local_middle",
        "local_middle_energy",
        "local_promotion_flag",
        "local_promotion_strength",
        "local_promotion_age",
        "local_promotion_score",
        "local_promotion_direction",
    ]

    def __init__(
        self,
        output_dir,
        aggregate_filename="demand_trace.csv",
        station_filename="demand_station_trace.csv",
        station_rows=True,
        bin_only=True,
        include_empty_stations=False,
    ):
        self.output_dir = Path(output_dir)
        self.aggregate_path = self.output_dir / aggregate_filename
        self.station_path = self.output_dir / station_filename
        self.station_rows = bool(station_rows)
        self.bin_only = bool(bin_only)
        self.include_empty_stations = bool(include_empty_stations)
        self.episode = 0
        self._seen_board_events = set()
        self._seen_action_events = set()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_header(self.aggregate_path, self.AGGREGATE_HEADER)
        if self.station_rows:
            self._ensure_header(self.station_path, self.STATION_HEADER)

    @staticmethod
    def _ensure_header(path, header):
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

    @staticmethod
    def _append_row(path, header, row):
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow({key: row.get(key, "") for key in header})

    def start_episode(self, episode):
        self.episode = int(episode)
        self._seen_board_events.clear()
        self._seen_action_events.clear()

    def flush(self):
        """Compatibility hook for env shutdown; rows are written eagerly."""
        return None

    @staticmethod
    def _mean(values, default=0.0):
        values = list(values)
        return float(np.mean(values)) if values else float(default)

    def _collect_board_events(self, buses, current_time_s):
        by_station = defaultdict(lambda: [0, 0.0])
        total_boarded = 0
        total_wait = 0.0
        for bus in buses:
            count = int(getattr(bus, "last_board_count", 0) or 0)
            if count <= 0:
                continue
            event_time = getattr(bus, "last_board_time", None)
            if event_time is None or float(event_time) > float(current_time_s):
                continue
            station_id = int(getattr(bus, "last_board_station_id", -1))
            key = (
                int(getattr(bus, "bus_id", -1)),
                int(getattr(bus, "trip_id", -1)),
                station_id,
                int(event_time),
                count,
            )
            if key in self._seen_board_events:
                continue
            self._seen_board_events.add(key)
            direction = bool(getattr(bus, "direction", False))
            wait_sum = float(getattr(bus, "last_board_wait_sum_s", 0.0) or 0.0)
            by_station[(station_id, direction)][0] += count
            by_station[(station_id, direction)][1] += wait_sum
            total_boarded += count
            total_wait += wait_sum
        return by_station, total_boarded, total_wait

    def _collect_action_events(self, buses, current_time_s):
        by_station = defaultdict(list)
        all_actions = []
        for bus in buses:
            event_time = getattr(bus, "last_action_time", None)
            if event_time is None or float(event_time) > float(current_time_s):
                continue
            station_id = int(getattr(bus, "last_action_station_id", -1))
            action = float(getattr(bus, "last_action_s", 0.0) or 0.0)
            key = (
                int(getattr(bus, "bus_id", -1)),
                int(getattr(bus, "trip_id", -1)),
                station_id,
                int(event_time),
                round(action, 6),
            )
            if key in self._seen_action_events:
                continue
            self._seen_action_events.add(key)
            direction = bool(getattr(bus, "direction", False))
            by_station[(station_id, direction)].append(action)
            all_actions.append(action)
        return by_station, all_actions

    @staticmethod
    def _local_state_summary(tracker, key):
        if tracker is not None and hasattr(tracker, "local_trace_summary"):
            return tracker.local_trace_summary(*key)
        state = getattr(tracker, "local_states", {}).get(key)
        if state is None:
            return {
                "local_low": 0.0,
                "local_high": 0.0,
                "local_high_raw": 0.0,
                "local_high_feature": 0.0,
                "local_high_noise_floor": 0.0,
                "local_high_energy": 0.0,
                "local_high_energy_feature": 0.0,
                "local_high_prior_share": 0.0,
                "local_middle": 0.0,
                "local_middle_energy": 0.0,
            }
        return {
            "local_low": float(getattr(state, "low", 0.0)),
            "local_high": float(getattr(state, "high", 0.0)),
            "local_high_raw": float(getattr(state, "high", 0.0)),
            "local_high_feature": float(getattr(state, "high", 0.0)),
            "local_high_noise_floor": 0.0,
            "local_high_energy": float(
                np.sqrt(max(getattr(state, "high_energy", 0.0), 0.0))),
            "local_high_energy_feature": float(
                np.sqrt(max(getattr(state, "high_energy", 0.0), 0.0))),
            "local_high_prior_share": 0.0,
            "local_middle": float(getattr(state, "middle", 0.0)),
            "local_middle_energy": float(
                np.sqrt(max(getattr(state, "middle_energy", 0.0), 0.0))),
        }

    def log_step(
        self,
        current_time_s,
        arrivals_by_station,
        stations,
        buses,
        tracker,
        bin_applied=False,
    ):
        if self.bin_only and not bin_applied:
            return
        arrivals_by_station = arrivals_by_station or {}
        stations = list(stations or [])
        buses = list(buses or [])
        board_by_station, boarded, board_wait_sum = self._collect_board_events(
            buses, current_time_s)
        action_by_station, actions = self._collect_action_events(
            buses, current_time_s)

        queue_lengths = [len(getattr(s, "waiting_passengers", [])) for s in stations]
        on_route = [b for b in buses if getattr(b, "on_route", False)]
        holding = [
            b for b in on_route
            if str(getattr(getattr(b, "state", None), "name", "")) == "HOLDING"
        ]
        headways = [
            float(getattr(b, "forward_headway", 0.0))
            for b in on_route
            if float(getattr(b, "forward_headway", 0.0) or 0.0) > 0.0
        ]
        targets = [
            float(getattr(b, "_target_headway", 0.0))
            for b in on_route
            if float(getattr(b, "_target_headway", 0.0) or 0.0) > 0.0
        ]
        summary = tracker.summary() if tracker is not None else {}
        aggregate = {
            "ep": self.episode,
            "time_s": int(current_time_s),
            "bin_index": int(summary.get("freq_updates", 0)),
            "arrivals": int(sum(arrivals_by_station.values())),
            "active_station_count": int(sum(1 for v in arrivals_by_station.values() if v)),
            "queue_total": int(sum(queue_lengths)),
            "queue_max": int(max(queue_lengths) if queue_lengths else 0),
            "boarded": int(boarded),
            "board_wait_mean_s": (
                float(board_wait_sum) / max(int(boarded), 1)
                if boarded > 0 else 0.0),
            "lower_action_count": int(len(actions)),
            "lower_action_mean_s": self._mean(actions),
            "headway_mean_s": self._mean(headways, default=360.0),
            "target_headway_mean_s": self._mean(targets, default=360.0),
            "holding_bus_count": int(len(holding)),
            "on_route_bus_count": int(len(on_route)),
        }
        for key in self.AGGREGATE_HEADER:
            if key.startswith("freq_"):
                aggregate[key] = float(summary.get(key, 0.0))
        self._append_row(self.aggregate_path, self.AGGREGATE_HEADER, aggregate)

        if not self.station_rows:
            return
        station_keys = set(arrivals_by_station) | set(board_by_station) | set(action_by_station)
        if self.include_empty_stations:
            station_keys.update(
                (int(getattr(s, "station_id", -1)), bool(getattr(s, "direction", False)))
                for s in stations
            )
        station_by_key = {
            (int(getattr(s, "station_id", -1)), bool(getattr(s, "direction", False))): s
            for s in stations
        }
        for key in sorted(station_keys, key=lambda x: (int(x[1]), int(x[0]))):
            station = station_by_key.get(key)
            if station is None and not self.include_empty_stations:
                queue = 0
            else:
                queue = len(getattr(station, "waiting_passengers", []))
            boarded_station, wait_station = board_by_station.get(key, [0, 0.0])
            station_actions = action_by_station.get(key, [])
            local = self._local_state_summary(tracker, key)
            promotion = (
                tracker.local_promotion_summary(*key)
                if tracker is not None and hasattr(tracker, "local_promotion_summary")
                else {"flag": 0.0, "strength": 0.0, "age": 0.0}
            )
            row = {
                "ep": self.episode,
                "time_s": int(current_time_s),
                "bin_index": int(summary.get("freq_updates", 0)),
                "station_id": int(key[0]),
                "direction": int(bool(key[1])),
                "arrivals": int(arrivals_by_station.get(key, 0)),
                "queue": int(queue),
                "boarded": int(boarded_station),
                "board_wait_mean_s": (
                    float(wait_station) / max(int(boarded_station), 1)
                    if boarded_station > 0 else 0.0),
                "lower_action_count": int(len(station_actions)),
                "lower_action_mean_s": self._mean(station_actions),
                **local,
                "local_promotion_flag": float(promotion.get("flag", 0.0)),
                "local_promotion_strength": float(promotion.get("strength", 0.0)),
                "local_promotion_age": float(promotion.get("age", 0.0)),
                "local_promotion_score": float(promotion.get("score", 0.0)),
                "local_promotion_direction": float(
                    promotion.get("direction", 0.0)),
            }
            self._append_row(self.station_path, self.STATION_HEADER, row)
