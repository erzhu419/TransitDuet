"""Causal learned residual service plans with common physical-cost credit."""

import numpy as np

from freq_hrl.core.prefix_cost_credit import PrefixCostCredit
from freq_hrl.domains.transit.native_service_plan import NativeServicePlan
from native_freqduet.env.evaluation import compute_wait_metrics, composite_service_cost

STATE_DIM = 34
RESIDUAL_SCALE_S = 20.0


def prefix_service_cost(env):
    """Use only already generated passengers and the current censoring clock."""
    wait = compute_wait_metrics(env.stations, censor_time_s=env.current_time)
    completed = len(env._completed_trip_ids.intersection(range(len(env.timetables))))
    return composite_service_cost(wait["restricted_wait_horizon_min"], env._peak_concurrent,
        env.headway_events.summary()["headway_cv"], env._n_fleet_target,
        wait["passenger_unserved_rate"], completed / max(len(env.timetables), 1))[0]


def residual_basis(intervals):
    x = np.linspace(-1, 1, intervals)
    columns = np.stack([x, x * x], axis=1)
    columns -= columns.mean(axis=0)
    return columns / np.maximum(np.max(np.abs(columns), axis=0), 1e-12)


class NativeResidualPlan(NativeServicePlan):
    def __init__(self, env, action_fn):
        super().__init__(env, "causal_forecast")
        self.action_fn = action_fn
        self.credit = PrefixCostCredit()
        self.decisions = []
        self.previous_actions = {True: np.zeros(2), False: np.zeros(2)}

    def observation(self, nominal, trip, now, preference):
        native = np.asarray(self.env._build_upper_state_v2(trip), dtype=np.float32)
        if native.shape != (16,):
            raise RuntimeError("Unexpected native frequency observation geometry")
        forecast_delta = (preference - np.diff(nominal)) / 120
        forecast_delta = np.pad(forecast_delta, (0, 5 - len(forecast_delta)), mode="edge")
        queues = [sum(len(st.waiting_passengers) for st in self.env.stations
            if bool(st.direction) == direction) / 1000 for direction in (bool(trip.direction), not bool(trip.direction))]
        buses = [bus for bus in self.env.bus_all if bus.on_route]
        load = sum(len(bus.passengers) for bus in buses) / max(sum(bus.capacity for bus in buses), 1)
        other = next((b for b in reversed(self.blocks) if b["direction"] != bool(trip.direction)), None)
        other_plan = np.zeros(5)
        age = 0.0
        if other is not None:
            values = (np.asarray(other["target_headways_s"])[1:] - 360) / 120
            other_plan[:len(values)] = values
            age = (now - other["decision_time_s"]) / 1800
        state = np.concatenate([native, forecast_delta, queues,
            [load, now / self.env.protocol.evaluation_end_time_s,
             (self.env._peak_concurrent - self.env._n_fleet_target) / self.env._n_fleet_target],
            self.previous_actions[bool(trip.direction)], other_plan, [age]]).astype(np.float32)
        if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
            raise RuntimeError("Invalid causal residual-plan observation")
        return state

    def preferred_intervals(self, nominal, trip, now):
        preference, rates = super().preferred_intervals(nominal, trip, now)
        state = self.observation(nominal, trip, now, preference)
        action = np.asarray(self.action_fn(state), dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.000001):
            raise ValueError("Residual policy needs two bounded unit-coordinate actions")
        self.credit.begin(state, action, prefix_service_cost(self.env), now)
        self.decisions.append({"state": state.copy(), "action": action.copy(), "time_s": now})
        self.previous_actions[bool(trip.direction)] = action.copy()
        return preference + RESIDUAL_SCALE_S * residual_basis(len(preference)) @ action, rates

    def finish(self, row):
        # Native step clears passenger lists after caching terminal measurements.
        outcome = self.env.measurement_details
        cost = composite_service_cost(outcome["restricted_wait_horizon_min"], outcome["peak_fleet"],
            outcome["headway_cv"], self.env._n_fleet_target, outcome["passenger_unserved_rate"],
            outcome["trip_completion_rate"])[0]
        if abs(cost - row["service_cost_restricted"]) > 1e-6:
            raise RuntimeError("Training terminal objective differs from reported physical service cost")
        return self.credit.finish(cost, self.env.current_time)
