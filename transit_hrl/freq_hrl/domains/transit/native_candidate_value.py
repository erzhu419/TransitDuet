"""Single native candidate interventions with exact causal-prefix checks."""

import copy
import random

import numpy as np
import torch

from freq_hrl.domains.transit.native_incumbent_window_plan import NativeIncumbentWindowPlan
from freq_hrl.domains.transit.native_residual_plan import prefix_service_cost


def numpy_rng_state(state):
    return [state[0], state[1].tolist(), *state[2:]]


def passenger_state(passenger):
    return [passenger.appear_time, passenger.boarding_time, passenger.arrive_time,
        int(passenger.appear_station.station_id), int(passenger.destination_station.station_id),
        bool(passenger.boarded), bool(passenger.arrived)]


def causal_snapshot(env):
    return {"cost_before": prefix_service_cost(env),
        "buses": [{**{k: getattr(bus, k) for k in ("bus_id", "trip_id", "launch_time", "direction",
            "on_route", "current_speed", "travel_distance", "holding_time", "dwelling_time",
            "forward_headway", "backward_headway")}, "state": bus.state.name,
            "stations": [int(bus.last_station.station_id), int(bus.next_station.station_id)],
            "passengers": [passenger_state(p) for p in bus.passengers]} for bus in env.bus_all],
        "queues": [[int(s.station_id), bool(s.direction), [passenger_state(p) for p in s.waiting_passengers]]
            for s in env.stations],
        "routes": [[int(r.route_id), float(r.speed_limit)] for r in env.routes],
        "timetables": [[int(t.launch_turn), bool(t.direction), bool(t.launched), float(t.launch_time),
            float(getattr(t, "_freqduet_scheduled_launch", t.launch_time)), float(t.target_headway)]
            for t in env.timetables],
        "completed_trips": copy.deepcopy(env.completed_trip_observations),
        "rng": {"python": random.getstate(), "numpy": numpy_rng_state(np.random.get_state()),
            "torch": torch.get_rng_state().tolist(), "scenario_seed": int(env.scenario_tape.seed),
            "scenario_streams": [(k, numpy_rng_state(rng.get_state()))
                for k, rng in env.scenario_tape._streams.items()]}}


class NativeCandidateValuePlan(NativeIncumbentWindowPlan):
    def __init__(self, env, model, probe_index, *, forced_candidate=None, reference=None):
        super().__init__(env, model, "rolling_plus_incumbent")
        self.probe_index, self.forced_candidate, self.reference = probe_index, forced_candidate, reference
        self.prefix, self.snapshot, self.probe = [], None, None

    def __call__(self, state, trip):
        self._incoming_state = np.asarray(state).tolist()
        return super().__call__(state, trip)

    def preferred_intervals(self, nominal, trip, now):
        self._incoming = {"time_s": float(now), "direction": bool(trip.direction),
            "trip_id": int(trip.launch_turn), "upper_state": self._incoming_state,
            "native_state_v2": self.env._build_upper_state_v2(trip).tolist(),
            "nominal_s": nominal.tolist()}
        return super().preferred_intervals(nominal, trip, now)

    def select_candidate(self, candidates):
        index, greedy = len(self.decisions), super().select_candidate(candidates)
        if index <= self.probe_index:
            prefix = {**self._incoming, "candidates": copy.deepcopy(candidates), "greedy": greedy,
                "historical_peak": int(self.env._peak_concurrent), "completed": len(self.seen_completions)}
            if self.reference is not None and prefix != self.reference.prefix[index]:
                raise RuntimeError("Native candidate intervention changed its causal action prefix")
            self.prefix.append(prefix)
        if index == self.probe_index:
            self.snapshot = causal_snapshot(self.env)
            if self.reference is not None and self.snapshot != self.reference.snapshot:
                raise RuntimeError("Native candidate intervention changed physical state, cost or RNG")
            selected = greedy if self.forced_candidate is None else self.forced_candidate
            self.probe = {**self._incoming, "index": index, "greedy": greedy, "selected": selected,
                "candidates": copy.deepcopy(candidates), "cost_before": self.snapshot["cost_before"],
                "prefix_decisions_qualified": index+1, "physical_state_matched": self.reference is not None,
                "rng_matched": self.reference is not None}
            return selected
        return greedy
