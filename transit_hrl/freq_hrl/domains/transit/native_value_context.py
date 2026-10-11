"""Passive occupancy context for the six already-labelled native decisions."""

import copy

from freq_hrl.domains.transit.native_candidate_value import NativeCandidateValuePlan
from freq_hrl.domains.transit.native_remaining_time import active_sample

PROBES = (8, 9, 22, 23, 36, 37)


class NativeValueContextAudit(NativeCandidateValuePlan):
    def __init__(self, env, model, probe_index, *, forced_candidate=None, reference=None):
        super().__init__(env, model, probe_index, forced_candidate=forced_candidate, reference=reference)
        self.contexts = []

    def select_candidate(self, candidates):
        index = len(self.decisions)
        selected = super().select_candidate(candidates)
        if index in PROBES:
            self.contexts.append({**copy.deepcopy(self._incoming), "index": index,
                "greedy": selected, "candidates": copy.deepcopy(candidates),
                "historical_peak": int(self.env._peak_concurrent),
                "active_buses": [{**active_sample(bus, self.env.current_time, self.trip_duration(bool(bus.direction))),
                    "onboard_count": len(bus.passengers), "capacity": int(bus.capacity)}
                    for bus in self.env.bus_all if bus.on_route],
                "queue_by_station": [[int(s.station_id), bool(s.direction), len(s.waiting_passengers)]
                    for s in self.env.stations],
                "route_speed_mps": [float(r.speed_limit) for r in self.env.routes]})
        return selected
