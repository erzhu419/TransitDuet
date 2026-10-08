"""Matched native observation controls with a common harmonic/OD estimator."""

from collections import defaultdict, deque

import numpy as np

from native_freqduet.frequency.demand_frequency import DemandFrequencyTracker


METHODS = ("correct", "raw_history_common", "swapped_common")


class NativeRoutingTracker(DemandFrequencyTracker):
    """Change actor inputs, retaining the native estimator and diagnostic state.

    Correct routing is numerically unchanged. Both controls retain the upper
    forecast, HF-energy and OD slots. Swapped routing exchanges the upper's
    dynamic value/slope and the lower's four-feature band block, not the common
    forecast background. The history control carries two global bins as
    current/difference and four trailing local bins.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing = "correct"
        self._global_raw_bins = deque(maxlen=2)
        self._local_raw_bins = defaultdict(lambda: deque(maxlen=4))

    @classmethod
    def from_config(cls, cfg, update_interval_s=1.0):
        routing = cfg.get("routing", "correct")
        if routing not in METHODS:
            raise ValueError(f"Unknown native routing: {routing}")
        tracker = super().from_config(cfg, update_interval_s)
        tracker.routing = routing
        return tracker

    def reset(self):
        super().reset()
        self._global_raw_bins.clear()
        self._local_raw_bins.clear()

    def _apply_update(self, arrivals_by_station, arrivals_by_od):
        super()._apply_update(arrivals_by_station, arrivals_by_od)
        scale = 60.0 / self.bin_interval_s
        self._global_raw_bins.append(sum(arrivals_by_station.values()) * scale)
        for key in self.local_states:
            self._local_raw_bins[key].append(arrivals_by_station.get(key, 0.0) * scale)

    def upper_features(self, mode=None):
        features = super().upper_features(mode)
        if self.routing == "swapped_common":
            features[:2] = super().upper_features("high")[:2]
        elif self.routing == "raw_history_common":
            history = list(self._global_raw_bins)
            current = history[-1] if history else 0.0
            delta = current - history[-2] if len(history) == 2 else 0.0
            features[:2] = [current / self.global_demand_norm, delta / self.slope_norm]
        return features

    def lower_features(self, station_id, direction, mode=None):
        if self.routing == "swapped_common":
            return super().lower_features(station_id, direction, "low")
        if self.routing == "raw_history_common":
            values = list(self._local_raw_bins.get((int(station_id), bool(direction)), ()))
            return np.asarray([0.0] * (4 - len(values)) + values, dtype=np.float32) / self.local_demand_norm
        return super().lower_features(station_id, direction, mode)
