"""Passive causal features and fit-only correction for in-flight trip times."""

import numpy as np

from freq_hrl.domains.transit.native_incumbent_window_plan import NativeIncumbentWindowPlan


class NativeRemainingTimeAudit(NativeIncumbentWindowPlan):
    def __init__(self, env, model):
        super().__init__(env, model, "incumbent_only")
        self.snapshots = []

    def preferred_intervals(self, nominal, trip, now):
        preference, rates = super().preferred_intervals(nominal, trip, now)
        active = []
        for bus in self.env.bus_all:
            if not bus.on_route:
                continue
            total = sum(float(r.distance) for r in bus.effective_route)
            fraction = float(np.clip(1-float(bus.travel_distance)/max(total, 1.), 0., 1.))
            mean = self.trip_duration(bool(bus.direction))
            active.append({"trip_id": int(bus.trip_id), "direction": bool(bus.direction),
                "launch_time_s": float(bus.launch_time), "elapsed_s": float(now-bus.launch_time),
                "remaining_fraction": fraction, "mean_total_s": mean,
                "predicted_remaining_s": max(mean*fraction, 1.),
                "load_fraction": len(bus.passengers)/bus.capacity, "speed_mps": float(bus.current_speed),
                "holding_s": float(bus.holding_time), "dwelling_s": float(bus.dwelling_time)})
        self.snapshots.append({"time_s": float(now), "direction": bool(trip.direction), "active": active})
        return preference, rates


def remaining_features(sample):
    f, age = sample["remaining_fraction"], sample["elapsed_s"]/sample["mean_total_s"]
    return np.asarray([1., f, f*f, age, age*f, sample["predicted_remaining_s"]/2000.,
        sample["load_fraction"], sample["speed_mps"]/20., sample["holding_s"]/60.,
        sample["dwelling_s"]/60., float(sample["direction"])])


def fit_remaining_model(rows):
    samples = [s for row in rows for snapshot in row["snapshots"] for s in snapshot["active"]]
    if not samples:
        raise ValueError("Remaining-time fitting needs passive in-flight observations")
    x = np.stack([remaining_features(s) for s in samples])
    y = np.asarray([(s["actual_remaining_s"]-s["predicted_remaining_s"])/2000. for s in samples])
    penalty = np.eye(x.shape[1])*1e-3
    penalty[0, 0] = 0.
    coefficients = np.linalg.solve(x.T@x+penalty, x.T@y)
    return {"coefficients": coefficients.tolist(), "fit_samples": len(samples),
        "fit_scene_seeds": [r["scene_seed"] for r in rows], "ridge": 1e-3,
        "target": "additive_remaining_seconds_correction_not_policy_value"}


def corrected_remaining(sample, model):
    return max(1., sample["predicted_remaining_s"]+2000.*float(remaining_features(sample)@model["coefficients"]))
