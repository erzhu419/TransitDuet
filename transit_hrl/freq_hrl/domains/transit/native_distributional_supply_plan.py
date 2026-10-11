"""Expected native peak cost under fit-only trip-duration variability."""

import numpy as np

from freq_hrl.domains.transit.native_calibrated_supply_plan import NativeCalibratedSupplyPlan

MEMBERS = 64
FLEET_MODEL = "empirical_centered_duration_residuals_expected_peak_penalty"


def ensemble_peaks(starts, durations, historical_peak=0):
    starts, durations = np.asarray(starts, dtype=float), np.asarray(durations, dtype=float)
    if durations.ndim != 2 or starts.shape != (durations.shape[1],) or np.any(durations <= 0):
        raise ValueError("Ensemble occupancy needs positive member-by-trip durations")
    if not len(starts):
        return np.full(len(durations), historical_peak, dtype=int)
    starts = np.broadcast_to(starts, durations.shape)
    times = np.concatenate((starts, starts + durations), axis=1)
    changes = np.broadcast_to(np.r_[np.ones(starts.shape[1], dtype=int),
        -np.ones(starts.shape[1], dtype=int)], times.shape)
    order = np.lexsort((changes, times), axis=1)
    counts = np.take_along_axis(changes, order, axis=1).cumsum(axis=1)
    return np.maximum(historical_peak, counts.max(axis=1))


def residual_draws(model, trips):
    rng = np.random.default_rng(model["ensemble_seed"])
    # Stable trip identities share the same draws across candidates and decisions.
    return {tid: rng.choice(model["centered_residuals_s"][str(direction)], size=model["members"])
        for tid, direction in sorted(trips)}


def validate_model(model):
    if model["members"] != MEMBERS or model["ensemble_seed"] != 166000 + model["root"]:
        raise ValueError("Changed registered ensemble size or seed")
    for direction in ("False", "True"):
        mean, residuals = model["duration_prior_s"][direction], np.asarray(model["centered_residuals_s"][direction])
        if not (np.isfinite(mean) and mean > 0 and residuals.ndim == 1 and len(residuals) > 0
                and np.all(np.isfinite(residuals)) and abs(float(residuals.mean())) < 1e-9
                and np.all(mean + residuals > 0)):
            raise ValueError("Expected centered positive fit-only duration observations")


class NativeDistributionalSupplyPlan(NativeCalibratedSupplyPlan):
    def __init__(self, env, model):
        validate_model(model)
        super().__init__(env, "supply_aware", {d: model["duration_prior_s"][str(d)] for d in (False, True)})
        self.model = model
        self.residuals = residual_draws(model, [(int(t.launch_turn), bool(t.direction)) for t in env.timetables])

    def sampled_intervals(self, planned_by_id, now):
        starts, durations = [], []
        for bus in self.env.bus_all:
            if not bus.on_route:
                continue
            total = sum(float(r.distance) for r in bus.effective_route)
            fraction = np.clip(1 - float(bus.travel_distance) / max(total, 1.), 0., 1.)
            starts.append(now)
            durations.append(np.maximum((self.trip_duration(bool(bus.direction))
                + self.residuals[int(bus.trip_id)]) * fraction, 1.))
        for trip in self.env.timetables:
            if trip.launched:
                continue
            starts.append(max(now, planned_by_id.get(int(trip.launch_turn),
                float(getattr(trip, "_freqduet_scheduled_launch", trip.launch_time)))))
            durations.append(np.maximum(self.trip_duration(bool(trip.direction))
                + self.residuals[int(trip.launch_turn)], 1.))
        matrix = np.stack(durations, axis=1) if durations else np.empty((MEMBERS, 0))
        return np.asarray(starts), matrix

    def candidate_fleet_cost(self, peak, planned_by_id, now):
        starts, matrix = self.sampled_intervals(planned_by_id, now)
        peaks = ensemble_peaks(starts, matrix, self.env._peak_concurrent)
        values, counts = np.unique(peaks, return_counts=True)
        cost = np.mean(np.maximum(peaks - self.env._n_fleet_target, 0)**2 / self.env._n_fleet_target)
        return float(cost), {"ensemble_peak_counts": {str(int(v)): int(c) for v, c in zip(values, counts)}}

    def supply_summary(self):
        return {**super().supply_summary(), "fleet_model": FLEET_MODEL,
            "ensemble_members": MEMBERS, "ensemble_seed": self.model["ensemble_seed"],
            "fit_scene_seeds": self.model["fit_scene_seeds"]}
