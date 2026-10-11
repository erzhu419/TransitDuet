"""Training-only cold-start trip times for the existing causal supply selector."""

import numpy as np

from freq_hrl.domains.transit.native_supply_plan import NativeSupplyRollingPlan


class NativeCalibratedSupplyPlan(NativeSupplyRollingPlan):
    def __init__(self, env, condition, duration_prior_s):
        super().__init__(env, condition)
        if (set(duration_prior_s) != {False, True}
                or not all(np.isfinite(v) and v > 0 for v in duration_prior_s.values())):
            raise ValueError("Calibrated supply needs positive fitted times for both directions")
        self.duration_prior_s = dict(duration_prior_s)

    def trip_duration(self, direction):
        # First real completion replaces the prior, exactly as in the old EWMA.
        return (super().trip_duration(direction) if direction in self.duration_mean
                else self.duration_prior_s[direction])

    def preferred_intervals(self, nominal, trip, now):
        preference, rates = super().preferred_intervals(nominal, trip, now)
        self.decisions[-1].update(historical_peak=int(self.env._peak_concurrent),
            prior_directions_in_use=[str(d) for d in (False, True) if d not in self.duration_mean],
            duration_estimates_s={str(d): self.trip_duration(d) for d in (False, True)})
        return preference, rates

    def supply_summary(self):
        return {**super().supply_summary(),
            "initial_duration_prior_s": {str(d): v for d, v in self.duration_prior_s.items()},
            "duration_model": "fit_only_prior_until_first_completion_then_original_EWMA"}
