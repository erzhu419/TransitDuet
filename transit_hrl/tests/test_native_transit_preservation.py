import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.encoders import count_harmonic as shared
from native_freqduet.frequency import demand_frequency as adapter
from native_freqduet.frequency import intensity_estimator as reference
from scripts.run_native_transit_preservation_stage145 import compare


class NativeTransitPreservationTest(unittest.TestCase):
    def tracker_pair(self, **extra):
        rates = 5.0 + np.sin(np.arange(24) / 3.0)
        prior = shared.fit_harmonic_prior(rates, 60, 1440, 2)
        cfg = {
            "method": "harmonic", "bin_sec": 60, "fourier_K": 2,
            "harmonic_period_s": 1440, "harmonic_forgetting": 0.9995,
            "harmonic_prior_var": 0.01, "forecast_horizon_s": 180,
            "od_features": True, "harmonic_prior": {
                "global": prior, "local": {(1, True): prior},
                "od": {(1, 2, True): prior},
            },
            **extra,
        }
        candidate = adapter.DemandFrequencyTracker.from_config(cfg, 30)
        with patch.object(adapter, "CausalHarmonicBandState", reference.CausalHarmonicBandState):
            original = adapter.DemandFrequencyTracker.from_config(cfg, 30)
        return candidate, original

    def test_production_uses_shared_count_core(self):
        tracker, _ = self.tracker_pair()
        self.assertIs(type(tracker.global_state), shared.CausalHarmonicBandState)
        self.assertIsNot(shared.CausalHarmonicBandState, reference.CausalHarmonicBandState)
        self.assertGreater(tracker.global_state.low, 0)
        self.assertEqual(tracker.upper_feature_dim, 6)
        self.assertEqual(tracker.lower_feature_dim, 4)

    def test_historical_fit_matches_reference(self):
        rates = [0, 2, 5, 3, 0, 11, 8, 4]
        np.testing.assert_array_equal(
            shared.fit_harmonic_prior(rates, 60, 480, 2),
            reference.fit_harmonic_prior(rates, 60, 480, 2),
        )

    def test_native_features_and_forecasts_match_over_sparse_bins(self):
        candidate, original = self.tracker_pair()
        for step in range(20):
            stations = {(1, True): float(step % 4)}
            od = {(1, 2, True): float(step % 4)}
            if step % 3 == 0:
                stations[(2, False)] = 7.0
                od[(2, 1, False)] = 7.0
            candidate.update(stations, od)
            with patch.object(adapter, "CausalHarmonicBandState", reference.CausalHarmonicBandState):
                original.update(stations, od)
            np.testing.assert_array_equal(candidate.upper_features(), original.upper_features())
            for key in stations:
                np.testing.assert_array_equal(candidate.lower_features(*key), original.lower_features(*key))
            for a, b in [(candidate.global_state, original.global_state), *[
                (candidate.local_states[k], original.local_states[k]) for k in candidate.local_states
            ], *[(candidate.od_states[k], original.od_states[k]) for k in candidate.od_states]]:
                np.testing.assert_array_equal(a.theta, b.theta)
                self.assertEqual(a.forecast(3), b.forecast(3))

    def test_explicit_clock_reset_and_promotion_match(self):
        prior = shared.fit_harmonic_prior([1, 2, 3, 2], 60, 240, 1)
        kwargs = dict(update_interval_s=60, period_s=240, fourier_k=1, prior_theta=prior,
                      prior_var=0.01, residual_alpha=0.3, middle_alpha=0.1)
        a = shared.CausalHarmonicBandState(**kwargs)
        b = reference.CausalHarmonicBandState(**kwargs)
        for step, value in [(0, 2), (5, 10), (6, 7), (14, 0)]:
            a.update(value, step=step)
            b.update(value, step=step)
            self.assertEqual(a.forecast(2), b.forecast(2))
            self.assertEqual(a.promote_residual(0.5, 0.1), b.promote_residual(0.5, 0.1))
            np.testing.assert_array_equal(a.theta, b.theta)
        a.reset()
        b.reset()
        self.assertEqual(a.low, b.low)
        np.testing.assert_array_equal(a.cov, b.cov)

    def test_online_prefix_is_independent_of_future_burst(self):
        def run(values):
            state = shared.CausalHarmonicBandState(60, fourier_k=1)
            out = []
            for value in values:
                state.update(value)
                out.append((state.low, state.high, state.forecast(2)))
            return out
        prefix = run([0, 2, 4, 1])
        self.assertEqual(prefix, run([0, 2, 4, 1, 1000])[:4])

    def test_native_negative_binomial_core_is_also_preserved(self):
        a = shared.CausalNegativeBinomialHarmonicBandState(60, period_s=1440, fourier_k=2)
        b = reference.CausalNegativeBinomialHarmonicBandState(60, period_s=1440, fourier_k=2)
        for step, count in [(0, 0), (1, 3), (4, 25), (5, 2), (10, 1)]:
            a.update(count, step=step)
            b.update(count, step=step)
            np.testing.assert_array_equal(a.theta, b.theta)
            np.testing.assert_array_equal(a.cov, b.cov)
            self.assertEqual(a.forecast(3), b.forecast(3))
            self.assertEqual(a.high, b.high)

    def test_native_raw_history_retains_bins(self):
        tracker = adapter.DemandFrequencyTracker.from_config({
            "method": "raw_history", "bin_sec": 60,
            "upper_history_bins": 4, "lower_history_bins": 4,
            "global_demand_norm": 1, "local_demand_norm": 1,
        }, 60)
        for count in (2, 5, 1):
            tracker.update({(1, True): count})
        np.testing.assert_array_equal(tracker.upper_features(), [0, 2, 5, 1])
        np.testing.assert_array_equal(tracker.lower_features(1, True), [0, 2, 5, 1])

    def test_gate_rejects_changed_action_and_noop_training(self):
        a = {
            "rows": [{"reward": 1}], "updates": {"upper": 1, "lower": 1},
            "actor_change_max_abs": {"upper": 0.1, "lower": 0.1},
            "native_steps": 10, "state_dims": {"upper": 16, "lower": 9},
            "endpoints": [], "estimator": "frequency.intensity_estimator",
        }
        b = {**a, "estimator": "freq_hrl.encoders.count_harmonic"}
        arrays = {"actions/0/lower": np.array([1.0, 2.0])}
        self.assertTrue(compare(a, b, arrays, arrays)["passed"])
        bad = compare(a, b, arrays, {"actions/0/lower": np.array([1.0, 3.0])})
        self.assertFalse(bad["passed"])
        self.assertIn("actions/0/lower", bad["differences"])
        noop = {**b, "updates": {"upper": 0, "lower": 0}}
        self.assertIn("no_upper_learning", compare(a, noop, arrays, arrays)["differences"])
        self.assertFalse(compare(a, {**b, "rows": [{"reward": 2}]}, arrays, arrays)["passed"])


if __name__ == "__main__":
    unittest.main()
