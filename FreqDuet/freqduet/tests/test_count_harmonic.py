import math
import unittest

import numpy as np

from frequency import (
    CausalHarmonicBandState,
    CausalNegativeBinomialHarmonicBandState,
    DemandFrequencyTracker,
)


def count_state(innovation_clip=6.0):
    return CausalNegativeBinomialHarmonicBandState(
        update_interval_s=60.0,
        period_s=3600.0,
        fourier_k=0,
        forgetting_factor=0.9995,
        prior_var=0.01,
        prior_theta=np.array([math.log1p(10.0), 0.0]),
        nb_dispersion=20.0,
        process_var=1e-5,
        innovation_clip=innovation_clip,
    )


class CountHarmonicTest(unittest.TestCase):
    def test_count_filter_remains_finite_and_positive_semidefinite(self):
        state = count_state()
        rng = np.random.default_rng(17)
        for step, value in enumerate(rng.poisson(10.0, size=120)):
            state.update(value, step=step)
        self.assertTrue(np.isfinite(state.theta).all())
        self.assertTrue(np.isfinite(state.cov).all())
        self.assertGreaterEqual(state.low, 0.0)
        self.assertGreaterEqual(np.linalg.eigvalsh(state.cov).min(), -1e-10)
        self.assertGreater(state.observation_variance, 0.0)

    def test_robust_innovation_keeps_single_burst_out_of_trend(self):
        clipped = count_state(innovation_clip=2.0)
        unbounded = count_state(innovation_clip=0.0)
        for step in range(30):
            clipped.update(10.0, step=step)
            unbounded.update(10.0, step=step)
        clipped.update(200.0, step=30)
        unbounded.update(200.0, step=30)
        self.assertLess(clipped.low, unbounded.low)
        self.assertGreater(clipped.high, 0.0)

    def test_tracker_exposes_same_frequency_interface(self):
        tracker = DemandFrequencyTracker(
            method="harmonic_nb",
            update_interval_s=60.0,
            bin_sec=60.0,
            od_features=True,
            upper_mode="low",
            lower_mode="high",
        )
        tracker.update({(1, True): 8}, {(1, 5, True): 8})
        self.assertEqual(tracker.method, "harmonic_nb")
        self.assertEqual(len(tracker.upper_features()), 6)
        self.assertEqual(len(tracker.lower_features(1, True)), 4)

    def test_forecast_horizon_uses_aggregated_bin_interval(self):
        tracker = DemandFrequencyTracker(
            method="harmonic",
            update_interval_s=1.0,
            bin_sec=60.0,
            forecast_horizon_s=1800.0,
        )
        self.assertEqual(tracker.bin_interval_s, 60.0)
        self.assertEqual(tracker.forecast_steps, 30.0)

    def test_legacy_forecast_mode_is_explicit_and_reproducible(self):
        tracker = DemandFrequencyTracker(
            method="harmonic",
            update_interval_s=1.0,
            bin_sec=60.0,
            forecast_horizon_s=1800.0,
            forecast_mode="legacy_seconds_linear",
        )
        self.assertEqual(tracker.forecast_steps, 1800.0)
        self.assertEqual(tracker.forecast_mode, "legacy_seconds_linear")

    def test_unknown_forecast_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forecast_mode"):
            DemandFrequencyTracker(forecast_mode="future_magic")

    def test_harmonic_forecast_evaluates_future_latent_phase(self):
        theta = np.array([math.log1p(10.0), 0.0, 0.2, 0.0])
        state = CausalHarmonicBandState(
            update_interval_s=60.0,
            period_s=3600.0,
            fourier_k=1,
            prior_theta=theta,
        )
        expected = math.expm1(math.log1p(10.0) + 0.2)
        self.assertAlmostEqual(state.forecast(15), expected, places=10)

    def test_harmonic_forecast_tracks_explicit_global_bin_phase(self):
        state = CausalHarmonicBandState(
            update_interval_s=60.0,
            period_s=3600.0,
            fourier_k=1,
            forgetting_factor=0.999,
            prior_var=0.01,
            prior_theta=np.array([math.log1p(10.0), 0.0, 0.1, -0.1]),
        )
        state.update(10.0, step=20)
        expected = state._rate_from_log(
            float(state._features(30) @ state.theta))
        self.assertAlmostEqual(state.forecast(10), expected, places=10)


if __name__ == "__main__":
    unittest.main()
