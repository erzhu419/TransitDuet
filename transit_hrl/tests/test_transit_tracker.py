import unittest

import numpy as np

from freq_hrl.domains.transit import TransitFrequencyTracker


class TransitTrackerTest(unittest.TestCase):
    def test_tracker_exposes_compatible_features(self):
        tracker = TransitFrequencyTracker(
            update_interval_s=1,
            bin_sec=2,
            method="ema",
            low_period_s=10,
            fast_period_s=2,
            global_demand_norm=10,
            local_demand_norm=5,
            promotion_enable=True,
        )
        tracker.update({(1, True): 2.0})
        self.assertEqual(tracker.summary()["freq_updates"], 0)
        tracker.update({(1, True): 4.0})
        summary = tracker.summary()
        self.assertEqual(summary["freq_updates"], 1)
        self.assertEqual(tracker.upper_features().shape[0], tracker.upper_feature_dim)
        self.assertEqual(tracker.lower_features(1, True).shape[0], tracker.lower_feature_dim)
        self.assertGreaterEqual(summary["freq_low_demand"], 0.0)

    def test_seen_local_entities_receive_zero_bins(self):
        tracker = TransitFrequencyTracker(
            update_interval_s=1,
            bin_sec=1,
            method="ema",
            low_period_s=8,
            fast_period_s=2,
            local_demand_norm=10,
        )
        tracker.update({(2, False): 10.0})
        first = tracker.local_low_value(2, False)
        tracker.update({})
        second = tracker.local_low_value(2, False)
        self.assertLess(second, first)

    def test_fourier_tracker_is_causal(self):
        values = [0.0, 1.0, 2.0, 3.0, 100.0]

        def run(prefix):
            tracker = TransitFrequencyTracker(
                update_interval_s=1,
                bin_sec=1,
                method="fourier",
                harmonic_period_s=12,
                fourier_k=1,
            )
            snapshots = []
            for value in prefix:
                tracker.update({(1, True): value})
                snapshots.append(tracker.upper_features().copy())
            return snapshots

        before_future = run(values[:4])[-1]
        with_future = run(values)[3]
        np.testing.assert_allclose(before_future, with_future)

    def test_adaptive_wavelet_tracker_runs(self):
        tracker = TransitFrequencyTracker(
            update_interval_s=1,
            bin_sec=1,
            method="adaptive_wavelet",
            global_demand_norm=10,
            local_demand_norm=5,
        )
        tracker.update({(1, True): 3.0})
        tracker.update({(1, True): 5.0})
        summary = tracker.summary()
        self.assertEqual(summary["freq_method"], "adaptive_wavelet")
        self.assertEqual(tracker.upper_features().shape[0], tracker.upper_feature_dim)
        self.assertEqual(tracker.lower_features(1, True).shape[0], tracker.lower_feature_dim)

    def test_raw_history_method_is_supported_for_copied_configs(self):
        tracker = TransitFrequencyTracker(
            update_interval_s=1,
            bin_sec=1,
            method="raw_history",
            global_demand_norm=10,
            local_demand_norm=5,
        )
        tracker.update({(1, True): 3.0})
        self.assertEqual(tracker.summary()["freq_method"], "raw_history")
        self.assertEqual(tracker.upper_features().shape[0], tracker.upper_feature_dim)
        self.assertEqual(tracker.lower_features(1, True).shape[0], tracker.lower_feature_dim)


if __name__ == "__main__":
    unittest.main()
