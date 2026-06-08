import unittest

import numpy as np

from freq_hrl.experiments.transit.native_real_demand_control_validation import (
    build_native_real_demand_profile,
    paired_checks,
)


class NativeRealDemandControlValidationTest(unittest.TestCase):
    def test_profile_maps_real_series_to_native_multipliers(self):
        profile = build_native_real_demand_profile(
            {
                "a": np.arange(1, 30, dtype=float),
                "b": np.arange(30, 1, -1, dtype=float),
            },
            source="afc",
            seed=3,
            bins_per_hour=1,
        )
        self.assertTrue(profile["hour_multipliers"])
        self.assertTrue(profile["station_multipliers"])
        self.assertIn("boundary", profile)

    def test_paired_checks_capture_native_real_direction(self):
        rows = []
        for source in ("afc", "apc"):
            for seed in (1, 2, 3):
                rows.append({
                    "source": source,
                    "seed": seed,
                    "variant": "native_real_interval",
                    "control_score": -10.0,
                    "ep_reward": -100.0,
                    "avg_wait_min": 5.0,
                    "native_avg_board_wait_min": 4.0,
                    "native_alighted_pax": 50.0,
                    "native_avg_onboard_load": 0.5,
                    "shared_ppo_wait_replan_adaptive_drift_scale_mean": 1.0,
                    "shared_ppo_wait_replan_throughput_score_mean": 0.0,
                    "shared_ppo_wait_replan_reward_floor_score_mean": 0.0,
                    "shared_ppo_wait_replan_pressure_override_count": 0.0,
                })
                rows.append({
                    "source": source,
                    "seed": seed,
                    "variant": "native_real_freqhrl",
                    "control_score": -8.0,
                    "ep_reward": -90.0,
                    "avg_wait_min": 4.0,
                    "native_avg_board_wait_min": 3.0,
                    "native_alighted_pax": 55.0,
                    "native_avg_onboard_load": 0.4,
                    "shared_ppo_wait_replan_adaptive_drift_scale_mean": 0.8,
                    "shared_ppo_wait_replan_throughput_score_mean": 0.4,
                    "shared_ppo_wait_replan_reward_floor_score_mean": 0.2,
                    "shared_ppo_wait_replan_pressure_override_count": 1.0,
                })
        checks = {row["metric"]: row for row in paired_checks(rows, min_pairs=3)}
        self.assertEqual(checks["control_score"]["status"], "supported")
        self.assertLess(checks["avg_wait_min"]["delta_mean"], 0.0)
        self.assertLess(
            checks["shared_ppo_wait_replan_adaptive_drift_scale_mean"]["delta_mean"],
            0.0,
        )
        self.assertGreater(
            checks["shared_ppo_wait_replan_reward_floor_score_mean"]["delta_mean"],
            0.0,
        )
        self.assertGreater(
            checks["shared_ppo_wait_replan_pressure_override_count"]["delta_mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
