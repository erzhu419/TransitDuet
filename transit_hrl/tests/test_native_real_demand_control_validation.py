import unittest

import numpy as np

from freq_hrl.experiments.transit.native_real_demand_control_validation import (
    build_native_real_demand_profile,
    control_score,
    paired_checks,
    variants_for_control_profile,
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
                    "shared_ppo_adaptive_drift_guard_rejects": 0.0,
                    "shared_ppo_gap_risk_guard_rejects": 0.0,
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
                    "shared_ppo_adaptive_drift_guard_rejects": 1.0,
                    "shared_ppo_gap_risk_guard_rejects": 1.0,
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
        self.assertGreater(
            checks["shared_ppo_adaptive_drift_guard_rejects"]["delta_mean"],
            0.0,
        )

    def test_alighting_safe_profile_tightens_replan_acceptance(self):
        variants = variants_for_control_profile("alighting_safe_v1")
        freq = variants["native_real_freqhrl"]
        self.assertEqual(freq["_promotion_gate_max_replans"], 1)
        self.assertLess(freq["_promotion_replan_max_shift_s"], 2.5)
        self.assertGreater(freq["_promotion_replan_throughput_guard_min_score"], 0.10)
        self.assertEqual(freq["_promotion_replan_throughput_floor_min_delta_fraction"], 0.0)
        damped = variants_for_control_profile("alighting_safe_v2")["native_real_freqhrl"]
        self.assertLess(damped["_lower_hf_wait_action_gain_s"], freq["_lower_hf_wait_action_gain_s"])
        self.assertGreater(damped["_lower_hf_wait_load_damping_weight"], 0.0)
        self.assertLess(damped["_lower_hf_wait_max_scale"], freq["_lower_hf_wait_max_scale"])
        rescue = variants_for_control_profile("alighting_rescue_v3")["native_real_freqhrl"]
        self.assertGreater(rescue["_lower_hf_wait_boarding_rescue_gain_s"], 0.0)
        self.assertGreater(rescue["_lower_hf_wait_boarding_rescue_max_s"], 0.0)
        self.assertLess(rescue["_lower_hf_wait_action_gain_s"], damped["_lower_hf_wait_action_gain_s"])
        wait = variants_for_control_profile("alighting_wait_v4")["native_real_freqhrl"]
        self.assertEqual(wait["_promotion_gate_max_replans"], 2)
        self.assertLess(wait["_promotion_gate_wait_pressure_override_min"], damped["_promotion_gate_wait_pressure_override_min"])
        self.assertGreater(wait["_promotion_replan_throughput_floor_min_delta_fraction"], 0.0)
        self.assertEqual(wait["_promotion_replan_final_delta_abs_min_s"], 0.03)
        self.assertLess(wait["_lower_hf_wait_action_gain_s"], rescue["_lower_hf_wait_action_gain_s"])
        self.assertGreater(wait["_lower_hf_wait_boarding_rescue_gain_s"], 0.0)
        self.assertGreater(wait["_adaptive_lower_drift_penalty_gain"], rescue["_adaptive_lower_drift_penalty_gain"])

    def test_control_score_penalizes_completed_throughput_loss(self):
        base = {
            "ep_reward": -100.0,
            "avg_wait_min": 5.0,
            "headway_cv": 0.2,
            "native_avg_board_wait_min": 4.0,
            "native_boarded_pax": 100.0,
            "native_alighted_pax": 100.0,
        }
        better_reward_less_throughput = {
            **base,
            "ep_reward": -20.0,
            "native_boarded_pax": 95.0,
            "native_alighted_pax": 95.0,
        }
        self.assertLess(
            control_score(better_reward_less_throughput),
            control_score(base),
        )


if __name__ == "__main__":
    unittest.main()
