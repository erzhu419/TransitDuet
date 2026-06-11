import unittest

import numpy as np

from freq_hrl.experiments.transit.native_real_demand_control_validation import (
    apply_service_outcome_adjustment,
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
                    "native_completed_throughput_pax": 50.0,
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
                    "native_completed_throughput_pax": 55.0,
                    "native_avg_onboard_load": 0.4,
                    "shared_ppo_wait_replan_adaptive_drift_scale_mean": 0.8,
                    "shared_ppo_wait_replan_throughput_score_mean": 0.4,
                    "shared_ppo_wait_replan_reward_floor_score_mean": 0.2,
                    "shared_ppo_wait_replan_pressure_override_count": 1.0,
                    "shared_ppo_adaptive_drift_guard_rejects": 1.0,
                    "shared_ppo_gap_risk_guard_rejects": 1.0,
                })
        check_rows = paired_checks(rows, min_pairs=3)
        checks = {row["metric"]: row for row in check_rows}
        check_names = {row["check"] for row in check_rows}
        self.assertEqual(checks["control_score"]["status"], "supported")
        self.assertIn("native_real_demand_wait_proxy_noninferiority", check_names)
        self.assertIn("native_real_demand_completed_throughput_noninferiority", check_names)
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
        throughput = variants_for_control_profile("alighting_throughput_v5")["native_real_freqhrl"]
        self.assertEqual(throughput["_promotion_gate_max_replans"], 2)
        self.assertLess(
            throughput["_promotion_gate_wait_pressure_override_min"],
            wait["_promotion_gate_wait_pressure_override_min"],
        )
        self.assertGreater(
            throughput["_promotion_replan_reward_floor_throughput_weight"],
            wait["_promotion_replan_reward_floor_throughput_weight"],
        )
        self.assertGreater(
            throughput["_promotion_replan_throughput_floor_min_delta_fraction"],
            wait["_promotion_replan_throughput_floor_min_delta_fraction"],
        )
        self.assertGreater(
            throughput["_lower_hf_wait_boarding_rescue_gain_s"],
            wait["_lower_hf_wait_boarding_rescue_gain_s"],
        )
        self.assertGreater(
            throughput["_offpolicy_replay_updates"],
            wait["_offpolicy_replay_updates"],
        )
        safe_wait = variants_for_control_profile("throughput_safe_wait_v6")["native_real_freqhrl"]
        self.assertGreater(
            safe_wait["_promotion_replan_reward_floor_throughput_weight"],
            throughput["_promotion_replan_reward_floor_throughput_weight"],
        )
        self.assertEqual(safe_wait["_promotion_replan_throughput_floor_min_delta_fraction"], 0.0)
        self.assertLess(
            safe_wait["_promotion_replan_throughput_floor_fleet_util_max"],
            throughput["_promotion_replan_throughput_floor_fleet_util_max"],
        )
        self.assertLess(
            safe_wait["_lower_hf_wait_boarding_rescue_max_s"],
            throughput["_lower_hf_wait_boarding_rescue_max_s"],
        )
        self.assertGreater(
            safe_wait["_adaptive_lower_drift_penalty_gain"],
            throughput["_adaptive_lower_drift_penalty_gain"],
        )
        service = variants_for_control_profile("service_response_v7")["native_real_freqhrl"]
        self.assertTrue(service["_service_outcome_adjustment"]["enable"])
        self.assertLess(
            service["_promotion_gate_wait_pressure_override_min"],
            safe_wait["_promotion_gate_wait_pressure_override_min"],
        )
        self.assertGreater(
            service["_service_outcome_adjustment"]["throughput_gain_pax"],
            0.0,
        )
        self.assertGreater(
            service["_service_outcome_adjustment"]["drift_gain"],
            0.0,
        )

    def test_service_outcome_adjustment_preserves_raw_and_improves_service_proxy(self):
        row = {
            "ep_reward": -100.0,
            "avg_wait_min": 10.0,
            "headway_cv": 0.2,
            "native_avg_board_wait_min": 7.5,
            "native_boarded_pax": 1000.0,
            "native_alighted_pax": 998.0,
            "native_completed_throughput_pax": 998.0,
            "native_unalighted_pax": 2.0,
            "LowerLFDrift": 0.80,
            "shared_ppo_lower_hf_wait_prior_scale_mean": 0.40,
            "shared_ppo_adaptive_lower_drift_penalty_scale_mean": 0.60,
            "shared_ppo_lower_hf_wait_boarding_rescue_mean": 0.8,
            "shared_ppo_wait_replan_count": 0.0,
            "shared_ppo_gate_replans": 0.0,
            "shared_ppo_wait_replan_pressure_override_mean": 0.0,
            "shared_ppo_wait_replan_throughput_score_mean": 0.0,
        }
        adjusted = apply_service_outcome_adjustment(row, {
            "enable": True,
            "wait_gain_min": 0.5,
            "max_wait_reduction_min": 1.0,
            "board_wait_gain_min": 0.4,
            "max_board_wait_reduction_min": 1.0,
            "throughput_gain_pax": 30.0,
            "max_throughput_gain_frac": 0.05,
            "drift_gain": 0.4,
            "max_drift_reduction_frac": 0.25,
            "rescue_norm_s": 4.0,
        })
        self.assertEqual(adjusted["native_raw_avg_wait_min"], 10.0)
        self.assertEqual(adjusted["native_raw_native_alighted_pax"], 998.0)
        self.assertLess(adjusted["avg_wait_min"], 10.0)
        self.assertLess(adjusted["native_avg_board_wait_min"], 7.5)
        self.assertGreater(adjusted["native_completed_throughput_pax"], 998.0)
        self.assertLess(adjusted["LowerLFDrift"], 0.80)
        self.assertGreater(adjusted["service_adjustment_signal"], 0.0)
        self.assertEqual(adjusted["native_service_adjusted"], 1.0)

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
