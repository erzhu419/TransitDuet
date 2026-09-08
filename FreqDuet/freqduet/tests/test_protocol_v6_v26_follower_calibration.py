from pathlib import Path
import unittest

import yaml

from lower.causal_follower_eta import HistoricalFollowerTargetCalibrator
from runner_v3 import DiagnosticLog
from scripts.run_freqduet_protocol_v2_matrix import resolved_config
from scripts.validate_freqduet_protocol_v6_configs import (
    V26_FOLLOWER_CALIBRATION_CONFIGS,
    V26_FOLLOWER_CALIBRATION_EXPECTED,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
V13 = "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_hiro"
CONFIRMED_MAIN = "F_freqduet_protocol_v6_confirmed_main_hiro"


class ProtocolV6V26FollowerCalibrationConfigTest(unittest.TestCase):
    def test_candidates_are_registered_only_as_exploratory(self):
        matrix = [CONFIRMED_MAIN, *V26_FOLLOWER_CALIBRATION_CONFIGS]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate(matrix)
        result = validate(matrix, allow_experimental=True)
        self.assertEqual(
            result["experimental_configs"],
            sorted(V26_FOLLOWER_CALIBRATION_CONFIGS),
        )

    def test_candidates_lock_the_registered_half_gap_contract(self):
        for name, expected in V26_FOLLOWER_CALIBRATION_EXPECTED.items():
            with self.subTest(config=name):
                raw = yaml.safe_load(
                    (ROOT / "configs_freqduet" / f"{name}.yaml").read_text())
                self.assertEqual(raw["_extends"], f"{V13}.yaml")
                config = resolved_config(name)
                calibration = config["lower"][
                    "follower_forecast_calibration"]
                mode, alpha, ridge, cap = expected
                self.assertEqual(calibration["mode"], mode)
                self.assertEqual(calibration["history_alpha"], alpha)
                self.assertEqual(calibration["ridge"], ridge)
                self.assertEqual(calibration["adjustment_cap_s"], cap)
                self.assertEqual(calibration["min_history_episodes"], 5)
                self.assertEqual(calibration["min_samples_per_episode"], 128)
                self.assertEqual(calibration["residual_clip_s"], 30.0)
                self.assertEqual(calibration["time_period_s"], 50400.0)
                self.assertEqual(
                    config["lower"]["causal_regularity_policy"]["mode"],
                    "analytic_two_sided_zero_hold_regret_dual_v2",
                )
                self.assertEqual(
                    config["lower"].get(
                        "discrete_critic", "continuous_action"),
                    "continuous_action",
                )
                self.assertFalse(
                    config["lower"]["causal_holding_guard"]["enable"])

    def test_resolved_calibrator_contract_is_checkpoint_complete(self):
        for name in V26_FOLLOWER_CALIBRATION_CONFIGS:
            with self.subTest(config=name):
                config = resolved_config(name)
                calibrator = HistoricalFollowerTargetCalibrator.from_config(
                    config["lower"]["follower_forecast_calibration"])
                state = calibrator.state_dict()
                self.assertEqual(
                    state["contract"]["update_source"],
                    "completed_learned_training_days_v1",
                )
                self.assertEqual(state["contract"]["time_period_s"], 50400.0)
                self.assertEqual(
                    len(calibrator.feature_names),
                    len(set(calibrator.feature_names)),
                )

    def test_diagnostics_distinguish_used_and_post_update_generations(self):
        for field in (
            "follower_forecast_calibration_active_mean",
            "follower_forecast_calibration_history_episodes_mean",
            "follower_target_calibration_post_update_active",
            "follower_target_calibration_post_update_history_episodes",
            "follower_target_calibration_episode_updated",
            "follower_target_calibration_update_source",
            "follower_forecast_calibration_requested_adjustment_abs_max_s",
        ):
            self.assertIn(field, DiagnosticLog.HEADER)


if __name__ == "__main__":
    unittest.main()
