from pathlib import Path
import unittest

import yaml

from runner_v3 import DiagnosticLog
from scripts.run_freqduet_protocol_v2_matrix import resolved_config
from scripts.validate_freqduet_protocol_v6_configs import (
    V27_MULTISTEP_VALUE_CONFIGS,
    V27_MULTISTEP_VALUE_EXPECTED,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
V13 = "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_hiro"
CONFIRMED_MAIN = "F_freqduet_protocol_v6_confirmed_main_hiro"


class ProtocolV6V27MultiStepValueTest(unittest.TestCase):
    def test_candidates_are_registered_as_exploratory_only(self):
        matrix = [CONFIRMED_MAIN, *V27_MULTISTEP_VALUE_CONFIGS]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate(matrix)
        result = validate(matrix, allow_experimental=True)
        self.assertEqual(
            result["experimental_configs"],
            sorted(V27_MULTISTEP_VALUE_CONFIGS),
        )

    def test_configs_replace_only_the_v13_regularity_target(self):
        for name, (horizon, ucb_beta) in (
                V27_MULTISTEP_VALUE_EXPECTED.items()):
            with self.subTest(config=name):
                raw = yaml.safe_load(
                    (ROOT / "configs_freqduet" / f"{name}.yaml")
                    .read_text())
                if name.endswith("h2_u000_r0010_hiro"):
                    self.assertEqual(raw["_extends"], f"{V13}.yaml")
                config = resolved_config(name)
                lower = config["lower"]
                policy = lower["causal_regularity_policy"]
                value = policy["multi_step_value"]
                self.assertEqual(
                    policy["mode"],
                    "causal_multistep_arrival_delta_regret_dual_v12",
                )
                self.assertEqual(policy["cost_limit"], 0.001)
                self.assertEqual(value["horizon_steps"], horizon)
                self.assertEqual(value["ucb_beta"], ucb_beta)
                self.assertEqual(
                    value["mode"],
                    "discounted_future_arrival_cost_change_v1",
                )
                self.assertEqual(
                    lower["action_bins"],
                    [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
                )
                self.assertEqual(
                    lower.get("discrete_critic", "continuous_action"),
                    "continuous_action",
                )
                self.assertFalse(lower["causal_holding_guard"]["enable"])
                self.assertFalse(lower.get(
                    "follower_forecast_calibration", {}).get(
                        "enable", False))

    def test_diagnostic_schema_exposes_training_and_freeze_evidence(self):
        for field in (
            "lower_multistep_value_enabled",
            "lower_multistep_value_horizon_steps",
            "lower_multistep_value_replay_size",
            "lower_multistep_value_targets_emitted",
            "lower_multistep_value_critic_updates",
            "lower_multistep_value_ready",
            "lower_multistep_value_critic_loss",
            "lower_multistep_value_action_span_mean",
            "lower_multistep_value_positive_regret_mean",
            "lower_multistep_value_frozen",
        ):
            self.assertIn(field, DiagnosticLog.HEADER)


if __name__ == "__main__":
    unittest.main()
