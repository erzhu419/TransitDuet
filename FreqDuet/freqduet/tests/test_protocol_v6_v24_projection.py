from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from runner_v3 import DiagnosticLog, TransitDuetV2Runner, load_config
from scripts.run_freqduet_protocol_v2_matrix import resolved_config
from scripts.validate_freqduet_protocol_v6_configs import (
    PROJECTION_CONFIGS,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "F_freqduet_protocol_v6_v24_jointproj_fkl_s1_hiro": (
        "forward_kl_v2", 1),
    "F_freqduet_protocol_v6_v24_jointproj_rkl_s4_hiro": (
        "reverse_kl_v1", 4),
    "F_freqduet_protocol_v6_v24_jointproj_fkl_s4_hiro": (
        "forward_kl_v2", 4),
    "F_freqduet_protocol_v6_v24_jointproj_fkl_s8_hiro": (
        "forward_kl_v2", 8),
}


class ProtocolV6V24ProjectionConfigTest(unittest.TestCase):
    def test_v23_and_v24_are_registered_as_experimental_configs(self):
        confirmed_main = "F_freqduet_protocol_v6_confirmed_main_hiro"
        matrix = [confirmed_main, *PROJECTION_CONFIGS]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate(matrix)
        result = validate(matrix, allow_experimental=True)
        self.assertEqual(
            result["experimental_configs"], sorted(PROJECTION_CONFIGS))

    def test_resolved_configs_lock_v24_distillation_factorial(self):
        for config_name, (distillation, steps) in CONFIGS.items():
            with self.subTest(config=config_name):
                config = resolved_config(config_name)
                lower = config["lower"]
                objective = lower["causal_regularity_policy"]
                projection = objective["categorical_projection"]
                self.assertEqual(
                    objective["mode"],
                    "analytic_two_sided_hf_aggregate_gain_projection_v11",
                )
                self.assertEqual(
                    projection["mode"],
                    "joint_kl_soft_policy_distillation_v2",
                )
                self.assertEqual(projection["distillation"], distillation)
                self.assertEqual(projection["distillation_steps"], steps)
                self.assertEqual(projection["regularity_target"], 0.036)
                self.assertEqual(projection["passenger_target"], 0.075)
                self.assertEqual(
                    lower["action_bins"],
                    [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
                )
                self.assertEqual(
                    lower["discrete_critic"], "zero_hold_advantage")
                self.assertFalse(lower["causal_holding_guard"]["enable"])

    def test_runner_exposes_post_update_actor_contract(self):
        config_name = "F_freqduet_protocol_v6_v24_jointproj_fkl_s4_hiro"
        config = load_config(
            ROOT / "configs_freqduet" / f"{config_name}.yaml")
        with TemporaryDirectory() as tmp:
            config.setdefault("logging", {})["logs_dir"] = tmp
            runner = TransitDuetV2Runner(config)

        trainer = runner.lower_trainer
        projection = trainer.regularity_policy_contract[
            "categorical_projection"]
        self.assertEqual(projection["distillation"], "forward_kl_v2")
        self.assertEqual(projection["distillation_steps"], 4)
        self.assertFalse(trainer.regularity_soft_dual_enabled)
        self.assertFalse(trainer.regularity_passenger_soft_dual_enabled)
        self.assertFalse(runner.lower_causal_holding_guard.enabled)
        for field in (
            "lower_regularity_projection_actor_post_forward_kl",
            "lower_regularity_projection_actor_post_regularity_cost",
            "lower_regularity_projection_actor_post_passenger_cost",
            "lower_regularity_projection_actor_post_constraints_met",
            "lower_regularity_projection_actor_reverse_kl_episode_mean",
            "lower_regularity_projection_actor_forward_kl_episode_mean",
            "lower_regularity_projection_actor_post_regularity_cost_episode_max",
            "lower_regularity_projection_target_action_change_abs_mean_s_episode_mean",
            "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean",
        ):
            self.assertIn(field, DiagnosticLog.HEADER)


if __name__ == "__main__":
    unittest.main()
