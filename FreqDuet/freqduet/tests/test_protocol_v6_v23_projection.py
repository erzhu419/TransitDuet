from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from runner_v3 import DiagnosticLog, TransitDuetV2Runner, load_config
from scripts.run_freqduet_protocol_v2_matrix import resolved_config


CONFIG = "F_freqduet_protocol_v6_v23_jointproj_r036_p075_hiro"
ROOT = Path(__file__).resolve().parents[1]


class ProtocolV6V23ProjectionConfigTest(unittest.TestCase):
    def test_resolved_config_locks_registered_v23_contract(self):
        config = resolved_config(CONFIG)
        lower = config["lower"]
        objective = lower["causal_regularity_policy"]
        projection = objective["categorical_projection"]

        self.assertEqual(
            objective["mode"],
            "analytic_two_sided_hf_aggregate_gain_projection_v10",
        )
        self.assertEqual(
            objective["regularity_gain_floor"]["mode"],
            "causal_hf_aggregate_gain_floor_v2",
        )
        self.assertEqual(objective["dual_update_mode"], "exact_projection_v1")
        self.assertEqual(
            objective["passenger_holding_constraint"]["dual_update_mode"],
            "exact_projection_v1",
        )
        self.assertEqual(projection["regularity_target"], 0.036)
        self.assertEqual(projection["passenger_target"], 0.075)
        self.assertEqual(projection["tolerance"], 1e-8)
        self.assertEqual(projection["max_iterations"], 200)
        self.assertEqual(projection["support_floor"], 1e-12)
        self.assertEqual(
            lower["action_bins"], [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0])
        self.assertEqual(lower["discrete_critic"], "zero_hold_advantage")
        self.assertFalse(lower["causal_holding_guard"]["enable"])

    def test_runner_exposes_projection_contract_without_execution_guard(self):
        config = load_config(
            ROOT / "configs_freqduet" / f"{CONFIG}.yaml")
        with TemporaryDirectory() as tmp:
            config.setdefault("logging", {})["logs_dir"] = tmp
            runner = TransitDuetV2Runner(config)

        trainer = runner.lower_trainer
        projection = trainer.regularity_policy_contract[
            "categorical_projection"]
        self.assertTrue(trainer.regularity_projection_enabled)
        self.assertFalse(trainer.regularity_soft_dual_enabled)
        self.assertFalse(trainer.regularity_passenger_soft_dual_enabled)
        self.assertEqual(
            trainer.regularity_constraint_cost_mode,
            "hf_aggregate_gain_shortfall_v4",
        )
        self.assertEqual(
            projection["base_policy"],
            "pessimistic_safe_soft_policy_v1",
        )
        self.assertEqual(projection["distillation"], "reverse_kl_v1")
        self.assertEqual(projection["distillation_steps"], 1)
        self.assertEqual(projection["execution_adjustment"], "none")
        self.assertFalse(runner.lower_causal_holding_guard.enabled)
        self.assertIn(
            "lower_regularity_projection_target_regularity_cost",
            DiagnosticLog.HEADER,
        )
        self.assertIn(
            "lower_regularity_projection_actor_reverse_kl",
            DiagnosticLog.HEADER,
        )


if __name__ == "__main__":
    unittest.main()
