import unittest

from scripts.run_freqduet_protocol_v2_matrix import resolved_config
from scripts.validate_freqduet_protocol_v6_configs import validate


CONFIGS = {
    name: f"F_freqduet_protocol_v6_{name}_hiro"
    for name in (
        "main", "nofreq", "rawhistory", "allfreq", "upperonly",
        "loweronly", "swapped", "nobudget", "noguard",
        "noloadcost", "waitonlycredit", "csac",
    )
}


class ProtocolV6ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {
            name: resolved_config(config) for name, config in CONFIGS.items()
        }

    def test_all_configs_lock_the_same_causal_execution_protocol(self):
        for name, config in self.configs.items():
            with self.subTest(config=name):
                protocol = config["protocol"]
                timetable = config["upper"]["timetable_planner"]
                self.assertEqual(protocol["version"], "freqduet-eval-v6")
                self.assertEqual(
                    protocol["objective_contract"],
                    "realized_restricted_passenger_journey_v6",
                )
                self.assertEqual(
                    timetable["terminal_schedule_mode"],
                    "exact_headway_curve",
                )
                self.assertTrue(timetable["terminal_dispatch"])
                self.assertEqual(
                    config["randomness"]["mode"], "isolated_streams_v4")
                self.assertEqual(
                    config["frequency"]["forecast_mode"], "causal")
                self.assertEqual(
                    config["frequency"]["observation_source"],
                    "apc_boardings",
                )

    def test_main_implements_frequency_split_and_historical_prior(self):
        main = self.configs["main"]
        frequency = main["frequency"]
        timetable = main["upper"]["timetable_planner"]
        credit = main["upper"]["interval_credit"]
        guard = main["lower"]["causal_holding_guard"]

        self.assertTrue(frequency["enable"])
        self.assertEqual(frequency["method"], "harmonic")
        self.assertTrue(frequency["use_historical_prior"])
        self.assertEqual(frequency["upper_mode"], "low")
        self.assertEqual(frequency["lower_mode"], "high")
        self.assertTrue(frequency["replace_upper_demand_with_low"])
        self.assertEqual(
            timetable["headway_budget_mode"],
            "rolling_zero_sum_delta_v6",
        )
        self.assertEqual(
            timetable["headway_budget_window_s"],
            timetable["replan_interval_s"],
        )
        self.assertEqual(guard["evidence_mode"], "pre_action_departure_v6")
        self.assertEqual(credit["assignment_mode"], "additive")
        self.assertGreater(credit["weights"]["onboard"], 0.0)
        self.assertGreater(credit["weights"]["dispatch_backlog"], 0.0)
        self.assertEqual(credit["weights"]["fleet"], 0.0)

    def test_locked_ablations_change_only_the_intended_mechanism(self):
        nofreq = self.configs["nofreq"]["frequency"]
        self.assertFalse(nofreq["enable"])
        self.assertFalse(nofreq["upper_features"])
        self.assertFalse(nofreq["lower_features"])

        raw = self.configs["rawhistory"]["frequency"]
        self.assertEqual(raw["method"], "raw_history")
        self.assertGreater(raw["upper_history_bins"], 1)
        self.assertGreater(raw["lower_history_bins"], 1)

        self.assertEqual(
            self.configs["allfreq"]["frequency"]["upper_mode"], "all")
        self.assertEqual(
            self.configs["allfreq"]["frequency"]["lower_mode"], "all")
        self.assertFalse(
            self.configs["upperonly"]["frequency"]["lower_features"])
        self.assertFalse(
            self.configs["loweronly"]["frequency"]["upper_features"])
        self.assertEqual(
            self.configs["swapped"]["frequency"]["upper_mode"], "high")
        self.assertEqual(
            self.configs["swapped"]["frequency"]["lower_mode"], "low")
        self.assertEqual(
            self.configs["nobudget"]["upper"]["timetable_planner"]
            ["headway_budget_mode"],
            "free",
        )
        self.assertFalse(
            self.configs["noguard"]["lower"]["causal_holding_guard"]
            ["enable"])
        self.assertFalse(
            self.configs["noloadcost"]["lower"]["load_weighted_holding"]
            ["enable"])
        wait_weights = self.configs["waitonlycredit"]["upper"][
            "interval_credit"]["weights"]
        self.assertEqual(wait_weights["onboard"], 0.0)
        self.assertEqual(wait_weights["dispatch_backlog"], 0.0)
        self.assertEqual(
            self.configs["csac"]["upper"]["algorithm_id"],
            "standard_sac_v4",
        )

    def test_fail_closed_cli_validator_accepts_the_locked_matrix(self):
        result = validate(list(CONFIGS.values()))
        self.assertEqual(result["status"], "valid")
        self.assertEqual(len(result["scenario_contract_sha256"]), 64)

    def test_fail_closed_cli_validator_rejects_unregistered_config(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate([
                CONFIGS["main"],
                "F_freqduet_protocol_v5_main_hiro",
            ])


if __name__ == "__main__":
    unittest.main()
