from tempfile import TemporaryDirectory
import unittest

from runner_v3 import TransitDuetV2Runner
from runner_v3 import load_config
from scripts.run_freqduet_protocol_v2_matrix import resolved_config
from scripts.validate_freqduet_protocol_v6_configs import (
    CONFIRMATION_CONFIGS,
    PROMOTED_CONFIGS,
    REGULARITY_POLICY_CONFIGS,
    ROOT,
    validate,
)


CONFIGS = {
    name: f"F_freqduet_protocol_v6_{name}_hiro"
    for name in (
        "main", "nofreq", "rawhistory", "allfreq", "upperonly",
        "loweronly", "swapped", "nobudget", "noguard",
        "noloadcost", "waitonlycredit", "csac",
    )
}
SOFT_CONFIGS = [
    f"F_freqduet_protocol_v6_{kind}_c{limit}_{rate}_hiro"
    for kind in (
        "softdual", "softreg_w025", "softreg_w05", "softreg_w1")
    for limit in ("035", "030")
    for rate in ("l3e4", "l1e3")
]
INCREMENTAL_CONFIGS = [
    "F_freqduet_protocol_v6_departctx_hiro",
    "F_freqduet_protocol_v6_avlctx_hiro",
    *[
        f"F_freqduet_protocol_v6_{kind}_w{weight}_hiro"
        for kind in ("fwdadv", "avlbal")
        for weight in ("05", "1", "2", "4")
    ],
]
COMPACT_CONFIGS = [
    "F_freqduet_protocol_v6_avlcompact_hiro",
    *[
        f"F_freqduet_protocol_v6_avlcompact_w{weight}_hiro"
        for weight in ("2", "4", "6", "8")
    ],
]
COMPACT_CONFIRMATION_CONFIGS = COMPACT_CONFIGS[:3]
COMPACT_EXPERIMENTAL_CONFIGS = COMPACT_CONFIGS[3:]
CONFIRMED_MAIN = "F_freqduet_protocol_v6_confirmed_main_hiro"


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

    def test_experimental_maskguard_requires_explicit_validator_opt_in(self):
        configs = [CONFIGS["main"], "F_freqduet_protocol_v6_maskguard_hiro"]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate(configs)
        result = validate(configs, allow_experimental=True)
        self.assertEqual(
            result["experimental_configs"],
            ["F_freqduet_protocol_v6_maskguard_hiro"],
        )

    def test_soft_regularity_factorial_requires_explicit_opt_in(self):
        configs = [CONFIGS["main"], CONFIGS["noguard"], *SOFT_CONFIGS]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate(configs)
        result = validate(configs, allow_experimental=True)
        self.assertEqual(result["experimental_configs"], sorted(SOFT_CONFIGS))

        for name in SOFT_CONFIGS:
            config = resolved_config(name)
            lower = config["lower"]
            regularity = lower.get("causal_departure_regularity", {})
            self.assertFalse(lower["causal_holding_guard"]["enable"])
            self.assertIn(lower["cost_limit"], {0.30, 0.35})
            self.assertIn(lower["lambda_lr"], {0.0003, 0.001})
            if "softreg_" in name:
                self.assertTrue(regularity["enable"])
                expected_weight = (
                    0.25 if "softreg_w025" in name
                    else 0.5 if "softreg_w05" in name
                    else 1.0
                )
                self.assertEqual(
                    regularity["cost_weight"], expected_weight)
                self.assertEqual(
                    regularity["evidence_mode"],
                    "pre_action_departure_v6",
                )
            else:
                self.assertFalse(regularity.get("enable", False))

    def test_incremental_regularity_factorial_is_causal_and_opt_in(self):
        configs = [
            CONFIGS["main"], CONFIGS["noguard"], *INCREMENTAL_CONFIGS]
        experimental = sorted(
            set(INCREMENTAL_CONFIGS) - set(CONFIRMATION_CONFIGS))
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate([CONFIGS["main"], CONFIGS["noguard"], *experimental])
        result = validate(configs, allow_experimental=True)
        self.assertEqual(
            result["experimental_configs"], experimental)
        self.assertEqual(
            result["confirmation_configs"], sorted(
                set(INCREMENTAL_CONFIGS).intersection(
                    CONFIRMATION_CONFIGS)))

        for name in INCREMENTAL_CONFIGS:
            config = resolved_config(name)
            lower = config["lower"]
            features = set(config["frequency"]["lower_context"]["features"])
            self.assertFalse(lower["causal_holding_guard"]["enable"])
            self.assertEqual(lower["cost_limit"], 0.5)
            self.assertEqual(lower["lambda_lr"], 0.0001)
            self.assertTrue({
                "departure_gap_norm", "departure_gap_valid"
            }.issubset(features))
            regularity = lower.get("causal_departure_regularity", {})
            if "fwdadv" in name:
                self.assertEqual(
                    regularity["objective_mode"],
                    "forward_incremental_reward",
                )
            elif "avlbal" in name:
                self.assertEqual(
                    regularity["objective_mode"],
                    "avl_two_sided_incremental_reward",
                )
                self.assertTrue({
                    "avl_follower_gap_norm", "avl_follower_gap_valid"
                }.issubset(features))
            else:
                self.assertFalse(regularity.get("enable", False))

    def test_selected_incremental_pair_is_registered_for_confirmation(self):
        result = validate([
            CONFIGS["main"],
            CONFIGS["noguard"],
            *CONFIRMATION_CONFIGS,
        ])
        self.assertEqual(result["experimental_configs"], [])
        self.assertEqual(
            result["confirmation_configs"], sorted(CONFIRMATION_CONFIGS))

    def test_compact_regularity_factorial_uses_only_sufficient_state(self):
        configs = [CONFIGS["main"], CONFIGS["noguard"], *COMPACT_CONFIGS]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate(configs)
        result = validate(configs, allow_experimental=True)
        self.assertEqual(
            result["experimental_configs"],
            sorted(COMPACT_EXPERIMENTAL_CONFIGS),
        )
        self.assertTrue(set(COMPACT_CONFIRMATION_CONFIGS).issubset(
            result["confirmation_configs"]))

        compact_features = {
            "regularity_hold_target_norm",
            "regularity_hold_target_valid",
        }
        raw_features = {
            "departure_gap_norm",
            "departure_gap_valid",
            "avl_follower_gap_norm",
            "avl_follower_gap_valid",
        }
        for name in COMPACT_CONFIGS:
            config = resolved_config(name)
            lower = config["lower"]
            features = set(config["frequency"]["lower_context"]["features"])
            self.assertTrue(compact_features.issubset(features))
            self.assertTrue(raw_features.isdisjoint(features))
            self.assertFalse(lower["causal_holding_guard"]["enable"])
            regularity = lower.get("causal_departure_regularity", {})
            if "_w" in name:
                self.assertTrue(regularity["enable"])
                self.assertEqual(
                    regularity["objective_mode"],
                    "avl_two_sided_incremental_reward",
                )
            else:
                self.assertFalse(regularity.get("enable", False))

    def test_compact_primary_pair_is_registered_for_confirmation(self):
        result = validate([
            CONFIGS["main"],
            CONFIGS["noguard"],
            *COMPACT_CONFIRMATION_CONFIGS,
        ])
        self.assertEqual(result["experimental_configs"], [])
        self.assertTrue(set(COMPACT_CONFIRMATION_CONFIGS).issubset(
            result["confirmation_configs"]))

    def test_confirmed_main_is_an_exact_behavioral_alias_of_compact_w2(self):
        result = validate([CONFIRMED_MAIN])
        self.assertEqual(result["canonical_main"], CONFIRMED_MAIN)
        self.assertEqual(result["promoted_configs"], PROMOTED_CONFIGS)

        confirmed = resolved_config(CONFIRMED_MAIN)
        selected = resolved_config(
            "F_freqduet_protocol_v6_avlcompact_w2_hiro")
        self.assertEqual(
            confirmed["protocol"]["role"],
            "confirmed_compact_avl_two_sided_regularity_w2_main_v6",
        )
        confirmed.pop("_name")
        selected.pop("_name")
        confirmed["protocol"].pop("role")
        selected["protocol"].pop("role")
        self.assertEqual(confirmed, selected)

    def test_action_regularity_dual_is_causal_separate_and_opt_in(self):
        configs = [CONFIRMED_MAIN, *REGULARITY_POLICY_CONFIGS]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            validate(configs)
        result = validate(configs, allow_experimental=True)
        self.assertEqual(
            result["experimental_configs"],
            sorted(REGULARITY_POLICY_CONFIGS),
        )

        for name in REGULARITY_POLICY_CONFIGS:
            config = resolved_config(name)
            lower = config["lower"]
            objective = lower["causal_regularity_policy"]
            features = set(config["frequency"]["lower_context"]["features"])
            self.assertFalse(lower["causal_holding_guard"]["enable"])
            self.assertEqual(
                objective["evidence_mode"], "compact_causal_target_v7")
            self.assertEqual(
                objective["mode"], "analytic_two_sided_target_dual_v1")
            self.assertIn(objective["cost_limit"], {0.0005, 0.001, 0.002})
            self.assertTrue({
                "regularity_hold_target_norm",
                "regularity_hold_target_valid",
            }.issubset(features))
            reward_objective = lower.get("causal_departure_regularity", {})
            self.assertEqual(
                bool(reward_objective.get("enable", False)),
                "w2actiondual" in name,
            )

    def test_action_regularity_runner_resolves_encoded_feature_indices(self):
        config = load_config(
            ROOT / "configs_freqduet"
            / "F_freqduet_protocol_v6_actiondual_c0010_hiro.yaml")
        with TemporaryDirectory() as tmp:
            config.setdefault("logging", {})["logs_dir"] = tmp
            runner = TransitDuetV2Runner(config)

        contract = runner.lower_trainer.regularity_policy_contract
        base = runner.env._base_state_dim
        features = runner.env.lower_context_features
        self.assertTrue(contract["enabled"])
        self.assertEqual(
            contract["target_feature_index"],
            base + features.index("regularity_hold_target_norm"),
        )
        self.assertEqual(
            contract["valid_feature_index"],
            base + features.index("regularity_hold_target_valid"),
        )
        self.assertEqual(contract["target_headway_feature_index"], 0)
        self.assertEqual(contract["action_target_scale_s"], 45.0)
        self.assertEqual(contract["target_headway_scale_s"], 600.0)

    def test_v6_nofrequency_state_dimension_is_derived_from_environment(self):
        config = load_config(
            ROOT / "configs_freqduet"
            / "F_freqduet_protocol_v6_nofreq_hiro.yaml")
        with TemporaryDirectory() as tmp:
            config.setdefault("logging", {})["logs_dir"] = tmp
            runner = TransitDuetV2Runner(config)

        self.assertEqual(
            runner.upper_state_dim,
            runner.env.upper_state_dim + runner.upper_plan_context_dim,
        )
        runner._current_ep = runner.upper_warmup
        runner._episode_training = False
        runner.env.reset()
        trip = runner.env.timetables[0]
        target = runner._upper_callback_v2(
            runner.env._build_upper_state(trip), trip)
        self.assertGreater(float(target), 0.0)

    def test_maskguard_uses_same_feasible_actions_in_policy_and_execution(self):
        config = load_config(
            ROOT / "configs_freqduet"
            / "F_freqduet_protocol_v6_maskguard_hiro.yaml")
        with TemporaryDirectory() as tmp:
            config.setdefault("logging", {})["logs_dir"] = tmp
            runner = TransitDuetV2Runner(config)

        expected_index = (
            runner.env._base_state_dim
            + runner.env.lower_context_features.index("causal_hold_limit"))
        self.assertTrue(runner.lower_causal_guard_policy_mask_enabled)
        self.assertEqual(runner.lower_action_limit_feature_index, expected_index)
        self.assertEqual(
            runner.lower_trainer.policy_net.action_limit_feature_index,
            expected_index,
        )
        runner._current_ep = 0
        runner._episode_training = False
        runner.env.reset()
        runner.env._upper_policy_callback = None
        states, _, _ = runner.env.initialize_state()
        key = next(key for key, value in states.items() if value)
        raw_state = states[key][0]
        action = runner._lower_action_for_agent(
            raw_state, key, deterministic=False)
        encoded = runner._augment_lower_state(raw_state, 0.0)
        limit_s = (
            float(encoded[expected_index])
            * float(runner.lower_action_bins.max()))
        self.assertLessEqual(float(action[0]), limit_s + 1e-6)
        self.assertEqual(
            runner._ep_lower_causal_guard_adjustments[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
