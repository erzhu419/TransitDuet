import math
import unittest

from freq_hrl.experiments.trading import full_method_hpo_v7 as v7


class FullMethodHPOV7Test(unittest.TestCase):
    def test_mechanism_gate_requires_executed_promotion_across_replicates(self):
        rows = []
        hf_rows = []
        for replicate in range(5):
            for scenario in (
                "stationary_low_noise",
                "localized_burst",
                "persistent_shift",
            ):
                rows.append({
                    "training_replicate_seed": replicate,
                    "scenario": scenario,
                    "promotion_gate_transition_count": 20,
                    "promotion_count": 0,
                    "promotion_replan_count": 0,
                    "promotion_gate_action_rate": 0.0,
                    "promotion_gate_probability_mean": 0.1,
                    "upper_plan_reference_coeff_abs": 0.1,
                    "upper_plan_residual_coeff_abs": 0.1,
                    "hard_hf_budget_projection": True,
                    "hf_leakage_budget_ratio_max": 0.9,
                    "hf_overlay_rms_after_projection_max": 0.001,
                })
            hf_rows.append({
                "training_replicate_seed": replicate,
                "paired_exogenous_path_identity": True,
                "lower_hf_action_sensitivity": 0.1,
            })
        evidence = v7._mechanism_activity_summary(rows, hf_rows)
        self.assertEqual(evidence["status"], "ineligible")
        self.assertEqual(evidence["promotion_execution_count"], 0.0)
        self.assertEqual(evidence["promotion_active_replicate_fraction"], 0.0)

        for row in rows:
            if (
                row["training_replicate_seed"] < 4
                and row["scenario"] != "stationary_low_noise"
            ):
                row["promotion_count"] = 1
                row["promotion_replan_count"] = 1
                row["promotion_gate_action_rate"] = 0.2
                row["promotion_gate_probability_mean"] = 0.2
        evidence = v7._mechanism_activity_summary(rows, hf_rows)
        self.assertEqual(evidence["status"], "eligible")
        self.assertEqual(evidence["promotion_active_replicate_fraction"], 0.8)
        self.assertEqual(evidence["hf_active_replicate_fraction"], 1.0)
        self.assertEqual(evidence["reference_active_replicate_fraction"], 1.0)
        self.assertEqual(
            evidence["upper_residual_active_replicate_fraction"], 1.0
        )
        self.assertEqual(
            evidence["hard_budget_compliant_replicate_fraction"], 1.0
        )
        self.assertEqual(
            evidence["promotion_selective_replicate_fraction"], 0.8
        )
        self.assertGreater(evidence["stress_low_promotion_rate_lift_mean"], 0.0)

    def test_mechanism_gate_rejects_always_on_promotion(self):
        rows = []
        hf_rows = []
        for replicate in range(5):
            for scenario in (
                "stationary_low_noise",
                "localized_burst",
                "persistent_shift",
            ):
                rows.append({
                    "training_replicate_seed": replicate,
                    "scenario": scenario,
                    "promotion_count": 10,
                    "promotion_replan_count": 10,
                    "promotion_gate_action_rate": 1.0,
                    "promotion_gate_probability_mean": 0.6,
                    "upper_plan_reference_coeff_abs": 0.1,
                    "upper_plan_residual_coeff_abs": 0.1,
                    "hard_hf_budget_projection": True,
                    "hf_leakage_budget_ratio_max": 0.9,
                    "hf_overlay_rms_after_projection_max": 0.001,
                })
            hf_rows.append({
                "training_replicate_seed": replicate,
                "paired_exogenous_path_identity": True,
                "lower_hf_action_sensitivity": 0.1,
            })
        evidence = v7._mechanism_activity_summary(rows, hf_rows)
        self.assertEqual(evidence["status"], "ineligible")
        self.assertEqual(
            evidence["promotion_selective_replicate_fraction"], 0.0
        )

    def test_registry_tunes_only_full_method_and_baselines_with_equal_budgets(self):
        self.assertEqual(len(v7.ALL_VARIANT_IDS), 12)
        self.assertEqual(len(v7.HPO_VARIANT_IDS), 7)
        self.assertEqual(
            set(v7.ALL_VARIANT_IDS) - set(v7.HPO_VARIANT_IDS),
            {
                "freq_hrl_no_promotion_v7",
                "freq_hrl_no_hf_lower_v7",
                "freq_hrl_no_leakage_v7",
                "freq_hrl_no_lf_reference_v7",
                "freq_hrl_anchor_only_v7",
            },
        )
        self.assertEqual({
            len(v7.candidate_ids_for_variant(variant_id))
            for variant_id in v7.ALL_VARIANT_IDS
        }, {6})
        self.assertEqual(v7.canonical_full_method_parameter_count(2), 50774)

    def test_ablations_inherit_full_candidate_and_keep_v7_protocol_fields(self):
        candidate_id = "v71_forecast"
        full = v7.effective_parameters_for_variant(
            "freq_hrl_full_v7", candidate_id
        )
        expected_changes = {
            "freq_hrl_no_promotion_v7": {
                "method_contract",
                "promotion_replan_cost",
            },
            "freq_hrl_no_hf_lower_v7": {
                "method_contract",
                "lower_hf_order_scale",
                "hard_hf_budget_projection",
            },
            "freq_hrl_no_leakage_v7": {
                "method_contract",
                "leakage_scale",
                "lower_lf_constraint_coef",
                "lower_lf_dual_lr",
                "lower_lf_objective_weight",
                "hard_hf_budget_projection",
            },
            "freq_hrl_no_lf_reference_v7": {
                "method_contract",
                "upper_plan_reference_mode",
            },
            "freq_hrl_anchor_only_v7": {
                "method_contract",
                "upper_residual_action_scale",
            },
        }
        for variant_id, expected in expected_changes.items():
            with self.subTest(variant_id=variant_id):
                ablated = v7.effective_parameters_for_variant(
                    variant_id, candidate_id
                )
                changed = {
                    key for key in full if full[key] != ablated[key]
                }
                self.assertEqual(changed, expected)
                self.assertEqual(
                    ablated["capacity_reference_method_contract"],
                    "full_freq_hrl_v7",
                )
                self.assertEqual(
                    ablated["promotion_credit_mode"],
                    "paired_plan_advantage",
                )
                self.assertEqual(
                    ablated["leakage_cost_mode"], "fixed_rms_budget"
                )
                self.assertTrue(ablated["include_hf_predictability"])
        self.assertTrue(full["hard_hf_budget_projection"])
        self.assertEqual(full["upper_plan_reference_mode"], "causal_lf")
        self.assertEqual(full["promotion_counterfactual_coef"], 1.0)
        for candidate in v7.FREQUENCY_CANDIDATES:
            params = candidate.parameters
            initial_probability = 1.0 / (
                1.0 + math.exp(-params["promotion_init_logit"])
            )
            self.assertLess(
                initial_probability,
                params["promotion_deterministic_threshold"],
            )

    def test_cell_trains_one_checkpoint_on_support_only_and_never_loads_ood(self):
        payload = v7.run_hpo_cell(
            candidate_id="v71_balanced",
            variant_id="freq_hrl_full_v7",
            training_replicate_seed=2026,
            train_seeds=[42],
            checkpoint_validation_seeds=[57721],
            tuning_validation_seeds=[68207],
            steps=24,
            assets=2,
            iterations=1,
        )
        summary = payload["cell_summary"]
        self.assertEqual(summary["cell_status"], "valid")
        self.assertEqual(summary["training_scenario"], "support_mixture")
        self.assertEqual(
            summary["training_episode_protocol"],
            "independent_full_episode_support_batch_v1",
        )
        self.assertEqual(summary["environment_steps_train"], 4 * 24)
        self.assertEqual(summary["ood_period_access_status"], "not_loaded")
        self.assertEqual(
            summary["promotion_recovery_access_status"], "not_loaded"
        )
        self.assertEqual(summary["heldout_test_seeds"], [])
        self.assertEqual(len(payload["tuning_rows"]), 4)
        self.assertEqual(len(payload["hf_intervention_rows"]), 4)
        self.assertEqual(
            {row["scenario"] for row in payload["tuning_rows"]},
            set(v7.SELECTION_SCENARIOS),
        )
        self.assertTrue(all(
            row["promotion_credit_mode"] == "paired_plan_advantage"
            and row["promotion_counterfactual_symmetric"] == 1.0
            and row["promotion_gate_action_rate"] == 0.0
            and row["promotion_continue_action_credit_mean"] > 0.0
            for row in payload["tuning_rows"]
        ))
        checkpoint_hashes = {
            row["frozen_checkpoint_sha256"]
            for row in payload["tuning_rows"]
        }
        self.assertEqual(
            checkpoint_hashes, {summary["frozen_checkpoint_sha256"]}
        )
        self.assertEqual(
            summary["capacity_actual_parameter_count"],
            summary["capacity_target_parameter_count"],
        )

    def test_hpo_rejects_independently_tuned_ablation(self):
        with self.assertRaisesRegex(ValueError, "non-ablation"):
            v7.run_hpo_cell(
                candidate_id="v71_balanced",
                variant_id="freq_hrl_no_hf_lower_v7",
                training_replicate_seed=2026,
                train_seeds=[42],
                checkpoint_validation_seeds=[57721],
                tuning_validation_seeds=[68207],
                steps=12,
                assets=2,
                iterations=1,
            )


if __name__ == "__main__":
    unittest.main()
