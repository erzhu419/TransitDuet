import math
from pathlib import Path
from tempfile import TemporaryDirectory
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

        for row in rows:
            row["promotion_deterministic_mode"] = (
                "counterfactual_advantage"
            )
            row["promotion_gate_probability_mean"] = 0.1
            row["promotion_gate_advantage_mean"] = (
                -0.01
                if row["scenario"] == "stationary_low_noise" else 0.02
            )
            row["promotion_gate_advantage_head_enabled"] = 1.0
            row["promotion_gate_advantage_alignment_valid"] = 1.0
            row["promotion_gate_advantage_sign_accuracy"] = 0.75
            row["promotion_gate_advantage_target_correlation"] = 0.4
        evidence = v7._mechanism_activity_summary(rows, hf_rows)
        self.assertEqual(evidence["status"], "eligible")
        self.assertEqual(
            evidence["stress_low_promotion_probability_lift_mean"], 0.0
        )
        self.assertGreater(
            evidence["stress_low_promotion_advantage_lift_mean"], 0.0
        )
        self.assertGreaterEqual(
            evidence["advantage_decision_accuracy_mean"],
            v7.MIN_ADVANTAGE_DECISION_ACCURACY,
        )

        for row in rows:
            row["promotion_gate_advantage_mean"] = 0.0
        evidence = v7._mechanism_activity_summary(rows, hf_rows)
        self.assertEqual(evidence["status"], "ineligible")

        for row in rows:
            row["promotion_gate_advantage_mean"] = (
                -0.01
                if row["scenario"] == "stationary_low_noise" else 0.02
            )
            row["promotion_gate_advantage_sign_accuracy"] = 0.5
        evidence = v7._mechanism_activity_summary(rows, hf_rows)
        self.assertEqual(evidence["status"], "ineligible")

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
        self.assertEqual(v7.canonical_full_method_parameter_count(2), 56855)

    def test_ablations_inherit_full_candidate_and_keep_v7_protocol_fields(self):
        candidate_id = "v72_forecast_margin"
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
                    ablated["promotion_deterministic_mode"],
                    "counterfactual_advantage",
                )
                self.assertGreater(
                    ablated["promotion_advantage_coef"], 0.0
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
            candidate_id="v72_balanced_margin",
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
            and row["promotion_deterministic_mode"]
            == "counterfactual_advantage"
            and row["promotion_gate_advantage_head_enabled"] == 1.0
            and row["promotion_gate_advantage_target_mae"] >= 0.0
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
                candidate_id="v72_balanced_margin",
                variant_id="freq_hrl_no_hf_lower_v7",
                training_replicate_seed=2026,
                train_seeds=[42],
                checkpoint_validation_seeds=[57721],
                tuning_validation_seeds=[68207],
                steps=12,
                assets=2,
                iterations=1,
            )

    def test_promotion_pilot_summary_is_diagnostic_only(self):
        manifest = v7.verify_current_freq_hrl_source_identity()[
            "source_manifest_sha256"
        ]
        payload = v7.run_hpo_cell(
            candidate_id="v72_balanced_margin",
            variant_id="freq_hrl_full_v7",
            training_replicate_seed=2026,
            train_seeds=[42],
            checkpoint_validation_seeds=[57721],
            tuning_validation_seeds=[68207],
            steps=24,
            assets=2,
            iterations=1,
            code_revision="a" * 40,
            expected_source_manifest_sha256=manifest,
        )
        with TemporaryDirectory() as directory:
            cell = Path(directory) / "cell"
            v7.write_hpo_cell(cell, payload)
            diagnostic = v7.summarize_selective_promotion_pilot(
                [cell],
                expected_candidate_ids=["v72_balanced_margin"],
                expected_replicate_seeds=[2026],
            )
        self.assertNotIn("frozen_config", diagnostic)
        self.assertTrue(diagnostic["summary"]["diagnostic_only"])
        self.assertFalse(diagnostic["summary"]["frozen_config_created"])
        self.assertEqual(
            diagnostic["summary"]["source_identity_status"], "verified"
        )
        self.assertEqual(len(diagnostic["leaderboard"]), 1)
        self.assertEqual(
            diagnostic["leaderboard"][0]["candidate_id"],
            "v72_balanced_margin",
        )


if __name__ == "__main__":
    unittest.main()
