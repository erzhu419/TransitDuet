import unittest

from freq_hrl.experiments.trading import full_method_hpo as v4
from freq_hrl.experiments.trading import full_method_hpo_v5 as v5


class FullMethodHPOV5Test(unittest.TestCase):
    def test_v5_registry_has_equal_search_budgets_and_exact_capacity(self):
        counts = {
            variant_id: len(v5.candidate_ids_for_variant(variant_id))
            for variant_id in v5.ALL_VARIANT_IDS
        }
        self.assertEqual(set(counts.values()), {6})
        self.assertEqual(len(v5.ALL_VARIANT_IDS), 10)
        self.assertEqual(v5.canonical_full_method_parameter_count(assets=2), 50134)

    def test_v5_ablation_effective_configs_change_one_mechanism(self):
        candidate_id = v5.candidate_ids_for_variant("freq_hrl_full_v5")[1]
        full = v5.effective_parameters_for_variant(
            "freq_hrl_full_v5", candidate_id
        )
        expected_changes = {
            "freq_hrl_no_promotion_v5": {
                "method_contract",
                "promotion_replan_cost",
            },
            "freq_hrl_no_hf_lower_v5": {
                "method_contract",
                "lower_hf_order_scale",
            },
            "freq_hrl_no_leakage_v5": {
                "method_contract",
                "leakage_scale",
                "lower_lf_constraint_coef",
                "lower_lf_constraint_target",
                "lower_lf_dual_lr",
                "lower_lf_objective_weight",
                "lower_lf_effect_filter_gain",
                "lower_lf_raw_recenter_scale",
            },
        }
        for variant_id, expected in expected_changes.items():
            with self.subTest(variant_id=variant_id):
                ablated = v5.effective_parameters_for_variant(
                    variant_id, candidate_id
                )
                changed = {
                    key for key in full if full[key] != ablated[key]
                }
                self.assertEqual(changed, expected)

    def test_v5_cell_uses_independent_hf_contract_without_heldout_access(self):
        payload = v5.run_hpo_cell(
            candidate_id="v5_u3_l3_h01_p005_balanced",
            variant_id="freq_hrl_full_v5",
            scenario="persistent_shift",
            training_replicate_seed=2026,
            train_seeds=[42],
            checkpoint_validation_seeds=[57721],
            tuning_validation_seeds=[68207],
            steps=24,
            assets=2,
            iterations=1,
        )
        summary = payload["cell_summary"]
        self.assertEqual(summary["heldout_test_seeds"], [])
        self.assertEqual(summary["heldout_test_access_status"], "not_loaded")
        self.assertEqual(
            summary["tuning_protocol_version"],
            v5.FULL_METHOD_TUNING_PROTOCOL_VERSION,
        )
        self.assertEqual(
            summary["full_method_implementation_version"],
            v5.FULL_METHOD_IMPLEMENTATION_VERSION,
        )
        self.assertEqual(summary["capacity_target_parameter_count"], 50134)
        self.assertEqual(summary["capacity_actual_parameter_count"], 50134)
        self.assertEqual(summary["hf_intervention_pair_count"], 1)
        row = payload["tuning_rows"][0]
        self.assertEqual(float(row["hf_tactical_stream_enabled"]), 1.0)
        self.assertLessEqual(
            float(row["task_credit_reconstruction_max_abs_error"]), 1e-10
        )

    def test_v5_scoped_registry_does_not_mutate_v4_protocol(self):
        original_variants = tuple(v4.ALL_VARIANT_IDS)
        original_protocol = v4.FULL_METHOD_TUNING_PROTOCOL_VERSION
        v5.candidate_ids_for_variant("freq_hrl_full_v5")
        v5.effective_parameters_for_variant(
            "freq_hrl_full_v5",
            v5.candidate_ids_for_variant("freq_hrl_full_v5")[0],
        )
        self.assertEqual(tuple(v4.ALL_VARIANT_IDS), original_variants)
        self.assertEqual(v4.FULL_METHOD_TUNING_PROTOCOL_VERSION, original_protocol)
        self.assertIn("freq_hrl_full_v4", v4.ALL_VARIANT_IDS)


if __name__ == "__main__":
    unittest.main()
