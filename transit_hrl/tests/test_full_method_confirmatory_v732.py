import unittest

from freq_hrl.experiments.trading import full_method_confirmatory_plan_v732 as plan
from freq_hrl.experiments.trading import full_method_confirmatory_v732 as confirm
from freq_hrl.experiments.trading import full_method_hpo_v7 as hpo


class FullMethodConfirmatoryV732Test(unittest.TestCase):
    @staticmethod
    def _frozen() -> dict:
        candidate = "v73_balanced_strict"
        selected = {}
        for variant_id in (
            "freq_hrl_full_v7",
            "freq_hrl_no_promotion_v7",
        ):
            selected[variant_id] = {
                "candidate_id": candidate,
                "effective_parameters": hpo.effective_parameters_for_variant(
                    variant_id, candidate
                ),
            }
        return {
            "selected": selected,
            "rollout_seed_roots": [42],
            "promotion_calibration_seeds": [43],
            "checkpoint_validation_seeds": [44],
            "tuning_validation_seeds": [45],
            "training_replicate_seeds": list(
                hpo.DEFAULT_FINAL_HPO_OPTIMIZER_SEEDS
            ),
            "steps": 12,
            "assets": 2,
            "iterations": 1,
            "confirmatory_plan_version": plan.CONFIRMATORY_PLAN_VERSION,
            "confirmatory_plan_sha256": plan.plan_sha256(),
        }

    def test_plan_is_stable_unique_and_source_bound(self):
        audit = plan.validate_plan()
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(audit["sha256"], plan.plan_sha256())
        self.assertEqual(len(plan.DEFAULT_CONFIRMATORY_REPLICATES), 24)
        self.assertEqual(len(plan.DEFAULT_HELDOUT_SEEDS), 8)
        self.assertFalse(
            set(plan.DEFAULT_CONFIRMATORY_REPLICATES)
            & set(plan.DEFAULT_HELDOUT_SEEDS)
        )

    def test_roles_require_exact_registered_replicates_and_paths(self):
        frozen = self._frozen()
        replicate = plan.DEFAULT_CONFIRMATORY_REPLICATES[0]
        heldout = confirm._validate_confirmatory_roles(
            frozen,
            training_replicate_seed=replicate,
            heldout_seeds=plan.DEFAULT_HELDOUT_SEEDS,
        )
        self.assertEqual(tuple(heldout), plan.DEFAULT_HELDOUT_SEEDS)
        with self.assertRaisesRegex(ValueError, "outside the registered plan"):
            confirm._validate_confirmatory_roles(
                frozen,
                training_replicate_seed=999,
                heldout_seeds=plan.DEFAULT_HELDOUT_SEEDS,
            )
        with self.assertRaisesRegex(ValueError, "differ"):
            confirm._validate_confirmatory_roles(
                frozen,
                training_replicate_seed=replicate,
                heldout_seeds=plan.DEFAULT_HELDOUT_SEEDS[:-1],
            )

    def test_frozen_training_replays_calibration_only_when_gate_exists(self):
        frozen = self._frozen()
        _, _, params, calibration, rows = confirm._train_frozen_variant(
            frozen=frozen,
            variant_id="freq_hrl_full_v7",
            training_replicate_seed=plan.DEFAULT_CONFIRMATORY_REPLICATES[0],
        )
        self.assertEqual(calibration["status"], "calibrated")
        self.assertTrue(rows)
        self.assertEqual(
            params["promotion_advantage_threshold"],
            calibration["calibrated_decision_threshold"],
        )
        _, _, _, ablated_calibration, ablated_rows = (
            confirm._train_frozen_variant(
                frozen=frozen,
                variant_id="freq_hrl_no_promotion_v7",
                training_replicate_seed=plan.DEFAULT_CONFIRMATORY_REPLICATES[0],
            )
        )
        self.assertEqual(ablated_calibration["status"], "not_applicable")
        self.assertEqual(ablated_rows, [])

    def test_effect_averages_paths_and_scenarios_before_inference(self):
        comparator = plan.PRIMARY_BASELINE_COMPARATORS[0]
        replicates = list(plan.DEFAULT_CONFIRMATORY_REPLICATES[:4])
        heldout = list(plan.DEFAULT_HELDOUT_SEEDS[:2])
        scenarios = tuple(plan.EVALUATION_SCENARIOS[:2])
        index = {}
        for variant_id in (hpo.ABLATION_PARENT_VARIANT, comparator):
            for replicate in replicates:
                for scenario in scenarios:
                    for seed in heldout:
                        index[(variant_id, replicate, scenario, seed)] = {
                            "total_return": (
                                "2.0"
                                if variant_id == hpo.ABLATION_PARENT_VARIANT
                                else "1.0"
                            )
                        }
        row = confirm._effect_row(
            index,
            comparator=comparator,
            metric="total_return",
            scenarios=scenarios,
            training_replicates=replicates,
            heldout_seeds=heldout,
            analysis_scope="test",
            hypothesis_role="primary_baseline",
            multiplicity_family="test",
        )
        self.assertEqual(row["independent_training_replicates"], 4)
        self.assertEqual(row["heldout_paths_per_replicate"], 2)
        self.assertEqual(row["directional_improvement_mean"], 1.0)
        self.assertEqual(row["directional_ci95_low"], 1.0)


if __name__ == "__main__":
    unittest.main()
