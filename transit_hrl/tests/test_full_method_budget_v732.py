from pathlib import Path
import unittest
from unittest.mock import patch

from freq_hrl.experiments.trading import full_method_budget_plan_v732 as plan
from freq_hrl.experiments.trading import full_method_budget_validation_v732 as budget
from freq_hrl.experiments.trading import full_method_confirmatory_plan_v732 as confirm
from freq_hrl.experiments.trading import full_method_hpo_v7 as hpo


class FullMethodBudgetV732Test(unittest.TestCase):
    def test_plan_is_complete_and_seed_roles_are_disjoint(self):
        audit = plan.validate_plan()
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(len(plan.experiment_cells([64, 96])), 90)
        used = set(hpo.DEFAULT_PILOT_OPTIMIZER_SEEDS)
        used.update(hpo.DEFAULT_FINAL_HPO_OPTIMIZER_SEEDS)
        used.update(confirm.DEFAULT_CONFIRMATORY_REPLICATES)
        self.assertFalse(
            set(plan.DEFAULT_BUDGET_OPTIMIZER_SEEDS).intersection(used)
        )
        self.assertEqual(set(plan.REPRESENTATIVE_CANDIDATES), set(hpo.HPO_VARIANT_IDS))

    def test_mandatory_ladder_selects_96_only_when_every_family_passes(self):
        directories = [
            Path(f"/fake/{iteration}/{index}")
            for iteration in (64, 96)
            for index in range(45)
        ]

        def cell_summary(path):
            return {"iterations": int(Path(path).parent.name)}

        def load_cells(paths):
            iteration = int(Path(paths[0]).parent.name)
            summaries = [{
                "iterations": iteration,
                "code_revision": "a" * 40,
                "source_manifest_sha256": "b" * 64,
                "source_identity_status": "verified",
            }]
            return summaries, [], [], budget._expected_keys(iteration)

        def leaderboard_row(**kwargs):
            return {
                "variant_id": kwargs["variant_id"],
                "candidate_id": kwargs["candidate_id"],
                "trained_checkpoint_fraction": 1.0,
                "validation_learning_gain_mean": 0.1,
                "checkpoint_boundary_replicate_fraction": 0.2,
            }

        with patch.object(budget, "_cell_summary", side_effect=cell_summary), patch.object(
            hpo, "_load_validated_hpo_cells", side_effect=load_cells
        ), patch.object(
            hpo, "_validate_common_hpo_fields", return_value=None
        ), patch.object(
            hpo, "_candidate_leaderboard_row", side_effect=leaderboard_row
        ):
            payload = budget.summarize_budget_cells(
                directories, expected_budgets=[64, 96]
            )
        self.assertEqual(payload["status"], "budget_selected")
        self.assertEqual(payload["selected_iterations"], 96)
        self.assertEqual(payload["budget_gate_by_iterations"]["96"], "pass")
        self.assertEqual(
            budget.validate_budget_decision(payload)["selected_iterations"],
            96,
        )

    def test_bound_hpo_cell_rejects_iteration_mismatch_before_training(self):
        with self.assertRaisesRegex(ValueError, "differ from the budget"):
            hpo.run_hpo_cell(
                candidate_id="ppo_lr1e4_std05",
                variant_id="flat_ppo_matched_v7",
                training_replicate_seed=7001,
                train_seeds=[42],
                promotion_calibration_seeds=[140001],
                checkpoint_validation_seeds=[57721],
                tuning_validation_seeds=[68207],
                steps=12,
                assets=2,
                iterations=96,
                training_budget_plan_sha256=plan.plan_sha256(),
                training_budget_decision_sha256="c" * 64,
                training_budget_selected_iterations=128,
            )


if __name__ == "__main__":
    unittest.main()
