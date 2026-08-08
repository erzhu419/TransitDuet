import unittest

from scripts.repair_full_method_hpo_v74_selection import (
    select_budget_eligible_rows,
)


VARIANT = "generic_hrl_ppo_matched_v7"


def row(
    candidate_id,
    *,
    rank,
    budget="sufficient",
    learning="eligible",
):
    return {
        "variant_id": VARIANT,
        "candidate_id": candidate_id,
        "rank": rank,
        "learning_gate_status": learning,
        "mechanism_activity_status": "not_applicable",
        "training_budget_status": budget,
        "robust_selection_score": 1.0 / rank,
        "checkpoint_boundary_replicate_fraction": 0.0,
    }


class FullMethodHpoV74SelectionRepairTest(unittest.TestCase):
    def test_budget_ineligible_top_score_cannot_block_valid_candidate(self):
        selected, top = select_budget_eligible_rows([
            row("ppo_lr3e4_std15", rank=1, budget="unstable_training_tail"),
            row("ppo_lr1e4_std15", rank=2, budget="unstable_training_tail"),
            row("ppo_lr3e4_std10", rank=3),
            row("ppo_lr1e4_std05", rank=4),
        ], variant_ids=(VARIANT,))
        self.assertEqual(selected[VARIANT]["candidate_id"], "ppo_lr3e4_std10")
        self.assertEqual(
            selected[VARIANT]["training_budget_status"], "sufficient"
        )
        self.assertEqual(top[VARIANT], ["ppo_lr3e4_std10", "ppo_lr1e4_std05"])

    def test_no_budget_eligible_candidate_is_a_hard_failure(self):
        with self.assertRaisesRegex(ValueError, "no learning/mechanism/budget"):
            select_budget_eligible_rows([
                row(
                    "ppo_lr3e4_std15",
                    rank=1,
                    budget="unstable_training_tail",
                ),
                row("ppo_lr3e4_std10", rank=2, learning="ineligible"),
            ], variant_ids=(VARIANT,))


if __name__ == "__main__":
    unittest.main()
