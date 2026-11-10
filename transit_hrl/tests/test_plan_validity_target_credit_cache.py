import unittest

from scripts.analyze_plan_validity_target_credit_cache import action_readout


def row(a, b):
    return {"factual_stop": False, "conditional_replicas": [
        {"local_advantage": v, "adaptive_window_advantage": v, "full_advantage": v}
        for v in [a] * 4 + [b] * 4]}


class CachedTargetCreditTest(unittest.TestCase):
    def test_training_heads_are_not_evaluated_on_their_fitting_future_block(self):
        result = action_readout([row(1, -1)], [1], target="G", threshold=0, training=True)
        self.assertEqual(set(result), {"A_to_B"})
        self.assertEqual(result["A_to_B"]["frozen_gain_over_factual"], -1)

    def test_information_actions_are_evaluated_on_the_other_future_block(self):
        result = action_readout([row(1, -1), row(-1, 1)], [0, 0], target="G", threshold=1, training=False)
        self.assertEqual(set(result), {"A_to_B", "B_to_A"})
        for direction in result.values():
            self.assertEqual(direction["full_information_gain_over_frozen"], -0.5)
            self.assertEqual(direction["full_information_gain_over_selected_constant"], -0.5)

    def test_threshold_and_prediction_interventions_are_separate(self):
        result = action_readout([row(1, 1)], [1], target="W", threshold=2, training=False)
        for direction in result.values():
            self.assertEqual(direction["same_cut_information_gain_over_frozen"], 0)
            self.assertEqual(direction["score_zero_gain_over_frozen"], 1)
            self.assertEqual(direction["target_zero_information_gain_over_frozen"], 1)


if __name__ == "__main__":
    unittest.main()
