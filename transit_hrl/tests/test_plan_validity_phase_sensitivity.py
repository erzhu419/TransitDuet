import unittest

from scripts.analyze_plan_validity_phase_sensitivity import sensitivity


class PhaseSensitivityTest(unittest.TestCase):
    def test_large_single_gain_does_not_replace_the_primary_cohort(self):
        rows = [(1, {"seed": 1, "a": 0, "b": 4}), (1, {"seed": 2, "a": 2, "b": 1}),
                (2, {"seed": 3, "a": 2, "b": 1})]
        result = sensitivity(rows, "a", "b")
        self.assertEqual(result["episodes"], 3)
        self.assertAlmostEqual(result["primary_mean_difference"], -2 / 3)
        self.assertEqual(result["descriptive_mean_without_largest_benefit"], 1)
        self.assertEqual(len(result["paired_effects"]), 3)
        self.assertEqual(result["largest_benefit_path"]["optimizer_seed"], 1)

    def test_zero_effects_do_not_count_as_improved_paths(self):
        result = sensitivity([(1, {"seed": i, "a": 2, "b": 2}) for i in range(2)], "a", "b")
        self.assertEqual(result["improved_paths"], 0)
        self.assertEqual(result["primary_mean_difference"], 0)
        self.assertEqual(result["descriptive_mean_without_largest_harm"], 0)


if __name__ == "__main__":
    unittest.main()
