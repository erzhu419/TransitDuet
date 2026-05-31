import unittest

from freq_hrl.core import RewardAttributionAccumulator


class RewardAttributionTest(unittest.TestCase):
    def test_accumulator_reports_required_frequency_costs(self):
        acc = RewardAttributionAccumulator()
        acc.log_step(
            task_reward=1.0,
            low_frequency_cost=0.2,
            high_frequency_cost=0.3,
            leakage_cost=0.04,
            promotion_adaptation_cost=0.01,
        )
        metrics = acc.episode_metrics()
        self.assertEqual(metrics["freq_attr_n"], 1.0)
        self.assertAlmostEqual(metrics["freq_attr_low_frequency_cost"], 0.2)
        self.assertAlmostEqual(metrics["freq_attr_high_frequency_cost"], 0.3)
        self.assertAlmostEqual(metrics["freq_attr_leakage_cost"], 0.04)
        self.assertAlmostEqual(metrics["freq_attr_promotion_adaptation_cost"], 0.01)
        self.assertLess(metrics["freq_attr_upper_credit"], 0.0)
        self.assertLess(metrics["freq_attr_lower_credit"], 0.0)


if __name__ == "__main__":
    unittest.main()
