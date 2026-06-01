import unittest

from freq_hrl.experiments.transit.native_promotion_replan_validation import paired_checks


class NativePromotionReplanValidationTest(unittest.TestCase):
    def test_paired_checks_gate_native_reward_and_wait(self):
        rows = [
            {
                "seed": 1,
                "variant": "interval_only",
                "ep_reward": -10.0,
                "avg_wait_min": 5.0,
                "score": -6.0,
                "upper_plan_decisions": 3.0,
            },
            {
                "seed": 1,
                "variant": "native_promotion_replan",
                "ep_reward": -8.0,
                "avg_wait_min": 4.0,
                "score": -5.0,
                "upper_plan_decisions": 5.0,
            },
            {
                "seed": 2,
                "variant": "interval_only",
                "ep_reward": -9.0,
                "avg_wait_min": 4.5,
                "score": -5.5,
                "upper_plan_decisions": 4.0,
            },
            {
                "seed": 2,
                "variant": "native_promotion_replan",
                "ep_reward": -7.5,
                "avg_wait_min": 4.1,
                "score": -5.0,
                "upper_plan_decisions": 6.0,
            },
        ]
        checks = {row["metric"]: row for row in paired_checks(rows)}
        self.assertGreater(checks["ep_reward"]["delta_mean"], 0.0)
        self.assertLess(checks["avg_wait_min"]["delta_mean"], 0.0)
        self.assertEqual(checks["ep_reward"]["n_common"], 2)


if __name__ == "__main__":
    unittest.main()
