import unittest

from freq_hrl.experiments.transit.native_wait_credit_validation import paired_checks


class NativeWaitCreditValidationTest(unittest.TestCase):
    def test_paired_checks_capture_native_wait_credit_direction(self):
        rows = [
            {
                "seed": 1,
                "variant": "no_wait_credit",
                "final_avg_wait_min": 5.0,
                "avg_wait_min_mean": 5.2,
                "final_ep_reward": -10.0,
                "final_score": -6.0,
                "wait_improvement": 0.2,
                "freq_wait_upper_credit_std": 0.0,
            },
            {
                "seed": 1,
                "variant": "native_wait_credit",
                "final_avg_wait_min": 4.5,
                "avg_wait_min_mean": 4.9,
                "final_ep_reward": -8.0,
                "final_score": -5.5,
                "wait_improvement": 0.8,
                "freq_wait_upper_credit_std": 0.4,
            },
            {
                "seed": 2,
                "variant": "no_wait_credit",
                "final_avg_wait_min": 6.0,
                "avg_wait_min_mean": 5.8,
                "final_ep_reward": -12.0,
                "final_score": -7.0,
                "wait_improvement": -0.1,
                "freq_wait_upper_credit_std": 0.0,
            },
            {
                "seed": 2,
                "variant": "native_wait_credit",
                "final_avg_wait_min": 5.1,
                "avg_wait_min_mean": 5.2,
                "final_ep_reward": -9.5,
                "final_score": -6.0,
                "wait_improvement": 0.7,
                "freq_wait_upper_credit_std": 0.5,
            },
        ]
        checks = {row["metric"]: row for row in paired_checks(rows, min_pairs=2)}
        self.assertLess(checks["final_avg_wait_min"]["delta_mean"], 0.0)
        self.assertGreater(checks["final_ep_reward"]["delta_mean"], 0.0)
        self.assertGreater(checks["freq_wait_upper_credit_std"]["delta_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
