import unittest

import numpy as np

from freq_hrl.domains.trading import PortfolioExecutionEnv, TradingFrequencyTracker


class TradingDomainTest(unittest.TestCase):
    def test_portfolio_execution_env_moves_toward_target(self):
        returns = np.zeros((5, 2), dtype=np.float64)
        env = PortfolioExecutionEnv(returns)
        env.set_target([0.5, -0.25])
        _, reward, done, info = env.lower_step(0.5)
        self.assertFalse(done)
        np.testing.assert_allclose(env.position, np.array([0.25, -0.125]))
        self.assertLessEqual(info["turnover"], 1.0)
        self.assertLess(reward, 0.0)

    def test_trading_tracker_updates_market_features(self):
        tracker = TradingFrequencyTracker(
            bar_sec=60,
            method="ema",
            feature_norm=[0.01, 100.0, 0.01],
            promotion_enable=True,
        )
        tracker.update_bar([0.001, 50.0, 0.002])
        tracker.update_bar([0.002, 70.0, 0.003])
        upper = tracker.upper_features()
        lower = tracker.lower_features(current_target=[0.1, 0.0, 0.0], current_position=[0.0, 0.0, 0.0])
        self.assertGreater(upper.shape[0], 0)
        self.assertGreater(lower.shape[0], 0)
        self.assertEqual(tracker.summary()["freq_updates"], 2)

    def test_trading_tracker_fourier_is_causal(self):
        values = [
            [0.0, 1.0],
            [0.01, 1.2],
            [0.02, 1.1],
            [0.03, 0.9],
            [10.0, 100.0],
        ]

        def run(prefix):
            tracker = TradingFrequencyTracker(
                bar_sec=60,
                method="fourier",
                harmonic_period_s=3600,
                fourier_k=1,
                promotion_enable=False,
            )
            snapshots = []
            for value in prefix:
                tracker.update_bar(value)
                snapshots.append(tracker.upper_features().copy())
            return snapshots

        before_future = run(values[:4])[-1]
        with_future = run(values)[3]
        np.testing.assert_allclose(before_future, with_future)


if __name__ == "__main__":
    unittest.main()
