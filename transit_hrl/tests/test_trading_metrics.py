import unittest

import numpy as np

from freq_hrl.experiments.trading.metrics import (
    METRIC_CONTRACT_VERSION,
    max_drawdown,
    periods_per_year_from_bar_seconds,
    summarize_pnl_series,
)


class TradingMetricsTest(unittest.TestCase):
    def test_annualized_sharpe_uses_sample_standard_deviation(self):
        returns = np.asarray([0.01, -0.01, 0.02], dtype=np.float64)
        equity = np.cumprod(1.0 + returns)
        stats = summarize_pnl_series(
            returns,
            equity,
            periods_per_year=252.0,
        )
        expected = np.sqrt(252.0) * returns.mean() / returns.std(ddof=1)
        self.assertAlmostEqual(stats["annualized_sharpe"], expected)
        self.assertEqual(stats["sharpe"], stats["annualized_sharpe"])
        self.assertEqual(stats["metric_contract_version"], METRIC_CONTRACT_VERSION)
        self.assertLess(stats["equity_reconstruction_max_abs_error"], 1e-12)

    def test_drawdown_includes_initial_equity(self):
        self.assertAlmostEqual(max_drawdown([0.90, 0.95]), 0.10)

    def test_additive_pnl_reconstructs_fixed_notional_equity(self):
        pnl = np.asarray([0.01, -0.02, 0.03], dtype=np.float64)
        equity = 1.0 + np.cumsum(pnl)
        stats = summarize_pnl_series(
            pnl,
            equity,
            periods_per_year=100.0,
            compounding="additive",
            min_annualization_years=0.0,
        )
        self.assertEqual(
            stats["return_series_kind"],
            "fixed_notional_normalized_pnl_increment",
        )
        self.assertLess(stats["equity_reconstruction_max_abs_error"], 1e-12)
        self.assertAlmostEqual(stats["annualized_return"], 100.0 * pnl.mean())

    def test_short_horizon_does_not_report_calmar(self):
        returns = np.asarray([0.01, -0.005, 0.002], dtype=np.float64)
        stats = summarize_pnl_series(
            returns,
            np.cumprod(1.0 + returns),
            periods_per_year=252.0,
        )
        self.assertFalse(stats["annualization_reliable"])
        self.assertTrue(np.isnan(stats["annualized_return"]))
        self.assertTrue(np.isnan(stats["calmar"]))
        self.assertTrue(np.isfinite(stats["episode_return_to_drawdown"]))

    def test_period_frequency_contract_is_explicit(self):
        self.assertEqual(periods_per_year_from_bar_seconds(24 * 3600), 252.0)
        self.assertEqual(
            periods_per_year_from_bar_seconds(60.0),
            252.0 * 6.5 * 60.0,
        )

    def test_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            summarize_pnl_series([0.01], [], periods_per_year=252.0)


if __name__ == "__main__":
    unittest.main()
