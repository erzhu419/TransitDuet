import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from freq_hrl.experiments.trading.diagnostic_plots import (
    plot_noleakage_drift,
    plot_signal_decomposition,
)
from freq_hrl.experiments.trading.public_market_data import (
    align_prices,
    price_returns,
    run_dataset_eval,
)


class TradingExperimentToolsTest(unittest.TestCase):
    def test_public_market_dataset_eval_uses_price_returns(self):
        dates = [f"2024-01-{day:02d}" for day in range(1, 8)]
        series = [
            ("AAA", [{"date": date, "close": 100.0 + i} for i, date in enumerate(dates)]),
            ("BBB", [{"date": date, "close": 80.0 + 0.5 * i} for i, date in enumerate(dates)]),
        ]
        _, prices = align_prices(series)
        returns = price_returns(prices)
        row = run_dataset_eval(returns)
        self.assertEqual(row["assets"], 2)
        self.assertEqual(row["freq_method"], "ema")
        self.assertIn("sharpe", row)
        state_row = run_dataset_eval(returns, freq_method="state_space")
        self.assertEqual(state_row["freq_method"], "state_space")

    def test_signal_plot_smoke(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_signal_decomposition(out, seed=42, steps=60, assets=2, scenario="persistent_shift")
            self.assertTrue((out / "signal_decomposition.png").exists())
            self.assertTrue((out / "promotion_timeline.png").exists())

    def test_noleakage_drift_plot_smoke(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            perf = out / "perf"
            perf.mkdir()
            (perf / "summary.csv").write_text(
                "baseline,LowerLFDrift_mean,sharpe_mean,FocusScore_mean\n"
                "freq_hrl,0.12,1.0,0.8\n"
                "no_leakage,0.42,0.7,0.1\n",
                encoding="utf-8",
            )
            plot_noleakage_drift(perf, out)
            self.assertTrue((out / "noleakage_drift_comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
