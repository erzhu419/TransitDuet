import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.order_book_matching_validation import (
    fill_market_order,
    make_synthetic_l2_order_book,
    read_l2_order_book_csv,
    run_validation,
)


class OrderBookMatchingValidationTest(unittest.TestCase):
    def test_fill_market_order_uses_multiple_levels(self):
        row = {
            "bid_prices": [99.9, 99.8],
            "ask_prices": [100.1, 100.2],
            "bid_sizes": [1.0, 5.0],
            "ask_sizes": [1.0, 5.0],
        }
        fill = fill_market_order(row, 3.0)
        self.assertAlmostEqual(fill["filled"], 3.0)
        self.assertGreater(fill["levels_used"], 1.0)
        self.assertGreater(fill["slippage_bps"], 0.0)

    def test_matching_validation_writes_outputs(self):
        self.assertGreater(len(make_synthetic_l2_order_book(seed=1, steps=8, levels=3)), 0)
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_validation(
                Path(tmp) / "out",
                seeds=[1, 2],
                latency_bins=[0, 2],
                methods=["ema", "adaptive_wavelet"],
                steps=64,
                levels=3,
                min_pairs=2,
            )
            self.assertTrue(payload["paired_checks"])
            self.assertTrue((Path(tmp) / "out" / "summary.json").exists())

    def test_matching_validation_reads_l2_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "book.csv"
            csv_path.write_text(
                "\n".join([
                    "timestamp,bid_price_1,ask_price_1,bid_size_1,ask_size_1,bid_price_2,ask_price_2,bid_size_2,ask_size_2",
                    "0,99.9,100.1,5,4,99.8,100.2,8,7",
                    "1,100.0,100.2,6,4,99.9,100.3,8,7",
                    "2,100.1,100.3,6,5,100.0,100.4,8,7",
                    "3,100.2,100.4,7,5,100.1,100.5,8,7",
                    "4,100.1,100.3,6,5,100.0,100.4,8,7",
                    "5,100.3,100.5,7,4,100.2,100.6,8,7",
                    "6,100.4,100.6,7,5,100.3,100.7,8,7",
                    "7,100.5,100.7,8,5,100.4,100.8,8,7",
                ]) + "\n",
                encoding="utf-8",
            )
            rows = read_l2_order_book_csv(csv_path, levels=2)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(rows[0]["bid_prices"]), 2)
            payload = run_validation(
                Path(tmp) / "out",
                seeds=[],
                latency_bins=[0],
                methods=["ema", "state_space"],
                steps=6,
                levels=2,
                min_pairs=1,
                csv_files=[csv_path],
            )
            self.assertEqual(payload["summary"][0]["source"], str(csv_path))


if __name__ == "__main__":
    unittest.main()
