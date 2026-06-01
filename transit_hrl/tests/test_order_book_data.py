import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.order_book_data import (
    make_synthetic_order_book,
    read_order_book_csv,
    run_order_book_eval,
    write_order_book_csv,
)


class OrderBookDataTest(unittest.TestCase):
    def test_order_book_csv_roundtrip_and_eval(self):
        rows = make_synthetic_order_book(seed=3, steps=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.csv"
            write_order_book_csv(path, rows)
            loaded = read_order_book_csv(path)
            self.assertEqual(len(loaded), len(rows))
            summary = run_order_book_eval(loaded, freq_method="ema", steps=48)
            self.assertEqual(summary["freq_method"], "ema")
            self.assertEqual(summary["bars"], 48)
            self.assertIn("sharpe", summary)
            self.assertGreater(summary["avg_spread_bps"], 0.0)


if __name__ == "__main__":
    unittest.main()
