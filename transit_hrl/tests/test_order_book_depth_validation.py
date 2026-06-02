import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.order_book_data import make_synthetic_order_book
from freq_hrl.experiments.trading.order_book_depth_validation import (
    apply_order_book_stress,
    run_validation,
)


class OrderBookDepthValidationTest(unittest.TestCase):
    def test_order_book_stress_changes_depth_and_spread(self):
        rows = make_synthetic_order_book(seed=3, steps=16)
        stressed = apply_order_book_stress(
            rows,
            spread_mult=2.0,
            depth_mult=0.5,
            latency_bins=2,
        )
        base_spread = rows[3]["ask"] - rows[3]["bid"]
        stressed_spread = stressed[3]["ask"] - stressed[3]["bid"]
        self.assertGreater(stressed_spread, base_spread)
        self.assertLess(stressed[3]["bid_size"], rows[3]["bid_size"])

    def test_run_validation_writes_paired_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_validation(
                Path(tmp) / "out",
                seeds=[1, 2],
                steps=48,
                methods=["ema", "state_space"],
                csv_files=[],
                min_pairs=2,
            )
            self.assertTrue(payload["paired_checks"])
            self.assertTrue((Path(tmp) / "out" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
