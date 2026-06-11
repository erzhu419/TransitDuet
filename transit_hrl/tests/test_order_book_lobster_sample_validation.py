import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.order_book_large_replay_manifest_validation import (
    run_manifest_validation,
)
from freq_hrl.experiments.trading.order_book_lobster_sample_validation import (
    convert_lobster_pair,
)


class OrderBookLobsterSampleValidationTest(unittest.TestCase):
    def test_lobster_pair_converts_to_venue_grade_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            message = root / "message.csv"
            orderbook = root / "orderbook.csv"
            message.write_text(
                "\n".join([
                    "34200.0,1,1001,50,2236800,1",
                    "34200.1,1,1002,40,2237000,-1",
                    "34200.2,4,1001,10,2236800,1",
                    "34200.3,2,1002,5,2237000,-1",
                    "34200.4,1,1003,60,2236700,1",
                    "34200.5,1,1004,45,2237100,-1",
                    "34200.6,3,1003,20,2236700,1",
                    "34200.7,5,0,15,2237100,-1",
                    "34200.8,1,1005,30,2236900,1",
                ]) + "\n",
                encoding="utf-8",
            )
            orderbook.write_text(
                "\n".join([
                    "2237000,40,2236800,50",
                    "2237000,40,2236800,50",
                    "2237000,40,2236800,40",
                    "2237000,35,2236800,40",
                    "2237000,35,2236700,60",
                    "2237100,45,2236700,60",
                    "2237100,45,2236700,40",
                    "2237100,30,2236700,40",
                    "2237100,30,2236900,30",
                ]) + "\n",
                encoding="utf-8",
            )

            converted = convert_lobster_pair(
                message_csv=message,
                orderbook_csv=orderbook,
                output_dir=root / "converted",
                max_rows=9,
            )
            payload = run_manifest_validation(
                root / "out",
                manifest=converted["manifest"],
                methods=["ema", "state_space"],
                steps=4,
                levels=1,
                latency_bins=[0],
                execution_modes=["market"],
                queue_ahead_fraction=0.5,
                min_pairs=1,
                require_venue_grade=True,
            )
            self.assertEqual(payload["coverage"]["venue_grade_l2_l3_session_pairs"], 1)
            self.assertEqual(payload["coverage"]["venue_grade_claim_status"], "supported")
            self.assertEqual(payload["coverage"]["source_quality_status"], "venue_grade_ready")


if __name__ == "__main__":
    unittest.main()
