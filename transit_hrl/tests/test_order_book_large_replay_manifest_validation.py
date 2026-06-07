import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.order_book_large_replay_manifest_validation import (
    load_manifest,
    run_manifest_validation,
)


class OrderBookLargeReplayManifestValidationTest(unittest.TestCase):
    def test_manifest_runner_writes_l2_l3_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            l2_path = root / "book_l2.csv"
            l2_path.write_text(
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
            l3_path = root / "events_l3.csv"
            l3_path.write_text(
                "\n".join([
                    "timestamp,event_type,side,price,size,order_id",
                    "0,add,bid,99.9,20,b1",
                    "0,add,ask,100.1,20,a1",
                    "1,trade,bid,99.9,12,t1",
                    "1,add,bid,99.8,10,b2",
                    "1,add,ask,100.2,10,a2",
                    "2,trade,ask,100.1,12,t2",
                    "2,add,bid,99.9,20,b3",
                    "2,add,ask,100.1,20,a3",
                    "3,cancel,bid,99.9,5,b3",
                    "3,trade,bid,99.9,10,t3",
                ]) + "\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "datasets": [
                        {"kind": "l2", "path": "book_l2.csv", "venue": "XNAS", "symbol": "AAA", "date": "2026-01-02"},
                        {"kind": "l3", "path": "events_l3.csv", "venue": "XNAS", "symbol": "AAA", "date": "2026-01-02"},
                    ]
                }),
                encoding="utf-8",
            )
            entries = load_manifest(manifest)
            self.assertEqual([entry.kind for entry in entries], ["l2", "l3"])
            payload = run_manifest_validation(
                root / "out",
                manifest=manifest,
                methods=["ema", "state_space"],
                steps=4,
                levels=2,
                latency_bins=[0],
                execution_modes=["market"],
                queue_ahead_fraction=0.5,
                min_pairs=1,
            )
            self.assertEqual(payload["coverage"]["l2_files"], 1)
            self.assertEqual(payload["coverage"]["l3_files"], 1)
            self.assertTrue((root / "out" / "summary.json").exists())
            self.assertTrue(any(row["book_kind"] == "l2" for row in payload["summary"]))
            self.assertTrue(any(row["book_kind"] == "l3" for row in payload["summary"]))


if __name__ == "__main__":
    unittest.main()
