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
                        {"kind": "l2", "path": "book_l2.csv", "venue": "XNAS", "symbol": "AAA", "date": "2026-01-02", "source_type": "fixture"},
                        {"kind": "l3", "path": "events_l3.csv", "venue": "XNAS", "symbol": "AAA", "date": "2026-01-02", "source_type": "fixture"},
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
            self.assertEqual(payload["coverage"]["real_l2_files"], 0)
            self.assertEqual(payload["coverage"]["real_l3_files"], 0)
            self.assertEqual(payload["coverage"]["fixture_l2_files"], 1)
            self.assertEqual(payload["coverage"]["fixture_l3_files"], 1)
            self.assertEqual(payload["coverage"]["venue_grade_l2_l3_session_pairs"], 0)
            self.assertEqual(payload["coverage"]["source_quality_status"], "mechanism_only")
            self.assertTrue((root / "out" / "summary.json").exists())
            self.assertTrue(any(row["book_kind"] == "l2" for row in payload["summary"]))
            self.assertTrue(any(row["book_kind"] == "l3" for row in payload["summary"]))

    def test_manifest_marks_explicit_real_sources(self):
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
                        {"kind": "l2", "path": "book_l2.csv", "source_type": "real"},
                        {
                            "kind": "l3",
                            "path": "events_l3.csv",
                            "source_type": "venue_grade",
                        },
                    ]
                }),
                encoding="utf-8",
            )
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
            self.assertEqual(payload["coverage"]["real_l2_files"], 1)
            self.assertEqual(payload["coverage"]["real_l3_files"], 1)
            self.assertEqual(payload["coverage"]["venue_grade_l2_l3_session_pairs"], 0)
            self.assertEqual(
                payload["coverage"]["source_quality_status"],
                "real_unpaired_or_metadata_incomplete",
            )

    def test_manifest_requires_paired_venue_sessions_for_ready_status(self):
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
                        {
                            "kind": "l2",
                            "path": "book_l2.csv",
                            "venue": "XNAS",
                            "symbol": "AAA",
                            "session": "2026-01-02",
                            "source_type": "real",
                        },
                        {
                            "kind": "l3",
                            "path": "events_l3.csv",
                            "venue": "XNAS",
                            "symbol": "AAA",
                            "session": "2026-01-02",
                            "source_type": "venue_grade",
                            "matching_semantics": "price_time",
                        },
                    ]
                }),
                encoding="utf-8",
            )
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
            self.assertEqual(payload["coverage"]["venue_grade_l2_l3_session_pairs"], 1)
            self.assertEqual(payload["coverage"]["source_quality_status"], "venue_grade_ready")
            self.assertGreater(payload["coverage"]["real_or_venue_grade_sessions"], 0)


if __name__ == "__main__":
    unittest.main()
