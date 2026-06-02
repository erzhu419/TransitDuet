import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.order_book_l3_replay_validation import (
    L3QueueBook,
    make_synthetic_l3_events,
    read_l3_events_csv,
    run_validation,
)


class OrderBookL3ReplayValidationTest(unittest.TestCase):
    def test_fifo_queue_fills_agent_after_ahead_order(self):
        book = L3QueueBook()
        book.add_order("m1", "bid", 99.9, 5.0)
        book.add_order("agent1", "bid", 99.9, 4.0, owner="agent")
        executed = book.trade("bid", 99.9, 7.0, timestamp=1.0)
        self.assertAlmostEqual(executed, 7.0)
        self.assertEqual(len(book.agent_fills), 1)
        self.assertAlmostEqual(book.agent_fills[0]["signed_qty"], 2.0)
        self.assertAlmostEqual(book.depth_at("bid", 99.9), 2.0)

    def test_l3_csv_reader_and_validation_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "events.csv"
            csv_path.write_text(
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
            events = read_l3_events_csv(csv_path)
            self.assertGreaterEqual(len(events), 8)
            payload = run_validation(
                Path(tmp) / "out",
                seeds=[],
                methods=["ema", "state_space"],
                steps=4,
                levels=2,
                min_pairs=1,
                csv_files=[csv_path],
            )
            self.assertTrue(payload["paired_checks"])
            self.assertTrue((Path(tmp) / "out" / "summary.json").exists())

    def test_synthetic_l3_validation_writes_outputs(self):
        self.assertGreater(len(make_synthetic_l3_events(seed=1, steps=8, levels=2)), 0)
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_validation(
                Path(tmp) / "out",
                seeds=[1, 2],
                methods=["ema", "state_space"],
                steps=48,
                levels=2,
                min_pairs=2,
            )
            self.assertTrue(payload["paired_checks"])
            self.assertTrue((Path(tmp) / "out" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
