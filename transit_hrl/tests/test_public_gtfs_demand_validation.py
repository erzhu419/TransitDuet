import tempfile
import unittest
import zipfile
from pathlib import Path

from freq_hrl.experiments.transit.public_gtfs_demand_validation import (
    load_gtfs_stop_event_counts,
    run_validation,
)


class PublicGTFSDemandValidationTest(unittest.TestCase):
    def test_load_gtfs_stop_events_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            gtfs = Path(tmp) / "sample_gtfs.zip"
            rows = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
            for trip_idx in range(24):
                hour = 5 + trip_idx // 3
                minute = (trip_idx % 3) * 20
                for stop_idx in range(4):
                    rows.append(
                        f"T{trip_idx},"
                        f"{hour:02d}:{minute + stop_idx:02d}:00,"
                        f"{hour:02d}:{minute + stop_idx:02d}:30,"
                        f"S{stop_idx},"
                        f"{stop_idx + 1}"
                    )
            with zipfile.ZipFile(gtfs, "w") as zf:
                zf.writestr("stop_times.txt", "\n".join(rows) + "\n")
            counts = load_gtfs_stop_event_counts(gtfs, bin_sec=600.0, max_series=3, min_events=5)
            self.assertEqual(len(counts), 3)
            self.assertTrue(all(series.sum() >= 5 for series in counts.values()))
            payload = run_validation(
                output_dir=Path(tmp) / "out",
                gtfs_zip=gtfs,
                methods=["ema", "fourier"],
                max_series=3,
                bin_sec=600.0,
                warmup=2,
                min_events=5,
            )
            self.assertTrue(payload["metadata"]["real_transit_feed"])
            self.assertFalse(payload["metadata"]["passenger_demand_ground_truth"])
            self.assertEqual(len(payload["summary"]), 2)


if __name__ == "__main__":
    unittest.main()
