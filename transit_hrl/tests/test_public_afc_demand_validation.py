import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.transit.public_afc_demand_validation import (
    rows_to_station_hour_series,
    run_validation,
)


class PublicAFCDemandValidationTest(unittest.TestCase):
    def test_station_hour_series_and_validation_from_cache(self):
        rows = []
        for station_idx in range(4):
            for hour in range(96):
                rows.append({
                    "transit_timestamp": f"2024-10-{1 + hour // 24:02d}T{hour % 24:02d}:00:00.000",
                    "station_complex_id": str(station_idx + 1),
                    "station_complex": f"Station {station_idx + 1}",
                    "ridership": str(10 + station_idx * 3 + (hour % 24)),
                })
        series = rows_to_station_hour_series(rows, max_series=3, min_hours=48)
        self.assertEqual(len(series), 3)
        self.assertTrue(all(values.size >= 96 for values in series.values()))
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "afc.csv"
            with cache.open("w", encoding="utf-8") as f:
                f.write("transit_timestamp,station_complex_id,station_complex,ridership\n")
                for row in rows:
                    f.write(
                        f"{row['transit_timestamp']},"
                        f"{row['station_complex_id']},"
                        f"{row['station_complex']},"
                        f"{row['ridership']}\n"
                    )
            payload = run_validation(
                output_dir=Path(tmp) / "out",
                cache_csv=cache,
                methods=["ema", "fourier"],
                max_series=3,
                min_hours=48,
                warmup=4,
            )
            self.assertTrue(payload["metadata"]["real_passenger_demand"])
            self.assertTrue(payload["metadata"]["afc_style_entries"])
            self.assertEqual(len(payload["summary"]), 2)
            self.assertTrue(payload["paired_deltas"])


if __name__ == "__main__":
    unittest.main()
