import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.transit.public_apc_demand_validation import (
    rows_to_route_halfhour_series,
    run_validation,
)


class PublicAPCDemandValidationTest(unittest.TestCase):
    def test_route_halfhour_series_and_validation_from_cache(self):
        rows = []
        for route_idx in range(4):
            for day in range(4):
                for halfhour in range(48):
                    rows.append({
                        "OBJECTID": str(len(rows) + 1),
                        "Route_Number": str(route_idx + 1),
                        "Route_Name": f"Route {route_idx + 1}",
                        "Ridership_Total": str(5 + route_idx * 2 + halfhour % 12),
                        "Route_Hour": str(halfhour / 2.0),
                        "Route_Hour_Description": "",
                        "Route_Date": f"2026-01-{day + 1:02d}",
                    })
        series = rows_to_route_halfhour_series(rows, max_series=3, min_bins=48)
        self.assertEqual(len(series), 3)
        self.assertTrue(all(values.size >= 192 for values in series.values()))
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "apc.csv"
            with cache.open("w", encoding="utf-8") as f:
                f.write(
                    "OBJECTID,Route_Number,Route_Name,Ridership_Total,"
                    "Route_Hour,Route_Hour_Description,Route_Date\n"
                )
                for row in rows:
                    f.write(
                        f"{row['OBJECTID']},"
                        f"{row['Route_Number']},"
                        f"{row['Route_Name']},"
                        f"{row['Ridership_Total']},"
                        f"{row['Route_Hour']},"
                        f"{row['Route_Hour_Description']},"
                        f"{row['Route_Date']}\n"
                    )
            payload = run_validation(
                output_dir=Path(tmp) / "out",
                cache_csv=cache,
                methods=["ema", "fourier", "apc_route_profile"],
                max_series=3,
                min_bins=48,
                warmup=4,
            )
            self.assertTrue(payload["metadata"]["real_passenger_demand"])
            self.assertTrue(payload["metadata"]["apc_style_boardings"])
            self.assertFalse(payload["metadata"]["apc_onboard_loads"])
            self.assertTrue(payload["metadata"]["apc_calibrated_profile"])
            self.assertEqual(len(payload["summary"]), 3)
            self.assertTrue(payload["paired_deltas"])


if __name__ == "__main__":
    unittest.main()
