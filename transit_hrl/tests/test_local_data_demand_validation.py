import tempfile
import unittest
from pathlib import Path

import pandas as pd

from freq_hrl.experiments.transit.local_data_demand_validation import (
    expand_hourly_counts,
    load_hourly_od_counts,
    run_validation,
)


class LocalDataDemandValidationTest(unittest.TestCase):
    def test_load_expand_and_validate_local_od_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "od.xlsx"
            rows = []
            for hour in ["06:00:00", "07:00:00", "08:00:00", "09:00:00"]:
                for origin_idx in range(5):
                    row = {"time": hour, "origin": f"X{origin_idx + 1:02d}"}
                    for dest_idx in range(5):
                        row[f"X{dest_idx + 1:02d}"] = (
                            0 if origin_idx == dest_idx else origin_idx + dest_idx + 1
                        )
                    rows.append(row)
            pd.DataFrame(rows).to_excel(path, index=False)
            counts = load_hourly_od_counts(path, max_series=5)
            self.assertEqual(len(counts), 5)
            expanded = expand_hourly_counts(next(iter(counts.values())), bins_per_hour=4)
            self.assertEqual(expanded.size, 16)
            payload = run_validation(
                output_dir=Path(tmp) / "out",
                od_path=path,
                methods=["ema", "fourier"],
                max_series=5,
                bins_per_hour=4,
                warmup=2,
            )
            self.assertEqual(len(payload["summary"]), 2)
            self.assertTrue(payload["paired_deltas"])


if __name__ == "__main__":
    unittest.main()
