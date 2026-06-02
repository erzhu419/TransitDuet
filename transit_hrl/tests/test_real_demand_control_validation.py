import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.transit.real_demand_control_validation import (
    build_trace_map,
    run_validation,
)


class RealDemandControlValidationTest(unittest.TestCase):
    def test_build_trace_map_normalizes_real_series(self):
        traces = build_trace_map(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [4.0, 3.0, 2.0, 1.0],
            },
            [1, 2],
            steps=6,
            corridors=2,
            demand_scale=18.0,
        )
        self.assertEqual(set(traces), {1, 2})
        self.assertEqual(traces[1].shape, (6, 2))
        self.assertGreater(traces[1].max(), 0.0)

    def test_run_validation_from_afc_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "afc.csv"
            with cache.open("w", encoding="utf-8") as f:
                f.write("transit_timestamp,station_complex_id,station_complex,ridership\n")
                for station_idx in range(3):
                    for hour in range(72):
                        f.write(
                            f"2024-10-{1 + hour // 24:02d}T{hour % 24:02d}:00:00.000,"
                            f"{station_idx},Station {station_idx},"
                            f"{10 + station_idx + hour % 12}\n"
                        )
            payload = run_validation(
                output_dir=Path(tmp) / "out",
                sources=["afc"],
                train_seeds=[1],
                eval_seeds=[2],
                steps=24,
                iterations=1,
                corridors=2,
                optimizer_seed=7,
                max_series=2,
                min_bins=24,
                demand_scale=12.0,
                afc_cache_csv=cache,
                apc_cache_csv=None,
                afc_start="2024-10-01T00:00:00",
                afc_end="2024-10-04T00:00:00",
                apc_start="2026-01-01",
                apc_end="2026-01-02",
                limit=1000,
                min_pairs=1,
            )
            self.assertEqual(payload["summary"]["sources"], ["afc"])
            self.assertTrue(payload["paired_checks"])
            self.assertTrue((Path(tmp) / "out" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
