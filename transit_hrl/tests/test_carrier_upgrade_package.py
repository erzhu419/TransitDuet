import csv
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.carrier_upgrade_package import build_carrier_upgrade_package


class CarrierUpgradePackageTest(unittest.TestCase):
    def test_build_carrier_upgrade_package_from_current_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "carrier"
            md = root / "md"
            payload = build_carrier_upgrade_package(
                results_root=Path("transit_hrl/results"),
                output_dir=out,
                md_dir=md,
                source_root=Path("."),
            )

            self.assertEqual(len(payload["documents"]), 8)
            self.assertGreaterEqual(payload["claims"], 1)
            self.assertGreaterEqual(payload["shared_core_supported"], 5)

            for path in payload["documents"].values():
                doc = Path(path)
                self.assertTrue(doc.exists(), f"missing {doc}")
                self.assertGreater(doc.stat().st_size, 500, f"empty {doc}")

            spec = (md / "freq_hrl_algorithm_spec_2026-06-27.md").read_text(encoding="utf-8")
            self.assertIn("frequency-responsibility protocol", spec)
            self.assertIn("Frozen Claims", spec)

            baseline_path = out / "baseline_manifest.csv"
            with baseline_path.open("r", newline="", encoding="utf-8") as f:
                baseline_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["baseline"] == "flat_sac" for row in baseline_rows))
            self.assertTrue(any(row["baseline"] == "swapped" for row in baseline_rows))

            repro = (md / "freq_hrl_reproducibility_package_2026-06-27.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("carrier_upgrade", repro)


if __name__ == "__main__":
    unittest.main()
