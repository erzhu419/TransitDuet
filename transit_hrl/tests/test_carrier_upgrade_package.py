import csv
import json
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

            self.assertEqual(len(payload["documents"]), 9)
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
            proof_path = out / "proof_manifest.csv"
            with proof_path.open("r", newline="", encoding="utf-8") as f:
                proof_rows = list(csv.DictReader(f))
            self.assertTrue(any("Theorem 1" in row["proof_item"] for row in proof_rows))
            self.assertTrue(any("Proposition 9" in row["proof_item"] for row in proof_rows))
            self.assertTrue(all(row["status"] == "formalized_statement" for row in proof_rows))

            spec_validation = json.loads((out / "spec_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(spec_validation["claim_freeze"]["status"], "supported")
            self.assertEqual(spec_validation["shared_core"]["status"], "supported")
            self.assertEqual(
                payload["spec_validation"]["version"],
                "freq_hrl_frozen_spec_2026_06_27",
            )
            shared_core_validation = json.loads(
                (out / "shared_core_validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(shared_core_validation["status"], "supported")
            self.assertEqual(shared_core_validation["core_boundary"]["violations"], [])
            self.assertEqual(
                payload["shared_core_validation"]["status"],
                "supported",
            )
            with (out / "scheduler_seed_manifest.csv").open("r", newline="", encoding="utf-8") as f:
                seed_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["artifact"].startswith("native_promotion") for row in seed_rows))
            self.assertTrue(any(int(row["seed_count"]) > 0 for row in seed_rows))
            with (out / "reproducibility_artifact_manifest.csv").open(
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                artifact_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["artifact"] == "figure_source_data" for row in artifact_rows))
            self.assertTrue(any(row["artifact"] == "external_transit_raw_cache" for row in artifact_rows))
            with (out / "cs_top_venue_readiness.csv").open("r", newline="", encoding="utf-8") as f:
                readiness_rows = list(csv.DictReader(f))
            readiness = {row["review_axis"]: row for row in readiness_rows}
            self.assertEqual(readiness["strong_rl_baselines"]["current_status"], "blocker")
            self.assertIn("strong_rl_baselines", payload["cs_top_venue_blockers"])

            repro = (md / "freq_hrl_reproducibility_package_2026-06-27.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("carrier_upgrade", repro)
            self.assertIn("external_truth_raw_cache", repro)
            self.assertIn("Scheduler Seed Ledger", repro)
            cs_strategy = (md / "freq_hrl_cs_top_venue_strategy_2026-06-27.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("CS top conference/journal", cs_strategy)
            self.assertIn("strong_rl_baselines", cs_strategy)


if __name__ == "__main__":
    unittest.main()
