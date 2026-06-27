import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.cs_top_venue_experiment_matrix import (
    build_cs_top_venue_experiment_matrix,
    write_outputs,
)


class CSTopVenueExperimentMatrixTest(unittest.TestCase):
    def test_builds_eight_reviewer_experiment_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strong = root / "strong_learned_baseline_validation_latest"
            strong.mkdir(parents=True)
            (strong / "summary.json").write_text(json.dumps({
                "summary": {
                    "ppo_strong_baseline_status": "partial",
                    "sac_td3_status": "registered_external_missing",
                    "scenario_count": 4,
                    "parameter_budget_status": "matched",
                }
            }), encoding="utf-8")
            agency = root / "agency_demand_onboard_coverage_latest"
            agency.mkdir()
            (agency / "summary.json").write_text(json.dumps({
                "summary": {"same_agency_native_control_status": "external_truth_not_control_linked"}
            }), encoding="utf-8")
            order_book = root / "order_book_lobster_venue_grade_multisymbol"
            order_book.mkdir()
            (order_book / "summary.json").write_text(json.dumps({
                "coverage": {"venue_grade_l2_l3_session_pairs": 3}
            }), encoding="utf-8")

            payload = build_cs_top_venue_experiment_matrix(root)
            self.assertEqual(payload["summary"]["experiment_count"], 8)
            rows = {row["id"]: row for row in payload["experiments"]}
            self.assertEqual(rows["E1"]["current_status"], "partial_ppo_supported")
            self.assertEqual(rows["E2"]["current_status"], "supported")
            self.assertEqual(rows["E4"]["current_status"], "matched")
            self.assertEqual(rows["E8"]["current_status"], "partial_scale")
            self.assertTrue(all("command" in row for row in payload["scheduler_manifest"]))

    def test_write_outputs_creates_scheduler_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = build_cs_top_venue_experiment_matrix(root)
            out = root / "out"
            write_outputs(out, payload)
            self.assertTrue((out / "cs_experiment_matrix.csv").exists())
            self.assertTrue((out / "scheduler_manifest.csv").exists())
            self.assertTrue((out / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
