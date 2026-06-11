import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.manuscript_submission_pack import build_submission_pack


class ManuscriptSubmissionPackTest(unittest.TestCase):
    def _write_json(self, root: Path, relative: str, payload: dict) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_submission_pack_generates_core_tables_and_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_json(
                root,
                "top_journal_unified_matrix_latest/summary.json",
                {
                    "claims": [
                        {
                            "id": "C1",
                            "claim": "Native learned promotion improves reward and wait",
                            "status": "supported",
                            "evidence": "reward=supported wait=supported",
                            "remaining_gap": "bounded claim",
                            "artifact": "artifact-a",
                        }
                    ],
                    "summary": {"supported": 1},
                },
            )
            self._write_json(
                root,
                "baseline_ablation_matrix_latest/summary.json",
                {
                    "summary": {
                        "claim_status": "supported",
                        "scenario_freq_family_win_rate": 1.0,
                        "required_baselines_positive": ["vanilla_rl"],
                    },
                    "paired_checks": [
                        {
                            "check": "freq_hrl_vs_vanilla_rl_sharpe",
                            "control": "vanilla_rl",
                            "metric": "sharpe",
                            "status": "supported",
                            "n_common": 5,
                            "delta_mean": 1.0,
                            "delta_ci95_low": 0.5,
                            "delta_ci95_high": 1.5,
                            "win_rate": 1.0,
                        }
                    ],
                },
            )
            self._write_json(
                root,
                "agency_demand_onboard_coverage_latest/summary.json",
                {
                    "summary": {
                        "evidence_scope": "real_afc_apc_external_board_alight_load_od_plus_native_service_response",
                        "supported_boundaries": 7,
                        "external_missing_boundaries": 3,
                    },
                    "claim_boundaries": [
                        {
                            "evidence_item": "real_public_bus_stop_onboard_load",
                            "status": "supported",
                            "allowed_wording": "load source",
                            "forbidden_wording": "load improvement",
                            "evidence": "rows=10",
                        }
                    ],
                },
            )
            self._write_json(
                root,
                "external_transit_truth_validation_latest/summary.json",
                {
                    "summary": {
                        "evidence_scope": "real_public_board_alight_load_and_estimated_od",
                        "supported_boundaries": 3,
                    },
                    "source_coverage": [
                        {
                            "source": "mbta_bus_stop_trip_ridership",
                            "claim_status": "supported",
                            "source_kind": "public_agency_observed_bus_apc",
                            "rows": 10,
                            "unique_routes": 2,
                            "unique_stops": 5,
                            "boundary": "fixture",
                        }
                    ],
                },
            )
            self._write_json(
                root,
                "order_book_lobster_venue_grade_multisymbol/summary.json",
                {
                    "coverage": {
                        "venue_grade_l2_l3_session_pairs": 3,
                        "source_quality_status": "venue_grade_ready",
                    }
                },
            )

            payload = build_submission_pack(
                results_root=root,
                output_dir=root / "pack",
                md_dir=root / "md",
            )

            self.assertEqual(payload["summary"]["claims"], 1)
            self.assertEqual(payload["summary"]["figures"], 5)
            self.assertTrue((root / "pack" / "claim_evidence_table.csv").exists())
            self.assertTrue((root / "pack" / "baseline_ablation_table.csv").exists())
            self.assertTrue((root / "pack" / "real_data_table.csv").exists())
            submission = (root / "md" / "freq_hrl_submission_package_2026-06-12.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Conservative Submission Package", submission)
            self.assertIn("full deployment validation", submission)


if __name__ == "__main__":
    unittest.main()
