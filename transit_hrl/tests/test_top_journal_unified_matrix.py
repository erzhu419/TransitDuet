import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.top_journal_unified_matrix import build_unified_matrix


class TopJournalUnifiedMatrixTest(unittest.TestCase):
    def test_v42_native_promotion_closes_c1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = (
                root
                / "scheduler_native_promotion_risk_banded_delta_floor_v42_512seed_merged"
            )
            out.mkdir(parents=True)
            checks = [
                {
                    "check": "native_wait_aware_replan_vs_interval_ep_reward",
                    "metric": "ep_reward",
                    "treatment": "native_wait_aware_replan",
                    "control": "interval_only",
                    "status": "supported",
                    "n_common": 512,
                },
                {
                    "check": "native_wait_aware_replan_vs_interval_avg_wait_min",
                    "metric": "avg_wait_min",
                    "treatment": "native_wait_aware_replan",
                    "control": "interval_only",
                    "status": "supported",
                    "n_common": 512,
                },
                {
                    "check": "native_wait_aware_replan_vs_interval_score",
                    "metric": "score",
                    "treatment": "native_wait_aware_replan",
                    "control": "interval_only",
                    "status": "supported",
                    "n_common": 512,
                },
            ]
            (out / "summary.json").write_text(
                json.dumps({"paired_checks": checks}),
                encoding="utf-8",
            )

            payload = build_unified_matrix(root)
            claims = {row["id"]: row for row in payload["claims"]}
            self.assertEqual(claims["C1"]["status"], "supported")
            self.assertIn("native_promotion_v42", claims["C1"]["evidence"])
            self.assertIn("v42", claims["C1"]["artifact"])
            self.assertIn("C7", claims)
            self.assertIn("C8", claims)

    def test_baseline_ablation_artifact_feeds_c8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "baseline_ablation_matrix"
            out.mkdir(parents=True)
            (out / "summary.json").write_text(
                json.dumps({
                    "summary": {
                        "claim_status": "supported",
                        "scenario_freq_family_win_rate": 0.75,
                    },
                    "paired_checks": [
                        {
                            "metric": "sharpe",
                            "control": "no_promotion",
                            "status": "supported",
                        }
                    ],
                }),
                encoding="utf-8",
            )
            payload = build_unified_matrix(root)
            claims = {row["id"]: row for row in payload["claims"]}
            self.assertEqual(claims["C8"]["status"], "supported")
            self.assertIn("no_promotion", claims["C8"]["evidence"])


if __name__ == "__main__":
    unittest.main()
