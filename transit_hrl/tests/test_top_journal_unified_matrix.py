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
            self.assertIn("C9", claims)

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

    def test_pressure_regime_winners_feed_c9(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "baseline_ablation_matrix"
            out.mkdir(parents=True)
            regimes = [
                "stationary_low_noise",
                "stationary_high_noise",
                "localized_burst",
                "persistent_shift",
                "ood_period",
            ]
            (out / "summary.json").write_text(
                json.dumps({
                    "summary": {
                        "claim_status": "partial",
                        "scenario_freq_family_win_rate": 1.0,
                    },
                    "scenario_winners": [
                        {"scenario": regime, "freq_family_wins": True}
                        for regime in regimes
                    ],
                    "paired_checks": [],
                }),
                encoding="utf-8",
            )
            payload = build_unified_matrix(root)
            claims = {row["id"]: row for row in payload["claims"]}
            self.assertEqual(claims["C9"]["status"], "supported")
            self.assertIn("stationary_low_noise", claims["C9"]["evidence"])

    def test_c5_requires_native_leakage_selector_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "leakage_no_tradeoff_matrix"
            out.mkdir(parents=True)
            (out / "summary.json").write_text(
                json.dumps({
                    "domain_verdicts": [
                        {
                            "domain": "trading_ppo_primal_dual",
                            "verdict": "no_tradeoff_strict_supported",
                        },
                        {
                            "domain": "native_real_demand_service_response_v7",
                            "verdict": "no_tradeoff_supported",
                        },
                    ],
                    "adaptive_native_real_demand_selector": {
                        "status": "supported",
                        "selected_domain": "native_real_demand_service_response_v7",
                        "selected_verdict": "no_tradeoff_supported",
                        "supported": True,
                        "strict_supported": False,
                    },
                }),
                encoding="utf-8",
            )

            payload = build_unified_matrix(root)
            claims = {row["id"]: row for row in payload["claims"]}
            self.assertEqual(claims["C5"]["status"], "supported")
            self.assertIn("native_selector_status=supported", claims["C5"]["evidence"])

    def test_c5_stays_partial_without_native_selector_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "leakage_no_tradeoff_matrix"
            out.mkdir(parents=True)
            (out / "summary.json").write_text(
                json.dumps({
                    "domain_verdicts": [
                        {
                            "domain": "trading_ppo_primal_dual",
                            "verdict": "no_tradeoff_strict_supported",
                        },
                        {
                            "domain": "transit_real_surrogate",
                            "verdict": "no_tradeoff_supported",
                        },
                    ],
                    "adaptive_native_real_demand_selector": {
                        "status": "blocked_no_native_no_tradeoff",
                        "selected_domain": "native_real_demand_service_response_v7",
                        "selected_verdict": "partial",
                        "supported": False,
                        "strict_supported": False,
                    },
                }),
                encoding="utf-8",
            )

            payload = build_unified_matrix(root)
            claims = {row["id"]: row for row in payload["claims"]}
            self.assertEqual(claims["C5"]["status"], "partial")
            self.assertIn("native_selector_status=blocked_no_native_no_tradeoff", claims["C5"]["evidence"])

    def test_c5_prefers_leakage_artifact_with_native_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_out = root / "leakage_no_tradeoff_matrix_latest_patch"
            old_out.mkdir(parents=True)
            (old_out / "summary.json").write_text(
                json.dumps({
                    "domain_verdicts": [
                        {
                            "domain": "transit_real_surrogate",
                            "verdict": "no_tradeoff_supported",
                        }
                    ]
                }),
                encoding="utf-8",
            )
            new_out = root / "leakage_no_tradeoff_matrix_latest"
            new_out.mkdir(parents=True)
            (new_out / "summary.json").write_text(
                json.dumps({
                    "domain_verdicts": [
                        {
                            "domain": "trading_ppo_primal_dual",
                            "verdict": "no_tradeoff_strict_supported",
                        },
                        {
                            "domain": "native_real_demand_service_response_v7",
                            "verdict": "no_tradeoff_supported",
                        },
                    ],
                    "adaptive_native_real_demand_selector": {
                        "status": "supported",
                        "selected_domain": "native_real_demand_service_response_v7",
                        "selected_verdict": "no_tradeoff_supported",
                        "supported": True,
                        "strict_supported": False,
                    },
                }),
                encoding="utf-8",
            )

            payload = build_unified_matrix(root)
            claims = {row["id"]: row for row in payload["claims"]}
            self.assertEqual(claims["C5"]["status"], "supported")
            self.assertIn("native_selector_status=supported", claims["C5"]["evidence"])
            self.assertIn("leakage_no_tradeoff_matrix_latest", claims["C5"]["artifact"])
            self.assertNotIn("latest_patch", claims["C5"]["artifact"])


if __name__ == "__main__":
    unittest.main()
