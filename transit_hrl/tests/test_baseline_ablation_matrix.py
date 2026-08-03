import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.baseline_ablation_matrix import (
    build_baseline_ablation_matrix,
    write_outputs,
)


class BaselineAblationMatrixTest(unittest.TestCase):
    def test_matrix_pairs_freq_hrl_against_required_baselines(self):
        rows = []
        for scenario in ("persistent_shift", "localized_burst"):
            for seed in (1, 2, 3):
                rows.extend([
                    {
                        "source_artifact": "synthetic",
                        "scenario": scenario,
                        "seed": seed,
                        "baseline": "freq_hrl",
                        "sharpe": 2.0,
                        "total_return": 0.10,
                        "FocusScore": 1.0,
                        "LowerLFDrift": 0.10,
                    },
                    {
                        "source_artifact": "synthetic",
                        "scenario": scenario,
                        "seed": seed,
                        "baseline": "no_promotion",
                        "sharpe": 1.0,
                        "total_return": 0.05,
                        "FocusScore": 0.5,
                        "LowerLFDrift": 0.20,
                    },
                    {
                        "source_artifact": "synthetic",
                        "scenario": scenario,
                        "seed": seed,
                        "baseline": "swapped",
                        "sharpe": 0.8,
                        "total_return": 0.02,
                        "FocusScore": -0.2,
                        "LowerLFDrift": 0.30,
                    },
                ])
        for row in rows:
            row["metric_contract_version"] = "trading_metrics_v2"
        payload = build_baseline_ablation_matrix(
            {"inline": Path("missing.json")},
            min_pairs=3,
        )
        self.assertEqual(payload["summary"]["claim_status"], "missing")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Directly exercise writer on a handcrafted payload, then exercise
            # paired statistics through the public builder by writing JSON.
            payload = {
                "per_seed": rows,
                "summary": [],
            }
            path = root / "pressure" / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            built = build_baseline_ablation_matrix(
                {"pressure": path},
                min_pairs=3,
            )
            checks = {row["check"]: row for row in built["paired_checks"]}
            self.assertEqual(checks["freq_hrl_vs_no_promotion_sharpe"]["status"], "supported")
            self.assertEqual(checks["freq_hrl_vs_swapped_FocusScore"]["status"], "supported")
            self.assertEqual(built["summary"]["claim_status"], "supported")
            self.assertEqual(built["summary"]["required_baselines_inconclusive"], [])
            self.assertIn("vanilla_rl", built["summary"]["required_baselines_missing"])
            self.assertEqual(
                built["summary"]["strong_learned_baseline_status"],
                "registered_missing",
            )
            learned = {
                row["baseline"]: row
                for row in built["summary"]["learned_baseline_manifest"]
            }
            self.assertEqual(learned["flat_ppo"]["evidence_status"], "registered_missing")
            write_outputs(root / "out", built)
            self.assertTrue((root / "out" / "summary.json").exists())
            self.assertTrue((root / "out" / "learned_baseline_manifest.csv").exists())

    def test_native_promotion_artifact_can_support_no_promotion_ablation_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pressure_rows = []
            for seed in (1, 2, 3):
                pressure_rows.extend([
                    {
                        "source_artifact": "pressure",
                        "scenario": "persistent_shift",
                        "seed": seed,
                        "baseline": "freq_hrl",
                        "sharpe": 1.0,
                        "total_return": 0.1,
                        "FocusScore": 1.0,
                        "LowerLFDrift": 0.1,
                    },
                    {
                        "source_artifact": "pressure",
                        "scenario": "persistent_shift",
                        "seed": seed,
                        "baseline": "no_promotion",
                        "sharpe": 1.0,
                        "total_return": 0.1,
                        "FocusScore": 1.0,
                        "LowerLFDrift": 0.1,
                    },
                ])
            for row in pressure_rows:
                row["metric_contract_version"] = "trading_metrics_v2"
            pressure_path = root / "pressure" / "summary.json"
            pressure_path.parent.mkdir(parents=True)
            pressure_path.write_text(
                json.dumps({"per_seed": pressure_rows}),
                encoding="utf-8",
            )
            promotion_path = root / "promotion" / "summary.json"
            promotion_path.parent.mkdir(parents=True)
            promotion_path.write_text(
                json.dumps({
                    "paired_checks": [
                        {
                            "metric": "ep_reward",
                            "treatment": "native_wait_aware_replan",
                            "control": "interval_only",
                            "status": "supported",
                        },
                        {
                            "metric": "avg_wait_min",
                            "treatment": "native_wait_aware_replan",
                            "control": "interval_only",
                            "status": "supported",
                        },
                    ],
                }),
                encoding="utf-8",
            )

            built = build_baseline_ablation_matrix(
                {
                    "pressure": pressure_path,
                    "native_promotion_v47": promotion_path,
                },
                min_pairs=3,
            )
            summary = built["summary"]
            self.assertEqual(summary["claim_status"], "supported")
            self.assertIn("no_promotion", summary["required_baselines_positive"])
            self.assertEqual(summary["required_baselines_inconclusive"], [])
            self.assertEqual(
                summary["ablation_support_overrides"][0]["status"],
                "supported",
            )

    def test_learned_baseline_rows_are_credited_only_when_present(self):
        rows = []
        for seed in (1, 2, 3, 4):
            rows.extend([
                {
                    "source_artifact": "learned",
                    "scenario": "persistent_shift",
                    "seed": seed,
                    "baseline": "freq_hrl",
                    "sharpe": 2.0,
                    "total_return": 0.10,
                    "FocusScore": 1.0,
                    "LowerLFDrift": 0.10,
                },
                {
                    "source_artifact": "learned",
                    "scenario": "persistent_shift",
                    "seed": seed,
                    "baseline": "flat_ppo",
                    "sharpe": 1.0,
                    "total_return": 0.04,
                    "FocusScore": 0.3,
                    "LowerLFDrift": 0.20,
                },
            ])
        for row in rows:
            row["metric_contract_version"] = "trading_metrics_v2"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "learned" / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"per_seed": rows}), encoding="utf-8")
            built = build_baseline_ablation_matrix({"learned": path}, min_pairs=3)
            learned = {
                row["baseline"]: row
                for row in built["summary"]["learned_baseline_manifest"]
            }
            self.assertEqual(learned["flat_ppo"]["evidence_status"], "supported")
            self.assertEqual(learned["flat_sac"]["evidence_status"], "registered_missing")
            self.assertEqual(built["summary"]["strong_learned_baseline_status"], "partial")

    def test_legacy_sharpe_rows_are_not_headline_eligible(self):
        rows = []
        for seed in (1, 2, 3):
            rows.extend([
                {"scenario": "x", "seed": seed, "baseline": "freq_hrl", "sharpe": 2.0},
                {"scenario": "x", "seed": seed, "baseline": "swapped", "sharpe": 1.0},
            ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps({"per_seed": rows}), encoding="utf-8")
            built = build_baseline_ablation_matrix({"legacy": path}, min_pairs=3)
            check = next(
                row for row in built["paired_checks"]
                if row["check"] == "freq_hrl_vs_swapped_sharpe"
            )
            self.assertEqual(check["status"], "invalid_legacy_metric_contract")
            self.assertFalse(check["metric_contract_valid"])


if __name__ == "__main__":
    unittest.main()
