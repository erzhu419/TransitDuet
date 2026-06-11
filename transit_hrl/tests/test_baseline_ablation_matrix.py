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
            write_outputs(root / "out", built)
            self.assertTrue((root / "out" / "summary.json").exists())

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


if __name__ == "__main__":
    unittest.main()
