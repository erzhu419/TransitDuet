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
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            built = build_baseline_ablation_matrix(
                {"pressure": path},
                min_pairs=3,
            )
            checks = {row["check"]: row for row in built["paired_checks"]}
            self.assertEqual(checks["freq_hrl_vs_no_promotion_sharpe"]["status"], "supported")
            self.assertEqual(checks["freq_hrl_vs_swapped_FocusScore"]["status"], "supported")
            self.assertEqual(built["summary"]["claim_status"], "supported")
            write_outputs(root / "out", built)
            self.assertTrue((root / "out" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
