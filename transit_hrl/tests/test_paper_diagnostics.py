import tempfile
import unittest
import json
from pathlib import Path

from freq_hrl.experiments.paper_diagnostics import (
    build_claim_matrix,
    build_statistical_checks,
    write_report,
)
from freq_hrl.experiments.statistics import (
    claim_status,
    noninferiority_status,
    paired_delta_stats,
    sign_test_p_value,
)


class PaperDiagnosticsTest(unittest.TestCase):
    def test_claim_matrix_builds_with_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims = build_claim_matrix(root / "results", root / "transit")
            self.assertGreaterEqual(len(claims), 1)
            self.assertIn("claim", claims[0])
            checks = build_statistical_checks(root / "results")
            write_report(root / "report.md", claims, checks)
            self.assertTrue((root / "report.md").exists())

    def test_paired_statistics_capture_direction(self):
        rows = []
        for seed in [1, 2, 3, 4]:
            rows.append({"variant": "base", "seed": seed, "wait": 5.0 + seed})
            rows.append({"variant": "freq", "seed": seed, "wait": 4.0 + seed})
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("seed",),
            metric="wait",
            treatment="freq",
            control="base",
            lower_is_better=True,
            n_boot=100,
            seed=7,
        )
        self.assertEqual(stats["n_common"], 4)
        self.assertAlmostEqual(stats["delta_mean"], -1.0)
        self.assertEqual(stats["win_rate"], 1.0)
        self.assertIn(claim_status(stats, min_pairs=4), {"supported", "positive_mixed"})
        self.assertLess(sign_test_p_value([1.0, 1.0, 1.0, 1.0]), 0.2)

    def test_noninferiority_status_uses_loss_margin(self):
        stats = {
            "n_common": 5,
            "improvement_mean": -0.002,
            "improvement_ci95_low": -0.004,
            "improvement_ci95_high": 0.001,
        }
        self.assertEqual(noninferiority_status(stats, max_loss=0.005, min_pairs=5), "supported")
        self.assertEqual(noninferiority_status(stats, max_loss=0.001, min_pairs=5), "inconclusive")

    def test_real_demand_control_rows_enter_statistical_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "results" / "transit_real_demand_control"
            out.mkdir(parents=True)
            rows = []
            for source in ("afc", "apc"):
                for seed in (1, 2, 3):
                    rows.append({
                        "source": source,
                        "seed": seed,
                        "variant": "base_real_ema",
                        "control_objective": -10.0,
                        "reward_mean": -8.0,
                        "wait_proxy": 7.0,
                        "LowerLFDrift": 2.0,
                        "RawLowerLFDriftAbs": 2.5,
                    })
                    rows.append({
                        "source": source,
                        "seed": seed,
                        "variant": "full_real_freqhrl",
                        "control_objective": -8.0,
                        "reward_mean": -6.0,
                        "wait_proxy": 5.0,
                        "LowerLFDrift": 1.5,
                        "RawLowerLFDriftAbs": 1.8,
                    })
            (out / "summary.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
            checks = {
                row["check"]: row
                for row in build_statistical_checks(root / "results")
            }
            self.assertEqual(
                checks["transit_real_demand_control_objective_vs_base"]["status"],
                "supported",
            )
            self.assertLess(
                checks["transit_real_demand_control_wait_vs_base"]["delta_mean"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
