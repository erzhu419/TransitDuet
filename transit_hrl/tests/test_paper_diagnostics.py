import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
